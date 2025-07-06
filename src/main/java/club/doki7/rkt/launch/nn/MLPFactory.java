package club.doki7.rkt.launch.nn;

import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.shaderc.ShaderCompiler;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.common.ShaderStage;
import club.doki7.rkt.vk.desc.DescriptorKind;
import club.doki7.rkt.vk.desc.DescriptorSetLayout;
import club.doki7.rkt.vk.desc.DescriptorSetLayoutBinding;
import club.doki7.rkt.vk.pipeline.ComputePipeline;
import club.doki7.rkt.vk.pipeline.PipelineLayout;
import club.doki7.rkt.vk.pipeline.ShaderModule;
import club.doki7.rkt.vk.pipeline.ShaderSpecialisation;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.shaderc.Shaderc;
import club.doki7.shaderc.ShadercUtil;
import club.doki7.shaderc.enumtype.ShadercIncludeType;
import club.doki7.shaderc.enumtype.ShadercShaderKind;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.*;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class MLPFactory implements AutoCloseable {
    public MLPFactory(RenderContext cx) throws RenderException {
        this.cx = cx;
        this.libShaderc = ILibraryLoader.platformLoader().loadLibrary("shaderc_shared");
        this.shaderCompiler = ShaderCompiler.create(
                new Shaderc(libShaderc),
                MLPFactory::rescDirResolve
        );

        String mlpForwardShaderCode;
        try (InputStream stream = MLPFactory.class.getResourceAsStream("/resc/nn/shader/mlp_forward.comp.glsl")) {
            if (stream == null) {
                throw new RenderException("找不到 MLP 前向传播着色器文件 /resc/nn/shader/mlp_forward.comp.glsl");
            }
            mlpForwardShaderCode = new String(stream.readAllBytes());
        } catch (IOException e) {
            throw new RenderException("找不到 MLP 前向传播着色器文件 /resc/nn/shader/mlp_forward.comp.glsl");
        }

        try (Arena arena = Arena.ofConfined()) {
            BytePtr spv = shaderCompiler.compileIntoSPV(
                    arena,
                    "mlp_forward.comp.glsl",
                    mlpForwardShaderCode,
                    ShadercShaderKind.COMPUTE_SHADER
            );
            mlpForwardModule = ShaderModule.create(cx, spv);
        }

        mlpForwardSetLayout = DescriptorSetLayout.create(cx, List.of(
                // layout(set = 0, binding = 0) uniform InferOptions {
                //     uint input_offset;
                //     uint batch_size;
                // };
                new DescriptorSetLayoutBinding(DescriptorKind.UNIFORM_BUFFER, ShaderStage.COMPUTE),
                // layout(set = 0, binding = 1) buffer InputBuffer {
                //     readonly float input_data[];
                // };
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE),
                // layout(set = 0, binding = 2) buffer WeightsBuffer {
                //     readonly float weights[];
                // };
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE),
                // layout(set = 0, binding = 3) buffer BiasBuffer {
                //     readonly float bias[];
                // };
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE),
                // layout(set = 0, binding = 4) buffer OutputBuffer {
                //     writeonly float output_data[];
                // };
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE)
        ), true);
        mlpForwardPipelineLayout = PipelineLayout.create(
                cx,
                List.of(mlpForwardSetLayout),
                List.of()
        );
    }

    public MLPInfer createInfer(MLPOptions options) throws RenderException {
        Buffer.OptionsInit storageOptionsInit = new Buffer.OptionsInit();
        storageOptionsInit.usage = Set.of(Buffer.Usage.STORAGE_BUFFER, Buffer.Usage.TRANSFER_DST);
        Buffer.Options stroageOptions = storageOptionsInit.build();

        List<Buffer> weightBufferList = new ArrayList<>();
        List<Buffer> biasBufferList = new ArrayList<>();
        List<ComputePipeline> computePipelineList = new ArrayList<>();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment specSegment = arena.allocate(MLP_FORWARD_SHADER_SPEC_LAYOUT);
            specSegment.set(ValueLayout.JAVA_INT, OFFSET_tx, options.perceptronWorkgroupSize);
            specSegment.set(ValueLayout.JAVA_INT, OFFSET_ty, 1);
            specSegment.set(ValueLayout.JAVA_BOOLEAN, OFFSET_useSharedMemory, options.useSharedMemory);

            int inputSize = options.inputSize;
            for (MLPOptions.Layer layer : options.layers) {
                specSegment.set(ValueLayout.JAVA_INT, OFFSET_perceptronCount, layer.size);
                specSegment.set(ValueLayout.JAVA_INT, OFFSET_inputSize, inputSize);
                specSegment.set(ValueLayout.JAVA_INT, OFFSET_activation, layer.activ.value);

                computePipelineList.add(ComputePipeline.create(
                        cx,
                        mlpForwardPipelineLayout,
                        mlpForwardModule,
                        new ShaderSpecialisation(SPEC_ENTRIES, specSegment)
                ));

                int weightBufferSize = inputSize * layer.size * Float.BYTES;
                int biasBufferSize = layer.size * Float.BYTES;

                weightBufferList.add(Buffer.create(
                        cx,
                        weightBufferSize,
                        false,
                        stroageOptions
                ));
                biasBufferList.add(Buffer.create(
                        cx,
                        biasBufferSize,
                        false,
                        stroageOptions
                ));
            }
        }

        return new MLPInfer(
                options,
                cx,
                weightBufferList,
                biasBufferList,
                computePipelineList
        );
    }

    @Override
    public void close() {
        mlpForwardPipelineLayout.close();
        mlpForwardSetLayout.close();

        mlpForwardModule.close();
        shaderCompiler.close();
        libShaderc.close();
    }

    private final RenderContext cx;
    private final ISharedLibrary libShaderc;
    private final ShaderCompiler shaderCompiler;

    final ShaderModule mlpForwardModule;
    final DescriptorSetLayout mlpForwardSetLayout;
    final PipelineLayout mlpForwardPipelineLayout;

    private static ShadercUtil.IncludeResult rescDirResolve(
            String requestedSource,
            @EnumType(ShadercIncludeType.class) int includeType,
            String requestingSource,
            long includeDepth
    ) throws IOException {
        try (InputStream inputStream = MLPFactory.class.getResourceAsStream("/resc/nn/shader/" + requestedSource)) {
            if (inputStream == null) {
                throw new IOException("无法找到着色器包含文件: " + requestedSource);
            }
            String content = new String(inputStream.readAllBytes());
            return new ShadercUtil.IncludeResult(requestedSource, content);
        }
    }

    private static final StructLayout MLP_FORWARD_SHADER_SPEC_LAYOUT = NativeLayout.structLayout(
            ValueLayout.JAVA_INT.withName("tx"), // const uint tx
            ValueLayout.JAVA_INT.withName("ty"), // const uint ty
            ValueLayout.JAVA_INT.withName("perceptron_count"), // const uint perceptron_count
            ValueLayout.JAVA_INT.withName("input_size"), // const uint input_size
            ValueLayout.JAVA_INT.withName("activation"), // const uint activation
            ValueLayout.JAVA_BOOLEAN.withName("use_shared_memory")  // const boolean use_shared_memory
    );

    private static final MemoryLayout.PathElement PATH_tx = MemoryLayout.PathElement.groupElement("tx");
    private static final MemoryLayout.PathElement PATH_ty = MemoryLayout.PathElement.groupElement("ty");
    private static final MemoryLayout.PathElement PATH_perceptronCount = MemoryLayout.PathElement.groupElement("perceptron_count");
    private static final MemoryLayout.PathElement PATH_inputSize = MemoryLayout.PathElement.groupElement("input_size");
    private static final MemoryLayout.PathElement PATH_activation = MemoryLayout.PathElement.groupElement("activation");
    private static final MemoryLayout.PathElement PATH_useSharedMemory = MemoryLayout.PathElement.groupElement("use_shared_memory");

    private static final int OFFSET_tx = (int) MLP_FORWARD_SHADER_SPEC_LAYOUT.byteOffset(PATH_tx);
    private static final int OFFSET_ty = (int) MLP_FORWARD_SHADER_SPEC_LAYOUT.byteOffset(PATH_ty);
    private static final int OFFSET_perceptronCount = (int) MLP_FORWARD_SHADER_SPEC_LAYOUT.byteOffset(PATH_perceptronCount);
    private static final int OFFSET_inputSize = (int) MLP_FORWARD_SHADER_SPEC_LAYOUT.byteOffset(PATH_inputSize);
    private static final int OFFSET_activation = (int) MLP_FORWARD_SHADER_SPEC_LAYOUT.byteOffset(PATH_activation);
    private static final int OFFSET_useSharedMemory = (int) MLP_FORWARD_SHADER_SPEC_LAYOUT.byteOffset(PATH_useSharedMemory);

    private static final List<ShaderSpecialisation.Entry> SPEC_ENTRIES = List.of(
            new ShaderSpecialisation.Entry(0, OFFSET_tx, Integer.BYTES),
            new ShaderSpecialisation.Entry(1, OFFSET_ty, Integer.BYTES),
            new ShaderSpecialisation.Entry(2, OFFSET_perceptronCount, Integer.BYTES),
            new ShaderSpecialisation.Entry(3, OFFSET_inputSize, Integer.BYTES),
            new ShaderSpecialisation.Entry(4, OFFSET_activation, Integer.BYTES),
            new ShaderSpecialisation.Entry(5, OFFSET_useSharedMemory, Integer.BYTES)
    );
}
