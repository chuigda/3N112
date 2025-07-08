package club.doki7.rkt.launch.nn;

import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.shaderc.ShaderCompiler;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.common.ShaderStage;
import club.doki7.rkt.vk.desc.DescriptorKind;
import club.doki7.rkt.vk.desc.DescriptorSetLayout;
import club.doki7.rkt.vk.desc.DescriptorSetLayoutBinding;
import club.doki7.rkt.vk.desc.PushConstantRange;
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

        mlpForwardSetLayout = createForwardSetLayout();
        mlpForwardPipelineLayout = PipelineLayout.create(cx, List.of(mlpForwardSetLayout), List.of());
        mlpForwardModule = createShaderModule("mlp_forward.comp.glsl");

        mlpWeightPrewarmSetLayout = createWeightPrewarmSetLayout();
        mlpWeightPrewarmPipelineLayout = PipelineLayout.create(
                cx,
                List.of(mlpWeightPrewarmSetLayout),
                // layout(push_constant) uniform PushConstants {
                //     float seed;
                // };
                List.of(new PushConstantRange(Float.BYTES, Set.of(ShaderStage.COMPUTE)))
        );
        mlpWeightPrewarmModule = createShaderModule("mlp_weight_prewarm.comp.glsl");

        mlpErrorSetLayout = createErrorSetLayout();
        mlpErrorPipelineLayout = PipelineLayout.create(cx, List.of(mlpErrorSetLayout), List.of());
        mlpErrorMSEModule = createShaderModule("mlp_error_mse.comp.glsl");
        mlpErrorCrossEntropyModule = createShaderModule("mlp_error_cross_entropy.comp.glsl");

        mlpUpdateWeightsSetLayout = createUpdateWeightsSetLayout();
        mlpUpdateWeightsPipelineLayout = PipelineLayout.create(cx, List.of(mlpUpdateWeightsSetLayout), List.of());
        mlpUpdateWeightsModule = createShaderModule("mlp_update_weights.comp.glsl");

        mlpBackpropSetLayout = createBackpropSetLayout();
        mlpBackpropPipelineLayout = PipelineLayout.create(cx, List.of(mlpBackpropSetLayout), List.of());
        mlpBackpropModule = createShaderModule("mlp_backprop.comp.glsl");
    }

    public MLP createModel(MLPOptions options) throws RenderException {
        Buffer.OptionsInit storageOptionsInit = new Buffer.OptionsInit();
        storageOptionsInit.usage = Set.of(Buffer.Usage.STORAGE_BUFFER, Buffer.Usage.TRANSFER_DST);
        Buffer.Options stroageOptions = storageOptionsInit.build();

        List<Buffer> weightBufferList = new ArrayList<>();
        List<Buffer> biasBufferList = new ArrayList<>();
        List<ComputePipeline> forwardPipelineList = new ArrayList<>();
        List<ComputePipeline> prewarmPipelineList = new ArrayList<>();
        List<ComputePipeline> backpropPipelineList = new ArrayList<>();
        List<ComputePipeline> updatePipelineList = new ArrayList<>();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment forwardSpec = arena.allocate(ForwardShaderSpec.LAYOUT);
            forwardSpec.set(ValueLayout.JAVA_INT, ForwardShaderSpec.OFFSET_ty, 1);
            forwardSpec.set(ValueLayout.JAVA_BOOLEAN, ForwardShaderSpec.OFFSET_useSharedMemory, options.useSharedMemory);

            MemorySegment updateWeightSpec = arena.allocate(UpdateWeightsShaderSpec.LAYOUT);
            updateWeightSpec.set(ValueLayout.JAVA_INT, UpdateWeightsShaderSpec.OFFSET_ty, 1);

            int inputSize = options.inputSize;
            for (int i = 0; i < options.layers.size(); i++) {
                MLPOptions.Layer layer = options.layers.get(i);

                forwardSpec.set(ValueLayout.JAVA_INT, ForwardShaderSpec.OFFSET_tx, layer.perceptronWorkgroupSize);
                forwardSpec.set(ValueLayout.JAVA_INT, ForwardShaderSpec.OFFSET_perceptronCount, layer.size);
                forwardSpec.set(ValueLayout.JAVA_INT, ForwardShaderSpec.OFFSET_inputSize, inputSize);
                forwardSpec.set(ValueLayout.JAVA_INT, ForwardShaderSpec.OFFSET_activation, layer.activ.value);

                forwardPipelineList.add(ComputePipeline.create(
                        cx,
                        mlpForwardPipelineLayout,
                        mlpForwardModule,
                        new ShaderSpecialisation(ForwardShaderSpec.SPEC_ENTRIES, forwardSpec)
                ));

                prewarmPipelineList.add(ComputePipeline.create(
                        cx,
                        mlpWeightPrewarmPipelineLayout,
                        mlpWeightPrewarmModule,
                        // memory layout and data are compatible, so we can reuse the same memory segment
                        new ShaderSpecialisation(WeightPrewarmShaderSpec.SPEC_ENTRIES, forwardSpec)
                ));

                updateWeightSpec.set(ValueLayout.JAVA_INT, UpdateWeightsShaderSpec.OFFSET_tx, layer.perceptronWorkgroupSize);
                updateWeightSpec.set(ValueLayout.JAVA_INT, UpdateWeightsShaderSpec.OFFSET_perceptronCount, layer.size);
                updateWeightSpec.set(ValueLayout.JAVA_INT, UpdateWeightsShaderSpec.OFFSET_inputSize, inputSize);

                updatePipelineList.add(ComputePipeline.create(
                        cx,
                        mlpUpdateWeightsPipelineLayout,
                        mlpUpdateWeightsModule,
                        new ShaderSpecialisation(UpdateWeightsShaderSpec.SPEC_ENTRIES, updateWeightSpec)
                ));

                if (i != options.layers.size() - 1) {
                    MLPOptions.Layer nextLayer = options.layers.get(i + 1);

                    MemorySegment backpropSpec = arena.allocate(BackpropShaderSpec.LAYOUT);
                    backpropSpec.set(ValueLayout.JAVA_INT, BackpropShaderSpec.OFFSET_tx, layer.perceptronWorkgroupSize);
                    backpropSpec.set(ValueLayout.JAVA_INT, BackpropShaderSpec.OFFSET_ty, 1);
                    backpropSpec.set(ValueLayout.JAVA_INT, BackpropShaderSpec.OFFSET_perceptronCount, layer.size);
                    backpropSpec.set(ValueLayout.JAVA_INT, BackpropShaderSpec.OFFSET_nextPerceptronCount, nextLayer.size);
                    backpropSpec.set(ValueLayout.JAVA_INT, BackpropShaderSpec.OFFSET_activation, layer.activ.value);

                    backpropPipelineList.add(ComputePipeline.create(
                            cx,
                            mlpBackpropPipelineLayout,
                            mlpBackpropModule,
                            new ShaderSpecialisation(BackpropShaderSpec.SPEC_ENTRIES, backpropSpec)
                    ));
                }

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

                inputSize = layer.size;
            }
        }

        return new MLP(
                this,
                options,
                cx,
                weightBufferList,
                biasBufferList,
                forwardPipelineList,
                prewarmPipelineList,
                backpropPipelineList,
                updatePipelineList
        );
    }

    @Override
    public void close() {
        mlpBackpropModule.close();
        mlpBackpropPipelineLayout.close();
        mlpBackpropSetLayout.close();

        mlpUpdateWeightsModule.close();
        mlpUpdateWeightsPipelineLayout.close();
        mlpUpdateWeightsSetLayout.close();

        mlpErrorCrossEntropyModule.close();
        mlpErrorMSEModule.close();
        mlpErrorPipelineLayout.close();
        mlpErrorSetLayout.close();

        mlpWeightPrewarmModule.close();
        mlpWeightPrewarmPipelineLayout.close();
        mlpWeightPrewarmSetLayout.close();

        mlpForwardModule.close();
        mlpForwardPipelineLayout.close();
        mlpForwardSetLayout.close();

        shaderCompiler.close();
        libShaderc.close();
    }

    private ShaderModule createShaderModule(String shaderName) throws RenderException {
        String shaderCode;
        try (InputStream stream = MLPFactory.class.getResourceAsStream("/resc/nn/shader/" + shaderName)) {
            if (stream == null) {
                throw new RenderException("无法读取着色器文件: " + shaderName);
            }
            shaderCode = new String(stream.readAllBytes());
        } catch (IOException e) {
            throw new RenderException("无法读取着色器文件: " + shaderName);
        }

        try (Arena arena = Arena.ofConfined()) {
            BytePtr spv = shaderCompiler.compileIntoSPV(
                    arena,
                    shaderName,
                    shaderCode,
                    ShadercShaderKind.COMPUTE_SHADER
            );
            return ShaderModule.create(cx, spv);
        }
    }

    private DescriptorSetLayout createForwardSetLayout() throws VulkanException {
        return DescriptorSetLayout.create(cx, List.of(
                // layout(set = 0, binding = 0) uniform InferOptions {
                //     uint input_offset;
                //     uint batch_size;
                // };
                UBO,
                // layout(set = 0, binding = 1) buffer InputBuffer {
                //     readonly float input_data[];
                // };
                SSBO,
                // layout(set = 0, binding = 2) buffer WeightsBuffer {
                //     readonly float weights[];
                // };
                SSBO,
                // layout(set = 0, binding = 3) buffer BiasBuffer {
                //     readonly float bias[];
                // };
                SSBO,
                // layout(set = 0, binding = 4) buffer OutputBuffer {
                //     writeonly float output_data[];
                // };
                SSBO
        ), true);
    }

    private DescriptorSetLayout createWeightPrewarmSetLayout() throws VulkanException {
        return DescriptorSetLayout.create(cx, List.of(
                // layout(set = 0, binding = 0) buffer WeightsBuffer {
                //     writeonly float weights[];
                // };
                SSBO,
                // layout(set = 0, binding = 1) buffer BiasBuffer {
                //     writeonly float bias[];
                // };
                SSBO
        ), true);
    }

    private DescriptorSetLayout createErrorSetLayout() throws VulkanException {
        return DescriptorSetLayout.create(cx, List.of(
                // layout(set = 0, binding = 0) uniform InferOptions {
                //     uint input_offset;
                //     uint batch_size;
                // };
                UBO,
                // layout(set = 0, binding = 1) buffer OutputBuffer {
                //     readonly float output_data[];
                // };
                SSBO,
                // layout(set = 0, binding = 2) buffer LabelBuffer {
                //     readonly uint label_data[];
                // };
                SSBO,
                // layout(set = 0, binding = 3) buffer GradientBuffer {
                //     writeonly float gradient_data[];
                // };
                SSBO
        ), true);
    }

    private DescriptorSetLayout createUpdateWeightsSetLayout() throws VulkanException {
        return DescriptorSetLayout.create(cx, List.of(
                // layout(set = 0, binding = 0) uniform UpdateOptions {
                //     float learning_rate;
                //     uint batch_size;
                // };
                UBO,
                // layout(set = 0, binding = 1) buffer InputBuffer {
                //     readonly float input_data[];
                // };
                SSBO,
                // layout(set = 0, binding = 2) buffer GradientBuffer {
                //     readonly float gradient_data[];
                // };
                SSBO,
                // layout(set = 0, binding = 3) buffer WeightsBuffer {
                //     float weights[];
                // };
                SSBO,
                // layout(set = 0, binding = 4) buffer BiasesBuffer {
                //     float biases[];
                // };
                SSBO
        ), true);
    }

    private DescriptorSetLayout createBackpropSetLayout() throws VulkanException {
        return DescriptorSetLayout.create(cx, List.of(
                // layout(set = 0, binding = 0) uniform InferOptions {
                //     uint input_offset;
                //     uint batch_size;
                // };
                UBO,
                // layout(set = 0, binding = 1) buffer NextLayerGradientBuffer {
                //     readonly float next_layer_gradient_data[];
                // };
                SSBO,
                // layout(set = 0, binding = 2) buffer NextLayerWeightsBuffer {
                //     readonly float next_layer_weights_data[];
                // };
                SSBO,
                // layout(set = 0, binding = 3) buffer OutputBuffer {
                //     readonly float current_layer_output_data[];
                // };
                SSBO,
                // layout(set = 0, binding = 4) buffer GradientBuffer {
                //     writeonly float gradient_data[];
                // };
                SSBO
        ), true);
    }

    private final RenderContext cx;
    private final ISharedLibrary libShaderc;
    private final ShaderCompiler shaderCompiler;

    final DescriptorSetLayout mlpForwardSetLayout;
    final PipelineLayout mlpForwardPipelineLayout;
    final ShaderModule mlpForwardModule;

    final DescriptorSetLayout mlpWeightPrewarmSetLayout;
    final PipelineLayout mlpWeightPrewarmPipelineLayout;
    final ShaderModule mlpWeightPrewarmModule;

    final DescriptorSetLayout mlpErrorSetLayout;
    final PipelineLayout mlpErrorPipelineLayout;
    final ShaderModule mlpErrorMSEModule;
    final ShaderModule mlpErrorCrossEntropyModule;

    final DescriptorSetLayout mlpUpdateWeightsSetLayout;
    final PipelineLayout mlpUpdateWeightsPipelineLayout;
    final ShaderModule mlpUpdateWeightsModule;

    final DescriptorSetLayout mlpBackpropSetLayout;
    final PipelineLayout mlpBackpropPipelineLayout;
    final ShaderModule mlpBackpropModule;

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

    private static int round2WorkgroupSize(int problemSize) {
        int[] workgroupSizes = { 2, 4, 8, 16, 32, 64 };
        for (int size : workgroupSizes) {
            if (problemSize % size > size / 2) {
                return problemSize / size;
            }
        }
        return 1;
    }

    private static final DescriptorSetLayoutBinding UBO = new DescriptorSetLayoutBinding(DescriptorKind.UNIFORM_BUFFER, ShaderStage.COMPUTE);
    private static final DescriptorSetLayoutBinding SSBO = new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE);

    static final class ForwardShaderSpec {
        static final StructLayout LAYOUT = NativeLayout.structLayout(
                ValueLayout.JAVA_INT.withName("tx"), // const uint tx
                ValueLayout.JAVA_INT.withName("ty"), // const uint ty
                ValueLayout.JAVA_INT.withName("perceptron_count"), // const uint perceptron_count
                ValueLayout.JAVA_INT.withName("input_size"), // const uint input_size
                ValueLayout.JAVA_INT.withName("activation"), // const uint activation
                ValueLayout.JAVA_INT.withName("use_shared_memory")  // const boolean use_shared_memory
        );

        static final MemoryLayout.PathElement PATH_tx = MemoryLayout.PathElement.groupElement("tx");
        static final MemoryLayout.PathElement PATH_ty = MemoryLayout.PathElement.groupElement("ty");
        static final MemoryLayout.PathElement PATH_perceptronCount = MemoryLayout.PathElement.groupElement("perceptron_count");
        static final MemoryLayout.PathElement PATH_inputSize = MemoryLayout.PathElement.groupElement("input_size");
        static final MemoryLayout.PathElement PATH_activation = MemoryLayout.PathElement.groupElement("activation");
        static final MemoryLayout.PathElement PATH_useSharedMemory = MemoryLayout.PathElement.groupElement("use_shared_memory");

        static final int OFFSET_tx = (int) LAYOUT.byteOffset(PATH_tx);
        static final int OFFSET_ty = (int) LAYOUT.byteOffset(PATH_ty);
        static final int OFFSET_perceptronCount = (int) LAYOUT.byteOffset(PATH_perceptronCount);
        static final int OFFSET_inputSize = (int) LAYOUT.byteOffset(PATH_inputSize);
        static final int OFFSET_activation = (int) LAYOUT.byteOffset(PATH_activation);
        static final int OFFSET_useSharedMemory = (int) LAYOUT.byteOffset(PATH_useSharedMemory);

        static final List<ShaderSpecialisation.Entry> SPEC_ENTRIES = List.of(
                new ShaderSpecialisation.Entry(0, OFFSET_tx, Integer.BYTES),
                new ShaderSpecialisation.Entry(1, OFFSET_ty, Integer.BYTES),
                new ShaderSpecialisation.Entry(2, OFFSET_perceptronCount, Integer.BYTES),
                new ShaderSpecialisation.Entry(3, OFFSET_inputSize, Integer.BYTES),
                new ShaderSpecialisation.Entry(4, OFFSET_activation, Integer.BYTES),
                new ShaderSpecialisation.Entry(5, OFFSET_useSharedMemory, Integer.BYTES)
        );
    }

    static final class WeightPrewarmShaderSpec {
        static final StructLayout LAYOUT = NativeLayout.structLayout(
                ValueLayout.JAVA_INT.withName("tx"), // const uint tx
                ValueLayout.JAVA_INT.withName("ty"), // const uint ty
                ValueLayout.JAVA_INT.withName("perceptron_count"), // const uint perceptron_count
                ValueLayout.JAVA_INT.withName("input_size"), // const uint input_size
                ValueLayout.JAVA_INT.withName("activation") // const uint activation
        );

        static final MemoryLayout.PathElement PATH_tx = MemoryLayout.PathElement.groupElement("tx");
        static final MemoryLayout.PathElement PATH_ty = MemoryLayout.PathElement.groupElement("ty");
        static final MemoryLayout.PathElement PATH_perceptronCount = MemoryLayout.PathElement.groupElement("perceptron_count");
        static final MemoryLayout.PathElement PATH_inputSize = MemoryLayout.PathElement.groupElement("input_size");
        static final MemoryLayout.PathElement PATH_activation = MemoryLayout.PathElement.groupElement("activation");

        static final int OFFSET_tx = (int) LAYOUT.byteOffset(PATH_tx);
        static final int OFFSET_ty = (int) LAYOUT.byteOffset(PATH_ty);
        static final int OFFSET_perceptronCount = (int) LAYOUT.byteOffset(PATH_perceptronCount);
        static final int OFFSET_inputSize = (int) LAYOUT.byteOffset(PATH_inputSize);
        static final int OFFSET_activation = (int) LAYOUT.byteOffset(PATH_activation);

        static final List<ShaderSpecialisation.Entry> SPEC_ENTRIES = List.of(
                new ShaderSpecialisation.Entry(0, OFFSET_tx, Integer.BYTES),
                new ShaderSpecialisation.Entry(1, OFFSET_ty, Integer.BYTES),
                new ShaderSpecialisation.Entry(2, OFFSET_perceptronCount, Integer.BYTES),
                new ShaderSpecialisation.Entry(3, OFFSET_inputSize, Integer.BYTES),
                new ShaderSpecialisation.Entry(4, OFFSET_activation, Integer.BYTES)
        );
    }

    static final class ErrorCrossEntropyShaderSpec {
        static final StructLayout LAYOUT = NativeLayout.structLayout(
                ValueLayout.JAVA_INT.withName("tx"), // const uint tx
                ValueLayout.JAVA_INT.withName("ty"), // const uint ty
                ValueLayout.JAVA_INT.withName("perceptron_count") // const uint perceptron_count
        );

        static final MemoryLayout.PathElement PATH_tx = MemoryLayout.PathElement.groupElement("tx");
        static final MemoryLayout.PathElement PATH_ty = MemoryLayout.PathElement.groupElement("ty");
        static final MemoryLayout.PathElement PATH_perceptronCount = MemoryLayout.PathElement.groupElement("perceptron_count");

        static final int OFFSET_tx = (int) LAYOUT.byteOffset(PATH_tx);
        static final int OFFSET_ty = (int) LAYOUT.byteOffset(PATH_ty);
        static final int OFFSET_perceptronCount = (int) LAYOUT.byteOffset(PATH_perceptronCount);

        static final List<ShaderSpecialisation.Entry> SPEC_ENTRIES = List.of(
                new ShaderSpecialisation.Entry(0, OFFSET_tx, Integer.BYTES),
                new ShaderSpecialisation.Entry(1, OFFSET_ty, Integer.BYTES),
                new ShaderSpecialisation.Entry(2, OFFSET_perceptronCount, Integer.BYTES)
        );
    }

    static final class ErrorMSEShaderSpec {
        static final StructLayout LAYOUT = NativeLayout.structLayout(
                ValueLayout.JAVA_INT.withName("tx"), // const uint tx
                ValueLayout.JAVA_INT.withName("ty"), // const uint ty
                ValueLayout.JAVA_INT.withName("perceptron_count"), // const uint perceptron_count
                ValueLayout.JAVA_INT.withName("activation") // const uint activation
        );

        static final MemoryLayout.PathElement PATH_tx = MemoryLayout.PathElement.groupElement("tx");
        static final MemoryLayout.PathElement PATH_ty = MemoryLayout.PathElement.groupElement("ty");
        static final MemoryLayout.PathElement PATH_perceptronCount = MemoryLayout.PathElement.groupElement("perceptron_count");
        static final MemoryLayout.PathElement PATH_activation = MemoryLayout.PathElement.groupElement("activation");

        static final int OFFSET_tx = (int) LAYOUT.byteOffset(PATH_tx);
        static final int OFFSET_ty = (int) LAYOUT.byteOffset(PATH_ty);
        static final int OFFSET_perceptronCount = (int) LAYOUT.byteOffset(PATH_perceptronCount);
        static final int OFFSET_activation = (int) LAYOUT.byteOffset(PATH_activation);

        static final List<ShaderSpecialisation.Entry> SPEC_ENTRIES = List.of(
                new ShaderSpecialisation.Entry(0, OFFSET_tx, Integer.BYTES),
                new ShaderSpecialisation.Entry(1, OFFSET_ty, Integer.BYTES),
                new ShaderSpecialisation.Entry(2, OFFSET_perceptronCount, Integer.BYTES),
                new ShaderSpecialisation.Entry(3, OFFSET_activation, Integer.BYTES)
        );
    }

    static final class BackpropShaderSpec {
        static final StructLayout LAYOUT = NativeLayout.structLayout(
                ValueLayout.JAVA_INT.withName("tx"), // const uint tx
                ValueLayout.JAVA_INT.withName("ty"), // const uint ty
                ValueLayout.JAVA_INT.withName("perceptron_count"), // const uint perceptron_count
                ValueLayout.JAVA_INT.withName("next_perceptron_count"), // const uint next_perceptron_count
                ValueLayout.JAVA_INT.withName("activation") // const uint activation
        );

        static final MemoryLayout.PathElement PATH_tx = MemoryLayout.PathElement.groupElement("tx");
        static final MemoryLayout.PathElement PATH_ty = MemoryLayout.PathElement.groupElement("ty");
        static final MemoryLayout.PathElement PATH_perceptronCount = MemoryLayout.PathElement.groupElement("perceptron_count");
        static final MemoryLayout.PathElement PATH_nextPerceptronCount = MemoryLayout.PathElement.groupElement("next_perceptron_count");
        static final MemoryLayout.PathElement PATH_activation = MemoryLayout.PathElement.groupElement("activation");

        static final int OFFSET_tx = (int) LAYOUT.byteOffset(PATH_tx);
        static final int OFFSET_ty = (int) LAYOUT.byteOffset(PATH_ty);
        static final int OFFSET_perceptronCount = (int) LAYOUT.byteOffset(PATH_perceptronCount);
        static final int OFFSET_nextPerceptronCount = (int) LAYOUT.byteOffset(PATH_nextPerceptronCount);
        static final int OFFSET_activation = (int) LAYOUT.byteOffset(PATH_activation);

        static final List<ShaderSpecialisation.Entry> SPEC_ENTRIES = List.of(
                new ShaderSpecialisation.Entry(0, OFFSET_tx, Integer.BYTES),
                new ShaderSpecialisation.Entry(1, OFFSET_ty, Integer.BYTES),
                new ShaderSpecialisation.Entry(2, OFFSET_perceptronCount, Integer.BYTES),
                new ShaderSpecialisation.Entry(3, OFFSET_nextPerceptronCount, Integer.BYTES),
                new ShaderSpecialisation.Entry(4, OFFSET_activation, Integer.BYTES)
        );
    }

    static final class UpdateWeightsShaderSpec {
        static final StructLayout LAYOUT = NativeLayout.structLayout(
                ValueLayout.JAVA_INT.withName("tx"), // const uint tx
                ValueLayout.JAVA_INT.withName("ty"), // const uint ty
                ValueLayout.JAVA_INT.withName("input_size"), // const uint input_size
                ValueLayout.JAVA_INT.withName("perceptron_count") // const uint perceptron_count
        );

        static final MemoryLayout.PathElement PATH_tx = MemoryLayout.PathElement.groupElement("tx");
        static final MemoryLayout.PathElement PATH_ty = MemoryLayout.PathElement.groupElement("ty");
        static final MemoryLayout.PathElement PATH_inputSize = MemoryLayout.PathElement.groupElement("input_size");
        static final MemoryLayout.PathElement PATH_perceptronCount = MemoryLayout.PathElement.groupElement("perceptron_count");

        static final int OFFSET_tx = (int) LAYOUT.byteOffset(PATH_tx);
        static final int OFFSET_ty = (int) LAYOUT.byteOffset(PATH_ty);
        static final int OFFSET_inputSize = (int) LAYOUT.byteOffset(PATH_inputSize);
        static final int OFFSET_perceptronCount = (int) LAYOUT.byteOffset(PATH_perceptronCount);

        static final List<ShaderSpecialisation.Entry> SPEC_ENTRIES = List.of(
                new ShaderSpecialisation.Entry(0, OFFSET_tx, Integer.BYTES),
                new ShaderSpecialisation.Entry(1, OFFSET_ty, Integer.BYTES),
                new ShaderSpecialisation.Entry(2, OFFSET_inputSize, Integer.BYTES),
                new ShaderSpecialisation.Entry(3, OFFSET_perceptronCount, Integer.BYTES)
        );
    }
}
