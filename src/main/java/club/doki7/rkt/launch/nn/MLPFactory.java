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
import club.doki7.rkt.vk.pipeline.PipelineLayout;
import club.doki7.rkt.vk.pipeline.ShaderModule;
import club.doki7.shaderc.Shaderc;
import club.doki7.shaderc.ShadercUtil;
import club.doki7.shaderc.enumtype.ShadercIncludeType;
import club.doki7.shaderc.enumtype.ShadercShaderKind;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

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

        // layout(set = 0, binding = 0) uniform InferOptions {
        //     uint input_offset;
        //     uint batch_size;
        // };
        mlpForwardSet0Layout = DescriptorSetLayout.create(cx, List.of(
                new DescriptorSetLayoutBinding(DescriptorKind.UNIFORM_BUFFER, ShaderStage.COMPUTE)
        ));
        // layout(set = 1, binding = 0) buffer InputBuffer {
        //     readonly float input_data[];
        // };
        mlpForwardSet1Layout = DescriptorSetLayout.create(cx, List.of(
                new DescriptorSetLayoutBinding(DescriptorKind.UNIFORM_BUFFER, ShaderStage.COMPUTE)
        ));
        // layout(set = 2, binding = 0) buffer WeightsBuffer {
        //     readonly float weights[];
        // };
        // layout(set = 2, binding = 1) buffer BiasBuffer {
        //     readonly float bias[];
        // };
        // layout(set = 2, binding = 2) buffer OutputBuffer {
        //     writeonly float output_data[];
        // };
        mlpForwardSet2Layout = DescriptorSetLayout.create(cx, List.of(
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE),
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE),
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE)
        ));
        mlpForwardPipelineLayout = PipelineLayout.create(cx, List.of(
                mlpForwardSet0Layout,
                mlpForwardSet1Layout,
                mlpForwardSet2Layout
        ), List.of());
    }

    public MLPInfer createInfer(MLPOptions options) throws RenderException {
        return null;
    }

    @Override
    public void close() throws Exception {
        mlpForwardPipelineLayout.close();
        mlpForwardSet0Layout.close();
        mlpForwardSet1Layout.close();
        mlpForwardSet2Layout.close();

        mlpForwardModule.close();
        shaderCompiler.close();
        libShaderc.close();
    }

    private final RenderContext cx;
    private final ISharedLibrary libShaderc;
    private final ShaderCompiler shaderCompiler;

    final ShaderModule mlpForwardModule;
    final DescriptorSetLayout
            mlpForwardSet0Layout,
            mlpForwardSet1Layout,
            mlpForwardSet2Layout;
    final PipelineLayout mlpForwardPipelineLayout;

    private static ShadercUtil.IncludeResult rescDirResolve(
            String requestedSource,
            @EnumType(ShadercIncludeType.class) int includeType,
            String requestingSource,
            long includeDepth
    ) throws IOException {
        Path path = Path.of("resc", "shader", requestedSource);
        String content = Files.readString(path);
        return new ShadercUtil.IncludeResult(path.toAbsolutePath().toString(), content);
    }

    private static final StructLayout MLP_FORWARD_SHADER_SPEC_LAYOUT = NativeLayout.structLayout(
            ValueLayout.JAVA_INT, // const uint tx
            ValueLayout.JAVA_INT, // const uint ty
            ValueLayout.JAVA_INT, // const uint perceptron_count
            ValueLayout.JAVA_INT, // const uint input_size
            ValueLayout.JAVA_INT, // const uint activation
            ValueLayout.JAVA_INT  // const uint max_shared_input_size
    );
}
