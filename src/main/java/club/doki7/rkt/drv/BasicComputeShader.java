package club.doki7.rkt.drv;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.shaderc.ShaderCompiler;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.shaderc.Shaderc;
import club.doki7.shaderc.ShadercUtil;
import club.doki7.shaderc.enumtype.ShadercIncludeType;
import club.doki7.shaderc.enumtype.ShadercShaderKind;
import club.doki7.vulkan.command.VulkanLoader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

public final class BasicComputeShader {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
    }

    public static void main() {
        try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
             ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma");
             ISharedLibrary libShaderc = ILibraryLoader.platformLoader().loadLibrary("shaderc_shared");
             Application app = new Application(libVulkan, libVMA, libShaderc)) {
            app.applicationStart();
        } catch (Throwable e) {
            e.printStackTrace(System.err);
        }
    }
}

final class Application implements AutoCloseable {
    @Override
    public void close() {
        cx.close();
        compiler.close();
    }

    Application(ISharedLibrary libVulkan, ISharedLibrary libVMA, ISharedLibrary libShaderc) throws RenderException {
        this.cx = RenderContext.createHeadless(libVulkan, libVMA, new RenderConfig());
        this.compiler = ShaderCompiler.create(new Shaderc(libShaderc), Application::unsupportedResolve);
    }

    void applicationStart() throws RenderException {
        String shaderSource;
        try {
            shaderSource = Files.readString(Path.of("resc/shader/forward.comp.glsl"));
        } catch (IOException e) {
            throw new RuntimeException("无法打开 shader 文件: forward.comp.glsl", e);
        }

        String spvAssembly = compiler.compileIntoAssembly(
                "forward.comp",
                shaderSource,
                ShadercShaderKind.COMPUTE_SHADER
        );
        System.out.println("Shader 编译成功，SPIR-V 汇编:\n" + spvAssembly);
    }

    private final RenderContext cx;
    private final ShaderCompiler compiler;

    private static ShadercUtil.IncludeResult unsupportedResolve(
            String requestedSource,
            @EnumType(ShadercIncludeType.class) int includeType,
            String requestingSource,
            long includeDepth
    ) {
        throw new UnsupportedOperationException();
    }

    private static final float perceptronWeights[][] = {
            // layer 1: 2 perceptron, 2 inputs for each perceptron
            {
                    // perceptron 1
                    20.0f, 20.0f,
                    // perceptron 2
                    -20.0f, -20.0f
            },
            // layer 2: 1 perceptron, 2 inputs
            {
                    // perceptron
                    20.0f, 20.0f
            }
    };
    private static final float perceptronBias[][] = {
            // layer 1: 2 biases for each perceptron
            {
                    -10.0f,
                    30.0f
            },
            // layer 2: 1 bias for the perceptron
            {
                    -10.0f
            }
    };
    private static final Logger logger = Logger.getLogger(Application.class.getName());
}
