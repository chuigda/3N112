package club.doki7.rkt.drv;

import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.shaderc.ShaderCompiler;
import club.doki7.rkt.util.Result;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.shaderc.Shaderc;
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

    public static void main(String[] args) {
        try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
             ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma");
             ISharedLibrary libShaderc = ILibraryLoader.platformLoader().loadLibrary("shaderc_shared")) {
            applicationStart(libVulkan, libVMA, libShaderc);
        } catch (Throwable e) {
            e.printStackTrace(System.err);
        }
    }

    private static void applicationStart(
            ISharedLibrary libVulkan,
            ISharedLibrary libVMA,
            ISharedLibrary libShaderc
    ) throws RenderException {
        RenderConfig config = new RenderConfig();

        try (RenderContext cx = RenderContext.createHeadless(libVulkan, libVMA, config)) {
            String shaderSource;
            try {
                shaderSource = Files.readString(Path.of("resc/shader/basic.comp"));
            } catch (IOException e) {
                throw new RuntimeException("无法打开 shader 文件: basic.comp", e);
            }

            Shaderc shaderc = new Shaderc(libShaderc);
            ShaderCompiler compiler = ShaderCompiler.create(
                    shaderc,
                    (_, _, _, _) -> {
                        throw new UnsupportedOperationException();
                    }
            );

            Result<String, String> compileResult = compiler.compileIntoAssembly(
                    "basic.comp",
                    shaderSource,
                    ShadercShaderKind.COMPUTE_SHADER
            );
            if (compileResult instanceof Result.Err<String, String> err) {
                log.severe("计算着色器编译失败: " + err.error);
                return;
            }

            String spvAssembly = ((Result.Ok<String, String>) compileResult).value;
            System.out.println("Shader 编译成功，SPIR-V 汇编:\n" + spvAssembly);
        }
    }

    private static final Logger log = Logger.getLogger(BasicComputeShader.class.getName());
}
