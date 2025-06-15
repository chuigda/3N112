package club.doki7.rkt.drv;

import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.RenderWindow;
import club.doki7.rkt.vk.Swapchain;
import club.doki7.glfw.GLFW;
import club.doki7.glfw.GLFWLoader;
import club.doki7.vulkan.command.VulkanLoader;

public final class Main {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
        VulkanLoader.loadVulkanLibrary();
        GLFWLoader.loadGLFWLibrary();
        System.loadLibrary("vma");
    }

    public static void main(String[] args) {
        try {
            applicationStart();
        } catch (Throwable e) {
            e.printStackTrace(System.err);
        }
    }

    private static void applicationStart() throws RenderException {
        GLFW glfw = GLFWLoader.loadGLFW();
        if (glfw.init() != GLFW.TRUE) {
            throw new RuntimeException("GLFW 初始化失败");
        }

        RenderConfig config = new RenderConfig();

        RenderWindow window = new RenderWindow(glfw, "Example window", 800, 600);
        RenderContext cx = RenderContext.create(glfw, window.rawWindow, config);
        Swapchain swapchain = Swapchain.create(cx, 800, 600);

        while (window.tick()) {
            if (window.framebufferResized) {
                cx.waitDeviceIdle();
                swapchain.close();
                swapchain = Swapchain.create(cx, window.width, window.height);
                window.framebufferResized = false;
            }
            cx.gc();
        }

        swapchain.close();
        cx.close();
        window.close();
    }
}
