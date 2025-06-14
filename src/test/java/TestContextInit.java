import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.vk.RenderConfig;
import club.doki7.cg112.vk.RenderContext;
import club.doki7.cg112.vk.RenderWindow;
import club.doki7.cg112.vk.Swapchain;
import club.doki7.glfw.GLFW;
import club.doki7.glfw.GLFWLoader;
import club.doki7.vulkan.command.VulkanLoader;

public class TestContextInit {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
        VulkanLoader.loadVulkanLibrary();
        GLFWLoader.loadGLFWLibrary();
        System.loadLibrary("vma");
    }

    public static void main(String[] args) throws RenderException {
        GLFW glfw = GLFWLoader.loadGLFW();
        glfw.initHint(GLFW.PLATFORM, GLFW.PLATFORM_X11);
        if (glfw.init() != GLFW.TRUE) {
            throw new RuntimeException("GLFW 初始化失败");
        }

        RenderConfig config = new RenderConfig();

        RenderWindow window = new RenderWindow(glfw, "Test Vulkan Context", 800, 600);
        RenderContext context = RenderContext.create(glfw, window.rawWindow, config);
        Swapchain swapchain = Swapchain.create(context, 800, 600);

        while (window.beforeTick()) {
            if (window.framebufferResized) {
                swapchain.close();
                swapchain = Swapchain.create(context, window.width, window.height);
            }
            window.afterTick();
        }

        swapchain.close();
        context.close();
        window.close();
    }
}
