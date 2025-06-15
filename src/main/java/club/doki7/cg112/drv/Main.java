package club.doki7.cg112.drv;

import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.vk.RenderConfig;
import club.doki7.cg112.vk.RenderContext;
import club.doki7.cg112.vk.RenderWindow;
import club.doki7.cg112.vk.Swapchain;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.glfw.GLFW;
import club.doki7.glfw.GLFWLoader;
import club.doki7.vulkan.VkConstants;
import club.doki7.vulkan.bitmask.VkAccessFlags;
import club.doki7.vulkan.bitmask.VkImageAspectFlags;
import club.doki7.vulkan.bitmask.VkPipelineStageFlags;
import club.doki7.vulkan.command.VulkanLoader;
import club.doki7.vulkan.datatype.VkCommandBufferBeginInfo;
import club.doki7.vulkan.datatype.VkImageMemoryBarrier;
import club.doki7.vulkan.datatype.VkSubmitInfo;
import club.doki7.vulkan.enumtype.VkImageLayout;

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

        RenderWindow window = new RenderWindow(glfw, "CG-112", 800, 600);
        RenderContext cx = RenderContext.create(glfw, window.rawWindow, config);
        Swapchain swapchain = Swapchain.create(cx, 800, 600);

        int currentFrame = 0;
        IntPtr pImageIndex = IntPtr.allocate(cx.prefabArena);
        VkCommandBufferBeginInfo cmdBufBeginInfo = VkCommandBufferBeginInfo.allocate(cx.prefabArena);
        VkSubmitInfo submitInfo = VkSubmitInfo.allocate(cx.prefabArena)
                .waitSemaphoreCount(1)
                .pWaitDstStageMask(IntPtr.allocateV(cx.prefabArena, VkPipelineStageFlags.TOP_OF_PIPE))
                .commandBufferCount(1)
                .signalSemaphoreCount(1);
        VkImageMemoryBarrier imageMemoryBarrier = VkImageMemoryBarrier.allocate(cx.prefabArena)
                .srcAccessMask(0)
                .dstAccessMask(VkAccessFlags.COLOR_ATTACHMENT_WRITE)
                .oldLayout(VkImageLayout.UNDEFINED)
                .newLayout(VkImageLayout.PRESENT_SRC_KHR)
                .srcQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                .image(swapchain.pSwapchainImages.read())
                .subresourceRange(it -> it
                        .aspectMask(VkImageAspectFlags.COLOR)
                        .baseMipLevel(0)
                        .levelCount(1)
                        .baseArrayLayer(0)
                        .layerCount(1));
        while (window.tick()) {
            if (window.framebufferResized) {
                cx.waitDeviceIdle();
                swapchain.close();
                swapchain = Swapchain.create(cx, window.width, window.height);
                window.framebufferResized = false;
            }
        }
        cx.waitDeviceIdle();

        swapchain.close();
        cx.close();
        window.close();
    }
}
