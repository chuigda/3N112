package club.doki7.cg112.vk.cleanup;

import club.doki7.cg112.vk.RenderContext;
import club.doki7.vulkan.handle.VkImageView;
import club.doki7.vulkan.handle.VkSemaphore;
import club.doki7.vulkan.handle.VkSwapchainKHR;

import java.util.logging.Logger;

public final class SwapchainCleanup implements IDisposable {
    public SwapchainCleanup(
            RenderContext cx,
            VkSwapchainKHR vkSwapchain,
            VkImageView.Ptr pSwapchainImageViews,
            VkSemaphore.Ptr pRenderFinishedSemaphores
    ) {
        this.cx = cx;
        this.vkSwapchain = vkSwapchain;
        this.pSwapchainImageViews = pSwapchainImageViews;
        this.pRenderFinishedSemaphores = pRenderFinishedSemaphores;
    }

    @Override
    public void dispose() {
        for (VkSemaphore semaphore : pRenderFinishedSemaphores) {
            cx.dCmd.destroySemaphore(cx.device, semaphore, null);
        }
        for (VkImageView imageView : pSwapchainImageViews) {
            cx.dCmd.destroyImageView(cx.device, imageView, null);
        }
        cx.dCmd.destroySwapchainKHR(cx.device, vkSwapchain, null);

        logger.info("已销毁交换链及其相关资源");
    }

    private final RenderContext cx;

    private final VkSwapchainKHR vkSwapchain;
    private final VkImageView.Ptr pSwapchainImageViews;
    private final VkSemaphore.Ptr pRenderFinishedSemaphores;

    private static final Logger logger = Logger.getLogger(SwapchainCleanup.class.getName());
}
