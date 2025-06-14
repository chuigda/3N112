package club.doki7.cg112.vk;

import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.vk.init.SwapchainInit;
import club.doki7.vulkan.datatype.*;
import club.doki7.vulkan.handle.*;

public final class Swapchain implements AutoCloseable {
    public final RenderContext cx;

    public final VkSurfaceFormatKHR surfaceFormat;
    public final VkExtent2D swapExtent;

    public final VkSwapchainKHR vkSwapchain;
    public final VkImage.Ptr pSwapchainImages;
    public final VkImageView.Ptr pSwapchainImageViews;
    public final VkSemaphore.Ptr pRenderFinishedSemaphores;

    public Swapchain(
            RenderContext cx,
            VkSurfaceFormatKHR surfaceFormat,
            VkExtent2D swapExtent,
            VkSwapchainKHR vkSwapchain,
            VkImage.Ptr pSwapchainImages,
            VkImageView.Ptr pSwapchainImageViews,
            VkSemaphore.Ptr pRenderFinishedSemaphores
    ) {
        this.cx = cx;
        this.surfaceFormat = surfaceFormat;
        this.swapExtent = swapExtent;
        this.vkSwapchain = vkSwapchain;
        this.pSwapchainImages = pSwapchainImages;
        this.pSwapchainImageViews = pSwapchainImageViews;
        this.pRenderFinishedSemaphores = pRenderFinishedSemaphores;
    }

    public static Swapchain create(RenderContext cx, int width, int height) throws RenderException {
        return new SwapchainInit(cx).init(width, height);
    }

    @Override
    public void close() {
        for (VkSemaphore semaphore : pRenderFinishedSemaphores) {
            cx.dCmd.destroySemaphore(cx.device, semaphore, null);
        }
        for (VkImageView imageView : pSwapchainImageViews) {
            cx.dCmd.destroyImageView(cx.device, imageView, null);
        }
        cx.dCmd.destroySwapchainKHR(cx.device, vkSwapchain, null);
    }
}
