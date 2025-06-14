package club.doki7.cg112.vk;

import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.vk.init.SwapchainInit;
import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.vulkan.datatype.*;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.*;
import org.jetbrains.annotations.Nullable;

import java.util.logging.Logger;

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

        this.presentInfo = VkPresentInfoKHR.allocate(cx.prefabArena)
                .waitSemaphoreCount(1)
                .swapchainCount(1)
                .pSwapchains(VkSwapchainKHR.Ptr.allocateV(cx.prefabArena, vkSwapchain));
    }

    public static Swapchain create(RenderContext cx, int width, int height) throws RenderException {
        return new SwapchainInit(cx).init(width, height);
    }

    public @EnumType(VkResult.class) int acquireNextImage(
            IntPtr pImageIndex,
            @Nullable VkSemaphore signalSemaphore
    ) {
        return cx.dCmd.acquireNextImageKHR(
                cx.device,
                vkSwapchain,
                NativeLayout.UINT64_MAX,
                signalSemaphore,
                null,
                pImageIndex
        );
    }

    public @EnumType(VkResult.class) int present(IntPtr pImageIndex) {
        presentInfo
                .pWaitSemaphores(pRenderFinishedSemaphores.offset(pImageIndex.read()))
                .pImageIndices(pImageIndex);
        try {
            cx.presentQueueLock.lock();
            return cx.dCmd.queuePresentKHR(cx.presentQueue, presentInfo);
        } finally {
            cx.presentQueueLock.unlock();
        }
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

        logger.info("已销毁交换链及其相关资源");
    }

    private final VkPresentInfoKHR presentInfo;

    private static final Logger logger = Logger.getLogger(Swapchain.class.getName());
}
