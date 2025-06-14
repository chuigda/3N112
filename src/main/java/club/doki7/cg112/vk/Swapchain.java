package club.doki7.cg112.vk;

import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.vk.cleanup.IDisposable;
import club.doki7.cg112.vk.cleanup.SwapchainCleanup;
import club.doki7.cg112.vk.init.SwapchainInit;
import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.vulkan.datatype.*;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.*;

import java.lang.ref.Cleaner;

public final class Swapchain implements IDisposable {
    public final RenderContext cx;

    public final VkSurfaceFormatKHR surfaceFormat;
    public final VkExtent2D swapExtent;

    public final VkSwapchainKHR vkSwapchain;
    public final VkImage.Ptr pSwapchainImages;
    public final VkImageView.Ptr pSwapchainImageViews;
    public final VkSemaphore.Ptr pRenderFinishedSemaphores;

    private final Cleaner.Cleanable cleanable;

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

        SwapchainCleanup cleanup = new SwapchainCleanup(
                cx,
                vkSwapchain,
                pSwapchainImageViews,
                pRenderFinishedSemaphores
        );
        this.cleanable = cleaner.register(this, cleanup::dispose);
    }

    public static Swapchain create(RenderContext cx, int width, int height) throws RenderException {
        return new SwapchainInit(cx).init(width, height);
    }

    public @EnumType(VkResult.class) int acquireNextImage(
            IntPtr pImageIndex,
            VkSemaphore signalSemaphore
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

    @Override
    public void dispose() {
        this.cleanable.clean();
    }

    private static final Cleaner cleaner = Cleaner.create();
}
