package club.doki7.rkt.vk;

import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.init.SwapchainInit;
import club.doki7.rkt.vk.sync.Fence;
import club.doki7.rkt.vk.sync.SemaphoreVK;
import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.vulkan.datatype.*;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.*;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.util.logging.Logger;

public final class Swapchain implements AutoCloseable {
    public final RenderContext cx;

    public final VkSurfaceFormatKHR surfaceFormat;
    public final VkExtent2D swapExtent;

    public final VkSwapchainKHR vkSwapchain;
    public final VkImage.Ptr pSwapchainImages;
    public final VkImageView.Ptr pSwapchainImageViews;

    public Swapchain(
            RenderContext cx,
            VkSurfaceFormatKHR surfaceFormat,
            VkExtent2D swapExtent,
            VkSwapchainKHR vkSwapchain,
            VkImage.Ptr pSwapchainImages,
            VkImageView.Ptr pSwapchainImageViews
    ) {
        this.cx = cx;
        this.surfaceFormat = surfaceFormat;
        this.swapExtent = swapExtent;
        this.vkSwapchain = vkSwapchain;
        this.pSwapchainImages = pSwapchainImages;
        this.pSwapchainImageViews = pSwapchainImageViews;

        this.presentInfo = VkPresentInfoKHR.allocate(cx.prefabArena)
                .swapchainCount(1)
                .pSwapchains(VkSwapchainKHR.Ptr.allocateV(cx.prefabArena, vkSwapchain));
    }

    public static Swapchain create(RenderContext cx, int width, int height) throws RenderException {
        return new SwapchainInit(cx).init(width, height);
    }

    public enum AcquireResult {
        OPTIMAL    (/*isSuccess=*/true,  /*isOptimal=*/true),
        SUBOPTIMAL (/*isSuccess=*/true,  /*isOptimal=*/false),
        OUTDATED   (/*isSuccess=*/false, /*isOptimal=*/false);

        public final boolean isSuccess;
        public final boolean isOptimal;

        AcquireResult(boolean isSuccess, boolean isOptimal) {
            this.isSuccess = isSuccess;
            this.isOptimal = isOptimal;
        }
    }

    public AcquireResult acquireNextImage(
            IntPtr pImageIndex,
            @Nullable SemaphoreVK signalSemaphore,
            @Nullable Fence fence
    ) throws VulkanException {
        @EnumType(VkResult.class) int result = cx.dCmd.acquireNextImageKHR(
                cx.device,
                vkSwapchain,
                NativeLayout.UINT64_MAX,
                signalSemaphore != null ? signalSemaphore.handle : null,
                fence != null ? fence.handle : null,
                pImageIndex
        );
        return switch (result) {
            case VkResult.SUCCESS -> AcquireResult.OPTIMAL;
            case VkResult.SUBOPTIMAL_KHR -> AcquireResult.SUBOPTIMAL;
            case VkResult.ERROR_OUT_OF_DATE_KHR -> AcquireResult.OUTDATED;
            default -> throw new VulkanException(result, "获取交换链图像索引失败");
        };
    }

    public AcquireResult acquireNextImage(IntPtr pImageIndex, Fence signalFence) throws RenderException {
        return acquireNextImage(pImageIndex, null, signalFence);
    }

    public AcquireResult acquireNextImage(IntPtr pImageIndex, SemaphoreVK signalSemaphore) throws RenderException {
        return acquireNextImage(pImageIndex, signalSemaphore, null);
    }

    public enum PresentResult { SUCCESS, SUBOPTIMAL }

    public PresentResult present(
            SemaphoreVK waitSemaphore,
            IntPtr pImageIndex
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            presentInfo
                    .waitSemaphoreCount(1)
                    .pWaitSemaphores(VkSemaphore.Ptr.allocateV(arena, waitSemaphore.handle))
                    .pImageIndices(pImageIndex);

            cx.presentQueueLock.lock();
            @EnumType(VkResult.class) int result =
                    cx.dCmd.queuePresentKHR(cx.presentQueue, presentInfo);
            return switch (result) {
                case VkResult.SUCCESS -> PresentResult.SUCCESS;
                case VkResult.SUBOPTIMAL_KHR -> PresentResult.SUBOPTIMAL;
                default -> throw new VulkanException(result, "提交交换链图像失败");
            };
        } finally {
            cx.presentQueueLock.unlock();
        }
    }

    @Override
    public void close() {
        cx.waitDeviceIdle();
        for (VkImageView imageView : pSwapchainImageViews) {
            cx.dCmd.destroyImageView(cx.device, imageView, null);
        }
        cx.dCmd.destroySwapchainKHR(cx.device, vkSwapchain, null);

        logger.info("已销毁交换链及其相关资源");
    }

    private final VkPresentInfoKHR presentInfo;

    private static final Logger logger = Logger.getLogger(Swapchain.class.getName());
}
