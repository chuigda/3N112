package club.doki7.cg112.drv;

import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.exc.VulkanException;
import club.doki7.cg112.vk.RenderConfig;
import club.doki7.cg112.vk.RenderContext;
import club.doki7.cg112.vk.RenderWindow;
import club.doki7.cg112.vk.Swapchain;
import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.glfw.GLFW;
import club.doki7.glfw.GLFWLoader;
import club.doki7.vulkan.VkConstants;
import club.doki7.vulkan.command.VulkanLoader;
import club.doki7.vulkan.datatype.VkCommandBufferBeginInfo;
import club.doki7.vulkan.datatype.VkSubmitInfo;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.VkCommandBuffer;
import club.doki7.vulkan.handle.VkFence;
import club.doki7.vulkan.handle.VkSemaphore;

import java.util.Objects;

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
                .commandBufferCount(1)
                .signalSemaphoreCount(1);
        while (window.tick()) {
            if (window.framebufferResized) {
                cx.waitDeviceIdle();
                swapchain.close();
                swapchain = Swapchain.create(cx, window.width, window.height);
                window.framebufferResized = false;
            }

            VkFence.Ptr pFence = cx.pInFlightFences.offset(currentFrame);

            cx.dCmd.waitForFences(cx.device, 1, pFence, VkConstants.TRUE, NativeLayout.UINT64_MAX);
            @EnumType(VkResult.class) int result = swapchain.acquireNextImage(
                    pImageIndex,
                    null
            );
            if (result == VkResult.ERROR_OUT_OF_DATE_KHR) {
                continue;
            }
            if (result != VkResult.SUCCESS && result != VkResult.SUBOPTIMAL_KHR) {
                throw new VulkanException(result, "获取交换链图像失败");
            }
            cx.dCmd.resetFences(cx.device, 1, pFence);
            int swapchainImageIndex = pImageIndex.read();
            VkSemaphore.Ptr pRenderFinishedSemaphore =
                    swapchain.pRenderFinishedSemaphores.offset(swapchainImageIndex);

            VkCommandBuffer.Ptr pCmdBuf = cx.graphicsCommandBuffers.offset(currentFrame);
            VkCommandBuffer cmdBuf = Objects.requireNonNull(pCmdBuf.read());
            cx.dCmd.resetCommandBuffer(cmdBuf, 0);
            cx.dCmd.beginCommandBuffer(cmdBuf, cmdBufBeginInfo);
            cx.dCmd.endCommandBuffer(cmdBuf);

            submitInfo
                    .pCommandBuffers(pCmdBuf)
                    .pSignalSemaphores(pRenderFinishedSemaphore);
            try {
                cx.graphicsQueueLock.lock();
                result = cx.dCmd.queueSubmit(cx.graphicsQueue, 1, submitInfo, pFence.read());
            } finally {
                cx.graphicsQueueLock.unlock();
            }
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "提交命令缓冲区失败");
            }

            result = swapchain.present(pImageIndex);
            if (result == VkResult.ERROR_OUT_OF_DATE_KHR || result == VkResult.SUBOPTIMAL_KHR) {
                continue;
            }
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法提交交换链图像");
            }
        }
        cx.waitDeviceIdle();

        cx.close();
        window.close();
    }
}
