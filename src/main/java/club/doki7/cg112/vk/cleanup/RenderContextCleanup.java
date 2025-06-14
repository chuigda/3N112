package club.doki7.cg112.vk.cleanup;

import club.doki7.vma.VMA;
import club.doki7.vma.handle.VmaAllocator;
import club.doki7.vulkan.command.VkDeviceCommands;
import club.doki7.vulkan.command.VkInstanceCommands;
import club.doki7.vulkan.handle.*;
import org.jetbrains.annotations.Nullable;

import java.util.logging.Logger;

public final class RenderContextCleanup implements IDisposable {
    public RenderContextCleanup(
            VkInstanceCommands iCmd,
            VkDeviceCommands dCmd,
            VMA vma,

            VkInstance instance,
            @Nullable VkDebugUtilsMessengerEXT debugMessenger,
            VkSurfaceKHR surface,

            VkDevice device,

            VmaAllocator vmaAllocator,

            VkSemaphore.Ptr pImageAvailableSemaphores,
            @Nullable VkSemaphore.Ptr pComputeFinishedSemaphores,
            VkFence.Ptr pInFlightFences,

            VkCommandPool graphicsCommandPool,
            VkCommandPool graphicsOnceCommandPool,
            @Nullable VkCommandPool transferCommandPool,
            @Nullable VkCommandPool computeCommandPool,
            @Nullable VkCommandPool computeOnceCommandPool
    ) {
        this.iCmd = iCmd;
        this.dCmd = dCmd;
        this.vma = vma;
        this.instance = instance;
        this.debugMessenger = debugMessenger;
        this.surface = surface;
        this.device = device;
        this.vmaAllocator = vmaAllocator;
        this.pImageAvailableSemaphores = pImageAvailableSemaphores;
        this.pComputeFinishedSemaphores = pComputeFinishedSemaphores;
        this.pInFlightFences = pInFlightFences;
        this.graphicsCommandPool = graphicsCommandPool;
        this.graphicsOnceCommandPool = graphicsOnceCommandPool;
        this.transferCommandPool = transferCommandPool;
        this.computeCommandPool = computeCommandPool;
        this.computeOnceCommandPool = computeOnceCommandPool;
    }

    @Override
    public void dispose() {
        // command pools
        dCmd.destroyCommandPool(device, graphicsCommandPool, null);
        dCmd.destroyCommandPool(device, graphicsOnceCommandPool, null);
        if (transferCommandPool != null) {
            dCmd.destroyCommandPool(device, transferCommandPool, null);
        }
        if (computeCommandPool != null) {
            dCmd.destroyCommandPool(device, computeCommandPool, null);
        }
        if (computeOnceCommandPool != null) {
            dCmd.destroyCommandPool(device, computeOnceCommandPool, null);
        }

        // synchronization objects
        for (VkSemaphore semaphore : pImageAvailableSemaphores) {
            dCmd.destroySemaphore(device, semaphore, null);
        }
        for (VkFence fence : pInFlightFences) {
            dCmd.destroyFence(device, fence, null);
        }
        if (pComputeFinishedSemaphores != null) {
            for (VkSemaphore semaphore : pComputeFinishedSemaphores) {
                dCmd.destroySemaphore(device, semaphore, null);
            }
        }

        // VMA allocator
        vma.destroyAllocator(vmaAllocator);

        // device
        dCmd.destroyDevice(device, null);

        // window surface
        iCmd.destroySurfaceKHR(instance, surface, null);

        // debug messenger
        if (debugMessenger != null) {
            iCmd.destroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
        }

        // instance
        iCmd.destroyInstance(instance, null);

        logger.info("已销毁 RenderContext 资源");
    }

    private final VkInstanceCommands iCmd;
    private final VkDeviceCommands dCmd;
    private final VMA vma;

    private final VkInstance instance;
    private final @Nullable VkDebugUtilsMessengerEXT debugMessenger;
    private final VkSurfaceKHR surface;

    private final VkDevice device;

    private final VmaAllocator vmaAllocator;

    private final VkSemaphore.Ptr pImageAvailableSemaphores;
    private final @Nullable VkSemaphore.Ptr pComputeFinishedSemaphores;
    private final VkFence.Ptr pInFlightFences;

    private final VkCommandPool graphicsCommandPool;
    private final VkCommandPool graphicsOnceCommandPool;
    private final @Nullable VkCommandPool transferCommandPool;
    private final @Nullable VkCommandPool computeCommandPool;
    private final @Nullable VkCommandPool computeOnceCommandPool;

    private static final Logger logger = Logger.getLogger(RenderContextCleanup.class.getName());
}
