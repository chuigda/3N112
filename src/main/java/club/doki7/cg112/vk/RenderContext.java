package club.doki7.cg112.vk;

import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.vk.cleanup.RenderContextCleanup;
import club.doki7.cg112.vk.init.ContextInit;
import club.doki7.glfw.GLFW;
import club.doki7.glfw.handle.GLFWwindow;
import club.doki7.vma.VMA;
import club.doki7.vma.handle.VmaAllocator;
import club.doki7.vulkan.command.VkDeviceCommands;
import club.doki7.vulkan.command.VkEntryCommands;
import club.doki7.vulkan.command.VkInstanceCommands;
import club.doki7.vulkan.command.VkStaticCommands;
import club.doki7.vulkan.handle.*;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public final class RenderContext implements AutoCloseable {
    public final Arena prefabArena;
    public final RenderConfig config;
    public final VkStaticCommands sCmd;
    public final VkEntryCommands eCmd;
    public final VkInstanceCommands iCmd;
    public final VkDeviceCommands dCmd;
    public final VMA vma;

    public final VkPhysicalDevice physicalDevice;
    public final int graphicsQueueFamilyIndex;
    public final int presentQueueFamilyIndex;
    public final int dedicatedTransferQueueFamilyIndex;
    public final int dedicatedComputeQueueFamilyIndex;

    public final VkInstance instance;
    public final @Nullable VkDebugUtilsMessengerEXT debugMessenger;
    public final VkSurfaceKHR surface;

    public final VkDevice device;
    public final VkQueue graphicsQueue;
    public final VkQueue presentQueue;
    public final @Nullable VkQueue transferQueue;
    public final @Nullable VkQueue computeQueue;

    public final VmaAllocator vmaAllocator;

    public final VkSemaphore.Ptr pImageAvailableSemaphores;
    public final @Nullable VkSemaphore.Ptr pComputeFinishedSemaphores;
    public final VkFence.Ptr pInFlightFences;

    public final VkCommandPool graphicsCommandPool;
    public final VkCommandPool graphicsOnceCommandPool;
    public final @Nullable VkCommandPool transferCommandPool;
    public final @Nullable VkCommandPool computeCommandPool;
    public final @Nullable VkCommandPool computeOnceCommandPool;

    public final VkCommandBuffer.Ptr graphicsCommandBuffers;
    public final @Nullable VkCommandBuffer.Ptr computeCommandBuffers;

    public final Lock graphicsQueueLock;
    public final Lock presentQueueLock;
    public final @Nullable Lock transferQueueLock;
    public final @Nullable Lock computeQueueLock;

    public RenderContext(
            Arena prefabArena,
            RenderConfig config,

            VkStaticCommands sCmd,
            VkEntryCommands eCmd,
            VkInstanceCommands iCmd,
            VkDeviceCommands dCmd,
            VMA vma,

            VkPhysicalDevice physicalDevice,
            int graphicsQueueFamilyIndex,
            int presentQueueFamilyIndex,
            int dedicatedTransferQueueFamilyIndex,
            int dedicatedComputeQueueFamilyIndex,

            VkInstance instance,
            @Nullable VkDebugUtilsMessengerEXT debugMessenger,
            VkSurfaceKHR surface,

            VkDevice device,
            VkQueue graphicsQueue,
            VkQueue presentQueue,
            @Nullable VkQueue transferQueue,
            @Nullable VkQueue computeQueue,

            VmaAllocator vmaAllocator,

            VkSemaphore.Ptr pImageAvailableSemaphores,
            @Nullable VkSemaphore.Ptr pComputeFinishedSemaphores,
            VkFence.Ptr pInFlightFences,

            VkCommandPool graphicsCommandPool,
            VkCommandPool graphicsOnceCommandPool,
            @Nullable VkCommandPool transferCommandPool,
            @Nullable VkCommandPool computeCommandPool,
            @Nullable VkCommandPool computeOnceCommandPool,

            VkCommandBuffer.Ptr graphicsCommandBuffers,
            @Nullable VkCommandBuffer.Ptr computeCommandBuffers
    ) {
        this.prefabArena = prefabArena;
        this.config = config;

        this.sCmd = sCmd;
        this.eCmd = eCmd;
        this.iCmd = iCmd;
        this.dCmd = dCmd;
        this.vma = vma;

        this.physicalDevice = physicalDevice;
        this.graphicsQueueFamilyIndex = graphicsQueueFamilyIndex;
        this.presentQueueFamilyIndex = presentQueueFamilyIndex;
        this.dedicatedTransferQueueFamilyIndex = dedicatedTransferQueueFamilyIndex;
        this.dedicatedComputeQueueFamilyIndex = dedicatedComputeQueueFamilyIndex;

        this.instance = instance;
        this.debugMessenger = debugMessenger;
        this.surface = surface;

        this.device = device;
        this.graphicsQueue = graphicsQueue;
        this.presentQueue = presentQueue;
        this.transferQueue = transferQueue;
        this.computeQueue = computeQueue;

        this.vmaAllocator = vmaAllocator;

        this.pImageAvailableSemaphores = pImageAvailableSemaphores;
        this.pComputeFinishedSemaphores = pComputeFinishedSemaphores;
        this.pInFlightFences = pInFlightFences;

        this.graphicsCommandPool = graphicsCommandPool;
        this.graphicsOnceCommandPool = graphicsOnceCommandPool;
        this.transferCommandPool = transferCommandPool;
        this.computeCommandPool = computeCommandPool;
        this.computeOnceCommandPool = computeOnceCommandPool;

        this.graphicsCommandBuffers = graphicsCommandBuffers;
        this.computeCommandBuffers = computeCommandBuffers;

        this.graphicsQueueLock = new ReentrantLock();
        if (this.graphicsQueue != this.presentQueue) {
            this.presentQueueLock = new ReentrantLock();
        } else {
            this.presentQueueLock = graphicsQueueLock;
        }

        if (transferQueue != null) {
            this.transferQueueLock = new ReentrantLock();
        } else {
            this.transferQueueLock = null;
        }
        if (computeQueue != null) {
            this.computeQueueLock = new ReentrantLock();
        } else {
            this.computeQueueLock = null;
        }

        RenderContextCleanup cleanup = new RenderContextCleanup(
                iCmd,
                dCmd,
                vma,

                instance,
                debugMessenger,
                surface,

                device,

                vmaAllocator,

                pImageAvailableSemaphores,
                pComputeFinishedSemaphores,
                pInFlightFences,

                graphicsCommandPool,
                graphicsOnceCommandPool,
                transferCommandPool,
                computeCommandPool,
                computeOnceCommandPool
        );
        cleanable = cleaner.register(this, cleanup::dispose);
    }

    public boolean hasTransferQueue() {
        return transferQueue != null;
    }

    public void waitDeviceIdle() {
        try {
            graphicsQueueLock.lock();
            if (graphicsQueueLock != presentQueueLock) {
                presentQueueLock.lock();
            }
            if (transferQueueLock != null) {
                transferQueueLock.lock();
            }
            if (computeQueueLock != null) {
                computeQueueLock.lock();
            }

            dCmd.deviceWaitIdle(device);
        } finally {
            graphicsQueueLock.unlock();
            if (graphicsQueueLock != presentQueueLock) {
                presentQueueLock.unlock();
            }
            if (transferQueueLock != null) {
                transferQueueLock.unlock();
            }
            if (computeQueueLock != null) {
                computeQueueLock.unlock();
            }
        }
    }

    public static RenderContext create(
            GLFW glfw,
            GLFWwindow window,
            RenderConfig config
    ) throws RenderException {
        return new ContextInit(glfw, window, config).init();
    }

    @Override
    public void close() {
        this.cleanable.clean();
    }

    private final Cleaner.Cleanable cleanable;
    private static final Cleaner cleaner = Cleaner.create();
}
