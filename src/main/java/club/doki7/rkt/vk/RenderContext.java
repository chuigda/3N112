package club.doki7.rkt.vk;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.util.Pair;
import club.doki7.rkt.util.Ref;
import club.doki7.rkt.vk.cmd.SubmitInfo;
import club.doki7.rkt.vk.init.ContextInit;
import club.doki7.ffm.annotation.Unsafe;
import club.doki7.glfw.GLFW;
import club.doki7.glfw.handle.GLFWwindow;
import club.doki7.vma.VMA;
import club.doki7.vma.handle.VmaAllocator;
import club.doki7.vulkan.bitmask.VkPipelineStageFlags;
import club.doki7.vulkan.command.VkDeviceCommands;
import club.doki7.vulkan.command.VkEntryCommands;
import club.doki7.vulkan.command.VkInstanceCommands;
import club.doki7.vulkan.command.VkStaticCommands;
import club.doki7.vulkan.datatype.VkSubmitInfo;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.*;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;
import java.util.LinkedList;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Logger;

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
    public final VmaAllocator vmaAllocator;

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

            VmaAllocator vmaAllocator
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

        this.cleaner = Cleaner.create();
        this.cleanables = new ConcurrentHashMap<>();
        this.disposeList = new Ref<>(new LinkedList<>());
        this.countedList = new LinkedList<>();
        this.gcQueue = new LinkedBlockingQueue<>();
        this.gcThread = new Thread(() -> {
            Logger logger = Logger.getLogger(Thread.currentThread().getName());
            while (true) {
                try {
                    @Nullable IDisposeOnContext item = gcQueue.take();
                    if (item == IDisposeOnContext.POISON) {
                        break;
                    }
                    try {
                        item.disposeOnContext(this);
                    } catch (Throwable e) {
                        logger.severe("在 RenderContext GC 线程中处理对象 " + item + " 时发生异常: " + e.getMessage());
                        e.printStackTrace(System.err);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            logger.info("RenderContext GC 线程已退出");
        }, "RenderContext-GC-Thread");
        this.gcThread.start();
    }

    public boolean hasTransferQueue() {
        return transferQueue != null;
    }

    public boolean hasComputeQueue() {
        return computeQueue != null;
    }

    public void submitGraphics(SubmitInfo info, @Nullable VkFence fence) throws VulkanException {
        submitToQueue(graphicsQueue, graphicsQueueLock, info, fence);
    }

    public void submitCompute(SubmitInfo info, @Nullable VkFence fence) throws VulkanException {
        if (!hasComputeQueue()) {
            throw new IllegalStateException("没有可用的计算队列");
        }
        submitToQueue(computeQueue, computeQueueLock, info, fence);
    }

    public void submitTransfer(SubmitInfo info, @Nullable VkFence fence) throws VulkanException {
        if (!hasTransferQueue()) {
            throw new IllegalStateException("没有可用的传输队列");
        }
        submitToQueue(transferQueue, transferQueueLock, info, fence);
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

    public Cleaner.Cleanable registerCleanup(Object item, IDisposeOnContext dispose, boolean local) {
        Cleaner.Cleanable cleanable;
        if (local) {
            cleanable = cleaner.register(item, () -> {
                disposeImmediate(dispose);
                cleanables.remove(dispose);
            });
        } else {
            cleanable = cleaner.register(item, () -> {
                dispose(dispose);
                cleanables.remove(dispose);
            });
        }
        cleanables.put(dispose, cleanable);
        return cleanable;
    }

    public void dispose(IDisposeOnContext item) {
        synchronized (disposeList) {
            disposeList.value.add(item);
        }
    }

    @Unsafe
    public void disposeImmediate(IDisposeOnContext item) {
        boolean result = gcQueue.offer(item);
        assert result;
    }

    public void gc() {
        countedList.removeIf(it -> {
            if (it.second >= config.maxFramesInFlight + 1) {
                it.first.disposeOnContext(this);
                return true;
            } else {
                return false;
            }
        });

        for (Pair<IDisposeOnContext, Integer> item : countedList) {
            item.second += 1;
        }

        LinkedList<IDisposeOnContext> list;
        synchronized (disposeList) {
            list = disposeList.value;
            disposeList.value = new LinkedList<>();
        }

        for (IDisposeOnContext item : list) {
            countedList.add(new Pair<>(item, 0));
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
        waitDeviceIdle();

        for (Cleaner.Cleanable cleanable : cleanables.values()) {
            cleanable.clean();
        }

        for (IDisposeOnContext item : disposeList.value) {
            boolean result = gcQueue.offer(item);
            assert result;
        }
        for (Pair<IDisposeOnContext, Integer> item : countedList) {
            boolean result = gcQueue.offer(item.first);
            assert result;
        }
        boolean result = gcQueue.offer(IDisposeOnContext.POISON);
        assert result;
        try {
            gcThread.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.warning("RenderContext GC 线程被中断: "  + e.getMessage());
        }

        vma.destroyAllocator(vmaAllocator);

        dCmd.destroyDevice(device, null);

        iCmd.destroySurfaceKHR(instance, surface, null);

        if (debugMessenger != null) {
            iCmd.destroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
        }

        iCmd.destroyInstance(instance, null);

        logger.info("已销毁 RenderContext 资源");
    }

    private void submitToQueue(
            VkQueue queue,
            Lock queueLock,
            SubmitInfo info,
            @Nullable VkFence fence
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkCommandBuffer.Ptr pCommandBuffers =
                    VkCommandBuffer.Ptr.allocate(arena, info.commandBuffers.size());
            for (int i = 0; i < info.commandBuffers.size(); i++) {
                pCommandBuffers.write(i, info.commandBuffers.get(i).handle);
            }
            VkSemaphore.Ptr pWaitSemaphores =
                    VkSemaphore.Ptr.allocate(arena, info.waitSemaphores.size());
            for (int i = 0; i < info.waitSemaphores.size(); i++) {
                pWaitSemaphores.write(i, info.waitSemaphores.get(i).handle);
            }
            @EnumType(VkPipelineStageFlags.class) IntPtr pWaitDstStageMask =
                    IntPtr.allocate(arena, info.waitDstStageMasks.size());
            for (int i = 0; i < info.waitDstStageMasks.size(); i++) {
                pWaitDstStageMask.write(i, info.waitDstStageMasks.get(i));
            }
            VkSemaphore.Ptr pSignalSemaphores =
                    VkSemaphore.Ptr.allocate(arena, info.signalSemaphores.size());
            for (int i = 0; i < info.signalSemaphores.size(); i++) {
                pSignalSemaphores.write(i, info.signalSemaphores.get(i).handle);
            }

            VkSubmitInfo submitInfoVk = VkSubmitInfo.allocate(arena)
                    .commandBufferCount(info.commandBuffers.size())
                    .pCommandBuffers(pCommandBuffers)
                    .waitSemaphoreCount(info.waitSemaphores.size())
                    .pWaitSemaphores(pWaitSemaphores)
                    .pWaitDstStageMask(pWaitDstStageMask)
                    .signalSemaphoreCount(info.signalSemaphores.size())
                    .pSignalSemaphores(pSignalSemaphores);

            queueLock.lock();
            @EnumType(VkResult.class) int result = dCmd.queueSubmit(queue, 1, submitInfoVk, fence);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法提交到图形队列");
            }
        } finally {
            queueLock.unlock();
        }
    }

    final VkQueue graphicsQueue;
    final VkQueue presentQueue;
    final @Nullable VkQueue transferQueue;
    final @Nullable VkQueue computeQueue;
    final Lock graphicsQueueLock;
    final Lock presentQueueLock;
    final @Nullable Lock transferQueueLock;
    final @Nullable Lock computeQueueLock;

    private final Cleaner cleaner;
    private final ConcurrentHashMap<IDisposeOnContext, Cleaner.Cleanable> cleanables;
    private final Ref<LinkedList<IDisposeOnContext>> disposeList;
    private final LinkedList<Pair<IDisposeOnContext, Integer>> countedList;
    private final BlockingQueue<IDisposeOnContext> gcQueue;
    private final Thread gcThread;
    private static final Logger logger = Logger.getLogger(RenderContext.class.getName());
}
