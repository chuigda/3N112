package club.doki7.cg112.vk;

import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.util.Pair;
import club.doki7.cg112.util.Ref;
import club.doki7.cg112.vk.init.ContextInit;
import club.doki7.ffm.annotation.Unsafe;
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
import java.util.LinkedList;
import java.util.concurrent.BlockingQueue;
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
    public final VkQueue graphicsQueue;
    public final VkQueue presentQueue;
    public final @Nullable VkQueue transferQueue;
    public final @Nullable VkQueue computeQueue;

    public final VmaAllocator vmaAllocator;

    public final Lock graphicsQueueLock;
    public final Lock presentQueueLock;
    public final @Nullable Lock transferQueueLock;
    public final @Nullable Lock computeQueueLock;

    public final Cleaner cleaner = Cleaner.create();

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

    private final Ref<LinkedList<IDisposeOnContext>> disposeList;
    private final LinkedList<Pair<IDisposeOnContext, Integer>> countedList;
    private final BlockingQueue<IDisposeOnContext> gcQueue;
    private final Thread gcThread;
    private static final Logger logger = Logger.getLogger(RenderContext.class.getName());
}
