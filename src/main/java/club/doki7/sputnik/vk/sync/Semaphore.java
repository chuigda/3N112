package club.doki7.sputnik.vk.sync;

import club.doki7.sputnik.exc.VulkanException;
import club.doki7.sputnik.vk.IDisposeOnContext;
import club.doki7.sputnik.vk.RenderContext;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.annotation.Unsafe;
import club.doki7.vulkan.datatype.VkSemaphoreCreateInfo;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.VkSemaphore;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;

public final class Semaphore implements AutoCloseable {
    public final VkSemaphore handle;

    public static Semaphore create(RenderContext cx) throws VulkanException {
        return create(cx, false);
    }

    @Unsafe
    public static Semaphore createLocal(RenderContext cx) throws VulkanException {
        return create(cx, true);
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private static Semaphore create(RenderContext cx, boolean local) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkSemaphore.Ptr pSemaphore = VkSemaphore.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result =
                    cx.dCmd.createSemaphore(cx.device, createInfo, null, pSemaphore);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 信号量");
            }
            return new Semaphore(pSemaphore.read(), cx, local);
        }
    }

    private Semaphore(VkSemaphore handle, RenderContext context, boolean local) {
        this.handle = handle;
        IDisposeOnContext d = cx -> cx.dCmd.destroySemaphore(cx.device, handle, null);
        if (local) {
            this.cleanable = context.cleaner.register(this, () -> context.disposeImmediate(d));
        } else {
            this.cleanable = context.cleaner.register(this, () -> context.dispose(d));
        }
    }

    private final Cleaner.Cleanable cleanable;
    private static final VkSemaphoreCreateInfo createInfo =
            VkSemaphoreCreateInfo.allocate(Arena.global());
}
