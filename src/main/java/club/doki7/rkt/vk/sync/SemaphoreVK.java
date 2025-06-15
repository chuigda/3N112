package club.doki7.rkt.vk.sync;

import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.annotation.Unsafe;
import club.doki7.vulkan.datatype.VkSemaphoreCreateInfo;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.VkSemaphore;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;

public final class SemaphoreVK implements AutoCloseable {
    public final VkSemaphore handle;

    public static SemaphoreVK create(RenderContext cx) throws VulkanException {
        return create(cx, false);
    }

    @Unsafe
    public static SemaphoreVK createLocal(RenderContext cx) throws VulkanException {
        return create(cx, true);
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private static SemaphoreVK create(RenderContext cx, boolean local) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkSemaphore.Ptr pSemaphore = VkSemaphore.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result =
                    cx.dCmd.createSemaphore(cx.device, createInfo, null, pSemaphore);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 信号量");
            }
            return new SemaphoreVK(pSemaphore.read(), cx, local);
        }
    }

    private SemaphoreVK(VkSemaphore handle, RenderContext context, boolean local) {
        this.handle = handle;

        IDisposeOnContext d = cx -> cx.dCmd.destroySemaphore(cx.device, handle, null);
        this.cleanable = context.registerCleanup(this, d, local);
    }

    private final Cleaner.Cleanable cleanable;
    private static final VkSemaphoreCreateInfo createInfo =
            VkSemaphoreCreateInfo.allocate(Arena.global());
}
