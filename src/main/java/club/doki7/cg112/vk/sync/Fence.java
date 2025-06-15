package club.doki7.cg112.vk.sync;

import club.doki7.cg112.exc.VulkanException;
import club.doki7.cg112.vk.IDisposeOnContext;
import club.doki7.cg112.vk.RenderContext;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.annotation.Unsafe;
import club.doki7.vulkan.bitmask.VkFenceCreateFlags;
import club.doki7.vulkan.datatype.VkFenceCreateInfo;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.VkFence;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;

public final class Fence implements AutoCloseable {
    public final VkFence handle;

    public static Fence create(
            RenderContext cx,
            @EnumType(VkFenceCreateFlags.class) int flags
    ) throws VulkanException {
        return create(cx, flags, false);
    }

    public static Fence create(RenderContext cx) throws VulkanException {
        return create(cx, 0, false);
    }

    @Unsafe
    public static Fence createLocal(
            RenderContext cx,
            @EnumType(VkFenceCreateFlags.class) int flags
    ) throws VulkanException {
        return create(cx, flags, true);
    }

    @Unsafe
    public static Fence createLocal(RenderContext cx) throws VulkanException {
        return create(cx, 0, true);
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private static Fence create(
            RenderContext cx,
            @EnumType(VkFenceCreateFlags.class) int flags,
            boolean local
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkFence.Ptr pFence = VkFence.Ptr.allocate(arena);
            VkFenceCreateInfo createInfo = VkFenceCreateInfo.allocate(arena).flags(flags);
            @EnumType(VkResult.class) int result =
                    cx.dCmd.createFence(cx.device, createInfo, null, pFence);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 栅栏");
            }
            return new Fence(pFence.read(), cx, local);
        }
    }

    private Fence(VkFence handle, RenderContext context, boolean local) {
        this.handle = handle;
        IDisposeOnContext d = cx -> cx.dCmd.destroyFence(cx.device, handle, null);
        if (local) {
            this.cleanable = context.cleaner.register(this, () -> context.disposeImmediate(d));
        } else {
            this.cleanable = context.cleaner.register(this, () -> context.dispose(d));
        }
    }

    private final Cleaner.Cleanable cleanable;
}
