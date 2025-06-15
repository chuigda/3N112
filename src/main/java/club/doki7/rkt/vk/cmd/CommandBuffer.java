package club.doki7.rkt.vk.cmd;

import club.doki7.rkt.vk.RenderContext;
import club.doki7.vulkan.handle.VkCommandBuffer;

public final class CommandBuffer {
    public final VkCommandBuffer handle;
    public final boolean canReset;

    public void reset(RenderContext cx) {
        if (!canReset) {
            throw new IllegalStateException("This command buffer cannot be reset.");
        }
        cx.dCmd.resetCommandBuffer(handle, 0);
    }

    CommandBuffer(VkCommandBuffer handle, boolean canReset, CommandPool commandPool) {
        this.handle = handle;
        this.canReset = canReset;
        this.commandPool = commandPool;
    }

    @SuppressWarnings({"unused", "FieldCanBeLocal"})
    private final CommandPool commandPool;
}
