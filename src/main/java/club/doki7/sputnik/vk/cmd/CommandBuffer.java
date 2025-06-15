package club.doki7.sputnik.vk.cmd;

import club.doki7.sputnik.vk.RenderContext;
import club.doki7.vulkan.handle.VkCommandBuffer;

public final class CommandBuffer {
    public final VkCommandBuffer vkCommandBuffer;

    public void reset(RenderContext cx) {
        if (!canReset) {
            throw new IllegalStateException("This command buffer cannot be reset.");
        }
        cx.dCmd.resetCommandBuffer(vkCommandBuffer, 0);
    }

    CommandBuffer(VkCommandBuffer vkCommandBuffer, boolean canReset) {
        this.vkCommandBuffer = vkCommandBuffer;
        this.canReset = canReset;
    }

    private final boolean canReset;
}
