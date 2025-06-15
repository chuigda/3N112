package club.doki7.rkt.vk.cmd;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.rkt.vk.sync.SemaphoreVK;
import club.doki7.vulkan.bitmask.VkPipelineStageFlags;

import java.util.List;

public final class SubmitInfo {
    public final List<CommandBuffer> commandBuffers;
    public final List<SemaphoreVK> waitSemaphores;
    public final @EnumType(VkPipelineStageFlags.class) List<Integer> waitDstStageMasks;
    public final List<SemaphoreVK> signalSemaphores;

    public SubmitInfo(
            List<CommandBuffer> commandBuffers,
            List<SemaphoreVK> waitSemaphores,
            @EnumType(VkPipelineStageFlags.class) List<Integer> waitDstStageMasks,
            List<SemaphoreVK> signalSemaphores
    ) {
        assert waitSemaphores.size() == waitDstStageMasks.size() :
                "waitSemaphores 和 waitDstStageMasks 的大小必须相同";

        this.commandBuffers = commandBuffers;
        this.waitSemaphores = waitSemaphores;
        this.waitDstStageMasks = waitDstStageMasks;
        this.signalSemaphores = signalSemaphores;
    }
}
