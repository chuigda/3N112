package club.doki7.rkt.vk.desc;

import club.doki7.ffm.annotation.Bitmask;
import club.doki7.rkt.vk.common.ShaderStage;
import club.doki7.vulkan.bitmask.VkShaderStageFlags;

import java.util.Collections;
import java.util.Set;

public final class PushConstantRange {
    public final int size;
    public final Set<ShaderStage> stages;

    public PushConstantRange(int size, Set<ShaderStage> stages) {
        this.size = size;
        this.stages = Collections.unmodifiableSet(stages);
    }

    public PushConstantRange(int size, ShaderStage... stages) {
        this(size, Set.of(stages));
    }

    public @Bitmask(VkShaderStageFlags.class) int shaderStageFlags() {
        @Bitmask(VkShaderStageFlags.class) int flags = 0;
        for (ShaderStage stage : stages) {
            flags |= stage.value;
        }
        return flags;
    }
}
