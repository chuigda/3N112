package club.doki7.rkt.vk.desc;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.rkt.vk.common.ShaderStage;
import club.doki7.vulkan.bitmask.VkShaderStageFlags;

import java.util.Collections;
import java.util.Set;

public final class DescriptorSetLayoutBinding {
    public final DescriptorKind kind;
    public final Set<ShaderStage> stages;

    public DescriptorSetLayoutBinding(DescriptorKind kind, Set<ShaderStage> stages) {
        this.kind = kind;
        this.stages = Collections.unmodifiableSet(stages);
    }

    public DescriptorSetLayoutBinding(DescriptorKind kind, ShaderStage... stages) {
        this.kind = kind;
        this.stages = Set.of(stages);
    }

    public @EnumType(VkShaderStageFlags.class) int shaderStageFlags() {
        @EnumType(VkShaderStageFlags.class) int flags = 0;
        for (ShaderStage stage : stages) {
            flags |= stage.value;
        }
        return flags;
    }
}
