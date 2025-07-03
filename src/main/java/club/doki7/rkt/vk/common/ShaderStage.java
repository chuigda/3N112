package club.doki7.rkt.vk.common;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.vulkan.bitmask.VkShaderStageFlags;

public enum ShaderStage {
    VERTEX(VkShaderStageFlags.VERTEX),
    FRAGMENT(VkShaderStageFlags.FRAGMENT),
    COMPUTE(VkShaderStageFlags.COMPUTE),
    TESS_CONTROL(VkShaderStageFlags.TESSELLATION_CONTROL),
    TESS_EVAL(VkShaderStageFlags.TESSELLATION_EVALUATION),
    GEOMETRY(VkShaderStageFlags.GEOMETRY);

    public final @EnumType(VkShaderStageFlags.class) int value;

    ShaderStage(@EnumType(VkShaderStageFlags.class) int value) {
        this.value = value;
    }
}
