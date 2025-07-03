package club.doki7.rkt.vk.desc;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.vulkan.enumtype.VkDescriptorType;

public enum DescriptorKind {
    COMBINED_IMAGE_SAMPLER(VkDescriptorType.COMBINED_IMAGE_SAMPLER),
    STORAGE_IMAGE(VkDescriptorType.STORAGE_IMAGE),
    UNIFORM_BUFFER(VkDescriptorType.UNIFORM_BUFFER),
    STORAGE_BUFFER(VkDescriptorType.STORAGE_BUFFER);

    public final @EnumType(VkDescriptorType.class) int value;

    DescriptorKind(@EnumType(VkDescriptorType.class) int value) {
        this.value = value;
    }
}
