package club.doki7.rkt.vk.desc;

import club.doki7.vulkan.datatype.VkWriteDescriptorSet;

public sealed interface IDescriptor permits
        UniformBufferObject,
        ShaderStorageBufferObject
{
    DescriptorKind kind();

    void updateWriteDescriptorSet(VkWriteDescriptorSet writeDescriptorSet);
}
