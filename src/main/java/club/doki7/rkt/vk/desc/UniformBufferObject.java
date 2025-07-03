package club.doki7.rkt.vk.desc;

import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.vulkan.datatype.VkDescriptorBufferInfo;
import club.doki7.vulkan.datatype.VkWriteDescriptorSet;
import club.doki7.vulkan.enumtype.VkDescriptorType;

public final class UniformBufferObject implements IDescriptor {
    public final Buffer buffer;
    public final VkDescriptorBufferInfo bufferInfo;

    public static UniformBufferObject create(RenderContext cx, Buffer buffer) {
        if (!buffer.options.usage.contains(Buffer.Usage.UNIFORM_BUFFER)) {
            throw new IllegalArgumentException("Buffer must be created with usage UNIFORM_BUFFER");
        }
        VkDescriptorBufferInfo bufferInfo = VkDescriptorBufferInfo.allocate(cx.prefabArena)
                .buffer(buffer.handle)
                .offset(0)
                .range(buffer.size);
        return new UniformBufferObject(buffer, bufferInfo);
    }

    private UniformBufferObject(Buffer buffer, VkDescriptorBufferInfo bufferInfo) {
        this.buffer = buffer;
        this.bufferInfo = bufferInfo;
    }

    @Override
    public DescriptorKind kind() {
        return DescriptorKind.UNIFORM_BUFFER;
    }

    @Override
    public void updateWriteDescriptorSet(VkWriteDescriptorSet writeDescriptorSet) {
        writeDescriptorSet
                .descriptorType(VkDescriptorType.UNIFORM_BUFFER)
                .pBufferInfo(bufferInfo);
    }
}
