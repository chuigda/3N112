package club.doki7.rkt.vk.desc;

import club.doki7.rkt.vk.RenderContext;
import club.doki7.vulkan.datatype.VkWriteDescriptorSet;

import java.util.Collections;
import java.util.List;

public final class DescriptorSet {
    public final VkWriteDescriptorSet.Ptr descriptorSetWrites;
    public final DescriptorSetLayout compatibleLayout;
    public final List<IDescriptor> descriptors;

    public static DescriptorSet create(
            RenderContext context,
            DescriptorSetLayout compatibleLayout,
            List<IDescriptor> descriptors
    ) {
        assert checkLayoutCompatibility(descriptors, compatibleLayout);

        VkWriteDescriptorSet.Ptr descriptorSetWrites = VkWriteDescriptorSet.allocate(
                context.prefabArena,
                descriptors.size()
        );
        for (int i = 0; i < descriptors.size(); i++) {
            IDescriptor descriptor = descriptors.get(i);
            VkWriteDescriptorSet descriptorSetWrite = descriptorSetWrites.at(i);
            descriptorSetWrite
                    .dstBinding(i)
                    .dstArrayElement(0)
                    .descriptorCount(1);
            descriptor.updateWriteDescriptorSet(descriptorSetWrite);
        }

        return new DescriptorSet(
                descriptorSetWrites,
                compatibleLayout,
                Collections.unmodifiableList(descriptors)
        );
    }

    private DescriptorSet(
            VkWriteDescriptorSet.Ptr descriptorSetWrites,
            DescriptorSetLayout compatibleLayout,
            List<IDescriptor> descriptors
    ) {
        this.descriptorSetWrites = descriptorSetWrites;
        this.compatibleLayout = compatibleLayout;
        this.descriptors = descriptors;
    }

    private static boolean checkLayoutCompatibility(
            List<IDescriptor> descriptors,
            DescriptorSetLayout compatibleLayout
    ) {
        if (descriptors.size() != compatibleLayout.bindings.size()) {
            return false;
        }
        for (int i = 0; i < descriptors.size(); i++) {
            IDescriptor descriptor = descriptors.get(i);
            DescriptorSetLayoutBinding binding = compatibleLayout.bindings.get(i);

            if (descriptor.kind() != binding.kind) {
                return false;
            }
        }
        return true;
    }
}
