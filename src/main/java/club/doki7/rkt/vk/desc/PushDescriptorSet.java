package club.doki7.rkt.vk.desc;

import club.doki7.rkt.vk.RenderContext;
import club.doki7.vulkan.datatype.VkWriteDescriptorSet;

import java.util.Collections;
import java.util.List;

public final class PushDescriptorSet {
    public final VkWriteDescriptorSet.Ptr descriptorSetWrites;
    public final DescriptorSetLayout compatibleLayout;
    public final List<IDescriptor> descriptors;

    public static PushDescriptorSet create(
            RenderContext cx,
            DescriptorSetLayout compatibleLayout,
            List<IDescriptor> descriptors
    ) {
        assert checkLayoutCompatibility(descriptors, compatibleLayout);

        VkWriteDescriptorSet.Ptr descriptorSetWrites = VkWriteDescriptorSet.allocate(
                cx.prefabArena,
                descriptors.size()
        );
        for (int i = 0; i < descriptors.size(); i++) {
            IDescriptor descriptor = descriptors.get(i);
            VkWriteDescriptorSet descriptorSetWrite = descriptorSetWrites.at(i);
            descriptorSetWrite
                    .dstBinding(i)
                    .descriptorCount(1);
            descriptor.updateWriteDescriptorSet(descriptorSetWrite);
        }

        return new PushDescriptorSet(
                descriptorSetWrites,
                compatibleLayout,
                Collections.unmodifiableList(descriptors)
        );
    }

    private PushDescriptorSet(
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
