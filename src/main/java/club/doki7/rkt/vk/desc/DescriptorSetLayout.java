package club.doki7.rkt.vk.desc;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.vulkan.bitmask.VkDescriptorSetLayoutCreateFlags;
import club.doki7.vulkan.datatype.VkDescriptorSetLayoutBinding;
import club.doki7.vulkan.datatype.VkDescriptorSetLayoutCreateInfo;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.VkDescriptorSetLayout;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class DescriptorSetLayout implements AutoCloseable {
    public final VkDescriptorSetLayout handle;
    public final List<DescriptorSetLayoutBinding> bindings;

    public static DescriptorSetLayout create(
            RenderContext cx,
            List<DescriptorSetLayoutBinding> bindings,
            boolean pushDescriptor
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkDescriptorSetLayoutBinding.Ptr nBindings = VkDescriptorSetLayoutBinding.allocate(arena, bindings.size());
            for (int i = 0; i < bindings.size(); i++) {
                DescriptorSetLayoutBinding binding = bindings.get(i);
                nBindings.at(i)
                        .binding(i)
                        .descriptorType(binding.kind.value)
                        .descriptorCount(1)
                        .stageFlags(binding.shaderStageFlags());
            }
            VkDescriptorSetLayoutCreateInfo createInfo = VkDescriptorSetLayoutCreateInfo.allocate(arena)
                    .flags(pushDescriptor ? VkDescriptorSetLayoutCreateFlags.PUSH_DESCRIPTOR : 0)
                    .bindingCount(bindings.size())
                    .pBindings(nBindings);
            VkDescriptorSetLayout.Ptr pLayout = VkDescriptorSetLayout.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result = cx.dCmd.createDescriptorSetLayout(
                    cx.device,
                    createInfo,
                    null,
                    pLayout
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 描述符集布局");
            }

            VkDescriptorSetLayout handle = Objects.requireNonNull(pLayout.read());
            return new DescriptorSetLayout(handle, Collections.unmodifiableList(bindings), cx);
        }
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private DescriptorSetLayout(
            VkDescriptorSetLayout handle,
            List<DescriptorSetLayoutBinding> bindings,
            RenderContext context
    ) {
        this.handle = handle;
        this.bindings = bindings;

        IDisposeOnContext d = cx -> cx.dCmd.destroyDescriptorSetLayout(cx.device, handle, null);
        this.cleanable = context.registerCleanup(this, d, false);
    }

    private final Cleaner.Cleanable cleanable;
}
