package club.doki7.rkt.vk.pipeline;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.desc.DescriptorSetLayout;
import club.doki7.rkt.vk.desc.PushConstantRange;
import club.doki7.vulkan.datatype.VkPipelineLayoutCreateInfo;
import club.doki7.vulkan.datatype.VkPushConstantRange;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.VkDescriptorSetLayout;
import club.doki7.vulkan.handle.VkPipelineLayout;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class PipelineLayout implements AutoCloseable {
    public final VkPipelineLayout handle;
    public final List<DescriptorSetLayout> descriptorSetLayouts;
    public final List<PushConstantRange> pushConstantRanges;
    public final List<Integer> pushConstantOffsets;
    public final int pushConstantSize;

    public static PipelineLayout create(
            RenderContext context,
            List<DescriptorSetLayout> descriptorSetLayout,
            List<PushConstantRange> pushConstantRanges
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            @Nullable VkDescriptorSetLayout.Ptr pSetLayouts;
            if (descriptorSetLayout.isEmpty()) {
                pSetLayouts = null;
            } else {
                pSetLayouts = VkDescriptorSetLayout.Ptr.allocate(arena, descriptorSetLayout.size());
                for (int i = 0; i < descriptorSetLayout.size(); i++) {
                    pSetLayouts.write(i, descriptorSetLayout.get(i).handle);
                }
            }

            @Nullable VkPushConstantRange.Ptr pPushConstantRanges;
            List<Integer> pushConstantOffsets;
            int pushConstantSize = 0;
            if (pushConstantRanges.isEmpty()) {
                pPushConstantRanges = null;
                pushConstantOffsets = List.of();
            } else {
                pPushConstantRanges = VkPushConstantRange.allocate(arena, pushConstantRanges.size());
                pushConstantOffsets = new ArrayList<>();
                for (int i = 0; i < pushConstantRanges.size(); i++) {
                    PushConstantRange range = pushConstantRanges.get(i);
                    VkPushConstantRange pRange = pPushConstantRanges.at(i);

                    pRange.stageFlags(range.shaderStageFlags())
                            .offset(pushConstantSize)
                            .size(range.size);
                    pushConstantOffsets.add(pushConstantSize);
                    pushConstantSize += range.size;
                }
            }

            VkPipelineLayoutCreateInfo createInfo = VkPipelineLayoutCreateInfo.allocate(arena)
                    .setLayoutCount(descriptorSetLayout.size())
                    .pSetLayouts(pSetLayouts)
                    .pushConstantRangeCount(pushConstantRanges.size())
                    .pPushConstantRanges(pPushConstantRanges);
            VkPipelineLayout.Ptr pLayout = VkPipelineLayout.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result = context.dCmd.createPipelineLayout(
                    context.device,
                    createInfo,
                    null,
                    pLayout
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建管线布局");
            }

            VkPipelineLayout handle = Objects.requireNonNull(pLayout.read());
            return new PipelineLayout(
                    handle,
                    Collections.unmodifiableList(descriptorSetLayout),
                    Collections.unmodifiableList(pushConstantRanges),
                    Collections.unmodifiableList(pushConstantOffsets),
                    pushConstantSize,
                    context
            );
        }
    }

    @Override
    public void close() throws Exception {
        cleanable.clean();
    }

    private PipelineLayout(
            VkPipelineLayout handle,
            List<DescriptorSetLayout> descriptorSetLayouts,
            List<PushConstantRange> pushConstantRanges,
            List<Integer> pushConstantOffsets,
            int pushConstantSize,
            RenderContext context
    ) {
        this.handle = handle;
        this.descriptorSetLayouts = descriptorSetLayouts;
        this.pushConstantRanges = pushConstantRanges;
        this.pushConstantOffsets = pushConstantOffsets;
        this.pushConstantSize = pushConstantSize;

        IDisposeOnContext d = cx -> cx.dCmd.destroyPipelineLayout(cx.device, handle, null);
        this.cleanable = context.registerCleanup(this, d, false);
    }

    private final Cleaner.Cleanable cleanable;
}
