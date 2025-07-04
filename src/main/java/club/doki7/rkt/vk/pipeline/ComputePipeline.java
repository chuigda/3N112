package club.doki7.rkt.vk.pipeline;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.vulkan.bitmask.VkShaderStageFlags;
import club.doki7.vulkan.datatype.VkComputePipelineCreateInfo;
import club.doki7.vulkan.datatype.VkSpecializationInfo;
import club.doki7.vulkan.datatype.VkSpecializationMapEntry;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.VkPipeline;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;
import java.util.Objects;

public final class ComputePipeline implements AutoCloseable {
    public final VkPipeline handle;
    public final PipelineLayout layout;

    public static ComputePipeline create(
            RenderContext cx,
            PipelineLayout layout,
            ShaderModule computeShader,
            @Nullable ShaderSpecialisation specialisation
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkSpecializationInfo specInfo;
            if (specialisation == null) {
                specInfo = null;
            } else {
                VkSpecializationMapEntry.Ptr entries = VkSpecializationMapEntry.allocate(arena, specialisation.entries.size());
                for (int i = 0; i < specialisation.entries.size(); i++) {
                    ShaderSpecialisation.Entry entry = specialisation.entries.get(i);
                    entries.at(i)
                            .constantID(entry.constantId)
                            .offset(entry.offset)
                            .size(entry.size);
                }

                specInfo = VkSpecializationInfo.allocate(arena)
                        .mapEntryCount(specialisation.entries.size())
                        .pMapEntries(entries)
                        .pData(specialisation.data);
            }

            VkComputePipelineCreateInfo createInfo = VkComputePipelineCreateInfo.allocate(arena)
                    .stage(it -> it
                            .stage(VkShaderStageFlags.COMPUTE)
                            .module(computeShader.handle)
                            .pName(entryName)
                            .pSpecializationInfo(specInfo))
                    .layout(layout.handle);
            VkPipeline.Ptr pPipeline = VkPipeline.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result = cx.dCmd.createComputePipelines(
                    cx.device,
                    null,
                    1,
                    createInfo,
                    null,
                    pPipeline
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建计算管线");
            }

            VkPipeline handle = Objects.requireNonNull(pPipeline.read());
            return new ComputePipeline(handle, layout, cx);
        }
    }

    @Override
    public void close() throws Exception {
        cleanable.clean();
    }

    private ComputePipeline(
            VkPipeline handle,
            PipelineLayout layout,
            RenderContext context
    ) {
        this.handle = handle;
        this.layout = layout;

        IDisposeOnContext d = cx -> cx.dCmd.destroyPipeline(cx.device, handle, null);
        this.cleanable = context.registerCleanup(this, d, false);
    }

    private final Cleaner.Cleanable cleanable;
    private static final BytePtr entryName = BytePtr.allocateString(Arena.global(), "main");
}
