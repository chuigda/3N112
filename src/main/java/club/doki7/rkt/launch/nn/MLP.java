package club.doki7.rkt.launch.nn;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.cmd.CommandBuffer;
import club.doki7.rkt.vk.cmd.CommandPool;
import club.doki7.rkt.vk.cmd.SubmitInfo;
import club.doki7.rkt.vk.pipeline.ComputePipeline;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.rkt.vk.sync.Fence;
import club.doki7.vulkan.bitmask.VkCommandBufferUsageFlags;
import club.doki7.vulkan.bitmask.VkCommandPoolCreateFlags;
import club.doki7.vulkan.datatype.VkBufferCopy;
import club.doki7.vulkan.datatype.VkCommandBufferBeginInfo;
import club.doki7.vulkan.enumtype.VkCommandBufferLevel;
import club.doki7.vulkan.enumtype.VkResult;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public final class MLP implements AutoCloseable {
    public final MLPFactory factory;
    public final MLPOptions options;

    public MLP(
            MLPFactory factory,
            MLPOptions options,
            RenderContext cx,
            List<Buffer> weightBufferList,
            List<Buffer> biasBufferList,
            List<ComputePipeline> computePipelineList
    ) {
        this.factory = factory;
        this.options = options;
        this.cx = cx;
        this.weightBufferList = weightBufferList;
        this.biasBufferList = biasBufferList;
        this.computePipelineList = computePipelineList;
    }

    public void uploadWeights(
            List<MemorySegment> weightList,
            List<MemorySegment> biasList
    ) throws RenderException {
        assert weightList.size() == options.layers.size();
        assert biasList.size() == options.layers.size();

        int queueFamilyIndex = cx.hasComputeQueue()
                ? cx.dedicatedComputeQueueFamilyIndex
                : cx.graphicsQueueFamilyIndex;
        Buffer.Options stagingOptions = Buffer.OptionsInit.stagingBufferPreset().build();

        List<Buffer> weightStagingBuffers = new ArrayList<>();
        List<Buffer> biasStagingBuffers = new ArrayList<>();
        try (Arena arena = Arena.ofConfined();
             CommandPool cmdPool = CommandPool.create(
                     cx,
                     VkCommandPoolCreateFlags.TRANSIENT,
                     queueFamilyIndex
             );
             Fence fence = Fence.createLocal(cx, 0x0)) {
            CommandBuffer cmdBuf = cmdPool.allocCmdBuf(cx, VkCommandBufferLevel.PRIMARY);
            cx.dCmd.beginCommandBuffer(cmdBuf.handle, VkCommandBufferBeginInfo.allocate(arena)
                    .flags(VkCommandBufferUsageFlags.ONE_TIME_SUBMIT));
            for (int i = 0; i < options.layers.size(); i++) {
                Buffer weightBuffer = weightBufferList.get(i);
                Buffer biasBuffer = biasBufferList.get(i);
                MemorySegment weightSegment = weightList.get(i);
                MemorySegment biasSegment = biasList.get(i);

                Buffer weightStagingBuffer = Buffer.create(cx, weightBuffer.size, true, stagingOptions);
                weightStagingBuffers.add(weightStagingBuffer);
                Buffer biasStagingBuffer = Buffer.create(cx, biasBuffer.size, true, stagingOptions);
                biasStagingBuffers.add(biasStagingBuffer);

                weightStagingBuffer.mapped.copyFrom(weightSegment);
                biasStagingBuffer.mapped.copyFrom(biasSegment);

                cx.dCmd.cmdCopyBuffer(
                        cmdBuf.handle,
                        weightStagingBuffer.handle,
                        weightBuffer.handle,
                        1,
                        VkBufferCopy.allocate(arena).size(weightBuffer.size)
                );
                cx.dCmd.cmdCopyBuffer(
                        cmdBuf.handle,
                        biasStagingBuffer.handle,
                        biasBuffer.handle,
                        1,
                        VkBufferCopy.allocate(arena).size(biasBuffer.size)
                );
            }
            @EnumType(VkResult.class) int result = cx.dCmd.endCommandBuffer(cmdBuf.handle);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法录制上传权重与偏置所需的命令缓冲");
            }

            SubmitInfo submitInfo = new SubmitInfo(
                    List.of(cmdBuf),
                    List.of(),
                    List.of(),
                    List.of()
            );
            if (cx.hasComputeQueue()) {
                cx.submitCompute(submitInfo, fence);
            } else {
                cx.submitGraphics(submitInfo, fence);
            }
            cx.waitForFence(fence);
        } finally {
            for (Buffer buffer : weightStagingBuffers) {
                buffer.close();
            }
            for (Buffer buffer : biasStagingBuffers) {
                buffer.close();
            }
        }
    }

    @Override
    public void close() {
        for (ComputePipeline pipeline : computePipelineList) {
            pipeline.close();
        }
        for (Buffer buffer : weightBufferList) {
            buffer.close();
        }
        for (Buffer buffer : biasBufferList) {
            buffer.close();
        }
    }

    final RenderContext cx;
    final List<Buffer> weightBufferList;
    final List<Buffer> biasBufferList;
    final List<ComputePipeline> computePipelineList;
}
