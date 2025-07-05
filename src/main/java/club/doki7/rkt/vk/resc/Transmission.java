package club.doki7.rkt.vk.resc;

import club.doki7.ffm.annotation.Bitmask;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.cmd.CommandBuffer;
import club.doki7.rkt.vk.cmd.CommandPool;
import club.doki7.rkt.vk.cmd.SubmitInfo;
import club.doki7.rkt.vk.common.QueueFamily;
import club.doki7.rkt.vk.sync.Fence;
import club.doki7.rkt.vk.sync.SemaphoreVK;
import club.doki7.vulkan.bitmask.VkAccessFlags;
import club.doki7.vulkan.bitmask.VkCommandBufferUsageFlags;
import club.doki7.vulkan.bitmask.VkCommandPoolCreateFlags;
import club.doki7.vulkan.bitmask.VkPipelineStageFlags;
import club.doki7.vulkan.datatype.VkBufferCopy;
import club.doki7.vulkan.datatype.VkBufferMemoryBarrier;
import club.doki7.vulkan.datatype.VkCommandBufferBeginInfo;
import club.doki7.vulkan.enumtype.VkCommandBufferLevel;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

public final class Transmission {
    public static void uploadBuffer(
            RenderContext cx,
            Buffer buffer,
            MemorySegment rawData,
            QueueFamily bufferAffinity
    ) throws VulkanException {
        if (buffer.options.mapped) {
            buffer.mapped.copyFrom(rawData);
            if (!buffer.options.coherent) {
                buffer.flush(cx);
            }
            return;
        }

        Buffer.Options stagingOptions = Buffer.OptionsInit.stagingBufferPreset().build();

        try (Buffer stagingBuffer = Buffer.create(cx, buffer.size, true, stagingOptions);
             Fence fence = Fence.createLocal(cx)) {
            stagingBuffer.mapped.copyFrom(rawData);

            if (cx.hasTransferQueue()) {
                if (buffer.options.shared || bufferAffinity == QueueFamily.TRANSFER) {
                    uploadBufferWithSpecificQueue(cx, buffer, stagingBuffer, QueueFamily.TRANSFER, fence);
                } else {
                    uploadBufferWithTransferQueue(cx, buffer, stagingBuffer, bufferAffinity, fence);
                }
            } else {
                uploadBufferWithSpecificQueue(cx, buffer, stagingBuffer, bufferAffinity, fence);
            }
        }
    }

    private static void uploadBufferWithTransferQueue(
            RenderContext cx,
            Buffer buffer,
            Buffer stagingBuffer,
            QueueFamily bufferAffinity,
            Fence fence
    ) throws VulkanException {
        int queueIndex = cx.getQueueFamilyIndex(QueueFamily.TRANSFER);
        int affinityQueueIndex = cx.getQueueFamilyIndex(bufferAffinity);

        @Bitmask(VkCommandPoolCreateFlags.class) int poolFlags = VkCommandPoolCreateFlags.TRANSIENT;

        try (Arena arena = Arena.ofConfined();
             CommandPool transferCmdPool = CommandPool.createLocal(cx, poolFlags, queueIndex);
             CommandPool affinityCmdPool = CommandPool.createLocal(cx, poolFlags, affinityQueueIndex);
             SemaphoreVK semaphore = SemaphoreVK.createLocal(cx)) {
            CommandBuffer transferCmdBuf = transferCmdPool.allocCmdBuf(cx, VkCommandBufferLevel.PRIMARY);
            cx.dCmd.beginCommandBuffer(transferCmdBuf.handle, beginInfo);
            cx.dCmd.cmdCopyBuffer(
                    transferCmdBuf.handle,
                    stagingBuffer.handle,
                    buffer.handle,
                    1,
                    VkBufferCopy.allocate(arena).size(buffer.size)
            );
            VkBufferMemoryBarrier releaseBarrier = VkBufferMemoryBarrier.allocate(arena)
                    .srcAccessMask(VkAccessFlags.TRANSFER_WRITE)
                    .dstAccessMask(0)
                    .srcQueueFamilyIndex(queueIndex)
                    .dstQueueFamilyIndex(affinityQueueIndex)
                    .buffer(buffer.handle)
                    .size(buffer.size);
            cx.dCmd.cmdPipelineBarrier(
                    transferCmdBuf.handle,
                    VkPipelineStageFlags.TRANSFER,
                    VkPipelineStageFlags.ALL_COMMANDS,
                    0x0,
                    0, null,
                    1, releaseBarrier,
                    0, null
            );
            cx.dCmd.endCommandBuffer(transferCmdBuf.handle);
            cx.submit(
                    new SubmitInfo(List.of(transferCmdBuf), List.of(), List.of(), List.of(semaphore)),
                    null,
                    QueueFamily.TRANSFER
            );

            CommandBuffer affinityCmdBuf = affinityCmdPool.allocCmdBuf(cx, VkCommandBufferLevel.PRIMARY);
            cx.dCmd.beginCommandBuffer(affinityCmdBuf.handle, beginInfo);
            VkBufferMemoryBarrier acquireBarrier = VkBufferMemoryBarrier.allocate(arena)
                    .srcAccessMask(0)
                    .dstAccessMask(VkAccessFlags.MEMORY_READ | VkAccessFlags.MEMORY_WRITE)
                    .srcQueueFamilyIndex(queueIndex)
                    .dstQueueFamilyIndex(affinityQueueIndex)
                    .buffer(buffer.handle)
                    .size(buffer.size);
            cx.dCmd.cmdPipelineBarrier(
                    affinityCmdBuf.handle,
                    VkPipelineStageFlags.TRANSFER,
                    VkPipelineStageFlags.ALL_COMMANDS,
                    0x0,
                    0, null,
                    1, acquireBarrier,
                    0, null
            );
            cx.dCmd.endCommandBuffer(affinityCmdBuf.handle);
            cx.submit(
                    new SubmitInfo(
                            List.of(affinityCmdBuf),
                            List.of(semaphore),
                            List.of(VkPipelineStageFlags.TRANSFER),
                            List.of()
                    ),
                    fence,
                    bufferAffinity
            );

            cx.waitForFence(fence);
        }
    }

    private static void uploadBufferWithSpecificQueue(
            RenderContext cx,
            Buffer buffer,
            Buffer stagingBuffer,
            QueueFamily bufferAffinity,
            Fence fence
    ) throws VulkanException {
        int queueIndex = cx.getQueueFamilyIndex(bufferAffinity);
        try (Arena arena = Arena.ofConfined();
             CommandPool cmdPool = CommandPool.createLocal(cx, VkCommandPoolCreateFlags.TRANSIENT, queueIndex)) {
            CommandBuffer cmdBuf = cmdPool.allocCmdBuf(cx, VkCommandBufferLevel.PRIMARY);
            cx.dCmd.beginCommandBuffer(cmdBuf.handle, beginInfo);
            cx.dCmd.cmdCopyBuffer(
                    cmdBuf.handle,
                    stagingBuffer.handle,
                    buffer.handle,
                    1,
                    VkBufferCopy.allocate(arena).size(buffer.size)
            );
            cx.dCmd.endCommandBuffer(cmdBuf.handle);
            cx.submit(
                    new SubmitInfo(List.of(cmdBuf), List.of(), List.of(), List.of()),
                    fence,
                    bufferAffinity
            );
            cx.waitForFence(fence);
        }
    }

    private static final VkCommandBufferBeginInfo beginInfo =
            VkCommandBufferBeginInfo.allocate(Arena.global())
                    .flags(VkCommandBufferUsageFlags.ONE_TIME_SUBMIT);
}
