package club.doki7.rkt.launch.nn;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.cmd.SubmitInfo;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.rkt.vk.sync.Fence;
import club.doki7.vulkan.datatype.VkCommandBufferBeginInfo;
import club.doki7.vulkan.enumtype.VkResult;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.Objects;

public final class MLPInferTask extends MLPTaskBase implements AutoCloseable {
    public MLPInferTask(
            MLP mlp,
            int batchSize,
            Buffer inputBuffer,
            boolean mappedOutputBuffer,
            boolean mappedHiddenLayerOutputBuffer
    ) throws VulkanException {
        super(mlp, batchSize, inputBuffer, mappedOutputBuffer, mappedHiddenLayerOutputBuffer);

        preRecordCommandBuffer();
        this.submitInfo = new SubmitInfo(List.of(cmdBuf), List.of(), List.of(), List.of());
    }

    public void executeBatch(int batchStart) throws VulkanException {
        long totalCount = inputBuffer.size / ((long) mlp.options.inputSize * Float.BYTES);
        long ehtotBatchSize = Math.min(totalCount - batchStart, batchSize);
        if (ehtotBatchSize <= 0) {
            throw new IllegalArgumentException("批次起始超出输入数据范围");
        }

        IntPtr pInferOptionsBuffer = Objects.requireNonNull(IntPtr.checked(inferOptionsBuffer.mapped));
        IntPtr pIOInferOptionsBuffer = Objects.requireNonNull(IntPtr.checked(ioInferOptionsBuffer.mapped));
        pInferOptionsBuffer.write(0, 0);
        pInferOptionsBuffer.write(1, (int) ehtotBatchSize);
        pIOInferOptionsBuffer.write(0, batchStart);
        pIOInferOptionsBuffer.write(1, (int) ehtotBatchSize);

        try (Fence fence = Fence.createLocal(cx)) {
            if (cx.hasComputeQueue()) {
                cx.submitCompute(submitInfo, fence);
            } else {
                cx.submitGraphics(submitInfo, fence);
            }
            cx.waitForFence(fence);
        }
    }

    private void preRecordCommandBuffer() throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            cx.dCmd.beginCommandBuffer(cmdBuf.handle, VkCommandBufferBeginInfo.allocate(arena));
            preRecordForwardCommandBuffer();
            @EnumType(VkResult.class) int result = cx.dCmd.endCommandBuffer(cmdBuf.handle);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法录制 MLP 推理任务所用的命令缓冲");
            }
        }
    }

    private final SubmitInfo submitInfo;
}
