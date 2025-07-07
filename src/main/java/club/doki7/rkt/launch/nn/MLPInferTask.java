package club.doki7.rkt.launch.nn;

import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.cmd.CommandBuffer;
import club.doki7.rkt.vk.cmd.CommandPool;
import club.doki7.rkt.vk.cmd.SubmitInfo;
import club.doki7.rkt.vk.desc.PushDescriptorSet;
import club.doki7.rkt.vk.desc.UniformBufferObject;
import club.doki7.rkt.vk.desc.ShaderStorageBufferObject;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.rkt.vk.sync.Fence;
import club.doki7.vulkan.VkConstants;
import club.doki7.vulkan.bitmask.VkAccessFlags;
import club.doki7.vulkan.bitmask.VkCommandPoolCreateFlags;
import club.doki7.vulkan.bitmask.VkPipelineStageFlags;
import club.doki7.vulkan.datatype.VkBufferMemoryBarrier;
import club.doki7.vulkan.datatype.VkCommandBufferBeginInfo;
import club.doki7.vulkan.datatype.VkDescriptorBufferInfo;
import club.doki7.vulkan.enumtype.VkCommandBufferLevel;
import club.doki7.vulkan.enumtype.VkPipelineBindPoint;
import club.doki7.vulkan.enumtype.VkResult;

import java.lang.foreign.Arena;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;

public final class MLPInferTask implements AutoCloseable {
    public final int batchSize;
    public final Buffer inputBuffer;
    public final List<Buffer> outputBufferList;

    public MLPInferTask(
            MLP mlp,
            int batchSize,
            Buffer inputBuffer,
            boolean mappedOutputBuffer,
            boolean mappedHiddenLayerOutputBuffer
    ) throws VulkanException {
        this.cx = mlp.cx;
        this.mlp = mlp;
        this.batchSize = batchSize;
        this.inputBuffer = inputBuffer;

        Buffer.OptionsInit uniformOptionsInit = new Buffer.OptionsInit();
        uniformOptionsInit.usage = Set.of(Buffer.Usage.UNIFORM_BUFFER);
        uniformOptionsInit.mapped = true;
        uniformOptionsInit.coherent = true;
        Buffer.Options uniformOptions = uniformOptionsInit.build();

        this.inferOptionsBuffer = Buffer.create(
                cx,
                INFER_OPTIONS_LAYOUT.byteSize(),
                false,
                uniformOptions
        );
        this.layer0InferOptionsBuffer = Buffer.create(
                cx,
                INFER_OPTIONS_LAYOUT.byteSize(),
                false,
                uniformOptions
        );

        Buffer.OptionsInit outputOptionsInit = new Buffer.OptionsInit();
        if (mappedOutputBuffer) {
            outputOptionsInit.usage = Set.of(Buffer.Usage.STORAGE_BUFFER);
            outputOptionsInit.mapped = true;
            outputOptionsInit.coherent = true;
        } else {
            outputOptionsInit.usage = Set.of(Buffer.Usage.STORAGE_BUFFER, Buffer.Usage.TRANSFER_SRC);
        }
        Buffer.Options outputOptions = outputOptionsInit.build();

        Buffer.OptionsInit hiddenOutputOptionsInit = new Buffer.OptionsInit();
        if (mappedHiddenLayerOutputBuffer) {
            hiddenOutputOptionsInit.usage = Set.of(Buffer.Usage.STORAGE_BUFFER);
            hiddenOutputOptionsInit.mapped = true;
            hiddenOutputOptionsInit.coherent = true;
        } else {
            hiddenOutputOptionsInit.usage = Set.of(Buffer.Usage.STORAGE_BUFFER, Buffer.Usage.TRANSFER_SRC);
        }
        Buffer.Options hiddenOutputOptions = hiddenOutputOptionsInit.build();

        this.outputBufferList = new ArrayList<>();
        this.descriptorSets = new ArrayList<>();
        for (int i = 0; i < mlp.options.layers.size(); i++) {
             MLPOptions.Layer layer = mlp.options.layers.get(i);
             Buffer.Options useOptions = i == mlp.options.layers.size() - 1
                     ? outputOptions
                     : hiddenOutputOptions;
             Buffer outputBuffer = Buffer.create(
                     cx,
                     (long) layer.size * batchSize * Float.BYTES,
                     false,
                     useOptions
             );
             outputBufferList.add(outputBuffer);

             Buffer ehtotInferOptionsBuffer = i == 0
                     ? layer0InferOptionsBuffer
                     : inferOptionsBuffer;
             Buffer ehtotInputBuffer = i == 0
                     ? inputBuffer
                     : outputBufferList.get(i - 1);
             descriptorSets.add(PushDescriptorSet.create(
                     cx,
                     mlp.factory.mlpForwardSetLayout,
                     List.of(
                             UniformBufferObject.create(cx, ehtotInferOptionsBuffer),
                             ShaderStorageBufferObject.create(cx, ehtotInputBuffer),
                             ShaderStorageBufferObject.create(cx, mlp.weightBufferList.get(i)),
                             ShaderStorageBufferObject.create(cx, mlp.biasBufferList.get(i)),
                             ShaderStorageBufferObject.create(cx, outputBuffer)
                     )
             ));
        }

        int queueFamilyIndex = cx.hasComputeQueue()
                ? cx.dedicatedComputeQueueFamilyIndex
                : cx.graphicsQueueFamilyIndex;
        this.cmdPool = CommandPool.create(cx, VkCommandPoolCreateFlags.TRANSIENT, queueFamilyIndex);
        this.cmdBuf = cmdPool.allocCmdBuf(cx, VkCommandBufferLevel.PRIMARY);

        preRecordCommandBuffer();
        this.submitInfo = new SubmitInfo(List.of(cmdBuf), List.of(), List.of(), List.of());
    }

    public void executeBatch(int batchStart) throws VulkanException {
        long totalCount = inputBuffer.size / ((long) mlp.options.inputSize * Float.BYTES);
        long ehtotBatchSize = Math.min(totalCount - batchStart, batchSize);
        if (ehtotBatchSize <= 0) {
            throw new IllegalArgumentException("批次大小超出输入数据范围");
        }

        IntPtr pInferOptionsBuffer = Objects.requireNonNull(IntPtr.checked(inferOptionsBuffer.mapped));
        IntPtr pLayer0InferOptionsBuffer = Objects.requireNonNull(IntPtr.checked(layer0InferOptionsBuffer.mapped));
        pInferOptionsBuffer.write(0, 0);
        pInferOptionsBuffer.write(1, (int) ehtotBatchSize);
        pLayer0InferOptionsBuffer.write(0, batchStart);
        pLayer0InferOptionsBuffer.write(1, (int) ehtotBatchSize);

        try (Fence fence = Fence.createLocal(cx)) {
            if (cx.hasComputeQueue()) {
                cx.submitCompute(submitInfo, fence);
            } else {
                cx.submitGraphics(submitInfo, fence);
            }
            cx.waitForFence(fence);
        }
    }

    @Override
    public void close() {
        cmdPool.close();
        inferOptionsBuffer.close();
        layer0InferOptionsBuffer.close();
    }

    private void preRecordCommandBuffer() throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            cx.dCmd.beginCommandBuffer(cmdBuf.handle, VkCommandBufferBeginInfo.allocate(arena));
            for (int i = 0; i < descriptorSets.size(); i++) {
                cx.dCmd.cmdBindPipeline(
                        cmdBuf.handle,
                        VkPipelineBindPoint.COMPUTE,
                        mlp.computePipelineList.get(i).handle
                );
                cx.dCmd.cmdPushDescriptorSetKHR(
                        cmdBuf.handle,
                        VkPipelineBindPoint.COMPUTE,
                        mlp.factory.mlpForwardPipelineLayout.handle,
                        0,
                        5,
                        descriptorSets.get(i).descriptorSetWrites
                );

                MLPOptions.Layer layer = mlp.options.layers.get(i);
                cx.dCmd.cmdDispatch(
                        cmdBuf.handle,
                        Math.ceilDiv(layer.size, mlp.options.perceptronWorkgroupSize),
                        batchSize,
                        1
                );

                if (i == descriptorSets.size() - 1) {
                    continue;
                }

                Buffer outputBuffer = outputBufferList.get(i);
                VkBufferMemoryBarrier barrier = VkBufferMemoryBarrier.allocate(arena)
                        .srcAccessMask(VkAccessFlags.SHADER_WRITE)
                        .dstAccessMask(VkAccessFlags.SHADER_READ)
                        .srcQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                        .dstQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                        .buffer(outputBuffer.handle)
                        .offset(0)
                        .size(outputBuffer.size);
                cx.dCmd.cmdPipelineBarrier(
                        cmdBuf.handle,
                        VkPipelineStageFlags.COMPUTE_SHADER,
                        VkPipelineStageFlags.COMPUTE_SHADER,
                        0x0,
                        0, null,
                        1, barrier,
                        0, null
                );
            }

            @EnumType(VkResult.class) int result = cx.dCmd.endCommandBuffer(cmdBuf.handle);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法录制 MLP 推理任务所用的命令缓冲");
            }
        }
    }

    private final RenderContext cx;
    private final MLP mlp;

    private final Buffer inferOptionsBuffer;
    private final Buffer layer0InferOptionsBuffer;
    private final List<PushDescriptorSet> descriptorSets;
    private final CommandPool cmdPool;
    private final CommandBuffer cmdBuf;
    private final SubmitInfo submitInfo;

    private static final StructLayout INFER_OPTIONS_LAYOUT = NativeLayout.structLayout(
            ValueLayout.JAVA_INT.withName("input_offset"),
            ValueLayout.JAVA_INT.withName("batch_size")
    );
    private static final VkDescriptorBufferInfo INFER_OPTIONS_BUFFER_INFO = VkDescriptorBufferInfo.allocate(Arena.global())
            .offset(0)
            .range(INFER_OPTIONS_LAYOUT.byteSize());
}
