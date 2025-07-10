package club.doki7.rkt.launch.nn;

import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.FloatPtr;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.util.Assertion;
import club.doki7.rkt.vk.cmd.CommandBuffer;
import club.doki7.rkt.vk.cmd.CommandPool;
import club.doki7.rkt.vk.cmd.SubmitInfo;
import club.doki7.rkt.vk.desc.PushDescriptorSet;
import club.doki7.rkt.vk.desc.ShaderStorageBufferObject;
import club.doki7.rkt.vk.desc.UniformBufferObject;
import club.doki7.rkt.vk.pipeline.ComputePipeline;
import club.doki7.rkt.vk.pipeline.ShaderSpecialisation;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.rkt.vk.sync.Fence;
import club.doki7.vulkan.VkConstants;
import club.doki7.vulkan.bitmask.*;
import club.doki7.vulkan.datatype.VkBufferMemoryBarrier;
import club.doki7.vulkan.datatype.VkCommandBufferBeginInfo;
import club.doki7.vulkan.enumtype.VkCommandBufferLevel;
import club.doki7.vulkan.enumtype.VkPipelineBindPoint;
import club.doki7.vulkan.enumtype.VkResult;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;

public final class MLPTrainTask extends MLPTaskBase implements AutoCloseable {
    public final List<Buffer> gradientBufferList;

    public MLPTrainTask(
            MLP mlp,
            int batchSize,
            Buffer inputBuffer,
            Buffer labelBuffer,
            LossFunction lossFunction
    ) throws VulkanException {
        super(mlp, batchSize, inputBuffer, Assertion.assertionEnabled, Assertion.assertionEnabled);
        this.labelBuffer = labelBuffer;
        this.lossFunction = lossFunction;

        try (Arena arena = Arena.ofConfined()) {
            MLPOptions.Layer lastLayer = mlp.options.layers.getLast();
            if (lossFunction == LossFunction.CROSS_ENTROPY) {
                MemorySegment spec = arena.allocate(MLPFactory.ErrorCrossEntropyShaderSpec.LAYOUT);
                spec.set(ValueLayout.JAVA_INT, MLPFactory.ErrorCrossEntropyShaderSpec.OFFSET_tx, batchSize);
                spec.set(ValueLayout.JAVA_INT, MLPFactory.ErrorCrossEntropyShaderSpec.OFFSET_ty, 1);
                spec.set(ValueLayout.JAVA_INT, MLPFactory.ErrorCrossEntropyShaderSpec.OFFSET_perceptronCount, lastLayer.size);

                this.errorPipeline = ComputePipeline.create(
                        cx,
                        mlp.factory.mlpErrorPipelineLayout,
                        mlp.factory.mlpErrorCrossEntropyModule,
                        new ShaderSpecialisation(MLPFactory.ErrorCrossEntropyShaderSpec.SPEC_ENTRIES, spec)
                );
            } else {
                MemorySegment spec = arena.allocate(MLPFactory.ErrorMSEShaderSpec.LAYOUT);
                spec.set(ValueLayout.JAVA_INT, MLPFactory.ErrorMSEShaderSpec.OFFSET_tx, lastLayer.perceptronWorkgroupSize);
                spec.set(ValueLayout.JAVA_INT, MLPFactory.ErrorMSEShaderSpec.OFFSET_ty, batchSize);
                spec.set(ValueLayout.JAVA_INT, MLPFactory.ErrorMSEShaderSpec.OFFSET_perceptronCount, lastLayer.size);
                spec.set(ValueLayout.JAVA_INT, MLPFactory.ErrorMSEShaderSpec.OFFSET_activation, lastLayer.activ.value);

                this.errorPipeline = ComputePipeline.create(
                        cx,
                        mlp.factory.mlpErrorPipelineLayout,
                        mlp.factory.mlpErrorMSEModule,
                        new ShaderSpecialisation(MLPFactory.ErrorMSEShaderSpec.SPEC_ENTRIES, spec)
                );
            }
        }

        Buffer.OptionsInit optionsInit = new Buffer.OptionsInit();
        optionsInit.usage = Set.of(Buffer.Usage.STORAGE_BUFFER);
        if (Assertion.assertionEnabled) {
            optionsInit.mapped = true;
            optionsInit.coherent = true;
        }
        Buffer.Options storageOnlyOptions = optionsInit.build();

        optionsInit.usage = Set.of(Buffer.Usage.UNIFORM_BUFFER);
        optionsInit.mapped = true;
        optionsInit.coherent = true;
        Buffer.Options uniformOptions = optionsInit.build();

        this.updateOptionsBuffer = Buffer.create(
                cx,
                UPDATE_OPTIONS_LAYOUT.byteSize(),
                false,
                uniformOptions
        );

        this.gradientBufferList = new ArrayList<>();
        this.weightsUpdateDescriptorSetList = new ArrayList<>();
        for (int i = 0; i < mlp.options.layers.size(); i++) {
            MLPOptions.Layer layer = mlp.options.layers.get(i);

            Buffer gradientBuffer = Buffer.create(
                    cx,
                    (long) layer.size * batchSize * Float.BYTES,
                    false,
                    storageOnlyOptions
            );
            gradientBufferList.add(gradientBuffer);

            weightsUpdateDescriptorSetList.add(PushDescriptorSet.create(
                    cx,
                    mlp.factory.mlpUpdateWeightsSetLayout,
                    List.of(
                            UniformBufferObject.create(cx, updateOptionsBuffer),
                            UniformBufferObject.create(cx, i == 0
                                    ? ioInferOptionsBuffer
                                    : inferOptionsBuffer),
                            ShaderStorageBufferObject.create(cx, i == 0
                                    ? inputBuffer
                                    : outputBufferList.get(i - 1)),
                            ShaderStorageBufferObject.create(cx, gradientBuffer),
                            ShaderStorageBufferObject.create(cx, mlp.weightBufferList.get(i)),
                            ShaderStorageBufferObject.create(cx, mlp.biasBufferList.get(i))
                    )
            ));
        }

        this.backpropDescriptorSetList = new ArrayList<>();
        for (int i = 0; i < mlp.options.layers.size() - 1; i++) {
            backpropDescriptorSetList.add(PushDescriptorSet.create(
                    cx,
                    mlp.factory.mlpBackpropSetLayout,
                    List.of(
                            UniformBufferObject.create(cx, inferOptionsBuffer),
                            ShaderStorageBufferObject.create(cx, gradientBufferList.get(i + 1)),
                            ShaderStorageBufferObject.create(cx, mlp.weightBufferList.get(i + 1)),
                            ShaderStorageBufferObject.create(cx, outputBufferList.get(i)),
                            ShaderStorageBufferObject.create(cx, gradientBufferList.get(i))
                    )
            ));
        }

        this.errorDescriptorSet = PushDescriptorSet.create(cx, mlp.factory.mlpErrorSetLayout, List.of(
                UniformBufferObject.create(cx, ioInferOptionsBuffer),
                ShaderStorageBufferObject.create(cx, outputBufferList.getLast()),
                ShaderStorageBufferObject.create(cx, labelBuffer),
                ShaderStorageBufferObject.create(cx, gradientBufferList.getLast())
        ));

        preRecordCommandBuffer();

        this.submitInfo = new SubmitInfo(
                List.of(cmdBuf),
                List.of(),
                List.of(),
                List.of()
        );
    }

    public void prewarm() throws VulkanException {
        int queueFamilyIndex = cx.hasComputeQueue()
                ? cx.dedicatedComputeQueueFamilyIndex
                : cx.graphicsQueueFamilyIndex;

        try (CommandPool cmdPool = CommandPool.createLocal(
                cx,
                VkCommandPoolCreateFlags.TRANSIENT,
                queueFamilyIndex
             );
             Fence fence = Fence.create(cx);
             Arena arena = Arena.ofConfined()) {
            FloatPtr rand = FloatPtr.allocate(arena);
            if (Assertion.assertionEnabled) {
                rand.write(0.5f);
            } else {
                rand.write((float) Math.random());
            }

            CommandBuffer cmdBuf = cmdPool.allocCmdBuf(cx, VkCommandBufferLevel.PRIMARY);
            cx.dCmd.beginCommandBuffer(cmdBuf.handle, VkCommandBufferBeginInfo.allocate(arena)
                    .flags(VkCommandBufferUsageFlags.ONE_TIME_SUBMIT));

            int inputSize = mlp.options.inputSize;
            for (int i = 0; i < mlp.options.layers.size(); i++) {
                PushDescriptorSet descriptorSet = PushDescriptorSet.create(
                        cx,
                        mlp.factory.mlpWeightPrewarmSetLayout,
                        List.of(
                                ShaderStorageBufferObject.create(cx, mlp.weightBufferList.get(i)),
                                ShaderStorageBufferObject.create(cx, mlp.biasBufferList.get(i))
                        )
                );

                MLPOptions.Layer layer = mlp.options.layers.get(i);
                cx.dCmd.cmdBindPipeline(
                        cmdBuf.handle,
                        VkPipelineBindPoint.COMPUTE,
                        mlp.prewarmPipelineList.get(i).handle
                );
                cx.dCmd.cmdPushDescriptorSetKHR(
                        cmdBuf.handle,
                        VkPipelineBindPoint.COMPUTE,
                        mlp.factory.mlpWeightPrewarmPipelineLayout.handle,
                        0,
                        descriptorSet.descriptors.size(),
                        descriptorSet.descriptorSetWrites
                );
                cx.dCmd.cmdPushConstants(
                        cmdBuf.handle,
                        mlp.factory.mlpWeightPrewarmPipelineLayout.handle,
                        VkShaderStageFlags.COMPUTE,
                        0,
                        Float.BYTES,
                        rand.segment()
                );
                cx.dCmd.cmdDispatch(
                        cmdBuf.handle,
                        Math.ceilDiv(layer.size, layer.perceptronWorkgroupSize),
                        inputSize,
                        1
                );

                inputSize = layer.size;
            }

            @EnumType(VkResult.class) int result = cx.dCmd.endCommandBuffer(cmdBuf.handle);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法录制预热 MLP 权重的命令缓冲");
            }

            SubmitInfo submitInfo = new SubmitInfo(List.of(cmdBuf), List.of(), List.of(), List.of());
            if (cx.hasComputeQueue()) {
                cx.submitCompute(submitInfo, fence);
            } else {
                cx.submitGraphics(submitInfo, fence);
            }
            cx.waitForFence(fence);
        }
    }

    public void executeBatch(int batchStart, float learnRate) throws VulkanException {
        long totalCount = inputBuffer.size / ((long) mlp.options.inputSize * Float.BYTES);
        long ehtotBatchSize = Math.min(totalCount - batchStart, batchSize);
        if (ehtotBatchSize <= 0) {
            throw new IllegalArgumentException("批次起始超出输入数据范围");
        }

        IntPtr pInferOptionsBuffer = Objects.requireNonNull(IntPtr.checked(inferOptionsBuffer.mapped));
        IntPtr pIOInferOptionsBuffer = Objects.requireNonNull(IntPtr.checked(ioInferOptionsBuffer.mapped));
        IntPtr pUpdateOptionsBuffer = Objects.requireNonNull(IntPtr.checked(updateOptionsBuffer.mapped));
        pInferOptionsBuffer.write(0, 0);
        pInferOptionsBuffer.write(1, (int) ehtotBatchSize);
        pIOInferOptionsBuffer.write(0, batchStart);
        pIOInferOptionsBuffer.write(1, (int) ehtotBatchSize);
        pUpdateOptionsBuffer.segment().set(ValueLayout.JAVA_FLOAT, 0, learnRate);
        pUpdateOptionsBuffer.write(1, (int) ehtotBatchSize);

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
        for (Buffer gradientBuffer : gradientBufferList) {
            gradientBuffer.close();
        }
        updateOptionsBuffer.close();
        errorPipeline.close();
        super.close();
    }

    private void preRecordCommandBuffer() throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            cx.dCmd.beginCommandBuffer(cmdBuf.handle, VkCommandBufferBeginInfo.allocate(arena));
            preRecordForwardCommandBuffer();

            MLPOptions.Layer lastLayer = mlp.options.layers.getLast();

            // region calculate the gradient of the last layer
            cx.dCmd.cmdBindPipeline(
                    cmdBuf.handle,
                    VkPipelineBindPoint.COMPUTE,
                    errorPipeline.handle
            );
            cx.dCmd.cmdPushDescriptorSetKHR(
                    cmdBuf.handle,
                    VkPipelineBindPoint.COMPUTE,
                    mlp.factory.mlpErrorPipelineLayout.handle,
                    0,
                    errorDescriptorSet.descriptors.size(),
                    errorDescriptorSet.descriptorSetWrites
            );
            if (lossFunction == LossFunction.CROSS_ENTROPY) {
                cx.dCmd.cmdDispatch(cmdBuf.handle, 1, 1, 1);
            } else {
                cx.dCmd.cmdDispatch(
                        cmdBuf.handle,
                        Math.ceilDiv(lastLayer.size, lastLayer.perceptronWorkgroupSize),
                        1,
                        1
                );
            }
            // endregion

            // region calculate the gradient of the hidden layers
            for (int i = mlp.options.layers.size() - 2; i >= 0; i--) {
                MLPOptions.Layer layer = mlp.options.layers.get(i);

                // region step 1. make sure gradient write is already visible
                Buffer gradientBuffer = gradientBufferList.get(i + 1);
                VkBufferMemoryBarrier barrier1 = VkBufferMemoryBarrier.allocate(arena)
                        .srcAccessMask(VkAccessFlags.SHADER_WRITE)
                        .dstAccessMask(VkAccessFlags.SHADER_READ)
                        .srcQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                        .dstQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                        .buffer(gradientBuffer.handle)
                        .offset(0)
                        .size(gradientBuffer.size);
                cx.dCmd.cmdPipelineBarrier(
                        cmdBuf.handle,
                        VkPipelineStageFlags.COMPUTE_SHADER,
                        VkPipelineStageFlags.COMPUTE_SHADER,
                        0x0,
                        0, null,
                        1, barrier1,
                        0, null
                );
                // endregion

                // region step 2. backpropagate the error
                cx.dCmd.cmdBindPipeline(
                        cmdBuf.handle,
                        VkPipelineBindPoint.COMPUTE,
                        mlp.backpropPipelineList.get(i).handle
                );
                cx.dCmd.cmdPushDescriptorSetKHR(
                        cmdBuf.handle,
                        VkPipelineBindPoint.COMPUTE,
                        mlp.factory.mlpBackpropPipelineLayout.handle,
                        0,
                        backpropDescriptorSetList.get(i).descriptors.size(),
                        backpropDescriptorSetList.get(i).descriptorSetWrites
                );
                cx.dCmd.cmdDispatch(
                        cmdBuf.handle,
                        Math.ceilDiv(layer.size, layer.perceptronWorkgroupSize),
                        batchSize,
                        1
                );
                // endregion
            }
            // endregion

            // region make sure the gradient of the first layer is visible
            VkBufferMemoryBarrier barrier2 = VkBufferMemoryBarrier.allocate(arena)
                    .srcAccessMask(VkAccessFlags.SHADER_WRITE)
                    .dstAccessMask(VkAccessFlags.SHADER_READ)
                    .srcQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                    .dstQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                    .buffer(gradientBufferList.getFirst().handle)
                    .offset(0)
                    .size(gradientBufferList.getFirst().size);
            cx.dCmd.cmdPipelineBarrier(
                    cmdBuf.handle,
                    VkPipelineStageFlags.COMPUTE_SHADER,
                    VkPipelineStageFlags.COMPUTE_SHADER,
                    0x0,
                    0, null,
                    1, barrier2,
                    0, null
            );
            // endregion

            // region update the weights and biases
            int inputSize = mlp.options.inputSize;
            int inputPerceptronWorkgroupSize = mlp.options.layers.getFirst().perceptronWorkgroupSize;
            for (int i = 0; i < mlp.options.layers.size(); i++) {
                MLPOptions.Layer layer = mlp.options.layers.get(i);

                cx.dCmd.cmdBindPipeline(
                        cmdBuf.handle,
                        VkPipelineBindPoint.COMPUTE,
                        mlp.updatePipelineList.get(i).handle
                );
                cx.dCmd.cmdPushDescriptorSetKHR(
                        cmdBuf.handle,
                        VkPipelineBindPoint.COMPUTE,
                        mlp.factory.mlpUpdateWeightsPipelineLayout.handle,
                        0,
                        weightsUpdateDescriptorSetList.get(i).descriptors.size(),
                        weightsUpdateDescriptorSetList.get(i).descriptorSetWrites
                );
                cx.dCmd.cmdDispatch(
                        cmdBuf.handle,
                        Math.ceilDiv(inputSize, inputPerceptronWorkgroupSize),
                        Math.ceilDiv(layer.size, layer.perceptronWorkgroupSize),
                        1
                );

                inputSize = layer.size;
                inputPerceptronWorkgroupSize = layer.perceptronWorkgroupSize;
            }
            // endregion

            // region make sure the weight and biases updates are visible
            VkBufferMemoryBarrier.Ptr barriers = VkBufferMemoryBarrier.allocate(arena, mlp.options.layers.size() * 2L);
            for (int i = 0; i < mlp.options.layers.size(); i++) {
                Buffer weightBuffer = mlp.weightBufferList.get(i);
                Buffer biasBuffer = mlp.biasBufferList.get(i);

                barriers.at(i * 2L, it -> it
                        .srcAccessMask(VkAccessFlags.SHADER_WRITE)
                        .dstAccessMask(VkAccessFlags.SHADER_READ)
                        .srcQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                        .dstQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                        .buffer(weightBuffer.handle)
                        .offset(0)
                        .size(weightBuffer.size));
                barriers.at(i * 2L + 1, it -> it
                        .srcAccessMask(VkAccessFlags.SHADER_WRITE)
                        .dstAccessMask(VkAccessFlags.SHADER_READ)
                        .srcQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                        .dstQueueFamilyIndex(VkConstants.QUEUE_FAMILY_IGNORED)
                        .buffer(biasBuffer.handle)
                        .offset(0)
                        .size(biasBuffer.size));
            }

            cx.dCmd.cmdPipelineBarrier(
                    cmdBuf.handle,
                    VkPipelineStageFlags.COMPUTE_SHADER,
                    VkPipelineStageFlags.COMPUTE_SHADER,
                    0x0,
                    0, null,
                    mlp.options.layers.size() * 2, barriers,
                    0, null
            );
            // endregion

            @EnumType(VkResult.class) int result = cx.dCmd.endCommandBuffer(cmdBuf.handle);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法录制 MLP 推理任务所用的命令缓冲");
            }
        }
    }

    private final Buffer labelBuffer;
    private final LossFunction lossFunction;
    private final ComputePipeline errorPipeline;

    private final Buffer updateOptionsBuffer;

    private final List<PushDescriptorSet> weightsUpdateDescriptorSetList;
    private final List<PushDescriptorSet> backpropDescriptorSetList;
    private final PushDescriptorSet errorDescriptorSet;

    private final SubmitInfo submitInfo;

    static final StructLayout UPDATE_OPTIONS_LAYOUT = NativeLayout.structLayout(
            ValueLayout.JAVA_FLOAT.withName("learning_rate"),
            ValueLayout.JAVA_INT.withName("batch_size")
    );
}
