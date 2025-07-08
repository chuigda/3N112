package club.doki7.rkt.launch.nn;

import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.FloatPtr;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.util.Assertion;
import club.doki7.rkt.vk.cmd.CommandBuffer;
import club.doki7.rkt.vk.cmd.CommandPool;
import club.doki7.rkt.vk.cmd.SubmitInfo;
import club.doki7.rkt.vk.desc.PushDescriptorSet;
import club.doki7.rkt.vk.desc.ShaderStorageBufferObject;
import club.doki7.rkt.vk.desc.UniformBufferObject;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.rkt.vk.sync.Fence;
import club.doki7.vulkan.bitmask.VkCommandBufferUsageFlags;
import club.doki7.vulkan.bitmask.VkCommandPoolCreateFlags;
import club.doki7.vulkan.bitmask.VkShaderStageFlags;
import club.doki7.vulkan.datatype.VkCommandBufferBeginInfo;
import club.doki7.vulkan.enumtype.VkCommandBufferLevel;
import club.doki7.vulkan.enumtype.VkPipelineBindPoint;
import club.doki7.vulkan.enumtype.VkResult;

import java.lang.foreign.Arena;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class MLPTrainTask extends MLPTaskBase implements AutoCloseable {
    public MLPTrainTask(
            MLP mlp,
            int batchSize,
            Buffer inputBuffer,
            Buffer labelBuffer,
            LossFunction lossFunction
    ) throws VulkanException {
        super(mlp, batchSize, inputBuffer, false, false);
        this.labelBuffer = labelBuffer;
        this.lossFunction = lossFunction;

        Buffer.OptionsInit optionsInit = new Buffer.OptionsInit();
        optionsInit.usage = Set.of(Buffer.Usage.STORAGE_BUFFER);
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
                rand.write(0.0f);
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

    @Override
    public void close() {
        for (Buffer gradientBuffer : gradientBufferList) {
            gradientBuffer.close();
        }
    }

    private void preRecordCommandBuffer() throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            cx.dCmd.beginCommandBuffer(cmdBuf.handle, VkCommandBufferBeginInfo.allocate(arena));
            preRecordForwardCommandBuffer();

            // TODO

            @EnumType(VkResult.class) int result = cx.dCmd.endCommandBuffer(cmdBuf.handle);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法录制 MLP 推理任务所用的命令缓冲");
            }
        }
    }

    private final Buffer labelBuffer;
    private final LossFunction lossFunction;

    private final Buffer updateOptionsBuffer;

    private final List<Buffer> gradientBufferList;
    private final List<PushDescriptorSet> weightsUpdateDescriptorSetList;
    private final List<PushDescriptorSet> backpropDescriptorSetList;
    private final PushDescriptorSet errorDescriptorSet;

    static final StructLayout UPDATE_OPTIONS_LAYOUT = NativeLayout.structLayout(
            ValueLayout.JAVA_INT.withName("input_offset"),
            ValueLayout.JAVA_INT.withName("batch_size")
    );
}
