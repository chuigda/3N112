package club.doki7.rkt.launch.nn;

import club.doki7.ffm.NativeLayout;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.cmd.CommandBuffer;
import club.doki7.rkt.vk.cmd.CommandPool;
import club.doki7.rkt.vk.desc.PushDescriptorSet;
import club.doki7.rkt.vk.desc.ShaderStorageBufferObject;
import club.doki7.rkt.vk.desc.UniformBufferObject;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.vulkan.VkConstants;
import club.doki7.vulkan.bitmask.VkAccessFlags;
import club.doki7.vulkan.bitmask.VkCommandPoolCreateFlags;
import club.doki7.vulkan.bitmask.VkPipelineStageFlags;
import club.doki7.vulkan.datatype.VkBufferMemoryBarrier;
import club.doki7.vulkan.enumtype.VkCommandBufferLevel;
import club.doki7.vulkan.enumtype.VkPipelineBindPoint;

import java.lang.foreign.Arena;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public abstract sealed class MLPTaskBase implements AutoCloseable
        permits MLPInferTask
{
    public final int batchSize;
    public final Buffer inputBuffer;
    public final List<Buffer> outputBufferList;

    public MLPTaskBase(
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
        this.ioInferOptionsBuffer = Buffer.create(
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
        this.forwardDescriptorSetList = new ArrayList<>();
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
                    ? ioInferOptionsBuffer
                    : inferOptionsBuffer;
            Buffer ehtotInputBuffer = i == 0
                    ? inputBuffer
                    : outputBufferList.get(i - 1);
            forwardDescriptorSetList.add(PushDescriptorSet.create(
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
    }

    @Override
    public void close() {
        cmdPool.close();
        inferOptionsBuffer.close();
        ioInferOptionsBuffer.close();
    }

    protected void preRecordForwardCommandBuffer() {
        try (Arena arena = Arena.ofConfined()) {
            for (int i = 0; i < forwardDescriptorSetList.size(); i++) {
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
                        forwardDescriptorSetList.get(i).descriptorSetWrites
                );

                MLPOptions.Layer layer = mlp.options.layers.get(i);
                cx.dCmd.cmdDispatch(
                        cmdBuf.handle,
                        Math.ceilDiv(layer.size, layer.perceptronWorkgroupSize),
                        batchSize,
                        1
                );

                if (i == forwardDescriptorSetList.size() - 1) {
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
        }
    }

    protected final RenderContext cx;
    protected final MLP mlp;

    protected final Buffer inferOptionsBuffer;
    protected final Buffer ioInferOptionsBuffer;

    protected final List<PushDescriptorSet> forwardDescriptorSetList;
    protected final CommandPool cmdPool;
    protected final CommandBuffer cmdBuf;

    static final StructLayout INFER_OPTIONS_LAYOUT = NativeLayout.structLayout(
            ValueLayout.JAVA_INT.withName("input_offset"),
            ValueLayout.JAVA_INT.withName("batch_size")
    );
}
