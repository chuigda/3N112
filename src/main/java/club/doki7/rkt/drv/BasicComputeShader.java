package club.doki7.rkt.drv;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.shaderc.ShaderCompiler;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.cmd.CommandBuffer;
import club.doki7.rkt.vk.cmd.CommandPool;
import club.doki7.rkt.vk.cmd.SubmitInfo;
import club.doki7.rkt.vk.common.ShaderStage;
import club.doki7.rkt.vk.desc.*;
import club.doki7.rkt.vk.pipeline.ComputePipeline;
import club.doki7.rkt.vk.pipeline.PipelineLayout;
import club.doki7.rkt.vk.pipeline.ShaderModule;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.rkt.vk.sync.Fence;
import club.doki7.shaderc.Shaderc;
import club.doki7.shaderc.ShadercUtil;
import club.doki7.shaderc.enumtype.ShadercIncludeType;
import club.doki7.shaderc.enumtype.ShadercShaderKind;
import club.doki7.vulkan.bitmask.VkAccessFlags;
import club.doki7.vulkan.bitmask.VkPipelineStageFlags;
import club.doki7.vulkan.bitmask.VkShaderStageFlags;
import club.doki7.vulkan.command.VulkanLoader;
import club.doki7.vulkan.datatype.VkBufferMemoryBarrier;
import club.doki7.vulkan.datatype.VkCommandBufferBeginInfo;
import club.doki7.vulkan.enumtype.VkCommandBufferLevel;
import club.doki7.vulkan.enumtype.VkPipelineBindPoint;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;

public final class BasicComputeShader {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
    }

    public static void main() {
        try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
             ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma");
             ISharedLibrary libShaderc = ILibraryLoader.platformLoader().loadLibrary("shaderc_shared");
             Application app = new Application(libVulkan, libVMA, libShaderc)) {
            app.applicationStart();
        } catch (Throwable e) {
            e.printStackTrace(System.err);
        }
    }
}

final class Application implements AutoCloseable {
    @Override
    public void close() {
        arena.close();
        cx.close();
        compiler.close();
    }

    Application(ISharedLibrary libVulkan, ISharedLibrary libVMA, ISharedLibrary libShaderc) throws RenderException {
        this.cx = RenderContext.createHeadless(libVulkan, libVMA, new RenderConfig());
        this.compiler = ShaderCompiler.create(new Shaderc(libShaderc), Application::unsupportedResolve);
    }

    void applicationStart() throws RenderException {
        // region 1. shader module creation
        String shaderSource;
        try {
            shaderSource = Files.readString(Path.of("resc/shader/forward.comp.glsl"));
        } catch (IOException e) {
            throw new RuntimeException("无法打开 shader 文件: forward.comp.glsl", e);
        }

        String spvAssembly = compiler.compileIntoAssembly(
                "forward.comp.glsl",
                shaderSource,
                ShadercShaderKind.COMPUTE_SHADER
        );
        logger.fine("Shader 编译成功，SPIR-V 汇编:\n" + spvAssembly);

        BytePtr spv = compiler.compileIntoSPV(
                arena,
                "forward.comp.glsl",
                shaderSource,
                ShadercShaderKind.COMPUTE_SHADER
        );
        logger.info("Shader 编译成功，已生成 SPIR-V 二进制数据，长度: " + spv.size());
        ShaderModule shaderModule = ShaderModule.create(cx, spv);
        logger.info("创建着色器模块成功，句柄: " + shaderModule.handle);
        // endregion

        // region 2. descriptor set layout creation
        DescriptorSetLayout layout = DescriptorSetLayout.create(cx, List.of(
                // layout(set = 0, binding = 0) uniform buffer InputBuffer { ... };
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE),
                // layout(set = 0, binding = 1) uniform buffer WeightsBuffer { ... };
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE),
                // layout(set = 0, binding = 2) uniform buffer BiasBuffer { ... };
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE),
                // layout(set = 0, binding = 3) uniform buffer OutputBuffer { ... };
                new DescriptorSetLayoutBinding(DescriptorKind.STORAGE_BUFFER, ShaderStage.COMPUTE)
        ));
        List<PushConstantRange> pushConstantRanges = List.of(
                new PushConstantRange(Integer.BYTES, ShaderStage.COMPUTE)
        );
        // endregion

        // region 3. pipeline creation
        PipelineLayout pipelineLayout =
                PipelineLayout.create(cx, List.of(layout), pushConstantRanges);
        ComputePipeline computePipeline =
                ComputePipeline.create(cx, pipelineLayout, shaderModule, null);
        // endregion

        // region 4. buffer creation
        Buffer.Options optionsMappedSSBO = Buffer.Options.init(it -> {
            it.mapped = true;
            it.coherent = true;
            it.usage = Set.of(Buffer.Usage.STORAGE_BUFFER);
        });

        Buffer inputBuffer =
                Buffer.create(cx, inputSize * Float.BYTES, true, optionsMappedSSBO);
        Buffer hiddenLayerOutputBuffer =
                Buffer.create(cx, hiddenLayerSize * Float.BYTES, true, optionsMappedSSBO);
        Buffer outputBuffer =
                Buffer.create(cx, outputLayerSize * Float.BYTES, true, optionsMappedSSBO);

        Buffer hiddenLayerWeightsBuffer = Buffer.create(
                cx,
                perceptronWeights[0].length * Float.BYTES,
                true,
                optionsMappedSSBO
        );
        Buffer hiddenLayerBiasBuffer = Buffer.create(
                cx,
                perceptronBias[0].length * Float.BYTES,
                true,
                optionsMappedSSBO
        );
        Buffer outputLayerWeightsBuffer = Buffer.create(
                cx,
                perceptronWeights[1].length * Float.BYTES,
                true,
                optionsMappedSSBO
        );
        Buffer outputLayerBiasBuffer = Buffer.create(
                cx,
                perceptronBias[1].length * Float.BYTES,
                true,
                optionsMappedSSBO
        );

        Objects.requireNonNull(hiddenLayerWeightsBuffer.mapped)
                .segment()
                .copyFrom(MemorySegment.ofArray(perceptronWeights[0]));
        Objects.requireNonNull(hiddenLayerBiasBuffer.mapped)
                .segment()
                .copyFrom(MemorySegment.ofArray(perceptronBias[0]));
        Objects.requireNonNull(outputLayerWeightsBuffer.mapped)
                .segment()
                .copyFrom(MemorySegment.ofArray(perceptronWeights[1]));
        Objects.requireNonNull(outputLayerBiasBuffer.mapped)
                .segment()
                .copyFrom(MemorySegment.ofArray(perceptronBias[1]));
        // endregion

        // region 5. descriptor set creation
        ShaderStorageBufferObject inputSSBO = ShaderStorageBufferObject.create(cx, inputBuffer);
        ShaderStorageBufferObject hiddenLayer1OutputSSBO =
                ShaderStorageBufferObject.create(cx, hiddenLayerOutputBuffer);
        ShaderStorageBufferObject outputSSBO =
                ShaderStorageBufferObject.create(cx, outputBuffer);

        ShaderStorageBufferObject hiddenLayerWeightsSSBO =
                ShaderStorageBufferObject.create(cx, hiddenLayerWeightsBuffer);
        ShaderStorageBufferObject hiddenLayerBiasSSBO =
                ShaderStorageBufferObject.create(cx, hiddenLayerBiasBuffer);
        ShaderStorageBufferObject outputLayerWeightsSSBO =
                ShaderStorageBufferObject.create(cx, outputLayerWeightsBuffer);
        ShaderStorageBufferObject outputLayerBiasSSBO =
                ShaderStorageBufferObject.create(cx, outputLayerBiasBuffer);

        DescriptorSet layer1Set = DescriptorSet.create(cx, layout, List.of(
                inputSSBO,
                hiddenLayerWeightsSSBO,
                hiddenLayerBiasSSBO,
                hiddenLayer1OutputSSBO
        ));
        DescriptorSet layer2Set = DescriptorSet.create(cx, layout, List.of(
                hiddenLayer1OutputSSBO,
                outputLayerWeightsSSBO,
                outputLayerBiasSSBO,
                outputSSBO
        ));

        IntPtr pushConstant1 = IntPtr.allocateV(arena, inputSize);
        IntPtr pushConstant2 = IntPtr.allocateV(arena, hiddenLayerSize);
        // endregion

        // region 6. command buffer recording
        CommandPool commandPool;
        int queueIndex;
        if (cx.hasComputeQueue()) {
            queueIndex = cx.dedicatedComputeQueueFamilyIndex;
        } else {
            queueIndex = cx.graphicsQueueFamilyIndex;
        }
        commandPool = CommandPool.createLocal(cx, 0, queueIndex);

        CommandBuffer commandBuffer = commandPool.allocCmdBuf(cx, VkCommandBufferLevel.PRIMARY);
        cx.dCmd.beginCommandBuffer(
                commandBuffer.handle,
                VkCommandBufferBeginInfo.allocate(arena)
        );
        cx.dCmd.cmdBindPipeline(
                commandBuffer.handle,
                VkPipelineBindPoint.COMPUTE,
                computePipeline.handle
        );

        // run hidden layer
        cx.dCmd.cmdPushDescriptorSetKHR(
                commandBuffer.handle,
                VkPipelineBindPoint.COMPUTE,
                pipelineLayout.handle,
                0,
                layer1Set.descriptors.size(),
                layer1Set.descriptorSetWrites
        );
        cx.dCmd.cmdPushConstants(
                commandBuffer.handle,
                pipelineLayout.handle,
                VkShaderStageFlags.COMPUTE,
                0,
                Integer.BYTES,
                pushConstant1.segment()
        );
        cx.dCmd.cmdDispatch(commandBuffer.handle, 2, 1, 1);

        // wait for writes into hiddenLayerOutputBuffer to be visible
        VkBufferMemoryBarrier barrier = VkBufferMemoryBarrier.allocate(arena)
                .srcAccessMask(VkAccessFlags.SHADER_WRITE)
                .dstAccessMask(VkAccessFlags.SHADER_READ)
                .srcQueueFamilyIndex(queueIndex)
                .dstQueueFamilyIndex(queueIndex)
                .buffer(hiddenLayerOutputBuffer.handle)
                .offset(0)
                .size(hiddenLayerOutputBuffer.size);
        cx.dCmd.cmdPipelineBarrier(
                commandBuffer.handle,
                VkPipelineStageFlags.COMPUTE_SHADER,
                VkPipelineStageFlags.COMPUTE_SHADER,
                0x0,
                0, null,
                1, barrier,
                0, null
        );

        // run output layer
        cx.dCmd.cmdPushDescriptorSetKHR(
                commandBuffer.handle,
                VkPipelineBindPoint.COMPUTE,
                pipelineLayout.handle,
                0,
                layer2Set.descriptors.size(),
                layer2Set.descriptorSetWrites
        );
        cx.dCmd.cmdPushConstants(
                commandBuffer.handle,
                pipelineLayout.handle,
                VkShaderStageFlags.COMPUTE,
                0,
                Integer.BYTES,
                pushConstant2.segment()
        );
        cx.dCmd.cmdDispatch(commandBuffer.handle, 1, 1, 1);

        cx.dCmd.endCommandBuffer(commandBuffer.handle);
        // endregion

        // region 7. command buffer submission and testing
        Fence fence = Fence.createLocal(cx);
        SubmitInfo submitInfo = new SubmitInfo(List.of(commandBuffer), List.of(), List.of(), List.of());

        for (int i = 0; i < 4; i++) {
            // copy input values to input buffer
            Objects.requireNonNull(inputBuffer.mapped)
                    .segment()
                    .copyFrom(MemorySegment.ofArray(inputValues[i]));

            // submit command buffer
            if (cx.hasComputeQueue()) {
                cx.submitCompute(submitInfo, fence);
            } else {
                cx.submitGraphics(submitInfo, fence);
            }

            // wait for the command buffer to finish execution
            cx.waitForFence(fence);
            cx.resetFence(fence);

            float hidden1 = Objects.requireNonNull(hiddenLayerOutputBuffer.mapped)
                    .segment()
                    .get(ValueLayout.JAVA_FLOAT, 0);
            float hidden2 = Objects.requireNonNull(hiddenLayerOutputBuffer.mapped)
                    .segment()
                    .get(ValueLayout.JAVA_FLOAT, Float.BYTES);

            // read output from output buffer
            float output = Objects.requireNonNull(outputBuffer.mapped)
                    .segment()
                    .get(ValueLayout.JAVA_FLOAT, 0);
            boolean boolResult = output >= 0.5f;

            logger.info("测试 [" + i + "]" +
                        " | 输入层: " + inputValues[i][0] + ", " + inputValues[i][1] +
                        " | 隐含层: " + hidden1 + ", " + hidden2 +
                        " | 输出层: " + boolResult + "(" + output + ")" +
                        " | 期望结果: " + expectedOutput[i] +
                        " | 成功: " + (boolResult == expectedOutput[i]));
        }
        // endregion
    }

    private final Arena arena = Arena.ofConfined();

    private final RenderContext cx;
    private final ShaderCompiler compiler;

    private static ShadercUtil.IncludeResult unsupportedResolve(
            String requestedSource,
            @EnumType(ShadercIncludeType.class) int includeType,
            String requestingSource,
            long includeDepth
    ) {
        throw new UnsupportedOperationException();
    }

    private static final int inputSize = 2;
    private static final int hiddenLayerSize = 2;
    private static final int outputLayerSize = 1;

    private static final float[][] inputValues = {
            { 0.0f, 0.0f },
            { 0.0f, 1.0f },
            { 1.0f, 0.0f },
            { 1.0f, 1.0f }
    };
    private static final boolean[] expectedOutput = {
            false, // 0.0f, 0.0f
            true,  // 0.0f, 1.0f
            true,  // 1.0f, 0.0f
            false  // 1.0f, 1.0f
    };

    private static final float[][] perceptronWeights = {
            // hidden layer: 2 perceptron, 2 inputs for each perceptron
            {
                    // perceptron 1
                    20.0f, 20.0f,
                    // perceptron 2
                    -20.0f, -20.0f
            },
            // output layer: 1 perceptron, 2 inputs
            {
                    // perceptron
                    20.0f, 20.0f
            }
    };
    private static final float[][] perceptronBias = {
            // hidden layer: 2 biases for each perceptron
            {
                    -10.0f,
                    30.0f
            },
            // output layer: 1 bias for the perceptron
            {
                    -30.0f
            }
    };
    private static final Logger logger = Logger.getLogger(Application.class.getName());
}
