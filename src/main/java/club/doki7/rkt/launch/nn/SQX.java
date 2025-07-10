package club.doki7.rkt.launch.nn;

import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.FloatPtr;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.util.Assertion;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.common.QueueFamily;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.rkt.vk.resc.Transmission;
import club.doki7.vulkan.command.VulkanLoader;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;

public final class SQX {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
    }

    public static void main(String[] ignored) {
        try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
             ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma");
             SQX_App app = new SQX_App(libVulkan, libVMA)) {
            app.applicationStart();
        } catch (Throwable e) {
            e.printStackTrace(System.err);
        }
    }
}

final class SQX_App implements AutoCloseable {
    @Override
    public void close() {
        cx.close();
    }

    SQX_App(ISharedLibrary libVulkan, ISharedLibrary libVMA) throws RenderException {
        this.cx = RenderContext.createHeadless(libVulkan, libVMA, new RenderConfig());
    }

    void applicationStart() throws IOException, RenderException {
        MLPOptions options = new MLPOptions(
                2,
                List.of(
                        new MLPOptions.Layer(16, Activation.RELU, 16),
                        new MLPOptions.Layer(16, Activation.RELU, 16),
                        new MLPOptions.Layer(2, Activation.LINEAR, 2)
                ),
                true
        );

        try (MLPFactory factory = new MLPFactory(cx);
             MLP model = factory.createModel(options)) {
            loadWeight(model, "sqx_trained_");
            infer(model);

            logger.info("开始自行训练模型，与预训练结果进行比对");

            train(model);
            infer(model);
        }
    }

    private void loadWeight(MLP model, String weightPrefix) throws IOException, RenderException {
        List<MemorySegment> weightList = new ArrayList<>();
        List<MemorySegment> biasList = new ArrayList<>();

        int inputSize = model.options.inputSize;
        for (int i = 0; i < model.options.layers.size(); i++) {
            MLPOptions.Layer layer = model.options.layers.get(i);

            String weightFileName = weightPrefix + weightFileNameList.get(i);
            String biasFileName = weightPrefix + biasFileNameList.get(i);

            byte[] weights = Files.readAllBytes(Path.of("resc", "nn", weightFileName));
            byte[] biases = Files.readAllBytes(Path.of("resc", "nn", biasFileName));
            assert weights.length == layer.size * inputSize * Float.BYTES;
            assert biases.length == layer.size * Float.BYTES;

            weightList.add(MemorySegment.ofArray(weights));
            biasList.add(MemorySegment.ofArray(biases));

            inputSize = layer.size;
        }

        model.uploadWeights(weightList, biasList);
    }

    private void train(MLP model) throws RenderException, IOException {
        final int trainDataSize = 2000;
        final int batchSize = 64;

        byte[] inputData = Files.readAllBytes(Path.of("resc", "nn", "sqx_train_inputs.bin"));
        assert inputData.length == 2 * trainDataSize * Float.BYTES;
        byte[] labelData = Files.readAllBytes(Path.of("resc", "nn", "sqx_train_labels.bin"));
        assert labelData.length == trainDataSize * Integer.BYTES;

        Buffer.Options ioBufferOptions = Buffer.OptionsInit.shaderStorageBufferPreset().build();

        try (Buffer inputBuffer = Buffer.create(cx, inputData.length, false, ioBufferOptions);
             Buffer labelBuffer = Buffer.create(cx, labelData.length, false, ioBufferOptions);
             MLPTrainTask trainTask = new MLPTrainTask(model, batchSize, inputBuffer, labelBuffer, LossFunction.CROSS_ENTROPY)) {

            QueueFamily queueAffinity = cx.hasComputeQueue() ? QueueFamily.COMPUTE : QueueFamily.GRAPHICS;
            Transmission.uploadBuffer(cx, inputBuffer, MemorySegment.ofArray(inputData), queueAffinity);
            Transmission.uploadBuffer(cx, labelBuffer, MemorySegment.ofArray(labelData), queueAffinity);

            if (Assertion.assertionEnabled) {
                loadWeight(model, "sqx_initial_");
            } else {
                trainTask.prewarm();
            }

            long startTime = System.nanoTime();
            for (int i = 0; i < 50; i++) {
                for (int batchStart = 0; batchStart < trainDataSize; batchStart += batchSize) {
                    trainTask.executeBatch(batchStart, 0.05f);
                }
            }
            long endTime = System.nanoTime();
            logger.info("训练耗时: " + (endTime - startTime) / 1000_000 + " ms");
        }
    }

    private void infer(MLP model) throws IOException, RenderException {
        final int testDataSize = 10_000;
        final int batchSize = 1000;

        byte[] inputData = Files.readAllBytes(Path.of("resc", "nn", "sqx_test_inputs.bin"));
        assert inputData.length == 2 * testDataSize * Float.BYTES;

        byte[] labelData = Files.readAllBytes(Path.of("resc", "nn", "sqx_test_labels.bin"));
        assert labelData.length == testDataSize * Integer.BYTES;

        Buffer.Options inputBufferOptions = Buffer.OptionsInit.shaderStorageBufferPreset().build();
        try (Buffer inputBuffer = Buffer.create(cx, inputData.length, false, inputBufferOptions);
             MLPInferTask inferTask = new MLPInferTask(model, batchSize, inputBuffer, true, false);
             Arena arena = Arena.ofConfined()) {
            Transmission.uploadBuffer(
                    cx,
                    inputBuffer,
                    MemorySegment.ofArray(inputData),
                    cx.hasComputeQueue() ? QueueFamily.COMPUTE : QueueFamily.GRAPHICS
            );

            IntPtr labelDataNormalised = IntPtr.allocate(arena, testDataSize);
            labelDataNormalised.segment().copyFrom(MemorySegment.ofArray(labelData));

            FloatPtr outputMapped = Objects.requireNonNull(FloatPtr.checked(
                    inferTask.outputBufferList.getLast().mapped
            ));
            assert outputMapped.size() == batchSize * 2;

            int correctCount = 0;
            for (int batchStart = 0; batchStart < testDataSize; batchStart += batchSize) {
                inferTask.executeBatch(batchStart);

                for (int outputIdx = 0; outputIdx < batchSize; outputIdx++) {
                    FloatPtr oneHot = outputMapped.slice(outputIdx * 2, (outputIdx + 1) * 2);
                    int predictedLabel = oneHot.read(0) > oneHot.read(1) ? 0 : 1;
                    int actualLabel = labelDataNormalised.read(batchStart + outputIdx);
                    if (predictedLabel == actualLabel) {
                        correctCount++;
                    }
                }
            }

            float accuracy = (float) correctCount / (float) testDataSize;
            logger.info("推理准确率: " + accuracy * 100.0f + "%");
        }
    }

    private final RenderContext cx;
    private static final Logger logger = Logger.getLogger(SQX_App.class.getName());

    private static final List<String> weightFileNameList = List.of(
            "weights_L1.bin",
            "weights_L2.bin",
            "weights_L3.bin"
    );
    private static final List<String> biasFileNameList = List.of(
            "biases_L1.bin",
            "biases_L2.bin",
            "biases_L3.bin"
    );
}
