package club.doki7.rkt.launch.nn;

import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.FloatPtr;
import club.doki7.rkt.exc.RenderException;
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

public final class MNIST {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
    }

    public static void main(String[] ignored) {
        try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
             ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma");
             Application app = new Application(libVulkan, libVMA)) {
            app.applicationStart();
        } catch (Throwable e) {
            e.printStackTrace(System.err);
        }
    }
}

final class Application implements AutoCloseable {
    @Override
    public void close() {
        cx.close();
    }

    Application(ISharedLibrary libVulkan, ISharedLibrary libVMA) throws RenderException {
        this.cx = RenderContext.createHeadless(libVulkan, libVMA, new RenderConfig());
    }

    void applicationStart() throws RenderException, IOException {
        MLPOptions options = new MLPOptions(
                MNIST_IMAGE_SIZE,
                List.of(
                        new MLPOptions.Layer(300, Activation.SIGMOID, 32),
                        new MLPOptions.Layer(100, Activation.SIGMOID, 32),
                        new MLPOptions.Layer(10, Activation.LINEAR, 2)
                ),
                true
        );

        try (MLPFactory factory = new MLPFactory(cx);
             MLP model = factory.createModel(options)) {
            train(model);
            loadWeights(model);
            infer(model);
        }
    }

    private void train(MLP model) throws RenderException, IOException {
        final int trainDataSize = 60_000;
        final int batchSize = 64;

        byte[] inputData = Files.readAllBytes(Path.of("resc", "nn", "train-images-idx3-ubyte.bin"));
        assert inputData.length == MNIST_IMAGE_SIZE * trainDataSize + MNIST_IMAGE_FILE_HEADER_SIZE;

        byte[] labelData = Files.readAllBytes(Path.of("resc", "nn", "train-labels-idx1-ubyte.bin"));
        assert labelData.length == trainDataSize + MNIST_LABEL_FILE_HEADER_SIZE;

        Buffer.Options ioBufferOptions = Buffer.OptionsInit.shaderStorageBufferPreset().build();
        try (Buffer inputBuffer = Buffer.create(cx, trainDataSize * MNIST_IMAGE_SIZE * Float.BYTES, false, ioBufferOptions);
             Buffer labelBuffer = Buffer.create(cx, trainDataSize, false, ioBufferOptions);
             MLPTrainTask trainTask = new MLPTrainTask(model, batchSize, inputBuffer, labelBuffer, LossFunction.CROSS_ENTROPY);
             Arena arena = Arena.ofConfined()) {

            FloatPtr normalisedInput = FloatPtr.allocate(arena, inputData.length - MNIST_IMAGE_FILE_HEADER_SIZE);
            for (int i = MNIST_IMAGE_FILE_HEADER_SIZE; i < inputData.length; i++) {
                normalisedInput.write(i - MNIST_IMAGE_FILE_HEADER_SIZE, (inputData[i] & 0xFF) / 255.0f);
            }

            QueueFamily queueAffinity = cx.hasComputeQueue() ? QueueFamily.COMPUTE : QueueFamily.GRAPHICS;
            Transmission.uploadBuffer(cx, inputBuffer, normalisedInput.segment(), queueAffinity);
            Transmission.uploadBuffer(
                    cx,
                    labelBuffer,
                    MemorySegment.ofArray(labelData).asSlice(MNIST_LABEL_FILE_HEADER_SIZE),
                    queueAffinity
            );

            trainTask.prewarm();
        }
    }

    private void loadWeights(MLP mlp) throws RenderException, IOException {
        List<MemorySegment> weightList = new ArrayList<>();
        List<MemorySegment> biasList = new ArrayList<>();
        for (int i = 0; i < weightFileNameList.size(); i++) {
            String weightFileName = weightFileNameList.get(i);
            String biasFileName = biasFileNameList.get(i);

            byte[] weights = Files.readAllBytes(Path.of("resc", "nn", weightFileName));
            byte[] biases = Files.readAllBytes(Path.of("resc", "nn", biasFileName));
            assert weights.length == weightFileSize.get(i) && biases.length == biasFileSize.get(i);

            weightList.add(MemorySegment.ofArray(weights));
            biasList.add(MemorySegment.ofArray(biases));
        }

        mlp.uploadWeights(weightList, biasList);
    }

    private void infer(MLP model) throws RenderException, IOException {
        final int testDataSize = 10_000;
        final int batchSize = 1000;

        byte[] inputData = Files.readAllBytes(Path.of("resc", "nn", "t10k-images-idx3-ubyte.bin"));
        assert inputData.length == MNIST_IMAGE_SIZE * testDataSize + MNIST_IMAGE_FILE_HEADER_SIZE;

        byte[] labelData = Files.readAllBytes(Path.of("resc", "nn", "t10k-labels-idx1-ubyte.bin"));
        assert labelData.length == testDataSize + MNIST_LABEL_FILE_HEADER_SIZE;

        Buffer.Options inputBufferOptions = Buffer.OptionsInit.shaderStorageBufferPreset().build();
        try (Buffer inputBuffer = Buffer.create(cx, testDataSize * MNIST_IMAGE_SIZE * Float.BYTES, false, inputBufferOptions);
             MLPInferTask inferTask = new MLPInferTask(model, batchSize, inputBuffer, true, false);
             Arena arena = Arena.ofConfined()) {
            FloatPtr normalisedInput = FloatPtr.allocate(arena, inputData.length - MNIST_IMAGE_FILE_HEADER_SIZE);
            for (int i = MNIST_IMAGE_FILE_HEADER_SIZE; i < inputData.length; i++) {
                normalisedInput.write(i - MNIST_IMAGE_FILE_HEADER_SIZE, (inputData[i] & 0xFF) / 255.0f);
            }
            Transmission.uploadBuffer(cx, inputBuffer, normalisedInput.segment(), cx.hasComputeQueue()
                    ? QueueFamily.COMPUTE
                    : QueueFamily.GRAPHICS);

            FloatPtr outputMapped = Objects.requireNonNull(FloatPtr.checked(inferTask.outputBufferList.getLast().mapped));
            assert outputMapped.size() == batchSize * 10;
            int correctCount = 0;
            for (int batchIdx = 0; batchIdx < testDataSize; batchIdx += batchSize) {
                long startTime = System.nanoTime();
                inferTask.executeBatch(batchIdx);
                long endTime = System.nanoTime();

                logger.info("批次 " + (batchIdx / batchSize + 1) + " 推理耗时: " + (endTime - startTime) / 1000_000 + " ms");

                for (int outputIdx = 0; outputIdx < batchSize; outputIdx++) {
                    FloatPtr oneHot = outputMapped.slice(outputIdx * 10, (outputIdx + 1) * 10);
                    byte actual = max10(oneHot);
                    byte expected = labelData[MNIST_LABEL_FILE_HEADER_SIZE + batchIdx + outputIdx];

                    if (actual == expected) {
                        correctCount++;
                    }
                }
            }

            float accuracy = (float) correctCount / (float) testDataSize;
            logger.info("推理准确率: " + accuracy * 100.0f + "%");
        }
    }

    private byte max10(FloatPtr oneHot) {
        float maxValue = oneHot.read();
        byte maxIndex = 0;

        for (byte i = 1; i < 10; i++) {
            float value = oneHot.read(i);
            if (value > maxValue) {
                maxValue = value;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private final RenderContext cx;

    private static final Logger logger = Logger.getLogger(Application.class.getName());
    private static final List<String> weightFileNameList = List.of(
            "weights_L1_784x300.bin",
            "weights_L2_300x100.bin",
            "weights_L3_100x10.bin"
    );
    private static final List<Long> weightFileSize = List.of(
            784L * 300 * Float.BYTES,
            300L * 100 * Float.BYTES,
            100L * 10 * Float.BYTES
    );
    private static final List<String> biasFileNameList = List.of(
            "biases_L1_784x300.bin",
            "biases_L2_300x100.bin",
            "biases_L3_100x10.bin"
    );
    private static final List<Long> biasFileSize = List.of(
            300L * Float.BYTES,
            100L * Float.BYTES,
            10L * Float.BYTES
    );
    private static final int MNIST_IMAGE_FILE_HEADER_SIZE = 16;
    private static final int MNIST_LABEL_FILE_HEADER_SIZE = 8;
    private static final int MNIST_IMAGE_SIZE = 28 * 28;
}
