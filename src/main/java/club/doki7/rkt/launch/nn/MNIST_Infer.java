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

public final class MNIST_Infer {
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
                        new MLPOptions.Layer(300, Activation.SIGMOID),
                        new MLPOptions.Layer(100, Activation.SIGMOID),
                        new MLPOptions.Layer(10, Activation.LINEAR)
                ),
                64,
                false
        );

        byte[] inputData = Files.readAllBytes(Path.of("resc", "nn", "t10k-images.idx3-ubyte.bin"));
        assert inputData.length == MNIST_IMAGE_SIZE * 10_000 + MNIST_IMAGE_FILE_HEADER_SIZE;

        byte[] labelData = Files.readAllBytes(Path.of("resc", "nn", "t10k-labels.idx1-ubyte.bin"));
        assert labelData.length == 10_000 + MNIST_LABEL_FILE_HEADER_SIZE;

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

        Buffer.Options inputBufferOptions = Buffer.OptionsInit.shaderStorageBufferPreset().build();
        long inputItemCount = inputData.length - MNIST_IMAGE_FILE_HEADER_SIZE;
        long inputBufferSize = inputItemCount * Float.BYTES;

        int batchSize = 1000;
        try (MLPFactory factory = new MLPFactory(cx);
             MLP model = factory.createModel(options);
             Buffer inputBuffer = Buffer.create(cx, inputBufferSize, false, inputBufferOptions);
             MLPInferTask inferTask = new MLPInferTask(model, batchSize, inputBuffer, true, false);
             Arena arena = Arena.ofConfined()) {
            model.uploadWeights(weightList, biasList);

            FloatPtr normalisedInput = FloatPtr.allocate(arena, inputItemCount);
            for (int i = MNIST_IMAGE_FILE_HEADER_SIZE; i < inputData.length; i++) {
                normalisedInput.write(i - MNIST_IMAGE_FILE_HEADER_SIZE, (inputData[i] & 0xFF) / 255.0f);
            }
            Transmission.uploadBuffer(cx, inputBuffer, normalisedInput.segment(), cx.hasComputeQueue()
                    ? QueueFamily.COMPUTE
                    : QueueFamily.GRAPHICS);

            FloatPtr outputMapped = Objects.requireNonNull(FloatPtr.checked(inferTask.outputBufferList.getLast().mapped));
            assert outputMapped.size() == batchSize * 10;
            int correctCount = 0;
            for (int i = 0; i < (inputItemCount / MNIST_IMAGE_SIZE); i += batchSize) {
                inferTask.executeBatch(i);

                for (int outputIdx = 0; outputIdx < batchSize; outputIdx++) {
                    FloatPtr oneHot = outputMapped.slice(outputIdx * 10, (outputIdx + 1) * 10);
                    byte actual = max10(oneHot);
                    byte expected = labelData[MNIST_LABEL_FILE_HEADER_SIZE + i + outputIdx];

                    if (actual == expected) {
                        correctCount++;
                    }
                }
            }

            float accuracy = (float) correctCount / 10_000.0f;
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
