package club.doki7.rkt.launch.nn;

import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.FloatPtr;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.vulkan.command.VulkanLoader;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;

public final class XOR {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
    }

    public static void main(String[] ignored) {
        try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
             ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma");
             XOR_Application app = new XOR_Application(libVulkan, libVMA)) {
            app.applicationStart();
        } catch (Throwable e) {
            e.printStackTrace(System.err);
        }
    }
}

final class XOR_Application implements AutoCloseable {
    @Override
    public void close() {
        cx.close();
    }

    XOR_Application(ISharedLibrary libVulkan, ISharedLibrary libVMA) throws RenderException {
        this.cx = RenderContext.createHeadless(libVulkan, libVMA, new RenderConfig());
    }

    void applicationStart() throws RenderException {
        MLPOptions options = new MLPOptions(
                2,
                List.of(
                        new MLPOptions.Layer(2, Activation.RELU, 2),
                        new MLPOptions.Layer(1, Activation.SIGMOID, 1)
                ),
                true
        );

        try (MLPFactory factory = new MLPFactory(cx);
             MLP model = factory.createModel(options)) {
            train(model);
            // loadWeight(model);
            infer(model);
        }
    }

    private void train(MLP model) throws RenderException {
        final int batchSize = 4;

        float[] inputData = {
                0.0f, 0.0f,
                0.0f, 1.0f,
                1.0f, 0.0f,
                1.0f, 1.0f
        };
        float[] labelData = {
                0.0f,
                1.0f,
                1.0f,
                0.0f
        };

        Buffer.OptionsInit optionsInit = Buffer.OptionsInit.shaderStorageBufferPreset();
        optionsInit.mapped = true;
        optionsInit.coherent = true;
        Buffer.Options ioBufferOptions = optionsInit.build();

        try (Buffer inputBuffer = Buffer.create(cx, inputData.length * Float.BYTES, false, ioBufferOptions);
             Buffer labelBuffer = Buffer.create(cx, labelData.length * Float.BYTES, false, ioBufferOptions);
             MLPTrainTask trainTask = new MLPTrainTask(model, batchSize, inputBuffer, labelBuffer, LossFunction.MEAN_SQUARED_ERROR)) {

            FloatPtr inputMapped = Objects.requireNonNull(FloatPtr.checked(inputBuffer.mapped));
            FloatPtr labelMapped = Objects.requireNonNull(FloatPtr.checked(labelBuffer.mapped));
            inputMapped.write(inputData);
            labelMapped.write(labelData);

            trainTask.prewarm();
            long startTime = System.nanoTime();
            for (int i = 0; i < 5000; i++) {
                for (int j = 0; j < inputData.length / 2; j += batchSize) {
                    trainTask.executeBatch(j, 0.1f);
                }
            }
            long endTime = System.nanoTime();
            logger.info("训练耗时: " + (endTime - startTime) / 1000_000 + " ms");
        }
    }

    private void loadWeight(MLP model) throws RenderException {
        List<MemorySegment> weightList = List.of(
                MemorySegment.ofArray(new float[] { 20.0f, 20.0f, -20.0f, -20.0f }),
                MemorySegment.ofArray(new float[] { 20.0f, 20.0f })
        );
        List<MemorySegment> biasList = List.of(
                MemorySegment.ofArray(new float[] { -10.0f, 30.0f }),
                MemorySegment.ofArray(new float[] { -30.0f })
        );
        model.uploadWeights(weightList, biasList);
    }

    private void infer(MLP model) throws RenderException {
        final int testDataSize = 4;
        final int batchSize = 4;

        float[] inputData = {
                0.0f, 0.0f,
                0.0f, 1.0f,
                1.0f, 0.0f,
                1.0f, 1.0f
        };
        float[] labelData = {
                0.0f,
                1.0f,
                1.0f,
                0.0f
        };

        Buffer.OptionsInit optionsInit = Buffer.OptionsInit.shaderStorageBufferPreset();
        optionsInit.mapped = true;
        optionsInit.coherent = true;
        Buffer.Options ioBufferOptions = optionsInit.build();

        try (Buffer inputBuffer = Buffer.create(cx, inputData.length * Float.BYTES, false, ioBufferOptions);
             MLPInferTask inferTask = new MLPInferTask(model, batchSize, inputBuffer, true, false)) {

            FloatPtr inputMapped = Objects.requireNonNull(FloatPtr.checked(inputBuffer.mapped));
            inputMapped.write(inputData);

            inferTask.executeBatch(0);

            FloatPtr outputMapped = Objects.requireNonNull(FloatPtr.checked(inferTask.outputBufferList.getLast().mapped));
            int correctCount = 0;
            for (int outputIdx = 0; outputIdx < batchSize; outputIdx++) {
                float actual = outputMapped.read(outputIdx);
                float expected = labelData[outputIdx];

                if ((actual >= 0.5) == (expected >= 0.5)) {
                    correctCount++;
                }
            }

            float accuracy = (float) correctCount / (float) testDataSize;
            logger.info("推理准确率: " + accuracy * 100.0f + "%");
        }
    }

    private final RenderContext cx;
    private static final Logger logger = Logger.getLogger(XOR_Application.class.getName());
}
