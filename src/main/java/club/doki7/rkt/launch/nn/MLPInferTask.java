package club.doki7.rkt.launch.nn;

import club.doki7.ffm.NativeLayout;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.resc.Buffer;

import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class MLPInferTask implements AutoCloseable {
    public final int batchSize;
    public final Buffer inputBuffer;
    public final List<Buffer> outputBufferList;

    public MLPInferTask(
            MLP infer,
            int batchSize,
            Buffer inputBuffer,
            boolean mappedOutputBuffer,
            boolean mappedHiddenLayerOutputBuffer
    ) throws VulkanException {
        this.cx = infer.cx;
        this.infer = infer;
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
        for (int i = 0; i < infer.options.layers.size(); i++) {
             MLPOptions.Layer layer = infer.options.layers.get(i);
             Buffer.Options useOptions = i == infer.options.layers.size() - 1
                     ? outputOptions
                     : hiddenOutputOptions;
             Buffer outputBuffer = Buffer.create(
                     cx,
                     (long) layer.size * batchSize * Float.BYTES,
                     false,
                     useOptions
             );
             outputBufferList.add(outputBuffer);
        }
    }

    @Override
    public void close() throws Exception {
        inferOptionsBuffer.close();
        layer0InferOptionsBuffer.close();
    }

    private final RenderContext cx;
    private final MLP infer;

    private final Buffer inferOptionsBuffer;
    private final Buffer layer0InferOptionsBuffer;

    private static final StructLayout INFER_OPTIONS_LAYOUT = NativeLayout.structLayout(
            ValueLayout.JAVA_INT.withName("input_offset"),
            ValueLayout.JAVA_INT.withName("batch_size")
    );
}
