package club.doki7.rkt.launch.nn;

import club.doki7.rkt.vk.pipeline.ComputePipeline;
import club.doki7.rkt.vk.resc.Buffer;

import java.util.List;

public final class MLPInfer implements AutoCloseable {
    public final MLPOptions options;

    public MLPInfer(
            MLPOptions options, Buffer inferOptionsL0Buffer, Buffer inferOptionsBuffer, List<Buffer> layerOptionsBuffer, List<Buffer> weightBuffer, List<Buffer> biasBuffer, List<ComputePipeline> computePipelines) {
        this.options = options;
        this.inferOptionsL0Buffer = inferOptionsL0Buffer;
        this.inferOptionsBuffer = inferOptionsBuffer;
        this.layerOptionsBuffer = layerOptionsBuffer;
        this.weightBuffer = weightBuffer;
        this.biasBuffer = biasBuffer;
        this.computePipelines = computePipelines;
    }

    @Override
    public void close() throws Exception {
        inferOptionsL0Buffer.close();
        inferOptionsL0Buffer.close();

        for (Buffer buffer : layerOptionsBuffer) {
            buffer.close();
        }
        for (Buffer buffer : weightBuffer) {
            buffer.close();
        }
        for (Buffer buffer : biasBuffer) {
            buffer.close();
        }

        for (ComputePipeline pipeline : computePipelines) {
            pipeline.close();
        }
    }

    private final Buffer inferOptionsL0Buffer;
    private final Buffer inferOptionsBuffer;
    private final List<Buffer> layerOptionsBuffer;
    private final List<Buffer> weightBuffer;
    private final List<Buffer> biasBuffer;
    private final List<ComputePipeline> computePipelines;
}
