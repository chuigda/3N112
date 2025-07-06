package club.doki7.rkt.launch.nn;

import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.pipeline.ComputePipeline;
import club.doki7.rkt.vk.resc.Buffer;

import java.util.List;

public final class MLPInfer implements AutoCloseable {
    public final MLPOptions options;

    public MLPInfer(
            MLPOptions options,
            RenderContext cx,
            List<Buffer> weightBuffer,
            List<Buffer> biasBuffer,
            List<ComputePipeline> computePipelines
    ) {
        this.options = options;
        this.cx = cx;
        this.weightBuffer = weightBuffer;
        this.biasBuffer = biasBuffer;
        this.computePipelines = computePipelines;
    }

    @Override
    public void close() throws Exception {
        for (ComputePipeline pipeline : computePipelines) {
            pipeline.close();
        }
        for (Buffer buffer : weightBuffer) {
            buffer.close();
        }
        for (Buffer buffer : biasBuffer) {
            buffer.close();
        }
    }

    final RenderContext cx;
    final List<Buffer> weightBuffer;
    final List<Buffer> biasBuffer;
    final List<ComputePipeline> computePipelines;
}
