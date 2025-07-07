package club.doki7.rkt.launch.nn;

import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.pipeline.ComputePipeline;
import club.doki7.rkt.vk.resc.Buffer;

import java.lang.foreign.MemorySegment;
import java.util.List;

public final class MLP implements AutoCloseable {
    public final MLPOptions options;

    public MLP(
            MLPOptions options,
            RenderContext cx,
            List<Buffer> weightBufferList,
            List<Buffer> biasBufferList,
            List<ComputePipeline> computePipelineList
    ) {
        this.options = options;
        this.cx = cx;
        this.weightBufferList = weightBufferList;
        this.biasBufferList = biasBufferList;
        this.computePipelineList = computePipelineList;
    }

    public void uploadWeights(
            List<MemorySegment> weightList,
            List<MemorySegment> biasList
    ) throws RenderException {
        if (cx.hasTransferQueue()) {
            uploadWithTransferQueue(weightList, biasList);
        } else {
            uploadWithAffinityQueue(weightList, biasList);
        }
    }

    private void uploadWithAffinityQueue(
            List<MemorySegment> weightList,
            List<MemorySegment> biasList
    ) throws RenderException {

    }

    private void uploadWithTransferQueue(
            List<MemorySegment> weightList,
            List<MemorySegment> biasList
    ) throws RenderException {
    }

    @Override
    public void close() throws Exception {
        for (ComputePipeline pipeline : computePipelineList) {
            pipeline.close();
        }
        for (Buffer buffer : weightBufferList) {
            buffer.close();
        }
        for (Buffer buffer : biasBufferList) {
            buffer.close();
        }
    }

    final RenderContext cx;
    final List<Buffer> weightBufferList;
    final List<Buffer> biasBufferList;
    final List<ComputePipeline> computePipelineList;
}
