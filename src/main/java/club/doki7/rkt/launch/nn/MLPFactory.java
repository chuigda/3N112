package club.doki7.rkt.launch.nn;

import club.doki7.ffm.NativeLayout;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.desc.DescriptorSetLayout;
import club.doki7.rkt.vk.pipeline.PipelineLayout;
import club.doki7.rkt.vk.pipeline.ShaderModule;

import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;

public final class MLPFactory implements AutoCloseable {
    public MLPFactory(RenderContext cx) {
        this.cx = cx;
    }

    @Override
    public void close() throws Exception {
        mlpForwardModule.close();
    }

    private final RenderContext cx;
    private final ShaderModule mlpForwardModule;
    private final DescriptorSetLayout mlpForwardDescSetLayout;
    private final PipelineLayout mlpForwardPipelineLayout;

    private static final StructLayout MLP_FORWARD_SHADER_SPEC_LAYOUT = NativeLayout.structLayout(
            ValueLayout.JAVA_INT, // const uint tx
            ValueLayout.JAVA_INT, // const uint ty
            ValueLayout.JAVA_INT, // const uint perceptron_count
            ValueLayout.JAVA_INT, // const uint input_size
            ValueLayout.JAVA_INT, // const uint activation
            ValueLayout.JAVA_INT  // const uint max_shared_input_size
    );
}
