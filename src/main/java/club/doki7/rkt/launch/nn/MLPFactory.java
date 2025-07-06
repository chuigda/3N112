package club.doki7.rkt.launch.nn;

import club.doki7.rkt.vk.pipeline.ShaderModule;

public final class MLPFactory implements AutoCloseable {
    @Override
    public void close() throws Exception {
    }

    private final ShaderModule mlpForwardModule;
}
