package club.doki7.rkt.vk.pipeline;

import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.vulkan.handle.VkShaderModule;

import java.lang.ref.Cleaner;

public final class ShaderModule implements AutoCloseable {
    public final VkShaderModule handle;

    @Override
    public void close() throws Exception {
        cleanable.clean();
    }

    private ShaderModule(VkShaderModule handle, RenderContext context) {
        this.handle = handle;

        IDisposeOnContext d = cx -> cx.dCmd.destroyShaderModule(cx.device, handle, null);
        this.cleanable = context.registerCleanup(this, d, true);
    }

    private final Cleaner.Cleanable cleanable;
}
