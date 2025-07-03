package club.doki7.rkt.vk.pipeline;

import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.desc.DescriptorSetLayout;
import club.doki7.rkt.vk.desc.PushConstantLayout;
import club.doki7.vulkan.handle.VkPipelineLayout;

import java.lang.ref.Cleaner;

public final class PipelineLayout implements AutoCloseable {
    public final VkPipelineLayout handle;
    public final DescriptorSetLayout descriptorSetLayout;
    public final PushConstantLayout pushConstantLayout;

    @Override
    public void close() throws Exception {
        cleanable.clean();
    }

    private PipelineLayout(
            VkPipelineLayout handle,
            DescriptorSetLayout descriptorSetLayout,
            PushConstantLayout pushConstantLayout,
            RenderContext context
    ) {
        this.handle = handle;
        this.descriptorSetLayout = descriptorSetLayout;
        this.pushConstantLayout = pushConstantLayout;

        IDisposeOnContext d = cx -> cx.dCmd.destroyPipelineLayout(cx.device, handle, null);
        this.cleanable = context.registerCleanup(this, d, false);
    }

    private final Cleaner.Cleanable cleanable;
}
