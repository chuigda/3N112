package club.doki7.rkt.vk.pipeline;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.vulkan.datatype.VkShaderModuleCreateInfo;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.VkShaderModule;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;
import java.util.Objects;

public final class ShaderModule implements AutoCloseable {
    public final VkShaderModule handle;

    public static ShaderModule create(RenderContext cx, byte[] code) throws VulkanException {
        assert code.length % 4 == 0 : "SPIRV 字节码长度必须是 4 的倍数";

        try (Arena arena = Arena.ofConfined()) {
            VkShaderModuleCreateInfo createInfo = VkShaderModuleCreateInfo.allocate(arena)
                    .pCode(IntPtr.allocate(arena, code))
                    .codeSize(code.length);
            VkShaderModule.Ptr pShaderModule = VkShaderModule.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result = cx.dCmd.createShaderModule(
                    cx.device,
                    createInfo,
                    null,
                    pShaderModule
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建着色器模块");
            }

            VkShaderModule handle = Objects.requireNonNull(pShaderModule.read());
            return new ShaderModule(handle, cx);
        }
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private ShaderModule(VkShaderModule handle, RenderContext context) {
        this.handle = handle;

        IDisposeOnContext d = cx -> cx.dCmd.destroyShaderModule(cx.device, handle, null);
        this.cleanable = context.registerCleanup(this, d, true);
    }

    private final Cleaner.Cleanable cleanable;
}
