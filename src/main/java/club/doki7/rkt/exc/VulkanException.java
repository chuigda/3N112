package club.doki7.rkt.exc;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.vulkan.enumtype.VkResult;

public final class VulkanException extends RenderException {
    public @EnumType(VkResult.class) int result;

    public VulkanException(@EnumType(VkResult.class) int result, String message) {
        super(message);
        this.result = result;
    }

    @Override
    public String getMessage() {
        return "Vulkan 错误: " + super.getMessage() + " (错误码: " + VkResult.explain(result) + ")";
    }
}
