package club.doki7.rkt.vk.init;

import club.doki7.ffm.annotation.Bitmask;
import club.doki7.ffm.annotation.NativeType;
import club.doki7.ffm.annotation.Pointer;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.vulkan.VkConstants;
import club.doki7.vulkan.VkFunctionTypes;
import club.doki7.vulkan.bitmask.VkDebugUtilsMessageSeverityFlagsEXT;
import club.doki7.vulkan.bitmask.VkDebugUtilsMessageTypeFlagsEXT;
import club.doki7.vulkan.datatype.VkDebugUtilsMessengerCallbackDataEXT;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.function.Consumer;
import java.util.logging.Logger;

class DebugCallback {
    private static @NativeType("VkBool32") int debugCallback(
            @Bitmask(VkDebugUtilsMessageSeverityFlagsEXT.class) int messageSeverity,
            @Bitmask(VkDebugUtilsMessageTypeFlagsEXT.class) int ignoredMessageType,
            @Pointer(comment="const VkDebugUtilsMessengerCallbackDataEXT*") MemorySegment pCallbackData,
            @Pointer(comment="void*") MemorySegment ignoredPUserData
    ) {
        VkDebugUtilsMessengerCallbackDataEXT callbackData =
                new VkDebugUtilsMessengerCallbackDataEXT(pCallbackData.reinterpret(VkDebugUtilsMessengerCallbackDataEXT.BYTES));
        @Nullable BytePtr pMessageIdName = callbackData.pMessageIdName();
        String messageIdName = pMessageIdName != null ? pMessageIdName.readString() : "UNKNOWN-ID";

        if (messageIdName.equals("Loader Message")) {
            return VkConstants.TRUE;
        }

        @Nullable BytePtr pMessage = callbackData.pMessage();
        String message = pMessage != null ? pMessage.readString() : "发生了未知错误, 没有诊断消息可用";
        message = messageIdName + ": " + message;

        Consumer<String> action = getSeverityLoggingFunction(messageSeverity);
        action.accept(message);

        if (messageSeverity >= VkDebugUtilsMessageSeverityFlagsEXT.ERROR) {
            StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
            StringBuilder sb = new StringBuilder();
            sb.append("JVM 调用栈:\n");
            for (StackTraceElement stackTraceElement : stackTrace) {
                sb.append("\t").append(stackTraceElement).append("\n");
            }
            action.accept(sb.toString());
        }

        return VkConstants.FALSE;
    }

    private static @NotNull Consumer<String> getSeverityLoggingFunction(@Bitmask(VkDebugUtilsMessageSeverityFlagsEXT.class) int messageSeverity) {
        Consumer<String> action;
        if (messageSeverity >= VkDebugUtilsMessageSeverityFlagsEXT.ERROR) {
            action = logger::severe;
        } else if (messageSeverity >= VkDebugUtilsMessageSeverityFlagsEXT.WARNING) {
            action = logger::warning;
        } else if (messageSeverity >= VkDebugUtilsMessageSeverityFlagsEXT.INFO) {
            action = logger::info;
        } else {
            action = logger::fine;
        }
        return action;
    }

    public static final MemorySegment UPCALL_debugCallback;
    static {
        try {
            MethodHandle handle = MethodHandles.lookup().findStatic(
                    DebugCallback.class,
                    "debugCallback",
                    VkFunctionTypes.PFN_vkDebugUtilsMessengerCallbackEXT.toMethodType()
            );
            UPCALL_debugCallback = Linker.nativeLinker().upcallStub(
                    handle,
                    VkFunctionTypes.PFN_vkDebugUtilsMessengerCallbackEXT,
                    Arena.global()
            );
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static final Logger logger = Logger.getLogger("vulkan.validation-layer");
}
