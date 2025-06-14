package club.doki7.cg112.vk;

import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.vk.cleanup.RenderWindowCleanup;
import club.doki7.ffm.annotation.Pointer;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.glfw.GLFW;
import club.doki7.glfw.GLFWFunctionTypes;
import club.doki7.glfw.handle.GLFWwindow;

import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.ref.Cleaner;

public final class RenderWindow implements AutoCloseable {
    public final GLFW glfw;
    public final GLFWwindow rawWindow;
    public int width;
    public int height;

    public boolean framebufferResized = true;
    private final IntPtr pWidthHeight = IntPtr.allocate(Arena.ofAuto(), 2);

    public RenderWindow(GLFW glfw, String title, int width, int height) throws RenderException {
        this.glfw = glfw;
        if (glfw.vulkanSupported() != GLFW.TRUE) {
            throw new RenderException("GLFW 报告其不支持 Vulkan");
        }

        glfw.windowHint(GLFW.CLIENT_API, GLFW.NO_API);
        try (Arena arena = Arena.ofConfined()) {
            BytePtr titleBuffer = BytePtr.allocateString(arena, title);
            this.rawWindow = glfw.createWindow(width, height, titleBuffer, null, null);
            if (this.rawWindow == null) {
                throw new RenderException("无法创建 GLFW 窗口");
            }
            this.width = width;
            this.height = height;

            MethodHandle handle = MethodHandles.lookup().findVirtual(
                    RenderWindow.class,
                    "framebufferSizeCallback",
                    GLFWFunctionTypes.GLFWframebuffersizefun.toMethodType()
            ).bindTo(this);
            MemorySegment pfn = Linker.nativeLinker().upcallStub(
                    handle,
                    GLFWFunctionTypes.GLFWframebuffersizefun,
                    Arena.global()
            );
            glfw.setFramebufferSizeCallback(rawWindow, pfn);

            RenderWindowCleanup cleanup = new RenderWindowCleanup(glfw, rawWindow);
            cleanable = cleaner.register(this, cleanup::dispose);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new RuntimeException("找不到回调函数 RenderWindow::framebufferSizeCallback", e);
        }
    }

    public boolean tick() {
        if (glfw.windowShouldClose(rawWindow) == GLFW.TRUE) {
            return false;
        }

        glfw.pollEvents();
        if (framebufferResized) {
            glfw.getFramebufferSize(rawWindow, pWidthHeight, pWidthHeight.offset(1));
            width = pWidthHeight.read(0);
            height = pWidthHeight.read(1);
        }
        return true;
    }

    private void framebufferSizeCallback(
            @Pointer(comment="GLFWwindow*") MemorySegment ignoredWindow,
            int ignoredWidth,
            int ignoredHeight
    ) {
        framebufferResized = true;
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private final Cleaner.Cleanable cleanable;
    private static final Cleaner cleaner = Cleaner.create();
}
