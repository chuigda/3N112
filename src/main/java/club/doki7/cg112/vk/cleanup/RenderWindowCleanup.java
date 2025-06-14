package club.doki7.cg112.vk.cleanup;

import club.doki7.glfw.GLFW;
import club.doki7.glfw.handle.GLFWwindow;

import java.util.logging.Logger;

public final class RenderWindowCleanup implements IDisposable {
    public RenderWindowCleanup(GLFW glfw, GLFWwindow rawWindow) {
        this.glfw = glfw;
        this.rawWindow = rawWindow;
    }

    @Override
    public void dispose() {
        glfw.destroyWindow(rawWindow);
        logger.info("已销毁绘图窗口");
    }

    private final GLFW glfw;
    private final GLFWwindow rawWindow;

    private static final Logger logger = Logger.getLogger(RenderWindowCleanup.class.getName());
}
