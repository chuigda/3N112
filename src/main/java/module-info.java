module club.doki7.cg112 {
    requires club.doki7.ffm;
    requires club.doki7.glfw;
    requires club.doki7.vma;
    requires club.doki7.vulkan;

    requires java.logging;
    requires org.jetbrains.annotations;

    exports club.doki7.cg112.exc;
    exports club.doki7.cg112.util;

    exports club.doki7.cg112.vk;
    exports club.doki7.cg112.vk.cmd;
    exports club.doki7.cg112.vk.sync;
}
