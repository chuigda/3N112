module club.doki7.rkt {
    requires club.doki7.ffm;
    requires club.doki7.glfw;
    requires club.doki7.vma;
    requires club.doki7.vulkan;

    requires java.logging;
    requires org.jetbrains.annotations;
    requires club.doki7.stb;
    requires club.doki7.shaderc;

    exports club.doki7.rkt.exc;
    exports club.doki7.rkt.util;

    exports club.doki7.rkt.vk;
    exports club.doki7.rkt.vk.cmd;
    exports club.doki7.rkt.vk.sync;
}
