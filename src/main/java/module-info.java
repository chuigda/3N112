module club.doki7.sputnik {
    requires club.doki7.ffm;
    requires club.doki7.glfw;
    requires club.doki7.vma;
    requires club.doki7.vulkan;

    requires java.logging;
    requires org.jetbrains.annotations;

    exports club.doki7.sputnik.exc;
    exports club.doki7.sputnik.util;

    exports club.doki7.sputnik.vk;
    exports club.doki7.sputnik.vk.cmd;
    exports club.doki7.sputnik.vk.sync;
}
