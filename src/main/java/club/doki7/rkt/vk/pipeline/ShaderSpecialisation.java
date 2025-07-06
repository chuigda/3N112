package club.doki7.rkt.vk.pipeline;

import club.doki7.ffm.annotation.Unsigned;

import java.lang.foreign.MemorySegment;
import java.util.Collections;
import java.util.List;

public final class ShaderSpecialisation {
    public static final class Entry {
        public @Unsigned final int constantId;
        public @Unsigned final int offset;
        public @Unsigned final long size;

        public Entry(int constantId, int offset, long size) {
            this.constantId = constantId;
            this.offset = offset;
            this.size = size;
        }
    }

    public final List<Entry> entries;
    public final MemorySegment data;

    public ShaderSpecialisation(List<Entry> entries, MemorySegment data) {
        this.entries = Collections.unmodifiableList(entries);
        this.data = data;
    }
}
