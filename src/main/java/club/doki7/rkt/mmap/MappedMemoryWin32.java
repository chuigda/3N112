package club.doki7.rkt.mmap;

import java.lang.foreign.MemorySegment;

public final class MappedMemoryWin32 implements IMappedMemory {
    final MemorySegment hFile;
    final MemorySegment hMapping;

    public MappedMemoryWin32(MemorySegment hFile, MemorySegment hMapping) {
        this.hFile = hFile;
        this.hMapping = hMapping;
    }

    @Override
    public void close() throws Exception {
    }
}
