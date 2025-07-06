package club.doki7.rkt.mmap;

import java.io.IOException;

public sealed interface IMMapImpl permits MMapImplWin32 {
    IMappedMemory mapMemory(String fileName) throws IOException;
}
