package club.doki7.rkt.mmap;

import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;

import java.io.IOException;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public final class MMapImplWin32 implements IMMapImpl {
    @Override
    public IMappedMemory mapMemory(String fileName) throws IOException {
        // before actually commencing operation, let
        return null;
    }

    MMapImplWin32() {
        kernel32 = ILibraryLoader.platformLoader().loadLibrary("kernel32");
        pfnCreateFileW = kernel32.load("CreateFileW");
        pfnCreateFileMappingW = kernel32.load("CreateFileMappingW");
        pfnCloseHandle = kernel32.load("CloseHandle");

        Linker linker = Linker.nativeLinker();
        Linker.Option ccs = Linker.Option.captureCallState("GetLastError");
        mhCreateFileW = linker.downcallHandle(pfnCreateFileW, DESC$CreateFileW, ccs);
        mhCreateFileMappingW = linker.downcallHandle(pfnCreateFileMappingW, DESC$CreateFileMappingW, ccs);
        mhCloseHandle = linker.downcallHandle(pfnCloseHandle, DESC$CloseHandle);
    }

    private final ISharedLibrary kernel32;
    private final MemorySegment pfnCreateFileW;
    private final MemorySegment pfnCreateFileMappingW;
    private final MemorySegment pfnCloseHandle;
    private final MethodHandle mhCreateFileW;
    private final MethodHandle mhCreateFileMappingW;
    private final MethodHandle mhCloseHandle;

    private static final FunctionDescriptor DESC$CreateFileW = FunctionDescriptor.of(
            ValueLayout.ADDRESS, // returns HANDLE
            ValueLayout.ADDRESS.withTargetLayout(NativeLayout.WCHAR_T), // LPCSTRW lpFileName,
            ValueLayout.JAVA_INT,                                       // DWORD dwDesiredAccess,
            ValueLayout.JAVA_INT,                                       // DWORD dwShareMode,
            ValueLayout.ADDRESS,                                        // LPSECURITY_ATTRIBUTES lpSecurityAttributes,
            ValueLayout.JAVA_INT,                                       // DWORD dwCreationDisposition,
            ValueLayout.JAVA_INT,                                       // DWORD dwFlagsAndAttributes,
            ValueLayout.ADDRESS                                         // HANDLE hTemplateFile
    );
    private static final FunctionDescriptor DESC$CreateFileMappingW = FunctionDescriptor.of(
            ValueLayout.ADDRESS,                                       // returns HANDLE
            ValueLayout.ADDRESS,                                       // HANDLE hFile,
            ValueLayout.ADDRESS,                                       // LPSECURITY_ATTRIBUTES lpFileMappingAttributes,
            ValueLayout.JAVA_INT,                                      // DWORD flProtect,
            ValueLayout.JAVA_INT,                                      // DWORD dwMaximumSizeHigh,
            ValueLayout.JAVA_INT,                                      // DWORD dwMaximumSizeLow,
            ValueLayout.ADDRESS.withTargetLayout(NativeLayout.WCHAR_T) // LPCWSTR lpName
    );
    private static final FunctionDescriptor DESC$CloseHandle = FunctionDescriptor.of(
            ValueLayout.JAVA_INT, // returns BOOL
            ValueLayout.ADDRESS   // HANDLE hObject
    );
}
