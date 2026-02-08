# Progress Report: Native Windows CUDA Port

## Current Status
The project is in the final stages of merging the **HorizonXP optimized CUDA backend** (Driver API based) into the main branch with full native Windows support (no WSL2/Linux required). 

## Achievements
1.  **Backend Integration:** Merged high-performance CUDA kernels, including **Attention v3** (chunked reduction + GQA shared-load) and the full GPU-resident inference pipeline.
2.  **Native Build Pipeline:**
    *   Created `scripts/gen_cuda_header.ps1` to compile CUDA kernels into embedded C binary headers (`.cubin.h`).
    *   Updated `build.ps1` to handle the full native Windows build using `nvcc` and `cl.exe` (MSVC).
3.  **Core OS Porting:**
    *   **Timing:** Implemented a high-resolution Windows timer (`QueryPerformanceCounter`) to replace `gettimeofday`.
    *   **Memory Mapping:** Replaced POSIX `mmap` with Windows `CreateFileMapping` and `MapViewOfFile`.
    *   **Atomics:** Replaced GCC `__atomic` builtins with Windows `Interlocked` intrinsics.
    *   **Signals:** Fixed POSIX signal handling for Windows console events.

## Current Blockers & Challenges
The build is currently hitting several compilation and linkage errors due to architectural differences between the fork and the base project:

1.  **Signature Mismatches:** 
    *   The fork uses a **Global State** model (no `ctx` pointer passed to most kernels).
    *   The core orchestration logic (`voxtral.c`, `voxtral_kernels.c`) still expects a `ctx` pointer for every call.
    *   *Result:* "Too many arguments" or "Incompatible pointer types" errors.
2.  **Header Definition Order:**
    *   `voxtral.h` has circular dependencies or missing type definitions (like `vox_backend_t` and `vox_ctx_t`) when used across multiple backend files.
3.  **MSVC-Specific Syntax Errors:**
    *   **OpenMP:** MSVC is more restrictive than GCC regarding the `if()` clause in `#pragma omp parallel for`. Previous automated fixes left trailing parentheses that need manual cleanup.
    *   **C Types:** Non-standard types like `dim3` (CUDA C++) are being used in pure C wrappers, causing syntax errors in `cl.exe`.
4.  **Timing Logic Cleanup:**
    *   Some parts of the code still attempt to access `.tv_sec` and `.tv_usec` on variables that have already been converted to `double` by the new `get_time_ms()` helper.

## Next Steps
1.  **Header Synchronization:** Flatten `voxtral.h` and ensure all enums and struct members are defined before they are used.
2.  **Kernel Signature Alignment:** Decide on a unified calling convention (preferably removing `ctx` from low-level GPU kernels to match the optimized fork).
3.  **Surgical Code Fixes:** Use manual edits (avoiding regex replaces) to clean up the OpenMP pragmas and timing calculations.
4.  **Verification:** Once linked, run `voxtral.exe` with the `-Cuda` flag to verify kernel loading and transcription accuracy.
