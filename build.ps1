# build.ps1 - Build script for voxtral.c on Windows
param(
    [switch]$Clean,
    [switch]$Blas,
    [switch]$Debug
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

function Test-Command ([string]$Name) {
    (Get-Command $Name -ErrorAction SilentlyContinue) -ne $null
}

function Import-VSEnv {
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        Write-Warning "vswhere.exe not found at $vswhere"
        return $false
    }
    
    $vsroot  = & $vswhere -latest -products * `
               -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
               -property installationPath 2>$null
    if (-not $vsroot) {
        Write-Warning "VS Build Tools not found."
        return $false
    }

    $vcvars = Join-Path $vsroot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) {
        Write-Warning "vcvars64.bat not found at $vcvars"
        return $false
    }

    Write-Host "Importing MSVC environment from $vcvars"
    $envDump = cmd /s /c "`"$vcvars`" && set"
    foreach ($line in $envDump -split "`r?`n") {
        if ($line -match '^(.*?)=(.*)$') {
            $name,$value = $Matches[1],$Matches[2]
            Set-Item -Path "Env:$name" -Value $value
        }
    }
    return $true
}

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

$SRCS = "voxtral.c", "voxtral_kernels.c", "voxtral_audio.c", "voxtral_encoder.c", "voxtral_decoder.c", "voxtral_tokenizer.c", "voxtral_safetensors.c", "main.c"
$TARGET = "voxtral.exe"

if ($Clean) {
    Write-Host "Cleaning..."
    if (Test-Path $TARGET) { Remove-Item $TARGET }
    Get-ChildItem -Filter *.o | Remove-Item
    Get-ChildItem -Filter *.obj | Remove-Item
}

# Try to find a compiler
$CC = ""
if (Test-Command gcc) {
    $CC = "gcc"
} elseif (Test-Command cl) {
    $CC = "cl"
} else {
    Write-Host "Compiler not found in PATH. Trying to import VS environment..."
    if (Import-VSEnv) {
        if (Test-Command cl) {
            $CC = "cl"
        }
    }
}

if (-not $CC) {
    Write-Error "No compiler (gcc or cl) found. Please install MinGW-w64 or Visual Studio Build Tools."
    exit 1
}

Write-Host "Using compiler: $CC"

if ($CC -eq "gcc") {
    $CFLAGS = "-Wall", "-Wextra", "-O3", "-march=native", "-ffast-math"
    $LDFLAGS = "-lm"
    if ($Debug) {
        $CFLAGS = "-Wall", "-Wextra", "-g", "-O0", "-DDEBUG"
    }
    if ($Blas) {
        $CFLAGS += "-DUSE_BLAS", "-DUSE_OPENBLAS"
        $LDFLAGS += "-lopenblas"
    }
    $cmd = "$CC $CFLAGS -o $TARGET $SRCS $LDFLAGS"
} else {
    # MSVC (cl.exe)
    # /O2: Optimization, /W3: Warning level, /MT: Static CRT, /D_CRT_SECURE_NO_WARNINGS, /openmp: Enable OpenMP
    $CFLAGS = "/O2", "/W3", "/MT", "/D_CRT_SECURE_NO_WARNINGS", "/openmp"
    if ($Debug) {
        $CFLAGS = "/Zi", "/Od", "/DDEBUG", "/D_CRT_SECURE_NO_WARNINGS", "/openmp"
    }
    if ($Blas) {
        $CFLAGS += "/DUSE_BLAS", "/DUSE_OPENBLAS"
        Write-Warning "BLAS support with MSVC in this script is experimental (expects openblas.lib in search path)."
        $LDFLAGS = "openblas.lib"
    } else {
        $LDFLAGS = ""
    }
    $cmd = "$CC $CFLAGS $SRCS /Fe$TARGET /link $LDFLAGS"
}

Write-Host "Building $TARGET..."
Write-Host "Running: $cmd"
Invoke-Expression $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful: $TARGET"
} else {
    Write-Host "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
