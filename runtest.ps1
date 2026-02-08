# runtest.ps1 - Voxtral regression test for Windows (CUDA focus + real streaming simulation)
# Requires: voxtral.exe, voxtral-model/, samples/jfk.wav, ffmpeg in PATH (for streaming test)

$ErrorActionPreference = "Stop"

# --- Paths (relative to current working directory) ---
$ModelDir   = "voxtral-model"
$InputWav   = "samples\jfk.wav"
$VoxtralExe = ".\voxtral.exe"

# --- Streaming parameters ---
$IntervalSec = "0.1"
$ChunkSec    = 0.1   # how we drip-feed audio (simulate streaming)
$RequireCuda = $true

# Phrases expected in JFK sample
$Phrases = @(
  "And so my fellow Americans",
  "ask not what your country can do for you",
  "ask what you can do for your country"
)

# ---------------- Helpers ----------------
function Fail-Fast([string]$msg) { Write-Host "FAIL: $msg" -ForegroundColor Red; exit 1 }

function Resolve-OrFail([string]$p, [string]$msg) {
  if (-not (Test-Path $p)) { Fail-Fast $msg }
  return $p
}

function Q([string]$s) {
  # Quote for Windows commandline
  return '"' + ($s -replace '"','\"') + '"'
}

function Normalize([string]$s) {
  if ($null -eq $s) { return "" }
  return ($s.ToLowerInvariant() -replace "[\r\n]", " " -replace "[,\.!\?]", "")
}

function Extract-TranscriptLines([string]$combinedText) {
  $lines = $combinedText -split "`r?`n"

  $transcript = @()
  foreach ($l in $lines) {
    $t = $l.Trim()
    if ($t.Length -eq 0) { continue }

    # Filter out known status/debug lines
    if ($t -match "^(Loading|Model loaded|Tokenizer:|Audio:|Encoder:|Decoder:|Stream finished|Listening|WASAPI:)" ) { continue }
    if ($t -match "^\[(cuda|kernels)\]" ) { continue }
    if ($t -match "layer\s+\d+\/\d+\s+loaded" ) { continue }
    if ($t -match "^voxtral\." -or $t -match "^Usage:" -or $t -match "^Required:" -or $t -match "^Options:" ) { continue }

    $transcript += $t
  }

  return $transcript
}

function Has-Cuda([string]$combinedText) {
  return ($combinedText -match "\[kernels\]\s*backend\s*=\s*CUDA") -or ($combinedText -match "\[cuda\]")
}

function Write-Log([string]$name, [string]$text) {
  $path = "runtest-{0}.log" -f $name
  Set-Content -Path $path -Value $text -Encoding UTF8
  return $path
}

function Check-Output([string]$name, [string]$combinedText, [bool]$requireCuda) {
  $ok = $true
  $clean = Normalize $combinedText

  $sawCuda = Has-Cuda $combinedText
  if ($requireCuda -and -not $sawCuda) {
    Write-Host "  FAIL: CUDA backend usage not detected in output!" -ForegroundColor Red
    $ok = $false
  } else {
    Write-Host ("  CUDA: {0}" -f ($(if ($sawCuda) { "YES" } else { "NO" }))) -ForegroundColor DarkGreen
  }

  foreach ($p in $Phrases) {
    $cp = Normalize $p
    if ($clean -notlike "*$cp*") {
      Write-Host "  MISSING: ""$p""" -ForegroundColor Yellow
      $ok = $false
    }
  }

  $transcriptLines = Extract-TranscriptLines $combinedText
  if ($transcriptLines.Count -gt 0) {
    Write-Host "  Transcript:" -ForegroundColor Cyan
    foreach ($tl in $transcriptLines) {
      Write-Host "    $tl"
    }
  } else {
    Write-Host "  Transcript: (none detected)" -ForegroundColor Yellow
  }

  if ($ok) {
    Write-Host "PASS: $name" -ForegroundColor Green
    $global:PASS++
  } else {
    Write-Host "FAIL: $name" -ForegroundColor Red
    $global:FAIL++
  }
  Write-Host ""
}

# Start a process with redirected output/error (optional stdin)
function Start-Proc([string]$exe, [string]$argString, [bool]$redirectStdin) {
  $p = New-Object System.Diagnostics.Process
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $exe
  $psi.Arguments = $argString
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError  = $true
  $psi.RedirectStandardInput  = $redirectStdin
  $p.StartInfo = $psi

  $null = $p.Start()

  # Start async draining immediately to avoid deadlocks
  $outTask = $p.StandardOutput.ReadToEndAsync()
  $errTask = $p.StandardError.ReadToEndAsync()

  return @{ Proc = $p; OutTask = $outTask; ErrTask = $errTask }
}

# ---------------- Tests ----------------
$global:PASS = 0
$global:FAIL = 0

$VoxtralExe = Resolve-OrFail $VoxtralExe "voxtral.exe not found. Run .\build.ps1 first."
$ModelDir   = Resolve-OrFail $ModelDir   "voxtral-model not found. Run .\download_model.ps1 first."
$InputWav   = Resolve-OrFail $InputWav   "samples\jfk.wav not found."

function Run-Batch-CUDA {
  Write-Host "=== Test: batch-cuda ==="

  # keep --debug OPTIONAL; it adds noise. enable if you really want backend line.
  $argString = ('-d {0} -i {1}' -f (Q $ModelDir), (Q $InputWav))
  Write-Host "  [cmd] $VoxtralExe $argString" -ForegroundColor DarkGray

  $h = Start-Proc $VoxtralExe $argString $false
  $h.Proc.WaitForExit()

  $out = $h.OutTask.GetAwaiter().GetResult()
  $err = $h.ErrTask.GetAwaiter().GetResult()
  $combined = ($out + "`n" + $err).Trim()

  $log = Write-Log "batch" $combined
  Write-Host "  Log: $log" -ForegroundColor DarkGray

  Check-Output "batch-cuda" $combined $RequireCuda
}

function Run-Streaming-CUDA {
  Write-Host ("=== Test: streaming-cuda -I {0} (chunked) ===" -f $IntervalSec)

  $ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
  if (-not $ff) {
    Write-Host "  [skip] ffmpeg not found, skipping streaming test." -ForegroundColor Yellow
    $global:PASS++
    Write-Host ""
    return
  }

  # Voxtral: read raw audio from stdin
  $voxArgs = ('-d {0} --stdin -I {1}' -f (Q $ModelDir), $IntervalSec)
  Write-Host "  [vox] $VoxtralExe $voxArgs" -ForegroundColor DarkGray

  $vox = Start-Proc $VoxtralExe $voxArgs $true

  # ffmpeg: decode/resample to raw s16le 16k mono on stdout
  $ffArgs = ('-hide_banner -loglevel error -i {0} -f s16le -ar 16000 -ac 1 -' -f (Q $InputWav))
  Write-Host "  [ff ] $($ff.Source) $ffArgs" -ForegroundColor DarkGray

  $ffp = New-Object System.Diagnostics.Process
  $ffpsi = New-Object System.Diagnostics.ProcessStartInfo
  $ffpsi.FileName = $ff.Source
  $ffpsi.Arguments = $ffArgs
  $ffpsi.UseShellExecute = $false
  $ffpsi.CreateNoWindow = $true
  $ffpsi.RedirectStandardOutput = $true  # binary audio
  $ffpsi.RedirectStandardError  = $true
  $ffp.StartInfo = $ffpsi

  $null = $ffp.Start()
  $ffErrTask = $ffp.StandardError.ReadToEndAsync()

  # Chunk sizes: 16k samples/sec * 2 bytes/sample * ChunkSec
  $bytesPerSec = 16000 * 2
  $chunkBytes = [int]([Math]::Max(320, [Math]::Round($bytesPerSec * $ChunkSec)))  # at least 320 bytes
  if ($chunkBytes % 2 -ne 0) { $chunkBytes++ } # keep sample alignment

  $buf = New-Object byte[] $chunkBytes
  $stdin = $vox.Proc.StandardInput.BaseStream
  $src   = $ffp.StandardOutput.BaseStream

  try {
    while ($true) {
      # FIX: Prüfen, ob Voxtral noch läuft, bevor wir schreiben
      if ($vox.Proc.HasExited) {
        Write-Host "  [info] Voxtral process exited early (EOF or Crash)." -ForegroundColor Yellow
        break
      }

      $n = $src.Read($buf, 0, $buf.Length)
      if ($n -le 0) { break }

      try {
        $stdin.Write($buf, 0, $n)
        $stdin.Flush()
      } catch {
        # FIX: Abfangen, wenn die Pipe während des Schreibens bricht
        Write-Host "  [info] Pipe closed while writing (Target process died)." -ForegroundColor Yellow
        break
      }

      # simulate realtime
      Start-Sleep -Milliseconds ([int]([Math]::Round($ChunkSec * 1000)))
    }
  }
  catch {
     Write-Host "  [err] Loop Exception: $_" -ForegroundColor Red
  }
  finally {
    # signal EOF to voxtral
    try { $vox.Proc.StandardInput.Close() } catch {}
  }

  $ffp.WaitForExit()
  $vox.Proc.WaitForExit()

  $voxOut = $vox.OutTask.GetAwaiter().GetResult()
  $voxErr = $vox.ErrTask.GetAwaiter().GetResult()
  $ffErr  = $ffErrTask.GetAwaiter().GetResult()

  $combined = ($voxOut + "`n" + $voxErr).Trim()
  
  # --- NEW: Show error immediately if Voxtral crashed ---
  if ($vox.Proc.ExitCode -ne 0) {
      Write-Host "`n[ERROR] Voxtral Crashed (Exit Code: $($vox.Proc.ExitCode))" -ForegroundColor Red
      Write-Host "--- Last Output ---" -ForegroundColor Gray
      $combined | Select-Object -Last 10 | Write-Host
      Write-Host "-------------------`n" -ForegroundColor Gray
  }
  # -------------------------------------------------------------

  if ($ffp.ExitCode -ne 0 -and $ffErr.Trim().Length -gt 0) {
    $combined += "`n`n[ffmpeg exit=$($ffp.ExitCode)]`n$ffErr"
  }

  $log = Write-Log "streaming" $combined
  Write-Host "  Log: $log" -ForegroundColor DarkGray

  Check-Output ("streaming-cuda -I {0}" -f $IntervalSec) $combined $RequireCuda
}

Run-Batch-CUDA
Run-Streaming-CUDA

Write-Host "=== Results: $global:PASS passed, $global:FAIL failed ==="
if ($global:FAIL -eq 0) { exit 0 } else { exit 1 }
