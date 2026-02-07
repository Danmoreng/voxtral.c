# download_model.ps1 - Download Voxtral Realtime 4B model from HuggingFace
param(
    [string]$Dir = "voxtral-model"
)

$MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
$FILES = "consolidated.safetensors", "params.json", "tekken.json"
$BASE_URL = "https://huggingface.co/$MODEL_ID/resolve/main"

if (-not (Test-Path $Dir)) {
    New-Item -ItemType Directory -Path $Dir | Out-Null
}

Write-Host "Downloading Voxtral Realtime 4B to ${Dir}/"
Write-Host "Model: ${MODEL_ID}"
Write-Host ""

foreach ($file in $FILES) {
    $dest = Join-Path $Dir $file
    if (Test-Path $dest) {
        Write-Host "  [skip] $file (already exists)"
    } else {
        Write-Host "  [download] $file..."
        $url = "$BASE_URL/$file"
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
        Write-Host "  [done] $file"
    }
}

Write-Host ""
Write-Host "Download complete. Model files in ${Dir}/"
Get-ChildItem $Dir
