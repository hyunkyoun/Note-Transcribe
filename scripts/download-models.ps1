<#
.SYNOPSIS
  Downloads the Whisper, VAD and speaker-embedding models into .\models

.EXAMPLE
  .\scripts\download-models.ps1
  .\scripts\download-models.ps1 -Model large-v3-turbo-q5_0 -Fallback base.en
#>
param(
  [string]$Model = "large-v3-turbo",
  [string]$Fallback = "base",
  [string]$Dest = ""
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar

if (-not $Dest) { $Dest = Join-Path $PSScriptRoot "..\models" }
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$Dest = (Resolve-Path $Dest).Path

$hf = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
$files = @(
  @{ Name = "ggml-$Model.bin";    Url = "$hf/ggml-$Model.bin" },
  @{ Name = "ggml-$Fallback.bin"; Url = "$hf/ggml-$Fallback.bin" },
  @{ Name = "silero_vad.onnx";    Url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" },
  @{ Name = "wespeaker_en_voxceleb_resnet34_LM.onnx"; Url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx" }
)

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
foreach ($f in $files) {
  $out = Join-Path $Dest $f.Name
  if (Test-Path $out) { Write-Host "already present: $($f.Name)"; continue }
  Write-Host "downloading $($f.Name) ..."
  $part = "$out.part"
  if ($curl) {
    & curl.exe -L --fail --retry 3 -o $part $f.Url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $($f.Url)" }
  } else {
    Invoke-WebRequest -Uri $f.Url -OutFile $part -UseBasicParsing
  }
  Move-Item -Force $part $out
}
Write-Host "`nModels are in $Dest"
Get-ChildItem $Dest -File | Where-Object { $_.Name -ne "README.md" } | Format-Table Name, @{n="MB";e={[math]::Round($_.Length/1MB,1)}}
