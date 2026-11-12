# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

param(
    [string]$InstallPath = "C:\openblas"
)

$ErrorActionPreference = "Stop"
# Windows PowerShell 5.1 redraws the progress bar on every read block, which
# costs an order of magnitude of wall-clock on a download this size.
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version Latest

# Bumping OpenBLAS means changing $openBlasVersion and every Hash below.
# Regenerate a hash with: Get-FileHash -Algorithm SHA256 <downloaded zip>
$openBlasVersion = "0.3.30"
$packages = @{
    arm64 = @{
        Asset        = "woa64-dll"
        Hash         = "5BBD8C6CA5A4C415FF58D5E18378B35BC1E81F46E6C385753EFF367FF7474819"
        Subdirectory = "OpenBLAS"
    }
    x64   = @{
        Asset        = "x64"
        Hash         = "8B04387766EFC05C627E26D24797EC0D4ED4C105EC14FA7400AA84A02DB22B66"
        Subdirectory = ""
    }
}

$nativeArchitectures = @(
    $env:PROCESSOR_ARCHITECTURE
    $env:PROCESSOR_ARCHITEW6432
)

if ($nativeArchitectures -contains "ARM64") {
    $architecture = "arm64"
} elseif ($nativeArchitectures -contains "AMD64") {
    $architecture = "x64"
} else {
    throw "Unsupported Windows architecture: $($nativeArchitectures -join ', ')"
}

$package = $packages[$architecture]
$archiveUrl = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$openBlasVersion/OpenBLAS-$openBlasVersion-$($package.Asset).zip"
$expectedHash = $package.Hash
$archiveSubdirectory = $package.Subdirectory

$workingPath = Join-Path ([System.IO.Path]::GetTempPath()) "faiss-openblas-$architecture-$PID"
$archivePath = Join-Path $workingPath "openblas.zip"
$extractPath = Join-Path $workingPath "extract"

New-Item -ItemType Directory -Path $workingPath | Out-Null

try {
    # A blip here aborts before-all, losing every interpreter's wheel in this
    # job. Retrying is safe: the hash check below rejects a partial download.
    # -UseBasicParsing keeps PS 5.1 off the Internet Explorer parsing engine,
    # which is absent on freshly provisioned images.
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $archiveUrl -OutFile $archivePath -UseBasicParsing
            break
        } catch {
            if ($attempt -eq 3) {
                throw
            }
            Start-Sleep -Seconds (5 * $attempt)
        }
    }

    $archiveStream = [System.IO.File]::OpenRead($archivePath)
    try {
        $hasher = [System.Security.Cryptography.SHA256]::Create()
        try {
            $actualHash = [System.BitConverter]::ToString(
                $hasher.ComputeHash($archiveStream)
            ).Replace("-", "")
        } finally {
            $hasher.Dispose()
        }
    } finally {
        $archiveStream.Dispose()
    }

    if ($actualHash -ne $expectedHash) {
        throw "OpenBLAS archive hash mismatch: expected $expectedHash, got $actualHash"
    }

    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath
    $sourcePath = if ($archiveSubdirectory) {
        Join-Path $extractPath $archiveSubdirectory
    } else {
        $extractPath
    }

    foreach ($subdirectory in @("bin", "include", "lib")) {
        if (-not (Test-Path -LiteralPath (Join-Path $sourcePath $subdirectory))) {
            throw "OpenBLAS archive is missing the '$subdirectory' directory"
        }
    }

    if (Test-Path -LiteralPath $InstallPath) {
        Remove-Item -LiteralPath $InstallPath -Recurse -Force
    }
    Move-Item -LiteralPath $sourcePath -Destination $InstallPath
} finally {
    if (Test-Path -LiteralPath $workingPath) {
        Remove-Item -LiteralPath $workingPath -Recurse -Force
    }
}
