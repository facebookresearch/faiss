param(
    [string]$InstallPath = "C:\openblas"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$nativeArchitectures = @(
    $env:PROCESSOR_ARCHITECTURE
    $env:PROCESSOR_ARCHITEW6432
)

if ($nativeArchitectures -contains "ARM64") {
    $architecture = "arm64"
    $archiveUrl = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30-woa64-dll.zip"
    $expectedHash = "5BBD8C6CA5A4C415FF58D5E18378B35BC1E81F46E6C385753EFF367FF7474819"
    $archiveSubdirectory = "OpenBLAS"
} elseif ($nativeArchitectures -contains "AMD64") {
    $architecture = "x64"
    $archiveUrl = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30-x64.zip"
    $expectedHash = "8B04387766EFC05C627E26D24797EC0D4ED4C105EC14FA7400AA84A02DB22B66"
    $archiveSubdirectory = ""
} else {
    throw "Unsupported Windows architecture: $($nativeArchitectures -join ', ')"
}

$workingPath = Join-Path ([System.IO.Path]::GetTempPath()) "faiss-openblas-$architecture-$PID"
$archivePath = Join-Path $workingPath "openblas.zip"
$extractPath = Join-Path $workingPath "extract"

New-Item -ItemType Directory -Path $workingPath | Out-Null

try {
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archivePath

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
