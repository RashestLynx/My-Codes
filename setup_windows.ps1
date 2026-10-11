# DVD Upscaler setup: puts everything dvd_upscale.py needs into this folder, so the folder
# is the whole program (copy or move it anywhere). Run setup.bat (double-click), not this file.
# Run it again any time: what is already here is kept, what is missing is fetched.
#
# What ends up in this folder:
#   python\          a private Python 3.12 (nothing is installed on the computer) with the
#                    face restoration packages (onnxruntime-directml: any GPU, OpenCV, numpy)
#   ffmpeg.exe, ffprobe.exe            (BtbN's Windows build: has vid.stab for --stabilize)
#   realesrgan-ncnn-vulkan.exe, vcomp140*.dll, models\      (the upscaler and its models)
#   models\realesrgan-x2plus.*         (from realesrgan-x2plus.zip, if it is in this folder)
#   face_models\     the face detector and GFPGAN 1.4
#   Movies\          put your movies here (or drag them onto Upscale.bat)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'    # (downloads are many times slower with the progress bar)
try {   # (older Windows PowerShell may still default to TLS 1.0, which these sites refuse)
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

$Here = $PSScriptRoot
$Tmp = Join-Path $Here '_setup_downloads'
$Models = Join-Path $Here 'models'
$Faces = Join-Path $Here 'face_models'
$PyDir = Join-Path $Here 'python'
$PyExe = Join-Path $PyDir 'python.exe'
$Problems = New-Object System.Collections.ArrayList

$URL_PYTHON = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip'
$URL_GETPIP = 'https://bootstrap.pypa.io/get-pip.py'
$URL_FFMPEG = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
$URL_ESRGAN = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip'
$URL_YUNET = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
$URL_GFPGAN = 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx'

function Step($text) {
    Write-Host ''
    Write-Host "== $text" -ForegroundColor Cyan
}

function Have($path, $minBytes) {
    # a file that is there and not cut short (a download that broke off)
    return (Test-Path -LiteralPath $path -PathType Leaf) -and ((Get-Item -LiteralPath $path).Length -ge $minBytes)
}

function Get-File($url, $dest, $minBytes) {
    # downloaded under a temporary name, renamed only when complete; three tries
    if (Have $dest $minBytes) { return }
    $name = Split-Path -Leaf $dest
    $part = "$dest.part"
    for ($try = 1; $try -le 3; $try++) {
        Write-Host "  downloading $name ..."
        try {
            Invoke-WebRequest -Uri $url -OutFile $part -UseBasicParsing
            $size = (Get-Item -LiteralPath $part).Length
            if ($size -lt $minBytes) { throw "it came out as only $size bytes" }
            Move-Item -LiteralPath $part -Destination $dest -Force
            return
        } catch {
            Write-Host "  that didn't work: $($_.Exception.Message)" -ForegroundColor Yellow
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
            if ($try -eq 3) { throw "Couldn't download $name from $url (check the internet connection, then run setup.bat again)" }
            Start-Sleep -Seconds (5 * $try)
        }
    }
}

function Expand-Fresh($zip, $name) {
    # a zip unpacked into an empty folder of its own under _setup_downloads
    $dir = Join-Path $Tmp $name
    if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force }
    Expand-Archive -LiteralPath $zip -DestinationPath $dir -Force
    return $dir
}

function Run($exe, $argv) {
    # a program run with its messages shown; returns its exit code. (Windows PowerShell 5.1 turns
    # a program's error-stream lines into script errors when they are redirected, which
    # "Stop" would make fatal: so not "Stop" while a program runs)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $exe @argv | Out-Host; return $LASTEXITCODE } finally { $ErrorActionPreference = $old }
}

function Run-Quiet($exe, $argv) {
    # the same with nothing shown: (exit code, everything it printed)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $exe @argv 2>&1 | ForEach-Object { "$_" } | Out-String
        return @($LASTEXITCODE, $out)
    } finally { $ErrorActionPreference = $old }
}

function Find-In($dir, $file) {
    $f = Get-ChildItem -LiteralPath $dir -Recurse -Filter $file | Select-Object -First 1
    if (-not $f) { throw "$file wasn't in the download (has the download page changed?)" }
    return $f
}

try {
Write-Host "Setting up the DVD Upscaler in: $Here"
New-Item -ItemType Directory -Force -Path $Tmp, $Models, $Faces, (Join-Path $Here 'Movies') | Out-Null

# ---- the upscaler and ffmpeg (needed for everything) ------------------------------------------
Step 'ffmpeg'
if ((Have (Join-Path $Here 'ffmpeg.exe') 1MB) -and (Have (Join-Path $Here 'ffprobe.exe') 1MB)) {
    Write-Host '  already here'
} else {
    $zip = Join-Path $Tmp 'ffmpeg.zip'
    Get-File $URL_FFMPEG $zip 20MB
    Write-Host '  unpacking ...'
    $dir = Expand-Fresh $zip 'ffmpeg'
    foreach ($n in 'ffmpeg.exe', 'ffprobe.exe') {
        Copy-Item -LiteralPath (Find-In $dir $n).FullName -Destination (Join-Path $Here $n) -Force
    }
    Write-Host '  done'
}

Step 'Real-ESRGAN upscaler'
if ((Have (Join-Path $Here 'realesrgan-ncnn-vulkan.exe') 1MB) -and (Have (Join-Path $Models 'realesrgan-x4plus.bin') 1MB)) {
    Write-Host '  already here'
} else {
    $zip = Join-Path $Tmp 'realesrgan.zip'
    Get-File $URL_ESRGAN $zip 10MB
    Write-Host '  unpacking ...'
    $dir = Expand-Fresh $zip 'realesrgan'
    $root = (Find-In $dir 'realesrgan-ncnn-vulkan.exe').DirectoryName
    foreach ($n in 'realesrgan-ncnn-vulkan.exe', 'vcomp140.dll', 'vcomp140d.dll') {
        $f = Join-Path $root $n
        if (Test-Path -LiteralPath $f) { Copy-Item -LiteralPath $f -Destination (Join-Path $Here $n) -Force }
    }
    # its models next to any already in models\ (realesrgan-x2plus, say, stays)
    Get-ChildItem -LiteralPath (Join-Path $root 'models') -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Models $_.Name) -Force
    }
    Write-Host '  done'
}

Step 'realesrgan-x2plus model (live action and movie tapes)'
$zip = Join-Path $Here 'realesrgan-x2plus.zip'
$installed = (Have (Join-Path $Models 'realesrgan-x2plus.bin') 1MB) -and (Have (Join-Path $Models 'realesrgan-x2plus.param') 1KB)
if (Test-Path -LiteralPath $zip) {
    # both files from the zip whenever either differs (a newer zip, or a pair that doesn't belong
    # together): they must come from the same download
    try {
        $dir = Expand-Fresh $zip 'x2plus'
        $same = $true
        foreach ($n in 'realesrgan-x2plus.param', 'realesrgan-x2plus.bin') {
            $dst = Join-Path $Models $n
            if (-not (Test-Path -LiteralPath $dst) -or
                (Get-FileHash -LiteralPath (Find-In $dir $n).FullName).Hash -ne (Get-FileHash -LiteralPath $dst).Hash) {
                $same = $false
            }
        }
        if ($same) {
            Write-Host '  already here'
        } else {
            foreach ($n in 'realesrgan-x2plus.param', 'realesrgan-x2plus.bin') {
                Copy-Item -LiteralPath (Find-In $dir $n).FullName -Destination (Join-Path $Models $n) -Force
            }
            Write-Host '  done (from realesrgan-x2plus.zip)'
        }
    } catch {
        # (a damaged zip: the rest of the setup still runs)
        Write-Host "  realesrgan-x2plus.zip can't be read ($($_.Exception.Message))" -ForegroundColor Yellow
        if ($installed) { Write-Host '  keeping the installed model' -ForegroundColor Yellow }
        [void]$Problems.Add('realesrgan-x2plus.zip is damaged: download it again')
    }
} elseif ($installed) {
    Write-Host '  already here'
} else {
    Write-Host '  realesrgan-x2plus.zip is not in this folder: live action uses the slower' -ForegroundColor Yellow
    Write-Host '  realesrgan-x4plus until it is (put the zip here and run setup.bat again).' -ForegroundColor Yellow
    [void]$Problems.Add('realesrgan-x2plus.zip missing (live action works, but slower, with realesrgan-x4plus)')
}

# ---- Python and the face restoration ----------------------------------------------------------
Step 'Python (a private copy, in the python folder)'
if (Test-Path -LiteralPath $PyExe) {
    Write-Host '  already here'
} else {
    $zip = Join-Path $Tmp 'python.zip'
    Get-File $URL_PYTHON $zip 5MB
    Write-Host '  unpacking ...'
    New-Item -ItemType Directory -Force -Path $PyDir | Out-Null
    Expand-Archive -LiteralPath $zip -DestinationPath $PyDir -Force
    Write-Host '  done'
}
# the embeddable Python ignores installed packages until "import site" is switched on in its
# ._pth file (python312._pth)
$pth = Get-ChildItem -LiteralPath $PyDir -Filter 'python*._pth' | Select-Object -First 1
if ($pth) {
    $lines = Get-Content -LiteralPath $pth.FullName
    if ($lines -contains '#import site') {
        $lines = $lines | ForEach-Object { if ($_ -eq '#import site') { 'import site' } else { $_ } }
        Set-Content -LiteralPath $pth.FullName -Value $lines -Encoding Ascii
    }
}

Step 'pip (installs Python packages into the python folder)'
$r = Run-Quiet $PyExe @('-m', 'pip', '--version')
if ($r[0] -eq 0) {
    Write-Host '  already here'
} else {
    $getpip = Join-Path $Tmp 'get-pip.py'
    Get-File $URL_GETPIP $getpip 100KB
    if ((Run $PyExe @($getpip, '--no-warn-script-location')) -ne 0) { throw "pip couldn't be set up (see the messages above)" }
}

Step 'Face restoration packages (onnxruntime-directml, OpenCV, numpy)'
# onnxruntime-directml runs on any GPU (NVIDIA, AMD, Intel) through DirectX 12; only one
# onnxruntime package may be installed, so others are removed first (this python folder only)
[void](Run-Quiet $PyExe @('-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu'))
if ((Run $PyExe @('-m', 'pip', 'install', '--upgrade', '--no-warn-script-location', 'onnxruntime-directml', 'opencv-python-headless', 'numpy')) -ne 0) {
    throw "The face restoration packages couldn't be installed (see the messages above)"
}

Step 'GPU engine for live action and CGI (the current ncnn)'
# realesrgan-ncnn-vulkan.exe's own engine is from 2022: on NVIDIA drivers from 570 on, the big
# live-action model makes the GPU reset now and then. dvd_upscale.py uses this one for it when
# it is installed. --no-deps: it needs only numpy (installed above); its other listed packages
# include opencv-python, which would clash with opencv-python-headless
if ((Run $PyExe @('-m', 'pip', 'install', '--upgrade', '--no-deps', '--no-warn-script-location', 'ncnn==1.0.20260526')) -ne 0) {
    Write-Host '  not installed: live action and CGI use the upscaler''s own (older) engine' -ForegroundColor Yellow
    [void]$Problems.Add('the current ncnn (GPU engine for live action and CGI) is not installed')
} else {
    Write-Host '  done'
}

Step 'Face restoration models'
Get-File $URL_YUNET (Join-Path $Faces 'face_detection_yunet_2023mar.onnx') 100KB
Get-File $URL_GFPGAN (Join-Path $Faces 'gfpgan_1.4.onnx') 300MB
Write-Host '  done'

# ---- check -----------------------------------------------------------------------------------
Step 'Checking that the face restoration runs'
$r = Run-Quiet $PyExe @((Join-Path $Here 'dvd_upscale.py'), '--faces-worker', '--check', '--models', $Faces)
$out = $r[1]
$provider = [regex]::Match($out, 'PROVIDER (\S+)').Groups[1].Value
if ($r[0] -eq 0 -and $provider) {
    if ($provider -eq 'CPUExecutionProvider') {
        Write-Host "  works, but on the processor ($provider): very slow. Is the graphics driver up to date?" -ForegroundColor Yellow
        [void]$Problems.Add('face restoration runs on the processor, not the GPU (update the graphics driver)')
    } else {
        Write-Host "  works, on the GPU ($provider)" -ForegroundColor Green
    }
} else {
    Write-Host $out
    Write-Host "  the face restoration check failed (see above)" -ForegroundColor Yellow
    [void]$Problems.Add('the face restoration check failed: upscaling works, --faces does not yet')
}

Remove-Item -LiteralPath $Tmp -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ''
if ($Problems.Count) {
    Write-Host 'Set up, with these to look at:' -ForegroundColor Yellow
    foreach ($p in $Problems) { Write-Host "  - $p" -ForegroundColor Yellow }
} else {
    Write-Host 'All set.' -ForegroundColor Green
}
Write-Host 'Put movies in the Movies folder and double-click Upscale.bat, or drag movies (or a folder)'
Write-Host 'onto Upscale.bat. Preview.bat does the same for a 60-second test.'
} catch {
    Write-Host ''
    Write-Host "Setup stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Fix that and run setup.bat again: what is already done is kept.'
    exit 1
}
