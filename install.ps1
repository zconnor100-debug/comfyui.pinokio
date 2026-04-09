# AI Video Generator - Windows Auto-Installer
# Usage: irm https://raw.githubusercontent.com/zconnor100-debug/comfyui.pinokio/main/install.ps1 | iex

$ErrorActionPreference = "Stop"
$repo    = "https://github.com/zconnor100-debug/comfyui.pinokio"
$branch  = "main"
$appName = "comfyui.pinokio"

function Write-Step  { param($msg) Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-OK    { param($msg) Write-Host "   OK  $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "   !!  $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "   XX  $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "  AI Video Generator Installer" -ForegroundColor Magenta
Write-Host "  ==============================" -ForegroundColor DarkGray
Write-Host ""

# ── 1. Check Git ─────────────────────────────────────────────────────────────
Write-Step "Checking for Git..."
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Warn "Git not found. Attempting install via winget..."
    try {
        winget install --id Git.Git -e --silent --accept-source-agreements --accept-package-agreements
        # Reload PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH","User")
        Write-OK "Git installed."
    } catch {
        Write-Fail "Could not auto-install Git."
        Write-Host "   Please install Git from https://git-scm.com and re-run this script." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-OK "Git found: $(git --version)"
}

# ── 2. Find Pinokio ───────────────────────────────────────────────────────────
Write-Step "Locating Pinokio..."

$candidates = @(
    "$env:USERPROFILE\pinokio",
    "$env:USERPROFILE\Pinokio",
    "C:\pinokio",
    "$env:LOCALAPPDATA\pinokio"
)

$pinokioRoot = $null
foreach ($c in $candidates) {
    if (Test-Path "$c\api") { $pinokioRoot = $c; break }
}

if (-not $pinokioRoot) {
    # Pinokio not installed — offer to download
    Write-Warn "Pinokio not found."
    $install = Read-Host "   Download and install Pinokio now? (y/n)"
    if ($install -match "^y") {
        Write-Step "Downloading Pinokio installer..."
        $dlUrl  = "https://github.com/pinokiocomputer/pinokio/releases/latest/download/Pinokio_Setup_win.exe"
        $dlPath = "$env:TEMP\Pinokio_Setup.exe"
        try {
            Invoke-WebRequest -Uri $dlUrl -OutFile $dlPath -UseBasicParsing
            Write-OK "Downloaded. Launching installer — complete it, then press Enter here."
            Start-Process $dlPath
            Read-Host "   Press Enter once Pinokio is installed"
            # Re-check
            foreach ($c in $candidates) {
                if (Test-Path "$c\api") { $pinokioRoot = $c; break }
            }
        } catch {
            Write-Fail "Download failed: $_"
        }
    }
    if (-not $pinokioRoot) {
        Write-Fail "Pinokio API directory not found. Install Pinokio from https://pinokio.computer first."
        exit 1
    }
}

$apiDir = "$pinokioRoot\api"
Write-OK "Pinokio found at: $pinokioRoot"

# ── 3. Clone / update the app ─────────────────────────────────────────────────
Write-Step "Installing AI Video Generator..."
$appDir = "$apiDir\$appName"

if (Test-Path $appDir) {
    Write-Warn "App folder already exists — pulling latest changes..."
    git -C $appDir fetch origin
    git -C $appDir checkout $branch
    git -C $appDir pull origin $branch
    Write-OK "Updated to latest version."
} else {
    git clone -b $branch $repo $appDir
    Write-OK "Cloned to: $appDir"
}

# ── 4. Launch Pinokio ─────────────────────────────────────────────────────────
Write-Step "Opening Pinokio..."

$exeCandidates = @(
    "$env:LOCALAPPDATA\Programs\Pinokio\Pinokio.exe",
    "$env:LOCALAPPDATA\pinokio\Pinokio.exe",
    "$pinokioRoot\Pinokio.exe"
)
$pinokioExe = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $pinokioExe) {
    # Try searching AppData
    $pinokioExe = Get-ChildItem "$env:LOCALAPPDATA" -Recurse -Filter "Pinokio.exe" -ErrorAction SilentlyContinue |
                  Select-Object -First 1 -ExpandProperty FullName
}

if ($pinokioExe) {
    Start-Process $pinokioExe
    Write-OK "Pinokio launched."
} else {
    Write-Warn "Could not auto-launch Pinokio — open it manually."
}

# ── 5. Done ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host "   1. Open Pinokio (it may already be launching)" -ForegroundColor Gray
Write-Host "   2. Find 'ComfyUI' in your app library" -ForegroundColor Gray
Write-Host "   3. Click Install — downloads models, sets up Python env" -ForegroundColor Gray
Write-Host "   4. Once installed: Launch ComfyUI first, then" -ForegroundColor Gray
Write-Host "      use 'Launch Prompt Video Generator' or 'Launch AI Video Narrator'" -ForegroundColor Gray
Write-Host ""
Write-Host "  For AI scripting (Ollama): https://ollama.com" -ForegroundColor DarkGray
Write-Host ""
