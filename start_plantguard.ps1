# PlantGuard AI Windows setup and launcher
# This script installs Python 3.12 when needed, creates an isolated
# environment, installs the project requirements, and starts Streamlit.

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements.txt"
$App = Join-Path $ProjectRoot "plantguard\app.py"

function Test-CompatiblePython([string]$Executable) {
    if (-not (Test-Path $Executable) -and $Executable -notin @("py", "python", "python3")) {
        return $false
    }

    try {
        & $Executable -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Find-CompatiblePython {
    foreach ($Candidate in @("py", "python", "python3")) {
        if (Get-Command $Candidate -ErrorAction SilentlyContinue) {
            if (Test-CompatiblePython $Candidate) {
                return $Candidate
            }
        }
    }

    $Installed = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    foreach ($Candidate in $Installed) {
        if (Test-CompatiblePython $Candidate.FullName) {
            return $Candidate.FullName
        }
    }

    return $null
}

Write-Host ""
Write-Host "PlantGuard AI setup" -ForegroundColor Cyan
Write-Host "-------------------"

$Python = Find-CompatiblePython

if (-not $Python) {
    Write-Host "Python 3.10 or newer was not found." -ForegroundColor Yellow

    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Windows Package Manager (winget) is unavailable. Install Python 3.12 from https://www.python.org/downloads/ and run this file again."
    }

    Write-Host "Installing Python 3.12 for your Windows account..."
    winget install --exact --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements

    if ($LASTEXITCODE -ne 0) {
        throw "Python installation did not complete successfully."
    }

    $Python = Find-CompatiblePython
    if (-not $Python) {
        throw "Python was installed but could not be located. Close this window, open a new terminal, and run start_plantguard.bat again."
    }
}

$Version = & $Python --version
Write-Host "Using $Version" -ForegroundColor Green

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating the PlantGuard virtual environment..."
    & $Python -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "The virtual environment could not be created."
    }
}

Write-Host "Installing/updating required packages..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip could not be updated."
}

& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "The PlantGuard packages could not be installed."
}

$LocalUrl = "http://localhost:8501"
Write-Host ""
Write-Host "Starting PlantGuard AI at $LocalUrl" -ForegroundColor Green
Write-Host "Keep this window open while using the dashboard."
Write-Host "Press Ctrl+C here when you are finished."
Write-Host ""

# Open the browser shortly after Streamlit begins starting.
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    "-NoProfile",
    "-Command",
    "Start-Sleep -Seconds 4; Start-Process '$LocalUrl'"
)

Push-Location (Join-Path $ProjectRoot "plantguard")
try {
    & $VenvPython -m streamlit run $App --server.address localhost --server.port 8501
}
finally {
    Pop-Location
}
