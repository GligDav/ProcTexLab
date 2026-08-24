[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repositoryRoot = $PSScriptRoot
$venvPath = Join-Path $repositoryRoot ".venv"

# Prefer the Windows Python launcher, but also support installations that only
# expose python.exe on PATH.
$pythonExecutable = $null
$pythonArguments = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExecutable = "py"
    $pythonArguments = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExecutable = "python"
}

if ($null -eq $pythonExecutable) {
    Write-Error "Python is not installed or is not available on PATH. Install Python 3.14, then run this script again." -ErrorAction Continue
    exit 1
}

Push-Location $repositoryRoot
try {
    if (-not (Test-Path -LiteralPath $venvPath -PathType Container)) {
        Write-Host "Creating virtual environment in .venv..."
        & $pythonExecutable @pythonArguments -m venv $venvPath
        if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }

        $gpuAnswer = Read-Host "Install the optional GPU dependency? [y/N]"
        $package = if ($gpuAnswer -match '^(?i:y|yes)$') { ".[gpu]" } else { "." }

        Write-Host "Installing project dependencies..."
        & (Join-Path $venvPath "Scripts\python.exe") -m pip install -e $package
        if ($LASTEXITCODE -ne 0) { throw "Failed to install project dependencies." }
    }

    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (-not (Test-Path -LiteralPath $activateScript -PathType Leaf)) {
        throw ".venv exists but does not contain a usable Windows virtual environment. Remove .venv and run this script again."
    }

    . $activateScript
    try {
        python -m gui.test_app
        $appExitCode = $LASTEXITCODE
    } finally {
        deactivate
    }

    exit $appExitCode
} catch {
    Write-Error $_
    exit 1
} finally {
    Pop-Location
}
