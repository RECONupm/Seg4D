[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CloudCompareDirectory,
    [string]$Seg4DRuntimeDirectory,
    [string]$ExpectedPython = "3.13",
    [string[]]$RequiredModules = @(
        "PIL.Image",
        "fitz",
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "matplotlib",
        "cv2",
        "yaml"
    )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$cloudCompareRoot = (Resolve-Path -LiteralPath $CloudCompareDirectory).Path
$python = Join-Path $cloudCompareRoot "plugins\Python\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "CloudCompare's embedded Python was not found: $python"
}
if (-not $Seg4DRuntimeDirectory) {
    $Seg4DRuntimeDirectory = Join-Path $cloudCompareRoot "plugins\Python\Lib\site-packages\seg4d_runtime"
}
$seg4dRuntime = (Resolve-Path -LiteralPath $Seg4DRuntimeDirectory).Path

$normalizedCloudCompareRoot = $cloudCompareRoot.TrimEnd("\") + "\"
$uninstallRoots = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"
)
$cloudCompareRegistration = $uninstallRoots |
    ForEach-Object { Get-ItemProperty -Path $_ -ErrorAction SilentlyContinue } |
    Where-Object {
        $displayName = $_.PSObject.Properties["DisplayName"]
        $installLocation = $_.PSObject.Properties["InstallLocation"]
        $displayName -and
        $installLocation -and
        $displayName.Value -like "CloudCompare*" -and
        $installLocation.Value -and
        $installLocation.Value.TrimEnd("\") + "\" -eq $normalizedCloudCompareRoot
    } |
    Select-Object -First 1

$runtimeJson = & $python -c @"
import json
import sys
import sysconfig

print(json.dumps({
    'executable': sys.executable,
    'version': sys.version,
    'major_minor': f'{sys.version_info.major}.{sys.version_info.minor}',
    'cache_tag': sys.implementation.cache_tag,
    'extension_suffix': sysconfig.get_config_var('EXT_SUFFIX'),
}))
"@
if ($LASTEXITCODE -ne 0) {
    throw "Could not query CloudCompare's embedded Python."
}

$runtime = $runtimeJson | ConvertFrom-Json
Write-Host "CloudCompare directory : $cloudCompareRoot"
if ($cloudCompareRegistration) {
    $displayVersion = $cloudCompareRegistration.PSObject.Properties["DisplayVersion"]
    if ($displayVersion) {
        Write-Host "CloudCompare version   : $($displayVersion.Value)"
    }
}
Write-Host "Python executable      : $($runtime.executable)"
Write-Host "Python version         : $($runtime.version)"
Write-Host "Extension ABI          : $($runtime.extension_suffix)"
Write-Host "Seg4D private runtime  : $seg4dRuntime"

$failedChecks = [System.Collections.Generic.List[string]]::new()
if ($runtime.major_minor -ne $ExpectedPython) {
    $failedChecks.Add(
        "Seg4D expects Python $ExpectedPython, but this PythonRuntime embeds Python $($runtime.major_minor)."
    )
}

foreach ($moduleName in $RequiredModules) {
    $moduleCode = "import importlib, pathlib, sys; root = pathlib.Path(r'$seg4dRuntime').resolve(); sys.path.insert(0, str(root)); module = importlib.import_module('$moduleName'); origin = pathlib.Path(module.__file__).resolve(); print('$($moduleName): ' + ('OK' if origin.is_relative_to(root) else 'WRONG RUNTIME') + ' (' + str(origin) + ')'); raise SystemExit(0 if origin.is_relative_to(root) else 2)"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $moduleOutput = & $python -c $moduleCode 2>&1
    $moduleExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($moduleExitCode -eq 0) {
        Write-Host ($moduleOutput | Out-String).Trim() -ForegroundColor Green
    }
    else {
        Write-Host "$($moduleName): FAILED" -ForegroundColor Red
        Write-Host ($moduleOutput | Out-String).Trim()
        $failedChecks.Add("$moduleName could not be imported.")
    }
}

if ($failedChecks.Count -gt 0) {
    Write-Host "`nCompatibility problems:" -ForegroundColor Red
    foreach ($failure in $failedChecks) {
        Write-Host " - $failure"
    }
    exit 1
}

Write-Host "`nCloudCompare's Python runtime is compatible with Seg4D." -ForegroundColor Green
