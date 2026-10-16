[CmdletBinding()]
param(
    [string[]]$Target = @("all"),
    [switch]$Bootstrap,
    [switch]$SkipSmokeTests,
    [string]$CondaExecutable,
    [string]$Configuration
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# Conda environments must not inherit packages installed in the Windows user
# profile. Otherwise a build can succeed only on the developer's machine.
$env:PYTHONNOUSERSITE = "1"

function Get-RepositoryRoot {
    return (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-CondaExecutable {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "Conda was not found. Pass its full path with -CondaExecutable."
}

function Invoke-CheckedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$Description
    )

    Write-Host "`n==> $Description" -ForegroundColor Cyan
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Assert-PathInsideRepository {
    param(
        [string]$Path,
        [string]$RepositoryRoot
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $rootWithSeparator = $RepositoryRoot.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar

    if (-not $fullPath.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to write outside the repository: $fullPath"
    }

    return $fullPath
}

function Get-CondaEnvironmentNames {
    param([string]$Conda)

    $rawJson = & $Conda env list --json
    if ($LASTEXITCODE -ne 0) {
        throw "Could not list Conda environments."
    }

    $environmentData = $rawJson | ConvertFrom-Json
    return @($environmentData.envs | ForEach-Object { Split-Path -Leaf $_ })
}

function Initialize-BuildEnvironment {
    param(
        [string]$Conda,
        [object]$Definition,
        [string]$RequirementsPath,
        [string[]]$ExistingEnvironmentNames
    )

    $environmentName = [string]$Definition.environment
    if ($environmentName -notin $ExistingEnvironmentNames) {
        Invoke-CheckedCommand -Executable $Conda -Arguments @(
            "create",
            "--yes",
            "--name",
            $environmentName,
            "python=$($Definition.pythonVersion)",
            "pip"
        ) -Description "Create Conda environment '$environmentName'"
    }

    if ($Definition.PSObject.Properties.Name -contains "runtimeRequirements") {
        $runtimeRequirements = (Resolve-Path -LiteralPath (
            Join-Path (Get-RepositoryRoot) ([string]$Definition.runtimeRequirements)
        )).Path
        Invoke-CheckedCommand -Executable $Conda -Arguments @(
            "run", "--no-capture-output", "--name", $environmentName,
            "python", "-m", "pip", "install", "--requirement", $runtimeRequirements
        ) -Description "Download/install pinned runtime dependencies for '$environmentName'"
    }

    $pipArguments = @(
        "run",
        "--no-capture-output",
        "--name",
        $environmentName,
        "python",
        "-m",
        "pip",
        "install",
        "--requirement",
        $RequirementsPath
    )
    if ($Definition.PSObject.Properties.Name -contains "buildToolsDirectory") {
        $toolsDirectory = Join-Path (Get-RepositoryRoot) ([string]$Definition.buildToolsDirectory)
        New-Item -ItemType Directory -Path $toolsDirectory -Force | Out-Null
        $pipArguments += @("--target", $toolsDirectory, "--upgrade")
    }
    Invoke-CheckedCommand -Executable $Conda -Arguments $pipArguments `
        -Description "Install pinned build dependencies for '$environmentName'"
}

function Test-BuildEnvironment {
    param(
        [string]$Conda,
        [object]$Definition
    )

    $requiredImportsJson = ConvertTo-Json -InputObject @($Definition.requiredImports) -Compress
    $pythonCode = @"
import importlib
import json
import sys

expected = tuple(int(value) for value in '$($Definition.pythonVersion)'.split('.'))
actual = sys.version_info[:2]
if actual != expected:
    raise SystemExit(f'Expected Python {expected[0]}.{expected[1]}, found {actual[0]}.{actual[1]}')

for module_name in json.loads(r'''$requiredImportsJson'''):
    importlib.import_module(module_name)

import PyInstaller
print(f'Python {actual[0]}.{actual[1]} / PyInstaller {PyInstaller.__version__}: OK')
"@
    $encodedPythonCode = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($pythonCode)
    )
    $pythonLauncher = "import base64; exec(compile(base64.b64decode('$encodedPythonCode'), '<seg4d-build-check>', 'exec'))"

    Invoke-CheckedCommand -Executable $Conda -Arguments @(
        "run",
        "--no-capture-output",
        "--name",
        [string]$Definition.environment,
        "python",
        "-c",
        $pythonLauncher
    ) -Description "Validate environment '$($Definition.environment)'"
}

function Publish-BuildOutput {
    param(
        [string]$BuiltDirectory,
        [string]$OutputDirectory,
        [string]$RepositoryRoot
    )

    $safeOutput = Assert-PathInsideRepository -Path $OutputDirectory -RepositoryRoot $RepositoryRoot
    $outputParent = Split-Path -Parent $safeOutput
    $backupDirectory = "$safeOutput.__seg4d_backup"

    if (-not (Test-Path -LiteralPath $outputParent)) {
        New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
    }

    if (Test-Path -LiteralPath $backupDirectory) {
        throw "A previous backup still exists: $backupDirectory. Inspect it before rebuilding."
    }

    $hadPreviousOutput = Test-Path -LiteralPath $safeOutput
    if ($hadPreviousOutput) {
        Move-Item -LiteralPath $safeOutput -Destination $backupDirectory
    }

    try {
        Move-Item -LiteralPath $BuiltDirectory -Destination $safeOutput
        if ($hadPreviousOutput) {
            Remove-Item -LiteralPath $backupDirectory -Recurse -Force
        }
    }
    catch {
        if ((Test-Path -LiteralPath $safeOutput) -and $hadPreviousOutput) {
            Remove-Item -LiteralPath $safeOutput -Recurse -Force
        }
        if (Test-Path -LiteralPath $backupDirectory) {
            Move-Item -LiteralPath $backupDirectory -Destination $safeOutput
        }
        throw
    }
}

$repositoryRoot = Get-RepositoryRoot
# Some PyInstaller hooks inspect site.getusersitepackages() even when user-site
# imports are disabled. Keep that lookup inside the repository so a locked or
# redirected Windows profile cannot break otherwise reproducible builds.
$env:PYTHONUSERBASE = Join-Path $repositoryRoot ".build\python-user-base"
New-Item -ItemType Directory -Path $env:PYTHONUSERBASE -Force | Out-Null
if (-not $Configuration) {
    $Configuration = Join-Path $PSScriptRoot "executable-builds.json"
}
$configurationPath = (Resolve-Path -LiteralPath $Configuration).Path
$configurationData = Get-Content -Raw -LiteralPath $configurationPath | ConvertFrom-Json
$conda = Resolve-CondaExecutable -RequestedPath $CondaExecutable

$availableTargets = @($configurationData.targets.PSObject.Properties.Name)
$requestedTargets = @(
    $Target |
        ForEach-Object { $_ -split "," } |
        ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { $_ } |
        Select-Object -Unique
)
if ("all" -in $requestedTargets) {
    $requestedTargets = $availableTargets
}

$unknownTargets = @($requestedTargets | Where-Object { $_ -notin $availableTargets })
if ($unknownTargets.Count -gt 0) {
    throw "Unknown target(s): $($unknownTargets -join ', '). Available targets: $($availableTargets -join ', ')."
}

$existingEnvironmentNames = Get-CondaEnvironmentNames -Conda $conda
$originalPythonPath = $env:PYTHONPATH

foreach ($targetName in $requestedTargets) {
    $definition = $configurationData.targets.$targetName
    $env:PYTHONPATH = $originalPythonPath
    if ($definition.PSObject.Properties.Name -contains "buildToolsDirectory") {
        $buildToolsDirectory = Join-Path $repositoryRoot ([string]$definition.buildToolsDirectory)
        if (Test-Path -LiteralPath $buildToolsDirectory) {
            if ($env:PYTHONPATH) {
                $env:PYTHONPATH = "$buildToolsDirectory;$($env:PYTHONPATH)"
            }
            else {
                $env:PYTHONPATH = $buildToolsDirectory
            }
        }
    }
    $entryScript = (Resolve-Path -LiteralPath (Join-Path $repositoryRoot $definition.entryScript)).Path
    $requirementsPath = (Resolve-Path -LiteralPath (Join-Path $repositoryRoot $definition.requirements)).Path
    $outputDirectory = Join-Path $repositoryRoot $definition.outputDirectory

    if ($Bootstrap) {
        Initialize-BuildEnvironment `
            -Conda $conda `
            -Definition $definition `
            -RequirementsPath $requirementsPath `
            -ExistingEnvironmentNames $existingEnvironmentNames
        $existingEnvironmentNames = Get-CondaEnvironmentNames -Conda $conda
    }

    # Bootstrap may have created this target-local tool directory after the
    # initial PYTHONPATH setup above.
    if ($definition.PSObject.Properties.Name -contains "buildToolsDirectory") {
        $buildToolsDirectory = Join-Path $repositoryRoot ([string]$definition.buildToolsDirectory)
        if ((Test-Path -LiteralPath $buildToolsDirectory) -and
            -not (($env:PYTHONPATH -split ";") -contains $buildToolsDirectory)) {
            if ($env:PYTHONPATH) {
                $env:PYTHONPATH = "$buildToolsDirectory;$($env:PYTHONPATH)"
            }
            else {
                $env:PYTHONPATH = $buildToolsDirectory
            }
        }
    }

    if ([string]$definition.environment -notin $existingEnvironmentNames) {
        throw "Conda environment '$($definition.environment)' is missing. Re-run with -Bootstrap."
    }

    Test-BuildEnvironment -Conda $conda -Definition $definition

    $buildRoot = Assert-PathInsideRepository `
        -Path (Join-Path $repositoryRoot ".build\executables\$targetName") `
        -RepositoryRoot $repositoryRoot
    $distPath = Join-Path $buildRoot "dist"
    $workPath = Join-Path $buildRoot "work"
    $specPath = Join-Path $buildRoot "spec"

    foreach ($directory in @($distPath, $workPath, $specPath)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }

    $pyInstallerArguments = @(
        "run",
        "--no-capture-output",
        "--name",
        [string]$definition.environment,
        "python",
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--name",
        [string]$definition.executableName,
        "--distpath",
        $distPath,
        "--workpath",
        $workPath,
        "--specpath",
        $specPath
    )
    if (-not (
        ($definition.PSObject.Properties.Name -contains "reuseBuildCache") -and
        [bool]$definition.reuseBuildCache
    )) {
        $pyInstallerArguments += "--clean"
    }

    if ($definition.PSObject.Properties.Name -contains "pythonPaths") {
        foreach ($pythonPath in @($definition.pythonPaths)) {
            $pyInstallerArguments += @(
                "--paths",
                (Join-Path $repositoryRoot ([string]$pythonPath))
            )
        }
    }
    foreach ($packageName in @($definition.collectAll)) {
        $pyInstallerArguments += @("--collect-all", [string]$packageName)
    }
    if ($definition.PSObject.Properties.Name -contains "excludeModules") {
        foreach ($moduleName in @($definition.excludeModules)) {
            $pyInstallerArguments += @("--exclude-module", [string]$moduleName)
        }
    }
    $pyInstallerArguments += $entryScript

    Invoke-CheckedCommand `
        -Executable $conda `
        -Arguments $pyInstallerArguments `
        -Description "Build '$targetName' with PyInstaller"

    $builtDirectory = Join-Path $distPath ([string]$definition.executableName)
    if (-not (Test-Path -LiteralPath $builtDirectory)) {
        throw "PyInstaller did not create the expected directory: $builtDirectory"
    }

    $builtExecutablePath = Join-Path $builtDirectory "$($definition.executableName).exe"
    if (-not (Test-Path -LiteralPath $builtExecutablePath)) {
        throw "PyInstaller did not create the expected executable: $builtExecutablePath"
    }

    if (-not $SkipSmokeTests) {
        $smokeTestArguments = @("--help")
        if ($definition.PSObject.Properties.Name -contains "smokeTestArguments") {
            $smokeTestArguments = @(
                $definition.smokeTestArguments | ForEach-Object { [string]$_ }
            )
        }
        Invoke-CheckedCommand `
            -Executable $builtExecutablePath `
            -Arguments $smokeTestArguments `
            -Description "Smoke-test '$targetName'"
    }

    Publish-BuildOutput `
        -BuiltDirectory $builtDirectory `
        -OutputDirectory $outputDirectory `
        -RepositoryRoot $repositoryRoot

    $executablePath = Join-Path $outputDirectory "$($definition.executableName).exe"
    Write-Host "Published: $executablePath" -ForegroundColor Green
}
$env:PYTHONPATH = $originalPythonPath

Write-Host "`nAll requested executables were built successfully." -ForegroundColor Green
