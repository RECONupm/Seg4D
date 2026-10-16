#define appName "seg4d"
#define appVersion "1.0"
#define appVerName "SEG4D_Installer"
#define appPublisher ""
#define appUrl ""
#define appExeName "seg4d.py"


[Setup]
AppName={#appName}
AppVersion={#appVersion}
AppVerName={#appVerName}
AppPublisher={#appPublisher}
DefaultDirName={commonpf}\CloudCompare
AllowNoIcons=true
Compression=lzma2/max
SolidCompression=true
VersionInfoVersion={#appVersion}
VersionInfoCompany={#appPublisher}
VersionInfoProductName={#appName}
VersionInfoProductVersion={#appVersion}
DirExistsWarning=no
AppPublisherURL={#appUrl}
UninstallFilesDir={app}
OutputBaseFilename={#appVerName} Setup
OutputDir=installer
DefaultGroupName={#appName}
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
DisableStartupPrompt=true
DiskSpanning=no
SetupIconFile=logo.ico
; It is for modyfing any environment variable
;ChangesEnvironment=true

[Files]
;Files to be copied
Source: "seg4d.py"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}"; Tasks: ; Languages: ; Flags: replacesameversion
Source: "assets\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\assets"; Flags: recursesubdirs
Source: "licenses\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\licenses"; Flags: recursesubdirs
Source: "configs\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\configs"; Flags: recursesubdirs
Source: "geometric-based_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\geometric-based_methods"; Flags: recursesubdirs; Excludes: "__pycache__\*,*\__pycache__\*,*.pyc"
Source: "main_module\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\main_module"; Flags: recursesubdirs; Excludes: "__pycache__\*,*\__pycache__\*,*.pyc"
Source: "other_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\other_methods"; Flags: recursesubdirs; Excludes: "__pycache__\*,*\__pycache__\*,*.pyc"
Source: "pre-requisites\cloudcompare-python313.txt"; Flags: dontcopy
Source: "scripts\validate_cloudcompare_runtime.py"; Flags: dontcopy
#ifdef IncludeExternalPayloads
Source: "point_clouds_examples\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\point_clouds_examples"; Flags: recursesubdirs
#endif
Source: "radiometric-based_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\radiometric-based_methods"; Flags: recursesubdirs; Excludes: "__pycache__\*,*\__pycache__\*,*.pyc"
;In the following line we exclude the folder point_transformer because we copy as a 7zip and then unzip it because the paths are too large
Source: "segmentation_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\segmentation_methods"; Flags: recursesubdirs; Excludes: "point_transformer\*,point_transformer.7z,__pycache__\*,*\__pycache__\*,*.pyc"
#ifdef IncludeLegacyPointTransformer
Source: "segmentation_methods\point_transformer.7z"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\segmentation_methods"; Flags: deleteafterinstall 
Source: "{#LegacyExtractor}"; DestDir: "{tmp}\seg4d-7zip"; DestName: "extractor.exe"; Flags: deleteafterinstall
Source: "{#LegacyExtractorLicense}"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\licenses\7zip"
#ifdef LegacyExtractorDLL
Source: "{#LegacyExtractorDLL}"; DestDir: "{tmp}\seg4d-7zip"; Flags: deleteafterinstall
#endif
#endif
Source: "seg4d_plugin\*"; DestDir: "{app}\plugins-python\seg4d_plugin"; Flags: recursesubdirs; Excludes: "__pycache__\*,*\__pycache__\*,*.pyc"

[InstallDelete]
Type: filesandordirs; Name: "{app}\plugins\Python\Lib\site-packages\seg4d\geometric-based_methods\__pycache__"
Type: filesandordirs; Name: "{app}\plugins\Python\Lib\site-packages\seg4d\main_module\__pycache__"
Type: filesandordirs; Name: "{app}\plugins\Python\Lib\site-packages\seg4d\other_methods\__pycache__"
Type: filesandordirs; Name: "{app}\plugins\Python\Lib\site-packages\seg4d\radiometric-based_methods\__pycache__"
Type: filesandordirs; Name: "{app}\plugins\Python\Lib\site-packages\seg4d\segmentation_methods\__pycache__"
Type: filesandordirs; Name: "{app}\plugins-python\seg4d_plugin\__pycache__"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\plugins\Python\Lib\site-packages\seg4d_runtime"
Type: filesandordirs; Name: "{app}\seg4d-pip-temp"

#ifdef IncludeExternalPayloads
[Files]
;Additional libraries for the proper instalation of the program
Source: "modified_libraries\tkPDFViewer\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\tkPDFViewer"; Tasks: ; Languages: ; Flags: replacesameversion
Source: "modified_libraries\tkPDFViewer-0.1.dist-info\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\tkPDFViewer-0.1.dist-info"; Tasks: ; Languages: ; Flags: replacesameversion
Source: "modified_libraries\7za\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\7za"; Tasks: ; Languages: ; Flags: replacesameversion

#endif

[Code]
#ifdef IncludeLegacyPointTransformer
procedure CurStepChanged(CurStep: TSetupStep);
var
  ArchivePath, OutputPath: String;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    ArchivePath := ExpandConstant('{app}\plugins\Python\Lib\site-packages\seg4d\segmentation_methods\point_transformer.7z');
    OutputPath := ExpandConstant('{app}\plugins\Python\Lib\site-packages\seg4d\segmentation_methods');
    ResultCode := -1;
    if (not Exec(ExpandConstant('{tmp}\seg4d-7zip\extractor.exe'),
        'x "' + ArchivePath + '" -o"' + OutputPath + '" -y', '', SW_HIDE,
        ewWaitUntilTerminated, ResultCode)) or (ResultCode <> 0) then
      RaiseException('Could not extract Point Transformer V1. 7-Zip exit code: ' + IntToStr(ResultCode));
  end;
end;
#endif

function SetProcessEnvironmentVariable(Name: String; Value: String): Boolean;
  external 'SetEnvironmentVariableW@kernel32.dll stdcall';

function InstallPythonDependencies: String;
var
  PythonExe: String;
  RequirementsFile: String;
  ValidatorFile: String;
  RuntimeDirectory: String;
  PipTempDirectory: String;
  PipLogFile: String;
  Parameters: String;
  ResultCode: Integer;
  PermissionsResultCode: Integer;
begin
  Result := '';
  PythonExe := ExpandConstant('{app}\plugins\Python\python.exe');
  ExtractTemporaryFile('cloudcompare-python313.txt');
  ExtractTemporaryFile('validate_cloudcompare_runtime.py');
  RequirementsFile := ExpandConstant('{tmp}\cloudcompare-python313.txt');
  ValidatorFile := ExpandConstant('{tmp}\validate_cloudcompare_runtime.py');
  RuntimeDirectory :=
    ExpandConstant('{app}\plugins\Python\Lib\site-packages\seg4d_runtime');
  PipTempDirectory := ExpandConstant('{app}\seg4d-pip-temp');
  PipLogFile := ExpandConstant('{app}\seg4d-pip-install.log');

  DelTree(RuntimeDirectory, True, True, True);
  DelTree(PipTempDirectory, True, True, True);
  ForceDirectories(PipTempDirectory);
  SetProcessEnvironmentVariable('TMP', PipTempDirectory);
  SetProcessEnvironmentVariable('TEMP', PipTempDirectory);
  DeleteFile(PipLogFile);
  Parameters :=
    '-m pip install --disable-pip-version-check --no-cache-dir --no-compile ' +
    '--only-binary=:all: --log "' + PipLogFile + '" --target "' +
    RuntimeDirectory + '" -r "' + RequirementsFile + '"';
  if (not Exec(PythonExe, Parameters, '', SW_HIDE, ewWaitUntilTerminated, ResultCode)) or
     (ResultCode <> 0) then
  begin
    DelTree(PipTempDirectory, True, True, True);
    Result :=
      'Could not install the CPython 3.13 dependencies for Seg4D. ' +
      'Check the Internet connection and this log: ' + PipLogFile + #13#10 +
      'Exit code: ' + IntToStr(ResultCode);
    exit;
  end;

  Parameters :=
    '/C icacls "' + RuntimeDirectory + '" /reset /T /C /Q';
  if (not Exec(ExpandConstant('{cmd}'), Parameters, '', SW_HIDE,
      ewWaitUntilTerminated, PermissionsResultCode)) or
     (PermissionsResultCode <> 0) then
  begin
    DelTree(PipTempDirectory, True, True, True);
    Result :=
      'Could not make the private Seg4D runtime readable by normal users.' + #13#10 +
      'Exit code: ' + IntToStr(PermissionsResultCode);
    exit;
  end;

  Parameters :=
    '"' + ValidatorFile + '" --runtime "' + RuntimeDirectory +
    '" --requirements "' + RequirementsFile + '" --expected-python 3.13';
  if (not Exec(PythonExe, Parameters, '', SW_HIDE, ewWaitUntilTerminated, ResultCode)) or
     (ResultCode <> 0) then
  begin
    Result :=
      'The private Seg4D runtime failed its CPython 3.13 validation. ' +
      'Check this log: ' + PipLogFile + #13#10 +
      'Exit code: ' + IntToStr(ResultCode);
  end;
  DelTree(PipTempDirectory, True, True, True);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  PythonExe: String;
  PythonCheck: String;
  ResultCode: Integer;
begin
  Result := '';
  ResultCode := -1;
  PythonExe := ExpandConstant('{app}\plugins\Python\python.exe');

  if not FileExists(PythonExe) then
  begin
    Result :=
      'CloudCompare PythonRuntime was not found in the selected directory.' + #13#10 +
      'Install CloudCompare 2.14 with the PythonRuntime plugin, then run this installer again.';
    exit;
  end;

  PythonCheck :=
    '-c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)"';
  if (not Exec(PythonExe, PythonCheck, '', SW_HIDE, ewWaitUntilTerminated, ResultCode)) or
     (ResultCode <> 0) then
  begin
    Result :=
      'This Seg4D release requires CloudCompare 2.14 PythonRuntime with CPython 3.13.' + #13#10 +
      'The selected CloudCompare installation embeds another Python ABI.' + #13#10 +
      'Use a clean CloudCompare 2.14 installation with its PythonRuntime plugin.';
  end;

  if Result = '' then
  begin
    Result := InstallPythonDependencies;
  end;
end;
