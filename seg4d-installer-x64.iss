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
SolidCompression=false
VersionInfoVersion={#appVersion}
VersionInfoCompany={#appPublisher}
VersionInfoProductName={#appName}
VersionInfoProductVersion={#appVersion}
DirExistsWarning=no
AppPublisherURL={#appUrl}
UninstallFilesDir={app}
OutputBaseFilename={#appVerName} Setup
OutputDir=Installer
DefaultGroupName={#appName}
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
DisableStartupPrompt=true
DiskSpanning=yes
SetupIconFile=logo.ico
; It is for modyfing any environment variable
;ChangesEnvironment=true

[Files]
;Files to be copied
Source: "seg4d.py"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}"; Tasks: ; Languages: ; Flags: replacesameversion
Source: "assets\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\assets"; Flags: recursesubdirs
Source: "configs\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\configs"; Flags: recursesubdirs
Source: "geometric-based_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\geometric-based_methods"; Flags: recursesubdirs
Source: "main_module\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\main_module"; Flags: recursesubdirs
Source: "other_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\other_methods"; Flags: recursesubdirs
Source: "point_clouds_examples\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\point_clouds_examples"; Flags: recursesubdirs
Source: "radiometric-based_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\radiometric-based_methods"; Flags: recursesubdirs
;In the following line we exclude the folder point_transformer because we copy as a 7zip and then unzip it because the paths are too large
Source: "segmentation_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\segmentation_methods"; Flags: recursesubdirs; Excludes: "point_transformer\*,point_transformer.7z"
Source: "segmentation_methods\point_transformer.7z"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\segmentation_methods"; Flags: deleteafterinstall 
Source: "seg4d_plugin\*"; DestDir: "{app}\plugins-python\seg4d_plugin"; Flags: recursesubdirs
  
[Run]
;Install the dependencies
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install scipy"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install scikit-learn==1.1.0"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install pandas"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install matplotlib"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install open3d"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install fuzzy-c-means"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install yellowbrick"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install opencv-python"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install PyYAML"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install tkpdfviewer==0.1"; Flags: runhidden

[Files]
;Additional libraries for the proper instalation of the program
Source: "modified_libraries\tkPDFViewer\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\tkPDFViewer"; Tasks: ; Languages: ; Flags: replacesameversion
Source: "modified_libraries\tkPDFViewer-0.1.dist-info\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\tkPDFViewer-0.1.dist-info"; Tasks: ; Languages: ; Flags: replacesameversion
Source: "modified_libraries\7za\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\7za"; Tasks: ; Languages: ; Flags: replacesameversion

;Unzip the pointtransformer application due its size

[Run]
;Unzip the point-transformer app
Filename: {app}\plugins\Python\Lib\site-packages\7za\7za.exe; Parameters: " x ""{app}\plugins\Python\Lib\site-packages\seg4d\segmentation_methods\point_transformer.7z"" -o""{app}\plugins\Python\Lib\site-packages\seg4d\segmentation_methods"" -y"; Flags: runhidden runascurrentuser waituntilterminated shellexec

[InstallDelete]
Type: files; Name: "{app}\plugins\Python\Lib\site-packages\seg4d\segmentation_methods\point_transformer.7z"