![title](https://github.com/Luisjupm/Seg4D/assets/107433987/07a2e9ef-6ae5-4121-a1aa-df4e7101e47a)

Plugin developed within the framework of the research project CAREEN (APOYO-JOVENES-21-RCDT1L-85- SL9E1R). This plugin allows to extend the functionalities of CloudCompare to support the diagnosis of historical constructions.

# Getting started
Be sure to check the [Documentation](https://github.com/Luisjupm/Seg4D/blob/main/assets_sheets/guide_construction-system-segmentation.pdf), which features detailed explanations on how the program works and an User Manual.

# Download
You can find out how to install Seg4D in the [Installation Guide](https://github.com/Luisjupm/Seg4D/blob/main/installer/seg4d_installation_guide.pdf).

Seg4D is available in Windows as a plugin in CloudCompare (2.13) thanks to CloudCompare PythonRuntime (see References). You can download the latest version CloudCompare (Windows installer version) including the CloudCompare PythonRuntime plugin here:

[CloudCompare](https://www.danielgm.net/cc/)

Simply install the latest version of CloudCompare and tick Python checkbox during the installation:

![ccpp](https://github.com/Luisjupm/Seg4D/assets/107433987/71bf7405-c45e-48fb-9334-7d44d65f578b)

**Seg4D plugin in CloudCompare.**
![open1b](https://github.com/Luisjupm/Seg4D/assets/107433987/156c4ea9-1caa-4da5-ada3-c80e85a8b22b)

![open3](https://github.com/Luisjupm/Seg4D/assets/107433987/5bd03527-dcbe-4345-82c6-8df9578c979f)

**Seg4D plugin GUI.**
![Captura de pantalla 2024-05-22 122517](https://github.com/Luisjupm/Seg4D/assets/107433987/2a84dd3b-5d17-4ca4-8d3f-852e7cc64ad1)

# Folders structure

- assets: Pictures for "About" tab.

- assets_sheets: Seg4d user guides.

- configs: Yaml file for execute ".exe" files.

- geometric-based_methods: Scripts files and executables of geometric based methods
	- "analysis_of_arches.py"
	- "analysis_of_deflections.py"
	- "analysis_of_inclinations.py"
	- "analysis_of_vaults.py"
	- "geometrical_features.py" and the executable "jakteristics-0.6.0"

- installer: Installation user guide.

- licenses: Licences of the libraries implemented in the software.

- main_module: Scripts files with functions imported to other scripts.
	- "main.py"
	- "main_gui.py"
	- "ransac.py"

- other_methods: Scripts files of other methods.
	- "anisotropic_denoising.py"
	- "potree_converter.py"
	- "voxelize.py"

- point_cloud_examples: ".bin" files with point cloud examples to train the different methods.

- pre-requisites: Requirements of libraries versions and CloudCompare-PythonPlugin install package.

- radiometric-based_methods: Scripts files of radiometic based methods
	- "color_conversion.py"
	- "stadistical_features.py"

- seg4d_plugin: Boot files.

- segmentation_methods: Scripts files and executables of segmentation methods
	- "deep_learning_segmentation.py" and the executable "point_transformer" 
	- "supervised_machine_learning.py" and the executables "optimal_flow-0.1.11", "scikit-learn-1.3.2" y "tpot-0.12.1"
	- "unsupervised_machine_learning.py" and the executable "scikit-learn-1.3.2"

# Citing Seg4D
You can cite the repository itself:

https://github.com/Luisjupm/Seg4D/

We are currently working on a scientific article about 3DFin, which may be published in 2024.
L.J. Sánchez-Aparicio, P. Sanz-Honrado, R. Santamaria Maestro, Seg4D: a CloudCompare plugin for supporting the diagnosis of historic constructions from 3D point clouds, n.d.

# References
CloudCompare-PythonRuntime, by Thomas Montaigu: [CloudCompare-PythonRuntime](https://github.com/tmontaigu/CloudCompare-PythonRuntime)

# Acknowledgement
The development of Seg4D plugin has been supported by the Community of Madrid and the Higher Polytechnic School of Madrid through the Project CAREEN (desarrollo de nuevos métodos basados en inteligenCia ARtificial para la caracterización de daños en construccionEs históricas a través de nubEs de puNtos 3D) with reference APOYO-JOVENES-21-RCDT1L-85-SL9E1R. Pablo Sanz's pre-doctoral contract is part of grant PID2022-140071OB-C21, funded by MCIN/AEI/10.13039/501100011033 and ESF+.
