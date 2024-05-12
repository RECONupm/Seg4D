# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:12:14 2024

@author: Digi_2
"""

import argparse
import os
import yaml

import pandas as pd
import numpy as np

import jakteristics.utils
from jakteristics.extension import compute_features


def main():
    # Import all the parameters form the CMD
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',type=str,help='Yaml configuration file')
    parser.add_argument('--o',type=str,help='Output_directory')    
    args=parser.parse_args() 
            
    # Read the configuration from the YAML file for the set-up
    with open(args.i, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    input_file= config_data.get('INPUT_POINT_CLOUD')
    output_directory= config_data.get('OUTPUT_DIRECTORY')
    features_selected=config_data['CONFIGURATION']['input_features']
    radius=config_data['CONFIGURATION']['radius']
 
    print("Input file located in " + str(input_file))
    print("Output file located in " + str(output_directory))
    print("Radius = " + str(radius))
    print("Features chosen = " + str(features_selected))
    
    xyz=pd.read_csv(input_file,delimiter=' ')
    xyz_array = np.ascontiguousarray(xyz.values)
    
    # Inicializamos una lista para almacenar las características para cada radio
    all_features = []

    # Iteramos sobre cada radio
    for r in radius:
        # Calculamos las características para el radio actual
        features = compute_features(xyz_array, search_radius=r, feature_names=features_selected)
        all_features.append(features)
    
    # Concatenamos todas las características en una matriz
    xyz_array_with_features = np.concatenate([xyz_array] + all_features, axis=1)
    
    # Añadimos los headers con los valores de radio correspondientes
    headers = list(xyz.columns)
    for r in radius:
        for feature in features_selected:
            headers.append(f"{feature}_({r})")
    
    # Creamos un DataFrame desde la matriz completa
    df = pd.DataFrame(xyz_array_with_features, columns=headers)
    
    # Guardamos el DataFrame como un archivo de texto en formato .txt
    df.to_csv(os.path.join(output_directory, 'pc_computed.txt'), sep=' ', index=False)
 
    
if __name__ == '__main__':
    main()
