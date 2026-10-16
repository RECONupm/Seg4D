# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:12:14 2024

@author: Digi_2
"""

import argparse
import os
import sys
import yaml

import pandas as pd
import numpy as np

import jakteristics.utils
from jakteristics.extension import compute_features


def run_self_test():
    points = np.ascontiguousarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = compute_features(
        points, search_radius=2.0, feature_names=["linearity"]
    )
    if result.shape != (len(points), 1):
        raise RuntimeError(f"Unexpected Jakteristics result shape: {result.shape}")
    print("Jakteristics runtime self-test passed.")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="jakteristics",
        description="Seg4D geometric-feature console runtime.",
    )
    parser.add_argument("-i", "--input", dest="i", help="YAML configuration file.")
    parser.add_argument(
        "-o",
        "--output",
        dest="o",
        help="Output directory (overrides the YAML value).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate the frozen Jakteristics runtime and exit.",
    )
    return parser


def main(arguments=None):
    parser = build_parser()
    command_line = list(sys.argv[1:] if arguments is None else arguments)
    if not command_line:
        parser.print_help()
        return 0
    args = parser.parse_args(command_line)
    if args.self_test:
        run_self_test()
        return 0
    if not args.i:
        parser.error("--input/--i is required unless --self-test is used.")
            
    # Read the configuration from the YAML file for the set-up
    with open(args.i, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    input_file= config_data.get('INPUT_POINT_CLOUD')
    output_directory = os.path.abspath(args.o or config_data.get('OUTPUT_DIRECTORY'))
    os.makedirs(output_directory, exist_ok=True)
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
    raise SystemExit(main())
