# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:34:55 2023

@author: Digi_2
"""

import argparse
import os
import yaml

import pandas as pd
from optimalflow.autoFS import dynaFS_clf

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
    selectors= config_data.get('SELECTORS_FILE')
    output_directory= config_data.get('OUTPUT_DIRECTORY')
    features=config_data['CONFIGURATION']['f']
    cross_val=config_data['CONFIGURATION']['cv']
    
    # input_file=args.i
    print("Input file located in " + str(input_file))
    # output_directory=args.o
    print("Output file located in " + str(output_directory))
    # selectors=args.s
    print("Selectors chosen = " + str(selectors))
    # features=args.f
    print("Features chosen = " + str(features))
    # cross_val=args.cv
    print("Value chosen for cross validation = " + str(cross_val))
            
    # Read the input file and prepare them. We delete the X,Y,Z and Classification from the features as well as the column of classification from the input point cloud
    tr = pd.read_csv(input_file,delimiter=' ')
    tr_features_copy=tr
    if 'Classification' in tr_features_copy.columns:
        tr_features_clean=tr_features_copy.drop (['Classification'], axis=1)
        tr_labels_clean=tr[['Classification']]   
    else:
        tr_features_clean=tr_features_copy
        tr_labels_clean=tr
    
    # Check if any missing values exist in the DataFrame
    if tr_features_clean.any().any():
          # Clean the dataframe, and drop all the line that contains a NaN (Not a Number) value.
          tr_features_filled=tr_features_clean.ffill()         
          def missing_values_checker():
            # Check for missing values
            missing_values = tr_features_filled.isnull()
            # To count the total number of missing values in the DataFrame
            total_missing_count = missing_values.sum().sum()
          missing_values_checker()
         
    #Creates an Array from labels
    tr_labels=tr_labels_clean.values
    tr_labels_2d = tr_labels.ravel()
    
    # Read the selectors
    # Open the file in read mode
    with open(selectors, 'r') as file:
        # Read all lines in the file and strip newline characters
        selectors_list = [line.strip() for line in file]
        print(selectors_list)
    # Run optimalflow
    reg_fs_demo = dynaFS_clf(
                            custom_selectors=selectors_list,
                            fs_num = int(features), 
                            random_state = None, 
                            cv = int(cross_val), 
                            input_from_file = False
                            )

    features_selected=reg_fs_demo.fit(tr_features_filled,tr_labels_2d)
            
    with open(os.path.join(output_directory, "features.txt"), "w") as file:
    # Write the list elements to the file separated by commas
        for index, item in enumerate(features_selected[1]):
            # Check if it's not the first item, then add a comma
            if index != 0:
                file.write(",")
            file.write(str(item))
            
if __name__=='__main__':
	main()