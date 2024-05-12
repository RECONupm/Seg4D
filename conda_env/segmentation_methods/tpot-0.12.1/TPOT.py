# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:29:32 2023

@author: Digi_2
"""

import argparse

import os
import subprocess
import sys
import yaml


import joblib

from sklearn.ensemble import *
from sklearn.kernel_approximation import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix

import pandas as pd
import numpy

from tpot import TPOTClassifier
from tpot import *
import tpot


def main():

    # Import all the parameters form the CMD
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',type=str,help='Yaml configuration file')
    parser.add_argument('--o',type=str,help='Output_directory')    
    args=parser.parse_args()  

    # Read the configuration from the YAML file for the set-up
    with open(args.i, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    test_file= config_data.get('INPUT_POINT_CLOUD_TESTING')
    train_file= config_data.get('INPUT_POINT_CLOUD_TRAINING')
    output_directory= config_data.get('OUTPUT_DIRECTORY')
    features2include_path= config_data.get('INPUT_FEATURES')                      
    generations=config_data['CONFIGURATION']['generations']
    population_size=config_data['CONFIGURATION']['population_size']
    mutation_rate=config_data['CONFIGURATION']['mutation_rate']
    crossover_rate=config_data['CONFIGURATION']['crossover_rate']
    cv=config_data['CONFIGURATION']['cv']
    max_time_mins=config_data['CONFIGURATION']['max_time_mins']
    max_eval_time_mins=config_data['CONFIGURATION']['max_eval_time_mins']
    early_stop=config_data['CONFIGURATION']['early_stop']
    scoring=config_data['CONFIGURATION']['scoring']
    
    # There are an issue with the f1 score. This score only accepts 0 and 1 lables. So if you introduce 3 and 4 labels, for example, throws and error
    if scoring=="f1":
        scoring="f1_macro"

    
    # test_file=args.te
    print("Test file located in " + test_file)
    # train_file=args.tr
    print("Train file located in " + train_file)
    # output_directory=args.o
    print("Output directory is " + output_directory)
    # # features2include=args.f
    print("Features to include = " + features2include_path)
    # # generations=args.ge
    print("Value chosen for generations = " + str(generations))
    # population_size=args.ps
    print("Value chosen for population size = " + str(population_size))
    # mutation_rate=args.mr
    print("Value chosen for mutation rate = " + str(mutation_rate))
    # crossover_rate=args.cr
    print("Value chosen for crossover rate = " + str(crossover_rate))
    # cv=args.cv
    print("Value chosen for cross validation = " + str(cv))
    # max_time_mins=args.mtm
    print("Value chosen for max time mins = " + str(max_time_mins))
    # max_eval_time_mins=args.metm
    print("Value chosen for max eval time mins = " + str(max_eval_time_mins))
    # early_stop=args.ng
    print("Value chosen for early stop = " + str(early_stop))
    # scoring=args.s
    print("Scoring chosen = " + str(scoring))
    
    
    #Store in a Pandas dataframe the content of the file
    pcd_training=pd.read_csv(train_file,delimiter=' ')
    pcd_testing=pd.read_csv(test_file,delimiter=' ')
    
    # Create the pandas with the features selected for the train and test case
    labels2include= ['Classification']
    pcd_training.dropna(inplace=True)
    pcd_testing.dropna(inplace=True)
    labels_train=pcd_training[labels2include]
    with open(features2include_path, "r") as file:
        features2include = [line.strip().split(',') for line in file]    
    features=pcd_training[features2include[0]]   
    X_train=features
    y_train=labels_train.to_numpy()
    y_train_reshaped = np.ravel(y_train)
    labels_evaluation=pcd_testing[labels2include]
    features=pcd_testing[features2include[0]]
    X_test=features
    y_test=labels_evaluation.to_numpy()
    y_test_reshaped=np.ravel(y_test)
    
    # Create the pipeline for the TPOT
    pipeline_optimizer = TPOTClassifier(
                                        generations=int(generations),
                                        population_size=int(population_size),
                                        mutation_rate=float(mutation_rate),
                                        crossover_rate=float(crossover_rate),
                                        scoring=scoring,
                                        cv=int(cv),
                                        max_time_mins=int(max_time_mins),
                                        max_eval_time_mins=int(max_eval_time_mins),
                                        early_stop=int(early_stop),
                                        random_state=None,
                                        verbosity=2,
                                        use_dask=False
                                        )
       
    
    # Run the TPOT with the training data and the pipeline
    pipeline_optimizer.fit(X_train, y_train_reshaped)
   
    
    # Serialize the best pipeline (model) into a .pkl file
    joblib.dump(pipeline_optimizer.fitted_pipeline_,os.path.join(output_directory, 'best_pipeline.pkl'))  # Replace 'fitted_pipeline_' with the appropriate attribute
    
    # Load the pickled model 
    loaded_model = joblib.load(os.path.join(output_directory, 'best_pipeline.pkl'))    

    # Prediction
    y_pred = loaded_model.predict(X_test)
    
    # Save the model to a file
    joblib.dump(loaded_model, os.path.join(output_directory, 'model.pkl'))
    
    # Create the confusion matrix
    cm= ConfusionMatrix(loaded_model, cmap='Blues')
    cm.score (X_test,y_test)
    cm.show(outpath=os.path.join(output_directory, 'confusion_matrix.png'))  # Save the confusion matrix to a file
    
    # Create the classification report
    report=classification_report(y_test, y_pred)
    # Write the report to a file
    with open(os.path.join(output_directory,'classification_report.txt'), 'w') as file:
        file.write(report)
    
    # Create the final point cloud with a layer of predictions
    pcd_testing_subset = pcd_testing[['X', 'Y', 'Z']].copy()
    pcd_testing_subset['Predictions'] = y_pred
    # Saving the DataFrame to a CSV file
    pcd_testing_subset.to_csv(os.path.join(output_directory, 'predictions.txt'), index=False)

if __name__=='__main__':
 	main()