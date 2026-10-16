# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:29:32 2023

@author: Digi_2
"""

import argparse
import json
import os
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")

from sklearn import (
    cluster,
    decomposition,
    ensemble,
    feature_selection,
    kernel_approximation,
    linear_model,
    naive_bayes,
    neighbors,
    neural_network,
    preprocessing,
    svm,
    tree,
)
from sklearn.metrics import classification_report
from tpot import TPOTClassifier
from tpot.config.classifier import classifier_config_dict
from tpot.operator_utils import source_decode
from yellowbrick.classifier import ConfusionMatrix


def run_self_test():
    """Exercise imports that TPOT resolves dynamically in a frozen build."""
    import xgboost

    flattened = np.ravel([[0], [1]])
    if flattened.tolist() != [0, 1]:
        raise RuntimeError("NumPy self-test failed.")
    unavailable_operators = [
        operator_name
        for operator_name in classifier_config_dict
        if source_decode(operator_name, verbose=0)[2] is None
    ]
    if unavailable_operators:
        raise RuntimeError(
            "TPOT operators missing from the frozen build: "
            + ", ".join(unavailable_operators)
        )

    TPOTClassifier(
        generations=1,
        population_size=1,
        cv=2,
        config_dict="TPOT light",
        disable_update_check=True,
        verbosity=0,
    )
    print(
        "TPOT self-test passed "
        f"(TPOT classifier and XGBoost {xgboost.__version__})."
    )


def write_progress(output_directory, percent, message):
    """Publish a small status file that the Seg4D window can poll."""
    status_path = os.path.join(output_directory, "automl_progress.json")
    temporary_path = status_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as status_file:
        json.dump({"percent": percent, "message": message}, status_file)
    os.replace(temporary_path, status_path)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tpot",
        description="Seg4D TPOT AutoML console runtime.",
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
        help="Validate the frozen AutoML runtime and exit.",
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
        return
    if not args.i:
        parser.error("--input/--i is required unless --self-test is used")

    # Read the configuration from the YAML file for the set-up
    with open(args.i, "r", encoding="utf-8") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    test_file = config_data["INPUT_POINT_CLOUD_TESTING"]
    train_file = config_data["INPUT_POINT_CLOUD_TRAINING"]
    output_directory = os.path.abspath(
        args.o or config_data["OUTPUT_DIRECTORY"]
    )
    features2include_path = config_data["INPUT_FEATURES"]
    configuration = config_data["CONFIGURATION"]
    generations = int(configuration["generations"])
    population_size = int(configuration["population_size"])
    mutation_rate = float(configuration["mutation_rate"])
    crossover_rate = float(configuration["crossover_rate"])
    cv = int(configuration["cv"])
    max_time_mins = int(configuration["max_time_mins"])
    max_eval_time_mins = int(configuration["max_eval_time_mins"])
    early_stop_value = configuration.get("early_stop")
    early_stop = (
        None
        if early_stop_value in (None, "", 0, "0")
        else int(early_stop_value)
    )
    scoring = configuration["scoring"]
    os.makedirs(output_directory, exist_ok=True)
    
    # There are an issue with the f1 score. This score only accepts 0 and 1 lables. So if you introduce 3 and 4 labels, for example, throws and error
    if scoring == "f1":
        scoring = "f1_macro"
    elif scoring == "precision":
        scoring = "precision_macro"

    
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
    
    
    write_progress(output_directory, 5, "Loading training and testing data...")
    pcd_training = pd.read_csv(train_file, sep=r"\s+")
    pcd_testing = pd.read_csv(test_file, sep=r"\s+")
    
    # Create the pandas with the features selected for the train and test case
    labels2include = ["Classification"]
    pcd_training.dropna(inplace=True)
    pcd_testing.dropna(inplace=True)
    labels_train = pcd_training[labels2include]
    with open(features2include_path, "r", encoding="utf-8") as file:
        features2include = [
            feature.strip()
            for feature in file.read().split(",")
            if feature.strip()
        ]
    if not features2include:
        raise ValueError("At least one input feature must be selected.")
    missing_training = sorted(set(features2include) - set(pcd_training.columns))
    missing_testing = sorted(set(features2include) - set(pcd_testing.columns))
    if missing_training or missing_testing:
        raise ValueError(
            "Selected features are missing. "
            f"Training: {missing_training}; testing: {missing_testing}"
        )

    X_train = pcd_training[features2include]
    y_train = labels_train.to_numpy()
    y_train_reshaped = np.ravel(y_train)
    labels_evaluation = pcd_testing[labels2include]
    X_test = pcd_testing[features2include]
    y_test = np.ravel(labels_evaluation.to_numpy())

    if len(np.unique(y_train_reshaped)) < 2:
        raise ValueError("The training cloud must contain at least two classes.")
    smallest_class = int(pd.Series(y_train_reshaped).value_counts().min())
    if cv > smallest_class:
        raise ValueError(
            f"Cross-validation ({cv}) exceeds the smallest class size "
            f"({smallest_class})."
        )
    
    # Create the pipeline for the TPOT
    pipeline_optimizer = TPOTClassifier(
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        scoring=scoring,
        cv=cv,
        max_time_mins=max_time_mins,
        max_eval_time_mins=max_eval_time_mins,
        early_stop=early_stop,
        random_state=None,
        verbosity=2,
        use_dask=False,
        n_jobs=1,
        disable_update_check=True,
    )
       
    
    # Run the TPOT with the training data and the pipeline
    write_progress(output_directory, 10, "Optimizing the classification pipeline...")
    pipeline_optimizer.fit(X_train, y_train_reshaped)
   
    
    # Serialize the best pipeline (model) into a .pkl file
    write_progress(output_directory, 88, "Saving the best pipeline...")
    best_pipeline_path = os.path.join(output_directory, "best_pipeline.pkl")
    joblib.dump(pipeline_optimizer.fitted_pipeline_, best_pipeline_path)
    
    # Load the pickled model 
    loaded_model = joblib.load(best_pipeline_path)

    # Prediction
    y_pred = loaded_model.predict(X_test)
    
    # Save the model to a file
    joblib.dump(loaded_model, os.path.join(output_directory, "model.pkl"))
    pipeline_optimizer.export(
        os.path.join(output_directory, "best_pipeline.py")
    )
    
    # Create the confusion matrix
    write_progress(output_directory, 93, "Creating evaluation reports...")
    cm = ConfusionMatrix(loaded_model, cmap="Blues")
    cm.score(X_test, y_test)
    cm.show(
        outpath=os.path.join(output_directory, "confusion_matrix.png"),
        clear_figure=True,
    )
    
    # Create the classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    # Write the report to a file
    with open(
        os.path.join(output_directory, "classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)
    
    # Create the final point cloud with a layer of predictions
    pcd_testing_subset = pcd_testing[["X", "Y", "Z"]].copy()
    pcd_testing_subset["Predictions"] = y_pred
    # Saving the DataFrame to a CSV file
    pcd_testing_subset.to_csv(
        os.path.join(output_directory, "predictions.txt"), index=False
    )
    write_progress(output_directory, 100, "AutoML completed.")


if __name__ == "__main__":
    raise SystemExit(main())
