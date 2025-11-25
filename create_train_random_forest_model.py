import os
from datetime import datetime
import constants as c
from typing import Dict, Tuple
import numpy as np
import sklearn.ensemble
from matplotlib import pyplot as plt
from scipy.stats import randint
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay
from tqdm import tqdm
import word_embeddings

from constants import BERT_MODEL


# Get unique name for output directory
def generate_run_dir_name() -> str:
    now = datetime.now()

    day   = now.strftime("%d")
    month = now.strftime("%m")
    hour  = now.strftime("%H")
    minute = now.strftime("%M")
    second = now.strftime("%S")

    return f"{day}_{month}_{hour}_{minute}_{second}"


# Write BERT variant used to file
def write_model_used_to_file(filepath: str, model_selection: BERT_MODEL):
    report_path = os.path.join(filepath, "embedding_model_selection_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Random Forest RandomizedSearchCV Report\n")
        f.write("======================================\n\n")
        f.write(f"BERT Model Used: {model_selection.name}\n")
        f.write(f"BERT Model Embeddings Filepath: {model_selection.value}\n")


# Read features in from file
# split into features and labels
def read_features(model_selection: BERT_MODEL) -> Tuple[np.ndarray, np.ndarray]:
    X_features = [] # joined array of all features for all desired classes together
    y_labels = [] # labels of features for all desired classes together
    # for each class, read its values into X_features and its labels into y_labels

    df_labels_embeddings = word_embeddings.get_embeddings_and_labels_for_model(model_selection = model_selection,
                                                                               use_cached_embeddings=True)

    embeddings = df_labels_embeddings["Embedding"].values
    X_features = np.vstack(embeddings).astype(np.float32)
    y_labels = df_labels_embeddings["GT"].astype(int).to_numpy()

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    return X_features, y_labels


# Find and plot accuracy vs number of estimators
# for random forest classifier
def find_best_n_estimators_random_forest(Xtrain:np.ndarray, Xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray):
    n_estimator_val = []
    rf_score = []

    # pretty print loading bars with tqdm just for fun
    for i in tqdm(range(10, 301), desc="Training Random Forests", unit="model"):
        rf_classifier = RandomForestClassifier(n_estimators=i)
        rf_classifier.fit(Xtrain, ytrain)
        score = rf_classifier.score(Xtest, ytest)
        n_estimator_val.append(i)
        rf_score.append(score)

    plt.figure(figsize=(8, 4.5))
    plt.plot(n_estimator_val, rf_score, linewidth=1)
    plt.xlabel("Number of Estimator Values (integer)")
    plt.ylabel("Accuracy")
    plt.title("Random Forest Classifier Accuracy vs. Number of Estimators")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


# Run classification report to assess classifier performance
# classifier should ALREADY be fitted
# Also compute AUC and graph ROC
def assess_clf_performance(clf: RandomForestClassifier,
                           Xtest: np.ndarray,
                           ytest: np.ndarray,
                           classes: Dict[str, int],
                           directory: str = None,
                           save_report_and_model: bool = False):

    ypred = clf.predict(Xtest)

    # evaluate model & print report
    y_score = clf.predict_proba(Xtest)[:, 1]
    fpr, tpr, _ = roc_curve(ytest, y_score)
    roc_auc = auc(fpr, tpr)

    report_str = classification_report(
        ytest,
        ypred,
        target_names=list(classes.keys())
    )

    print("----- Simple Test/Train Split Random Forest Classification Report -----")
    print(report_str)
    print(f"AUC: {roc_auc:.3f}")

    # Save Reprot & Model
    if directory is not None and save_report_and_model:
        manually_dir = os.path.join(directory, "manually_instantiated_model")
        os.makedirs(manually_dir, exist_ok=True)

        # Create & Save Report
        report_path = os.path.join(manually_dir, "random_forest_manual_split_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Random Forest Manual Train/Test Split Report\n")
            f.write("=============================================\n\n")
            f.write("Classifier parameters\n")
            f.write("---------------------\n")
            for k, v in clf.get_params().items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

            f.write("Classification report\n")
            f.write("---------------------\n")
            f.write(report_str + "\n\n")

            f.write("ROC / AUC\n")
            f.write("---------\n")
            f.write(f"AUC: {roc_auc:.4f}\n")

        print(f"Saved manual split report to: {report_path}")

    # Save model to same directory
        model_path = os.path.join(manually_dir, "random_forest_manual_model.joblib")
        joblib.dump(clf, model_path)
        print(f"Saved manual split model to: {model_path}")

    # Plot ROC
    plot_filepath = os.path.join(manually_dir, "roc_curve")
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name="RandomForest").plot()
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title("ROC Curve (Hold-out Split)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.show()


# Train and test a random forest model with a simple test/training split
def train_randomforest(clf: RandomForestClassifier,
                       X_features: np.ndarray,
                       y_labels: np.ndarray,
                       classes: Dict[str, int],
                       random_state: int = 42,
                       find_n_estimators: bool = False,
                       test_size=0.3,
                       print_perf_metrics: bool = True,
                       report_directory: str = None,
                       save_report_and_model: bool = False) -> RandomForestClassifier:

    print("---- BEGIN Training Random Forest Classifier ---")
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_features, y_labels, test_size=test_size, stratify=y_labels,
        random_state=random_state
    )

    # Train the model
    clf.fit(Xtrain, ytrain)
    # Print Performance Metrics
    if print_perf_metrics:
        assess_clf_performance(clf,
                               Xtest,
                               ytest,
                               classes,
                               directory=report_directory,
                               save_report_and_model=save_report_and_model)

    # (Optional) find & print perf graphs with different numbers of estimators
    if find_n_estimators:
        find_best_n_estimators_random_forest(Xtrain, Xtest, ytrain, ytest)
    print("---- END Training Random Forest Classifier ---")
    return clf


# Perform K-Fold validation and optionally save report
def perform_k_fold_randomforest(X_features: np.ndarray,
                                y_labels: np.ndarray,
                                report_directory: str,
                                n_estimators: int = 200,
                                random_state: int = 42,
                                save_report: bool = False):
    print(f"---- BEGIN Cross Validation (Random Forest) ----")
    # Run Cross Validation and get scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    scores = cross_val_score(rf, X_features, y_labels, cv=cv, scoring="accuracy")
    # Compute mean & STDEV
    mean_score = scores.mean()
    std_score = scores.std()

    print(f"Cross-validation accuracy: {mean_score:.3f} ± {std_score:.3f}")
    print(f"---- END Cross Validation (Random Forest) ----")

    if save_report:
        kfold_dir = os.path.join(report_directory, "k_fold_validation")
        os.makedirs(kfold_dir, exist_ok=True)

        report_path = os.path.join(kfold_dir, "random_forest_kfold_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Random Forest K-Fold Cross-Validation Report\n")
            f.write("===========================================\n\n")
            f.write(f"n_estimators: {n_estimators}\n")
            f.write(f"random_state: {random_state}\n")
            f.write(f"n_splits: {cv.get_n_splits()}\n\n")

            f.write("Fold accuracies:\n")
            for i, s in enumerate(scores, start=1):
                f.write(f"  Fold {i}: {s:.4f}\n")

            f.write("\n")
            f.write(f"Mean accuracy: {mean_score:.4f}\n")
            f.write(f"Std accuracy:  {std_score:.4f}\n")

        print(f"Saved k-fold report to: {report_path}")


# run a random hyperparameter search for random forest
# return the model with the best accuracy AND save a report
def run_random_param_search(X_train: np.ndarray,
                            y_train: np.ndarray,
                            directory: str) -> sklearn.ensemble.RandomForestClassifier:
    print("----- BEGIN Randomized Search CV -----")
    # define the estimator
    rf_classifier = RandomForestClassifier(random_state=42)

    # define the parameter distributions
    param_distributions = {
        "n_estimators": randint(100, 400),
        "max_depth": [None] + list(range(10, 61, 10)),  # none = unlimited
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ['sqrt', 'log2', None],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy", "log_loss"],
    }

    # create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        estimator=rf_classifier,
        param_distributions=param_distributions,
        n_iter=40,
        cv=5,
        scoring='accuracy',
        random_state=36,
        n_jobs=-1,
        verbose=2
    )

    random_search.fit(X_train, y_train)

    # Access the best parameters and best score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    print("----- Completed Randomized Search CV -----")

    # ---- Save report to directory ----
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(directory, "random_forest_random_search_report.txt")

    cv_results = random_search.cv_results_
    mean_scores = cv_results["mean_test_score"]
    std_scores = cv_results["std_test_score"]
    params_list = cv_results["params"]

    # sort configurations from best to worst
    sorted_indices = np.argsort(mean_scores)[::-1]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Random Forest RandomizedSearchCV Report\n")
        f.write("======================================\n\n")

        f.write("Best configuration\n")
        f.write("------------------\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nBest mean CV accuracy: {best_score:.4f}\n\n")

        f.write("Search space\n")
        f.write("-----------\n")
        f.write(str(param_distributions) + "\n\n")

        f.write("All tried configurations (sorted by mean accuracy)\n")
        f.write("-------------------------------------------------\n")
        for rank, idx in enumerate(sorted_indices, start=1):
            f.write(f"Rank {rank}\n")
            f.write(f"  mean_accuracy: {mean_scores[idx]:.4f}\n")
            f.write(f"  std_accuracy:  {std_scores[idx]:.4f}\n")
            f.write(f"  params:        {params_list[idx]}\n\n")

    print(f"Saved random search report to: {report_path}")
    print("----- END Randomized Search CV -----")
    return random_search.best_estimator_


# run random search and save best result to file
def perform_random_param_search(X_features: np.ndarray,
                                y_labels: np.ndarray,
                                directory: str,
                                save_to_file: bool = False) -> RandomForestClassifier:
    # Split into Test/Train sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_features, y_labels, test_size=0.3, stratify=y_labels,
        random_state=42
    )
    # make the output directory
    output_dir = directory + "/random_search_model/"
    os.makedirs(output_dir, exist_ok=True)

    clf = run_random_param_search(Xtrain, ytrain, output_dir)

    if save_to_file:
        joblib.dump(clf, f'{output_dir}/random_forest_model.joblib')

    return clf


# Create & Save various Random Forest Models
def create_new_trained_models(run_k_fold_validation: bool,
                              run_new_simple_rf_classifier: bool,
                              run_random_param_search: bool,
                              model_selection: BERT_MODEL):
    # ---- Run Model -----
    directory_to_save_models = (
            c.RANDOM_FOREST_TRAINED_MODEL_DIR_PREFIX + "sandbox/" + generate_run_dir_name()
    )

    os.makedirs(directory_to_save_models, exist_ok=True)
    write_model_used_to_file(directory_to_save_models, model_selection)

    classes = {"humour": 1, "non_humour": 0}

    # Read features in from file
    X_features, y_labels = read_features(model_selection)

    # Perform k-fold validation random forest
    if run_k_fold_validation:
        perform_k_fold_randomforest(X_features,
                                    y_labels,
                                    report_directory=directory_to_save_models,
                                    save_report=True)

    # Fit Random Forest Model
    if run_new_simple_rf_classifier:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf = train_randomforest(clf,
                                 X_features,
                                 y_labels,
                                 classes,
                                 report_directory=directory_to_save_models,
                                 save_report_and_model=True)

    # Perform Random Search
    if run_random_param_search:
        perform_random_param_search(X_features,
                                    y_labels,
                                    directory=directory_to_save_models,
                                    save_to_file=True)



if __name__ == "__main__":
    create_new_trained_models(run_k_fold_validation=True,
                              run_new_simple_rf_classifier=True,
                              run_random_param_search=True,
                              model_selection=BERT_MODEL.BERT)