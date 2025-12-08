import os
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline

import corpus_loader

import constants as c
from typing import Dict, Tuple
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import randint
import joblib
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score, PrecisionRecallDisplay
)
import word_embeddings

from constants import BERT_MODEL

from sklearn.metrics import f1_score, make_scorer

f1_macro_scorer = make_scorer(f1_score, average="macro")

# Get unique name for bert_embeddings directory
def generate_run_dir_name(model_selection: BERT_MODEL, use_smote:bool) -> str:
    now = datetime.now()

    day = now.strftime("%d")
    month = now.strftime("%m")
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    second = now.strftime("%S")

    if use_smote:
        return f"{day}_{month}_{hour}_{minute}_{second}_{model_selection.name}_smote"  # model selection at end so ordered by date
    else:
        return f"{day}_{month}_{hour}_{minute}_{second}_{model_selection.name}_no_smote"  # model selection at end so ordered by date


# Read features in from file
# split into features and labels
def read_features(model_selection: BERT_MODEL) -> Tuple[np.ndarray, np.ndarray]:
    # for each class, read its values into X_features and its labels into y_labels
    df_labels_embeddings = word_embeddings.get_embeddings_and_labels_for_model(model_selection = model_selection,
                                                                               use_cached_embeddings=True)

    embeddings = df_labels_embeddings["Embedding"].values
    X_features = np.vstack(embeddings).astype(np.float32)
    y_labels = df_labels_embeddings["GT"].astype(int).to_numpy()

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    return X_features, y_labels


# Train and test a random forest model with a simple test/training split
def train_randomforest(Xtrain: np.ndarray,
                       ytrain: np.ndarray,
                       Xtest: np.ndarray,
                       ytest: np.ndarray,
                       classes: Dict[int, str],
                       model_selection: BERT_MODEL,
                       random_state: int = 42,
                       report_directory: str = None) -> RandomForestClassifier:


    sm = SMOTE(random_state=42)
    Xtrain, ytrain = sm.fit_resample(Xtrain, ytrain)

    print("---- BEGIN Training Random Forest Classifier ---")
    clf = RandomForestClassifier(n_estimators=500,
                                 max_depth=10,
                                 max_features="sqrt",
                                 min_samples_leaf=9,
                                 min_samples_split=17,
                                 bootstrap=False,
                                 criterion="gini",
                                 class_weight="balanced",
                                 random_state=random_state
                                 )

    # Train the model
    clf.fit(Xtrain, ytrain)

    # Assess Performance and Save Report to file
    full_output_dir = os.path.join(report_directory, "manually_instantiated_model")
    os.makedirs(full_output_dir, exist_ok=True)
    generate_model_analysis_report(Xtrain,
                                   ytrain,
                                   Xtest,
                                   ytest,
                                   clf,
                                   full_output_dir,
                                   classes,
                                   model_selection=model_selection
                                   )

    # Save the trained model to file
    joblib.dump(clf, f'{full_output_dir}/random_forest_model.joblib')

    print("---- END Training Random Forest Classifier ---")
    return clf


# Perform K-Fold validation and optionally save report
def perform_k_fold_randomforest(X_features: np.ndarray,
                                y_labels: np.ndarray,
                                report_directory: str,
                                model_selection: BERT_MODEL,
                                n_estimators: int = 200,
                                random_state: int = 42):

    print(f"---- BEGIN Cross Validation (Random Forest) ----")
    # Run Cross Validation and get scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    scores = cross_val_score(rf, X_features, y_labels, cv=cv, scoring="accuracy")

    # Compute mean & STDEV
    mean_score = scores.mean()
    std_score = scores.std()

    # Generate & Save Report
    kfold_dir = os.path.join(report_directory, "k_fold_validation")
    os.makedirs(kfold_dir, exist_ok=True)
    report_path = os.path.join(kfold_dir, "random_forest_kfold_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Random Forest K-Fold Cross-Validation Report\n")
        f.write("===========================================\n\n")
        f.write(f"Embedding Used: {model_selection.name}\n")
        f.write(f"n_estimators: {n_estimators}\n")
        f.write(f"random_state: {random_state}\n")
        f.write(f"n_splits: {cv.get_n_splits()}\n\n")

        f.write("Fold accuracies:\n")
        for i, s in enumerate(scores, start=1):
            f.write(f"  Fold {i}: {s:.4f}\n")

        f.write("\n")
        f.write(f"Mean accuracy: {mean_score:.4f}\n")
        f.write(f"Std accuracy: {std_score:.4f}\n")

    print(f"Saved k-fold report to: {report_path}")
    print(f"---- END Cross Validation (Random Forest) ----")


# run a random hyperparameter search for random forest
# return the model with the best accuracy AND save a report
def run_random_param_search(X_train: np.ndarray,
                            y_train: np.ndarray) -> RandomForestClassifier:
    # Define the estimator
    rf_classifier = RandomForestClassifier() #    class_weight = {0: 1.0, 1: 5.0}

    # Define SMOTE inside of pipeline to use inside the model itself
    pipeline = ImbPipeline([
        ("smote", SMOTE()),
        ("rf", rf_classifier),
    ])

    # Params stat with rf__ to indicate they apply to random forest and not smote
    param_distributions = {
        "rf__n_estimators": randint(200, 400),
        "rf__max_depth": list(range(5, 41, 5)),
        "rf__min_samples_split": randint(2, 20),
        "rf__min_samples_leaf": randint(1, 10),
        "rf__max_features": ['sqrt', 'log2'],
        "rf__bootstrap": [True, False],
        "rf__criterion": ["gini"],
        "rf__class_weight": [None, "balanced", "balanced_subsample"]
    }

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=40,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=36),
        scoring=f1_macro_scorer,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    # Run the search
    random_search.fit(X_train, y_train)

    # Return the best model
    return random_search.best_estimator_


# Generate & Save a report about a fitted classifier
# Statistics such as test/train accuracy & ROC curve
def generate_model_analysis_report(Xtrain: np.ndarray,
                                   ytrain: np.ndarray,
                                   Xtest: np.ndarray,
                                   ytest: np.ndarray,
                                   already_fitted_clf: RandomForestClassifier, # ignore "unresolved attribute for class BaseEstimator" warnings
                                   directory: str,
                                   classes: Dict[int, str],
                                   model_selection: BERT_MODEL):
    # ---- Generate & Save ROC Chart to Directory ----
    # Get ROC/AUC and Plot It
    # Find the numeric label for “humour”
    positive_label = [k for k, v in classes.items() if v == "humour"][0]

    # Find which column that label corresponds to in predict_proba
    pos_idx = list(already_fitted_clf.classes_).index(positive_label)

    # Calculate ROC with correct index
    y_pred = already_fitted_clf.predict_proba(Xtest)[:, pos_idx]
    fpr, tpr, _ = roc_curve(ytest, y_pred, pos_label=positive_label) # force positive label manually
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure(figsize=(6, 6))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name="RandomForest").plot()
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title(f"ROC Curve (Random Forest Model)\nBig Bang Theory Humour Classification\nEmbeddings Model: {model_selection.name}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Save ROC to file
    plot_filepath = os.path.join(directory, "roc_curve_random_forest.png")
    plt.savefig(plot_filepath)
    plt.close()
    print(f"Saved ROC curve to: {plot_filepath}")

    # ---- Statistics on Test/Training Set Distribution ----
    # Statistics on Class Distribution (Training Set)
    train_humour_class_count = np.sum(ytrain == c.GT_HUMOUR)
    train_non_humour_class_count = np.sum(ytrain == c.GT_NON_HUMOUR)
    training_total_classes = train_humour_class_count + train_non_humour_class_count
    training_humour_class_percent = round((train_humour_class_count / training_total_classes) * 100,1)
    training_non_humour_class_percent = round((train_non_humour_class_count / training_total_classes) * 100,1)

    # Statistics on Class Distribution (Test Set)
    test_humour_class_count = np.sum(ytest == c.GT_HUMOUR)
    test_non_humour_class_count = np.sum(ytest == c.GT_NON_HUMOUR)
    test_total_classes = test_humour_class_count + test_non_humour_class_count
    test_humour_class_percent = round((test_humour_class_count / test_total_classes) * 100, 1)
    test_non_humour_class_percent = round((test_non_humour_class_count / test_total_classes) * 100,1)

    # ---- Generate & Save AUPRC Chart to Directory ----
    # Calculate AUPRC
    auprc = average_precision_score(ytest, y_pred)

    # Plot Precision-Recall Curve
    plt.figure(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(ytest, y_pred)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title(f"Precision-Recall Curve (Random Forest Model)\nBig Bang Theory Humour Classification\nEmbeddings Model: {model_selection.name}", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)

    # Note: Since we are proportionally balance classes between test & training we can use either one here
    plt.annotate(
        f"Humor Class Prevalence: {test_humour_class_percent}%\n"
        f"Non-Humor Class Prevalence: {test_non_humour_class_percent}%",
        xy=(0.5, -0.20),
        xycoords="axes fraction",
        ha="center",
        fontsize=8
    )

    plt.tight_layout()

    # Save ROC to file
    plot_filepath = os.path.join(directory, "pr_curve_random_forest.png")
    plt.savefig(plot_filepath)
    plt.close()
    print(f"Saved PR curve to: {plot_filepath}")

    # ---- Generate & Save Report to Directory ----

    # Make directory & path to store final report
    report_path = os.path.join(directory, "random_forest_model_report.txt")

    # Training accuracy
    y_train_pred = already_fitted_clf.predict(Xtrain)
    train_acc = accuracy_score(ytrain, y_train_pred)

    # Test accuracy
    y_test_pred = already_fitted_clf.predict(Xtest)
    test_acc = accuracy_score(ytest, y_test_pred)

    # Classification Report infers label order, this forces the order to be correct
    labels = sorted(classes.keys())
    target_names = [classes[l] for l in labels]

    # Generate Classification Report
    report_str = classification_report(
        ytest,
        y_test_pred,
        labels=labels,
        target_names=target_names)

    # Generate Confusion Matrix
    label_order = sorted(classes.keys()) # force ascending label order
    conf_matrix = confusion_matrix(ytest, y_test_pred, labels=label_order)
    display_names = [classes[l] for l in label_order] # force class names in the same order as keys
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                  display_labels=display_names)
    disp.plot(cmap="Blues")
    plt.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    matrix_filepath = os.path.join(directory, "confusion_matrix.png")
    plt.savefig(matrix_filepath)
    plt.close()
    print(f"Saved Confusion Matrix to: {plot_filepath}")

    # Build .txt file to save final report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("----- Random Forest Model Report -----\n")
        f.write("======================================\n\n")
        f.write(f"Embedding Used: {model_selection.name}\n")

        f.write("--------- Performance Metrics --------\n")
        f.write(f"Training accuracy: {train_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Random Forest AUC (Test Set): {roc_auc:.3f}\n")
        f.write(f"Random Forest AUPRC (Test Set): {auprc:.3f}\n\n")
        f.write(f"Classification Report: \n{report_str}\n")
        f.write("----------- General Metrics ----------\n")
        f.write(f"Train Set Instances of Humor Class: {train_humour_class_count} ({training_humour_class_percent})\n")
        f.write(f"Train Set Instances of Non-Humor Class: {train_non_humour_class_count} ({training_non_humour_class_percent})\n")
        f.write(f"Test Set Instances of Humor Class: {test_humour_class_count} ({test_humour_class_percent})\n")
        f.write(f"Test Set Instances of Non-Humor Class: {test_non_humour_class_count} ({test_non_humour_class_percent})\n")

# run random search and save best result to file
def perform_random_param_search(Xtrain: np.ndarray,
                                ytrain: np.ndarray,
                                Xtest: np.ndarray,
                                ytest: np.ndarray,
                                model_selection: BERT_MODEL,
                                directory: str,
                                classes: Dict[int, str]) -> RandomForestClassifier:
    print("----- BEGIN Randomized Search CV -----")

    # Make the bert_embeddings directory to store model & report
    output_dir = directory + "/random_search_model/"
    os.makedirs(output_dir, exist_ok=True)

    # Run random search for best configuration of parameters
    clf = run_random_param_search(Xtrain,
                                  ytrain)

    # Save best model to file
    joblib.dump(clf, f'{output_dir}/random_forest_model.joblib')

    # Generate Report about our best model found with Random Search
    os.makedirs(output_dir, exist_ok=True)

    # Generate report on model performance
    generate_model_analysis_report(Xtrain,
                                   ytrain,
                                   Xtest,
                                   ytest,
                                   clf,
                                   output_dir,
                                   classes,
                                   model_selection=model_selection)

    print("----- END Randomized Search CV -----")
    return clf


# Create & Save various Random Forest Models
def create_new_trained_models(run_k_fold_validation: bool,
                              run_new_simple_rf_classifier: bool,
                              run_random_param_search: bool,
                              use_smote: bool,
                              model_selection: BERT_MODEL,
                              split_dialog_by_episode: bool = True):

    directory_to_save_models = (
            c.RANDOM_FOREST_TRAINED_MODEL_DIR_PREFIX + "sandbox/" + generate_run_dir_name(model_selection, use_smote)
    )

    os.makedirs(directory_to_save_models, exist_ok=True)

    # Read features in from file
    X_features, y_labels = read_features(model_selection)

    if split_dialog_by_episode:
        Xtrain, Ytrain, Xtest, Ytest = corpus_loader.corpus_test_train_split_by_episode(test_size = 0.3,
                                                                                        model_selection=model_selection)
    else:
        # Split into train/test
        # ordered differently since this is the order train_test_split returns them
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_features,
                                                        y_labels,
                                                        test_size=0.3,
                                                        stratify=y_labels)

    # Perform k-fold validation random forest
    if run_k_fold_validation:
        perform_k_fold_randomforest(X_features,
                                    y_labels,
                                    model_selection=model_selection,
                                    report_directory=directory_to_save_models)

    # Fit Random Forest Model
    if run_new_simple_rf_classifier:
        clf = train_randomforest(Xtrain,
                                 Ytrain,
                                 Xtest,
                                 Ytest,
                                 model_selection = model_selection,
                                 classes=c.CLASSES,
                                 report_directory=directory_to_save_models)

    # Perform Random Search
    if run_random_param_search:
        perform_random_param_search(Xtrain,
                                    Ytrain,
                                    Xtest,
                                    Ytest,
                                    model_selection=model_selection,
                                    directory=directory_to_save_models,
                                    classes=c.CLASSES)



if __name__ == "__main__":

    # Create trained model with Sentence Bert embeddings and SMOTE oversampling
    create_new_trained_models(run_k_fold_validation=True,
                              run_new_simple_rf_classifier=True,
                              run_random_param_search=True,
                              use_smote=True,
                              model_selection=BERT_MODEL.S_BERT,
                              split_dialog_by_episode=False)


    # Create trained model with Bert embeddings and SMOTE oversampling
    create_new_trained_models(run_k_fold_validation=True,
                              run_new_simple_rf_classifier=True,
                              run_random_param_search=True,
                              use_smote=True,
                              model_selection=BERT_MODEL.BERT,
                              split_dialog_by_episode=False)