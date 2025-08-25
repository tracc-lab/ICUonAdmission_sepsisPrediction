import os
import sys
import ast
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff() # this is just for VS Code so it doesn't wait to manually close a plot before moving on to the next

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
)
from imblearn.under_sampling import RandomUnderSampler

import Methods_utils.methods_cm_time as custom_cm
import Methods_utils.methods as custom

def extract_inner_list(s):
    try:
        parsed = ast.literal_eval(s)
        # If it's a list of a single list, return the single list
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
            return parsed[0]
        return parsed  # fallback: if it's a flat list already
    except Exception:
        return []

# Tee class for logging (prints to file and terminal)
class Tee:
    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration from a file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def metrics_model(y_test, probabilities, predictions, model):
    """Compute and print common metrics."""
    print(f"probs: {probabilities}")
    precision, recall, thresh = precision_recall_curve(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    auprc = average_precision_score(y_test, probabilities)
    print(f"Precision for {model}: {precision}")
    print(f"Recall for {model}: {recall}")
    print(f"Threshold for PR for {model}: {thresh}")
    print(f"AUC for {model}: {auc}")
    print(f"AUPRC for {model}: {auprc}")
    return auc, fpr, tpr, auprc, precision, recall

def heatmap_featureSelection(data_heatmap):
    """Get top features from the last row of heatmap results."""
    return data_heatmap['Selected Features'].iloc[-1]

def cv_featureSelection(data_CV):
    """Get best features (excluding LASSO and NONE) from cross-validation table based on AUPRC."""
    best_row = data_CV.loc[data_CV['Test AUPRC'].idxmax()]
    print("--------Looking at CV -------------------------------------------")
    print("Best AUPRC among the folds:", best_row['Test AUPRC'])
    print("Corresponding Features:", best_row['Selected Features'])
    print("Feature Selection Method:", best_row['Current Feature Selection'])
    print("Model:", best_row['Model'])
    # Remove 'lasso' and 'none' feature selection methods before further analysis
    filtered = data_CV[
        (data_CV['Current Feature Selection'] != 'lasso') &
        (data_CV['Current Feature Selection'] != 'none')
    ]
    best_row_noLasso = filtered.loc[filtered['Test AUPRC'].idxmax()]
    print("CV10: Best AUPRC (excluding LASSO):", best_row_noLasso['Test AUPRC'])
    print("Features:", best_row_noLasso['Selected Features'])
    return best_row_noLasso['Selected Features']

def averages_AUROC(data_CV, config):
    """Get best features by AUROC, excluding LASSO and handling ['All'] placeholder."""
    best_row = data_CV.loc[data_CV['Test AUROC'].idxmax()]
    print("--------Choosing the best features from CV -------------------------------------------")
    print("Best AUROC among the folds:", best_row['Test AUROC'])
    print("Corresponding Features:", best_row['Selected Features'])
    print("Feature Selection Method:", best_row['Current Feature Selection'])
    print("Model:", best_row['Model'])

    filtered = data_CV[data_CV['Current Feature Selection'] != 'lasso']
    best_row_noLasso = filtered.loc[filtered['Test AUROC'].idxmax()]
    if 'All' in best_row_noLasso['Selected Features']:
        best_features = config["data_aspects"]["data_columns"]
    else:
        best_features = best_row_noLasso['Selected Features']
    print("Best AUROC (excluding LASSO):", best_row_noLasso['Test AUROC'])
    print("Corresponding Features:", best_features)
    return best_features

def cv_bestAverageModel(data_CV):
    """Find model with best mean AUPRC across folds."""
    print("--------Choosing the best performing model on average -------------------------------------------")
    model_avg = data_CV.groupby('Model')['Test AUPRC'].mean()
    best_model = model_avg.idxmax()
    print("Average Performance for each Model across folds:")
    print(model_avg)
    print(f"Best Model: {best_model}, AUPRC: {model_avg.max()*100:.2f}")
    return best_model

def cv_bestAverageModel_AUROC_Table(data_CV):
    """Find model with best mean AUROC across folds."""
    print("--------Choosing the best performing model on average (AUROC) -------------------------------------------")
    model_avg = data_CV.groupby('Model')['Test AUROC'].mean()
    best_model = model_avg.idxmax()
    print("Average Performance for each Model (AUROC):")
    print(model_avg)
    print(f"Best AUROC Model: {best_model}, AUROC: {model_avg.max()*100:.2f}")
    return best_model

def ml_model_cm(model_name, X_train, y_train, X_test, y_test, iteration, onset_days_arr, plot_number, results_dir):
    """Train, evaluate and return model and metrics using the correct method by model name."""
    name = f"{model_name.capitalize()}_{iteration}"
    if model_name == 'rf':
        out = custom_cm.random_forest
    elif model_name == 'svm':
        out = custom_cm.svm
    elif model_name == 'xgb':
        out = custom_cm.xgboost_clf
    elif model_name == 'ridge':
        out = custom_cm.ridge
    elif model_name == 'logistic':
        out = custom_cm.logistic
    else:
        raise ValueError(f"Model name '{model_name}' not recognized.")
    return out(X_train, y_train, X_test, y_test, True, name, onset_days_arr, plot_number, results_dir)

def plot_ROC(df, title, save_path):
    """Plot ROC curve given DataFrame and save to file."""
    colors = {'PT10': '#630C3A', 'CV10': '#27C3C1', 'All': '#FFC107', 'Baseline': '#7E34F9'}
    plt.figure(figsize=(10, 9))
    plt.rcParams['font.family'] = 'Arial'
    for _, row in df.iterrows():
        method = row['Iteration Counter']
        label = {
            'PT10': 'PT10 Features',
            'CV10': 'CV10 Features',
            'All': 'All Features'
        }.get(method, method)
        plt.plot(row['FPR'], row['TPR'], marker='o', linestyle='-', color=colors.get(method, '#000000'), label=f"{label}: {row['AUROC']:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Baseline: 0.5')

    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    legend = plt.legend(prop={'size': 22}, loc='lower right', frameon=False)
    for text in legend.get_texts():
        parts = text.get_text().split(':')
        if len(parts) > 1:
            text.set_text(f"{parts[0]}: $\\mathbf{{{parts[1]}}}$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plotAUROC_trainAndTest(plot_AUROC_df_train_grouped, plot_AUROC_df_test_grouped, results_directory):
    """Plot AUROC curves for train and test sets, grouped by iteration count."""
    for counter_iter, group_train in plot_AUROC_df_train_grouped:
        group_test = plot_AUROC_df_test_grouped.get_group(counter_iter)
        plot_ROC(group_train, f'ROC Curve for Training Data', f"{results_directory}/{counter_iter}_training_holdout.png")
        plot_ROC(group_test, f'ROC Curve for Testing Data', f"{results_directory}/{counter_iter}_testing_holdout.png")

def plotViolin(plot_info_df, results_directory):
    """Plot and save violin plots of correct prediction percentages."""
    def single_violin(df, label, filename):
        df['Percentage Correct'] = pd.to_numeric(df['Correct'] / df['Total'] * 100, errors='coerce')
        sns.set_palette(sns.color_palette(['#8c95c5', '#4d004b',  '#b6cde2']))
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='time_categories', y='Percentage Correct')
        plt.title(f'Violin Plot - {label}')
        plt.xlabel('Time intervals')
        plt.tick_params(axis='x', length=0)
        plt.ylabel('Percentage Correct Sepsis Predictions')
        plt.savefig(f"{results_directory}/{filename}", dpi=600)
        plt.close()
        df.to_csv(f"{results_directory}/{label}_violinData.csv", index=False)
    single_violin(plot_info_df[plot_info_df['Feature Selection Method'] == 'PT10'], "PT10 Features", "violin_plot_PT10.png")
    single_violin(plot_info_df[plot_info_df['Feature Selection Method'] == 'CV10'], "CV10 Features", "violin_plot_CV10.png")

def extract_model_class_name(model_obj_or_str):
    if isinstance(model_obj_or_str, str):
        # If looks like e.g. "LogisticRegression(...)", strip to "LogisticRegression"
        return model_obj_or_str.split('(')[0].strip()
    elif hasattr(model_obj_or_str, '__class__'):
        return model_obj_or_str.__class__.__name__
    else:
        return str(model_obj_or_str)
    
def print_AUROCandAUPRC_andSTD(results_dict, best_model_name):
    """
    Print summary: Model name, Features (name_of_set), Average AUROC, STD, and likewise for AUPRC.
    """
    # Gather results only for the best model
    indices = [i for i, model in enumerate(results_dict['Model']) if model == best_model_name]
    if not indices:
        print(f"No models matched type '{best_model_name}'")
        return

    # Group AUROC and AUPRC by feature set (tuple-ized to be hashable)
    group_stats = {}
    for i in indices:
        features_set = tuple(results_dict['Features'][i])
        if features_set not in group_stats:
            group_stats[features_set] = {'AUROC': [], 'AUPRC': []}
        group_stats[features_set]['AUROC'].append(results_dict['AUROC'][i])
        group_stats[features_set]['AUPRC'].append(results_dict['AUPRC'][i])

    for features_set, metrics in group_stats.items():
        # Convert tuple to clean, human-readable set name
        features_display = ', '.join(str(f) for f in features_set)
        print(f"Model name: {best_model_name}")
        print(f"Features (name_of_set): {features_display}")
        print(f"Average AUROC: {np.mean(metrics['AUROC']):.4f}, STD: {np.std(metrics['AUROC']):.4f}")
        print(f"Average AUPRC: {np.mean(metrics['AUPRC']):.4f}, STD: {np.std(metrics['AUPRC']):.4f}")
        print("-" * 70)

def getData(data_path, CV_nr, config):
    """Load and return data, onset times, selected features and best-avg model from the specified paths and config."""
    cv_resPath = config['results_feat_sel']['dir_path'] + f'/resultsAllCVs_pipeline_{CV_nr}__split_{CV_nr}.csv'
    data_CV = pd.read_csv(cv_resPath, encoding='latin-1', sep='~', engine='python')

    heatmap_resPath = config["results_feat_sel"]["PT10_res"]
    data_heatmap = pd.read_csv(heatmap_resPath, encoding='latin-1', sep=',')
    
    data = pd.read_csv(data_path, encoding='latin-1', sep='~')
    
    onset_time_path = config["data_aspects"]["onset_data_path"]
    data_onset = pd.read_csv(onset_time_path, encoding='latin-1', sep='~')
    
    heatmap_featSel = heatmap_featureSelection(data_heatmap)
    cv_featSel = cv_featureSelection(data_CV)
    
    best_avg_model = cv_bestAverageModel(data_CV)
    
    featureSelection_options_str = [heatmap_featSel, cv_featSel]
    featureSelection_options = [extract_inner_list(s) for s in featureSelection_options_str]
    featureSelection_options.append(config["data_aspects"]["data_columns"])
    X = data[config["data_aspects"]["data_columns"]].copy()
    y_toSplit = data['event']
    return X, y_toSplit, featureSelection_options, data_onset, best_avg_model

def trainModels_andTest(X, y_toSplit, featureSelection_options, data_onset, best_avg_model, run_number, n_iters, config):
    """Train models and compute/test performance, saving everything to defined output folder."""
    print("--------Starting training and testing on hold-out set -------------------------------------------")

    print(f"DEBUG: n_iters parameter received = {n_iters}")
    results_dir = config["results_holdOut"]["results_folder"]
    os.makedirs(results_dir, exist_ok=True)
    print(f"....Created results directory at {results_dir}...")

    iteration_labels = ['PT10', 'CV10', 'All']
    results_dict = {'Model': [], 'Features': [], 'AUROC': [], 'AUPRC': [], 'Precision': [], 'Recall': []}
    plot_info_df = pd.DataFrame(columns=['Iteration Counter', 'Feature Selection Method', 'time_categories', 'Total', 'Correct', 'Incorrect'])
    plot_AUROC_df_train = pd.DataFrame(columns=['Count', 'Iteration Counter', 'Feature Selection Method', 'AUROC', 'FPR', 'TPR'])
    plot_AUROC_df_test = pd.DataFrame(columns=['Count', 'Iteration Counter', 'Feature Selection Method', 'AUROC', 'FPR', 'TPR'])

    undersample = RandomUnderSampler(sampling_strategy=1)
    X_train_unscaled_imbal, X_test_unscaled_imbal, y_train_imbal, y_test = train_test_split(
        X, y_toSplit, stratify=y_toSplit, test_size=0.1, random_state=1)

    print("Cases and controls hold-out aka test data:\n", y_test.value_counts())
    print("Cases and controls training data:\n", y_train_imbal.value_counts())
    subjects_index_with_sepsis = y_test[y_test == 1].index
    onset_array = data_onset.loc[subjects_index_with_sepsis, 'n_onset_days']

    for counter_iter in range(n_iters):
        print("Entered the for loop for iterations")
        print(f"DEBUG: Starting iteration {counter_iter} of {n_iters-1} (0-indexed)")
        X_train_bal, y_train = undersample.fit_resample(X_train_unscaled_imbal, y_train_imbal)
        # Format iteration counter with leading zeros (01, 02, 03, etc.)
        iteration_suffix = f"{counter_iter:02d}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create the filename prefix - this will be prepended to all plot filenames
        # plot_number = f"run_{run_number:02d}_iter_{iteration_suffix}"
        plot_number = f"it_{iteration_suffix}"
        print(f"Plot number (filename prefix): {plot_number}")
        print(f"Files will be saved in: {results_dir}")
        
        for idx, features in enumerate(featureSelection_options):
            label = iteration_labels[idx] if idx < len(iteration_labels) else str(idx)
            # Scale
            scaler = StandardScaler()
            X_train_unscaled = X_train_bal[features].copy()
            X_test_unscaled = X_test_unscaled_imbal[features].copy()
            columns_to_scale = [col for col in X_train_unscaled.columns if not col.startswith('c_')]
            X_train_scaled = X_train_unscaled.copy()
            X_test_scaled = X_test_unscaled.copy()
            X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train_unscaled[columns_to_scale])
            X_test_scaled[columns_to_scale] = scaler.transform(X_test_unscaled[columns_to_scale])
            X_train = X_train_scaled.to_numpy()
            X_test = X_test_scaled.to_numpy()

            print(f"Cases and controls training data balanced:\n{y_train.value_counts()}")
            model, auroc_model, fpr_model, tpr_model, auprc_model, precision_model, recall_model, plot_info = ml_model_cm(
                best_avg_model, X_train, y_train, X_test, y_test, label, onset_array, plot_number, results_dir)
            results_dict['Model'].append(best_avg_model) #chnaged this, because we are interested in the name, not the whole object from sklearn
            results_dict['Features'].append(features)
            results_dict['AUROC'].append(auroc_model)
            results_dict['AUPRC'].append(auprc_model)
            results_dict['Precision'].append(precision_model)
            results_dict['Recall'].append(recall_model)
            plot_info = plot_info.reset_index()
            plot_info['Iteration Counter'] = counter_iter
            plot_info['Feature Selection Method'] = label
            plot_info_df = pd.concat([plot_info_df, plot_info], ignore_index=True)

            # Test results
            plot_AUROC_df_test = pd.concat([
                plot_AUROC_df_test,
                pd.DataFrame([{
                    'Count': counter_iter,
                    'Iteration Counter': label,
                    'Feature Selection Method': features,
                    'AUROC': auroc_model,
                    'FPR': fpr_model,
                    'TPR': tpr_model
                }])
            ], ignore_index=True)

            # Train results: Use probabilities for classifiers with predict_proba, else decision_function
            if hasattr(model, "predict_proba"):
                model_prob_train = model.predict_proba(X_train)[:, 1]
            elif hasattr(model, "decision_function"):
                model_prob_train = model.decision_function(X_train)
            else:
                model_prob_train = np.zeros(X_train.shape[0])
            predictions_model_train = model.predict(X_train)
            auc_train, fpr_train, tpr_train, auprc_train, precision_train, recall_train = metrics_model(
                y_train, model_prob_train, predictions_model_train, model)
            plot_AUROC_df_train = pd.concat([
                plot_AUROC_df_train,
                pd.DataFrame([{
                    'Count': counter_iter,
                    'Iteration Counter': label,
                    'Feature Selection Method': features,
                    'AUROC': auc_train,
                    'FPR': fpr_train,
                    'TPR': tpr_train
                }])
            ], ignore_index=True)

    # Save & plot
    plotAUROC_trainAndTest(plot_AUROC_df_train.groupby('Count'), plot_AUROC_df_test.groupby('Count'), results_dir)
    plotViolin(plot_info_df, results_dir)
    # THIS LINE IS THE KEY: Always extract class name from best_avg_model for filtering!
    best_model_name = extract_model_class_name(best_avg_model)
    print_AUROCandAUPRC_andSTD(results_dict, best_model_name)
    print(results_dict)

def wrapAdvancedAnalysis(CV_nr, run_number, number_ofIterations):
    config = load_config()
    RES_DIR = config["results_holdOut"]["results_folder"]
    os.makedirs(RES_DIR, exist_ok=True)
    log_file_path = f"{RES_DIR}/run_log_hold-out.txt"
    sys.stdout = Tee(log_file_path)
    sys.stderr = sys.stdout
    sns.set_palette(sns.color_palette(config["misc"]["colors_holdOut"]))
    data_path = config["data_paths"]["original_data_path"]
    X, y_toSplit, featureSelection_options, data_onset, best_avg_model = getData(data_path, CV_nr, config)
    trainModels_andTest(X, y_toSplit, featureSelection_options, data_onset, best_avg_model, run_number, number_ofIterations, config)

if __name__ == "__main__":
    run_number = 11
    wrapAdvancedAnalysis(10, run_number, 3)
