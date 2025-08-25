#%% Imports
import os
import sys

### we need this to both print and save logs
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

def setup_results_directory_and_logging(config):
    """Setup results directory and logging based on config."""
    results_dir = config["results_feat_sel"]["dir_path"]
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging to the results directory
    log_file_path = os.path.join(results_dir, "run_log.txt")
    sys.stdout = Tee(log_file_path)
    sys.stderr = sys.stdout  # send stderr to same "tee"
    
    print("This will print on terminal AND be logged in file.")
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")
    
    return results_dir



import yaml
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

# Append utils path and import custom modules
sys.path.append('./Methods_utils')
import Methods_utils.methods as custom
import Methods_utils.methods_heatmap as heatmap


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration from given path."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_data(data_path: str, config: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list]:
    """
    Load and clean data according to config.

    Returns:
        x (pd.DataFrame): Feature data
        y (pd.Series): Target variable
        data (pd.DataFrame): Raw loaded data
        feature_names (list): List of feature names
    """
    print(config["misc"]["entering_method_print"] + "Entering the method to read the data.")

    data = pd.read_csv(data_path, encoding='latin-1', sep=config["data_aspects"]["original_data_delimiter"])
    print(f"--- All columns of the read data are: \n {data.columns.values}")

    x = data[config["data_aspects"]["data_columns"]].copy()

    if x.isna().any().any():
        print("!!! Missing values found!!!")
        print("\nMissing value counts per column:")
        print(x.isna().sum())
        print("\nRows and columns with missing data:")
        print(x[x.isna().any(axis=1)])
    else:
        print("No missing values found.")
    print()

    y = data[config["data_aspects"]["target_feature"]]
    feature_names = x.columns.tolist()
    print(f"--- Working with the following {len(feature_names)} features:\n {feature_names}")

    return x, y, data, feature_names


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, exclude_prefix: str = "c_") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale numerical features excluding columns starting with exclude_prefix.

    Args:
      X_train: Training features dataframe.
      X_test: Testing features dataframe.
      exclude_prefix: Prefix of columns to exclude from scaling.

    Returns:
      Tuple of scaled training and testing DataFrames.
    """
    scaler = StandardScaler()
    columns_to_scale = [col for col in X_train.columns if not col.startswith(exclude_prefix)]

    # Create copies to preserve original dataframes
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train_scaled, X_test_scaled


def pt10_method(cv_nr: int, shap_folds: list, rf_folds: list, xgb_folds: list, ridge_folds: list,
                logistic_folds: list, X_pool_orig: pd.DataFrame, y_pool_orig: pd.Series,
                experim: str, config: dict, results_dir: str, wish_toPlot_AUROC: bool = False, wish_toPlot_AUPRC: bool = False) -> None:
    """
    Process the top 10 features across folds and retrain models, save results and optionally plot.

    Args:
        cv_nr: Cross-validation fold number
        shap_folds, rf_folds, xgb_folds, ridge_folds, logistic_folds: Lists of selected features per fold per method
        X_pool_orig: Original feature dataframe (imbalanced)
        y_pool_orig: Original target series (imbalanced)
        experim: Experiment name string
        config: Configuration dictionary
        results_dir: Directory to save results
        wish_toPlot_AUROC, wish_toPlot_AUPRC: Flags to plot performance curves.
    """
    print(config["misc"]["entering_method_print"] + "Entering PT10 method.")

    if config["misc"].get("PT10_extras", False):
        for fold_name, fold_data in zip(
            ['shap', 'rf', 'xgb', 'ridge', 'logistic'],
            [shap_folds, rf_folds, xgb_folds, ridge_folds, logistic_folds]
        ):
            print(f"{experim}{cv_nr}{fold_name}_folds")
            heatmap.heatmap_oneFeatureSelectionCV(fold_data, f"{experim}{cv_nr}{fold_name}_folds")

    save_name = f"{experim}{cv_nr}"
    print(save_name)

    top10_across_folds = heatmap.original_heatmap(save_name, shap_folds, rf_folds, xgb_folds, ridge_folds, logistic_folds)
    print(f"--- PT10: \n{top10_across_folds}")
    print("#" * 120)

    # Retrain using top 10 features
    X_pool = X_pool_orig[top10_across_folds].copy()
    X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(
        X_pool, y_pool_orig, stratify=y_pool_orig, test_size=0.2, random_state=cv_nr - 1
    )

    X_train_scaled, X_test_scaled = scale_features(X_train_unscaled, X_test_unscaled)

    X_train = X_train_scaled.to_numpy()
    X_test = X_test_scaled.to_numpy()

    # Initialize performance dictionaries
    auc_dict_new = {model: [] for model in ['dummy_majority', 'dummy_minority', 'rf', 'svm', 'xgb', 'ridge', 'logistic']}
    auprc_dict_new = auc_dict_new.copy()

    # Train models and collect metrics
    results = {
        "dummy_majority": custom.dummy_clf_majority0,
        "dummy_minority": custom.dummy_clf_minority1,
        "rf": custom.random_forest,
        "svm": custom.svm,
        "xgb": custom.xgboost_clf,
        "ridge": custom.ridge,
        "logistic": custom.logistic,
    }

    model_metrics = {}  # to hold full returned metrics for plotting if needed

    for model_name, func in results.items():
        metrics = func(X_train, y_train, X_test, y_test)
        model_metrics[model_name] = metrics
        auc_dict_new[model_name].append(metrics[1])    # auc is second returned value
        auprc_dict_new[model_name].append(metrics[4])  # auprc is fifth returned value

    # Save results in dataframe
    featureSel_andPerformance_top10 = pd.DataFrame()

    iteration = cv_nr - 1
    feature_selection_str = "top_10_acrossfold"
    selected_features_list = [top10_across_folds]

    data_rows = []
    for model_name in results.keys():
        data_rows.append({
            'Iteration': iteration,
            'Stage': cv_nr,
            'Current Feature Selection': feature_selection_str,
            'Selected Features': selected_features_list,
            'Model': model_name,
            'Test AUROC': auc_dict_new[model_name][-1],
            'Test AUPRC': auprc_dict_new[model_name][-1],
        })

    featureSel_andPerformance_top10 = pd.DataFrame(data_rows)
    
    # Create full path for PT10 results using results directory
    pt10_file_path = os.path.join(results_dir, os.path.basename(config["results_feat_sel"]["PT10_res"]))
    featureSel_andPerformance_top10.to_csv(pt10_file_path, index=False)

    # Plotting AUROC if requested
    if wish_toPlot_AUROC:
        fprs = [model_metrics[m][2] for m in results.keys()]
        tprs = [model_metrics[m][3] for m in results.keys()]
        aucs = [model_metrics[m][1] for m in results.keys()]
        model_labels = ['Dummy_majority', 'Dummy_minority', 'RF', 'SVM', 'XGBoost', 'Ridge', 'Logistic']

        custom.plot_auc_models(fprs, tprs, aucs, model_labels, f"{experim}{iteration}final_stratif")

    # Plotting AUPRC if requested
    if wish_toPlot_AUPRC:
        recalls = [model_metrics[m][6] for m in results.keys()]
        precisions = [model_metrics[m][5] for m in results.keys()]
        auprcs = [model_metrics[m][4] for m in results.keys()]
        model_labels = ['Dummy_majority', 'Dummy_minority', 'RF', 'SVM', 'XGBoost', 'Ridge', 'Logistic']

        custom.plot_auprc_models(recalls, precisions, auprcs, model_labels, f"{experim}{cv_nr}{iteration}final_stratif")


def train_feat_sel_heatmap_top10(cv_nr: int, config: dict,
                                extras: bool = False,
                                wish_toPlot_AUROC: bool = False,
                                wish_toPlot_AUPRC: bool = False) -> None:
    """
    Main training function that performs stratified K-fold CV, feature selection,
    model training, evaluation and results saving.
    """
    
    # Setup results directory and logging
    results_dir = setup_results_directory_and_logging(config)

    experim = f"_pipeline_{cv_nr}_"

    # DataFrames to hold results
    featureSel_andPerformance = pd.DataFrame(
        columns=['Iteration', 'Stage', 'Current Feature Selection', 'Selected Features', 'Model', 'Test AUROC', 'Test AUPRC']
    )
    featureSel_andPerformance_CV = pd.DataFrame(
        columns=['SplitNo', 'Iteration', 'Stage', 'Current Feature Selection', 'Selected Features', 'Model', 'Test AUROC', 'Test AUPRC']
    )
    featureSel_andPerformance_top10 = pd.DataFrame(
        columns=['Iteration', 'Stage', 'Current Feature Selection', 'Selected Features', 'Model', 'Test AUROC', 'Test AUPRC']
    )

    crt_feat_sel_options = ['none', 'lasso', 'shap', 'rf', 'xgb', 'ridge', 'logistic']

    # Pre-filled LASSO features from your code (should be loaded or defined more generally)
    features_imp_lasso = [
        'c_gender', 'c_vor_alko', 'c_mechventil', 'c_picco',
        'o_sofa_resp', 'o_sofa_liver', 'n_alter', 'n_bdmit',
        'n_bdsys', 'n_balance', 'n_laktat', 'n_ptt', 'n_ery',
        'o_sofa_cardio', 'o_sofa_liver', 'n_thrombo', 'n_crp',
        'n_crp', 'n_sofa_total', 'n_meanlambda', 'n_delta', 'n_c'
    ]

    shap_folds = []
    rf_folds = []
    xgb_folds = []
    ridge_folds = []
    logistic_folds = []

    allAUROCs = pd.DataFrame(columns=['Iteration', 'Stage', 'Model name', 'AUROC', 'TPR', 'FPR'])

    # Load CV data (already split from hold-out)
    print("="*80)
    print("LOADING PRE-SPLIT CV DATA (90% of original)")
    print("="*80)
    x, y, data, feature_names = get_data(config["data_paths"]["cv_data_path"], config)
    
    # Use all CV data for cross-validation (no further splitting needed)
    X_pool_orig_imbalanced = x
    y_pool_orig_imbalanced = y
    
    print("Cases and controls in CV data: \n", y_pool_orig_imbalanced.value_counts())
    print(f"CV data shape: {X_pool_orig_imbalanced.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Hold-out data location: {config['data_paths']['holdout_data_path']}")

    skf = StratifiedKFold(n_splits=cv_nr, shuffle=True, random_state=42)

    # Map of feature importances; initially empty except lasso preset
    features_imp_rf = []
    features_imp_xgb = []
    features_imp_ridge = []
    features_imp_logistic = []
    features_imp_shap = features_imp_lasso.copy()  # start with LASSO for shap placeholder

    # Model functions dictionary (same as in PT10 method)
    model_funcs = {
        'dummy_majority': custom.dummy_clf_majority0,
        'dummy_minority': custom.dummy_clf_minority1,
        'rf': custom.random_forest,
        'svm': custom.svm,
        'xgb': custom.xgboost_clf,
        'ridge': custom.ridge,
        'logistic': custom.logistic,
    }
    ml_models = list(model_funcs.keys())

    for iteration_x, (train_index, test_index) in enumerate(skf.split(X_pool_orig_imbalanced, y_pool_orig_imbalanced), 1):
        print(f"-------------- Started working on fold {iteration_x} --------------")
        for stage_cnt in range(len(crt_feat_sel_options)):
            print(f"\nCurrently working on fold {iteration_x}, feature selection stage {stage_cnt}...")

            # Select features according to stage
            if stage_cnt == 0:
                X_pool = X_pool_orig_imbalanced
            elif stage_cnt == 1:
                X_pool = X_pool_orig_imbalanced[features_imp_lasso].copy()
            elif stage_cnt == 2:
                X_pool = X_pool_orig_imbalanced[features_imp_shap].copy()
            elif stage_cnt == 3:
                X_pool = X_pool_orig_imbalanced[features_imp_rf].copy()
            elif stage_cnt == 4:
                X_pool = X_pool_orig_imbalanced[features_imp_xgb].copy()
            elif stage_cnt == 5:
                X_pool = X_pool_orig_imbalanced[features_imp_ridge].copy()
            else:  # stage_cnt == 6
                X_pool = X_pool_orig_imbalanced[features_imp_logistic].copy()

            # Split using indices
            X_train_unscaled, X_test_unscaled = X_pool.iloc[train_index], X_pool.iloc[test_index]
            y_train, y_test = y_pool_orig_imbalanced.iloc[train_index], y_pool_orig_imbalanced.iloc[test_index]

            # Scaling
            X_train_scaled, X_test_scaled = scale_features(X_train_unscaled, X_test_unscaled)

            X_train, X_test = X_train_scaled.to_numpy(), X_test_scaled.to_numpy()

            # Train models and collect metrics
            auc_dict = {m: [] for m in ml_models}
            auprc_dict = {m: [] for m in ml_models}
            model_metrics_stage = {}

            # we take the model name and the actual function for the model
            """ The function actually returns:
            Index	Meaning	             From Function
            0	the trained rf model	      rf
            1	AUROC	                  auc_rf
            2	FPR list	              fpr_rf
            3	TPR list	              tpr_rf
            4	AUPRC	                auprc_rf
            5	Precision list	    precision_rf
            6	Recall list	           recall_rf
            """
            for model_name, func in model_funcs.items():
                metrics = func(X_train, y_train, X_test, y_test)
                model_metrics_stage[model_name] = metrics
                auc_dict[model_name].append(metrics[1])
                auprc_dict[model_name].append(metrics[4])

                new_row = {
                    'Iteration': iteration_x,
                    'Stage': stage_cnt,
                    'Model name': model_name,
                    'AUROC': metrics[1],
                    'TPR': metrics[3],
                    'FPR': metrics[2],
                }

                allAUROCs = pd.concat([allAUROCs, pd.DataFrame([new_row])], ignore_index=True)

            # Compute feature importance only in stage 0
            if stage_cnt == 0:
                features_imp_rf = custom.feat_imp_rf(model_metrics_stage['rf'][0], feature_names)
                features_imp_xgb = custom.feat_imp_xgb(model_metrics_stage['xgb'][0], feature_names)
                features_imp_ridge = custom.feat_imp_ridge(model_metrics_stage['ridge'][0], feature_names)
                features_imp_logistic = custom.feat_imp_logistic(model_metrics_stage['logistic'][0], feature_names)

                if extras:
                    print(f"SHAP used in iteration: {iteration_x}")
                    list_allMLmodels = ml_models
                    models_auprc_list = [auprc_dict[m][0] for m in ml_models]
                    models_list = [model_metrics_stage[m][0] for m in ml_models]

                    max_auprc = max(models_auprc_list)
                    max_model_index = models_auprc_list.index(max_auprc)
                    shap_model = models_list[max_model_index]

                    shap_kind_map = {
                        'rf': 'rf',
                        'svm': 'svm',
                        'xgb': 'xgb',
                        'ridge': 'linear',
                        'logistic': 'linear'
                    }
                    shap_kind = shap_kind_map.get(list_allMLmodels[max_model_index], '')

                    print(f"HIGHEST AUPRC MODEL: {list_allMLmodels[max_model_index]}")
                    print(shap_kind)
                    features_imp_shap = custom.feat_imp_shap(shap_model, feature_names, shap_kind, X_train, X_test)

                shap_folds.append(features_imp_shap)
                rf_folds.append(features_imp_rf)
                xgb_folds.append(features_imp_xgb)
                ridge_folds.append(features_imp_ridge)
                logistic_folds.append(features_imp_logistic)

            # Save results into DataFrames
            current_feat_sel = crt_feat_sel_options[stage_cnt]
            selected_features_by_stage = [
                ['All'], features_imp_lasso, features_imp_shap, features_imp_rf,
                features_imp_xgb, features_imp_ridge, features_imp_logistic
            ]

            iteration_val = iteration_x

            # Accumulate info for saving
            df_entries = []
            df_entries_cv = []

            for model_idx, model_name in enumerate(ml_models):
                entry = {
                    'Iteration': iteration_val,
                    'Stage': stage_cnt,
                    'Current Feature Selection': current_feat_sel,
                    'Selected Features': [selected_features_by_stage[stage_cnt]],
                    'Model': model_name,
                    'Test AUROC': auc_dict[model_name][-1],
                    'Test AUPRC': auprc_dict[model_name][-1]
                }
                df_entries.append(entry)

                entry_cv = entry.copy()
                entry_cv['SplitNo'] = iteration_x
                df_entries_cv.append(entry_cv)

            if extras:
                featureSel_andPerformance = pd.concat(
                    [featureSel_andPerformance, pd.DataFrame(df_entries)],
                    ignore_index=True
                )
            featureSel_andPerformance_CV = pd.concat(
                [featureSel_andPerformance_CV, pd.DataFrame(df_entries_cv)],
                ignore_index=True
            )

            print(experim)
            # Plot AUROC curves if requested
            if wish_toPlot_AUROC:
                fprs = [model_metrics_stage[m][2] for m in ml_models]
                tprs = [model_metrics_stage[m][3] for m in ml_models]
                aucs = [model_metrics_stage[m][1] for m in ml_models]

                custom.plot_auc_models(fprs, tprs, aucs,
                                       ['Dummy_majority', 'Dummy_minority', 'RF', 'SVM', 'XGBoost', 'Ridge', 'Logistic'],
                                       f"{experim}{stage_cnt}final_stratif")

            # Plot AUPRC curves if requested
            if wish_toPlot_AUPRC:
                recalls = [model_metrics_stage[m][6] for m in ml_models]
                precisions = [model_metrics_stage[m][5] for m in ml_models]
                auprcs = [model_metrics_stage[m][4] for m in ml_models]

                custom.plot_auprc_models(recalls, precisions, auprcs,
                                        ['Dummy_majority', 'Dummy_minority', 'RF', 'SVM', 'XGBoost', 'Ridge', 'Logistic'],
                                        f"{experim}{stage_cnt}{iteration_x}final_stratif")

        print("Cases and controls hold-out data: ", y_test.value_counts())

        if extras:
            featureSel_andPerformance.to_csv(os.path.join(results_dir, f'results{experim}_split_{iteration_x}.csv'), index=False, sep="~")

    # Save combined CV results
    featureSel_andPerformance_CV.to_csv(os.path.join(results_dir, f'resultsAllCVs{experim}_split_{iteration_x}.csv'), index=False, sep="~")

    # Save all AUROCs for external plotting
    allAUROCs['TPR'] = allAUROCs['TPR'].apply(lambda x: ','.join(map(str, x)))
    allAUROCs['FPR'] = allAUROCs['FPR'].apply(lambda x: ','.join(map(str, x)))
    allAUROCs.to_csv(os.path.join(results_dir, f'allAUROCs{experim}.csv'), index=False, sep="~")

    # Run PT10 final step
    pt10_method(cv_nr, shap_folds, rf_folds, xgb_folds, ridge_folds, logistic_folds,
                X_pool_orig_imbalanced, y_pool_orig_imbalanced, experim,
                config, results_dir, wish_toPlot_AUROC, wish_toPlot_AUPRC)


def feature_selection_and_predictions(cv_nr: int, extras: bool, wish_toPlot_AUROC: bool, wish_toPlot_AUPRC: bool) -> None:
    """
    Entry point to run full training and evaluation pipeline.
    """
    config = load_config()
    train_feat_sel_heatmap_top10(cv_nr, config, extras, wish_toPlot_AUROC, wish_toPlot_AUPRC)


if __name__ == '__main__':
    feature_selection_and_predictions(10, extras=True, wish_toPlot_AUROC=False, wish_toPlot_AUPRC=False)