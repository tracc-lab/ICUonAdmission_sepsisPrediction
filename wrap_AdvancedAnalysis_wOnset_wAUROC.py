# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:49:49 2024

@author: aa36
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
import ast 

# import sys
# sys.path.append('../Methods_utils')  # Add the path to the custom folder
# print(sys.path)

import Methods_utils.methods_cm_time as custom_cm
import Methods_utils.methods as custom
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

import os

colors = ['#630C3A', '#27C3C1', '#FFC107', '#7E34F9', '#E01889', '#617111','#fe6100',  '#7d413c',
'#423568', '#5590b4']
sns.set_palette(sns.color_palette(colors))


    
## Def plot correct predictions, incorrect predictions vs onset time
def metrics_model (y_test, probabilities, predictions, model):
    print("probs: ", probabilities)
    precision, recall, thresh = precision_recall_curve(y_test,predictions )
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    
    auc = roc_auc_score(y_test, probabilities)
    auprc = average_precision_score(y_test, probabilities)
    
    print("Precision for ", model, " : ", precision)
    print("Recall for ", model, " : ", recall)
    print("Threshold for PR for ", model, " : ", thresh)

    print("AUC for ", model, " : ", auc)
    print("AUPRC for ", model, " : ", auprc)

    return auc, fpr, tpr, auprc, precision, recall

#%% Top 10 selected features from Hetmap and CV10
def heatmap_featureSelection (data_heatmap):
    heatmap_featSel = data_heatmap['Selected Features'].iloc[-1]
    # print("Top 10 most selected features from the CV, as seen in the heatmap:", heatmap_featSel)
    
    return heatmap_featSel

def cv_featureSelection (data_CV):
    best_auprc_row = data_CV.loc[data_CV['Test AUPRC'].idxmax()]

    best_AUPRC = best_auprc_row['Test AUPRC']
    best_features = best_auprc_row['Selected Features']
    feature_selection_method = best_auprc_row['Current Feature Selection']
    model_best_folds = best_auprc_row['Model']

    print("\n--------Choosing the best features from CV -------------------------------------------")
    print("Best AUPRC among the folds:", best_AUPRC)
    print("Corresponding Features:", best_features)
    print("Feature Selection Method:", feature_selection_method)
    print("Model:", model_best_folds)

    # Filter out rows where the feature selection method is LASSO
    data_CV_filtered = data_CV[data_CV['Current Feature Selection'] != 'lasso']

    # Find the row with the highest AUPRC among the remaining rows
    best_auprc_row_noLasso = data_CV_filtered.loc[data_CV_filtered['Test AUPRC'].idxmax()]

    # Step 4: Retrieve the corresponding features, feature selection method, and model for that row
    best_AUPRC_noLasso = best_auprc_row_noLasso['Test AUPRC']
    #if best_auprc_row_noLasso['Selected Features'] == ['All']:
    # If the best set from CV10 is ['All'], then force the full set of features in here
    if 'All' in best_auprc_row_noLasso['Selected Features']:
        print(":::::::::::::::::::::::::::::::::::::::: CASE 1")
        best_features_noLasso = "['c_gender', 'c_vor_diab', 'c_vor_herz' ,'c_vor_atem' ,'c_vor_alko','c_vor_smok', 'c_vor_kidn' ,'c_vor_canc', 'c_ek', 'c_pct', 'c_mechventil','c_dialyse', 'c_ecmo_pecla', 'c_picco' ,'o_sofa_resp', 'o_sofa_cardio','o_sofa_coag' ,'o_sofa_renal', 'o_sofa_liver','n_alter', 'n_kat', 'n_sapsii','n_bddia' ,'n_bdmit', 'n_bdsys', 'n_herzfr', 'n_temp', 'n_ph', 'n_po2' ,'n_pco2','n_fio2pro' ,'n_sbe', 'n_balance', 'n_laktat', 'n_hb' ,'n_blutz', 'n_calcium','n_kalium' ,'n_leuko' ,'n_thrombo' ,'n_bili', 'n_inr' ,'n_ptt' ,'n_ery', 'n_hct','n_crp', 'n_krea' ,'n_harn' ,'n_sofa_total' ,'n_meanlambda' ,'n_delta', 'n_c']"
    else:
        print(":::::::::::::::::::::::::::::::::::::::: CASE 2")
        best_features_noLasso = best_auprc_row_noLasso['Selected Features']
    feature_selection_method_noLasso = best_auprc_row_noLasso['Current Feature Selection']
    model_noLasso = best_auprc_row_noLasso['Model']

    print("\nBest AUPRC (excluding LASSO):", best_AUPRC_noLasso)
    print("Corresponding Features:", best_features_noLasso)
    print("Feature Selection Method:", feature_selection_method_noLasso)
    print("Model:", model_noLasso)
    
    return best_features_noLasso

def averages_AUROC (data_CV):
    best_auprc_row = data_CV.loc[data_CV['Test AUROC'].idxmax()]

    best_AUPRC = best_auprc_row['Test AUROC']
    best_features = best_auprc_row['Selected Features']
    feature_selection_method = best_auprc_row['Current Feature Selection']
    model_best_folds = best_auprc_row['Model']

    print("\n--------Choosing the best features from CV -------------------------------------------")
    print("Best AUROC among the folds:", best_AUPRC)
    print("Corresponding Features:", best_features)
    print("Feature Selection Method:", feature_selection_method)
    print("Model:", model_best_folds)

    # Filter out rows where the feature selection method is LASSO
    data_CV_filtered = data_CV[data_CV['Current Feature Selection'] != 'lasso']

    # Find the row with the highest AUPRC among the remaining rows
    best_auprc_row_noLasso = data_CV_filtered.loc[data_CV_filtered['Test AUROC'].idxmax()]
    best_AUPRC_noLasso = best_auprc_row_noLasso['Test AUROC']
    # if best_auprc_row_noLasso['Selected Features'] == ['All']:
    # If the best set from CV10 is ['All'], then force the full set of features in here
    if 'All' in best_auprc_row_noLasso['Selected Features']:
        print(":::::::::::::::::::::::::::::::::::::::: CASE !")
        best_features_noLasso = "['c_gender', 'c_vor_diab', 'c_vor_herz' ,'c_vor_atem' ,'c_vor_alko','c_vor_smok', 'c_vor_kidn' ,'c_vor_canc', 'c_ek', 'c_pct', 'c_mechventil','c_dialyse', 'c_ecmo_pecla', 'c_picco' ,'o_sofa_resp', 'o_sofa_cardio','o_sofa_coag' ,'o_sofa_renal', 'o_sofa_liver','n_alter', 'n_kat', 'n_sapsii','n_bddia' ,'n_bdmit', 'n_bdsys', 'n_herzfr', 'n_temp', 'n_ph', 'n_po2' ,'n_pco2','n_fio2pro' ,'n_sbe', 'n_balance', 'n_laktat', 'n_hb' ,'n_blutz', 'n_calcium','n_kalium' ,'n_leuko' ,'n_thrombo' ,'n_bili', 'n_inr' ,'n_ptt' ,'n_ery', 'n_hct','n_crp', 'n_krea' ,'n_harn' ,'n_sofa_total' ,'n_meanlambda' ,'n_delta', 'n_c']"
    else:
        print(":::::::::::::::::::::::::::::::::::::::: CASE 2")
        best_features_noLasso = best_auprc_row_noLasso['Selected Features']
    feature_selection_method_noLasso = best_auprc_row_noLasso['Current Feature Selection']
    model_noLasso = best_auprc_row_noLasso['Model']

    print("\nBest AUROC (excluding LASSO):", best_AUPRC_noLasso)
    print("Corresponding Features:", best_features_noLasso)
    print("Feature Selection Method:", feature_selection_method_noLasso)
    print("Model:", model_noLasso)
    
        
    return best_features_noLasso

#%% Best model name retrieval and ML model
def cv_bestAverageModel (data_CV):
    print("\n--------Choosing the best performing model on average -------------------------------------------")
    # group by model and calculate the mean performance
    average_performance_per_model = data_CV.groupby('Model')['Test AUPRC'].mean()

    # calculate the mean performance across all metrics for each model
    average_performance_per_model = data_CV.groupby('Model')['Test AUPRC'].mean()
    
    # find the best average performing model
    best_avg_model = average_performance_per_model.idxmax()
    best_average_auprc = average_performance_per_model.max()

    print("Average Performance of each Model accross the folds:")
    print(average_performance_per_model)

    print("\nBest Average Performing Model:")
    print("Model:", best_avg_model)
    print("Average Performance:", best_average_auprc)

    print("Average best performing model is: ", best_avg_model, best_average_auprc*100)
    
    return best_avg_model 

def cv_bestAverageModel_AUROC_Table (data_CV):
    print("\n--------Choosing the best performing model on average -------------------------------------------")
    # group by model and calculate the mean performance
    average_performance_per_model = data_CV.groupby('Model')['Test AUROC'].mean()

    # calculate the mean performance across all metrics for each model
    average_performance_per_model = data_CV.groupby('Model')['Test AUROC'].mean()
    
    # find the best average performing model
    best_avg_model = average_performance_per_model.idxmax()
    best_average_auprc = average_performance_per_model.max()

    print("Average Performance of each Model accross the folds:")
    print(average_performance_per_model)

    print("\nBest Average Performing Model:")
    print("Model:", best_avg_model)
    print("Average Performance:", best_average_auprc)

    print("Average best performing model AUROC is: ", best_avg_model, best_average_auprc*100)
    
    return best_avg_model 

def ml_model_cm (model_name, X_train, y_train, X_test, y_test, iteration, onset_days_arr, plot_number):
    
    if model_name == 'rf':
        model, auroc, fpr, tpr, auprc, precision, recall, plot_info = custom_cm.random_forest(X_train, y_train, X_test, y_test, True, "RF_" + iteration, onset_days_arr, plot_number)
    elif model_name == 'svm':
        model, auroc, fpr, tpr, auprc, precision, recall, plot_info = custom_cm.svm(X_train, y_train, X_test, y_test, True, "SVM_" + iteration, onset_days_arr, plot_number)
    elif model_name == 'xgb':
        model, auroc, fpr, tpr, auprc, precision, recall, plot_info = custom_cm.xgboost_clf(X_train, y_train, X_test, y_test, True, "XGB_" + iteration, onset_days_arr, plot_number)
    elif model_name == 'ridge':
        model, auroc, fpr, tpr, auprc, precision, recall, plot_info = custom_cm.ridge(X_train, y_train, X_test, y_test, True, "Ridge_" + iteration, onset_days_arr, plot_number)
    elif model_name == 'logistic':
        model, auroc, fpr, tpr, auprc, precision, recall, plot_info = custom_cm.logistic(X_train, y_train, X_test, y_test, True, "Logistic_" + iteration,onset_days_arr, plot_number)
    else:
        print("ERROR")
        pass
    
    return  model, auroc, fpr, tpr, auprc, precision, recall, plot_info

#%% Plot AUROC train and test, plotViolin, print AUROC and AUPRC and sd

def plot_ROC (df_train_OrTest, title_name, save_name):
    colors = {'HeatmapTop10': '#630C3A', 'cv10_FeatSel': '#27C3C1', 'AllFeatures': '#FFC107', 'Baseline': '#7E34F9'}

    plt.figure(figsize=(10, 9))
    plt.rcParams['font.family'] = 'Arial'
    
# 'HeatmapTop10 Features', 'CV10TopAUPRC Features', 'All Features'
    for _, row in df_train_OrTest.iterrows():
        method = row['Iteration Counter']
        label = ''
  
        if method == 'HeatmapTop10':
            label = 'HeatmapTop10 Features'
        elif method == 'cv10_FeatSel':
            label = 'CV10TopAUPRC Features'
        elif method == 'AllFeatures':
            label = 'All Features'
            
        plt.plot(row['FPR'], row['TPR'], marker='o', linestyle='-', color=colors[method], label=f"{label}: {row['AUROC']:.2f}")

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Baseline: 0.5')

    plt.xlabel('False Positive Rate', fontsize = 18+4)
    plt.ylabel('True Positive Rate', fontsize = 18+4)
    plt.title(title_name, fontsize = 20+4)

    plt.xticks(fontsize=16+4)
    plt.yticks(fontsize=16+4)

    # Increase the size of the text in the legend
    legend = plt.legend(prop={'size': 18+4}, loc='lower right')  # Adjust size and location as needed
      # Adjust size as needed

    for text in legend.get_texts():
        parts = text.get_text().split(':')  # Split text at ":"
        if len(parts) > 1:  # Ensure there is a part after ":"
            text.set_text(f"{parts[0]}: $\\mathbf{{{parts[1]}}}$")  # Set LaTeX format for bold text

    plt.tight_layout()
    plt.savefig( save_name + '_' + str(df_train_OrTest['Count'].unique()) + '.png' , dpi=600)
    plt.show()

def plotAUROC_trainAndTest(plot_AUROC_df_train_grouped, plot_AUROC_df_test_grouped, results_directory):
    # Plot each group separately
    for counter_iter, group_train in plot_AUROC_df_train_grouped:
        print("Counter iter is: ", counter_iter)
        print("Counter iter is: ", group_train)
        
        group_test = plot_AUROC_df_test_grouped.get_group(counter_iter)
        print("Group test AUROC: ", group_test['AUROC'])
        
        title_name_train = f'ROC Curve for Training Data'
        save_name_train = results_directory + str(counter_iter) + '_training_holdout'
        plot_ROC(group_train, title_name_train, save_name_train)
        
        title_name_test = f'ROC Curve for Testing Data'
        save_name_test = results_directory + str(counter_iter) + '_testing_holdout'
        plot_ROC(group_test, title_name_test, save_name_test)

##########################################################################
# Violin plot for the variation of the % of correct predictions          #
# of the best average performing model with HeatmapTop10 and CV feats    #
##########################################################################
def plotViolin (plot_info_df, results_directory):
    heatmap_df = plot_info_df[plot_info_df['Feature Selection Method'] == 'HeatmapTop10']

    # Calculate percentage of correct values for each row
    heatmap_df['Percentage Correct'] = (heatmap_df['Correct'] / heatmap_df['Total']) * 100
    # print("This is heatmap_df: _______________________________", heatmap_df)

    unique_values = heatmap_df['Percentage Correct'].unique()
    print("Unique values in 'Percentage Correct' column:", unique_values)

    heatmap_df['Percentage Correct'] = pd.to_numeric(heatmap_df['Percentage Correct'], errors='coerce')

    colors_violin = ['#8c95c5', '#4d004b',  '#b6cde2']
    sns.set_palette(sns.color_palette(colors_violin))


    # Create violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data = heatmap_df, x = 'time_categories', y = 'Percentage Correct', cmap = colors_violin)
    plt.title('Violin Plot - HeatmapTop10 Features')
    plt.xlabel('Time Categories')
    plt.ylabel('Percentage Correct Sepsis Predictions')
    plt.savefig(results_directory + "violin_plot_heatmap10.png", dpi = 600)
    plt.show()

    heatmap_df.to_csv(results_directory + "heatmap_df_violinData.csv")

    ##### Violin plot CV10
    cv10_df_violin = plot_info_df[plot_info_df['Feature Selection Method'] == 'cv10_FeatSel']

    # Calculate percentage of correct values for each row
    cv10_df_violin['Percentage Correct'] = (cv10_df_violin['Correct'] / cv10_df_violin['Total']) * 100
    unique_values_CV10 = cv10_df_violin['Percentage Correct'].unique()

    cv10_df_violin['Percentage Correct'] = pd.to_numeric(cv10_df_violin['Percentage Correct'], errors='coerce')

    colors_violin = ['#8c95c5', '#4d004b', '#b6cde2']
    sns.set_palette(sns.color_palette(colors_violin))


    # Create violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data = cv10_df_violin, x = 'time_categories', y = 'Percentage Correct', cmap = colors_violin)
    plt.title('Violin Plot - CV10 Features')
    plt.xlabel('Time Categories')
    plt.ylabel('Percentage Correct Sepsis Predictions')
    plt.savefig(results_directory + "violin_plot_CV10.png", dpi = 600)
    plt.show()

    cv10_df_violin.to_csv(results_directory + "cv10_df_violinData.csv")

##########################################################################
# AUROC and AUPRC values with their respective standard deviations after #
# taking into accounts all the iterations (here 20 (0, 19))              #
##########################################################################
def print_AUROCandAUPRC_andSTD (results_dict):
    logistic_results = {k: [] for k in results_dict.keys()}
    for i, model in enumerate(results_dict['Model']):
        if 'LogisticRegression' in str(model):
            for key, value in results_dict.items():
                logistic_results[key].append(value[i])

    # Group by features
    grouped_results = {}
    for model, features, aurocs in zip(logistic_results['Model'], logistic_results['Features'], logistic_results['AUROC']):
        features_tuple = tuple(features)  # Convert list to tuple
        if features_tuple not in grouped_results:
            grouped_results[features_tuple] = []
        grouped_results[features_tuple].append(aurocs)


    # Compute standard deviation for each feature set
    mean_stddev_dict = {}
    for features, aurocs in grouped_results.items():
        mean_stddev_dict[features] = {
            'mean': np.mean(aurocs),
            'stddev': np.std(aurocs)
        }

    # Print mean and standard deviation for each feature set
    print("Mean and standard deviation of AUROC for models containing 'LogisticRegression' grouped by features:")
    for features, values in mean_stddev_dict.items():
        print(f"Features: {features}, Mean AUROC: {values['mean']}, Stddev: {values['stddev']}")
        
        
        
    ### AUPRC mean and stddev
    print("")
    logistic_results = {k: [] for k in results_dict.keys()}
    for i, model in enumerate(results_dict['Model']):
        if 'LogisticRegression' in str(model):
            for key, value in results_dict.items():
                logistic_results[key].append(value[i])

    # Group by features
    grouped_results = {}
    for model, features, aurocs in zip(logistic_results['Model'], logistic_results['Features'], logistic_results['AUPRC']):
        features_tuple = tuple(features)  # Convert list to tuple
        if features_tuple not in grouped_results:
            grouped_results[features_tuple] = []
        grouped_results[features_tuple].append(aurocs)


    # Compute standard deviation for each feature set
    mean_stddev_dict = {}
    for features, aurocs in grouped_results.items():
        mean_stddev_dict[features] = {
            'mean': np.mean(aurocs),
            'stddev': np.std(aurocs)
        }

    # Print mean and standard deviation for each feature set
    print("Mean and standard deviation of AUPRC for models containing 'LogisticRegression' grouped by features:")
    for features, values in mean_stddev_dict.items():
        print(f"Features: {features}, Mean AUPRC: {values['mean']}, Stddev: {values['stddev']}")

#%% Retrieve data and make it usable
def getData(data_path):
    # read the all CVs to extract the avg perf model ^ best score
    cv_resPath = './resultsAllCVs_pipeline_10__split_10.csv' #'C:/Users/aa36.MEDMA/Desktop/ML_paper/Restructured_withConfusionMatrix_Balanced/resultsAllCVs_pipeline_10__split_10.csv'
    data_CV = pd.read_csv(cv_resPath, encoding='latin-1', sep='~')

    # read heatmap cv to get the features
    heatmap_resPath = './final_stratif.csv'
    data_heatmap = pd.read_csv(heatmap_resPath, encoding='latin-1', sep=',')

    data = pd.read_csv(data_path, encoding='latin-1', sep='~')
    # print(data.columns.values)


    onset_time_path = 'C:/Users/aa36.MEDMA/Desktop/ML_paper/fbentriesProgV3.csv'
    data_onset = pd.read_csv(onset_time_path, encoding='latin-1', sep='~')

    heatmap_featSel = heatmap_featureSelection(data_heatmap)
    print("\nTop 10 most selected features from the CV, as seen in the heatmap:", heatmap_featSel)

    cv_featSel = cv_featureSelection (data_CV) #"['c_gender', 'c_vor_diab', 'c_vor_herz' ,'c_vor_atem' ,'c_vor_alko','c_vor_smok', 'c_vor_kidn' ,'c_vor_canc', 'c_ek', 'c_pct', 'c_mechventil','c_dialyse', 'c_ecmo_pecla', 'c_picco' ,'o_sofa_resp', 'o_sofa_cardio','o_sofa_coag' ,'o_sofa_renal', 'o_sofa_liver','n_alter', 'n_kat', 'n_sapsii','n_bddia' ,'n_bdmit', 'n_bdsys', 'n_herzfr', 'n_temp', 'n_ph', 'n_po2' ,'n_pco2','n_fio2pro' ,'n_sbe', 'n_balance', 'n_laktat', 'n_hb' ,'n_blutz', 'n_calcium','n_kalium' ,'n_leuko' ,'n_thrombo' ,'n_bili', 'n_inr' ,'n_ptt' ,'n_ery', 'n_hct','n_crp', 'n_krea' ,'n_harn' ,'n_sofa_total' ,'n_meanlambda' ,'n_delta', 'n_c']"  
        #cv_featureSelection (data_CV)

    print("\nTop 10 most selected features from the CV, based on best AUPRC:", cv_featSel)

    best_avg_model = cv_bestAverageModel (data_CV)

    # aurocs_averages = cv_bestAverageModel_AUROC_Table(data_CV)
    # auprcs_averages_manuscriptTable = cv_bestAverageModel (data_CV)
    ## Although it does not guarantee that the best model will actually do great 
    ## with the features from the best auprc in the folds
    ## still the model "saw" those features at a certain point
    ## and because that was specific to the split which model does best

    #%% Retrain avg_best_model on the 90% data 
    # data split
    data2 = data[['c_gender', 'c_vor_diab', 'c_vor_herz' ,'c_vor_atem' ,'c_vor_alko',
    'c_vor_smok', 'c_vor_kidn' ,'c_vor_canc', 'c_ek', 'c_pct', 'c_mechventil',
    'c_dialyse', 'c_ecmo_pecla', 'c_picco' ,'o_sofa_resp', 'o_sofa_cardio',
    'o_sofa_coag' ,'o_sofa_renal', 'o_sofa_liver','n_alter', 'n_kat', 'n_sapsii',
    'n_bddia' ,'n_bdmit', 'n_bdsys', 'n_herzfr', 'n_temp', 'n_ph', 'n_po2' ,'n_pco2',
    'n_fio2pro' ,'n_sbe', 'n_balance', 'n_laktat', 'n_hb' ,'n_blutz', 'n_calcium',
    'n_kalium' ,'n_leuko' ,'n_thrombo' ,'n_bili', 'n_inr' ,'n_ptt' ,'n_ery', 'n_hct',
    'n_crp', 'n_krea' ,'n_harn' ,'n_sofa_total' ,'n_meanlambda' ,'n_delta', 'n_c']].copy()

    y_toSplit = data['event']
    X = data2

    featureSelection_options_str = [heatmap_featSel, cv_featSel]

    featureSelection_options = [ast.literal_eval(s) for s in featureSelection_options_str]
    print(featureSelection_options)

    featureSelection_options.append(['c_gender', 'c_vor_diab', 'c_vor_herz' ,'c_vor_atem' ,'c_vor_alko',
    'c_vor_smok', 'c_vor_kidn' ,'c_vor_canc', 'c_ek', 'c_pct', 'c_mechventil',
    'c_dialyse', 'c_ecmo_pecla', 'c_picco' ,'o_sofa_resp', 'o_sofa_cardio',
    'o_sofa_coag' ,'o_sofa_renal', 'o_sofa_liver','n_alter', 'n_kat', 'n_sapsii',
    'n_bddia' ,'n_bdmit', 'n_bdsys', 'n_herzfr', 'n_temp', 'n_ph', 'n_po2' ,'n_pco2',
    'n_fio2pro' ,'n_sbe', 'n_balance', 'n_laktat', 'n_hb' ,'n_blutz', 'n_calcium',
    'n_kalium' ,'n_leuko' ,'n_thrombo' ,'n_bili', 'n_inr' ,'n_ptt' ,'n_ery', 'n_hct',
    'n_crp', 'n_krea' ,'n_harn' ,'n_sofa_total' ,'n_meanlambda' ,'n_delta', 'n_c'])

    return X, y_toSplit, featureSelection_options, data_onset, best_avg_model

#%% Train models and use other functions
def trainModels_andTest(X, y_toSplit, featureSelection_options, data_onset, best_avg_model):
    results_dir = "./Results_iterationPlots/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print("....Created results directory...")
    
    iteration = ['HeatmapTop10', 'cv10_FeatSel', 'AllFeatures']
    count = 0

    results_dict = {'Model': [], 'Features': [], 'AUROC': [], 'AUPRC': [], 'Precision': [], 'Recall': []}
    plot_info_df = pd.DataFrame(columns=['Iteration Counter', 'Feature Selection Method', 'time_categories', 'Total', 'Correct', 'Incorrect'])
    plot_AUROC_df_train = pd.DataFrame(columns=['Count', 'Iteration Counter', 'Feature Selection Method', 'AUROC', 'FPR', 'TPR'])
    plot_AUROC_df_test = pd.DataFrame(columns=['Count','Iteration Counter', 'Feature Selection Method', 'AUROC', 'FPR', 'TPR'])

    undersample = RandomUnderSampler(sampling_strategy=1)

    X_train_unscaled_imbalanced, X_test_unscaled, y_train_imbalanced, y_test = train_test_split(X, y_toSplit, 
                                                                        stratify=y_toSplit,
                                                                        test_size=0.1 , 
                                                                        random_state = 1)

    print("Cases and controls hold-out aka test data: \n", y_test.value_counts())
    print("Cases and controls training data: \n", y_train_imbalanced.value_counts())

    subjects_index_with_sepsis = y_test[y_test == 1].index
    # Filter based on the index values where y_test is equal to 1, because onset can be only for sepsis (thus 1)
    onset_days_arr = data_onset.loc[subjects_index_with_sepsis, 'n_onset_days']
    onset_array = onset_days_arr

    y_test.reset_index(drop=True)

    for counter_iter in range (0,3):
        count = 0
        
        plot_number = results_dir + str(counter_iter)

        for features in featureSelection_options:
            print("-----------------------> Using the features from ", iteration[count] )
            X_toScale = X_train_unscaled_imbalanced[features].copy()
            X_test_featsGood = X_test_unscaled[features].copy()
            print("Currently working with: ", X_toScale)
            
            X_train_unscaled, y_train = undersample.fit_resample(X_toScale, y_train_imbalanced)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_unscaled)
            X_test = scaler.fit_transform(X_test_featsGood)
            
            print("Cases and controls training data balanced: \n", y_train.value_counts())
            
            model, auroc_model, fpr_model, tpr_model, auprc_model, precision_model, recall_model, plot_info = ml_model_cm (best_avg_model, X_train, y_train, X_test, y_test, iteration[count], onset_array, plot_number)
            results_dict['Model'].append(model)
            results_dict['Features'].append(features)
            results_dict['AUROC'].append(auroc_model)
            results_dict['AUPRC'].append(auprc_model)
            results_dict['Precision'].append(precision_model)
            results_dict['Recall'].append(recall_model)
            
            plot_info = plot_info.reset_index()
            # print("PLOTTING INFORMATIONNNNNNNNNNN: ", plot_info)
            plot_info['Iteration Counter'] = counter_iter
            plot_info['Feature Selection Method'] = iteration[count]
            
            # Append the current iteration's plot_info to the main plot_info DataFrame
            plot_info_df = pd.concat([plot_info_df, plot_info], ignore_index=True)
            
            plot_AUROC_df_test = plot_AUROC_df_test.append({'Count': counter_iter,
                                                            'Iteration Counter': iteration[count],
                                                    'Feature Selection Method': features,
                                                    'AUROC': auroc_model,
                                                    'FPR': fpr_model,
                                                    'TPR': tpr_model}, ignore_index=True)
        
            predictions_model_train = model.predict(X_train)
            
            model_Grid_probabilities_train = model.predict_proba(X_train)
            model_probabilities_train = model_Grid_probabilities_train[:,1]
            
            
            auc_train, fpr_train, tpr_train, auprc_train, precision_train, recall_train = metrics_model(y_train, model_probabilities_train, predictions_model_train, model)
            
            plot_AUROC_df_train = plot_AUROC_df_train.append({'Count': counter_iter,
                                                            'Iteration Counter': iteration[count],
                                                        'Feature Selection Method': features,
                                                        'AUROC': auc_train,
                                                        'FPR': fpr_train,
                                                        'TPR': tpr_train}, ignore_index=True)
        
            # Append the current iteration's plot_info to the main plot_info dataframe
            # plot_info_df = pd.concat([plot_info_df, iteration_plot_info_df], ignore_index=True)
        
            count = count + 1
        
    title_name_train = 'ROC Curve for Training Data'
    title_name_test = 'ROC Curve for Testing Data'

    save_name_train = results_dir + str(plot_number) + "training_holdout" + ".png"  # plot number is actually the iteration number to be used in saving the plot
    save_name_test = results_dir + str(plot_number) + "testing_holdout.png"

    plot_AUROC_df_train_grouped = plot_AUROC_df_train.groupby('Count')
    plot_AUROC_df_test_grouped = plot_AUROC_df_test.groupby('Count')

    # Plot each group separately
    plotAUROC_trainAndTest (plot_AUROC_df_train_grouped, plot_AUROC_df_test_grouped, results_dir)

    
    #%% Dummies: majority, minority, stratified
    model_dummy_majority, auroc_dummy_majority, fpr_dummy_majority, tpr_dummy_majority, auprc_dummy_majority, precision_dummy_majority, recall_dummy_majority = custom.dummy_clf_majority0 (X_train, y_train, X_test, y_test)#, True, "Dummy_majority_" + iteration[count-1])
    results_dict['Model'].append(model_dummy_majority)
    results_dict['Features'].append(featureSelection_options[1])
    results_dict['AUROC'].append(auroc_dummy_majority)
    results_dict['AUPRC'].append(auprc_dummy_majority)
    results_dict['Precision'].append(precision_dummy_majority)
    results_dict['Recall'].append(recall_dummy_majority)

    model_dummy_minority, auroc_dummy_minority, fpr_dummy_minority, tpr_dummy_minority, auprc_dummy_minority, precision_dummy_minority, recall_dummy_minority = custom.dummy_clf_minority1(X_train, y_train, X_test, y_test)#, True, "Dummy_minority_" + iteration[count-1])
    results_dict['Model'].append(model_dummy_minority)
    results_dict['Features'].append(featureSelection_options[1])
    results_dict['AUROC'].append(model_dummy_minority)
    results_dict['AUPRC'].append(auprc_dummy_minority)
    results_dict['Precision'].append(precision_dummy_minority)
    results_dict['Recall'].append(recall_dummy_minority)

    model_dummy_stratif, auroc_dummy_stratif, fpr_dummy_stratif, tpr_dummy_stratif, auprc_dummy_stratif, precision_dummy_stratif, recall_dummy_stratif = custom.dummy_clf(X_train, y_train, X_test, y_test)#, True, "Dummy_stratif_" + iteration[count-1])
    results_dict['Model'].append(model_dummy_stratif)
    results_dict['Features'].append(featureSelection_options[1])
    results_dict['AUROC'].append(model_dummy_stratif)
    results_dict['AUPRC'].append(auprc_dummy_stratif)
    results_dict['Precision'].append(precision_dummy_stratif)
    results_dict['Recall'].append(recall_dummy_stratif)

    #%% Results
    print(results_dict)

    plotViolin (plot_info_df, results_dir)
    print_AUROCandAUPRC_andSTD (results_dict)

#%% Wrap this .py script
def wrapAdvancedAnalysis (data_path):
    X, y_toSplit, featureSelection_options, data_onset, best_avg_model = getData(data_path)
    trainModels_andTest(X, y_toSplit, featureSelection_options, data_onset, best_avg_model)


#data_path = 'C:/Users/aa36.MEDMA/Desktop/Franzi/CC_QtJune/New_Bianka/fbentriesProgV2.csv'
#wrapAdvancedAnalysis (data_path)
