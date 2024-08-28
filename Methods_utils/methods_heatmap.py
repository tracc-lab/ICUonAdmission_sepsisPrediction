# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:15:21 2024

@author: Asus
"""
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

from collections import Counter



#%% heatmap_oneFeatureSelectionCV (featSel_folds, save_name);   savename should be experim + iteration_max aka CV + foldName
## savename should be experim + iteration_max aka CV + foldName
def heatmap_oneFeatureSelectionCV (featSel_folds, save_name):
    # Flatten the array of arrays and get unique elements
    flat_data = [item for sublist in featSel_folds for item in sublist]
    unique_elements = np.unique(flat_data)
    print("Entered heatmap_oneFeatureSelectionCV and this is the save name: ", save_name)
    # Create a dictionary to store counts for each element
    counts_dict = {element: [sublist.count(element) for sublist in featSel_folds] + [flat_data.count(element)] for element in unique_elements}
    
    # Create a DataFrame for seaborn
    df = pd.DataFrame(counts_dict, index=[f"Array {i}" for i in range(1, len(featSel_folds) + 1)] + ['Total'])
    name_to_save = ' '.join(["Heatmap_", save_name, ".png"])
    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(df, cmap='viridis', annot=True, fmt="d", cbar=True)
    plt.xlabel('Element Name')
    plt.ylabel('Array Index')
    plt.title('Heatmap of Element Counts in Arrays')
    # plt.savefig("Heatmap_" + save_name + ".png")
    plt.savefig(name_to_save)
    plt.show()

## we don't include lasso here beauase it never changes
def original_heatmap(*args):
    save_name, shap_folds, rf_folds, xgb_folds, ridge_folds, logistic_folds = args
    print("Entered Original Heatmap and this si the save name: ", save_name)
    # Flatten the arrays of arrays and get unique elements

    flat_data_rf = [item for sublist in rf_folds for item in sublist]
    flat_data_xgb = [item for sublist in xgb_folds for item in sublist]
    flat_data_ridge = [item for sublist in ridge_folds for item in sublist]
    flat_data_logistic = [item for sublist in logistic_folds for item in sublist]
    flat_data_shap = [item for sublist in shap_folds for item in sublist]
    
    unique_elements_rf = np.unique(flat_data_rf)
    unique_elements_xgb = np.unique(flat_data_xgb)
    unique_elements_ridge = np.unique(flat_data_ridge)
    unique_elements_logistic = np.unique(flat_data_logistic)
    unique_elements_shap = np.unique(flat_data_shap)
    
    
    # Create a dictionary to store counts for each element
    counts_dict_rf = {element: flat_data_rf.count(element) for element in unique_elements_rf}
    counts_dict_xgb= {element: flat_data_xgb.count(element) for element in unique_elements_xgb}
    counts_dict_ridge = {element: flat_data_ridge.count(element) for element in unique_elements_ridge}
    counts_dict_logistic = {element: flat_data_logistic.count(element) for element in unique_elements_logistic}
    counts_dict_shap = {element: flat_data_shap.count(element) for element in unique_elements_shap}
    
    
    # Create a DataFrame for seaborn
    df_rf = pd.DataFrame(list(counts_dict_rf.items()), columns=['Element', 'Total RF'])
    df_xgb = pd.DataFrame(list(counts_dict_xgb.items()), columns=['Element', 'Total XGB'])
    df_ridge = pd.DataFrame(list(counts_dict_ridge.items()), columns=['Element', 'Total Ridge'])
    df_logistic = pd.DataFrame(list(counts_dict_logistic.items()), columns=['Element', 'Total Logistic'])
    df_shap = pd.DataFrame(list(counts_dict_shap.items()), columns=['Element', 'Total Shap'])
    
    # Set index for both DataFrames
    df_rf.set_index('Element', inplace=True)
    df_xgb.set_index('Element', inplace=True)
    df_ridge.set_index('Element', inplace=True)
    df_logistic.set_index('Element', inplace=True)
    df_shap.set_index('Element', inplace=True)
    
    # Combine the DataFrames
    df_combined = pd.concat([df_shap, df_rf, df_xgb, df_ridge, df_logistic], axis=1)
    
    # Convert the DataFrame values to integers
    df_combined = df_combined.fillna(0)
    
    df_combined = df_combined.astype(int)
 
    total_totals = df_combined.sum(axis=1)
    total_totals = total_totals.sort_values(ascending=False)
    print("Total of included features: ", total_totals.size)
    
    top_10_absolute = total_totals.nlargest(10)

    # Print the absolute top 10 most selected features
    print("Absolute Top 10 most selected features:")
    print(top_10_absolute)
    top10_ever = top_10_absolute.index
    top10_ever_list = top10_ever.tolist()

    
    plt.figure(figsize=(10, 18))
    ax = sns.heatmap(pd.DataFrame(total_totals, columns=['Total']), cmap=plt.cm.BuPu, annot=True, fmt="d", cbar=True, annot_kws={"size": 18})
    
    plt.xticks(fontsize=18)  # X-axis ticks
    plt.yticks(fontsize=18)  # Y-axis ticks
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    ax.set_xticklabels([])  # so the 'Total' doesn't appear right under the plot and only then 'Prevalence'. this removes the 'Total'

    plt.ylabel('Feature Name', fontsize=18)
    plt.xlabel('Prevalence', fontsize=18)

    plt.title('Heatmap of Feature Selection Prevalence', fontsize = 20)
    plt.savefig("Heatmap of features" + save_name + ".png", bbox_inches='tight', dpi = 600)
    plt.show()
    
    return top10_ever_list
    
# #%%
# def heatmaps_allFeatureSelectionsCV (*args):
#     x = 10
#     arrays, arra2 = args
#     nr_models = len(auc_score)
#     count = 0
#     while count < nr_models:
#         auc = auc_score[count]
#         plt.plot(fpr[count], tpr[count], linestyle = '-', label = model[count] + ' AUROC ' + str(auc))
#         count = count + 1

#     plt.title("ROC AUC plot")
#     plt.xlabel("False Positive Rate (FPR)")
#     plt.ylabel("True Positive Rate (TPR)")
    
#     plt.legend()
#     plt.savefig("AUC ROC" + experim + "baseline_allFeats_models.png")
#     plt.show()
    
# def plot_auprc_models (*args):
    
#     recall, precision, auprc_score, model, experim = args
# #     print("fpr", fpr)
# #     print("auc", auc_score)
# #     print(model)
#     nr_models = len(auprc_score)
#     count = 0
#     while count < nr_models:
#         auprc = auprc_score[count]
#         plt.plot(recall[count], precision[count], linestyle = '-', label = model[count] + ' AUPRC ' + str(auprc))
#         count = count + 1

#     plt.title("AUPRC plot")
#     plt.xlabel("Recall (Sensitivity, TPR)")
#     plt.ylabel("Precision (PPV))")
    
#     plt.legend()
#     plt.savefig("AUPRC" + experim + "baseline_allFeats_models.png")
#     plt.show()
    


# def feat_imp_xgb(model_xgb, names):
#     xgb_feat_imp = model_xgb.feature_importances_
#     # print(xgb_feat_imp)

#     res_xgb = {}
     
#     for i in range(0,len(xgb_feat_imp)):
#         res_xgb[names[i]] = xgb_feat_imp[i]

#     print(" ----------------------------------------------------------------------- ")
#     #print(" All features with their XGBoost Importance")
#     sorted_res_xgb = dict(sorted(res_xgb.items(), key=lambda item: item[1], reverse = True))
#     #print(sorted_res_xgb)
    
#     # print(" ----------------------------------------------------------------------- ")
#     # print(" Selected XGBoost features based on importance")
#     selected_xgb = {}
#     count_xgb = 1
#     for key, value in sorted_res_xgb.items():
#         if value >= 0.01 and count_xgb <= 10:
#             selected_xgb[key] = value
#             count_xgb = count_xgb + 1
#     # print(selected_xgb)
#     # print(selected_xgb.keys())
    
#     keys = [k for k, v in selected_xgb.items()]
#     # print(keys)
#     # print(len(keys))
    
#     return keys
    
# def feat_imp_ridge(model_ridge, names):
#     coefficients = model_ridge.coef_[0]

#     feature_importance_ridge = pd.DataFrame({'Feature': names, 'Importance': np.abs(coefficients)})
#     feature_importance_ridge = feature_importance_ridge.sort_values('Importance', ascending=False)
#     #feature_importance_ridge.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
    
#     feature_importance_ridge_arr = feature_importance_ridge.query('Importance > 0.1')['Feature'].values
#     #print(feature_importance_ridge_arr)
#     #print(len(feature_importance_ridge_arr))
#     print(feature_importance_ridge_arr[0:10])
#     #print(len(feature_importance_ridge_arr[0:10]))
#     keys = feature_importance_ridge_arr[0:10].tolist()
    
#     return keys 
    
        
# def feat_imp_logistic(model_logistic, names):
#     coefficients = model_logistic.coef_[0]

#     feature_importance_logistic = pd.DataFrame({'Feature': names, 'Importance': np.abs(coefficients)})
#     feature_importance_logistic = feature_importance_logistic.sort_values('Importance', ascending=False)
#     #feature_importance_logistic.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
#     #print(feature_importance_logistic)
    
#     feature_importance_logistic_arr = feature_importance_logistic.query('Importance > 0.1')['Feature'].values
#     #print(feature_importance_logistic_arr)
#     #print(len(feature_importance_logistic_arr))
#     keys = feature_importance_logistic_arr[0:10].tolist()
#     print(keys)
#     #print(len(feature_importance_logistic_arr[0:10]))
    
#     return keys
    
# def feat_imp_shap(model, names, kind, subset): # it is just model because it may change every time
#     #make the explainer type based on a persed string. like TreeExplainer, LinearExplainer    
#     #subset means that we can use this to explain what was going on in the training or in the test
#     #  but according to practice, it is more useful to see what it does to test data.
#     if kind == 'rf' or kind == 'random forest' or kind == 'svm':
#         explainer = shap.KernelExplainer(model.predict, subset)
#         shap_values = explainer.shap_values(subset, check_additivity=False)
#         print(shap_values)
#         # Get top 10 features based on SHAP values
#         vals = np.abs(shap_values).mean(axis=0)
#         top_10_features_indices = np.argsort(vals)[::-1][:10]
#         top_10_features = names[top_10_features_indices]
#         return top_10_features.tolist()
    
#     elif kind == 'xgb':
#         explainer = shap.TreeExplainer(model, subset)
    
#     elif kind == 'linear':
#         explainer = shap.LinearExplainer(model, subset)
    
#     # Calculate SHAP values for the training set
#     shap_values = explainer.shap_values(subset)
    
#     # Get top 10 features based on SHAP values
#     vals = np.abs(shap_values).mean(axis=0)
#     top_10_features_indices = np.argsort(vals)[::-1][:10]
#     top_10_features = names[top_10_features_indices]
    
#     # Create a DataFrame with SHAP values and top 10 features
#     shap_df = pd.DataFrame(shap_values, columns=names)
#     #shap_df['target'] = y_train  # Assuming 'target' is your target variable
#     shap_df['abs_shap_values_mean'] = np.abs(shap_values).mean(axis=1)
    
#     # Add top 10 features to the DataFrame
#     #shap_df_top_10 = shap_df[['target', 'abs_shap_values_mean'] + top_10_features.tolist()]
#     shap_df_top_10 = shap_df[['abs_shap_values_mean'] + top_10_features.tolist()]
    
#     # Display the DataFrame with top 10 features
#     print(shap_df_top_10.head())
    
#     return top_10_features.tolist()

# #%% How to combine the feat imp into a pd df