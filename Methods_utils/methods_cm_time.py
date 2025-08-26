# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:15:21 2024

@author: Asus
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
import time
import matplotlib.pyplot as plt

from sklearn.metrics import  average_precision_score, precision_recall_curve, roc_curve, roc_auc_score

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import shap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

colors = ['#630C3A', '#27C3C1', '#FFC107', '#7E34F9', '#E01889', '#617111','#fe6100',  '#7d413c',
'#423568', '#5590b4']
sns.set_palette(sns.color_palette(colors))

#%% Needed methods like metrics, plot, etc.
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

def plot_auc_models (*args):
    
    fpr, tpr, auc_score, model, experim = args
    nr_models = len(auc_score)
    count = 0
    while count < nr_models:
        auc = auc_score[count]
        plt.plot(fpr[count], tpr[count], linestyle = '-', label = model[count] + ' AUROC ' + str(auc))
        count = count + 1

    plt.title("ROC AUC plot")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    
    plt.legend()
    plt.savefig("AUC ROC" + experim + "baseline_allFeats_models.png")
    #plt.show()
    plt.close()
    
def plot_auprc_models (*args):
    
    recall, precision, auprc_score, model, experim = args
    nr_models = len(auprc_score)
    count = 0
    while count < nr_models:
        auprc = auprc_score[count]
        plt.plot(recall[count], precision[count], linestyle = '-', label = model[count] + ' AUPRC ' + str(auprc))
        count = count + 1

    plt.title("AUPRC plot")
    plt.xlabel("Recall (Sensitivity, TPR)")
    plt.ylabel("Precision (PPV))")
    
    plt.legend()
    plt.savefig("AUPRC" + experim + "baseline_allFeats_models.png")
    #plt.show()
    plt.close()

# categorize time into bins to plot_onset with correct, incorrect, total predictions
# def categorize_time(onset_array):
#     categories = pd.cut(onset_array, bins=[-float('inf'), 5, 7, 10, float('inf')], labels=['< 5', '>=5 and <7', '>=7 and < 10', '>=10'])
#     return categories

def categorize_time(onset_array):
    categories = pd.cut(onset_array, bins=[-float('inf'), 4, 8, float('inf')], labels=['< 4 days', '>=4 and <8', '>=8'])
    return categories

# def categorize_time(onset_array):
#     categories = pd.cut(onset_array, bins=[-float('inf'), 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')], labels=['< 2','< 3','< 4' ,'< 5', '>=5 and <6','< 7', '>=7 and < 8','< 9', '< 10','>=10'])
#     return categories

# plot total, correct, and incorrect predictions based on onset time categories.
def plot_onset(y_test, predictions, name, onset_array, plot_number, results_dir):
    print("Length of y_test onset:", len(y_test), y_test)
    print("Length of predictions onset:", len(predictions), predictions)
    print("Length of onset_array onset:", len(onset_array))
    
    # Filter predictions for class 1 based on y_test
    predictions_class_1 = predictions[y_test == 1]
    
    # Categorize time
    time_categories = categorize_time(onset_array)
    print("The time categories are: ", time_categories)

    df = pd.DataFrame({
        'y_test': y_test[y_test == 1],  # Filtered for class 1
        'predictions': predictions_class_1,
        'time_categories': time_categories[y_test == 1]  # Filtered for class 1
    })

    df.to_csv("predictionsDebugCM.csv", index=False)

    grouped = df.groupby('time_categories').apply(lambda x: pd.Series({
        'Total': len(x),  # Total count
        'Correct': ((x['y_test'] == 1) & (x['predictions'] == 1)).sum(),  # Correct predictions (True Positives)
        'Incorrect': ((x['y_test'] == 1) & (x['predictions'] == 0)).sum() + ((x['y_test'] == 0) & (x['predictions'] == 1)).sum()  # Incorrect predictions (False Positives + False Negatives)
    }))
    
    print("Plot Onset Intermediate Results:")
    print(grouped)
    
    # Plotting
    plt.figure(figsize=(16, 12))
    ax = grouped.plot(kind='bar', width=0.6)
    plt.title(f'Predictions for {name}', fontsize = 16)
    plt.xlabel('Time intervals (days) before onset', fontsize=16-2)
    plt.ylabel('Number of cases', fontsize=16-2)
    
    plt.xticks(rotation=0, fontsize=14)  # Adjust x-tick font size and rotation
    plt.tick_params(axis='x', length=0) # removes the small tick lines
    plt.yticks(fontsize=14)

    plt.legend(title='Prediction Type')

    max_height = max([p.get_height() for p in ax.patches])
    
    # Annotating bars with values excluding the max height
    for i in ax.patches:
        if i.get_height() != max_height:
            ax.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.1, str(int(i.get_height())), ha='center', va='bottom', fontsize=16-2)

    # Save plot
    plt.tight_layout()
    
    # Construct the full save path using absolute path and shortened filename
    import os
    abs_results_dir = os.path.abspath(results_dir)
    os.makedirs(abs_results_dir, exist_ok=True)  # Ensure directory exists
    
    # Extract model name and feature set from name (e.g., "Logistic_PT10" -> "Logistic" + "PT10")
    if '_' in name:
        model_part, feature_part = name.split('_', 1)
        model_short = model_part[:8]  # Shorten model name
        feature_short = feature_part[:4]  # Shorten feature set name
        save_path = os.path.join(abs_results_dir, f"{plot_number}_{model_short}_{feature_short}_onset.png")
    else:
        model_short = name[:8]
        save_path = os.path.join(abs_results_dir, f"{plot_number}_{model_short}_onset.png")
    print(f"DEBUG: Saving onset plot to: {save_path}")
    
    plt.savefig(save_path, dpi = 600)
    #plt.show()
    plt.close()
    
    return grouped
    
def plot_custom_confusion_matrix(cm, classes, title, cmap):
    plt.figure(figsize=(11, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)  # Adjust font size for title
    
    # colorbar = plt.colorbar()
    # colorbar.ax.tick_params(labelsize=14)
    
    ax = plt.gca()
    colorbar = plt.colorbar(ax=ax, fraction=0.046, pad=0.04)  # Adjust fraction and pad as needed
    colorbar.ax.tick_params(labelsize=24)   # adjust size of the numbers on the colorbar
    
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=28)  # Adjust font size for tick labels
    plt.yticks(tick_marks, classes, fontsize=28)  

    plt.tick_params(axis='x', length=0) # removes the small tick lines
    plt.tick_params(axis='y', length=0) # removes the small tick lines

    fmt = 'd'  ### format decimal (integer)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     fontsize=30,  # Adjust font size for numbers in boxes   ideal 18
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=28)  # Adjust font size for ylabel
    plt.xlabel('Predicted label', fontsize=28)  # Adjust font size for xlabel
    
    plt.tight_layout()

def plot_confusionMatrix(y_test, predictions, name, onset_array, plot_number, results_dir):
    import os
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    print(cm[1][0])
    print("Correctly classified septic patients: ", cm[1][1])
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    print("TP: ", tp)
    
    # Plot and save confusion matrix
    plot_custom_confusion_matrix(cm, ['Sepsis-free', 'Sepsis'], 'Confusion matrix', cmap = plt.cm.BuPu)
    
    plt.tight_layout()
    
    # Construct the full save path using absolute path and shortened filename
    abs_results_dir = os.path.abspath(results_dir)
    os.makedirs(abs_results_dir, exist_ok=True)  # Ensure directory exists
    
    # Extract model name and feature set from name (e.g., "Logistic_PT10" -> "Logistic" + "PT10")
    if '_' in name:
        model_part, feature_part = name.split('_', 1)
        model_short = model_part[:8]  # Shorten model name
        feature_short = feature_part[:4]  # Shorten feature set name
        save_path = os.path.join(abs_results_dir, f"{plot_number}_cm_{model_short}_{feature_short}.png")
    else:
        model_short = name[:8]
        save_path = os.path.join(abs_results_dir, f"{plot_number}_cm_{model_short}.png")
    print(f"DEBUG: Saving confusion matrix to: {save_path}")
    
    plt.savefig(save_path, dpi = 600)
    #plt.show()
    plt.close()
    
    # Plot the onset time as well 
    grouped = plot_onset(y_test, predictions, name, onset_array, plot_number, results_dir)
    
    return grouped

#%% The Classifiers general returns: model, auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy
def dummy_clf(X_train, y_train, X_test, y_test, confusion_matrix, name_cm, onset_array):
    dummy_clf = DummyClassifier(strategy='stratified', random_state=1)
    dummy_clf.fit(X_train, y_train)
    
    predictions_dummy = dummy_clf.predict(X_test)
        
    dummy_Grid_probabilities = dummy_clf.predict_proba(X_test)
    dummy_probabilities = dummy_Grid_probabilities[:,1]
    
    auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy = metrics_model (y_test, dummy_probabilities, predictions_dummy, "Dummy")
    if confusion_matrix == True:
        plot_confusionMatrix(y_test, predictions_dummy, name_cm, onset_array)
    return dummy_clf, auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy

def dummy_clf_majority0(X_train, y_train, X_test, y_test, confusion_matrix, name_cm, onset_array):
    dummy_clf = DummyClassifier(strategy='constant', constant=0, random_state = 1)
    dummy_clf.fit(X_train, y_train)
    
    predictions_dummy = dummy_clf.predict(X_test)
        
    dummy_Grid_probabilities = dummy_clf.predict_proba(X_test)
    dummy_probabilities = dummy_Grid_probabilities[:,1]
    
    auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy = metrics_model (y_test, dummy_probabilities, predictions_dummy, "Dummy All Majority")
    
    if confusion_matrix == True:
        plot_confusionMatrix(y_test, predictions_dummy, name_cm, onset_array)
    
    return dummy_clf, auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy


def dummy_clf_minority1(X_train, y_train, X_test, y_test, confusion_matrix, name_cm, onset_array):
    dummy_clf = DummyClassifier(strategy='constant', constant=1, random_state = 1)
    dummy_clf.fit(X_train, y_train)
    
    predictions_dummy = dummy_clf.predict(X_test)
        
    dummy_Grid_probabilities = dummy_clf.predict_proba(X_test)
    dummy_probabilities = dummy_Grid_probabilities[:,1]
    
    auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy = metrics_model (y_test, dummy_probabilities, predictions_dummy, "Dummy All Minority")
    
    if confusion_matrix == True:
        plot_info = plot_confusionMatrix(y_test, predictions_dummy, name_cm, onset_array)
        
    return dummy_clf, auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy


def random_forest(X_train, y_train, X_test, y_test, confusion_matrix, name_cm, onset_array, plot_number, results_dir):
    rf = RandomForestClassifier(random_state=1)
    # rf.set_params(n_estimators = 100, max_features = 'sqrt', max_leaf_nodes = 9,
    #                              min_samples_split = 2, min_samples_leaf = 1, 
    #                              warm_start = True, bootstrap = True)
    rf.set_params(n_estimators = 100, max_features = 'sqrt', max_leaf_nodes = 9,
                                     min_samples_split = 10, min_samples_leaf = 4, 
                                     warm_start = True, bootstrap = True)
    rf.fit(X_train, y_train)
    
    predictions_rf = rf.predict(X_test)
        
    rf_Grid_probabilities = rf.predict_proba(X_test)
    rf_probabilities = rf_Grid_probabilities[:,1]
    
    auc_rf, fpr_rf, tpr_rf, auprc_rf, precision_rf, recall_rf = metrics_model (y_test, rf_probabilities, predictions_rf, "Random Forest")
    
    if confusion_matrix == True:
        plot_info = plot_confusionMatrix(y_test, predictions_rf, name_cm, onset_array, plot_number, results_dir)
    
    return rf, auc_rf, fpr_rf, tpr_rf, auprc_rf, precision_rf, recall_rf, plot_info
    
def svm(X_train, y_train, X_test, y_test, confusion_matrix, name_cm, onset_array, plot_number, results_dir):
    svm = SVC (random_state=1, probability=True)
    svm.set_params(C = 1, degree = 1, gamma = 0.01, kernel = 'rbf')
    # svm.set_params(C = 10, degree = 1, gamma = 0.01, kernel = 'linear')
    
    svm.fit(X_train, y_train)
      
    y_pred_svm = svm.predict(X_test)
        
    svm_Grid_probabilities = svm.predict_proba(X_test)
    svm_probabilities = svm_Grid_probabilities[:,1]
    
    auc_svm, fpr_svm, tpr_svm, auprc_svm, precision_svm, recall_svm = metrics_model (y_test, svm_probabilities, y_pred_svm, "SVM")
    print("from inside the SVM, the predictions are: ", list(y_pred_svm))
    print("\nThe actual true values are: ", list(y_test))
    
    y_pred_svm_series = pd.Series(y_pred_svm, index=y_test.index)
    
    if confusion_matrix == True:
        plot_info = plot_confusionMatrix(y_test, y_pred_svm_series, name_cm, onset_array, plot_number, results_dir)
        
    return svm, auc_svm, fpr_svm, tpr_svm, auprc_svm, precision_svm, recall_svm, plot_info

def xgboost_clf(X_train, y_train, X_test, y_test, confusion_matrix, name_cm, onset_array, plot_number, results_dir):
    xgboost = xgb.XGBClassifier(random_state=1)
    xgboost.set_params(colsample_bytree= 1, gamma = 1, max_depth= 18, 
                       min_child_weight= 10, n_estimators= 100, reg_alpha= 1, reg_lambda= 0)
    # xgboost.set_params(colsample_bytree= 0.5, gamma = 9, max_depth= 18, 
    #                min_child_weight= 10, n_estimators= 500, reg_alpha= 1, reg_lambda= 0)
    
    xgboost.fit(X_train, y_train) 
    
    predictions_xgboost = xgboost.predict(X_test)
    
    xgboost_Grid_probabilities = xgboost.predict_proba(X_test)
    xgboost_probabilities = xgboost_Grid_probabilities[:,1]

    auc_xgboost, fpr_xgboost, tpr_xgboost, auprc_xgboost, precision_xgboost, recall_xgboost = metrics_model (y_test, xgboost_probabilities, predictions_xgboost, "XGBoost")
    
    if confusion_matrix == True:
        plot_info = plot_confusionMatrix(y_test, predictions_xgboost, name_cm, onset_array, plot_number, results_dir)
        
    return xgboost, auc_xgboost, fpr_xgboost, tpr_xgboost, auprc_xgboost, precision_xgboost, recall_xgboost, plot_info

def ridge(X_train, y_train, X_test, y_test, confusion_matrix, name_cm, onset_array, plot_number, results_dir):
    ridge = RidgeClassifier(random_state=1)
    ridge.set_params(alpha = 34.30469286314926)
    # ridge.set_params(alpha = 0.029150530628251816)
    
    ridge.fit(X_train, y_train)
    
    predictions_ridge = ridge.predict(X_test)
    
    ridge_probabilities = ridge.decision_function(X_test) 
    
    auc_ridge, fpr_ridge, tpr_ridge, auprc_ridge, precision_ridge, recall_ridge = metrics_model (y_test, ridge_probabilities, predictions_ridge, "Ridge")

    if confusion_matrix == True:
        plot_info = plot_confusionMatrix(y_test, predictions_ridge, name_cm, onset_array, plot_number, results_dir)
        
    return ridge, auc_ridge, fpr_ridge, tpr_ridge, auprc_ridge, precision_ridge, recall_ridge, plot_info
    
def logistic(X_train, y_train, X_test, y_test, confusion_matrix, name_cm, onset_array, plot_number, results_dir):
    logistic = LogisticRegression(random_state=1)
    # logistic.set_params(penalty ='l1', C = 2.310129700083163,  solver = 'saga', max_iter = 500)
    logistic.set_params(penalty ='l2', C = 0.08111308307896872,  solver = 'saga', max_iter = 100)
    logistic.fit(X_train, y_train)
    
    y_pred_logistic = logistic.predict(X_test)
        
    probs = logistic.predict_proba(X_test) 
    logistic_probabilities = probs[:,1]
    
   
    auc_logistic, fpr_logistic, tpr_logistic, auprc_logistic, precision_logistic, recall_logistic = metrics_model (y_test, logistic_probabilities, y_pred_logistic, "Logistic")
    
    if confusion_matrix == True:
        plot_info = plot_confusionMatrix(y_test, y_pred_logistic, name_cm, onset_array, plot_number, results_dir)
        
    return logistic, auc_logistic, fpr_logistic, tpr_logistic, auprc_logistic, precision_logistic, recall_logistic, plot_info
