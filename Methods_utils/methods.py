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

from sklearn.metrics import  average_precision_score, precision_recall_curve, roc_curve, roc_auc_score

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import shap


#%% Needed methods like metrics, plot, etc.
def metrics_model (y_test, probabilities, predictions, model):
    # print("probs: ", probabilities)
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
    plt.show()

def plot_auc_allModels (*args):
    models, fprs, tprs, aucs, experim = args
    
    plt.figure(figsize=(8, 6))
    for model, fpr, tpr, auc in zip(models, fprs, tprs, aucs):
        plt.plot(fpr, tpr, label=f'{model} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{experim}_AUROC_all_folds.png')
    plt.show()

    
    
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
    plt.show()
    
#%% The Classifiers general returns: model, auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy
# we use dummy as a baselines classifier. the baseline for AUROC is always 0.5 and the baseline for AUPRC could be computed as well. 
def dummy_clf(X_train, y_train, X_test, y_test):
    dummy_clf = DummyClassifier(strategy='constant', constant=0, random_state = 1)  #DummyClassifier(strategy='stratified', random_state=1)
    
    dummy_clf.fit(X_train, y_train)
    
    predictions_dummy = dummy_clf.predict(X_test)
        
    dummy_Grid_probabilities = dummy_clf.predict_proba(X_test)
    dummy_probabilities = dummy_Grid_probabilities[:,1]
    
    auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy = metrics_model (y_test, dummy_probabilities, predictions_dummy, "Dummy All Majority")
    
    return dummy_clf, auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy


def dummy_clf_majority0(X_train, y_train, X_test, y_test):
    dummy_clf = DummyClassifier(strategy='constant', constant=0, random_state = 1)
    
    dummy_clf.fit(X_train, y_train)

    predictions_dummy = dummy_clf.predict(X_test)
        
    dummy_Grid_probabilities = dummy_clf.predict_proba(X_test)
    dummy_probabilities = dummy_Grid_probabilities[:,1]
    
    auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy = metrics_model (y_test, dummy_probabilities, predictions_dummy, "Dummy All Majority")
    
    return dummy_clf, auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy


def dummy_clf_minority1(X_train, y_train, X_test, y_test):
    dummy_clf = DummyClassifier(strategy='constant', constant=1, random_state = 1)

    dummy_clf.fit(X_train, y_train)
    
    predictions_dummy = dummy_clf.predict(X_test)
        
    dummy_Grid_probabilities = dummy_clf.predict_proba(X_test)
    dummy_probabilities = dummy_Grid_probabilities[:,1]
    
    auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy = metrics_model (y_test, dummy_probabilities, predictions_dummy, "Dummy All Minority")
    
    return dummy_clf, auc_dummy, fpr_dummy, tpr_dummy, auprc_dummy, precision_dummy, recall_dummy

# Now the "proper" classifiers
def random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(random_state=1)
    rf.set_params(n_estimators = 100, max_features = 'sqrt', max_leaf_nodes = 9,
                                     min_samples_split = 10, min_samples_leaf = 4, 
                                     warm_start = True, bootstrap = True)
    
    rf.fit(X_train, y_train)
    
    predictions_rf = rf.predict(X_test)
        
    rf_Grid_probabilities = rf.predict_proba(X_test)
    rf_probabilities = rf_Grid_probabilities[:,1]
    
    auc_rf, fpr_rf, tpr_rf, auprc_rf, precision_rf, recall_rf = metrics_model (y_test, rf_probabilities, predictions_rf, "Random Forest")
    
    return rf, auc_rf, fpr_rf, tpr_rf, auprc_rf, precision_rf, recall_rf
    
def svm(X_train, y_train, X_test, y_test):
    svm = SVC (random_state=1, probability=True)
    
    svm.set_params(C = 1, degree = 1, gamma = 0.01, kernel = 'rbf')
    
    svm.fit(X_train, y_train)
      
    y_pred_svm = svm.predict(X_test)
        
    svm_Grid_probabilities = svm.predict_proba(X_test)
    svm_probabilities = svm_Grid_probabilities[:,1]
    
    auc_svm, fpr_svm, tpr_svm, auprc_svm, precision_svm, recall_svm = metrics_model (y_test, svm_probabilities, y_pred_svm, "SVM")

    return svm, auc_svm, fpr_svm, tpr_svm, auprc_svm, precision_svm, recall_svm

def xgboost_clf(X_train, y_train, X_test, y_test):
    xgboost = xgb.XGBClassifier(random_state=1)
    xgboost.set_params(colsample_bytree= 1, gamma = 1, max_depth= 18, 
                       min_child_weight= 10, n_estimators= 100, reg_alpha= 1, reg_lambda= 0)
    
    xgboost.fit(X_train, y_train) #, early_stopping_rounds=10 it needs validation aka train, test, val
    
    predictions_xgboost = xgboost.predict(X_test)
    
    xgboost_Grid_probabilities = xgboost.predict_proba(X_test)
    xgboost_probabilities = xgboost_Grid_probabilities[:,1]

    auc_xgboost, fpr_xgboost, tpr_xgboost, auprc_xgboost, precision_xgboost, recall_xgboost = metrics_model (y_test, xgboost_probabilities, predictions_xgboost, "XGBoost")
    
    return xgboost, auc_xgboost, fpr_xgboost, tpr_xgboost, auprc_xgboost, precision_xgboost, recall_xgboost

def ridge(X_train, y_train, X_test, y_test):
    ridge = RidgeClassifier(random_state=1)
    ridge.set_params(alpha = 34.30469286314926)
  
    ridge.fit(X_train, y_train)
    
    predictions_ridge = ridge.predict(X_test)
    
    ridge_probabilities = ridge.decision_function(X_test)
    
    auc_ridge, fpr_ridge, tpr_ridge, auprc_ridge, precision_ridge, recall_ridge = metrics_model (y_test, ridge_probabilities, predictions_ridge, "Ridge")

    return ridge, auc_ridge, fpr_ridge, tpr_ridge, auprc_ridge, precision_ridge, recall_ridge
    
def logistic(X_train, y_train, X_test, y_test):
    logistic = LogisticRegression(random_state=1)
    logistic.set_params(penalty ='l2', C = 0.08111308307896872,  solver = 'saga', max_iter = 100)
    
    logistic.fit(X_train, y_train)
    
    y_pred_logistic = logistic.predict(X_test)
        
    probs = logistic.predict_proba(X_test)  
    logistic_probabilities = probs[:,1]
    
    auc_logistic, fpr_logistic, tpr_logistic, auprc_logistic, precision_logistic, recall_logistic = metrics_model (y_test, logistic_probabilities, y_pred_logistic, "Logistic")

    return logistic, auc_logistic, fpr_logistic, tpr_logistic, auprc_logistic, precision_logistic, recall_logistic

#%% Feature Importance
def feat_imp_rf(model_rf, names):
    importances_rf = model_rf.feature_importances_
    res_rf_ft = {}
    
    for i in range(0,len(importances_rf)):
        res_rf_ft[names[i]] = importances_rf[i]
    
    sorted_res_rf = dict(sorted(res_rf_ft.items(), key=lambda item: item[1], reverse = True))
    
    count = 1
    selected_rf = {}
    for key, value in sorted_res_rf.items():
        if value >= 0.01 and count <= 10:
            selected_rf[key] = value
            count = count + 1
    keys = [k for k, v in selected_rf.items()]

    return keys


def feat_imp_xgb(model_xgb, names):
    xgb_feat_imp = model_xgb.feature_importances_

    res_xgb = {}
     
    for i in range(0,len(xgb_feat_imp)):
        res_xgb[names[i]] = xgb_feat_imp[i]

    print(" ----------------------------------------------------------------------- ")
    sorted_res_xgb = dict(sorted(res_xgb.items(), key=lambda item: item[1], reverse = True))

    selected_xgb = {}
    count_xgb = 1
    for key, value in sorted_res_xgb.items():
        if value >= 0.01 and count_xgb <= 10:
            selected_xgb[key] = value
            count_xgb = count_xgb + 1
    
    keys = [k for k, v in selected_xgb.items()]
    
    return keys
    
def feat_imp_ridge(model_ridge, names):
    coefficients = model_ridge.coef_[0]

    feature_importance_ridge = pd.DataFrame({'Feature': names, 'Importance': np.abs(coefficients)})
    feature_importance_ridge = feature_importance_ridge.sort_values('Importance', ascending=False)
    
    feature_importance_ridge_arr = feature_importance_ridge.query('Importance > 0.1')['Feature'].values
    print(feature_importance_ridge_arr[0:10])

    keys = feature_importance_ridge_arr[0:10].tolist()
    
    return keys 
    
        
def feat_imp_logistic(model_logistic, names):
    coefficients = model_logistic.coef_[0]

    feature_importance_logistic = pd.DataFrame({'Feature': names, 'Importance': np.abs(coefficients)})
    feature_importance_logistic = feature_importance_logistic.sort_values('Importance', ascending=False)
    
    feature_importance_logistic_arr = feature_importance_logistic.query('Importance > 0.1')['Feature'].values

    keys = feature_importance_logistic_arr[0:10].tolist()
    print(keys)
    
    
    return keys

#############################################################################################################
# As said in the previous .py file, there are a few things to take into account when using SHAP:            #
# Different types of algorithms need a slightly different SHAP. I tried and it didn't really work otherwise #
# In this case, the subset is X_test. Now, SHAP can be used on training as well as on test                  #
# but because we actually apply SHAP to that model that had highest AUPRC, it is more interesting to see    #
# which features truly contributed to the prediction stage on the X_test.                                   #
# Oh, moreover, this test is the test among the folds. *NOT* the hold-out                                   #
#############################################################################################################
def feat_imp_shap(model, names, kind, subset): # it is just model because it may change every time          
    if kind == 'rf' or kind == 'random forest' or kind == 'svm':
        explainer = shap.KernelExplainer(model.predict, subset)
        shap_values = explainer.shap_values(subset, check_additivity=False)
        print(shap_values)
        # Get top 10 features based on SHAP values
        vals = np.abs(shap_values).mean(axis=0)
        top_10_features_indices = np.argsort(vals)[::-1][:10]
        top_10_features = names[top_10_features_indices]
        return top_10_features.tolist()  
    elif kind == 'xgb':
        explainer = shap.TreeExplainer(model, subset)  #shap.Explainer(model, subset) 
    elif kind == 'linear':
        explainer = shap.LinearExplainer(model, subset)
    
    shap_values = explainer.shap_values(subset)
    
    # Selected top 10 features from SHAP values
    vals = np.abs(shap_values).mean(axis=0)
    top_10_features_indices = np.argsort(vals)[::-1][:10]
    top_10_features = names[top_10_features_indices]
    
    # Create a DataFrame with SHAP values and top 10 features
    shap_df = pd.DataFrame(shap_values, columns=names)
    shap_df['abs_shap_values_mean'] = np.abs(shap_values).mean(axis=1)

    shap_df_top_10 = shap_df[['abs_shap_values_mean'] + top_10_features.tolist()]
    print(shap_df_top_10.head())
    
    return top_10_features.tolist()
