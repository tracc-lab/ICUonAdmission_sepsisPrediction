
#%% Imports
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import train_test_split


import sys
sys.path.append('./Methods_utils')
import Methods_utils.methods as custom
import Methods_utils.methods_heatmap as heatmap


extras = False
wish_toPlot_AUROC = False
wish_toPlot_AUPRC = False


#%% Data reading and cleaning
def getData(data_location):
    data_path = 'C:/Users/aa36.MEDMA/Desktop/Franzi/CC_QtJune/New_Bianka/fbentriesProgV2.csv'
    data = pd.read_csv(data_path, encoding='latin-1', sep='~')
    print(data.columns.values)

    data2 = data[['c_gender', 'c_vor_diab', 'c_vor_herz' ,'c_vor_atem' ,'c_vor_alko',
    'c_vor_smok', 'c_vor_kidn' ,'c_vor_canc', 'c_ek', 'c_pct', 'c_mechventil',
    'c_dialyse', 'c_ecmo_pecla', 'c_picco' ,'o_sofa_resp', 'o_sofa_cardio',
    'o_sofa_coag' ,'o_sofa_renal', 'o_sofa_liver','n_alter', 'n_kat', 'n_sapsii',
    'n_bddia' ,'n_bdmit', 'n_bdsys', 'n_herzfr', 'n_temp', 'n_ph', 'n_po2' ,'n_pco2',
    'n_fio2pro' ,'n_sbe', 'n_balance', 'n_laktat', 'n_hb' ,'n_blutz', 'n_calcium',
    'n_kalium' ,'n_leuko' ,'n_thrombo' ,'n_bili', 'n_inr' ,'n_ptt' ,'n_ery', 'n_hct',
    'n_crp', 'n_krea' ,'n_harn' ,'n_sofa_total' ,'n_meanlambda' ,'n_delta', 'n_c']].copy()
    data2.fillna(-1, inplace = True)
    data2.isna()

    y = data['event']
    x = data2
    feature_names = x.columns.values
    print("Working with the following features: ", x.columns.values)

    return x, y, data, feature_names

#%% Heatmap
#*********************************************************************************************************************
#* This block tackles the save of top 10 most selected features among folds and plots the heatmap with the prevalence*
#* of each feature among all feature selection algorithms.                                                           *
#* It can also plot what features were selected how many times by each feat sel algo if extras = True                *
#*********************************************************************************************************************
def heatmapTop10 (CV_nr, shap_folds, rf_folds, xgb_folds, ridge_folds, logistic_folds, X_pool_orig_imbalanced, y_pool_orig_imbalanced, experim):
    if extras == True:
        folds_name_param = experim + str(CV_nr) + 'rf_folds'
        print(experim + str(CV_nr) + 'shap_folds')

        heatmap.heatmap_oneFeatureSelectionCV( shap_folds, experim + str(CV_nr) + 'shap_folds')
        heatmap.heatmap_oneFeatureSelectionCV( rf_folds, experim + str(CV_nr) + 'rf_folds')
        heatmap.heatmap_oneFeatureSelectionCV( xgb_folds, experim + str(CV_nr) + 'xgb_folds')
        heatmap.heatmap_oneFeatureSelectionCV( ridge_folds, experim + str(CV_nr) + 'ridge_folds')
        heatmap.heatmap_oneFeatureSelectionCV( logistic_folds, experim + str(CV_nr) + 'logistic_folds')

    save_name = experim + str(CV_nr)
    print(save_name)
    top10_acrossFolds = heatmap.original_heatmap(save_name, shap_folds, rf_folds, xgb_folds, ridge_folds, logistic_folds)
    # top10_ever = top10_acrossFolds.index
    # top10_ever_list = top10_ever.tolist()
    print(top10_acrossFolds)

    print("#############################################################################################################################")
    #%% Retraining using the top 10 across
    X_pool = X_pool_orig_imbalanced[top10_acrossFolds].copy()
    X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X_pool, y_pool_orig_imbalanced, 
                                                                        stratify=y_pool_orig_imbalanced,
                                                                        test_size=0.2 , 
                                                                        random_state= CV_nr - 1)

    scaler = preprocessing.StandardScaler()#MinMaxScaler()
    X_train = scaler.fit_transform(X_train_unscaled)
    X_test = scaler.fit_transform(X_test_unscaled)

    ## maybe this was not 100% needed, but it is an elegant solution to make sure nothing gets overwritten
    auc_dict_new = {'dummy_majority': [], 'dummy_minority': [] ,'rf': [], 'svm': [], 'xgb': [], 'ridge': [], 'logistic': []}
    auprc_dict_new = {'dummy_majority': [], 'dummy_minority': [] ,'rf': [], 'svm': [], 'xgb': [], 'ridge': [], 'logistic': []}
    #%%Models
    dummy_majority, auc_dummy_majority, fpr_dummy_majority, tpr_dummy_majority, auprc_dummy_majority, precision_dummy_majority, recall_dummy_majority = custom.dummy_clf_majority0(X_train, y_train, X_test, y_test)
    dummy_minority, auc_dummy_minority, fpr_dummy_minority, tpr_dummy_minority, auprc_dummy_minority, precision_dummy_minority, recall_dummy_minority = custom.dummy_clf_minority1(X_train, y_train, X_test, y_test)
    rf, auc_rf, fpr_rf, tpr_rf, auprc_rf, precision_rf, recall_rf = custom.random_forest(X_train, y_train, X_test, y_test)
    svm, auc_svm, fpr_svm, tpr_svm, auprc_svm, precision_svm, recall_svm = custom.svm(X_train, y_train, X_test, y_test)
    xgboost_model, auc_xgboost, fpr_xgboost, tpr_xgboost, auprc_xgboost, precision_xgboost, recall_xgboost = custom.xgboost_clf(X_train, y_train, X_test, y_test)
    ridge, auc_ridge, fpr_ridge, tpr_ridge, auprc_ridge, precision_ridge, recall_ridge = custom.ridge(X_train, y_train, X_test, y_test)
    logistic, auc_logistic, fpr_logistic, tpr_logistic, auprc_logistic, precision_logistic, recall_logistic = custom.logistic(X_train, y_train, X_test, y_test)

    # AUC Dictionary
    auc_dict_new['dummy_majority'].append(auc_dummy_majority)
    auc_dict_new['dummy_minority'].append(auc_dummy_minority)
    auc_dict_new['rf'].append(auc_rf)
    auc_dict_new['svm'].append(auc_svm)
    auc_dict_new['xgb'].append(auc_xgboost)
    auc_dict_new['ridge'].append(auc_ridge)
    auc_dict_new['logistic'].append(auc_logistic)
    # AUPRC Dictionary
    auprc_dict_new['dummy_majority'].append(auprc_dummy_majority)
    auprc_dict_new['dummy_minority'].append(auprc_dummy_minority)
    auprc_dict_new['rf'].append(auprc_rf)
    auprc_dict_new['svm'].append(auprc_svm)
    auprc_dict_new['xgb'].append(auprc_xgboost)
    auprc_dict_new['ridge'].append(auprc_ridge)
    auprc_dict_new['logistic'].append(auprc_logistic)


    #%% Save in a df
    featureSel_andPerformance_top10 = pd.DataFrame(columns=['Iteration', 'Stage', 'Current Feature Selection', 'Selected Features', 'Model', 'Test AUROC', 'Test AUPRC'])


    new_iteration_data = CV_nr - 1 #, iteration_x, iteration_x, iteration_x, iteration_x]

    # new_features_selected = [['All'], features_imp_lasso, features_imp_shap, features_imp_rf,
    #                         features_imp_xgb, features_imp_ridge, features_imp_logistic]

    ml_models = ['dummy_majority', 'dummy_minority', 'rf', 'svm','xgb', 'ridge', 'logistic']
    # Convert the dictionaries to lists to use in the results df

    print("___________________ Printing info about things for df __________")

    # we populate the data one model at a time and the while take care of the feature selection stage
    count_model_entry = 0
    for model_entry in ml_models:
        new_entries_df = pd.DataFrame({'Iteration': new_iteration_data,
                                    'Stage': CV_nr, 
                                    'Current Feature Selection': 'top_10_acrossfold',
                                    'Selected Features': [top10_acrossFolds],
                                    'Model': ml_models[count_model_entry],
                                    'Test AUROC': auc_dict_new[model_entry][-1],  # Use the last value for the current model
                                    'Test AUPRC': auprc_dict_new[model_entry][-1] # because we add the vals of current stage
        })
        # Append the new DataFrame to the original DataFrame
        featureSel_andPerformance_top10 = pd.concat([featureSel_andPerformance_top10, new_entries_df], axis=0, ignore_index=True)
        count_model_entry = count_model_entry + 1

    print("This")
    print(experim)

    featureSel_andPerformance_top10.to_csv("final_stratif.csv")

    #%% AUC Plot for HeatmapTop10 features
    if wish_toPlot_AUROC == True:
        new_rates_fpr = []
        new_rates_fpr.append(fpr_dummy_majority)
        new_rates_fpr.append(fpr_dummy_minority)
        new_rates_fpr.append(fpr_rf)
        new_rates_fpr.append(fpr_svm)
        new_rates_fpr.append(fpr_xgboost)
        new_rates_fpr.append(fpr_ridge)
        new_rates_fpr.append(fpr_logistic)
        # print(new_rates_fpr)

        new_rates_tpr = []
        new_rates_tpr.append(tpr_dummy_majority)
        new_rates_tpr.append(tpr_dummy_minority)
        new_rates_tpr.append(tpr_rf)
        new_rates_tpr.append(tpr_svm)
        new_rates_tpr.append(tpr_xgboost)
        new_rates_tpr.append(tpr_ridge)
        new_rates_tpr.append(tpr_logistic)
        # print(new_rates_tpr)

        new_rates_auc = []
        new_rates_auc.append(auc_dummy_majority)
        new_rates_auc.append(auc_dummy_minority)
        new_rates_auc.append(auc_rf)
        new_rates_auc.append(auc_svm)
        new_rates_auc.append(auc_xgboost)
        new_rates_auc.append(auc_ridge)
        new_rates_auc.append(auc_logistic)
        # print(new_rates_auc)
        custom.plot_auc_models(new_rates_fpr, new_rates_tpr, new_rates_auc, ['Dummy_majority', 'Dummy_minority', 'RF', 'SVM','XGBoost', 'Ridge', 'Logistic'], experim + str(CV_nr - 1) + "final_stratif")

    #%% AUPRC Plot using HeatmapTop10 features
    if wish_toPlot_AUPRC == True:
        new_rates_recall = []
        new_rates_recall.append(recall_dummy_majority)
        new_rates_recall.append(recall_dummy_minority)
        new_rates_recall.append(recall_rf)
        new_rates_recall.append(recall_svm)
        new_rates_recall.append(recall_xgboost)
        new_rates_recall.append(recall_ridge)
        new_rates_recall.append(recall_logistic)
        # print(new_rates_recall)

        new_rates_precision = []
        new_rates_precision.append(precision_dummy_majority)
        new_rates_precision.append(precision_dummy_minority)
        new_rates_precision.append(precision_rf)
        new_rates_precision.append(precision_svm)
        new_rates_precision.append(precision_xgboost)
        new_rates_precision.append(precision_ridge)
        new_rates_precision.append(precision_logistic)
        # print(new_rates_precision)

        new_rates_auprc = []
        new_rates_auprc.append(auprc_dummy_majority)
        new_rates_auprc.append(auprc_dummy_minority)
        new_rates_auprc.append(auprc_rf)
        new_rates_auprc.append(auprc_svm)
        new_rates_auprc.append(auprc_xgboost)
        new_rates_auprc.append(auprc_ridge)
        new_rates_auprc.append(auprc_logistic)
        # print(new_rates_auprc)

        custom.plot_auprc_models(new_rates_recall, new_rates_precision, new_rates_auprc, ['Dummy_majority','Dummy_minority' , 'RF', 'SVM','XGBoost', 'Ridge', 'Logistic'], experim + str(CV_nr) + str(CV_nr - 1) + "final_stratif")


#%% Training the models and using the heatmapTop10 function
#*********************************************************************************************************************
#* This block tackles the save of top 10 most selected features among folds and plots the heatmap with the prevalence*
#* of each feature among all feature selection algorithms.                                                           *
#* It can also plot what features were selected how many times by each feat sel algo if extras = True                *
#*********************************************************************************************************************
def train_featSel_heatmapTop10 (CV_nr):
    # CV_nr = 3#10
    experim = "_pipeline_" + str(CV_nr) + "_"

    featureSel_andPerformance = pd.DataFrame(columns=['Iteration', 'Stage', 'Current Feature Selection', 'Selected Features', 'Model', 'Test AUROC', 'Test AUPRC'])
    featureSel_andPerformance_top10 = pd.DataFrame(columns=['Iteration', 'Stage', 'Current Feature Selection', 'Selected Features', 'Model', 'Test AUROC', 'Test AUPRC'])
    featureSel_andPerformance_CV = pd.DataFrame(columns=['SplitNo', 'Iteration', 'Stage', 'Current Feature Selection', 'Selected Features', 'Model', 'Test AUROC', 'Test AUPRC'])
    crt_feat_sel_options = ['none', 'lasso', 'shap', 'rf', 'xgb', 'ridge', 'logistic' ]
    features_imp_rf = []
    features_imp_xgb =[]
    features_imp_ridge = []
    features_imp_logistic = []
    features_imp_shap = []
    features_imp_lasso = ['c_gender', 'c_vor_alko', 'c_mechventil', 'c_picco', 
                'o_sofa_resp', 'o_sofa_liver', 'n_alter', 'n_bdmit', 
                'n_bdsys', 'n_balance', 'n_laktat', 'n_ptt', 'n_ery', 
                'o_sofa_cardio', 'o_sofa_liver', 'n_thrombo', 'n_crp',  
                'n_crp', 'n_sofa_total', 'n_meanlambda', 'n_delta', 'n_c']

    shap_folds = []
    rf_folds = []
    xgb_folds = []
    ridge_folds = []
    logistic_folds = []

    allAUROCs = pd.DataFrame(columns=['Iteration', 'Stage', 'Model name', 'AUROC', 'TPR', 'FPR'])

    #%% Data split, ml training, feature selection etc
    x,y, data, names = getData(data_location='C:/Users/aa36.MEDMA/Desktop/Franzi/CC_QtJune/New_Bianka/fbentriesProgV2.csv')
    X_pool_orig_imbalanced, X_test_holdout, y_pool_orig_imbalanced, y_test_holdout = train_test_split(x, y,
                                                    stratify=y, 
                                                    test_size=0.1,
                                                    random_state=1)
    print("Cases and controls hold-out data: \n", y_test_holdout.value_counts())
    print("Cases and controls remaining data imbalanced: \n", y_pool_orig_imbalanced.value_counts())

    # Get the indices of the holdout set
    holdout_indices = X_test_holdout.index

    # Retrieve the corresponding IDs from the original dataset
    holdout_ids = data.loc[holdout_indices, 'id']

    # Print the IDs of subjects in the holdout set
    print("IDs of subjects in the holdout set:")
    print(holdout_ids.to_list())

    print("Cases and controls undersampled data BALANCED: ", y_pool_orig_imbalanced.value_counts() )

    skf = StratifiedKFold(n_splits=CV_nr, shuffle=True, random_state=42)
        
    #%% Training the models
    #*********************************************************************************************************************
    #* This big chunk of code contains a massive for that iterates through all the folds.                                *
    #* It also saves information about feature selection stage and ml performance in a df to become .csv                 *
    #* It can also print AUROC and AUPRC for the models                                                                  *
    #*********************************************************************************************************************

    ## beginning of very big for
    for iteration_x, (train_index, test_index) in enumerate(skf.split(X_pool_orig_imbalanced, y_pool_orig_imbalanced), 1):
        print("-------------- Started working on fold " + str(iteration_x) + " --------------")
        stage_cnt = 0  # the feature selection stage
        iteration_arr = []
        stage_arr = []
        while stage_cnt <= 6:
            print("Currently doing magic in fold " + str(iteration_x) + ", feature selection stage " + str(stage_cnt) + "...")
            # Feature selection based on stage count
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
            elif stage_cnt == 6:
                X_pool = X_pool_orig_imbalanced[features_imp_logistic].copy()

            # Split data into train and test using KFold indices
            X_train_unscaled, X_test_unscaled = X_pool.iloc[train_index], X_pool.iloc[test_index]
            y_train, y_test = y_pool_orig_imbalanced.iloc[train_index], y_pool_orig_imbalanced.iloc[test_index]

            scaler = preprocessing.StandardScaler()#MinMaxScaler()
            X_train = scaler.fit_transform(X_train_unscaled)
            X_test = scaler.fit_transform(X_test_unscaled)
        
            ## maybe this was not 100% needed, but it is an elegant solution to make sure nothing gets overwritten
            auc_dict = {'dummy_majority': [], 'dummy_minority': [] ,'rf': [], 'svm': [], 'xgb': [], 'ridge': [], 'logistic': []}
            auprc_dict = {'dummy_majority': [], 'dummy_minority': [] ,'rf': [], 'svm': [], 'xgb': [], 'ridge': [], 'logistic': []}
            #%%Models
            dummy_majority, auc_dummy_majority, fpr_dummy_majority, tpr_dummy_majority, auprc_dummy_majority, precision_dummy_majority, recall_dummy_majority = custom.dummy_clf_majority0(X_train, y_train, X_test, y_test)
            dummy_minority, auc_dummy_minority, fpr_dummy_minority, tpr_dummy_minority, auprc_dummy_minority, precision_dummy_minority, recall_dummy_minority = custom.dummy_clf_minority1(X_train, y_train, X_test, y_test)
            rf, auc_rf, fpr_rf, tpr_rf, auprc_rf, precision_rf, recall_rf = custom.random_forest(X_train, y_train, X_test, y_test)
            svm, auc_svm, fpr_svm, tpr_svm, auprc_svm, precision_svm, recall_svm = custom.svm(X_train, y_train, X_test, y_test)
            xgboost_model, auc_xgboost, fpr_xgboost, tpr_xgboost, auprc_xgboost, precision_xgboost, recall_xgboost = custom.xgboost_clf(X_train, y_train, X_test, y_test)
            ridge, auc_ridge, fpr_ridge, tpr_ridge, auprc_ridge, precision_ridge, recall_ridge = custom.ridge(X_train, y_train, X_test, y_test)
            logistic, auc_logistic, fpr_logistic, tpr_logistic, auprc_logistic, precision_logistic, recall_logistic = custom.logistic(X_train, y_train, X_test, y_test)

            # AUC Dictionary
            auc_dict['dummy_majority'].append(auc_dummy_majority)
            auc_dict['dummy_minority'].append(auc_dummy_minority)
            auc_dict['rf'].append(auc_rf)
            auc_dict['svm'].append(auc_svm)
            auc_dict['xgb'].append(auc_xgboost)
            auc_dict['ridge'].append(auc_ridge)
            auc_dict['logistic'].append(auc_logistic)
            # AUPRC Dictionary
            auprc_dict['dummy_majority'].append(auprc_dummy_majority)
            auprc_dict['dummy_minority'].append(auprc_dummy_minority)
            auprc_dict['rf'].append(auprc_rf)
            auprc_dict['svm'].append(auprc_svm)
            auprc_dict['xgb'].append(auprc_xgboost)
            auprc_dict['ridge'].append(auprc_ridge)
            auprc_dict['logistic'].append(auprc_logistic)
            
            #%%Feature importance
            ## the feature importance is computed only once, in stage 0, when we use all the features to make a prediction
            if stage_cnt == 0:
                features_imp_rf = custom.feat_imp_rf(rf, names)
                features_imp_xgb = custom.feat_imp_xgb(xgboost_model, names)
                features_imp_ridge = custom.feat_imp_ridge(ridge, names)
                features_imp_logistic = custom.feat_imp_logistic(logistic, names)
        
                shap_kind = ''
                if extras == True:
                    print("SHAP used in iteration: ", iteration_x)
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                list_allMLmodels = ['dummy_majority', 'dummy_minority', 'rf', 'svm','xgb', 'ridge', 'logistic']
                models_auprc_list = [auprc_dummy_majority, auprc_dummy_minority, auprc_rf, auprc_svm, auprc_xgboost, auprc_ridge, auprc_logistic]
                models_list = [dummy_majority, dummy_minority , rf, svm, xgboost_model, ridge, logistic]
                
                # the shap is used only on the most performing method for this particular data split according to AUPRC
                # because of how shap is implemented, this elif is needed. see methods for more. it's a whole thing
                temp = 0
                maxim_auprc = max(models_auprc_list)
                for auprc in models_auprc_list:
                    if auprc == maxim_auprc:
                        shap_model = models_list[temp]
                        print("HIGHEST AUPRC MODEL: ", list_allMLmodels[temp])
                        if list_allMLmodels[temp] == 'rf':
                            shap_kind = 'rf'
                        elif list_allMLmodels[temp] == 'svm':
                            shap_kind = 'svm'
                        elif list_allMLmodels[temp] == 'xgb':
                            shap_kind = 'xgb'
                        elif list_allMLmodels[temp] == 'ridge' or list_allMLmodels[temp] == 'logistic':
                            shap_kind = 'linear'
                    temp = temp + 1
                    
                print(shap_kind)
                features_imp_shap = custom.feat_imp_shap(shap_model, names, shap_kind, X_test)
                
                shap_folds.append(features_imp_shap)
                rf_folds.append(features_imp_rf)
                xgb_folds.append(features_imp_xgb)
                ridge_folds.append(features_imp_ridge)
                logistic_folds.append(features_imp_logistic)
            #%% Save in a df
            crt_feat_sel = crt_feat_sel_options[stage_cnt]
            new_iteration_data = iteration_x #, iteration_x, iteration_x, iteration_x, iteration_x]
            
            new_features_selected = [['All'], features_imp_lasso, features_imp_shap, features_imp_rf,
                                    features_imp_xgb, features_imp_ridge, features_imp_logistic]
        
            ml_models = ['dummy_majority', 'dummy_minority', 'rf', 'svm','xgb', 'ridge', 'logistic']
            # Convert the dictionaries to lists to use in the results df
            auc_list = [auc_dict[model] for model in ml_models]
            auprc_list = [auprc_dict[model] for model in ml_models]
            
            if extras == True:
                print("___________________ Printing info about things for df __________")
                # print("Iter length: ", len(new_iteration_data), new_iteration_data)
                print("Crt feat sel length: ", len(crt_feat_sel), crt_feat_sel)
                print("Sel Feat length: ", len(new_features_selected[stage_cnt]), new_features_selected[stage_cnt])
                print("Model length: ", len(ml_models))
                print("AUROC length: ", len(auc_list), auc_list)
                print("AUPRC length: ", len(auprc_list), auprc_list)
            
            # we populate the data one model at a time and the while take care of the feature selection stage
            count_model_entry = 0
            for model_entry in ml_models:
                new_entries_df = pd.DataFrame({'Iteration': new_iteration_data,
                                            'Stage': stage_cnt, 
                                            'Current Feature Selection': crt_feat_sel,
                                            'Selected Features': [new_features_selected[stage_cnt]],
                                            'Model': ml_models[count_model_entry],
                                            'Test AUROC': auc_dict[model_entry][-1],  # Use the last value for the current model
                                            'Test AUPRC': auprc_dict[model_entry][-1] # because we add the vals of current stage
                })
                
                featureSel_andPerformance_CV_newEntries = pd.DataFrame({'SplitNo': iteration_x,
                                                            'Iteration': new_iteration_data,
                                                            'Stage': stage_cnt, 
                                                            'Current Feature Selection': crt_feat_sel,
                                                            'Selected Features': [new_features_selected[stage_cnt]],
                                                            'Model': ml_models[count_model_entry],
                                                            'Test AUROC': auc_dict[model_entry][-1], 
                                                            'Test AUPRC': auprc_dict[model_entry][-1] 
                                                            })
                # append the new df to the original df. aka populate needed df
                if extras == True:
                    featureSel_andPerformance = pd.concat([featureSel_andPerformance, new_entries_df], axis=0, ignore_index=True) # each iteration will have a .csv
                featureSel_andPerformance_CV = pd.concat([featureSel_andPerformance_CV, featureSel_andPerformance_CV_newEntries], axis=0, ignore_index=True) #single .csv to contain all model info
                count_model_entry = count_model_entry + 1
        
            print("This")
            print(experim)
            #%% AUC Plot
            new_rates_fpr = []
            new_rates_fpr.append(fpr_dummy_majority)
            new_rates_fpr.append(fpr_dummy_minority)
            new_rates_fpr.append(fpr_rf)
            new_rates_fpr.append(fpr_svm)
            new_rates_fpr.append(fpr_xgboost)
            new_rates_fpr.append(fpr_ridge)
            new_rates_fpr.append(fpr_logistic)
            # print(new_rates_fpr)

            new_rates_tpr = []
            new_rates_tpr.append(tpr_dummy_majority)
            new_rates_tpr.append(tpr_dummy_minority)
            new_rates_tpr.append(tpr_rf)
            new_rates_tpr.append(tpr_svm)
            new_rates_tpr.append(tpr_xgboost)
            new_rates_tpr.append(tpr_ridge)
            new_rates_tpr.append(tpr_logistic)
            # print(new_rates_tpr)

            new_rates_auc = []
            new_rates_auc.append(auc_dummy_majority)
            new_rates_auc.append(auc_dummy_minority)
            new_rates_auc.append(auc_rf)
            new_rates_auc.append(auc_svm)
            new_rates_auc.append(auc_xgboost)
            new_rates_auc.append(auc_ridge)
            new_rates_auc.append(auc_logistic)
            # print(new_rates_auc)
            
            if wish_toPlot_AUROC == True:   
                custom.plot_auc_models(new_rates_fpr, new_rates_tpr, new_rates_auc, ['Dummy_majority', 'Dummy_minority', 'RF', 'SVM','XGBoost', 'Ridge', 'Logistic'], experim + str(stage_cnt) + "final_stratif")
                
            ### store plotting info so you can print different aspects later as needed
            
            counter_aucAll = 0
            iteration_index = iteration_x - 1
            model_to_add = ''
            iteration = 0
            stage_to_add = 0
            auc_rates_to_add = 0
            fpr_rates_to_add = []
            tpr_rates_to_add = []
            
            for model_entry in ml_models:
                model_to_add = model_entry
                iteration = iteration_x
                stage_to_add = stage_cnt
                auc_rates_to_add = new_rates_auc[counter_aucAll]
                fpr_rates_to_add = [new_rates_fpr[counter_aucAll]]
                tpr_rates_to_add = [new_rates_tpr[counter_aucAll]]
        
                print("Adding now the TPRs: ", type(tpr_rates_to_add), tpr_rates_to_add)
                allAUROCs_plus = pd.DataFrame({'Iteration': iteration, 
                                        'Stage': stage_to_add,
                                        'Model name': model_to_add ,
                                        'AUROC': auc_rates_to_add,
                                        'TPR': tpr_rates_to_add ,
                                        'FPR': fpr_rates_to_add   
                                        })
                allAUROCs = pd.concat([allAUROCs, allAUROCs_plus], axis=0, ignore_index=True)
                counter_aucAll = counter_aucAll + 1
            
            #%% AUPRC Plot
            if wish_toPlot_AUPRC == True:
                new_rates_recall = []
                new_rates_recall.append(recall_dummy_majority)
                new_rates_recall.append(recall_dummy_minority)
                new_rates_recall.append(recall_rf)
                new_rates_recall.append(recall_svm)
                new_rates_recall.append(recall_xgboost)
                new_rates_recall.append(recall_ridge)
                new_rates_recall.append(recall_logistic)
                # print(new_rates_recall)

                new_rates_precision = []
                new_rates_precision.append(precision_dummy_majority)
                new_rates_precision.append(precision_dummy_minority)
                new_rates_precision.append(precision_rf)
                new_rates_precision.append(precision_svm)
                new_rates_precision.append(precision_xgboost)
                new_rates_precision.append(precision_ridge)
                new_rates_precision.append(precision_logistic)
                # print(new_rates_precision)

                new_rates_auprc = []
                new_rates_auprc.append(auprc_dummy_majority)
                new_rates_auprc.append(auprc_dummy_minority)
                new_rates_auprc.append(auprc_rf)
                new_rates_auprc.append(auprc_svm)
                new_rates_auprc.append(auprc_xgboost)
                new_rates_auprc.append(auprc_ridge)
                new_rates_auprc.append(auprc_logistic)
                # print(new_rates_auprc)

                custom.plot_auprc_models(new_rates_recall, new_rates_precision, new_rates_auprc, ['Dummy_majority','Dummy_minority' , 'RF', 'SVM','XGBoost', 'Ridge', 'Logistic'], experim + str(stage_cnt) + str(iteration_x) + "final_stratif")

            stage_cnt = stage_cnt + 1
            print("Beginning stage: ", stage_cnt)
            pass
        
        #%%
        print("Cases and controls hold-out data: ", y_test.value_counts() )
        
        if extras == True:
            featureSel_andPerformance.to_csv('results' + experim + '_split_' + str(iteration_x) + '.csv', index=False, sep = "~")
    ## end of very big for

    # this is where we save a .csv file with all the results from all folds of all feature selection strategies
    featureSel_andPerformance_CV.to_csv('resultsAllCVs' + experim + '_split_' + str(iteration_x) + '.csv', index=False, sep = "~")

    # save the TPRm FPR and AUROC information for all models accross all folds and all feature selection for further plots 
    # outside of this script
    allAUROCs['TPR'] = allAUROCs['TPR'].apply(lambda x: ','.join(map(str, x)))
    allAUROCs['FPR'] = allAUROCs['FPR'].apply(lambda x: ','.join(map(str, x)))
    allAUROCs.to_csv('allAUROCs' + experim  + '.csv', index=False, sep = "~")
    # custom.plot_auc_allModels (models_allModels, fprs_allModels, tprs_allModels, auc_allModels, experim)

    heatmapTop10 (CV_nr, shap_folds, rf_folds, xgb_folds, ridge_folds, logistic_folds, X_pool_orig_imbalanced, y_pool_orig_imbalanced, experim)


def featureSelection_andPredictions (CV_nr, extras, wish_toPlot_AUROC, wish_toPlot_AUPRC):
    train_featSel_heatmapTop10(CV_nr)
    
