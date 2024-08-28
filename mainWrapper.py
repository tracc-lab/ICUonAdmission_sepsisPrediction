from Methods_utils import methods as FeatSel
import wrap_AdvancedAnalysis_wOnset_wAUROC as HoldoutAnalysis
import pipeline_wHeatmap_imbalanced_AllAUROC as FeatureSelector

first_part = False
if __name__ == '__main__':
    print("Welcome: ")

    data = input("Do you want to insert custom data path? Hit enter if you want to skip and use the default. ")
    print("Your choice was: ", data )
    
    if data == '' or data == '\n':
        data_path = 'C:/Users/aa36.MEDMA/Desktop/Franzi/CC_QtJune/New_Bianka/fbentriesProgV2.csv'
    else:
        data_path = data

    #data_path = 'C:/Users/aa36.MEDMA/Desktop/Franzi/CC_QtJune/New_Bianka/fbentriesProgV2.csv'
    
    CV_nr = input("Please type how many folds should the cross-validation in the feature selection and prediction part to have/had. If no input is given, 10 is the default. ")
    if CV_nr == '' or data == '\n':
        CV_nr = 10
        
    print("\n")
    first_part = input("Do you wish to run the first part of the pipeline with the feature selection and prediction? \n If yes, type 1. Otherwise type 0. \n If no input is given, 0 (no) is the default. ")
    if first_part == '' or first_part == '\n':
        first_part = False
    elif first_part == 1:
        first_part = True
    elif first_part == 0:
        first_part = False

    if first_part == True:
        print("\n")
        print("The extras is a short term for extra printing of a heatmap showing if a feature was selected in each of the folds. \n This is repeated for each feature selection strategy.\n This is also a short term for each fold having a .csv file.")
        extras = input("Please type 1 if you wish to print extras or 0 if not. If no input is given, 0 is the default. ")
        if extras == '' or extras == '\n':
            extras = False
        elif extras == 1:
            extras = True
        elif extras == 0:
            extras = False
        
        print("\n")
        wish_toPlot_AUROC = input("Please type 1 if you wish to plot the AUROCs or type 0 if not. If no input is given, 0 is the default. ")
        if wish_toPlot_AUROC == '' or wish_toPlot_AUROC == '\n':
            wish_toPlot_AUROC = False
        elif wish_toPlot_AUROC == 1:
            wish_toPlot_AUROC = True
        elif wish_toPlot_AUROC == 0:
            wish_toPlot_AUROC = False
            
        print("\n")    
        wish_toPlot_AUPRC= input("Please type 1 if you wish to plot the AUPRCs or type 0 if not. If no input is given, 0 is the default. ")
        if wish_toPlot_AUPRC == '' or wish_toPlot_AUPRC == '\n':
            wish_toPlot_AUPRC = False
        elif wish_toPlot_AUPRC == 1:
            wish_toPlot_AUPRC = True
        elif wish_toPlot_AUPRC == 0:
            wish_toPlot_AUPRC = False
        
        
        print(f"Your selections are \n - Number of folds: {CV_nr}, \n - Extras {extras}, \n - Plot AUROCs? {wish_toPlot_AUROC}, \n - Plot AUPRCs? {wish_toPlot_AUPRC}")
    
    
    print("\n")    
    number_ofIterations= input("Please type how many runs do you wish to happen for the hold-out evaluation in the last part of the pipeline. Minimum is 2. Default is 3. ")
    if number_ofIterations == '' or number_ofIterations == '\n':
        number_ofIterations = 3 #minimum is 2
    else:
        number_ofIterations = int(number_ofIterations)
    
    if first_part == True:
        FeatureSelector.featureSelection_andPredictions(CV_nr, extras, wish_toPlot_AUROC, wish_toPlot_AUPRC)
        
    HoldoutAnalysis.wrapAdvancedAnalysis(data_path, CV_nr, number_ofIterations)