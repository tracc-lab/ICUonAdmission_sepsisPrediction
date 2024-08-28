from Methods_utils import methods as FeatSel
import wrap_AdvancedAnalysis_wOnset_wAUROC as HoldoutAnalysis

first_part = False
if __name__ == '__main__':
    print("THIS IS IT: ")

    data = input("Do you want to insert custom data path? Hit enter if you want to skip and use the default. ")
    print("Your choice was: ", data )
    
    if data == '' or data == '\n':
        data_path = 'C:/Users/aa36.MEDMA/Desktop/Franzi/CC_QtJune/New_Bianka/fbentriesProgV2.csv'
    else:
        data_path = data

    #data_path = 'C:/Users/aa36.MEDMA/Desktop/Franzi/CC_QtJune/New_Bianka/fbentriesProgV2.csv'
    
    if first_part == True:
        FeatSel(data_path)
        
    
    HoldoutAnalysis.wrapAdvancedAnalysis(data_path)