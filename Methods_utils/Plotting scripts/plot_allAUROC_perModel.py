# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:00:25 2024

@author: Asus
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plotPerModel_colorByIteration_givenStage(allAUROCs, model_name, given_stage):
    if given_stage == 1:
        print("Skipping plotting for Stage 1.")
        return
    
    custom_strings = {
        0: "No Feature Selection",  #skip 1 because it's LASSO
        2: "SHAP",
        3: "rf feature selection",
        4: "xgb feature selection",
        5: "ridge feature selection",
        6: "logistic feature selection",
        # Add more custom strings for other stages if needed
    }
    
    if given_stage not in custom_strings:
        print(f"Custom string not defined for stage {given_stage}.")
        return
    
    custom_str = custom_strings[given_stage]
    
    print("Plotting the AUROC from all iterations for " + model_name + " model for stage: " + custom_str)
    # Group data by iteration
    filtered_data = allAUROCs[(allAUROCs['Model name'] == model_name) & (allAUROCs['Stage'] == given_stage)]
    
    # Group data by iteration
    grouped = filtered_data.groupby('Iteration')
    
    plt.figure(figsize=(8, 6))
    
    # Plot AUROC curves for all models from the specified stage for each iteration
    for iteration, group in grouped:
        for index, row in group.iterrows():
            plt.plot(row['FPR'], row['TPR'], label=f"Data Split {iteration}, {model_name} (AUROC={row['AUROC']:.2f})")
                
    # Plot baseline (AUROC=0.5)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline (AUROC=0.5)', color='gray')
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUROC curves - {model_name} - Feature selection strategy {given_stage}')
    plt.legend()
    
    directory = "./plots_perModel_" + model_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the plot as an image
    plt.savefig(directory + f"{model_name}_Stage_{given_stage}_allIterations.png")
    
    plt.show()

        
def plotPerModel_colorByStage(allAUROCs, model_name):
    print("Plotting the AUROC from all iterations, all stages for " + model_name + "coloring by Stage..." )
    
    # Filter data by model name and given stage
    filtered_data = allAUROCs[(allAUROCs['Model name'] == model_name)]
    
    # Group data by iteration
    grouped = filtered_data.groupby('Iteration')
    
    plt.figure(figsize=(8, 6))
    
    # Define a color map for different stages
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped)))
    
    # Plot AUROC curves for all models from the specified stage for each iteration
    for i, (iteration, group) in enumerate(grouped):
        for index, row in group.iterrows():
            plt.plot(row['FPR'], row['TPR'], label=f"Data split {iteration}, {model_name} (AUROC={row['AUROC']:.2f})", color=colors[i])
                
    # Plot baseline (AUROC=0.5)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline (AUROC=0.5)', color='gray')
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUROC curves - {model_name}')
    plt.legend()
    plt.savefig("./plots_perModel_" + model_name + "/" + f"{model_name}_ColorByStage_allIterationsBIG.png")
    plt.show()
    # plots_perModel_rf
    
plot_csv_path = '../allAUROCs_pipeline_10_.csv'
data_plots = pd.read_csv(plot_csv_path, encoding='latin-1', sep='~')

allAUROCs = data_plots
print(data_plots)

# Convert string representations back to lists
allAUROCs['TPR'] = allAUROCs['TPR'].apply(lambda x: list(map(float, x.split(','))))
allAUROCs['FPR'] = allAUROCs['FPR'].apply(lambda x: list(map(float, x.split(','))))

# Iterate over each row in the DataFrame
for index, row in allAUROCs.iterrows():
    # Plot the AUROC curve
    plt.plot(row['FPR'], row['TPR'], label=f"{row['Iteration']}, Stage {row['Stage']}, {row['Model name']}")

# Add labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC Curves')
plt.legend()

plt.savefig("All_omg.png")
# Show plot
plt.show()


# plotPerModel_colorByStage(allAUROCs, 'rf')  # not intelligible

for stage_number in range (0,7):
    plotPerModel_colorByIteration_givenStage(allAUROCs, 'rf', stage_number)
    plotPerModel_colorByIteration_givenStage(allAUROCs, 'svm', stage_number)
    plotPerModel_colorByIteration_givenStage(allAUROCs, 'xgb', stage_number)
    plotPerModel_colorByIteration_givenStage(allAUROCs, 'ridge', stage_number)
    plotPerModel_colorByIteration_givenStage(allAUROCs, 'logistic', stage_number)
    