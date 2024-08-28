# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:00:25 2024

@author: Asus
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# gets a df ad plots one plot for each iteration and colors the lines by stage
def plotPerIteration_colorByStage (allAUROCs):
        # Group data by iteration
    grouped = allAUROCs.groupby('Iteration')
    
    # Determine unique stages for coloring
    unique_stages = allAUROCs['Stage'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_stages))  # Choosing a colormap with enough distinct colors
    
    # Plot each iteration on separate subplots
    for i, (iteration, group) in enumerate(grouped):
        plt.figure(figsize=(8, 6))
        for stage, color in zip(unique_stages, colors.colors):
            stage_data = group[group['Stage'] == stage]
            for index, row in stage_data.iterrows():
                plt.plot(row['FPR'], row['TPR'], label=f"Feature selection strategy {stage}, {row['Model name']}", color=color)
                
        # Plot baseline (AUROC=0.5)
        plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline (AUROC=0.5)', color='gray')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUROC curves - Data split {iteration}')
        plt.legend()
        plt.show()

def plotPerIteration_colorByStage_bestStage(allAUROCs):
    # Group data by iteration
    grouped = allAUROCs.groupby('Iteration')
    
    # Plot each iteration on separate subplots
    for i, (iteration, group) in enumerate(grouped):
        plt.figure(figsize=(8, 6))
        
        # Find the stage with the best AUROC for this iteration
        best_stage_data = group.loc[group['AUROC'].idxmax()]
        best_stage = best_stage_data['Stage']
        
        # Filter data to include only models from the best stage
        best_stage_models = group[group['Stage'] == best_stage]
        
        # Plot AUROC curves for all models from the best stage
        for index, row in best_stage_models.iterrows():
            plt.plot(row['FPR'], row['TPR'], label=f"Feature selection strategy {best_stage}, {row['Model name']} (AUROC={row['AUROC']:.2f})")
                
        # Plot baseline (AUROC=0.5)
        plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline (AUROC=0.5)', color='gray')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUROC curves - Data split {iteration}')
        plt.legend()
        plt.savefig("BestStage_iter_" + str(iteration) + ".png")
        plt.show()
        
def plotPerIteration_colorByStage_bestModel (allAUROCs):
    # Group data by iteration
    grouped = allAUROCs.groupby('Iteration')
    
    # Plot each iteration on separate subplots
    for i, (iteration, group) in enumerate(grouped):
        plt.figure(figsize=(8, 6))
        for stage in group['Stage'].unique():
            stage_data = group[group['Stage'] == stage]
            
            # Find the row with the highest AUROC for the current stage
            best_model_row = stage_data.loc[stage_data['AUROC'].idxmax()]
            
            # Plot the AUROC curve for the best model in the stage
            plt.plot(best_model_row['FPR'], best_model_row['TPR'],
                     label=f"Feature selection strategy {stage}, {best_model_row['Model name']} (AUROC={best_model_row['AUROC']:.2f})")
        
        # Plot baseline (AUROC=0.5)
        plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline (AUROC=0.5)', color='gray')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUROC curves - Data split {iteration}')
        plt.legend()
        plt.savefig("BestModels_iter_" + str(iteration) + ".png")
        plt.show()

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


plotPerIteration_colorByStage(allAUROCs)
plotPerIteration_colorByStage_bestModel (allAUROCs)
plotPerIteration_colorByStage_bestStage(allAUROCs)
