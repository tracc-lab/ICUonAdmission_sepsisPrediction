
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Arial'

plot_csv_path = '../allAUROCs_pipeline_10_.csv'
data_plots = pd.read_csv(plot_csv_path, encoding='latin-1', sep='~')

allAUROCs = data_plots

# remove all entries from dummy in Models and lasso in Stage 1. 
allAUROCs = allAUROCs[~allAUROCs['Model name'].str.contains('dummy', case=False)]
allAUROCs = allAUROCs[allAUROCs['Stage'] != 1]
print(allAUROCs)

#%% Plot Heatmap for all Models, one at a time, in 5 heatmaps with max aUROC
replacement_mapping = {0: 'None', 2: 'SHAP', 3: 'Random Forest', 4: 'XGBoost', 5: 'Ridge Regression', 6: 'Logistic Regression'}
allAUROCs['Stage'] = allAUROCs['Stage'].replace(replacement_mapping)

replacement_mapping_models = {'rf': 'Random Forest', 'svm': 'SVM', 'xgb': 'XGBoost', 'ridge': 'Ridge Regression', 'logistic': 'Logistic Regression'}
allAUROCs['Model name'] = allAUROCs['Model name'].replace(replacement_mapping_models)

columns_order = ['None', 'SHAP', 'Random Forest', 'XGBoost', 'Ridge Regression', 'Logistic Regression']

model_names = allAUROCs['Model name'].unique()

for model_name in model_names:
    model_df = allAUROCs[allAUROCs['Model name'] == model_name]
    
    if model_name == 'rf':
        model_name = 'Random Forest'
    elif  model_name == 'svm':
        model_name = 'SVM'
    elif  model_name == 'xgb':
        model_name = 'XGBoost'
    elif  model_name == 'ridge':
        model_name = 'Ridge Regression'
    elif  model_name == 'logistic':
        model_name = 'Logistic Regression'

    pivot_df = model_df.pivot_table(index='Iteration', columns='Stage', values='AUROC', aggfunc='first')
    
    # Reorder the columns according to desired_order
    pivot_df = pivot_df[columns_order]
    
    # Reorder the columns to have them in ascending order
    # pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    
    # Plotting the heatmap
    plt.figure(figsize=(8.5, 6))
    sns.heatmap(pivot_df, cmap='BuPu', annot=True, fmt=".2f", linewidths=0.5,
                annot_kws={"fontsize": 15+1})
    
    plt.title(f'AUROC Heatmap for {model_name}', fontsize=16+1)
    # plt.title('AUROC Heatmap for ' + r'$\mathbf{' + model_name + '}$', fontsize=13.5) #bolds, but makes RidgeRegression instead of Ridge Regression
    # plt.title('AUROC Heatmap for', fontsize=13.5)
    # plt.text(0.5, 0.95, model_name, fontsize=13.5, fontweight='bold', ha='center', va='center', transform=plt.gca().transAxes)

    plt.xlabel('Feature selection strategy', fontsize = 15+1)
    plt.ylabel('Data split', fontsize = 15+1)
    
    plt.yticks(fontsize=15+1, rotation = 0)  # Adjust fontsize as needed
    plt.xticks(fontsize=15+1, rotation=30, ha='right') 
    
    plt.tight_layout()
    
    plt.savefig("./Heatmap_AUROCs_perModel/AUROC_Heatmap_V2_" + model_name + ".png", dpi = 600)
    plt.show()

#%% Plot Heatmap for all Models in one heatmap with max aUROC
pivot_df_oneHeatmap = allAUROCs.pivot_table(index='Iteration', columns='Stage', values='AUROC', aggfunc='max')

plt.rcParams['font.family'] = 'Arial'

# Step 2: Iterate over each split and find the max AUROC for each stage across models
for i in range(1, 11):
    for stage in pivot_df_oneHeatmap.columns:
        max_auroc = allAUROCs[(allAUROCs['Iteration'] == i) & (allAUROCs['Stage'] == stage)].groupby('Model name')['AUROC'].max()
        max_model = max_auroc.idxmax()
        max_value = max_auroc.max()
        pivot_df_oneHeatmap.loc[i, stage] = max_value

pivot_df_oneHeatmap = pivot_df_oneHeatmap[columns_order]
model_colors = {'Random Forest': '#630C3A', 'SVM': '#27C3C1', 'XGBoost': '#FFC107', 'Ridge Regression': '#7E34F9', 'Logistic Regression': '#E01889'}

# colors = ['#630C3A', '#27C3C1', '#FFC107', '#7E34F9', '#E01889', '#617111','#fe6100',  '#7d413c',
# '#423568', '#5590b4']
# sns.set_palette(sns.color_palette(colors))

unique_models = allAUROCs['Model name'].unique()
print(allAUROCs['Model name'].unique())

cmap = ListedColormap([model_colors[model] for model in allAUROCs['Model name'].unique()])


################### width, height in inches
plt.figure(figsize=(10, 6))
# plt.figure(figsize=(20, 12))
sns.heatmap(pivot_df_oneHeatmap, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, cbar=False,
            annot_kws={"fontsize": 15+1})

plt.title('Maximum AUROC across models', fontsize=16+1)
plt.xlabel('Feature selection strategy', fontsize = 15+1)
plt.ylabel('Data split', fontsize = 15+1)

# Step 6: Create legend patches
legend_patches = [mpatches.Patch(color=color, label=model) for model, color in model_colors.items()]
plt.legend(handles=legend_patches, title='Model Name', bbox_to_anchor=(1, 1), loc='upper left', fontsize=15+1, title_fontsize=15+1)

plt.yticks(fontsize=15+1, rotation = 0)  # Adjust fontsize as needed
plt.xticks(fontsize=15+1, rotation=30, ha='right')  # Adjust fontsize and rotation as needed

plt.tight_layout()

plt.savefig("./Heatmap_AUROCs_bestModels/maxAUROC_Heatmap_5MLs_V2.png", dpi = 600)
plt.show()

