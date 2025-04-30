import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Sample benchmark results for demonstration
models = ["DeepCMorph", "Shest (User Model)"]
dice_scores = [0.832, 0.790]
pq_scores = [0.368, 0.305]
accuracy = [0.827, 0.781]
f1_lymphocyte = [0.85, 0.78]

# Create DataFrame
benchmark_df = pd.DataFrame({
    "Model": models,
    "Mean Dice": dice_scores,
    "Panoptic Quality": pq_scores,
    "Total Accuracy": accuracy,
    "F1-score (Lymphocyte)": f1_lymphocyte
})

# Display the table
import ace_tools as tools; tools.display_dataframe_to_user(name="Cell Type Prediction Benchmark", dataframe=benchmark_df)

# Plot a bar chart comparison for visual reference
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.2

ax.bar(x - 1.5*width, dice_scores, width, label='Mean Dice')
ax.bar(x - 0.5*width, pq_scores, width, label='PQ Score')
ax.bar(x + 0.5*width, accuracy, width, label='Total Accuracy')
ax.bar(x + 1.5*width, f1_lymphocyte, width, label='F1 (Lymphocyte)')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Scores")
ax.set_title("Cell Type Prediction Model Benchmark")
ax.legend()
plt.tight_layout()
plt.show()
