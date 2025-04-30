import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

# Sample function to evaluate predictions vs ground truth for Lizard dataset
def evaluate_cell_type_predictions(y_true, y_pred, class_names):
    """
    y_true and y_pred are expected to be 2D numpy arrays of the same shape
    containing integer class labels.
    """
    assert y_true.shape == y_pred.shape, "Shape mismatch between prediction and ground truth"
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    results = {
        "Accuracy": accuracy_score(y_true_flat, y_pred_flat),
        "Macro Precision": precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
        "Macro Recall": recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
        "Macro F1-score": f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
    }

    # Per-class F1-score
    per_class_f1 = f1_score(y_true_flat, y_pred_flat, average=None, labels=range(len(class_names)), zero_division=0)
    results.update({f"F1 ({name})": f1 for name, f1 in zip(class_names, per_class_f1)})

    return results

# Example placeholder arrays
num_classes = 6
class_names = ["Epithelial", "Connective", "Lymphocyte", "Plasma", "Neutrophil", "Eosinophil"]
y_true_sample = np.random.randint(0, num_classes, (256, 256))
y_pred_sample = np.random.randint(0, num_classes, (256, 256))

# Run evaluation
results_dict = evaluate_cell_type_predictions(y_true_sample, y_pred_sample, class_names)
results_df = pd.DataFrame(list(results_dict.items()), columns=["Metric", "Value"])
tools.display_dataframe_to_user(name="Lizard Dataset Evaluation Example", dataframe=results_df)
