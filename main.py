from sklearn.metrics import classification_report
import numpy as np

y_true = np.load("lizard_gt_labels.npy")  # 정답
y_pred = np.load("shest_pred_labels.npy")  # 예측 결과

print(classification_report(y_true.flatten(), y_pred.flatten(), target_names=cell_type_labels))
