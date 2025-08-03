import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Veri seti
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Train/Test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 4. Olasılık tahminleri (ROC için gereklidir)
y_probs = model.predict_proba(X_test)[:, 1]  # sadece pozitif sınıfın olasılığı

# 5. ROC Curve verileri
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

# 6. ROC Grafiği
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange')
plt.plot([0,1], [0,1], linestyle='--', color='gray', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
