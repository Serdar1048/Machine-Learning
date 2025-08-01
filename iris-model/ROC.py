from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Veri setini yükle
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Eğitimi bozmak için çok az veriyle eğitim yapıyoruz!
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, random_state=42)

# 3. Modeli tanımla ve eğit (veri az olduğu için model kötü olacak)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 4. Test verisiyle tahmin yap
y_probs = model.predict_proba(X_test)[:, 1]

# 5. ROC eğrisi ve AUC hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

# 6. ROC grafiğini çiz
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Kötü Eğitimli Modelin ROC Eğrisi")
plt.legend()
plt.grid(True)
plt.show()
