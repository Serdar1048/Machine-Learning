# 1. Gerekli kütüphaneleri içe aktar
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
import numpy as np

# 2. Veriyi yükle
iris = load_iris()
X = iris.data
y = iris.target

# 3. Modeli oluştur
model = LogisticRegression(max_iter=200)

# 4. K-Fold objesi oluştur (veriyi 5 parçaya bölecek)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5. Cross-validation uygula (doğruluk metrikleri döner)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# 6. Sonuçları yazdır
print("Her fold için doğruluk skorları:", scores)
print(f"Ortalama doğruluk: %{np.mean(scores) * 100:.2f}")
