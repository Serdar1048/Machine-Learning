# 1. Gerekli kütüphaneleri içe aktar
from sklearn.linear_model import LinearRegression  
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np  # Sıralama için eklendi

# 2. underfitting_example.csv dosyasını oku
df = pd.read_csv("underfitting/underfitting_example.csv")

# 3. Veriyi X ve y olarak ayır
X = df[["feature"]]
Y = df["target"]

# 4. Veriyi eğitim ve test olarak böl (train_test_split)
x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size= 0.2, random_state=42)

# 5. Çok basit bir model oluştur (örneğin LinearRegression)
model = LinearRegression()
# ↑ Bu model yeterince karmaşık değil, veriyi düzgün öğrenemiyor (underfitting)

# underfitting'i düzeltmek için Polynomial Regression kullanıyoruz (3. dereceden polinom)
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

# 6. Modeli eğit (fit)
model.fit(x_train, y_train)

# 7. Tahmin yap
y_pred = model.predict(x_test)

# 🔧 NOT: x_test sıralı olmadığı için plt.plot() yeşil çizgiyi düzgün çizemiyor.
# Bu yüzden sıralama yapılır:
sorted_index = np.argsort(x_test.values.flatten())  # x_test değerlerini sırala
x_sorted = x_test.values.flatten()[sorted_index]    # sıralı x değerleri
y_sorted = y_pred[sorted_index]                     # aynı sırayla tahmin edilen y değerleri

# 8. Gerçek verileri ve tahminleri çizerek karşılaştır

# Eğitim verisini kırmızı noktalarla göster
plt.scatter(x_train, y_train, color="red", label="Eğitim Verisi")

# Test verisini mavi noktalarla göster
plt.scatter(x_test, y_test, color="blue", label="Gerçek Test Verisi")

# ❌ Önceki hali (çizgiler dağınıktı):
# plt.plot(x_test, y_pred, color="green", linewidth=2, label="Tahmin Doğrusu")

# ✅ DÜZELTİLMİŞ HALİ: Sıralı veriyle düzgün çizgi elde ederiz
plt.plot(x_sorted, y_sorted, color="green", linewidth=2, label="Tahmin Doğrusu")

# Grafik başlık ve etiketler
plt.title("Normal Model")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()
