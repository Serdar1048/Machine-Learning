#SUPERVISED
"""
# 1. Gerekli kütüphaneleri içe aktar
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. iris.csv dosyasını pandas ile oku
df = pd.read_csv("iris-model/iris.csv")

# 3. Verinin ilk 5 satırını yazdır (kontrol etmek için)
#print(df.head())

# 4. Özellik sütunlarını (X) ve hedef sütunu (y) ayır
# X = sepal length, sepal width, petal length, petal width
# y = target
X = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
Y = df["target"]

# 5. Veriyi eğitim ve test verisi olarak ayır (test oranı %20 olsun)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 6. Lojistik regresyon modeli oluştur ve eğitim verisi ile eğit
model = LogisticRegression()
model.fit(x_train, y_train)

# 7. Test verisi ile tahmin yap
y_pred = model.predict(x_test)

# 8. Modelin doğruluk oranını hesapla
acc = accuracy_score(y_test, y_pred)

# 9. Tahminleri gerçek test verileriyle karşılaştır, doğruluğu hesapla
# (Bunu accuracy_score ile zaten yaptık)

# 10. Model doğruluğunu ekrana yazdır 
print(f"Model Doğruluğu: %{acc * 100:.2f}")
"""

#UNSUPERVISED

# 1. Gerekli kütüphaneleri içe aktar
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 2. iris.csv dosyasını pandas ile oku
df = pd.read_csv("iris-model/iris.csv")
# 3. Sadece özellik sütunlarını seç (4 özellik: sepal/petal)
x = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]

# 4. Veriyi standartlaştır (StandardScaler)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# 5. K-Means modelini oluştur ve veriye uygula (3 küme)
kmeans = KMeans(n_clusters=3, random_state=42)

# 6. Küme sonuçlarını veri çerçevesine yeni sütun olarak ekle
df["cluster"] = kmeans.fit_predict(x_scaled)

# 7. Sonuçları yazdır (ilk 10 satır)
print(df.head(10))

# (İsteğe bağlı) 8. Küme dağılımını görselleştir (matplotlib ile)
# 8. Küme sonuçlarını 2 boyutlu grafikle çizmek için gerekli kütüphaneleri içe aktar

# 9. Grafik çiz (örnek olarak ilk iki sütunu al: sepal length & sepal width)
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=df["cluster"], cmap="viridis")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Küme Dağılımı (K-Means)")
plt.show()
