# Gerekli kütüphaneleri içe aktar (pandas, sklearn)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # Gerekli metrik fonksiyonunu import ettik
from sklearn.metrics import confusion_matrix  # Confusion matrix fonksiyonunu import ettik
import seaborn as sns  # Görselleştirme için seaborn'u import ettik
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib'i import ettik

# iris.csv dosyasını oku
df = pd.read_csv("iris-model/iris.csv")

# 1. Iris veri setinden sadece 0 ve 1 sınıfını filtrele
df = df[df["target"].isin([0, 1])]

# 2. Özellik sütunlarını (X) ve hedef sütunu (y) ayır
X = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
Y = df["target"]

# 3. Veriyi eğitim ve test olarak ayır (test oranı %20, rastgelelik için random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Lojistik regresyon modelini oluştur ve eğitim verisi ile eğit
model = LogisticRegression()
model.fit(x_train, y_train)

# 5. Test verisi ile tahmin yap
y_pred = model.predict(x_test)

# 6. Precision ve Recall değerlerini hesapla ve yazdır
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix'i hesapla ve görselleştir
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.title("Confusion Matrix")
plt.show()
