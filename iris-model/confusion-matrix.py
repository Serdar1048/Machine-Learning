# 1. Gerekli kütüphaneleri içe aktar (pandas, sklearn, matplotlib)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 2. iris.csv dosyasını oku
df = pd.read_csv("iris-model/iris.csv")

# 3. Özellik sütunlarını (X) ve hedef sütunu (y) ayır

X = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
Y = df["target"]
# 4. Veriyi eğitim ve test olarak ayır
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
# 5. Model oluştur (Logistic Regression kullanabilirsin)
model = LogisticRegression()
# 6. Modeli eğit
model.fit(x_train,y_train)
# 7. Test verisi ile tahmin yap
y_pred = model.predict(x_test)
# 8. Confusion matrix hesapla ve yazdır (sklearn.metrics.confusion_matrix kullan)
cm = confusion_matrix(y_test,y_pred)
# 9. Confusion matrix’i görselleştir (isteğe bağlı matplotlib ile heatmap)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot= True, cmap="Blues", fmt="d", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Tahmin edilen")
plt.ylabel("Gerçek Değer")
plt.title("COnfusion Matrix")
plt.show()

# 10. Accuracy, precision, recall ve f1-score metriklerini yazdır (sklearn.metrics ile)
print(classification_report(y_test,y_pred))