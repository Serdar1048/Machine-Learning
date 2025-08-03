#SUPERVISED 

"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. veriyi oku
df = pd.read_csv("titanic-model/titanic.csv")

# 2. gerekli sütunları seç
df = df[["Survived", "Pclass", "Sex", "Age"]]
df = df.dropna() #eski verileri sil

# 3.kategorik veriyi sayıya çevir (sex sütunu)
df["Sex"] = df["Sex"].map({"male": 0, "female":1})

# 4. x ve y'yi ayır
x = df[["Pclass", "Sex", "Age"]] #girdi özellikleri
y = df["Survived"] #etiket (çıktı)

# 5. veriyi eğitim ve test olarak böl
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=42)

# 6. modeli kur ve eğit
model = LogisticRegression()
model.fit(x_train, y_train)

# 7. test verisiyle tahmin yap
y_pred = model.predict(x_test)

# 8. başarıyı ölç
acc = accuracy_score(y_test, y_pred)

print(f"Model Doğruluğu: %{acc * 100:2f}")"""

#UNSUPERVISED
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. CSV dosyasını oku
df = pd.read_csv("titanic-model/titanic.csv")

# 2. Gerekli sütunları al ve eksik verileri sil
df = df[["Pclass", "Sex", "Age", "Fare"]]
df = df.dropna()

# 4. Cinsiyeti sayıya çevir (male: 0, female: 1)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# 5. girdi verilerini seç
x = df[["Pclass", "Sex" , "Age" ,"Fare"]]

# 6. veriyi standartlaştır (K-Means için çok önemli)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 7. K-Means modelini oluştur (2 küme)

kmeans = KMeans(n_clusters=2,random_state=0)
df["Cluster"] = kmeans.fit_predict(x_scaled)

print(df)
