import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
data = pd.read_csv("hiyerarşik/musteri.csv")

X = data.iloc[0:20,2:4]

model = AgglomerativeClustering()
tahmin = model.fit_predict(X)

X["Label"] = tahmin

print(X["Age"][X["Label"]==0])

#plt.scatter(X["Age"][X["Label"]==0],X["Annual Income (k$)"][X["Label"]==0],color="red")
#plt.scatter(X["Age"][X["Label"]==1],X["Annual Income (k$)"][X["Label"]==1],color="black")
#plt.show()

link = linkage(X)
dendrogram(link)
plt.xlabel("Veri Notaları")
plt.ylabel("Mesafe")
plt.show()



