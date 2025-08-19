import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans,AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage,dendrogram

sonuc = pd.read_csv("bist30/veri.csv")

ms = MinMaxScaler()
X = ms.fit_transform(sonuc.iloc[:,[2,3]])
X = pd.DataFrame(X,columns=["Gelir","Oynaklık"])

model = AgglomerativeClustering(n_clusters=4,linkage="single")
tahmin = model.fit_predict(X)
labels = model.labels_

sonuc["Labels"] = labels
sns.scatterplot(x="Labels", y="Hisse", data=sonuc, hue="Labels", palette="deep")

plt.show()

#hc = linkage(X,method="single")
#dendrogram(hc)
#plt.show()

#kmodel = KMeans(n_clusters=6, random_state=0)
#kfit = kmodel.fit(X)
#labels = kfit.labels_

#sonuc["Labels"] = labels

#sns.scatterplot(x="Labels", y="Hisse", data=sonuc, hue=labels, palette="deep")
#plt.show()

