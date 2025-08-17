import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd. read_csv("K-means/musteri.csv")

data = data.drop(columns="CustomerID",axis=1)

X = data.iloc[:,1:3]
#plt.scatter(X.iloc[:,0],X.iloc[:,1],color="black")
#plt.show()

kmodel = KMeans(n_clusters=2,random_state=0)
kfit = kmodel.fit(X)
kumeler = kfit.labels_
merkezler = kfit.cluster_centers_

figure, axis = plt.subplots(1,2)
axis[0].scatter(X.iloc[:,0],X.iloc[:,1],color="black")
axis[1].scatter(X.iloc[:,0],X.iloc[:,1],c = kumeler,cmap="winter")
axis[1].scatter(merkezler[:,0],merkezler[:,1],c = "red",s=200)

plt.show()
#print(kmodel.cluster_centers_)

