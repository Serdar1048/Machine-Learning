import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
data = pd. read_csv("K-means/musteri.csv")

data = data.drop(columns="CustomerID",axis=1)

X = data.iloc[:,1:3]

#wcss = []
#
#for k in range(1,20):
#    kmodel = KMeans(n_clusters=k,random_state=0)
#    kmodel.fit(X)
#    wcss.append(kmodel.inertia_)
#
#plt.plot(range(1,20),wcss)
#plt.xlabel("Küme Sayısı")
#plt.ylabel("WCSS")
#plt.show()

kmodel = KMeans(random_state=0)
grafik = KElbowVisualizer(kmodel,k=(1,20))                    
grafik.fit(X)
grafik.poof()
