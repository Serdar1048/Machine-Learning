import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("knn/kanser.csv")

M = data[data["diagnosis"]=="M"]
B = data[data["diagnosis"]=="B"]

plt.scatter(M.radius_mean, M.texture_mean,color="red",label="Kötü huylu")
plt.scatter(B.radius_mean, B.texture_mean,color="green",label="İyi huylu")
plt.legend()
plt.show()