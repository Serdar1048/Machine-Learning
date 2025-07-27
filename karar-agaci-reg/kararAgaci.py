import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt
data = pd.read_csv("karar-agaci-reg/Position_Salaries.csv")

y = data["Salary"] 
X = data["Level"]

y = np.array(y).reshape(-1,1)
X = np.array(X).reshape(-1,1)

dtr = DecisionTreeRegressor(random_state=0,max_leaf_nodes=3)
dtr.fit(X,y)
tahmin = dtr.predict(X)

plt.figure(figsize=(20,10),dpi=100)
plot_tree(dtr,feature_names=["Level"],class_names=["Salary"],rounded=True,filled=True)
plt.show()



 
