import pandas as pd 

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as mt

data = pd.read_csv("https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv")

y = data["Revenue"]
X = data["Temperature"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train.values.reshape(-1,1),y_train.values.reshape(-1,1))
tahmin = model.predict(X_test.values.reshape(-1,1))

r2 = mt.r2_score(y_test,tahmin)
mse  =mt.mean_squared_error(y_test,tahmin)

print("R2: {}  MSE: {}".format(r2,mse))

parametreler = {"min_samples_split" : range(2,50), "max_leaf_nodes" : range(2,50)}

grid = GridSearchCV(estimator=model,param_grid=parametreler,cv=10)
grid.fit(X_train.values.reshape(-1,1),y_train.values.reshape(-1,1))

print(grid.best_params_)

