import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
import sklearn.metrics as mt

data = pd.read_csv("toplu-model-ornegi/reklam.csv")

y = data["Sales"]
X = data.drop(columns="Sales",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)

def modeltahmin(model):
    model.fit(X_train, y_train)
    tahmin = model.predict(X_test)
    r2 = mt.r2_score(y_test,tahmin)
    rmse = mt.mean_squared_error(y_test, tahmin)
    return [r2,rmse]

modeller = [LinearRegression(),Ridge(),Lasso(),ElasticNet(),SVR(),DecisionTreeRegressor(random_state=0),BaggingRegressor(random_state=0) ,RandomForestRegressor(random_state=0)]

ad = ["Linear Model", "Ridege Model" , "Lasso Model" ,"ElasticNet Model", "SVR Model", "Karar Ağacı Modeli","Bag Model", "Random Forest Model"]

sonuc = []

for i in modeller:
    sonuc.append(modeltahmin(i))
 
df = pd.DataFrame(ad, columns= ["Model Adı"])
df2 = pd.DataFrame(sonuc,columns=["R2", "RMSE"])
df = df.join(df2)                      

print(df)