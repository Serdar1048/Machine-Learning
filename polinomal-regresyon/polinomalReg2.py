import pandas as pd 
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as mt

data = pd.read_csv("polinomal-regresyon/data.csv")
data.drop(columns=["No", "X1 transaction date", "X5 latitude", "X6 longitude"],axis=1, inplace=True)
data= data.rename(columns={"X2 house age":"Ev Yaşı", "X3 distance to the nearest MRT station":"Metroya Uzaklık", "X4 number of convenience stores":"Market Sayısı", "Y house price of unit area":"Ev Fiyatı"})

y= data["Ev Fiyatı"]
X = data.drop(columns="Ev Fiyatı", axis=1)

pol = PolynomialFeatures(degree=2)
X_pol = pol.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pol,y,test_size=0.2,random_state=42)
pol_reg = LinearRegression()
pol_reg.fit(X_train,y_train)
tahmin = pol_reg.predict(X_test)

r2 = mt.r2_score(y_test, tahmin)
mse = mt.mean_squared_error(y_test, tahmin)


print("R2: {} MSE: {}".format(r2,mse))          