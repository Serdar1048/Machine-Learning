import pandas as pd
import numpy as np #numpy ne

from sklearn.impute import SimpleImputer #imputer ne
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
data = pd.read_excel("advertise/AdvertiseData2.xlsx")


imputer = SimpleImputer(missing_values=np.nan,strategy="mean") #bu satır ne işe yarıyor
imputer = imputer.fit(data)
data.iloc[:,:] = imputer.transform(data) #iloc nedir ne işimize yarıyor

y = data["Sales"]
X = data[["TV", "Radio"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression(fit_intercept=True,positive=False,tol=0.001)
lr.fit(X_train,y_train)
tahmin = lr.predict(X_test)

r2 = mt.r2_score(y_test,tahmin)
mse = mt.mean_squared_error(y_test,tahmin)
#rmse = mt.mean_squared_error(y_test,tahmin,squared=False)
rmse = np.sqrt(mt.mean_squared_error(y_test, tahmin))
mae = mt.mean_absolute_error(y_test,tahmin)


#Model Tuning
params = { "fit_intercept": [True, False],
    "positive": [True, False],
    "tol": [1e-3, 1e-4, 1e-5]}

grid = GridSearchCV(LinearRegression(),param_grid=params,cv=10)
grid.fit(X_train, y_train)

print(grid.best_params_) #{'fit_intercept': True, 'positive': False, 'tol': 0.001}

print("R2: {}  MSE:{}  RMSE: {}  MAE: {}".format(r2,mse,rmse,mae))