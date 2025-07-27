import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import sklearn.metrics as mt
from sklearn.model_selection import GridSearchCV
data = yf.download("THYAO.IS", start="2022-08-01", end="2022-09-01")
data = data.reset_index()
data["Day"] = data["Date"].astype(str).str.split("-").str[2]

print(data)
y = data["Close"]
X = data["Day"]

y = np.array(y).reshape(-1,1)
X = np.array(X).reshape(-1,1)

scy = StandardScaler()
scx = StandardScaler()

X = scx.fit_transform(X)
y = scy.fit_transform(y)

svrrbf = SVR(kernel="rbf")
svrrbf.fit(X,y)
tahminrbf = svrrbf.predict(X)

r2 = mt.r2_score(y,tahminrbf )
parametreler = {"C" : [1,10,100,1000,10000], "gamma" : [1,0.1,0.001], "kernel" : ["rbf","linear","poly"]}

tuning = GridSearchCV(estimator=SVR(),param_grid=parametreler,cv=10)
tuning.fit(X,y)
print(tuning.best_params_)

plt.scatter(X,y,color="red")
plt.plot(X,tahminrbf,color="green",label="RBF Model")
plt.legend()
plt.show()

