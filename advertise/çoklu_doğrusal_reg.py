import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
data = sns.load_dataset("tips")

kategori = []
kategorik = data.select_dtypes(include=["category"])

for i in kategorik.columns:
    kategori.append(i)

#kukla değişken tuzağı
data = pd.get_dummies(data,columns=kategori,drop_first=True)

y = data["tip"]
X = data.drop(columns="tip",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lr = LinearRegression(fit_intercept=True,positive=False,tol=0.001)
lr.fit(X_train, y_train)
tahmin = lr.predict(X_test)

params = { "fit_intercept": [True, False],
    "positive": [True, False],
    "tol": [1e-3, 1e-4, 1e-5]}

grid = GridSearchCV(LinearRegression(),param_grid=params,cv = 10,scoring="neg_mean_squared_error")
grid.fit(X_train,y_train)
print(grid.best_params_)
y_test = y_test.sort_index()
r2 = mt.r2_score(y_test,tahmin)
print(r2)
df = pd.DataFrame({"Gerçek": y_test,"Tahmin": tahmin})
df.plot(kind="line")
plt.show()