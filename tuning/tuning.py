import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt

data = pd.read_csv("tuning/AdvertiseData.csv")
data.drop(columns=["Index"],axis=1,inplace=True)
y  = data["Sales"]
X = data.drop(columns="Sales", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression()
model = lr.fit(X_train,y_train)


def skor(model, x_train, x_test, y_train, y_test):
    egitimtahmin = model.predict(x_train)
    testtahmin = model.predict(x_test)

    r2_eğitim = mt.r2_score(y_train, egitimtahmin)
    r2_test = mt.r2_score(y_test, testtahmin)

    mse_eğitim = mt.mean_squared_error(y_train, egitimtahmin) 
    mse_test = mt.mean_squared_error(y_test, testtahmin) 

    return[r2_eğitim, r2_test, mse_eğitim, mse_test]

sonuc1 = skor(model = lr, x_train = X_train, x_test = X_test, y_train = y_train, y_test = y_test)

print("Eğitim R2= {}  Eğitim MSE = {}".format(sonuc1[0], sonuc1[2]))
print("Test R2= {}  Test MSE = {}".format(sonuc1[1], sonuc1[3]))

lr_cv = LinearRegression()
k = 5
iterasyon = 1 
cv = KFold(n_splits = k)

for egitimindex, testindex in cv.split(X):
    X_train, X_test = X.loc[egitimindex], X.loc[testindex]
    y_train, y_test = y.loc[egitimindex], y.loc[testindex]
    lr_cv.fit(X_train, y_train)

    sonuc2 = skor(model = lr_cv, x_train = X_train, x_test = X_test, y_train = y_train, y_test = y_test)

    print("İterasyon: {}".format(iterasyon))
    print("Eğitim R2= {}  Eğitim MSE = {}".format(sonuc2[0], sonuc2[2]))
    print("Test R2= {}  Test MSE = {}".format(sonuc2[1], sonuc2[3]))
    iterasyon += 1
