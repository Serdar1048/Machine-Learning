import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = pd.read_csv("PCA/winequality-red.csv")

y = data["quality"]
X = data.drop(columns="quality",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# n <= min(değşiken sayısı=11, sınıf sayısı=6 -1 )
lda = LinearDiscriminantAnalysis(n_components=5)
X_train2 = lda.fit_transform(X_train,y_train)
X_test2 = lda.transform(X_test)

print(np.cumsum(lda.explained_variance_ratio_) * 100)
print(len(np.unique(y_train))) # -> sınıf sayısı = 6