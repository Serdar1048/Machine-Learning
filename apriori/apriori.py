import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules
data = pd.read_csv("apriori/bakkal.csv",header=None)
veri = data.copy()
veri.columns=["Ürün"]

veri = list(veri["Ürün"].apply(lambda x:x.split(",")))
te = TransactionEncoder()
teveri = te.fit_transform(veri)
veri = pd.DataFrame(teveri,columns=te.columns_)

df1 = apriori(veri,min_support=0.05,use_colnames=True)

df2 = association_rules(df1,metric="confidence",min_threshold=0.5)
print(df1)
print(df2)




