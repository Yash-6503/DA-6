#slip14
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd
#Createthedataset
TID={1:['apple','mango','banana'],
2:['mango','banana','cabbage','carrots'],
3:['mango','banana','carrots'],
4:['mango','carrots']}
#Convertthecategoricalvaluesintonumericformat
Te=TransactionEncoder()
Te_ary=Te.fit([TID[i]for i in TID]).transform([TID[i]for i in TID])
Df=pd.DataFrame(Te_ary,columns=Te.columns_)
#Applytheapriorialgorithmwithdifferentmin_supvalues
Min_sup_values=[0.25,0.5,0.75]
for min_sup in Min_sup_values:
    Frequent_itemsets=apriori(Df, min_support = min_sup, use_colnames=True)
    print('Frequent_itemsets with min_sup=',min_sup)
    print(Frequent_itemsets)
    print('\n')