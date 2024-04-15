#slip6
'''Create the following dataset in python & Convert the categorical values into numeric format.Apply
the apriori algorithm on the above dataset to generate the frequent itemsets and association rules. Repeat
the process with different min_sup values. '''

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
#createthedataset
TID={1:['bread','milk'],2:['bread','diaper','beer','eggs'],3:['milk','diaper','beer','coke'],4:['bread','milk','diaper','beer'],5:['bread','milk','diaper','coke']}
Transactions=[]
for key,value in TID.items():
    Transactions.append(value)
#convertthecategoricalvaluesintonumericformat
Te=TransactionEncoder()
Te_ary=Te.fit_transform(Transactions)
Df=pd.DataFrame(Te_ary,columns=Te.columns_)
#applytheapriorialgorithmwithdifferentmin_supvalues
Min_sup_values=[0.2,0.4,0.6]
for min_sup in Min_sup_values:
    Frequent_itemsets=apriori(Df,min_support=min_sup,use_colnames=True)
    Rules=association_rules(Frequent_itemsets, metric = 'confidence', min_threshold=0.7)
    print('Min_sup:',min_sup)
    print('FrequentItemsets:')
    print(Frequent_itemsets)
    print('AssociationRules:')
    print(Rules)