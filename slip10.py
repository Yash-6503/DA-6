#slip10
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
#Createthedataset
Dataset={
    1:['eggs','milk','bread'],
    2:['eggs','apple'],
    3:['milk','bread'],
    4:['apple','milk'],
    5:['milk','apple','bread']
}
#Convertcategoricalvaluesintonumericformat
Te=TransactionEncoder()
Te_ary=Te.fit(Dataset.values()).transform(Dataset.values())
Df=pd.DataFrame(Te_ary,columns=Te.columns_)

#ApplyApriorialgorithmtogeneratefrequentitemsetsandassociationrulesMin_sup=0.4
Frequent_itemsets=apriori(Df,min_support=min_sup,use_colnames=True)
Association_rules=association_rules(Frequent_itemsets,metric='confidence',min_threshold=0.6)
#Printthefrequentitemsetsandassociationrules
print('FrequentItemsets:\n',Frequent_itemsets)
print('\nAssociationRules:\n',Association_rules)