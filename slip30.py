'''Create the dataset . transactions = [['eggs', 'milk','bread'], ['eggs', 'apple'], ['milk', 'bread'], ['apple',
'milk'], ['milk', 'apple', 'bread']] .
Convert the categorical values into numeric format.Apply the apriori algorithm on the above dataset to
generate the frequent itemsets and association rules.'''

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Create the dataset
transactions = [['eggs', 'milk', 'bread'],
                ['eggs', 'apple'],
                ['milk', 'bread'],
                ['apple', 'milk'],
                ['milk', 'apple', 'bread']]

# Step 2: Convert categorical values into numeric format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 3: Apply Apriori algorithm to generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Step 4: Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Step 5: Display the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)
