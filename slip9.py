'''Create your own transactions dataset and apply the above process on your dataset.'''

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transactions dataset
transactions = [
    ["bread", "milk", "coffee"],
    ["bread", "butter", "jam"],
    ["milk", "tea", "sugar"],
    ["bread", "milk", "tea", "butter"],
    ["coffee", "sugar", "jam"]
]

# Convert transactions into DataFrame
df = pd.DataFrame(transactions)

# Display dataset information
print("Dataset Information:")
print(df.info())

# Preprocess the data
df.dropna(inplace=True)  # Drop null values
print("\nAfter dropping null values:")
print(df.info())

# Convert categorical values into numeric format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display frequent itemsets and association rules
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(association_rules_df)
