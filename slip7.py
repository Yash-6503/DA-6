'''Download the Market basket dataset. Write a python program to read the dataset and display its
information. Preprocess the data (drop null values etc.) Convert the categorical values into numeric
format. Apply the apriori algorithm on the above dataset to generate the frequent itemsets and association
rules. '''

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
df = pd.read_csv("./Datasets/Market_Basket.csv")

# Display dataset information
print("Dataset Information:")
print(df.info())

# Preprocess the data
df.dropna(inplace=True)  # Drop null values
print("\nAfter dropping null values:")
print(df.info())

# Convert categorical values into numeric format
te = TransactionEncoder()
encoded_data = te.fit(df.values).transform(df.values)
df_encoded = pd.DataFrame(encoded_data, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display frequent itemsets and association rules
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(association_rules_df)
