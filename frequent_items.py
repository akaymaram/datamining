import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

text = open("input3.txt","r")
dataset = []
for line in text:
  line = line.strip()
  line = line.lower()
  #split line to words
  words = line.split()
  #add words to dataset array
  dataset.append(words)

# convert dataset array to Pandas dataframe
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns = te.columns_)
#apriori algo
from mlxtend.frequent_patterns import apriori
freqItemSets =apriori(df, min_support = 0.65, use_colnames=True)
print(freqItemSets.head(20))
#can filter by length of itemsets as well

