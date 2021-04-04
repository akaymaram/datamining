import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = open("input3.txt", "r")
#empty dictionary 
d = {}

for line in text:
  line = line.strip()
  line = line.lower()
  #splitting line into words
  words = line.split(" ")
  for word in words:
    if word in d:
      d[word] = d[word] + 1
    else:
      d[word] = 1
      
for key in list(d.keys()):
  print(key, ":", d[key])
      
