import operator
text = open("input3.txt", "r")
#empty dictionary 
d = {}

for line in text:
  line = line.strip()
  line = line.lower()
  #splitting line into words
  words = line.split(" ")
  for word in words: #preprocessing to check for non-letters and remove them
    iLast = len(word) - 1
    if len(word) == 0: #There can be weird spacing when processing a text
      continue
    #make sure to use if below to avoid removing dates from data
    if(ord(word[0]) > 65 and ord(word[0]) < 122): #starts with a letter
      while (ord(word[iLast]) < 65 or ord(word[iLast]) > 122): #remove footnotes, quotes, periods, etc.
        word = word[:-1] #remove last char
        iLast = len(word) - 1 #update last index
        print(word, iLast)
    # leave word alone if doesn't start with a letter since it may be number
    if len(word) <= 4: #filter out stopwords which are usually short
      continue
    if word in d:
      d[word] = d[word] + 1
    else:
      d[word] = 1
      
#for key in sorted(d.keys()):
# print(key, ":", d[key])
sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))

for item in sorted_d.items():
  print(item)
