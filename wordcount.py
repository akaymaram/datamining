import operator
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
      
#for key in sorted(d.keys()):
# print(key, ":", d[key])
sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))

for item in sorted_d.items():
  print(item)
