# main.py 1.0.5

import numpy as np
import sys
import random
import ea


import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

comparison = {'similarily', 'likewise', 'also', 'comparison'}
reason = {'cause', 'reason'}
result = {'result', 'consequence', 'therefore', 'thus','consequently', 'hence'}
contrast = {'however', 'on the other hand', 'on the contrary', 'contrast'}
sequential = {'Firstly', 'secondly', 'thirdly', 'next','last', 'finally', 'in addition', 'furthermore', 'also'}
order_of_importance = {'most', 'more', 'importantly', 'significantly','above', 'all', 'primarily', 'essential'}



myfile = open("input3.txt", "r")
content = myfile.read()
myfile.close()


## removing stop words and Lemmatization
stop_words = set(stopwords.words('english'));
word_tokens = word_tokenize(content)
lemmatizer = WordNetLemmatizer()
filtered_sentence = []
for w in word_tokens:
	if w.lower() not in stop_words:
		filtered_sentence.append(lemmatizer.lemmatize(w))
final_sentence= ' '.join(filtered_sentence)



list_of_sentences = [sentence for sentence in final_sentence.split(".") if len(sentence) > 0]



def summary_matrix(summary):
	row_numbers = [-1]
	for x in summary:
		row_numbers.append(x)
	matrix = np.array(row_numbers)

	for i in summary:
		row = [i]
		sentence = list_of_sentences[i]
		set_of_words = {word for word in sentence.split()}
		sentence_size = len(set_of_words)
		for j in summary:
			if j == i:
				row.append(1)
				continue

			second_sentence = list_of_sentences[j]
			second_set_of_words = {word for word in second_sentence.split()}
			second_sentence_size = len(second_set_of_words)

			count = len(set_of_words.intersection(second_set_of_words))

			edge_weight = round(count/(sentence_size+second_sentence_size),3)


			row.append(edge_weight)

		matrix = np.vstack([matrix,row])
	return matrix




zero_to_n = list(range(len(list_of_sentences)))
max_matrix = summary_matrix(zero_to_n)

 


doc_length = len(max_matrix)-1

summary_length = 5

num_iterations = 100

population_size = 10

r_cross = 0.9

mutation_coefficient = .1

selection_rate = .5


best, score = ea.evolutionary_algorithm(summary_matrix, doc_length, summary_length, num_iterations, population_size, r_cross, mutation_coefficient, selection_rate)
print('Done!')
print('best summary: %s \ncohesion score: %f' % (best, score))



for index in best:
	print(index)
	print(list_of_sentences[index])


