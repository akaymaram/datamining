# dag.py 1.0.4

import numpy as np
import sys
import random


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




zero_to_n = [i for i in range(len(list_of_sentences))]
max_matrix = summary_matrix(zero_to_n)





### GA optimizer ###
 
def fitness_function(summary):
	total = 0
	matrix = summary_matrix(summary)
	matrix = matrix[1:]
	for row in matrix:
			total+=np.sum(row[1:])

	return total

 

 
def mutation(summary, r_mut, doc_length):
	num_sentences_to_change = max(int(r_mut*len(summary)),1)
	mutated_summary = []
	for x in summary:
		mutated_summary.append(x)

	sentenceID_sum_tuple = []

	matrix = summary_matrix(summary)
	matrix = matrix[1:]
	for row in matrix:
		row_sum =np.sum(row[1:])
		sentenceID_sum_tuple.append((row[0], row_sum))

	sentenceID_sum_tuple.sort(key = lambda x: x[1])

	sentenceID_sum_tuple = sentenceID_sum_tuple[0:num_sentences_to_change]

	for tup in sentenceID_sum_tuple:
		mutated_summary.remove(tup[0])


	while len(mutated_summary) < len(summary):
		sample_index = random.randint(0, doc_length-1)
		if sample_index not in mutated_summary:
			mutated_summary.append(sample_index) 


	return sorted(mutated_summary)



def genetic_algorithm(function, doc_length, summary_length, number_of_iterations, population_size, r_cross, mutation_coefficient, selection_rate):

	
	population = []

	for _ in range(population_size):
		summary = set()
		while len(summary) < summary_length:
			sample_index = random.randint(0, doc_length-1)
			if sample_index not in summary:
				summary.add(sample_index)


		population.append(sorted(summary))



	best_summary, best_score = 0, 0
	for gen in range(number_of_iterations):

		scores = [fitness_function(p) for p in population]
		for i in range(population_size):
			if scores[i] > best_score:
				best_summary, best_score = population[i], scores[i]
				print(">%d, new best f(%s) = %f" % (gen,  best_summary, best_score))


		selected = []
		while len(selected) < population_size*selection_rate:
			top_score = max(scores)
			index = scores.index(top_score)
			selected.append(population[index])
			del population[index]
			del scores[index]


		children = []
		for x in selected:
			children.append(x)
		for summary in selected:
			children.append(mutation(summary,mutation_coefficient,doc_length))
			

		
		
		population = children
	return [best_summary, best_score]


### GA optimizer ###
 


doc_length = len(max_matrix)-1

summary_length = 5

num_iterations = 100

population_size = 10

r_cross = 0.9

mutation_coefficient = .1

selection_rate = .5


best, score = genetic_algorithm(fitness_function, doc_length, summary_length, num_iterations, population_size, r_cross, mutation_coefficient, selection_rate)
print('Done!')
print('f(%s) = %f' % (best, score))


for index in best:
	print(index)
	print(list_of_sentences[index])


