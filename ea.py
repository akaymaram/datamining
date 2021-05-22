import numpy as np
import random

def fitness_function(summary, summary_matrix):
	total = 0
	matrix = summary_matrix(summary)
	matrix = matrix[1:]
	for row in matrix:
		total+=np.sum(row[1:])
	return total
 
def mutation(summary, r_mut, doc_length, summary_matrix):
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

def evolutionary_algorithm(summary_matrix, doc_length, summary_length, number_of_iterations, population_size, r_cross, mutation_coefficient, selection_rate,total_bests_scores = []):

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

		scores = [fitness_function(p, summary_matrix) for p in population]
		for i in range(population_size):
			if scores[i] > best_score:
				best_summary, best_score = population[i], scores[i]
				#print(">%d, new best f(%s) = %f" % (gen,  best_summary, best_score))


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
			children.append(mutation(summary,mutation_coefficient,doc_length, summary_matrix))
			

		
		
		population = children
	total_bests_scores.append((best_summary, best_score))
	return best_summary,best_score


