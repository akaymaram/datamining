# main.py 1.0.6

import numpy as np
import sys
import random
import ea
import fitz
import nltk
from termcolor import colored
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

##list of logical words
# comparison = {'similarily', 'likewise', 'also', 'comparison'}
# reason = {'cause', 'reason'}
# result = {'result', 'consequence', 'therefore', 'thus','consequently', 'hence'}
# contrast = {'however', 'on the other hand', 'on the contrary', 'contrast'}
# sequential = {'Firstly', 'secondly', 'thirdly', 'next','last', 'finally', 'in addition', 'furthermore', 'also','before', }
# order_of_importance = {'most', 'more', 'importantly', 'significantly','above', 'all', 'primarily', 'essential'}




doc = fitz.open("first_two_pages.pdf")
content = ""
for page in doc:
	text = page.get_text('text')
	content += text

print(colored('hello', 'red'), colored('world', 'green'))
print(content)
print(colored('hello', 'red'), colored('world', 'green'))
# sys.exit()
## removing stop words and Lemmatization
stop_words=set(['a', 'about', 'above', 'after', 'again', 'against',
	'all', 'am', 'an', 'and', 'any', 'are',
	'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
	'could', 'did','do', 'does','doing','down', 'during', 'each', 'few', 'for', 'from', 'further', 'had','has','have','having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is',
	 'it', "it's", 'its', 'itself', "let's",'me','more', 'most','my', 'myself','nor','of','on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own','same','she', "she'd", "she'll", "she's", 'should',                                      'so', 'some', 'such',      'than', 'that',            "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',       'very', 'was',                   'we', "we'd", "we'll", "we're", "we've", 'were',                     'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why',         "why's", 'with',                 'would',                            'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])

## still using word_tokens, due to differences in treatment of comma
word_tokens = word_tokenize(content)

# lemmatizer so far only trims down "ing", "s", etc
lemmatizer = WordNetLemmatizer()
filtered_sentence = []
for w in word_tokens:
	if w.lower() not in stop_words:
		filtered_sentence.append(lemmatizer.lemmatize(w))
final_sentence= ' '.join(filtered_sentence)
initial_list_of_sentences = [sentence for sentence in final_sentence.split(".") if len(sentence) > 0]



extra_information= {"content downloaded", "http", "/", "Vol", "©"}


list_of_sentences = []

for sentence in range(len(initial_list_of_sentences)):
	flag = True;
	word_list = initial_list_of_sentences[sentence].split()
	number_of_words = len(word_list)
	for word in word_list:
		if word in extra_information:
			flag = False
			continue
	print(flag)
	if number_of_words > 5 and flag == True:
		list_of_sentences.append(initial_list_of_sentences[sentence])



print(list_of_sentences)




def retrieved_matrix(summary):

	row_numbers = [-1]
	for x in summary:
		row_numbers.append(x)
	matrix = np.array(row_numbers)

	## Use summary length instead to make retrieving easier
	for i in range(len(summary)):
		row = [summary[i]]
		sentence = list_of_sentences[summary[i]]
		set_of_words = {word for word in sentence.split()}
		sentence_size = len(set_of_words)
		for j in range(len(summary)):
			if j == i:
				row.append(1)
				continue

			## if i>j ( repeated) just grab it from previous
			if j < i:
				retrieved_value = matrix.item(j + 1, i + 1)
				row.append(retrieved_value)
				continue

			second_sentence = list_of_sentences[summary[j]]
			second_set_of_words = {word for word in second_sentence.split()}
			second_sentence_size = len(second_set_of_words)

			##not count stopwords
			count = len(set_of_words.intersection(second_set_of_words))

			edge_weight = round(count / (sentence_size + second_sentence_size), 3)

			row.append(edge_weight)
		## adds a row to matrix (row by row)
		matrix = np.vstack([matrix, row])

	return matrix





## calculates table for whole document
zero_to_n = list(range(len(list_of_sentences)))
## save table for the first time
max_matrix = retrieved_matrix(zero_to_n)
## function to grab data form table
def summary_matrix(summary):
	row_numbers = [-1]
	for x in summary:
		row_numbers.append(x)
	matrix = np.array(row_numbers)

	for i in summary:
		row = [i]
		for j in summary:
			value = max_matrix.item(i+1,j+1)
			row.append(value)

		matrix = np.vstack([matrix, row])
	return matrix


doc_length = len(max_matrix)-1

summary_length = 5

num_iterations = 100

population_size = 10

r_cross = 0.9

mutation_coefficient = .1

selection_rate = .5


best, score = ea.evolutionary_algorithm(summary_matrix, doc_length, summary_length, num_iterations, population_size, r_cross, mutation_coefficient, selection_rate)
#print('Done!')
#print('best summary: %s \ncohesion score: %f' % (best, score))


## want stopwords here
list_of_sentences_with_stopwords = [sentence for sentence in content.split(".") if len(sentence) > 0]

my_terms=[]

for index in best:
	print(' ['+str(index)+'] ', end='')
	print(list_of_sentences_with_stopwords[index], end='.')
	my_terms.append(list_of_sentences_with_stopwords[index])



Adversative = ["however", "nevertheless", "in fact","actually", "instead", "contrary"]
Sequential = ["then", "next", "last", "finally", "up to now", "to sum up"]
Causal = ["therefore", "consequently", "then", "otherwise"]
Additive = ["in addition","moreover", "that is", "for instance"
"likewise","similarly"]


for x in my_terms:
	text = x
	for page in doc:
		text_instances = page.searchFor(text)
		for inst in text_instances:
			highlight = page.addUnderlineAnnot(inst)




doc.save("output.pdf", garbage=4, deflate=True, clean=True)