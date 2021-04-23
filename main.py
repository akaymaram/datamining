#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install frontend')
get_ipython().run_line_magic('pip', 'install PyMuPDF')
get_ipython().run_line_magic('pip', 'install mlxtend')
import fitz
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import sys
import random
import ea
#import fitz
get_ipython().run_line_magic('pip', 'install nltk')
import nltk
nltk.download('punkt')
nltk.download('wordnet')
get_ipython().run_line_magic('pip', 'install termcolor')
from termcolor import colored
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from operator import itemgetter
import json


# In[2]:


def fonts(doc, granularity=False):

    """Extracts fonts and their usage in PDF documents.



    :param doc: PDF document to iterate through

    :type doc: <class 'fitz.fitz.Document'>

    :param granularity: also use 'font', 'flags' and 'color' to discriminate text

    :type granularity: bool



    :rtype: [(font_size, count), (font_size, count}], dict

    :return: most used fonts sorted by count, font style information

    """

    styles = {}

    font_counts = {}



    for page in doc:

        blocks = page.getText("dict")["blocks"]

        for b in blocks:  # iterate through the text blocks

            if b['type'] == 0:  # block contains text

                for l in b["lines"]:  # iterate through the text lines

                    for s in l["spans"]:  # iterate through the text spans

                        if granularity:

                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])

                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],

                                                  'color': s['color']}

                        else:

                            identifier = "{0}".format(s['size'])

                            styles[identifier] = {'size': s['size'], 'font': s['font']}



                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage



    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)



    if len(font_counts) < 1:

        raise ValueError("Zero discriminating fonts found!")



    return font_counts, styles


# In[3]:


def font_tags(font_counts, styles):

    """Returns dictionary with font sizes as keys and tags as value.



    :param font_counts: (font_size, count) for all fonts occuring in document

    :type font_counts: list

    :param styles: all styles found in the document

    :type styles: dict



    :rtype: dict

    :return: all element tags based on font-sizes

    """

    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)

    p_size = p_style['size']  # get the paragraph's size



    # sorting the font sizes high to low, so that we can append the right integer to each tag

    font_sizes = []

    for (font_size, count) in font_counts:

        font_sizes.append(float(font_size))

    font_sizes.sort(reverse=True)



    # aggregating the tags for each font size

    idx = 0

    size_tag = {}

    for size in font_sizes:

        idx += 1

        if size == p_size:

            idx = 0

            size_tag[size] = '<p> '

        if size > p_size:

            size_tag[size] = '<h{0}>'.format(idx)

        elif size < p_size:

            size_tag[size] = '<s{0}>'.format(idx)



    return size_tag


# In[4]:


def headers_para(doc, size_tag):

    """Scrapes headers & paragraphs from PDF and return texts with element tags.



    :param doc: PDF document to iterate through

    :type doc: <class 'fitz.fitz.Document'>

    :param size_tag: textual element tags for each size

    :type size_tag: dict



    :rtype: list

    :return: texts with pre-prended element tags

    """

    header_para = []  # list with headers and paragraphs

    first = True  # boolean operator for first header

    previous_s = {}  # previous span



    for page in doc:

        blocks = page.getText("dict")["blocks"]

        for b in blocks:  # iterate through the text blocks

            if b['type'] == 0:  # this block contains text



                # REMEMBER: multiple fonts and sizes are possible IN one block



                block_string = ""  # text found in block

                for l in b["lines"]:  # iterate through the text lines

                    for s in l["spans"]:  # iterate through the text spans

                        if s['text'].strip():  # removing whitespaces:

                            if first:

                                previous_s = s

                                first = False

                                block_string = size_tag[s['size']] + s['text']

                            else:

                                if s['size'] == previous_s['size']:



                                    if block_string and all((c == "|") for c in block_string):

                                        # block_string only contains pipes

                                        block_string = size_tag[s['size']] + s['text']

                                    if block_string == "":

                                        # new block has started, so append size tag

                                        block_string = size_tag[s['size']] + s['text']

                                    else:  # in the same block, so concatenate strings

                                        block_string += " " + s['text']



                                else:

                                    header_para.append(block_string)

                                    block_string = size_tag[s['size']] + s['text']



                                previous_s = s



                    # new block started, indicating with a pipe

                    block_string += "|"



                header_para.append(block_string)



    return header_para


# In[20]:


# main.py 1.0.6

##list of logical words
# comparison = {'similarily', 'likewise', 'also', 'comparison'}
# reason = {'cause', 'reason'}
# result = {'result', 'consequence', 'therefore', 'thus','consequently', 'hence'}
# contrast = {'however', 'on the other hand', 'on the contrary', 'contrast'}
# sequential = {'Firstly', 'secondly', 'thirdly', 'next','last', 'finally', 'in addition', 'furthermore', 'also','before', }
# order_of_importance = {'most', 'more', 'importantly', 'significantly','above', 'all', 'primarily', 'essential'}


doc = fitz.open("pdf_full.pdf")
content = ""
for page in doc:
	text = page.get_text('text')
	content += text

#print(colored('hello', 'red'), colored('world', 'green'))
#print(content)
#print(colored('hello', 'red'), colored('world', 'green'))
# sys.exit()


font_counts, styles = fonts(doc, granularity=False)
size_tag = font_tags(font_counts, styles)
#print(size_tag)
elements = headers_para(doc, size_tag)

paragraphs=""

for elem in elements:
    if(len(elem)>=4 and elem[1]=='p'):
        paragraphs+=elem


#print(colored('paragraphs extracted from the pdf', 'red'))
#print(paragraphs)
#print(colored("paragraphs extracted from the pdf", 'red'))


## removing stop words and Lemmatization
stop_words=set(['<','p', '>', '|', 'a', 'about', 'above', 'after', 'again', 'against',
	'all', 'am', 'an', 'and', 'any', 'are',
	'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
	'could', 'did','do', 'does','doing','down', 'during', 'each', 'few', 'for', 'from', 'further', 'had','has','have','having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is',
	 'it', "it's", 'its', 'itself', "let's",'me','more', 'most','my', 'myself','nor','of','on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own','same','she', "she'd", "she'll", "she's", 'should', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',       'very', 'was',                   'we', "we'd", "we'll", "we're", "we've", 'were',                     'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why',         "why's", 'with',                 'would',                            'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])
''''
## still using word_tokens, due to differences in treatment of comma
word_tokens = word_tokenize(paragraphs)

# lemmatizer so far only trims down "ing", "s", etc
lemmatizer = WordNetLemmatizer()
filtered_sentence = []
for w in word_tokens:
	#if w.lower() not in stop_words:
    filtered_sentence.append(lemmatizer.lemmatize(w))

print(colored('filtered sentence', 'red'))
print(filtered_sentence)
print(colored('filtered sentence', 'red'))
final_sentence= ' '.join(filtered_sentence)
list_of_sentences = [sentence for sentence in final_sentence.split(".") if len(sentence) > 0]
'''
#stop_words=set(['<p>', '|'])
list_of_sentences=[]

for sent in paragraphs.split('.'):
    if(len(sent)>4):
        x=sent.replace('<p>', "")
        x=x.replace('|', "")
        list_of_sentences.append(x)

list_of_sentences_with_stopwords=[] 
for x in list_of_sentences:
    new_sent = ""
    for y in x.split():
        if(y in stop_words):
            continue 
        new_sent += y
        new_sent += " "
    list_of_sentences_with_stopwords.append(new_sent)

print(colored('list of sent with stopwords removed', 'red'))
print(list_of_sentences_with_stopwords)
print(colored('list of sent with stopwords removed', 'red'))
#print(colored('the list after stopwords', 'red'))
#print(list_of_sentences)
#print(colored("the list after stopwords", 'red'))


###################################################### FREQUENT ITEMSETS ####################################################################


#text = open("input3.txt","r")
text = list_of_sentences_with_stopwords
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
freqItemSets = apriori(df, min_support = 0.05, use_colnames=True)
freq_items = set()
for x in freqItemSets['itemsets']:
    for y in x:
        freq_items.add(y)
#print(freqItemSets.head(20))
print(colored("frequent itemsets are shown above", 'green'))

############################################################## ENDS HERE #####################################################################
    
    


def retrieved_matrix(summary):

	row_numbers = [-1]
	for x in summary:
		row_numbers.append(x)
	matrix = np.array(row_numbers)

	## Use summary length instead to make retrieving easier
	for i in range(len(summary)):
		row = [summary[i]]
		sentence = list_of_sentences_with_stopwords[summary[i]]
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

			second_sentence = list_of_sentences_with_stopwords[summary[j]]
			second_set_of_words = {word for word in second_sentence.split()}
			second_sentence_size = len(second_set_of_words)

			##not count stopwords
			#count = len(set_of_words.intersection(second_set_of_words))
			#edge_weight = round(count / (sentence_size + second_sentence_size), 3)
            
            
            
			set_of_commonalities = set_of_words.intersection(second_set_of_words)
			count = len(set_of_commonalities)
			for x in set_of_commonalities:
				if x in freq_items: 
					print("Im here")
					count+=1
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

print(summary_matrix([8,17,19,21,23]))
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
#list_of_sentences_with_stopwords = [sentence for sentence in list_of_sentences if len(sentence) > 0]

my_terms=[]

for index in best:
	print('['+str(index)+'] ', end='')
	print(list_of_sentences[index], end='.')
	my_terms.append(list_of_sentences[index])



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


# In[ ]:





# In[ ]:




