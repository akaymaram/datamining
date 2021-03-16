# dag.py 1.0.0

import networkx as nx
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


graph = nx.DiGraph()

myfile = open("input.txt", "r")
content = myfile.read()
myfile.close()

list_of_sentences = [sentence for sentence in content.split(".") if len(sentence) > 0]
print(list_of_sentences)

for i in range(0, len(list_of_sentences)):
	set_of_words_in_the_sentence = {word for word in list_of_sentences[i].split()}
	for j in range(i+1, len(list_of_sentences)):
		count = 0
		for word_in_next_sentence in list_of_sentences[j].split():
			if word_in_next_sentence in set_of_words_in_the_sentence:
				count+=1
		graph.add_edges_from([(i, j)], weight=count)
		j = j + 1
	i+= 1


pos = nx.spring_layout(graph, seed=7)

# plt.tight_layout()
# nx.draw_networkx_nodes(graph, pos, node_size=700)
# nx.draw_networkx_edges(graph, pos, width=6)
nx.draw_networkx_edges(graph,pos,edge_color='b',alpha = 0.6)
nx.draw_networkx_edge_labels(graph,pos,edge_labels = nx.get_edge_attributes(graph,'weight'))
nx.draw_networkx(graph, pos,arrows=False)
plt.show()
# plt.savefig("g.png", format="PNG")
plt.clf()