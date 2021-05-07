# from adagio.core.apicalls import *
file_path = r'D:\2. Research\My thesis\SVM\Wakelock\buggy\AndTweet-0.2.4.apk'#\multilock\CSipSimple-rev-da248d1132.apk'
leak_dir = r'C:\Users\Umair Khan\PycharmProjects\malware\wakeleakdata'
clean_dir = r'G:\Wakelock apps\apksfiles\fcg'#r'C:\Users\Umair Khan\PycharmProjects\malware\wakecleandata'#
# print(list_calls(file_path))
# print(list_XREF(file_path))
##///////////////////////////////////////////////
import networkx as nx
import time
# from adagio.common import ml
# from adagio.core import analysis
# from adagio.core import graphs
# import numpy as np
# def compute_label_histogram(g):
#     """ Compute the neighborhood hash of a graph g and return
#         the histogram of the hashed labels.
#     """
#
#     g_hash = ml.neighborhood_hash(g)
#     print("GHash=", g_hash)
#     g_x = ml.label_histogram(g_hash)
#     return g_x
# g = nx.read_gpickle(r'C:\Users\Umair Khan\PycharmProjects\malware\cleandata\7c4df6b2dd0a1dfc7abd66fb755370d4f2bbd7be81c77aa3bf0c7e425c92146b.pz')
# size = g.number_of_nodes()
# print("Size=", size)
# max_node_size = 0
# feature_vector_times = []
# label_dist = np.zeros(2**15)
# sample_sizes = []
# neighborhood_sizes = []
# class_dist = np.zeros(15)
# X = []
# Y = np.array([])
# fnames = []
# fcg = graphs.FCG(file_path).get_graph()
# if size < max_node_size or max_node_size == 0:
#     if size > 0:
#         t0 = time.time()
#         x_i = compute_label_histogram(g)
#         print("Histogram = ", x_i)
#         # save feature vector computing time for
#         # performance evaluation
#         feature_vector_times.append(time.time() - t0)
#         # save distribution of generated labels
#         label_dist = np.sum([label_dist, x_i], axis=0)
#         # save sizes of the sample for further analysis
#         # of the dataset properties
#         sample_sizes.append(size)
#         neighborhood_sizes += ml.neighborhood_sizes(g)
#         for n, l in g.node.items():
#             class_dist = np.sum([class_dist, l["label"]], axis=0)
#         # delete nx object to free memory
#         del g
#         X.append(x_i)
#         Y = np.append(Y, [int(0)])
#         fnames.append(r'C:\Users\Umair Khan\PycharmProjects\malware\cleandata\7c4df6b2dd0a1dfc7abd66fb755370d4f2bbd7be81c77aa3bf0c7e425c92146b.pz')
#

##////////////////////////////////////////////////
from adagio.core.analysis import Analysis
import warnings
warnings.filterwarnings("ignore")
# a = Analysis([leak_dir, clean_dir], labels=[1,0], split=0.8)
# a.run_linear_experiment('roc.pz')
# a.plot_average_roc("average_roc.png", boundary=1.0)
# a.save_data()
##/////////////////////////////////////////////////
# from adagio.core import featureAnalysis as fa
# w_agg = fa.aggregate_binary_svm_weights(w, 13)

##/////////////////////////////////////////////////
# from adagio.common.ml import *
#
# from adagio.core.graphs import *
# fcg = FCG(file_path).get_graph()
# print(type(fcg))

# rwk_example()
# print(neighborhood_hash(fcg)) ##////////Error because of label xor
# print(count_sensitive_neighborhood_hash(fcg))##////////Error because of label xor
# print(xor_neighborhood_hash(fcg)) ##//////////////Error because of label xor////////////////////////
# print(array_labels_to_str(fcg)) ##///////////////No ouput/////////////////////////
# print(str_labels_to_array(fcg)) ##//////////////Error because dtyp = np.int64////////////////////////
# print(neighborhood_sizes(fcg)) ##///////////////////////Working//////////////////
# print(neighborhood_size_distribution(fcg)) ##///////////////////////Working//////////////////

# ##/////////////////////////////////////////////////
# from adagio.core.graphs import *
# import matplotlib.pyplot as plt
# import networkx as nx
# fcg = FCG(file_path)
# print(type(fcg.get_graph()))
# graph = fcg.get_graph()
# print(len(graph))
# # for nodes in nx.nodes(graph):
# #     print(nodes)
# pos = nx.spring_layout(graph, iterations=500)
# # print(len(nx.nodes(graph)), len(nx.edges(graph)))
# # pos = nx.circular_layout(graph)
# # print(nx.all_simple_paths(graph,))
# nx.draw_networkx_nodes(graph, pos=pos, node_color='r', node_size=15)
# nx.draw_networkx_edges(graph, pos, arrow=True)
# # nx.draw_networkx_labels(graph, pos=pos, labels={x: str(x) for x in graph.nodes}, font_size=8)
# plt.axis('off')
# import os
# print(os.getcwd())
# # plt.draw()
# plt.savefig("CFG.png", dpi=300)
# plt.show()
##////////////////////Experiment Code/////////////////////////////////
clean_dir = r'D:\2. Research\Journal\Access\Dataset\Clean'
leak_dir = r'D:\2. Research\Journal\Access\Dataset\Leak'
from adagio.core.graphs import *
process_dir(leak_dir, leak_dir, mode='FCG')
process_dir(clean_dir, clean_dir, mode='FCG')
from adagio.core.analysis import Analysis
import warnings
warnings.filterwarnings("ignore")
a = Analysis([leak_dir, clean_dir], labels=[1,0], split=0.8)
# a.run_linear_experiment('roc.pz')
# a.plot_average_roc("average_roc.png", boundary=1.0)
a.save_data()