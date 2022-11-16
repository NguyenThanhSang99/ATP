import pickle 
from hierarchical_matrix import  build_hierarchical_matrix
import numpy as np 
import os.path
from tools import run_command
from tools import dir_tail_name
from matrix_factorization import run_nmf
import scipy

def embedding(original_graph,deleted_edges_file,rank = 30, is_dense_matrix = False,using_GPU = False, using_svd = False, strategy = "ln"):
	
	matrix,ranking_difference_file = build_hierarchical_matrix(original_graph,deleted_edges_file,is_dense_matrix = is_dense_matrix, strategy = strategy)
	
	dir_name,tail = dir_tail_name(original_graph)
	if is_dense_matrix:
		print("matrix shape: %s" % str(matrix.shape))
		if using_svd:
			W, s, V = np.linalg.svd(matrix, full_matrices=False)
			S = np.diag(s)
			H = np.dot(S,V)
			print("(SVD) W matrix shape: %s" % str(W.shape))
			print("(SVD) H matrix shape: %s" % str(H.shape))
			return W,H
		
		# from sklearn.decomposition import NMF
		# model = NMF(n_components= rank, init='random', random_state=0)
		# W = model.fit_transform(matrix)
		# H = model.components_
		
		
		W,H = run_nmf(matrix,rank = rank)
		print("(NMF) W matrix shape: %s" % str(W.shape))
		print("(NMF) H matrix shape: %s" % str(H.shape))
		# print np.matmul(W,H)
		return W,H
	else:
		if using_svd:
			
			W, s, V = scipy.sparse.linalg.svds(matrix,k = rank,)
			S = np.diag(s)
			H = np.dot(S,V)
			print("(SVDs) W matrix shape: %s" % str(W.shape))
			print("(SVDs) H matrix shape: %s" % str(H.shape))
			return W,H
		saved_matrix_file_name = os.path.join(dir_name,tail.split(".")[0]+"_HM.pkl")
		saved_WH_file_name = os.path.join(dir_name,tail.split(".")[0]+"_HM_WH.pkl")

		print("saved matrix file name: %s " % saved_matrix_file_name)
		with open(saved_matrix_file_name,"wb") as f:
			pickle.dump(matrix,f)
		command = "python libpmf-1.41/python/pmf_main.py --matrix " + saved_matrix_file_name + " --model " + saved_WH_file_name + " --rank " + str(rank)
		run_command(command)

		with open(saved_WH_file_name,"rb") as f:
			model = pickle.load(f)
			return model['W'],model['H'].T
