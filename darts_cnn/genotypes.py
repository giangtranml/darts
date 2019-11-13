from collections import namedtuple
import torch
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

DARTS_ = Genotype(
  normal=[
    [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],  
    [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
    [('sep_conv_3x3', 1), ('skip_connect', 0)], 
    [('skip_connect', 0), ('dil_conv_3x3', 2)]
  ],
  normal_concat=[2, 3, 4, 5], 
  reduce=[
    [('max_pool_3x3', 0), ('max_pool_3x3', 1)], 
    [('skip_connect', 2), ('max_pool_3x3', 1)], 
    [('max_pool_3x3', 0), ('skip_connect', 2)], 
    [('skip_connect', 2), ('max_pool_3x3', 1)]
  ],
  reduce_concat=[2, 3, 4, 5])

def parse(weights, num_nodes, k_strongest):
	"""
	Parse the continous parameters `alphas` with softmax to discrete connections.
	The authors choose to keep 2 edges with a strongest operation

	`weights` has shape = (n_edges, n_ops).
	If `num_nodes` = 4 then `n_edges` = 16.
		- weights[0, 1] for first node.
		- weights[2, 3, 4] for second node.
		- weights[5, 6, 7, 8] for third node.
		- weights[9, 10, 11, 12, 13] for fourth node.
	 """
	gene = []
	start = 0
	offset = 2
	for i in range(num_nodes):
		end = offset + start
		ops_max, ops_max_ind = torch.max(weights[start:end, 1:], axis=1)  # choose strongest operations for every edges, exclude 'none' operation.
		k_max, k_max_ind = torch.topk(ops_max, k=k_strongest, axis=0) # from those strongest operations for every edges, choose k strongest edges.
		gene.append([])
		for k, ind in zip(k_max, k_max_ind):
			gene[-1].append((PRIMITIVES[ops_max_ind[ops_max == k] + 1], ind.item()))
		start += offset
		offset += 1
	return gene