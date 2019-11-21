import torch
import torch.nn as nn
from genotypes import PRIMITIVES
from operations import OPS, FactorizedReduce, ReLUConvBN, Identity


class MixedOp(nn.Module):

	def __init__(self, C, stride):
		super(MixedOp, self).__init__()
		self._ops = nn.ModuleList()
		for primitive in PRIMITIVES:
			op = OPS[primitive](C, stride, False)
			if 'pool' in primitive:
				op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
		self._ops.append(op)

	def forward(self, x, weights):
		"""
		Apply operation to a pair of nodes (i, j). Each element in the `weights` vector = (alpha_1, alpha_2, ..., alpha_o) 
		is multiplied with corresponding operation in `ops` apply to input `x`. 
		
		Result to a big vector = (alpha_1 * ops_1(x), alpha_2 * ops_2(x), ..., alpha_o * ops_o(x))
		where each element `alpha_o` is a scalar and `ops_o(x)` is a tensor with any sizes.
		
		Then, take the sum of that big vector, this is the same as take dot product between 2 vectors,
		but since `weights` and `ops` is not the same type (`weights` is a vector of scalars, `ops` is 
		a vector of Operations class) we must do that manually.

		Parameters
		----------
		x: input tensor to this Mixed Operation.
		weights: vector weight corresponding to the Mixed Operation.

		"""
		# w_dot_ops = []
		# for w, op in zip(weights, self._ops):
		# 	w_dot_ops.append(w * op(x))
		# return sum(w_dot_ops)
		return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
	"""
	Abstract Cell class. A Cell is a Directed Acyclic Graph (DAG) of `n` nodes, with first two nodes are initial nodes
	or input nodes from 2 previous cells C[k-2] and C[k-1]. 

	The edges, each edge connects 2 nodes (xi, xj) applying a operation to transform xi -> xj. 
	`k` edges apply `k` operations to xi -> xj.
	"""

	def __init__(self, num_nodes, C_pp, C_p, C, reduction, reduction_prev, op_cls=MixedOp):
		"""
		Initialize a cell.

		Parameters
		----------
		C_pp: number channels of the previous previous cell C[k-2].
		C_p: number channels of the previous cell C[k-1].
		C: number channels of the current cell C[k]. 
		reduction: flag to check whether this cell is reduction cell or not.
		reduction_prev: flag to check whether previous cell is reduction cell.
		ops_cls: operation class 
		"""
		super(Cell, self).__init__()
		self.num_nodes = num_nodes
		self.C_pp = C_pp
		self.C_p = C_p
		self.C = C
		self.reduction = reduction
		self.reduction_prev = reduction_prev
		self._init_nodes(op_cls)

	def _init_nodes(self, op_cls):
		"""
		Initialize nodes to create DAG with 2 input nodes come from 2 previous cell C[k-2] and C[k-1]
		"""
		self.node_ops = nn.ModuleList()
		if self.reduction_prev:
			self.node0 = FactorizedReduce(self.C_pp, self.C, affine=False)
		else:
			self.node0 = ReLUConvBN(self.C_pp, self.C, 1, 1, 0, affine=False)
		self.node1 = ReLUConvBN(self.C_p, self.C, 1, 1, 0, affine=False)

		for i in range(self.num_nodes):
			# Creating edges connect node `i` to other nodes `j`. `j < i` 
			for j in range(2+i):
				stride = 2 if self.reduction and j < 2 else 1
				op = op_cls(self.C, stride)
				self.node_ops.append(op)

	def forward(self, s0, s1, weights_alpha):
		"""
		Apply node-level operations with mixed operation in the cell (DAG). 
		"""
		s0 = self.node0(s0)
		s1 = self.node1(s1)

		states = [s0, s1]
		offset = 0
		for i in range(self.num_nodes):
			# Apply operation to transform node `j` to node `i`, refer equation (1)
			compute_node = sum((self.node_ops[offset+j](h, weights_alpha[offset+j]) for j, h in enumerate(states)))  
			offset += len(states)
			states.append(compute_node)

		return torch.cat(states[-self.num_nodes:], dim=1) # concat with respect to channel dimension. (N, C, H, W)


class DerivedCell(nn.Module):

	def __init__(self, genotype, C_pp, C_p, C, reduction, reduction_prev, dropout_rate):
		super(DerivedCell, self).__init__()
		self.reduction = reduction
		if reduction_prev:
			self.node0 = FactorizedReduce(C_pp, C)
		else:
			self.node0 = ReLUConvBN(C_pp, C, 1, 1, 0)
		self.node1 = ReLUConvBN(C_p, C, 1, 1, 0)
		self.dropout = nn.Dropout(dropout_rate)

		if reduction:
			dag = genotype.reduce
			concat = genotype.reduce_concat
		else:
			dag = genotype.normal
			concat = genotype.normal_concat
		self.num_nodes = len(dag)
		self.concat = concat
		self.ops, self.nodes = self._compile_dag(C, dag)

	def _compile_dag(self, C, dag):
		ops = nn.ModuleList()
		nodes = []
		for i, connections in enumerate(dag):
			ops.append(nn.ModuleList())
			nodes.append([])
			for conn in connections:
				name, node = conn
				stride = 2 if self.reduction and node < 2 else 1
				op = OPS[name](C, stride, True)
				ops[-1].append(op)
				nodes[-1].append(node)
		return ops, nodes

	def forward(self, s0, s1):
		s0 = self.node0(s0)
		s1 = self.node1(s1)

		states = [s0, s1]
		for i in range(self.num_nodes):
			s = []
			for op, node in zip(self.ops[i], self.nodes[i]):
				h = op(states[node])
				if not isinstance(op, Identity):
					h = self.dropout(h)
				s.append(h)
			states.append(sum(s))
		return torch.cat([states[i] for i in self.concat], dim=1)


class MixedOpPCDarts(nn.Module):
	
	def __init__(self, C, stride, k=4):
		"""
		Parameters
		----------
		C: number input channels of the operation.
		stride: stride apply to conv/pool layer.
		k: partial channels connection hyper-parameter.
		"""
		super(MixedOpPCDarts, self).__init__()
		self.k = k
		self.ops = nn.ModuleList()
		for primitive in PRIMITIVES:
			op = OPS[primitive](C//k, stride, False)
			if 'pool' in primitive:
				op = nn.Sequential(op, nn.BatchNorm2d(C//k, affine=False))
		self.ops.append(op)
		self.mp = nn.MaxPool2d(2, 2)

	def forward(self, x, weights):
		channel_dim = x.shape[1]
		x_process = x[:, :channel_dim//self.k, :, :]
		x_bypass = x[:, channel_dim//self.k:, :, :]

		out_process = sum(w * op(x_process) for w, op in zip(weights, self.ops))
		if out_process.shape[2] == x.shape[2]:
			out = torch.cat([out_process, x_bypass], dim=1)
		else:
			out = torch.cat([out_process, self.mp(x_bypass)], dim=1)
		out = channel_shuffle(out, self.k)
		return out

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class CellPCDarts(Cell):

	def __init__(self, num_nodes, C_pp, C_p, C, reduction, reduction_prev, op_cls=MixedOpPCDarts):
		super(CellPCDarts, self).__init__(num_nodes, C_pp, C_p, C, reduction, reduction_prev, op_cls)

	def forward(self, s0, s1, weights_alpha, weights_beta):
		s0 = self.node0(s0)
		s1 = self.node1(s1)

		states = [s0, s1]
		offset = 0
		for i in range(self.num_nodes):
			# Apply operation to transform node `j` to node `i`, refer equation (1)
			compute_node = sum(weights_beta[offset+j]*self.node_ops[offset+j](h, weights_alpha[offset+j]) for j, h in enumerate(states))  
			offset += len(states)
			states.append(compute_node)

		return torch.cat(states[-self.num_nodes:], dim=1) # concat with respect to channel dimension. (N, C, H, W)	

