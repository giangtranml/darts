import torch
import torch.nn as nn
F = nn.functional
from genotypes import PRIMITIVES, Genotype, parse, parse_pcdarts
from cell_darts import Cell, DerivedCell, CellPCDarts
from utils import flatten_params
from copy import deepcopy

def print_gpu_usage():
	device = torch.cuda.current_device()
	print("Memory usage of GPU %s: %s" % (device, torch.cuda.memory_allocated(device)))


class Darts(nn.Module):
	"""
	Differentiable Architecture Search (Liu et al., 2018) https://arxiv.org/pdf/1806.09055.pdf

	Authors observed that typical CNN is the stack of motif cells. Then instead of searching for the whole architecture,
		they search for the best cell, then stack those cells to form a CNN.

	2 generic cells: Normal Cell and Reduction Cell. A cell is Directed Acyclic Graph (DAG) with `N` nodes. 

	There are 2 main stages: Searching and Training.
	In Searching stage, darts will jointly optimize `alphas` (architecture parameters) base on validation set 
		and `weights` (network parameters) on training set. Then derive the best cell (genotype) from `alphas`.

	In Training stage, use the searched cell to train normally to optimize `weights` (network parameters)
	"""

	def __init__(self, C, num_cells, num_nodes, num_classes, criterion, cell_cls=Cell, found_genotype=None):
		"""
		Parameters
		----------
		C: the initial channels to the DARTS.
		num_cells: number of cells/layers in the DARTS. 
					8 cells for architecture searching, and 20 cells for searched architecture.
		num_nodes: number of nodes in a cell.
		num_classes: number of classes for classification problem.
		criterion: use what kind of criterion to evaluate the training.
		found_genotype: if not None, it should be the searched cell, then use for training instead of searching.
		"""
		super(Darts, self).__init__()
		self.C = C
		self.num_cells = num_cells
		self.num_nodes = num_nodes
		self.num_classes = num_classes
		self.criterion = criterion
		self.found_genotype = found_genotype

		self._init_cells(cell_cls)
		if not found_genotype:
			self._init_alphas()

	def _init_cells(self, cell_cls):
		self.cells = nn.ModuleList()
		
		stem_multiplier = 3

		C_curr = stem_multiplier*self.C
		self.initial_cell = nn.Sequential(
			nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
			nn.BatchNorm2d(C_curr)
		)

		multiplier = 4
		C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
		
		reduction_prev = False
		for i in range(self.num_cells):
			# 1/3 and 2/3 of the length of the cell is reduction cell
			if i in [self.num_cells//3, 2*self.num_cells//3]:
				C_curr *= 2
				reduction = True
			else:
				reduction = False
			if self.found_genotype:
				cell = DerivedCell(self.found_genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, 0.2)
			else:
				cell = cell_cls(self.num_nodes, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
			reduction_prev = reduction
			self.cells += [cell]
			C_prev_prev, C_prev = C_prev, multiplier*C_curr

		self.global_pooling = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(C_prev, self.num_classes)

	def _init_alphas(self):
		"""
		Initialize architecture parameters: `alphas_normal` and `alphas_reduce`.
		The shape of `alphas_normal` and `alphas_reduce` is (num_edges_connect, num_ops).

		By default, 2 initial nodes in a cell are output nodes from 2 previous cells (c[k-2] and c[k-1]).
			So, if num_nodes = 4, then we have total number of edges connect: 2 + 3 + 4 + 5 = 14
			- First node connects 2 intial nodes.
			- Second node connects 2 initial nodes and first node.
			- Third node connects 2 initial nodes, first node and second node.
			- And so on.
		"""
		num_edges = sum(2+i for i in range(self.num_nodes))
		num_ops = len(PRIMITIVES)
		self.alphas_normal = nn.Parameter(1e-3*torch.randn(num_edges, num_ops), requires_grad=True)
		self.alphas_reduce = nn.Parameter(1e-3*torch.randn(num_edges, num_ops), requires_grad=True)
		self.arch_parameters = [self.alphas_normal, self.alphas_reduce]

	def set_optimizers(self, alphas_optimizer, weights_optimizer):
		self.alphas_optimizer = alphas_optimizer
		self.weights_optimizer = weights_optimizer

	def forward(self, input):
		s0 = s1 = self.initial_cell(input)
		for i, cell in enumerate(self.cells):
			if not self.found_genotype:
				if cell.reduction:
					weights = F.softmax(self.alphas_reduce, dim=-1)
				else:
					weights = F.softmax(self.alphas_normal, dim=-1)
				s0, s1 = s1, cell(s0, s1, weights)
			else:
				s0, s1 = s1, cell(s0, s1)
		out = self.global_pooling(s1)
		logits = self.classifier(out.view(out.size(0),-1))
		return logits

	def genotype(self):

		with torch.no_grad():
			gene_normal = parse(weights=F.softmax(self.alphas_normal, dim=-1), num_nodes=self.num_nodes, k_strongest=2)
			gene_reduce = parse(weights=F.softmax(self.alphas_reduce, dim=-1), num_nodes=self.num_nodes, k_strongest=2)

		concat = range(2, self.num_nodes+2)
		genotype = Genotype(
			normal=gene_normal, normal_concat=concat,
			reduce=gene_reduce, reduce_concat=concat
		)

		return genotype

	def loss(self, X, Y):
		logits = self(X)
		return logits, self.criterion(logits, Y)

	def first_order_approximation(self, X_val, Y_val):
		_, loss_val = self.loss(X_val, Y_val)
		loss_val.backward()

	def _hessian_vector_product(self, w_prime_grads, X_train, Y_train, r=1e-2):
		R = r/flatten_params(w_prime_grads).norm()
		
		with torch.no_grad():
			for p, v in zip(self.parameters(), w_prime_grads):
				p.add_(R, v)
		_, loss = self.loss(X_train, Y_train)
		grads_p = torch.autograd.grad(loss, self.arch_parameters)

		with torch.no_grad():
			for p, v in zip(self.parameters(), w_prime_grads):
				p.sub_(2*R, v)
		_, loss = self.loss(X_train, Y_train)
		grads_n = torch.autograd.grad(loss, self.arch_parameters)

		with torch.no_grad():
			for p, v in zip(self.parameters(), w_prime_grads):
				p.add_(R, v)

		return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

	def second_order_approximation(self, X_train, Y_train, X_val, Y_val):
		_, loss_train = self.loss(X_train, Y_train)
		w = flatten_params(self.parameters())
		momentum = self.weights_optimizer.param_groups[0]["momentum"]
		weight_decay = self.weights_optimizer.param_groups[0]["weight_decay"]
		eta = self.weights_optimizer.param_groups[0]["lr"]
		try:
			velocity = flatten_params(self.weights_optimizer.state[v]['momentum_buffer'] for v in self.parameters())
		except KeyError:
			velocity = torch.zeros_like(w)
		# gradient of weight parameters, plus L2 regularization if neccessary.
		w_grad = flatten_params(torch.autograd.grad(loss_train, self.parameters())) + weight_decay*w
		velocity = momentum*velocity + w_grad
		w_prime = w - eta*velocity

		unrolled_model = deepcopy(self)
		params, offset = unrolled_model.state_dict(), 0
		for k, v in self.named_parameters():
			v_length = torch.prod(torch.Tensor(list(v.size()))).int()
			params[k] = w_prime[offset: offset+v_length].view(v.size())
			offset += v_length

		_, loss_val = unrolled_model.loss(X_val, Y_val)
		loss_val.backward()

		alpha_grads = [alpha.grad for alpha in unrolled_model.arch_parameters] # derivative of loss_val w.r.t architecture parameters 
		w_prime_grads = [w_prime.grad for w_prime in unrolled_model.parameters()] # derivative of loss_val w.r.t w' parameters
		hessian_vector_grads = self._hessian_vector_product(w_prime_grads, X_train, Y_train)

		with torch.no_grad():
			for alpha, alpha_grad, hessian_vector_grad in zip(self.arch_parameters, alpha_grads, hessian_vector_grads):
				alpha.grad = alpha_grad - hessian_vector_grad


	def alphas_step(self, X_train, Y_train, X_val, Y_val, order=1):
		assert order in [1, 2], "Either first_order_appoximation (1) or second_order_approximation (2)"
		assert not self.found_genotype, "Use this function in searching phase only."
		self.alphas_optimizer.zero_grad()
		if order == 1:
			self.first_order_approximation(X_val, Y_val)
		else:
			self.second_order_approximation(X_train, Y_train, X_val, Y_val)
		self.alphas_optimizer.step()

	def weights_step(self, X_train, Y_train, grad_clip):
		self.weights_optimizer.zero_grad()
		logits, loss = self.loss(X_train, Y_train)
		loss.backward()
		nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
		self.weights_optimizer.step()
		return logits, loss


class PCDarts(Darts):
	"""
	Partial Channel Connection Differentiable Architecture Search (PC-DARTS)
	Paper: https://arxiv.org/pdf/1907.05737.pdf
	"""

	def __init__(self, C, num_cells, num_nodes, num_classes, criterion, cell_cls=CellPCDarts, found_genotype=None):
		super(PCDarts, self).__init__(C, num_cells, num_nodes, num_classes, criterion, cell_cls, found_genotype)

	def _init_alphas(self):
		"""
		Initialize `betas_normal` and `betas_reduce` use in edge normalization mentioned in the paper. 
		`betas_normal` and `betas_reduce` has shape = (num_edges_connect).
		"""
		super(PCDarts, self)._init_alphas()
		num_edges = sum(2+i for i in range(self.num_nodes))
		self.betas_normal = nn.Parameter(1e-3*torch.randn(num_edges), requires_grad=True)
		self.betas_reduce = nn.Parameter(1e-3*torch.randn(num_edges), requires_grad=True)
		self.arch_parameters += [self.betas_normal, self.betas_reduce]

	def forward(self, input):
		s0 = s1 = self.initial_cell(input)
		for i, cell in enumerate(self.cells):
			if cell.reduction:
				weights_alpha = F.softmax(self.alphas_reduce, dim=-1)
				n = 3
				start = 2
				weights_beta = F.softmax(self.betas_reduce[0:2], dim=-1)
				for i in range(self.num_nodes-1):
					end = start + n
					temp_weights_beta = F.softmax(self.betas_reduce[start:end], dim=-1)
					start = end
					n += 1
					weights_beta = torch.cat([weights_beta, temp_weights_beta], dim=0)
			else:
				weights_alpha = F.softmax(self.alphas_normal, dim=-1)
				n = 3
				start = 2
				weights_beta = F.softmax(self.betas_normal[0:2], dim=-1)
				for i in range(self.num_nodes-1):
					end = start + n
					temp_weights_beta = F.softmax(self.betas_normal[start:end], dim=-1)
					start = end
					n += 1
					weights_beta = torch.cat([weights_beta, temp_weights_beta], dim=0)
			s0, s1 = s1, cell(s0, s1, weights_alpha, weights_beta)
		out = self.global_pooling(s1)
		logits = self.classifier(out.view(out.size(0),-1))
		return logits

	def genotype(self):
		n = 3
		start = 2
		weights_beta_reduce = F.softmax(self.betas_reduce[0:2], dim=-1)
		weights_beta_normal = F.softmax(self.betas_normal[0:2], dim=-1)
		for i in range(self.num_nodes-1):
			end = start + n
			temp_weights_beta_reduce = F.softmax(self.betas_reduce[start:end], dim=-1)
			temp_weights_beta_normal = F.softmax(self.betas_normal[start:end], dim=-1)
			start = end
			n += 1
			weights_beta_reduce = torch.cat([weights_beta_reduce, temp_weights_beta_reduce], dim=0)
			weights_beta_normal = torch.cat([weights_beta_normal, temp_weights_beta_normal], dim=0)
		
		gene_normal = parse_pcdarts(F.softmax(self.alphas_normal, dim=-1), weights_beta_normal, self.num_nodes, 2)
		gene_reduce = parse_pcdarts(F.softmax(self.alphas_reduce, dim=-1), weights_beta_reduce, self.num_nodes, 2)

		concat = range(2, self.num_nodes+2)
		genotype = Genotype(
			normal=gene_normal, normal_concat=concat,
			reduce=gene_reduce, reduce_concat=concat
		)
		return genotype