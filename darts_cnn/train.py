import torch
import torch.nn as nn
F = nn.functional
from darts import DARTS, CustomDataParallel
import argparse
import utils
import torchvision
import logging, os
from genotypes import Genotype, DARTS_
import torch.backends.cudnn as cudnn


def training_phase(args, device, genotype):
	cudnn.benchmark = True
	cudnn.enabled = True
	torch.manual_seed(args.manual_seed)
	torch.cuda.manual_seed(args.manual_seed)
	C = args.init_channels
	num_cells = args.num_cells
	num_nodes = args.num_nodes
	num_classes = args.num_classes
	criterion = nn.CrossEntropyLoss()
	logging.info("args = %s", args)

	train_transform, test_transform = utils.data_transform_cifar10()
	train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, transform=train_transform, download=True)
	test_data = torchvision.datasets.CIFAR10(root=args.data, train=False, transform=test_transform)

	darts = DARTS(C, num_cells, num_nodes, num_classes, criterion, found_genotype=genotype)
	if torch.cuda.device_count() > 1:
		darts = CustomDataParallel(darts).to(device)
	else:
		darts = darts.to(device)
	weights_optimizer = torch.optim.SGD(darts.parameters(),
										lr=args.learning_rate,
										momentum=args.momentum,
										weight_decay=args.weight_decay)
	darts.set_optimizers(None, weights_optimizer)

	train_queue = torch.utils.data.DataLoader(
		train_data, batch_size=args.batch_size,
		shuffle=True, pin_memory=True, num_workers=2)

	test_queue = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, pin_memory=True, num_workers=2)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weights_optimizer, float(args.epochs), eta_min=args.learning_rate_min)

	for epoch in range(args.epochs):
		logging.info("EPOCH %d", epoch)
		lr = weights_optimizer.param_groups[0]["lr"]
		logging.info('Learning rate %e', lr)

		# training
		train_acc, train_loss = train(train_queue, darts, device)
		logging.info('Train Accuracy: %.2f', train_acc)
		logging.info("Train Loss: %.4f", train_loss)

		# validation
		valid_acc, valid_loss = evaluate(test_queue, darts, device)
		logging.info('Validation Accuracy: %.2f', valid_acc)
		logging.info("Validation Loss: %.4f", valid_loss)

		utils.save(darts, os.path.join(args.save, 'weights%d.pt' % epoch))
		scheduler.step()
		logging.info("-------------------------------------------------------------------")

def train(train_queue, model, device):
	objs = utils.AverageMeter()
	top1 = utils.AverageMeter()

	for step, (input_train, target_train) in enumerate(train_queue):
		model.train()
		n = input_train.size(0)
		input_train = input_train.to(device)
		target_train = target_train.squeeze().long().to(device)

		logits, loss = model.weights_step(input_train, target_train, args.grad_clip)

		acc = utils.metric(logits, target_train)
		objs.update(loss.item(), n)
		top1.update(acc, n)

		if step % args.report_freq == 0:
			logging.info('Train | Step: %d | Loss: %e | Accuracy: %.3f', step, objs.avg, top1.avg)

	return top1.avg, objs.avg	

def evaluate(test_queue, model, device):
	objs = utils.AverageMeter()
	top1 = utils.AverageMeter()
	model.eval()

	for step, (input_test, target_test) in enumerate(test_queue):
		input_test = input_test.to(device)
		target_test = target_test.squeeze().long().to(device)
		with torch.no_grad():
			logits, loss = model.loss(input_test, target_test)

		acc = utils.metric(logits, target_test)
		n = input_test.size(0)
		objs.update(loss.item(), n)
		top1.update(acc, n)

	return top1.avg, objs.avg

if __name__ == "__main__":
	device = "cuda" if torch.cuda.is_available() else "cpu"
	genotype = DARTS_
	args = utils.config(searching=False)
	training_phase(args, device, genotype)
