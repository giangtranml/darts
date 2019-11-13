import torch
import utils
from datasets import MarkInspectionDataset
import logging
import torchvision


def mobilenetv2(device):
	model = torch.hub.load("pytorch/vision", "mobilenet_v2", pretrained=True)

	last_classifier = torch.nn.Sequential(
		torch.nn.Dropout(0.2),
		torch.nn.Linear(in_features=1280, out_features=512),
		torch.nn.ReLU(),
		torch.nn.Dropout(0.1),
		torch.nn.Linear(in_features=512, out_features=128),
		torch.nn.ReLU(),
		torch.nn.Dropout(0.1),
		torch.nn.Linear(in_features=128, out_features=2),
	)

	criterion = torch.nn.CrossEntropyLoss()

	model.classifier = last_classifier

	# for p in model.features.parameters():
	# 	p.requires_grad = False
	
	model = model.to(device)
	return model


def main():
	device = "cuda" if torch.cuda.is_available() else "cpu"
	args = utils.config(searching=False)

	criterion = torch.nn.CrossEntropyLoss()

	train_transform, test_transform = utils.data_transform_cifar10()
	train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, transform=train_transform, download=True)
	test_data = torchvision.datasets.CIFAR10(root=args.data, train=False, transform=test_transform)

	model = mobilenetv2(device)

	weights_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
										lr=args.learning_rate,
										momentum=args.momentum,
										weight_decay=args.weight_decay)

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
		train_acc, train_loss, train_cm = train(train_queue, model, device, weights_optimizer, criterion, args)
		logging.info("Train Accuracy: %f", train_acc)
		logging.info("Train Confusion Matrix: \n %s", train_cm)
		# validation
		valid_acc, valid_loss, valid_cm = evaluate(test_queue, model, device, criterion)
		logging.info('Validation Accuracy: %f', valid_acc)
		logging.info("Validation Confusion Matrix: \n %s", valid_cm)

		#utils.save(darts, os.path.join(args.save, 'weights%d.pt' % epoch))
		scheduler.step()
		logging.info("-------------------------------------------------------------------")
		
def train(train_queue, model, device, optimizer, criterion, args):
	objs = utils.AverageMeter()
	top1 = utils.AverageMeter()
	model.train()
	confusion_matrix = None
	for step, (input_train, target_train) in enumerate(train_queue):
		n = input_train.size(0)
		input_train = input_train.to(device)
		target_train = target_train.squeeze().long().to(device)
		logits = model(input_train)
		loss = criterion(logits, target_train)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		acc, confusion = utils.metric(logits, target_train)
		objs.update(loss.item(), n)
		top1.update(acc, n)

		if step % args.report_freq == 0:
			logging.info('Train | Step: %d | Loss: %e | Accuracy: %.3f', step, objs.avg, top1.avg)

		if confusion_matrix is None:
			confusion_matrix = confusion
		else:
			confusion_matrix += confusion

	return top1.avg, objs.avg, confusion_matrix

def evaluate(test_queue, model, device, criterion):
	objs = utils.AverageMeter()
	top1 = utils.AverageMeter()
	model.eval()
	confusion_matrix = None

	for step, (input_test, target_test) in enumerate(test_queue):
		input_test = input_test.to(device)
		target_test = target_test.squeeze().long().to(device)
		with torch.no_grad():
			logits = model(input_test)
			loss = criterion(logits, target_test)

		acc, confusion = utils.metric(logits, target_test)
		n = input_test.size(0)
		objs.update(loss.item(), n)
		top1.update(acc, n)
		if confusion_matrix is None:
			confusion_matrix = confusion
		else:
			confusion_matrix += confusion

	return top1.avg, objs.avg, confusion_matrix

if __name__ == "__main__":
	main()