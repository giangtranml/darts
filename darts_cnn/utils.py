import torch
import torchvision.transforms as transforms
import os, sys, logging, argparse, time
from sklearn.metrics import confusion_matrix, classification_report

class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def flatten_params(xs):
	return torch.cat([x.view(-1) for x in xs])


def data_transform():
	train_transform = transforms.Compose([
		transforms.RandomRotation(5),
		transforms.RandomGrayscale(),
		transforms.ToTensor()
	])

	test_transform = transforms.Compose([
		transforms.ToTensor()
	])

	return train_transform, test_transform

def data_transform_cifar10():
	CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
	CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])

	valid_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	return train_transform, valid_transform

def metric(output, target,):
	softmax = torch.softmax(output, dim=-1)
	pred = torch.argmax(softmax, dim=-1)
	acc = pred[pred == target].shape[0]/pred.shape[0] * 100.
	#confusion = confusion_matrix(target.cpu().numpy(), pred.squeeze().cpu().numpy())
	return acc

def save(model, model_path):
  	torch.save(model.state_dict(), model_path)

def load(model, model_path):
  	model.load_state_dict(torch.load(model_path))

def create_exp_dir(path):
	if not os.path.exists(path):
		os.mkdir(path)
	print('Experiment dir : {}'.format(path))

def config(searching=True):
	parser = argparse.ArgumentParser("DARTS (Differentiable Architecture Search)")
	parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
	parser.add_argument('--model', type=str, default='darts', help='use models: [darts, pcdarts]')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
	parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
	parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
	parser.add_argument('--gpus', type=str, default="0", help='gpu device ids')
	parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
	parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
	parser.add_argument('--num_cells', type=int, default=8, help='total number of layers')
	parser.add_argument('--num_nodes', type=int, default=4, help='total number of nodes in a cell')
	parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
	parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
	parser.add_argument('--save', type=str, default='EXP', help='experiment name')
	parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
	parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
	parser.add_argument('--manual_seed', type=int, default=2, help='manual seed of the experiment')
	parser.add_argument('--order', type=int, default=2, help='either use first-order approximation or second-order approximation')
	parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for architecture encoding')
	parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for architecture encoding')
	args = parser.parse_args()
	
	string = "Search" if searching else "Train"
	args.save = '{}-{}'.format(string, time.strftime("%Y%m%d-%H%M%S"))
	create_exp_dir(args.save)

	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
		format=log_format, datefmt='%m/%d %I:%M:%S %p')
	fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logging.getLogger().addHandler(fh)
	return args