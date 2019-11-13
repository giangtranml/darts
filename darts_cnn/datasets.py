from torchvision.datasets import VisionDataset, CIFAR10
import os
import pickle
from PIL import Image

class MarkInspectionDataset(VisionDataset):

	def __init__(self, root, transforms=None, transform=None, target_transform=None, train=True):
		super(MarkInspectionDataset, self).__init__(root, transforms, transform, target_transform)
		if train:
			X_path = os.path.join(root, "X_train.pkl")
			y_path = os.path.join(root, "y_train.pkl")
		else:
			X_path = os.path.join(root, "X_test.pkl")
			y_path = os.path.join(root, "y_test.pkl")

		with open(X_path, "rb") as f:
			self.data = pickle.load(f)
		with open(y_path, "rb") as f:
			self.targets = pickle.load(f)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)
			
		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)


