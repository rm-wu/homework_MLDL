import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn

from torchvision.models import alexnet

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

import copy

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)

DATA_DIR = './Caltech101'
NUM_CLASSES = 101
from torch.utils.data import Dataset

class MySubset(Dataset):
	"""
	Subset of a dataset at specified indices.

	Arguments:
	dataset (Dataset): The whole Dataset
	indices (sequence): Indices in the whole set selected for subset
	"""
	def __init__(self, dataset, indices, transform):
		self.dataset = dataset
		self.indices = indices
		self.transform = transform

	def __getitem__(self, idx):
		im, labels = self.dataset[self.indices[idx]]
		return self.transform(im), labels

	def __len__(self):
		return len(self.indices)



def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


class Caltech(VisionDataset):
	def __init__(self, root, split='train', transform=None, target_transform=None):
		super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

		if split not in ['train', 'test']:
			raise ValueError("split must be equal to 'train' or 'test'")
		print(f"Split : {split}")

		if os.path.isdir(self.root):
			with open(os.path.join(self.root, f"{split}.txt")) as f:
				file_list = f.readlines()
		else:
			raise ValueError(f"{self.root} is not a directory")

		#remove class "BACKGROUND_Google"
		file_list = [f.strip('\n') for f in file_list if not f.startswith("BACKGROUND_Google/")]

		self.categories = set([f.split('/')[0] for f in file_list])
		self.categories = sorted(list(self.categories))

		self.y = []
		self.index = []
		self.images = []

		for (i, c) in enumerate(self.categories):
			#print(c)
			files = [f.split('/')[1].split('_')[1] for f in file_list if f.startswith(f"{c}/")]
			n = len(files)
			#print(len(files))
			images = []
			for f in files:
				img_name = os.path.join(self.root, 
										"101_ObjectCategories", 
										c, 
										f"image_{f}")
				image = pil_loader(img_name)
				images.append(image)
			self.index.extend(files)
			self.images.extend(images)
			self.y.extend(n*[i])


		
	def __getitem__(self, idx):
		'''
		__getitem__ should access an element through its index
		Args:
			index (int): Index

		Returns:
			tuple: (sample, target) where target is class_index of the target class.
		'''

		image = self.images[idx]
		label = self.y[idx]

		if self.transform is not None:
				image = self.transform(image)

		return image, label

	def __len__(self):
		'''
		The __len__ method returns the length of the dataset
		It is mandatory, as this is used by several other components
		'''
		length = len(self.index)
		return length



#class PyTorchWorker(Worker):
class PyTorchWorker(Worker):
	def __init__(self, batch_size=256, **kwargs):
		super().__init__(**kwargs)
		norm_imagenet_mean = (0.485, 0.456, 0.406)
		norm_imagenet_std  = (0.229, 0.224, 0.225)
		
		train_transform = transforms.Compose([
										#transforms.RandomCrop(200),
										transforms.RandomHorizontalFlip(),
										transforms.Resize((256, 256)),     
										transforms.CenterCrop(224),
										transforms.ColorJitter(hue=.05, saturation=.05),
										transforms.ToTensor(),
										transforms.Normalize(norm_imagenet_mean, norm_imagenet_std)])
		# Define transforms for the evaluation phase
		eval_transform = transforms.Compose([transforms.Resize((256, 256)),
										transforms.CenterCrop(224),                                   
										transforms.ToTensor(),
										transforms.Normalize(norm_imagenet_mean, norm_imagenet_std)])

		DATA_DIR = './Caltech101'
		# Prepare Pytorch train/test Datasets
		train_dataset_ = Caltech(DATA_DIR, split='train',  transform=None)
		test_dataset= Caltech(DATA_DIR, split='test', transform=eval_transform)

		train_idx, val_idx = train_test_split(np.arange(0, len(train_dataset_)), train_size=0.5,
											shuffle=True, stratify=train_dataset_.y)

		train_dataset = MySubset(train_dataset_, train_idx, train_transform)
		val_dataset = MySubset(train_dataset_, val_idx, eval_transform)
		train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
		val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
		test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
		
		# instantiate train_loader, validation_loader and test_loader
		self.train_loader = train_dataloader
		self.validation_loader = val_dataloader
		self.test_loader = test_dataloader


	def compute(self, config, budget, working_directory, *args, **kwargs):
		device = 'cuda'
		model = torchvision.models.alexnet(pretrained=True);print('Alexnet')
		model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
		model.to('cuda')
		criterion = torch.nn.CrossEntropyLoss()
		if config['optimizer'] == 'Adam':
			optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
		elif config['optimizer'] == 'SGD':
			optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])
		else:
			optimizer = torch.optim.SGD(model.parameter(), lr= config['lr'], momentum=config['sgd_momentum'], nesterov=True)

		train_accuracy = 0.0
		validation_accuracy = 0.0
		t_l = 0.0
		t_a = 0.0
		train_loss = -1
		validation_loss = -1
		dataloader = {'train': self.train_loader, 'val':self.validation_loader}
		for epoch in range(int(budget)):
			for phase in ['train', 'val']:
				running_loss = 0.0
				running_corrects = 0
				for i, (images, labels) in enumerate(dataloader[phase]):
					images = images.to(device)
					labels = labels.to(device)

					optimizer.zero_grad()

					with torch.set_grad_enabled(phase == 'train'):
						outputs = model(images)
						_, preds = torch.max(outputs.data, 1)
						loss = criterion(outputs, labels)

					if phase == 'train':
						loss.backward()
						optimizer.step()

					running_loss += loss.item() * images.size(0)
					running_corrects += torch.sum(preds == labels.data) 

				epoch_loss = running_loss / len(dataloader[phase].dataset)
				epoch_acc = (running_corrects.double() / len(dataloader[phase].dataset)).item()
				print(f"{epoch}  {phase} {epoch_loss} {epoch_acc}")
				if phase == 'val':
					if validation_loss == -1 or epoch_loss < validation_loss:
						validation_loss = epoch_loss
						validation_accuracy = epoch_acc
						train_loss = t_l 
						train_accuracy = t_a
				else:
					t_l = epoch_loss
					t_a = epoch_acc


		#train_accuracy = self.evaluate_accuracy(model, self.train_loader)
		#validation_accuracy = self.evaluate_accuracy(model, self.validation_loader)
		test_accuracy = self.evaluate_accuracy(model, self.test_loader)

		return ({
			'loss': validation_loss, 
			'info': {	'train loss': train_loss,
						'train accuracy': train_accuracy,
						'validation_loss': validation_loss,
						'validation accuracy': validation_accuracy,
						'test accuracy': test_accuracy,
					}
						
		})

	def evaluate_accuracy(self, model, data_loader):
		model.eval()
		correct=0
		with torch.no_grad():
			for x, y in data_loader:
				x,y =x.to('cuda'), y.to('cuda');
				output = model(x)
				#test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
				pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(y.view_as(pred)).sum().item()
		#import pdb; pdb.set_trace()	
		accuracy = correct/len(data_loader.sampler)
		return(accuracy)


	@staticmethod
	def get_configspace():
		cs = CS.ConfigurationSpace()

		lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-4', log=True)

		# For demonstration purposes, we add different optimizers as categorical hyperparameters.
		# To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
		# SGD has a different parameter 'momentum'.
		optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

		sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

		cs.add_hyperparameters([lr, optimizer, sgd_momentum])

		# The hyperparameter sgd_momentum will be used,if the configuration
		# contains 'SGD' as optimizer.
		cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
		cs.add_condition(cond)

		return cs


