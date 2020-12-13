import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging

logging.basicConfig(level=logging.DEBUG)


class PyTorchWorker(Worker):
    def __init__(self, batch_size=128, num_workers=4, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size

        # Caltech101 Dataset  --> take only a subset
        train_dataset = None  # train dataset torchvision
        test_dataset = None  # test dataset
        # --- Subset of the train dataset
        train_sampler = None
        validation_sampler = None

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=num_workers)
        self.validation_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                             batch_size=self.batch_size,
                                                             sampler=validation_sampler,
                                                             num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=self.batch_size,
                                                       num_workers=num_workers,
                                                       shuffle=False)

    def compute(self, model, config, budget, working_directory, *args, **kwargs):
        device = 'cuda'
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=config['lr'],
                                        momentum=config['sgd_momentum'])

        for epoch in range(int(budget)):
            loss = 0
            model.train()
            for i, (x, y) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
        train_accuracy = self.evaluate_accuracy(model, self.train_loader)
        val_accuracy = self.evaluate_accuracy(model, self.validation_loader)
        test_accuracy = self.evaluate_accuracy(model, self.test_loader)

        return {
            'loss': val_accuracy,  # note in the example it was 1-val_accuracy
            'info': {
                'test_acc': test_accuracy,
                'train_acc': train_accuracy,
                'val_acc': val_accuracy,
            }
        }

    def evaluate_accuracy(self, model, data_loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in data_loader:
                output = model(x)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
        accuracy = correct / len(data_loader.sampler)
        return accuracy

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9,
                                                      log=False)

        cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        return cs
