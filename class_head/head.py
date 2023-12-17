import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import no_grad

from torch.utils.data import DataLoader

from HCLIP import create_model
from dataset import load_dataset, CustomCollator

import numpy as np

import HCLIP
from main import args


class BinaryHeadNetwork(nn.Module):
    def __init__(self):
        super(BinaryHeadNetwork, self).__init__()

        model = HCLIP.CLIPClassifier.load_from_checkpoint(
            checkpoint_path='checkpoints/elated-waterfall-2-epoch=12.ckpt',
            args=args, strict=False)

        self.layers = [
            nn.Linear(768, 200),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(200, 1)
        ]

        self.seq = nn.Sequential(*self.layers)
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.seq(x)
        return F.log_softmax(x)


def train(model, n_epochs, train_dataloader, test_dataloader):
    losses = []
    accuracies = []
    for epoch in range(n_epochs):

        # обучение
        for x_train, y_train in train_dataloader:
            y_pred = model.forward(x_train)
            # loss = F.cross_entropy(y_pred, y_train)
            loss = model.cross_entropy_loss(y_pred, y_train)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

        val_loss = []
        val_accuracy = []
        # валидация
        if epoch % 2 == 0:
            with no_grad():
                for x_val, y_val in test_dataloader:
                    y_pred = model.foward(x_val)
                    loss = model.cross_entropy_loss(y_pred, y_val)
                    val_loss.append(loss.numpy())
                    val_accuracy.extend((torch.argmax(y_pred, dim=-1) == y_val).numpy().tolist())

        losses.append(np.mean(val_loss))
        accuracies.append(np.mean(val_accuracy))

        print(f'Epoch: {epoch}, loss: {np.mean(val_loss)}, accuracy: {np.mean(val_accuracy)}')


collator = CustomCollator(args)

dataset_train = load_dataset(args=args, split='train')
dataset_val = load_dataset(args=args, split='dev')

dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=collator)
dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=4, collate_fn=collator)
