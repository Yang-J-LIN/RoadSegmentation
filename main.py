import os
import argparse

import numpy as np

import torch

from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from dataset import RoadSegmentationDataset
from unet import U_Net

# intialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float)
parser.add_argument("--images_dir",
                    default='training/images',
                    type=str)
parser.add_argument("--groundtruth_dir",
                    default='training/groundtruth',
                    type=str)
parser.add_argument("--cuda",
                    default=True,
                    type=bool)
parser.add_argument("--test",
                    default=1,
                    type=bool)
parser.add_argument("--load_model",
                    default='model.pt',
                    type=str)
parser.add_argument("--save_model",
                    default='model.pt',
                    type=str)
parser.add_argument("--cross_validation",
                    default=1,
                    type=int)
args = parser.parse_args()

print(args)

DEVICE = 'cuda' if args.cuda is True else 'cpu'

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def train(train_images_list, dev_image_list, images_dir, groundtruth_dir):
    # images_list = images_list[0:80]
    print(train_images_list)
    train_images_paths = [os.path.join(images_dir, i) for i in train_images_list]
    train_groundtruth_paths = [os.path.join(groundtruth_dir, i) for i in train_images_list]

    train_set = RoadSegmentationDataset(
        train_images_paths,
        train_groundtruth_paths
    )

    if args.test == 1 and args.cross_validation != 1:
        dev_images_paths = [os.path.join(images_dir, i) for i in dev_image_list]
        dev_groundtruth_paths = [os.path.join(groundtruth_dir, i) for i in dev_image_list]
        dev_set = RoadSegmentationDataset(
            dev_images_paths,
            dev_groundtruth_paths,
        )

        max_f1 = 0

    # data augmentation
    print('Train size:', len(train_set))
    train_set.augment()
    print('Train size after augmentation:', len(train_set))

    train_loader = DataLoader(train_set, batch_size=10)

    if args.test == 1 and args.cross_validation != 1:
        train_test_loader = DataLoader(train_set, batch_size=10, sampler=RandomSampler(train_set, num_samples=100, replacement=True))
        dev_loader = DataLoader(dev_set, batch_size=10)

    # load model
    model = U_Net().cuda(DEVICE)

    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))

    print('Number of parameters:', get_n_params(model))

    # define the hyperparameters
    optimizer = AdamW(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    loss_func = torch.nn.BCELoss()

    # start training
    for epoch in range(3):
        print('Training epoch:', epoch)

        # training
        model.train()
        loss_total = 0
        for inputs, truth in train_loader:
            inputs = inputs.to(DEVICE)
            truth = truth.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs).squeeze(1)
            loss = loss_func(outputs, truth)
            loss_total += loss.item()

            loss.backward()

            optimizer.step()

        # evaluation
        if args.test == 1 and args.cross_validation != 1:
            model.eval()

            # evaluate the model on the training set (randomly pick 100 samples)
            outputs = []
            dev_truths = []
            for dev_inputs, dev_truth in train_test_loader:
                dev_inputs = dev_inputs.to(DEVICE)
                dev_truth = dev_truth.to(DEVICE)
                outputs.append(model(dev_inputs).squeeze(1).detach().cpu().numpy().flatten())
                dev_truths.append(dev_truth.detach().cpu().numpy().flatten())

            outputs = np.concatenate(outputs, axis=0)
            dev_truths = np.concatenate(dev_truths, axis=0)

            train_f1 = f1_score(dev_truths.astype(int), (outputs > 0.5).astype(int))

            # evaluate the model on the test set
            outputs = []
            dev_truths = []
            for dev_inputs, dev_truth in dev_loader:
                dev_inputs = dev_inputs.to(DEVICE)
                dev_truth = dev_truth.to(DEVICE)
                outputs.append(model(dev_inputs).squeeze(1).detach().cpu().numpy().flatten())
                dev_truths.append(dev_truth.detach().cpu().numpy().flatten())

            outputs = np.concatenate(outputs, axis=0)
            dev_truths = np.concatenate(dev_truths, axis=0)

            dev_f1 = f1_score(dev_truths.astype(int), (outputs > 0.5).astype(int))

            # early stop
            if dev_f1 > max_f1:
                max_f1 = dev_f1
                if args.save_model is not None:
                    torch.save(model.state_dict(), args.save_model)
        else:
            if args.save_model is not None:
                torch.save(model.state_dict(), args.save_model)
        print('Training loss:', loss_total / len(train_loader), train_f1, dev_f1)

        scheduler.step()

    if args.test == 1 and args.cross_validation != 1:
        return max_f1


def main():
    # load the dataset
    images_dir = args.images_dir
    groundtruth_dir = args.groundtruth_dir

    images_list = os.listdir(images_dir)
    images_list.sort()

    f1_list = []
    
    if args.cross_validation != 1:
        kf = KFold(n_splits=args.cross_validation)
        for train_index, test_index in kf.split(images_list):
            print(train_index, test_index)
            f1_list.append(train(
                np.array(images_list)[train_index],
                np.array(images_list)[test_index],
                images_dir,
                groundtruth_dir
            ))
        print('F1 Score:', np.array(f1_list).mean())
    else:
        train(
            images_list,
            None,
            images_dir,
            groundtruth_dir
        )


if __name__ == '__main__':
    main()
