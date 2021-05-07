#!/usr/bin/env python
# coding: utf-8

import logging
import os
import random
import sys
import time
import warnings
from time import sleep

import albumentations as A
import cv2
import numpy as np
import timm
import torch
from albumentations import (Compose, Cutout, HorizontalFlip, HueSaturationValue, IAAAdditiveGaussianNoise, MedianBlur,
                            Normalize, RandomBrightnessContrast, Resize)
from albumentations.pytorch import ToTensorV2
from catalyst.data.sampler import BalanceClassSampler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from ranger import Ranger  # https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer


def init_script():
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.INFO)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_test_valid_dataloaders(data_path, test_data_path, seed, image_size, batch_size):
    """
    Utility function for the model.
    """
    def build_data(data_path):
        content_list = []
        labels_list = []

        for image in tqdm(os.listdir(data_path)):
            if ".jpg" in image:
                content = cv2.imread(data_path + image)
                content_list.append(content)
            elif ".txt" in image:
                with open(data_path + image, "r") as f:
                    labels = f.read()
                labels = np.array(labels.split(" "), dtype=int)
                labels[0] = 0 if labels[0] == 1 else 1
                labels = np.roll(labels, -1)
                labels_list.append(labels)
        data = np.array([list(a) for a in zip(content_list, labels_list)])

        return data

    train_data = build_data(data_path=data_path)
    test_data = build_data(data_path=test_data_path)

    train_data, valid_data = train_test_split(train_data, shuffle=True, test_size=0.1, random_state=seed)

    train_clf_labels = [a[-1] for a in train_data[:, 1]]

    transform = Compose(
        [
            Resize(width=image_size, height=image_size),
            HorizontalFlip(p=0.4),
            # ShiftScaleRotate(p=0.3),
            MedianBlur(blur_limit=7, always_apply=False, p=0.3),
            IAAAdditiveGaussianNoise(scale=(0, 0.15 * 255), p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.4),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # in this implementation imagenet normalization is used
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            Cutout(p=0.4),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc"),
    )

    test_transform = Compose(
        [
            # only resize and normalization is used for testing
            # no TTA is implemented in this solution
            Resize(width=image_size, height=image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc"),
    )

    train_dataset = Dataset(train_data, transforms=transform)
    valid_dataset = Dataset(valid_data, transforms=transform)
    test_dataset = Dataset(test_data, transforms=test_transform)

    train_dataloader = DataLoader(
        train_dataset,
        # balanced sampler is used to minimize harmful effects of dataset not being fully balanced
        sampler=BalanceClassSampler(labels=train_clf_labels, mode="upsampling"),
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=1)
    valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=batch_size)

    return train_dataloader, test_dataloader, valid_dataloader


class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class that augments data during training.
    Imagenet normalization is used for efficientnet fine-tuning.
    """

    def __init__(self, dataset, transforms=None):
        super().__init__()
        self.data = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index][0], self.data[index][1]
        if image.shape[-1] == 4:
            # removing alpha channel if present
            image = image[..., :3]
        if len(image.shape) == 2:
            # converting single channel image to 3 channel for possible greyscales
            image = np.stack((image,) * 3, axis=-1)
        # fast BGR to RGB
        image = image[:, :, ::-1]

        if self.transforms is not None:
            # bboxes=[labels] labels are wrapped into list for correct augmentation
            transformed = self.transforms(image=image, bboxes=[label])

            image = transformed["image"]
            label = torch.Tensor(transformed["bboxes"])[0]

        return image, label


class EFN_Classifier(nn.Module):
    """
    NN itself as a root logic of our model.
    """

    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 2048)

        self.fc1 = nn.Linear(2048, 1536)
        self.fc2 = nn.Linear(2048, 512)

        self.dropout = nn.Dropout(0.35)

        self.fc_bbox = nn.Linear(1536, 4)
        self.fc_clf = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)

        x_bbox = self.fc1(x)
        x_bbox = self.dropout(x_bbox)

        x = self.fc2(x)
        x = self.dropout(x)

        x_bbox = self.fc_bbox(x_bbox)
        x = self.fc_clf(x)

        return x, x_bbox


class Brain(object):
    """
    High-level model logic and tuning nuggets encapsulated.

    Based on efficientNet: https://arxiv.org/abs/1905.11946
    fine tuning the efficientnet for classification and object detection
    in this implementation, no weights are frozen
    ideally, batchnorm layers can be frozen for marginal training speed increase
    """

    def __init__(self, gradient_accum_steps=5, lr=0.0005, epochs=100, n_class=2, lmb=30):
        self.device = self.set_cuda_device()
        self.net = EFN_Classifier("tf_efficientnet_b1_ns", n_class).to(self.device)
        self.loss_function = nn.MSELoss()
        self.clf_loss_function = nn.CrossEntropyLoss()
        self.optimizer = Ranger(self.net.parameters(), lr=lr, weight_decay=0.999, betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingLR(self.optimizer, epochs * 0.5, lr * 0.0001)
        self.scheduler.last_epoch = epochs
        self.scaler = GradScaler()
        self.epochs = epochs
        self.gradient_accum_steps = gradient_accum_steps
        self.lmb = lmb

    @staticmethod
    def set_cuda_device():
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logging.info(f"Running on {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logging.info("Running on a CPU")
        return device

    def run_training_loop(self, train_dataloader, valid_dataloader, model_filename):
        best_loss = float("inf")

        for epoch in range(self.epochs):
            if epoch != 0 and epoch > 0.5 * self.epochs:  # cosine anneal the last 50% of epochs
                self.scheduler.step()
            logging.info(f"Epoch {epoch+1}")

            logging.info("Training")
            train_losses, train_accuracies, train_miou = self.forward_pass(train_dataloader, train=True)

            logging.info("Validating")
            val_losses, val_accuracies, val_miou = self.forward_pass(valid_dataloader)

            logging.info(
                f"Training accuracy: {sum(train_accuracies)/len(train_accuracies):.2f}"
                f" | Training loss: {sum(train_losses)/len(train_losses):.2f}"
                f" | Training mIoU: {sum(train_miou)/len(train_miou):.2f}"
            )
            logging.info(
                f"Validation accuracy: {sum(val_accuracies)/len(val_accuracies):.2f}"
                f" | Validation loss: {sum(val_losses)/len(val_losses):.2f}"
                f" | Validation mIoU: {sum(val_miou)/len(val_miou):.2f}"
            )

            epoch_val_loss = sum(val_losses) / len(val_losses)

            if best_loss > epoch_val_loss:
                best_loss = epoch_val_loss
                torch.save(self.net.state_dict(), model_filename)
                logging.info(f"Saving with loss of {epoch_val_loss}, improved over previous {best_loss}")

    def bbox_iou(self, true_boxes, pred_boxes):
        iou_list = []
        for true_box, pred_box in zip(true_boxes, pred_boxes):

            x_left = max(true_box[0], pred_box[0]).item()
            y_top = max(true_box[1], pred_box[1]).item()

            x_right = min(true_box[2], pred_box[2]).item()
            y_bottom = min(true_box[3], pred_box[3]).item()

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            overlap = (x_right - x_left) * (y_bottom - y_top)

            true_box_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
            pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            iou = overlap / float(true_box_area + pred_box_area - overlap)
            iou_list.append(iou)

        iou = torch.tensor(iou)
        iou = torch.mean(iou)

        return iou

    def draw_boxes(self, images, bboxes, labels):
        label_dict = {0: "Cat", 1: "Dog"}

        for batch in zip(images, bboxes, labels):
            cv2.destroyAllWindows()
            image, bbox, label = batch[0].cpu().numpy(), batch[1].cpu().numpy(), torch.argmax(batch[2]).cpu().item()
            image = np.rollaxis(image, 0, 3)
            image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype("uint8")
            image = cv2.UMat(image)

            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness=2)

            cv2.putText(
                image,
                f"{label_dict[label]}",
                (bbox[1], bbox[3]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("test", image)
            cv2.waitKey(1)
            sleep(1)
            cv2.destroyAllWindows()

    def forward_pass(self, dataloader, draw=False, train=False):
        def get_loss(inputs, bbox_labels, clf_labels):
            label_outputs, bbox_outputs = self.net(inputs)
            bbox_loss = self.loss_function(bbox_outputs, bbox_labels)
            clf_loss = self.clf_loss_function(label_outputs, clf_labels)
            loss = torch.mean(bbox_loss + clf_loss * self.lmb)
            return loss, label_outputs, bbox_outputs

        if train:
            self.net.train()
        else:
            self.net.eval()

        losses = []
        accuracies = []
        miou = []

        for step, batch in enumerate(dataloader):
            inputs = batch[0].to(self.device).float()
            labels = batch[1].to(self.device).float()

            # splitting labels for separate loss calculation
            bbox_labels = labels[:, :4]
            clf_labels = labels[:, 4:].long()
            clf_labels = clf_labels[:, 0]

            with autocast():
                if train:
                    loss, label_outputs, bbox_outputs = get_loss(inputs, bbox_labels, clf_labels)
                    self.scaler.scale(loss).backward()
                else:
                    with torch.no_grad():
                        loss, label_outputs, bbox_outputs = get_loss(inputs, bbox_labels, clf_labels)
                    if draw:
                        self.draw_boxes(inputs, bbox_outputs, label_outputs)

            matches = [torch.argmax(i) == j for i, j in zip(label_outputs, clf_labels)]
            acc = matches.count(True) / len(matches)
            iou = self.bbox_iou(bbox_labels, bbox_outputs)

            miou.append(iou)
            losses.append(loss)
            accuracies.append(acc)

            if train and (step + 1) % self.gradient_accum_steps == 0:
                # gradient accumulation to train with bigger effective batch size
                # with less memory use
                # fp16 is used to speed up training and reduce memory consumption
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                logging.info(
                    f"Step {step} of {len(dataloader)},\t"
                    f"Accuracy: {sum(accuracies)/len(accuracies):.2f},\t"
                    f"mIoU: {sum(miou)/len(miou):.2f},\t"
                    f"Loss: {sum(losses)/len(losses):.2f}"
                )

        return losses, accuracies, miou


def main():

    main_ret_status = 0

    SEED = 55555
    init_script()
    seed_everything(seed=SEED)

    train_dataloader, test_dataloader, valid_dataloader = get_train_test_valid_dataloaders(
        data_path=r"./data/train/", test_data_path=r"./data/valid/", seed=SEED, image_size=300, batch_size=16
    )

    # train the model
    model_filename = "best.pth"
    brain = Brain(gradient_accum_steps=5, lr=0.0005, epochs=100, n_class=2, lmb=30)
    brain.run_training_loop(
        train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, model_filename=model_filename
    )
    # test the freshbaked model loading
    brain.net.load_state_dict(torch.load(model_filename))

    # run test pass to measure our model performance
    start = time.time()
    test_losses, test_accuracies, test_miou = brain.forward_pass(dataloader=test_dataloader, draw=False, train=False)
    total_time = time.time() - start

    logging.info(f"Average inference time is: {total_time/len(test_dataloader):.3f}")
    logging.info(
        f"Test accuracy: {sum(test_accuracies)/len(test_accuracies):.2f}"
        f" | Test loss: {sum(test_losses)/len(test_losses):.2f} | Test mIoU: {sum(test_miou)/len(test_miou):.2f}"
    )

    return main_ret_status


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
