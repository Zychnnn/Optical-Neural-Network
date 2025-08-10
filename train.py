import os
import csv
import random
import pathlib
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import onn

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(args):
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
    val_dataset = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    model = onn.Net().cuda()

    if args.whether_load_model:
        model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) + args.model_name))
        print(f'Model "{args.model_save_path}{args.start_epoch}{args.model_name}" loaded.')
    else:
        if os.path.exists(args.result_record_path):
            os.remove(args.result_record_path)
        with open(args.result_record_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train_Loss', "Train_Acc", 'Val_Loss', "Val_Acc", "LR"])

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.num_epochs):
        log = [epoch]
        model.train()

        train_len = 0.0
        train_running_counter = 0.0
        train_running_loss = 0.0

        tk0 = tqdm(train_dataloader, ncols=100, total=len(train_dataloader))
        for _, (train_images, train_labels) in enumerate(tk0):
            train_images = train_images.cuda()
            train_labels = train_labels.cuda()

            # Padding to match ONN input size
            train_images = F.pad(train_images, pad=(86, 86, 86, 86))
            train_images = torch.squeeze(torch.cat(
                (train_images.unsqueeze(-1), torch.zeros_like(train_images.unsqueeze(-1))),
                dim=-1), dim=1)

            train_outputs = model(train_images)
            train_loss_ = criterion(train_outputs, train_labels)
            train_counter_ = torch.eq(train_labels, torch.argmax(train_outputs, dim=1)).float().sum()

            optimizer.zero_grad()
            train_loss_.backward()
            optimizer.step()

            train_len += len(train_labels)
            train_running_loss += train_loss_.item()
            train_running_counter += train_counter_

            train_loss = train_running_loss / train_len
            train_accuracy = train_running_counter / train_len

            tk0.set_description_str(f'Epoch {epoch}/{args.start_epoch + args.num_epochs}')
            tk0.set_postfix({'Train_Loss': f'{train_loss:.5f}', 'Train_Accuracy': f'{train_accuracy:.5f}'})

        log.append(train_loss)
        log.append(train_accuracy)

        # Validation
        with torch.no_grad():
            model.eval()
            val_len = 0.0
            val_running_counter = 0.0
            val_running_loss = 0.0

            tk1 = tqdm(val_dataloader, ncols=100, total=len(val_dataloader))
            for _, (val_images, val_labels) in enumerate(tk1):
                val_images = val_images.cuda()
                val_labels = val_labels.cuda()

                val_images = F.pad(val_images, pad=(86, 86, 86, 86))
                val_images = torch.squeeze(torch.cat(
                    (val_images.unsqueeze(-1), torch.zeros_like(val_images.unsqueeze(-1))),
                    dim=-1), dim=1)

                val_outputs = model(val_images)
                val_loss_ = criterion(val_outputs, val_labels)
                val_counter_ = torch.eq(val_labels, torch.argmax(val_outputs, dim=1)).float().sum()

                val_len += len(val_labels)
                val_running_loss += val_loss_.item()
                val_running_counter += val_counter_

                val_loss = val_running_loss / val_len
                val_accuracy = val_running_counter / val_len

                tk1.set_description_str(f'Epoch {epoch}/{args.start_epoch + args.num_epochs}')
                tk1.set_postfix({'Val_Loss': f'{val_loss:.5f}', 'Val_Accuracy': f'{val_accuracy:.5f}'})

            log.append(val_loss)
            log.append(val_accuracy)

        torch.save(model.state_dict(), args.model_save_path + str(epoch) + args.model_name)
        print(f'Model "{args.model_save_path}{epoch}{args.model_name}" saved.')

        with open(args.result_record_path, 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--whether-load-model', type=bool, default=False, help="Load existing model to continue training")
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch')
    parser.add_argument('--model-name', type=str, default='_model.pth')
    parser.add_argument('--model-save-path', type=str, default="./saved_model/")
    parser.add_argument('--result-record-path', type=pathlib.Path, default="./result.csv", help="CSV result record path")

    torch.backends.cudnn.benchmark = True
    args_ = parser.parse_args()

    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)

    main(args_)
