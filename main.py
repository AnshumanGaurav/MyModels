import os
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from datetime import datetime
from torchvision import transforms
from Models.resnet import resnet50
from Dataset.ilsvrc2012 import get_ilsvrc2012_train_dataset
from torch.utils.data import DataLoader, DistributedSampler


def train(gpu, args):

    rank = gpu

    if args.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.gpus, rank=rank)

    torch.manual_seed(0)
    model = resnet50()
    optimizer = torch.optim.SGD(model.parameters(), 1e-1)

    epoch = 0
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    batch_size = 250

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    if args.gpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    transform = torch.nn.Sequential(
        transforms.Pad(224),
        transforms.CenterCrop((224, 224))
    )

    train_dataset = get_ilsvrc2012_train_dataset(transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.gpus, rank=gpu, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8,
                              pin_memory=True, sampler=train_sampler)

    total_step = len(train_loader)
    for epoch in range(epoch, args.epochs):
        start = datetime.now()

        for i, (images, labels) in enumerate(train_loader):

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item()
                ))

        if gpu == 0:
            print('This Epoch complete in: ' + str(datetime.now() - start))
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, args.dump_checkpoint + '_{}'.format(epoch))

    if args.gpus > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default=2, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--dump_checkpoint', default='checkpoint', type=str)
    parser.add_argument('--load_checkpoint', default=None, type=str)
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8890'

    if args.gpus > 1:
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        train(0, args)


if __name__ == '__main__':
    main()


