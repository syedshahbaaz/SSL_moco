import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
import logging
from moco_wrapper import ModelMoCo
import sys

parser = argparse.ArgumentParser(description='PyTorch MoCo Linear Eval')
parser.add_argument('-pt-ssl','--pre-train-ssl', action='store_true', \
    help='What backend to use for pretraining. Boolean. \
        Default pt_ssl=False, i.e, ImageNet pretrained network. ')
parser.add_argument('--model-dir', default='', type=str, metavar='PATH', help='path to directory where pretrained model is saved')
parser.add_argument('--epochs', '-e', default=100, type=int, metavar='N', help='number of epochs')
parser.add_argument('--dataset-ft', type=str, help='name of the dataset to fine tune the pretrained model on')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.STL10('./data', split='train', download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.STL10('./data', split='test', download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR10('./data', train=True, download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.CIFAR10('./data', train=False, download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def main():
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #update args.device
    print("Using device:", device) 
    logging.basicConfig(filename=os.path.join(args.model_dir, 'linear_eval.log'), level=logging.DEBUG)
    
    if not args.pre_train_ssl: #use ImageNet pretrained network
        pass
    else: # use SSL pre_trained network
        with open(os.path.join(args.model_dir ,'config.yml'), 'r') as file: #should be run_dir+config.yml. run_dir is user input
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        if config['arch'] == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
        elif config['arch'] == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)
        # for k in model.state_dict():
        #     print(k)
        checkpoint = torch.load(os.path.join(args.model_dir, 'model.pth'))
        state_dict = checkpoint['state_dict']
        # for k in state_dict:
        #     print(k)
        # sys.exit()
        for k in list(state_dict.keys()):
                # if(k == 'encoder_q.net.0.weight'):
                #     state_dict['conv1.weight']= state_dict[k]
                # elif(k=='encoder_q.net.1.weight'):
                #     state_dict['bn1.weight']= state_dict[k]
                # elif(k=='encoder_q.net.1.bias'):
                #     state_dict['bn1.bias']= state_dict[k]
                # elif(k=='encoder_q.net.1.running_mean'):
                #     state_dict['bn1.running_mean']= state_dict[k]
                # elif(k=='encoder_q.net.1.running_var'):
                #     state_dict['bn1.running_var']= state_dict[k]
                # elif(k=='encoder_q.net.1.num_batches_tracked'):
                #     state_dict['bn1.num_batches_tracked']= state_dict[k]
                if k.startswith('encoder_q.net.'):
                    # print(k, k[len("encoder_q.net."):])
                    # print('')
                    #print(state_dict[k])
                    
                    if k.startswith('encoder_q.net.'):
                         if(k.startswith('encoder_q.net.fc') or k.startswith('encoder_q.net.layer') or k.startswith('encoder_q.net.conv') or k.startswith('encoder_q.net.b')): 
                            state_dict[k[len("encoder_q.net."):]] = state_dict[k] # remove prefix
                         else:
                            name = k[len("encoder_q.net.")+1:]
                        #  print(k[len("encoder_q.net."):len("encoder_q.net.")+1])
                            num = int(k[len("encoder_q.net."):len("encoder_q.net.")+1])
                            if(num == 3): num = 1
                            elif(num==4): num =2
                            elif(num==5): num =3
                            elif(num==6): num=4
                            num = str(num)
                            state_dict['layer' + num + str(name)] = state_dict[k]
                                 
                             
                    
                del state_dict[k]
        # for k in state_dict.keys():
        #     print(k)
        # sys.exit()
        #sys.exit()
        log = model.load_state_dict(state_dict, strict=False)
        
        if(args.dataset_ft):
            logging.info(f'Fine tune on Dataset: {args.dataset_ft}')
            #print(f'Fine tune on Dataset: {args.dataset_ft}')
            if(args.dataset_ft == 'cifar10'):
                train_loader, test_loader = get_cifar10_data_loaders(download=True)
            elif(args.dataset_ft =='stl10'):
                train_loader, test_loader = get_stl10_data_loaders(download=True)
        else:
            logging.info(f'Fine tune on Dataset: {config["dataset"]}')
            print(f'Fine tune on Dataset: {config["dataset"]}')
            if config['dataset'] == 'cifar10':
                train_loader, test_loader = get_cifar10_data_loaders(download=True)
            elif config['dataset'] == 'stl10':
                train_loader, test_loader = get_stl10_data_loaders(download=True)
        
        

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = torch.nn.CrossEntropyLoss().to(device)


    epochs = args.epochs
    for epoch in range(1, epochs+1):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0

        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        logging.info("Epoch {}\tTrain Acc@1 {:.2f}\tTest Acc@1: {:.2f}\tTest Acc@5: {:.2f}".format(epoch,top1_train_accuracy.item(),top1_accuracy.item(),top5_accuracy.item()))
        print("Epoch {}\tTrain Acc@1 {:.2f}\tTest Acc@1: {:.2f}\tTest Acc@5: {:.2f}".format(epoch,top1_train_accuracy.item(),top1_accuracy.item(),top5_accuracy.item()))


if __name__ == "__main__":
    main()
