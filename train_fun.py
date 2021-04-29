import math
from tqdm import tqdm
import os
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import yaml


class TrainUtils:
    def __init__(self, model, train_loader, optimizer, args, args_dict, memory_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.args = args
        self.memory_loader = memory_loader
        self.test_loader = test_loader
        self.writer = SummaryWriter() #auto creates runs/Apr07_18-01-17_d1007 DIR
        self.path = os.path.join(self.args.results_dir,self.writer.log_dir)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with open(os.path.join(self.path,'config.yml'), 'w') as file:
            documents = yaml.dump(args_dict, file)
        
        logging.basicConfig(filename=os.path.join(self.path, 'training.log'), level=logging.DEBUG)
        print("\n\n directory: {}".format(self.path))
    
    def train_one_epoch(self, net , data_loader, train_optimizer, epoch, args):
        # train for one epoch
        net.train()
        #self.model.train()
        self.adjust_learning_rate(train_optimizer, epoch, args)
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
        
        for images, _ in train_bar:
            im_1 = images[0]
            im_2 = images[1]
            im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

            loss, logits, labels = net(im_1, im_2)
            if(self.n_iter % 50 == 0):
                self.top1, self.top5 = self.accuracy(logits, labels, topk=(1, 5))
            self.n_iter +=1
            # self.writer.add_scalar('loss', loss, global_step=n_iter)
            # self.writer.add_scalar('acc/top1', top1[0], global_step=epoch)
            # self.writer.add_scalar('acc/top5', top5[0], global_step=epoch)
            # self.writer.add_scalar('learning_rate', train_optimizer.param_groups[0]['lr'], global_step=epoch)
            
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size
            train_bar.set_description(
                'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'],
                                                                        total_loss / total_num))
            
        return total_loss / total_num

# lr scheduler for training
    def adjust_learning_rate(self, optimizer, epoch, args):
        """Decay the learning rate based on schedule"""
        lr = args.lr
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def accuracy(self, output, target, topk=(1,)):
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
    # test using a knn monitor
    def test(self,net, memory_data_loader, test_data_loader, epoch, args):
        net.eval()
        classes = len(memory_data_loader.dataset.classes)
        total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
                feature = net(data.cuda(non_blocking=True))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            if(args.dataset == 'stl10'):
                feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)
            else:
                feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(test_data_loader)
            for data, target in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = net(data)
                feature = F.normalize(feature, dim=1)
                
                pred_labels = self.knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

        return total_top1 / total_num * 100

    # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
    # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        print(sim_labels.view(-1,1))
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

    def train(self, epoch_start=1):
        # training loop
        
        epoch_start = 1
        self.n_iter= 0

        if(self.args.resume is ''):
            logging.info(f"Start MoCo training for {self.args.epochs} epochs.")

        for epoch in range(epoch_start, self.args.epochs + 1):
            train_loss = self.train_one_epoch(self.model, self.train_loader, self.optimizer, epoch, self.args)
            
            logging.info("Epoch: {}\ttrain_loss: {:.3f}\tAcc@1: {:.3f}\tAcc@5: {:.3f}".format(epoch,train_loss,self.top1[0], self.top5[0]))
            torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer' : self.optimizer.state_dict(),}, os.path.join(self.args.results_dir, self.writer.log_dir,'model.pth'))
        
        logging.info(f"Model, metadata and training log has been saved at {self.path}.")

    def knn_train(self, epoch_start=1):
        # training loop
        
        
        self.n_iter= 0

        if(self.args.resume is ''):
            logging.info(f"Start MoCo training for {self.args.epochs} epochs. knn testing")
    
        for epoch in range(epoch_start, self.args.epochs + 1):
            train_loss = self.train_one_epoch(self.model, self.train_loader, self.optimizer, epoch, self.args)
            test_acc_1 = self.test(self.model.encoder_q, self.memory_loader, self.test_loader, epoch, self.args)
            logging.info("Epoch: {}\ttrain_loss: {:.3f}\ttest_Acc@1: {:.3f}".format(epoch,train_loss,test_acc_1))
            torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer' : self.optimizer.state_dict(),}, os.path.join(self.args.results_dir, self.writer.log_dir,'model.pth'))
        
        logging.info(f"Model, metadata and training log has been saved at {self.path}.")