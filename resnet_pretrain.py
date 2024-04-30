from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10,CIFAR100
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet_new import ResNet18,ResNet34
# from advertorch.attacks import LinfPGDAttack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR100')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="cln", help="cln | adv")
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=100, type=int)
    parser.add_argument('--log_interval', default=200, type=int)
    parser.add_argument('--data-dir', default='./data/cifar100',
                    help='path to dataset CIFAR-100')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "cln":
        flag_advtrain = False
        nb_epoch = 10
        model_filename = "resnet18_cifar100_cln.pth"
    elif args.mode == "adv":
        flag_advtrain = True
        nb_epoch = 50
        model_filename = "tiny_linf_adv_RN34.pth"
    else:
        raise


     # setup data loader
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    """
    LOAD DATASETS
    """
    
    
    trainset = CIFAR100(args.data_dir,train=True,download=False,transform=trans_train)
    train_loader = DataLoader(trainset,batch_size=args.train_batch_size,shuffle=True,drop_last=True,pin_memory=torch.cuda.is_available())
    print("训练样本个数：",len(trainset))
    testset = CIFAR100(args.data_dir,train=False,download=False,transform=trans_test)
    test_loader = DataLoader(testset,batch_size=args.test_batch_size,shuffle=True,drop_last=True,pin_memory=torch.cuda.is_available())
    print("测试样本个数：",len(testset))

    model = ResNet18(100)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    '''
    MadryLab/cifar10_challengePublic standard AT
    epsilon:the maximum allowed perturbation per pixel      -8.0/255
    k:the number of PGD iterations used by the adversary    -10  
    a: the size of the PGD adversary steps                  -2.0
    '''

    if flag_advtrain:
        None
        # adversary = LinfPGDAttack(
        #     model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8.0/255,
        #     nb_iter=10, eps_iter=0.007, rand_init=True, clip_min=0.0,
        #     clip_max=1.0, targeted=False)

    for epoch in range(nb_epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ori = data
            if flag_advtrain:
                # data = adversary.perturb(data, target)
                None

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(
                output, target, reduction='elementwise_mean')
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        model.eval()
        test_clnloss = 0
        clncorrect = 0

        if flag_advtrain:
            test_advloss = 0
            advcorrect = 0

        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)
            with torch.no_grad():
                output = model(clndata)
            test_clnloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            if flag_advtrain:
                None
                # advdata = adversary.perturb(clndata, target)
                # with torch.no_grad():
                #     output = model(advdata)
                # test_advloss += F.cross_entropy(
                #     output, target, reduction='sum').item()
                # pred = output.max(1, keepdim=True)[1]
                # advcorrect += pred.eq(target.view_as(pred)).sum().item()

        test_clnloss /= len(test_loader.dataset)
        print('\nTest set: avg cln loss: {:.4f},'
            ' cln acc: {}/{} ({:.0f}%)\n'.format(
                test_clnloss, clncorrect, len(test_loader.dataset),
                100. * clncorrect / len(test_loader.dataset)))
        if flag_advtrain:
            test_advloss /= len(test_loader.dataset)
            print('Test set: avg adv loss: {:.4f},'
                ' adv acc: {}/{} ({:.0f}%)\n'.format(
                    test_advloss, advcorrect, len(test_loader.dataset),
                    100. * advcorrect / len(test_loader.dataset)))

    torch.save(
        model.state_dict(),
        os.path.join('./', model_filename))



'''

此 Python 脚本设计用于在 CIFAR100 数据集上训练卷积神经网络 （CNN），并具有可选的对抗性训练，以增强模型对对抗性攻击的鲁棒性。以下是其关键组件和功能的细分：

导入库：
torch， ， 用于使用 PyTorch 构建和训练神经网络。torch.nntorch.optim
torchvision.datasets以及用于数据加载和预处理。torchvision.transforms
自定义使用预定义的 ResNet18 和 ResNet34 架构。models.resnet_new
注释掉的导入表明，曾考虑过对抗性训练功能（使用对抗性攻击，如 LinfPGDAttack），但目前已被禁用。advertorch.attacks
参数解析：
该脚本用于处理各种训练设置（例如，批处理大小、纪元、训练模式和数据集目录）的命令行参数。argparse
环境设置：
设置可重现性的随机种子。
配置用于训练的 GPU 设置（如果可用）。
训练模式配置：
根据参数，脚本将训练配置为正常（用于干净）或对抗（）。它还相应地设置训练周期数和模型的文件名。--modeclnadv
数据加载：
为训练和测试数据集（例如，随机裁剪、翻转）定义了数据转换。
假设 CIFAR100 数据集已在本地可用，则无需下载即可加载用于训练和测试。
模型和优化器：
初始化修改为处理 100 个输出类 （CIFAR100） 的 ResNet18 模型。
Adam 优化器用于更新网络权重。
对抗性训练占位符：
本节包括一个用于对抗性训练的占位符，它将利用对抗性攻击机制（例如，LinfPGDAttack）来生成扰动输入，旨在提高模型对此类攻击的鲁棒性。
培训和评估循环：
该模型针对指定数量的 epoch 进行训练。对于每个 epoch，它都会处理来自训练加载器的批处理，如果启用了对抗性训练，则可以选择干扰数据。
它根据干净的数据评估模型，如果启用了对抗性训练，则还会评估对抗性扰动数据。记录损失和准确性等指标。
保存模型：
训练后，模型的权重将保存到指定的文件中。
该脚本的结构具有灵活性，允许在正常和对抗性训练模式之间直接切换。但是，为了完全实现对抗性训练，相关的注释部分与“广告

'''
