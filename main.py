from __future__ import print_function

import os
import time
import argparse
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import config as cf
from networks import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0, 1, 2])
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--seed', type=int, default=3333)
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
use_cuda_parallel = use_cuda and len(args.gpu_ids) > 1
device = f"cuda:{args.gpu_ids[0]}" if use_cuda and args.gpu_ids else "cpu"

best_acc = 0
start_epoch, num_epochs = cf.start_epoch, cf.num_epochs

print(f"pid     : {os.getpid()}")
print(f"device  : {device}")
print(f"use_cuda: {use_cuda}")
print(f"use_cuda_parallel: {use_cuda_parallel}")
print(f"torch.manual_seed({args.seed})")
torch.manual_seed(args.seed)  # Set the random seed manually for reproducibility.

# Data Preparation
def get_data_loader():
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ]) # mean-std transformation

    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    if args.dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=t_train)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=t_test)
        num_classes = 10
    elif args.dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=t_train)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=t_test)
        num_classes = 100
    else:
        raise Exception(f"Unknown args.dataset: {args.dataset}")

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=cf.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes

def get_model(num_classes):
    if args.net_type == "lenet":
        model = LeNet(num_classes)
    elif args.net_type == 'vggnet':
        model = VGG(args.depth, num_classes)
    elif args.net_type == 'resnet':
        model = ResNet(args.depth, num_classes)
    elif args.net_type == 'wide-resnet':
        model = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)
    return model

def get_model_file_name():
    if args.net_type == 'lenet':
        file_name = 'lenet'
    elif args.net_type == 'vggnet':
        file_name = 'vgg-'+str(args.depth)
    elif args.net_type == 'resnet':
        file_name = 'resnet-'+str(args.depth)
    elif args.net_type == 'wide-resnet':
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return file_name

# Test only option
def test_only(testloader):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    file_name = get_model_file_name()
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    model = checkpoint['net']

    if use_cuda:
        model.to(device)
        if use_cuda_parallel:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        cudnn.benchmark = True

    model.eval()
    model.training = False
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" % acc)

# Model
def setup_model(num_classes):
    file_name = get_model_file_name()
    if args.resume:
        # Load checkpoint
        print('| Resuming from checkpoint...')
        assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
        model = checkpoint['net']
        global best_acc, start_epoch
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('| Building net type [' + args.net_type + ']...')
        model = get_model(num_classes)
        model.apply(conv_init)

    if use_cuda:
        model.to(device)
        if use_cuda_parallel:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        cudnn.benchmark = True
    return model, file_name

# Training
def train(epoch, model, trainloader, criterion):
    model.train()
    model.training = True
    train_loss = 0
    correct = 0
    total = 0
    lr = cf.learning_rate(args.lr, epoch)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, lr))
    b_cnt = len(trainloader)
    for b_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device) # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, b_idx+1, b_cnt, loss.item(), 100.*correct/total))
        sys.stdout.flush()

def test(epoch, model, testloader, criterion, file_name):
    global best_acc
    model.eval()
    model.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.item(), acc))

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % acc)
            state = {
                    'net': model.module if use_cuda_parallel else model,
                    'acc': acc,
                    'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'+args.dataset+os.sep
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, save_point+file_name+'.t7')
            best_acc = acc

def main():
    print('\n[Phase 1] : Data Preparation')
    trainloader, testloader, num_classes = get_data_loader()

    if args.testOnly:
        test_only(testloader)
        return

    print('\n[Phase 2] : Model setup')
    model, file_name = setup_model(num_classes)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))

    criterion = nn.CrossEntropyLoss()
    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch+num_epochs):
        start_time = time.time()

        train(epoch, model, trainloader, criterion)
        test(epoch, model, testloader, criterion, file_name)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d'  % (cf.get_hms(elapsed_time)))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % best_acc)
# main()

if __name__ == '__main__':
    main()
