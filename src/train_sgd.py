from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np

from model import _netlocalD,_netG
import utils



parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.998,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'streetview':
    transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch=0

netG = _netG(opt)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
print(netG)


netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']
print(netD)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize//2, opt.imageSize//2)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
    real_center = real_center.cuda()


input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)


real_center = Variable(real_center)

# setup optimizer
optimizerD = optim.SGD(netD.parameters(), lr=opt.lr, momentum = 0.9)
optimizerG = optim.SGD(netD.parameters(), lr=opt.lr, momentum = 0.9)
l2_list = []
adv_list = []
total_list =[]
for epoch in range(resume_epoch,opt.niter):
    print(epoch)
    mean_l2_list = []
    mean_adv_list = []
    mean_total_list =[]
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        real_center_cpu = real_cpu[:,:,int(opt.imageSize//4):int(opt.imageSize//4)+int(opt.imageSize//2),int(opt.imageSize//4):int(opt.imageSize//4)+int(opt.imageSize//2)]
        batch_size = real_cpu.size(0)
        with torch.no_grad():
            input_real.resize_(real_cpu.size()).copy_(real_cpu)
            input_cropped.resize_(real_cpu.size()).copy_(real_cpu)
       	    real_center.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        input_cropped.data[:,0,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*117.0/255.0 - 1.0
        input_cropped.data[:,1,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*104.0/255.0 - 1.0
        input_cropped.data[:,2,int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred),int(opt.imageSize/4+opt.overlapPred):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred)] = 2*123.0/255.0 - 1.0

        # train with real
        netD.zero_grad()
        with torch.no_grad():
            label.resize_(batch_size).fill_(real_label)

        output = netD(real_center)
        output = output.reshape(label.size())
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        # noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)
        fake = netG(input_cropped)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        output = output.reshape(label.size())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        output = output.reshape(label.size())
        errG_D = criterion(output, label)
        # errG_D.backward(retain_variables=True)

        # errG_l2 = criterionMSE(fake,real_center)
        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(wtl2*overlapL2Weight)
        wtl2Matrix.data[:,:,int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred),int(opt.overlapPred):int(opt.imageSize/2 - opt.overlapPred)] = wtl2
        
        errG_l2 = (fake-real_center).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()

        errG = (1-wtl2) * errG_D + wtl2 * errG_l2

        mean_l2_list.append(errG_l2.item())
        mean_adv_list.append(errG_D.item())
        mean_total_list.append(errG.item())

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()

        print(epoch, opt.niter, i, len(dataloader),errD.item(), errG_D.item(),errG_l2.item(), D_x,D_G_z1)
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    'result/train/real/real_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(input_cropped.data,
                    'result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            recon_image = input_cropped.clone()
            recon_image.data[:,:,int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)] = fake.data
            vutils.save_image(recon_image.data,
                    'result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))


    # do checkpointing
    l2_list.append(sum(mean_l2_list)/len(mean_l2_list))
    adv_list.append(sum(mean_adv_list)/len(mean_adv_list))
    total_list.append(sum(mean_total_list)/len(mean_total_list))
    # torch.save({'epoch':epoch+1,
    #             'state_dict':netG.state_dict()},
    #             'model/netG_streetview.pth' )
    torch.save({'epoch':epoch+1,
                'state_dict':netD.state_dict()},
                'model/netlocalD.pth' )
    if((epoch+1)%20==0):
        torch.save({'epoch':epoch+1,'state_dict':netG.state_dict()},'model/netG_streetview'+str(epoch+1)+'.pth')
        

total_list = np.array(total_list)
adv_list = np.array(adv_list)
l2_list = np.array(l2_list)
np.save("total_loss.npy", total_list)
np.save("adv_loss.npy", adv_list)
np.save("l2_loss.npy", l2_list)

