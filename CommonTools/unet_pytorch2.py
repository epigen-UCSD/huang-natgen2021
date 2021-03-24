import torch
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pylab as plt
import os
from torch.nn.parallel.data_parallel import DataParallel
class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        #self.up_sampling = torch.nn.functional.interpolate(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet(torch.nn.Module):
    def __init__(self,N=1):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(1, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, True)
        self.down_block7 = UNet_down_block(512, 1024, True)

        self.mid_conv1 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.mid_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm2d(16)
        self.last_conv2 = torch.nn.Conv2d(16, N, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)
        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x

def show_example(input_, mask_, prediction_):
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    input__=input_.cpu().numpy()[0,0]
    ax1.imshow(input_.cpu().numpy()[0,0])
    if mask_ is not None:
        ax2.imshow(mask_.cpu().numpy()[0,0])
        #ax3.imshow(mask_.cpu().numpy()[0,1])
    ax3.imshow(prediction_.cpu().detach().numpy()[0,0])
    plt.show()
    #f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    #ax1.imshow(prediction_.cpu().detach().numpy()[0,0])
    #ax2.imshow(prediction_.cpu().detach().numpy()[0,1])
    #ax3.imshow(prediction_.cpu().detach().numpy()[0,2])
    plt.show()

def dice_loss2(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1.0 - (((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))


def train(generate_X_Y,nepochs=50,batch_size = 1000,init_lr = 0.01,save_root='v4unet1024-',load_fl=''):
    model = DataParallel(UNet()).cuda()
    #model = UNet().cuda()
    loss_fn = torch.nn.BCELoss(reduction='elementwise_mean')
    opt = torch.optim.RMSprop(model.parameters(), lr=init_lr)
    #opt.zero_grad()
    
    epoch0 = 0
    if load_fl:
        if os.path.exists(load_fl):
            load_dic = torch.load(load_fl)
            model.module.load_state_dict(load_dic['state_dict'])
            opt.load_state_dict(load_dic['optimizer'])
            epoch0 = load_dic['epoch']-1
            print('Loaded model from: '+load_fl)
        
    
    for epoch_ in range(nepochs):
        #data_loader = [dataset[np.random.randint(len(dataset))] for i in range(batch_size)]
        #Update learning curve as we learn more and more - currently dissabled
        epoch = epoch_+epoch0
        lr = init_lr * (0.5 ** (epoch // 20))
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        pbar = tqdm(np.arange(batch_size))
        losses = []
        for idata in pbar:
            X,Y = generate_X_Y(idata)
            batch_input = Variable(torch.cuda.FloatTensor(X))
            batch_gt_mask = Variable(torch.cuda.FloatTensor(Y))

            #pass through net
            pred_mask = model(batch_input)
            spred_mask = torch.sigmoid(pred_mask)
            
            #compute loss and backpropagate
            loss = loss_fn(spred_mask, batch_gt_mask)
            #loss += dice_loss2(spred_mask, batch_gt_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_ = loss.cpu().data.numpy()
            losses.append(loss_)
            #update progress bar
            
            text = 'Loss:{:>1.5f}'.format(loss_)
            pbar.set_description(text )
        
        #show_example(batch_input, batch_gt_mask, spred_mask)
        #save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            #'state_dict': model.module.state_dict(),
            'state_dict': model.state_dict(),
            'optimizer' : opt.state_dict()}
        #torch.save(checkpoint, save_root+'{}'.format((epoch+1)%5))
        torch.save(checkpoint, save_root+'_last')
        print("Finished epoch: "+str(int(epoch)))
        print("Mean loss: "+str(np.mean(losses)))
        show_example(batch_input, batch_gt_mask, spred_mask)
def load_model(fl=None,cuda=False):
    model = UNet()
    #if cuda:
        #model = DataParallel(model).cuda()
    if fl is not None:
        load_dic = torch.load(fl)
        model.load_state_dict(load_dic['state_dict'])
        print('Loaded model from: '+fl)
    return model
def get_model(model_fl):
    modelfl = model
    model = DataParallel(unet.UNet()).cuda()
    load_fl = modelfl
    load_dic = torch.load(load_fl)
    model.load_state_dict(load_dic['state_dict'])
    return model
def apply_model(im2D,model,cuda=False):
    if type(model) is str:
        model = get_model(model)
    
    im_reshape = np.array(im2D,dtype=np.float32)[np.newaxis,np.newaxis,...]
    im_reshape = torch.FloatTensor(im_reshape)
    if cuda:
        im_reshape = im_reshape.cuda()
    pred_mask = model(im_reshape)
    return torch.sigmoid(pred_mask).cpu().detach().numpy()[0,0]