import glob
from torch.utils.tensorboard import SummaryWriter
import os, losses, utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
from natsort import natsorted
from models import MSDP
import random
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=True

same_seeds(24)
GPU_iden = 0
def main():
    batch_size = 1

    train_dir = 'abct_data/Train/'
    val_dir = 'abct_data/Val/'
    weights = [1, 1, 0.5]  # loss weights
    lr = 0.0001
    head_dim = 6
    num_heads = [8, 4, 2, 1, 1]

    channels=8
    save_dir = 'msdp-nh({}{}{}{}{})_hd_{}_c_{}_ncc_{}_dsc_{}_reg_{}_lr_{}_4r/'.format(*num_heads, head_dim,channels,*weights, lr)
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    f = open(os.path.join('logs/'+save_dir, 'losses and dice' + ".txt"), "a")

    epoch_start = 0
    max_epoch = 30
    img_size = (160, 160, 192)
    cont_training = False

    '''
    Initialize model
    '''
    model = MSDP(img_size, num_heads=num_heads, channels=channels, head_dim=head_dim, scale=1)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        # epoch_start = 384
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
        print(model_dir + natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([#trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    val_composed = transforms.Compose([trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.InferDatasetS2S(natsorted(glob.glob(train_dir + '*.pkl')), transforms=train_composed)
    val_set = datasets.InferDatasetS2S(natsorted(glob.glob(val_dir + '*.pkl')), transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    Lncc = losses.NCC_vxm()
    Ldsc = losses.DiceLoss(num_class=5)
    Lreg = losses.Grad3d(penalty='l2')

    best_dsc = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=5)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

            moved, flow = model(x,y)
            def_seg = model.transformer[0](x_seg_oh.float(), flow.float())

            loss_ncc = Lncc(moved, y) * weights[0]
            loss_dsc = Ldsc(def_seg, y_seg.long()) * weights[1]
            loss_reg = Lreg(flow) * weights[2]

            loss = loss_ncc + loss_dsc + loss_reg

            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, DSC: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_ncc.item(), loss_reg.item(), loss_dsc.item()))

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                output = model(x,y)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(epoch, ':',eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        print(eval_dsc.avg, file=f)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        loss_all.reset()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''

    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()