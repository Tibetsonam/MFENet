
import argparse
from dataset import Dataset
from torchvision import transforms
import transform
from torch.utils import data
import torch
from collections import OrderedDict
from models.MobieN2_backbone import Model
import os
import numpy as np
import IOU
import datetime
import torch.nn.functional as F
import random
from tqdm import tqdm
import cv2
from torchsummary import summary
from torch.autograd import profiler
from thop import profile
from utils import clip_gradient, LR_Scheduler

os.environ["CUDA_VISIBLE_DEVICES"] = ''

GPU_NUMS = torch.cuda.device_count()
p = OrderedDict()
p['lr_bone'] = 1e-4  # Learning rate1e-4 
p['lr_branch'] = 1e-3
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
# lr_decay_epoch = [9, 20]
lr_decay_epoch = [50]
showEvery = 100


CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)

best_s = 0
best_epoch = 0
all_s = []
all_F = []
all_M = []


def FocalLossBalance(pred, mask, gamma=2.0, alpha=0.25):
    pred_sigmoid = torch.sigmoid(pred)
    pt = (1 - pred_sigmoid) * mask + pred_sigmoid * (1 - mask)
    focal_weight = (alpha * mask + (1 - alpha) * (1 - mask)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none') * focal_weight
    return loss.mean()


def focal_e_loss(pred, mask):
    wfoc = FocalLossBalance(pred, mask)

    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred
    mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2], mask.shape[3])
    phiGT = mask - mmask
    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    # weighted iou loss function
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (wfoc + eloss + wiou).mean()


def structure_loss(pred, mask):
    bce = CE(pred, mask)
    iou = IOU(torch.nn.Sigmoid()(pred), mask) 
    return bce + iou                        


# def structure_loss(pred, mask):
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
#     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask) * weit).sum(dim=(2, 3))
#     union = ((pred + mask) * weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1) / (union - inter + 1)
#     return (wbce + wiou).mean()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser()
print(torch.cuda.is_available())

parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
# train
parser.add_argument('--epoch', type=int, default=101)
parser.add_argument('--gpu_id', type=str, default='0', help='the gpu id')
parser.add_argument('--epoch_save', type=int, default=5)
parser.add_argument('--save_fold', type=str, default='./Modelsave') 
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_thread', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')

parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

# test in train
parser.add_argument('--test_batch_size', type=int, default=8)
parser.add_argument('--train_dataset', type=list, default=[''])
# Misc
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
config = parser.parse_args()

config.save_fold = config.save_fold + '/' + 'New_RDVS'
if not os.path.exists("%s" % (config.save_fold)):
    os.mkdir("%s" % (config.save_fold))


def test_intrain(net_bone, test_loader, epoch):
    global best_s, best_epoch, all_s
    global best_F, all_F
    global best_M, best_epoch, all_M
    net_bone.eval()
    s_sum = 0
    f_measure_sum = 0
    mae_sum = 0
    counter = 0
    for i, data_batch in enumerate(test_loader):
        if counter % 200 == 0:
            print("progress  {}/{}".format(i + 1, len(test_loader)),end='————')
        counter+=1
        
        
        image, flow, depth, name, split, size, label = data_batch['image'], data_batch['flow'], data_batch['depth'], \
                                                       data_batch['name'], data_batch['split'], data_batch['size'], \
                                                       data_batch['label']

        if config.cuda:
            image, flow, depth, label = image.cuda(), flow.cuda(), depth.cuda(), label.cuda()
        with torch.no_grad():

            decoder_out1, decoder_out2, decoder_out3, decoder_out4,decoder_out5,edge= net_bone(image, flow, depth)
            decoder = decoder_out1.size()
            for j in range(config.test_batch_size):
                pre1 = torch.nn.Sigmoid()(decoder_out1[j])
                # pre1 = torch.nn.Sigmoid()(decoder_final[j])
                pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
                gt = label[j]
                gt[gt >= 0.5] = 1
                gt[gt < 0.5] = 0
            #1 
                f_measure = compute_f_measure(pre1, gt)
            #2
                mae = compute_mae(pre1, gt)
                f_measure_sum += f_measure.item()
                mae_sum += mae.item()
                alpha = 0.5
                y = gt.mean()
                if y == 0:
                    x = pre1.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pre1.mean()
                    Q = x
                else:
                    Q = alpha * S_object(pre1, gt) + (1 - alpha) * S_region(pre1, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                s_sum += Q.item()
    s = s_sum / len(test_loader.dataset)
    f = f_measure_sum / len(test_loader.dataset)
    m = mae_sum / len(test_loader.dataset)
    if epoch == 1:
        best_s = s
    else:
        if s > best_s:
            best_s = s
            best_epoch = epoch
            torch.save(net_bone.state_dict(), config.save_fold + '/epoch_best.pth')
            print('best epoch:{}'.format(epoch))
    all_s.append(s)
    print()
    print('Epoch: {} S-measure: {} ####  bestS: {} bestEpoch: {}'.format(epoch, s, best_s, best_epoch))
    print()


# F-measure\MAE 
def compute_f_measure(prediction, target):
    true_positives = torch.sum(prediction * target)
    precision = true_positives / (torch.sum(prediction) + 1e-6)
    recall = true_positives / (torch.sum(target) + 1e-6)
    f_measure = (2 * precision * recall) / (precision + recall + 1e-6)
    return f_measure

def compute_mae(prediction, target):
    return torch.mean(torch.abs(prediction - target))


def S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg

    return Q


def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    if torch.isnan(score):
        raise
    return score


def S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    return Q


def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        if config.cuda:
            X = torch.eye(1).cuda() * round(cols / 2)
            Y = torch.eye(1).cuda() * round(rows / 2)
        else:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        if config.cuda:
            i = torch.from_numpy(np.arange(0, cols)).cuda().float()
            j = torch.from_numpy(np.arange(0, rows)).cuda().float()
        else:
            i = torch.from_numpy(np.arange(0, cols)).float()
            j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)

    return X.long(), Y.long()


def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0

    return Q


def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3

    return LT, RT, LB, RB, w1, w2, w3, w4


def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]

    return LT, RT, LB, RB



if __name__ == '__main__':
    set_seed(1024)

    composed_transforms_ts = transforms.Compose([
        # transform.RandomFlip(),
        transform.RandomRotate(),
        transform.colorEnhance(),
        transform.randomPeper(),
        transform.RandomHorizontalFlip(),
        transform.FixedResize(size=(config.input_size, config.input_size)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])
    dataset_train = Dataset(datasets=config.train_dataset,
                            transform=composed_transforms_ts, mode='train')
    # datasampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=dist.get_world_size(),
    #                                                               rank=args.local_rank, shuffle=True)
    # dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, sampler=datasampler,
    #                                          num_workers=8)

    dataloader = data.DataLoader(dataset_train, batch_size=config.batch_size, num_workers=config.num_thread,
                                 drop_last=True,
                                 shuffle=True)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}, batch_size:{}".format(len(dataset_train), len(dataloader),config.batch_size))
    
    # for test in train
    composed_transforms_te = transforms.Compose([
        transform.FixedResize(size=(config.input_size, config.input_size)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])

    dataset_test = Dataset(datasets=['RDVS'], transform=composed_transforms_te, mode='test')
    test_loader = data.DataLoader(dataset_test, batch_size=config.test_batch_size, num_workers=config.num_thread,
                                  drop_last=True, shuffle=False)
    print("Testing Set, DataSet Size:{}, DataLoader Size:{}, batch_size:{}".format(len(dataset_test), len(test_loader),config.test_batch_size))
    iter_num = len(dataloader)
    lr_scheduler = LR_Scheduler('poly', config.lr, config.epoch, iter_num)

    # all train
    net_bone = Model(mode=config.mode)
    if config.cuda:
        net_bone = net_bone.cuda()
    if GPU_NUMS == 1:
        print(f"Loading model, and using single GPU - {config.gpu_id}")
    elif GPU_NUMS > 1:
        print(f"Loading model, and using multiple GPUs - {config.gpu_id}")
        net_bone = torch.nn.DataParallel(net_bone)
    print(f"Loading model, and using single GPU - {config.gpu_id}")
    if config.cuda:
        net_bone = net_bone.cuda()

    # optimizer_bone = torch.optim.SGD(filter(lambda p:p.requires_grad,net_bone.parameters()),lr=p['lr_bone'], momentum=p['momentum'],
    #                                   weight_decay=p['wd'], nesterov=True)
    optimizer_bone = torch.optim.Adam(net_bone.parameters(),config.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    optimizer_bone.zero_grad()
    iter_num = len(dataloader)

    loss_write = []
    loss_write_epoch = []

    for epoch in range(1, config.epoch + 1):
        loss_all = 0
        for param_group in optimizer_bone.param_groups:
            param_group['lr'] = config.lr
        net_bone.zero_grad()


        net_bone.train()
        for i, data_batch in enumerate(dataloader):#start=1
            image, label, flow, depth, size ,name ,edge = data_batch['image'], data_batch['label'], data_batch['flow'], data_batch[
                'depth'],data_batch['size'],data_batch['name'],data_batch['edge']
            if image.size()[2:] != label.size()[2:]:
                print("Skip this batch")
                continue

            # cur_lr = lr_scheduler(optimizer_bone,i+1, epoch)
           
            if config.cuda:
                image, label, flow, depth ,edge= image.cuda(), label.cuda(), flow.cuda(), depth.cuda(),edge.cuda()

            decoder_out1, decoder_out2, decoder_out3, decoder_out4,decoder_out5,edge_up= net_bone(image, flow, depth)

            loss1 = structure_loss(decoder_out1, label)
            loss2 = structure_loss(decoder_out2, label)
            loss3 = structure_loss(decoder_out3, label)
            loss4 = structure_loss(decoder_out4, label)
            loss5 = structure_loss(decoder_out5, label)
            loss_edge = CE(edge_up, edge)

            loss = loss1 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16 + loss_edge / 16

            optimizer_bone.zero_grad()
            loss.backward()
            clip_gradient(optimizer_bone,config.clip)
            optimizer_bone.step()
            loss_all += loss.data

            if i % showEvery == 0:
            # if i  != 100:

                print(
                    '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss1 : %10.4f  || sum : %10.4f' % (
                        datetime.datetime.now(), epoch, config.epoch, i, iter_num,
                        loss1.data, loss_all / (i + 1)))
                print('\033[91m' + 'Learning rate: %.7f ' % (config.lr) + '\033[0m')
                # print('Learning rate: ' + str(optimizer_bone.param_groups[0]['lr']))

            loss_write.append(loss_all / (i + 1))

        test_intrain(net_bone, test_loader, epoch)
        if (epoch) % config.epoch_save == 0:
            torch.save(net_bone.state_dict(),
                       '%s/epoch_%d_bone.pth' % (config.save_fold, epoch))
        loss_write_epoch.append(loss_all / (i + 1))
    with open("./train_loss_epoch100_2.txt", 'w') as train_los_e:
        for loss in loss_write_epoch:
            train_los_e.write(str(loss) + '\n')


    with open("./s_measure_3.txt", 'w') as s_measure:
        s_measure.write(str(all_s))
    with open("./MAE_3.txt", 'w') as MAE:
        MAE.write(str(all_M))
    torch.save(net_bone.state_dict(), '%s/final_bone.pth' % config.save_fold)
    model_path = config.save_fold + '/' + '/epoch_best.pth'
    model_size_bytes = os.path.getsize(model_path)
    print(f"模型 Size: {model_size_bytes/ (1024 * 1024)} MB")
    


