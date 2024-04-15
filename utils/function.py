# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate



def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_loss_location  = AverageMeter()
    avg_iou = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    # writer = writer_dict['writer']
    # global_steps = writer_dict['train_global_steps']
    # nums = config.MODEL.NUM_OUTPUTS
    # confusion_matrix = np.zeros(
    #     (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))

    for i_iter, batch in enumerate(trainloader, 0):
        images, _, _, bbox = batch
        # size = labels.size()
        images = images.cuda()
        # labels = labels.long().cuda()
        # bd_gts = bd_gts.float().cuda()
        bbox = bbox.float().cuda()
        # print(bbox)
        

        losses, pred, loss_list = model(images, bbox)
        # if not isinstance(pred, (list, tuple)):
        #         pred = [pred]
        # for i, x in enumerate(pred):
        #     x = F.interpolate(
        #         input=x, size=size[-2:],
        #         mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        #     )

        #     confusion_matrix[..., i] += get_confusion_matrix(
        #         labels,
        #         x,
        #         size,
        #         config.DATASET.NUM_CLASSES,
        #         config.TRAIN.IGNORE_LABEL
        #     )
        loss = losses.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        avg_loss.update(loss.item())
        avg_loss_location.update(loss_list[0].mean().item())
        avg_iou.update(loss_list[1].mean().item())

        # lr = adjust_learning_rate(optimizer,
        #                           base_lr,
        #                           num_iters,
        #                           i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, location:{:.6f}, iou: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], avg_loss.average(),
                      avg_loss_location.average(), avg_iou.average())
            logging.info(msg)
    # for i in range(nums):
    #     pos = confusion_matrix[..., i].sum(1)
    #     res = confusion_matrix[..., i].sum(0)
    #     tp = np.diag(confusion_matrix[..., i])
    #     IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    #     mean_IoU = IoU_array.mean()

    # writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    # # writer.add_scalar('train_mIoU', mean_IoU, global_steps)
    # # writer.add_scalar('train_ave_acc', ave_acc.average(), global_steps)
    # # writer.add_scalar('train_avg_sem_loss', avg_sem_loss.average(), global_steps)
    # # writer.add_scalar('train_avg_bce_loss', avg_bce_loss.average(), global_steps)
    # # writer.add_scalar('train_avg_sb_loss', ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average(), global_steps)
    # writer_dict['train_global_steps'] = global_steps + 1
    return avg_loss.average(), avg_loss_location.average(), avg_iou.average()

def validate(config, testloader, model, writer_dict):
    model.eval()
    avg_loss = AverageMeter()
    avg_loss_location  = AverageMeter()
    avg_iou = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, _,  _, bbox = batch
            # size = label.size()
            image = image.cuda()
            # label = label.long().cuda()/
            # bd_gts = bd_gts.float().cuda()
            bbox = bbox.float().cuda()
            losses, pred, loss_list = model(image, bbox)
            # if not isinstance(pred, (list, tuple)):
            #     pred = [pred]
            # for i, x in enumerate(pred[2:]):
            #     x = F.interpolate(
            #         input=x, size=size[-2:],
            #         mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            #     )

            #     confusion_matrix[..., i] += get_confusion_matrix(
            #         label,
            #         x,
            #         size,
            #         config.DATASET.NUM_CLASSES,
            #         config.TRAIN.IGNORE_LABEL
            #     )

            # if idx % 10 == 0:
            #     print(idx)

            loss = losses.mean()
            # update average loss
            avg_loss.update(loss.item())
            avg_loss_location.update(loss_list[0].mean().item())
            avg_iou.update(loss_list[1].mean().item())

    # for i in range(nums):
    #     pos = confusion_matrix[..., i].sum(1)
    #     res = confusion_matrix[..., i].sum(0)
    #     tp = np.diag(confusion_matrix[..., i])
    #     IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    #     mean_IoU = IoU_array.mean()
        
        # logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    # writer = writer_dict['writer']
    # global_steps = writer_dict['valid_global_steps']
    # writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    # # writer.add_scalar('valid_ave_acc', ave_acc.average(), global_steps)
    # # writer.add_scalar('valid_avg_sem_loss', avg_sem_loss.average(), global_steps)
    # # writer.add_scalar('valid_avg_bce_loss', avg_bce_loss.average(), global_steps)
    # # writer.add_scalar('valid_avg_sb_loss', ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average(), global_steps)
    # writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    # writer_dict['valid_global_steps'] = global_steps + 1
    return avg_loss.average(), avg_loss_location.average(), avg_iou.average()



def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
