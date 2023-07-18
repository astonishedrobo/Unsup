from collections import defaultdict
from shutil import copyfile
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
from torch.utils.data.dataloader import default_collate
import pickle as pkl
from torch.utils.tensorboard import SummaryWriter


def prep_img(img):
    return Variable(img.unsqueeze(0)).cuda()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _fit_epoch(model, loader, criterion, optimizer, depth_est= True, color_seg = False,ignore_index = None):
    model.train()
    loss_meter = AverageMeter()
    t = tqdm(loader, total=len(loader))
    # counter = 0
    # pr = False
    for data, anno,target in t:
        if torch.cuda.is_available():
            data = Variable(data.cuda())
            anno = Variable(anno.cuda())
            if depth_est:
                target['x_A'] = target['x_A'].cuda()
                target['y_A'] = target['y_A'].cuda()
                target['x_B'] = target['x_B'].cuda()
                target['y_B'] = target['y_B'].cuda()
                target['ordinal_relation'] = Variable(target['ordinal_relation']).cuda()

            if depth_est ==True and color_seg == False:
                output = model(data)
                loss = criterion(output, target)
            elif depth_est == True and color_seg == True:
                output,color = model(data)
                if ignore_index == None:
                    loss = criterion(output, target)+nn.functional.cross_entropy(color, anno)
                else:
                    loss = criterion(output, target)+nn.functional.cross_entropy(color, anno,ignore_index = ignore_index)
            elif color_seg == True:
                color = model(data)
                
                if ignore_index == None:
                    loss = nn.functional.cross_entropy(color, anno)
                else:
                    loss = nn.functional.cross_entropy(color, anno, ignore_index = ignore_index)

            # print("training loss",loss.item())
            
            if torch.isnan(loss).any():
                print("Model Pretrained")
            #     # foo_bar = torch.isnan(data)
            #     # print(foo_bar.any(), data[foo_bar])
                
            #     if pr is False:
            #         if counter == 3:
            #             # print(color)
            #             pass
            #         else:
            #             print("Loss NaN")

                print("Vals:",torch.max(color), torch.min(color))
                exit(0)
            #             # print(color)
            #             pr = True
                
            # counter += 1
                
            
            loss_meter.update(loss.item())
            t.set_description("[ loss: {:.4f} ]".format(loss_meter.avg))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # # Sum up all gradients
            # gradient_sum = 0
            # for param in model.parameters():
            #     if param.grad is not None:
            #         gradient_sum += param.grad.sum().item()

            # # Print the sum of gradients
            # print("Sum of gradients:", gradient_sum)
    return loss_meter.avg

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    images, anno, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_anno = cat_list(anno, fill_value=255)
    batched_targets = default_collate(targets)
        
    return batched_imgs, batched_anno, batched_targets

def fit(model, train, criterion, optimizer,  save_path, depth_est= True, color_seg = False, batch_size=32,
        shuffle=True, nb_epoch=1, validation_data=None, cuda=True, num_workers=0, ignore_index = None, loss_log_dir = None):
    # TODO: implement CUDA flags, optional metrics and lr scheduler
    if validation_data:
        print('Train on {} samples, Validate on {} samples'.format(len(train), len(validation_data)))
    else:
        print('Train on {} samples'.format(len(train)))

    # Tensorboard Log
    if save_path is not None and loss_log_dir is None:
        log_directory_path = os.path.dirname(os.path.dirname(save_path))
        os.makedirs(os.path.join(log_directory_path, "log_directory"), exist_ok=True)
        log_directory = os.path.join(log_directory_path, "log_directory")
        folder_name = os.path.splitext(os.path.basename(save_path))[0]
        os.makedirs(os.path.join(log_directory, folder_name), exist_ok=True)
        lg = os.path.join(log_directory, folder_name)
        writer = SummaryWriter(lg)

    elif loss_log_dir is not None:
        os.makedirs(loss_log_dir, exist_ok=True)
        writer = SummaryWriter(loss_log_dir)

    train_loader = DataLoader(train, batch_size, shuffle, num_workers=num_workers, pin_memory=True,collate_fn = collate_fn)

    t = tqdm(range(nb_epoch), total=nb_epoch)
    training_loss = []
    validation_loss = []
    epoch_count = 1
    for epoch in t:
        tl = _fit_epoch(model, train_loader, criterion, optimizer, depth_est, color_seg, ignore_index = ignore_index )
        training_loss.append(tl)
        try:
            writer.add_scalar('Loss/train', tl, epoch_count)
        except:
            pass
        epoch_count += 1
        print(training_loss)
        if epoch % 5 ==0 :
            save_checkpoint(model.state_dict(), optimizer.state_dict(), save_path)
            with open(save_path.replace('.pth','_t.pkl'),'wb') as fp:
                pkl.dump(training_loss,fp)
            
            if validation_data:
                vl = validate(model, validation_data, criterion, batch_size, depth_est, color_seg)
                print("Valid: ", validation_loss)
                validation_loss.append(vl)
                writer.add_scalar('Loss/validation', vl, epoch_count)
                with open(save_path.replace('.pth','_v.pkl'),'wb') as fp:
                    pkl.dump(validation_loss,fp)
    try:
        writer.close()
    except:
        pass
    return training_loss, validation_loss
                
def validate(model, validation_data, criterion, batch_size, depth_est= True, color_seg = False):
    model.eval()
    val_loss = AverageMeter()
    loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    with torch.inference_mode():
        for data, anno,target in loader:
            data = Variable(data.cuda())
            anno = Variable(anno.cuda())
            # target['x_A'] = target['x_A'].cuda()
            # target['y_A'] = target['y_A'].cuda()
            # target['x_B'] = target['x_B'].cuda()
            # target['y_B'] = target['y_B'].cuda()
            # target['ordinal_relation'] = Variable(target['ordinal_relation']).cuda()
            if depth_est and not color_seg:
                output = model(data)
                loss = criterion(output, target)
            elif depth_est and color_seg:
                output,color = model(data)
                loss = criterion(output, target)+nn.functional.cross_entropy(color, anno)
            elif color_seg:
                color = model(data)
                loss = nn.functional.cross_entropy(color, anno)

            #output = model(data)
            #loss = criterion(output, target)
            # print(loss.item())
            val_loss.update(loss.item())
    return val_loss.avg


def save_checkpoint(model_state, optimizer_state, filename, epoch=None, is_best=False):
    state = dict(model_state=model_state,
                 optimizer_state=optimizer_state,
                 epoch=epoch)
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')
