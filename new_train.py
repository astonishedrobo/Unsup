import argparse
from datetime import datetime
import os
import dpt.transforms as T
from torch.optim import RMSprop, Adam
import torch
import matplotlib.pyplot as plt

from datasets import NYUDepth, NYUSeg
from models import HourGlass
from criterion import RelativeDepthLoss
from train_utils import fit, save_checkpoint
from torch.backends import cudnn
from dpt.models import DPTDepthModel, DPTSegmentationModel
from Sophia import SophiaG
import presets

def get_transform(train):
    if train:
        return presets.SegmentationPresetTrain(base_size=256, crop_size=160)
    
    else:
        return presets.SegmentationPresetVal(base_size=256)

def get_dataset(data_type, path_img, path_target, transforms):
    paths = {
        # class: dataset_fn
        "seg": NYUSeg,
        "depth": NYUDepth,
    }
    ds_fn = paths[data_type]
    num_classes = 256
    ignore_index = None

    return ds_fn(path_img, path_target, transforms), num_classes, ignore_index

def main(train_data_path, train_label_path, nb_epoch, save_path, start_path=None, batch_size=4, lr=0.001, depth_est = True, color_seg = False,
         plot_history=True,val_data_path = None, val_label_path = None, grayscale=False, model_type ='hourglass', no_transform = False, data_type='depth', loss_log_dir=None, distributed = False):
    cudnn.benchmark = True

    if depth_est ==True and color_seg == False:
        print("only depth estimation")
    elif depth_est == True and color_seg == True:
        print("joint depth and color")
        #grayscale = True
        #no_transform = False
    elif color_seg == True:
        print("only color")
        #grayscale = True
        #no_transform = False   

    # Define Dataset
    dataset_train, num_classes, ignore_index = get_dataset(data_type, train_data_path, train_label_path, get_transform(True))
    dataset_valid, num_classes, _ = get_dataset(data_type, val_data_path, val_label_path, get_transform(False))
    
    # Define Data Sampler
    # if distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_valid, shuffle=False)
    # else:
    #     train_sampler = torch.utils.data.RandomSampler(dataset_train)
    #     test_sampler = torch.utils.data.SequentialSampler(dataset_valid)

    # Define Models
    if model_type == 'hourglass':
        model = HourGlass(depth_est, color_seg)
        model.cuda()
    elif model_type == 'dpt':
        if depth_est and not color_seg:
            model =  DPTDepthModel(
                path=None,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
        elif color_seg:
            print("Checkpoint Seg")
            model = DPTSegmentationModel(
                num_classes,
                path=None,
                backbone="vitb_rn50_384",
            )
        model.cuda()
    
    optimizer = SophiaG(model.parameters(), lr=lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=0)

    criterion = RelativeDepthLoss()

    # TODO: Implement Distributed Training
    history = fit(model = model, train = dataset_train, criterion= criterion, optimizer = optimizer, save_path = save_path, batch_size = batch_size,
                  nb_epoch = nb_epoch, depth_est = depth_est, color_seg = color_seg, validation_data = dataset_valid, ignore_index = ignore_index, loss_log_dir = loss_log_dir)

if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='/home/soumyajit/Dataset/images/train')
    parser.add_argument('--train_label_path', default='/home/soumyajit/Dataset/annotations/train')
    parser.add_argument('--val_data_path', default='/home/soumyajit/ADEChallengeData2016/images/validation')
    parser.add_argument('--val_label_path', default='/home/soumyajit/ADEChallengeData2016/images/val_Slic_Merged_Segments')
    # parser.add_argument('--train_data_path', default='/home/soumyajit/ADEChallengeData2016/images/training')
    # parser.add_argument('--train_label_path', default='/home/soumyajit/ADEChallengeData2016/annotations/training')
    # parser.add_argument('--val_data_path', default='/home/soumyajit/ADEChallengeData2016/images/validation')
    # parser.add_argument('--val_label_path', default='/home/soumyajit/ADEChallengeData2016/annotations/validation')
    parser.add_argument('--color_seg',default = True,type=str2bool)
    parser.add_argument('--depth_est',default = False,type=str2bool)
    parser.add_argument('--grayscale',default = False,type=str2bool)
    parser.add_argument('--nb_epoch',default = 150,type = int)
    parser.add_argument('--save_path',default=os.path.join('/home/soumyajit/Unsup/saved_models', datetime.now().strftime('%mM-%dD_%Hh-%Mm-%Ss')) + '.pth')
    parser.add_argument('--loss_log_dir',default=None)
    parser.add_argument('--start_path', default=None)
    parser.add_argument('--batch_size', default=18,type  = int)
    parser.add_argument('--lr', default=1e-5,type = float)
    parser.add_argument('--model_type',default = 'dpt')
    parser.add_argument('--data_type',default = 'seg')
    parser.add_argument('--no_transform',default = False,type=str2bool)
    parser.add_argument('--distributed', default=True, help='If training is wanted to be distributed over multiple GPUs')
    args = parser.parse_args()
    
    main(args.train_data_path, args.train_label_path, args.nb_epoch, args.save_path, args.start_path, args.batch_size, args.lr, args.depth_est,args.color_seg,
         False, val_data_path = args.val_data_path, val_label_path = args.val_label_path, grayscale = args.grayscale, model_type = args.model_type, no_transform = args.no_transform,
         data_type=args.data_type, loss_log_dir = args.loss_log_dir, distributed= args.distributed)
    
