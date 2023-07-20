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
#from torch.optim.lr_scheduler import PolynomialLR

def _get_aug_transform(train,grayscale = True, validation = False):
        base_size =400//2#520#240
        crop_size =320//2#480

        min_size = int((0.5 if train else 1.0) * base_size)
        max_size = int((2.0 if train else 1.0) * base_size)
        transforms = []
        print("Doing transform")

        if validation:
            transforms.append(T.RandomResize(crop_size,crop_size))
            # return T.Compose(transforms)    


        if train:
                transforms.append(T.RandomResize(min_size,max_size))
                transforms.append(T.RandomHorizontalFlip(0.5))
                transforms.append(T.RandomCrop(crop_size))



        if grayscale:
            print("using grayscale")
            transforms.append(T.Grayscale(3))
            
        transforms.append(T.ToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        if grayscale:
            transforms.append(T.Normalize(mean=[0.456, 0.456, 0.456],
                                  std=[0.224, 0.224, 0.224]))
        else:
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
            

        return T.Compose(transforms)

def main(train_data_path, train_label_path, nb_epoch, save_path, start_path=None, batch_size=4, lr=0.001, depth_est = True, color_seg = False,
         plot_history=True,val_data_path = None, val_label_path = None, grayscale=False, model_type ='hourglass', no_transform = False, data_type='depth', loss_log_dir=None):
        
    # torch.cuda.empty_cache()

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
        
    if no_transform:
        print("using no transform")
        trans  = T.Compose([T.ToTensor(),T.ConvertImageDtype(torch.float)])
        if grayscale:
                print("using grayscale")
                trans  = T.Compose([T.Grayscale(3),T.ToTensor(),T.ConvertImageDtype(torch.float)])
    else:
        trans = _get_aug_transform(True,grayscale)
        
    if data_type == 'depth':
        train = NYUDepth(train_data_path, train_label_path, trans)
        num_classes = 255
        ignore_index = None
    elif data_type == 'seg':
        train = NYUSeg(train_data_path, train_label_path, trans)
        num_classes = 256#150#
        ignore_index = None#255#
    val = None
    if val_label_path is not None:
        if no_transform:
            trans  = T.Compose([T.ToTensor(),T.ConvertImageDtype(torch.float)])
        else:
            trans = _get_aug_transform(False,grayscale,True)
        if data_type == 'depth':
            val = NYUDepth(val_data_path, val_label_path, trans)
        elif data_type == 'seg':
            val = NYUSeg(val_data_path, val_label_path, trans)

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
    #optimizer = RMSprop(model.parameters(), lr, momentum=0.9)
    #optimizer = DecoupledSophia(model.parameters(), lr=1e-3, betas=(0.9, 0.999), rho=0.04, weight_decay=1e-1,     estimator="Hutchinson")
    optimizer = SophiaG(model.parameters(), lr=lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=0)
    # optimizer = Adam(model.parameters(), lr)
    #optimizer = SGD(model.parameters(), lr=2e-4, momentum=0.9)
    #optimizer = PolynomialLR(optimizer_non_lrsch)

    if start_path:
        experiment = torch.load(start_path,torch.device('cpu'))
        model.load_state_dict(experiment['model_state'])
        optimizer.load_state_dict(experiment['optimizer_state'])
        # model.to('cuda')

    criterion = RelativeDepthLoss()
    
    # optimizer.to('cuda')
    history = fit(model = model, train = train, criterion= criterion, optimizer = optimizer, save_path = save_path, batch_size = batch_size,
                  nb_epoch = nb_epoch, depth_est = depth_est, color_seg = color_seg, validation_data = val, ignore_index = ignore_index, loss_log_dir = loss_log_dir)
    
    save_checkpoint(model.state_dict(), optimizer.state_dict(), save_path)
    if plot_history:
        plt.plot(history['loss'], label='loss')
        plt.xlabel('epoch')
        plt.ylabel('relative depth loss')
        plt.legend()
        plt.show()


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
    parser.add_argument('--train_data_path', default='/mnt/efs/Data/ADEChallengeData2016/images/ade_training')
    parser.add_argument('--train_label_path', default='/mnt//efs/Data/ADEChallengeData2016/annotations/ade_training_segm')
    parser.add_argument('--val_data_path', default='/mnt//efs/Data/ADEChallengeData2016/images/validation')
    parser.add_argument('--val_label_path', default='/mnt//efs/Data/ADEChallengeData2016/annotations/validation_segm')
    parser.add_argument('--color_seg',default = True,type=str2bool)
    parser.add_argument('--depth_est',default = False,type=str2bool)
    parser.add_argument('--grayscale',default = False,type=str2bool)
    parser.add_argument('--nb_epoch',default = 150,type = int)
    parser.add_argument('--save_path',default=os.path.join('./saved_models', datetime.now().strftime('%mM-%dD_%Hh-%Mm-%Ss')) + '.pth')
    parser.add_argument('--loss_log_dir',default=None)
    parser.add_argument('--start_path', default=None)
    parser.add_argument('--batch_size', default=20,type  = int)
    parser.add_argument('--lr', default=1e-5,type = float)
    parser.add_argument('--model_type',default = 'dpt')
    parser.add_argument('--data_type',default = 'seg')
    parser.add_argument('--no_transform',default = False,type=str2bool)
    args = parser.parse_args()
    
    main(args.train_data_path, args.train_label_path, args.nb_epoch, args.save_path, args.start_path, args.batch_size, args.lr, args.depth_est,args.color_seg,
         False, val_data_path = args.val_data_path, val_label_path = args.val_label_path, grayscale = args.grayscale, model_type = args.model_type, no_transform = args.no_transform,
         data_type=args.data_type, loss_log_dir = args.loss_log_dir)
    

