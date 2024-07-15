# ------------------------------------------
# DAFN: Dual Attention Fusion Network
# Licensed under the MIT License.
# written By Ruixin Yang
# ------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torchvision import transforms   
from torch.nn.modules.loss import CrossEntropyLoss,BCELoss
from torch.nn import functional as F

class ISICLoss(nn.Module):
    def __init__(self,n_classes, alpha=0.4, beta=0.6) -> None:
        super().__init__()
        assert(alpha + beta == 1.)
        self.dice = DiceLoss(n_classes)
        self.ce = CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
    def forward(self,y_, y):
        loss_dice = self.dice(y_, y, softmax=True)
        loss_ce = self.ce(y_,y.long())
        loss = self.alpha*loss_ce + self.beta*loss_dice
        # print(f'{self.dice.get_dice_class()}')
        return loss,loss_ce,loss_dice

class SegPCLoss(nn.Module):
    def __init__(self,n_classes, alpha=0.4, beta=0.6) -> None:
        super().__init__()
        assert(alpha + beta == 1.)
        self.dice = DiceLossWithLogtis()
        self.ce = CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
    def forward(self,y_, y):
        loss_dice = self.dice(y_, y)
        loss_ce = self.ce(y_,y)
        loss = self.alpha*loss_ce + self.beta*loss_dice
        # print(f'{self.dice.get_dice_class()}')
        return loss,loss_ce,loss_dice


class SegPCLoss_full(nn.Module):
    def __init__(self,n_classes, alpha=0.4, beta=0.6) -> None:
        super().__init__()
        assert(alpha + beta == 1.)
        self.dice = DiceLossNoBackgorund(n_classes)
        self.ce = CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
    def forward(self,y_, y):
        loss_dice = self.dice(y_, y, softmax=True)
        loss_ce = self.ce(y_,y.long())
        loss = self.alpha*loss_ce + self.beta*loss_dice
        # print(f'{self.dice.get_dice_class()}')
        return loss,loss_ce,loss_dice

class DiceLossWithLogtis(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        prob = F.softmax(pred, dim=1)
        true_1_hot = mask.type(prob.type())
        
        dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
        intersection = torch.sum(prob * true_1_hot, dims)
        cardinality = torch.sum(prob + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + 1e-6)).mean()
        return (1 - dice_loss)

class SynapseLoss(nn.Module):
    def __init__(self,n_classes, alpha=0.4, beta=0.6) -> None:
        super().__init__()
        assert(alpha + beta == 1.)
        self.dice = DiceLoss(n_classes)
        self.ce = CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
    def forward(self,y_, y):
        loss_ce = self.ce(y_, y.long())
        loss_dice = self.dice(y_, y, softmax=True)
        loss = self.alpha*loss_ce + self.beta*loss_dice

        return loss,loss_ce,loss_dice

class DiceLossNoBackgorund(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossNoBackgorund, self).__init__()
        assert(n_classes > 1)
        self.n_classes = n_classes
        self.dice_class = None

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def get_dice_class(self):
        return torch.tensor(self.dice_class)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            if i == 0: # do not calculate backgorund
                continue
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]

        self.dice_class = class_wise_dice
        return loss / self.n_classes

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.dice_class = None

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def get_dice_class(self):
        return torch.tensor(self.dice_class)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]

        self.dice_class = class_wise_dice
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def calculate_metric_percase_PGDM(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        miou = metric.binary.jc(pred,gt)
        return miou,dice,hd95, 
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 1, 0, 
    else:
        return 0, 0, 0


def test_single_volume(image, label, net, device, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            img = image[ind, :, :]
            h, w = img.shape[0], img.shape[1]
            if h != patch_size[0] or w != patch_size[1]:
                img = zoom(img, (patch_size[0] / h, patch_size[1] / w), order=3)  # previous using 0
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            x = x_transforms(img).unsqueeze(0).float().to(device) # type: ignore

            net.eval()
            with torch.no_grad():
                y_ = net(x)
                # outputs = F.interpolate(outputs, size=img.shape[:], mode='bilinear', align_corners=False)
                out = torch.argmax(torch.softmax(y_, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if h != patch_size[0] or w != patch_size[1]:
                    pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(x), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def test_single_volume_PDGM(image, label, net, device, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 4:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]): # for each slice
            img = image[ind, :, :]
            h, w = img.shape[0], img.shape[1]
            if h != patch_size[0] or w != patch_size[1]:
                img = zoom(img, (patch_size[0] / h, patch_size[1] / w), order=3)  # previous using 0
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            x = x_transforms(img).unsqueeze(0).float().to(device) # type: ignore

            net.eval()
            with torch.no_grad():
                y_ = net(x)
                # outputs = F.interpolate(outputs, size=img.shape[:], mode='bilinear', align_corners=False)
                out = torch.argmax(torch.softmax(y_, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if h != patch_size[0] or w != patch_size[1]:
                    pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(x), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_PGDM(prediction == i, label == i))

    if test_save_path is not None:
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        for i in range(image.shape[-1]):
            img_itk = sitk.GetImageFromArray(image[:,:,:,i].astype(np.float32))
            sitk.WriteImage(img_itk, test_save_path + '/'+ case + f"_img{i}.nii.gz")
        
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


