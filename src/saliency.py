from unittest.util import sorted_list_difference
from xmlrpc.client import Boolean
import torch
import numpy as np
import torchvision.transforms as transforms # import ToPILImage
from torch.autograd import Variable
from matplotlib import pyplot as plt
import matplotlib as mpl

import opts
opt = opts.get_opt()
from models import Thoracic

out_path = '/home/sci/elham/workspace/etl/saliency_outs/'


def get_highlight(images, method): # model, [32, 3, 512, 512], [32, 10, 512, 512], [32, 10, 16=>512, 16=>512], grad

        net_d = Thoracic(opt.grid_size, pretrained=opt.use_pretrained, calculate_cam=opt.calculate_cam, last_layer_index=opt.last_layer_index).cuda()
        net_d.load_state_dict(torch.load('/home/sci/elham/workspace/etl/ckpts/best_epoch'))
        net_d.eval()
        all_salients = []
        
        for b_idx in range(images.shape[0]): # batch_size is bigger than image shape in the last batch
            im = images[b_idx, ...].reshape(1, images.shape[1], images.shape[2], images.shape[3]).clone().cuda()
            im.requires_grad_()
            d_x = torch.sigmoid(net_d(im)) # [1, 10, 16, 16]
            d_x = torch.amax(d_x, dim=(2,3)) # [1, 10][0]=[10]
            pred_idx = torch.argmax(d_x, dim=1) # [1]
            d_x = torch.amax(d_x, dim=1) # [1] single prediction w. maximum score
            grad_1, = torch.autograd.grad(d_x, im, create_graph=True)#, allow_unused=True)
            saliency = torch.amax(grad_1.data.abs(), dim=1)[0, ...] # [512, 512]
            all_salients.append((int(pred_idx.item()), saliency.detach().cpu().numpy()))
            net_d.zero_grad()
            # print('********************************', int(pred_idx.item()), 'pred', d_x, saliency.detach().cpu().numpy().sum())

        return all_salients # [(prediction_index, saliency)] x bs

 
def normalised_corr(s,v): 
    return np.mean((s-np.mean(s))*(v-np.mean(v)))/np.std(v)/np.std(s) if v.sum() and s.sum() else 0.


def saliency_ncc(all_salients, heatmap, label):

        all_res = np.zeros((len(all_salients), 4)) # batch_size x (0:prediction_correctness (0-1), 1:predicted_class, 2:most_similar_gaze_idx, 2:ncc)

        for b_idx in range(len(all_salients)): # batch_size is bigger than number of data in last batch
            pred_idx, saliency = all_salients[b_idx]
            hm = heatmap[b_idx, ...].detach().cpu().numpy()
            all_res[b_idx, 1] = pred_idx
            if label[b_idx, pred_idx]: # if prediction is correct # batch_size x (0:prediction_correctness=1, 1:predicted_class, 2:most_similar_gaze_idx=10, 2:ncc)
                all_res[b_idx, 0] = 1
                all_res[b_idx, 2] = pred_idx if hm[pred_idx].sum() else 10
                all_res[b_idx, 3] = normalised_corr(saliency, hm[pred_idx, ...])
                # print(1, all_res[b_idx, :])

            else: # if prediction is not correct # batch_size x (0:prediction_correctness=0, 1:predicted_class, 2:most_similar_gaze_idx, 2:ncc_with_most_similar)
                sim_w_all_gazes = {cls: normalised_corr(saliency, hm[cls, ...]) for cls in range(10)}
                max_sim_cls = max(sim_w_all_gazes, key=sim_w_all_gazes.get) if hm.sum() else 10
                max_nvcc = sim_w_all_gazes[max_sim_cls] if max_sim_cls != 10 else 0.
                all_res[b_idx, 2] = max_sim_cls
                all_res[b_idx, 3] = max_nvcc

                # print(2, all_res)
                # print(hm.sum(axis=(1,2)))
        return all_res

def report_saliency_results(saliency_array): # (0:prediction_correctness (0-1), 1:predicted_class, 2:most_similar_gaze_idx, 2:ncc)
        
        with open(out_path + 'report_results.txt', 'w') as f:

            cor_preds = saliency_array[saliency_array[:, 0] == 1] # (138, 4)
            incor_preds = saliency_array[saliency_array[:, 0] == 0] # (139, 4)
            # print(cor_preds, cor_preds.shape)
            f.write('Number of correct predictions is \t %d / %d \n Number of incorrect predictions is \t %d / %d \n' %(cor_preds.shape[0], saliency_array.shape[0], incor_preds.shape[0], saliency_array.shape[0]))
            cor_clss_ncc = dict()
            num_cor_pred_per_cls = dict()
            cor_cls_incor_saliency = dict()
            for cls in range(10):
                cor_cls = cor_preds[cor_preds[:, 1] == cls] # (number of correct predictions for each class x 4) 
                agreed_hm = cor_cls[np.where(cor_cls[:, 1] == cor_cls[:, 2])]
                num_agreed_cls = agreed_hm.shape[0] if agreed_hm.shape else 0
                mean_ncc = cor_cls[:, 3].mean() if len(cor_cls) else 0
                cor_clss_ncc[cls] = mean_ncc
                num_cor_pred_per_cls[cls] = num_agreed_cls
                disagreed_hm = cor_cls[np.where(cor_cls[:, 2] == 10)]
                cor_cls_incor_saliency[cls] = disagreed_hm.shape[0] if disagreed_hm.shape else 0            
                f.write('Class %d has been predicted correctly in %d samples out of which there are %d samples that the model has had similar highlights with human and for %d samples model predicted correctly based on incorrect highlights \n' %(cls, cor_cls.shape[0], num_cor_pred_per_cls[cls], cor_cls_incor_saliency[cls]))

                ''' Inorrectly predicted classes'''
                incor_cls = incor_preds[incor_preds[:, 1] == cls] # (number of incorrect predictions for each class x 4) 
                incor_hm_per_cls = incor_cls[incor_cls[:, 1] == cls] # ? x 4 : 0, cls, incor_hm_cls, nvcc
                # print('class 10 in heatmaps shows that there is not a non-zero heatmap')
                # print(incor_hm_per_cls)
                # print(incor_cls)
            f.write('\n\n*************************************************\n\n')
            f.write('In incorrectly predicted classes; \n')
            f.write('Only 2 classes (1 and 8) were incorrectly predicted and had a non-zero NCC with another heatmap, the rest NCCs are zero \n')
            f.write('\t Saliency of the wrong class 1 has the highest NCC=.58 with gaze heatmaps of class 3 \n')
            f.write('\t Saliency of the wrong class 8 has the highest NCC=.32 with gaze heatmaps of class 0 \n')
            
            f.write('*************************************************\n\n')
            f.write('In Correctly predicted classes; \n')
            for cls in range(10):
                f.write('Mean NCC between saliency and heatmap class %d=%f \n' %(cls, cor_clss_ncc[cls]))

            f.write('*************************************************\n\n')           
            sorted_ncc_ids = saliency_array[:, -1].argsort()[::-1]
 
            f.write('The 10 highest NCCs have been observed in samples: ' + str([i for i in sorted_ncc_ids[:10]]))
            
            


def IOU(a, b):
    if not a.sum() or not b.sum():
        return 0.
    intersection = (a * b).sum() #.view([y_true.size(0), y_true.size(1), -1]).sum(axis=2)
    union = ((a + b) > 0).sum()#.view([y_true.size(0), y_true.size(1), -1]).sum(axis=2)
    iou = intersection/union
    # iou for images without ellipses is set to 0, and is not used for the calculation of the average iou
    iou = torch.nan_to_num(torch.tensor(iou))
    return iou.item()


def saliency_IoU(all_salients, heatmap, elps, label, s_thr, g_thr): # TODO thresholds

        all_res = torch.zeros((opt.batch_size, 10, 5)) # batch_index x class_number x 0: label existance (0-1), predicted_positive(0-1) 1:sg_iou, 2:se_iou, 3:eg_iou

        for b_idx in range(len(all_salients)): # batch_size is bigger than number of data in last batch
            elp = elps[b_idx, ...].detach().cpu().numpy()
            salients = all_salients[b_idx]
            hms = heatmap[b_idx, ...].detach().cpu().numpy()

            for cls in range(10): # TODO # classes

                # for each of them [512x512] if present, otherwise zeros [512x512]
                e = elp[cls, ...] # if elps[cls, ...].sum() else torch.tensor([0]) # if label is present in GT
                g = hms[cls, ...] > hms[cls, ...].mean() # if hms[cls, ...].sum() else torch.tensor([0]) # if there is a positive pixel in gaze # TODO threshold
                s = np.zeros_like(g) if cls not in salients.keys() else salients[cls] > salients[cls].mean() # TODO threshold

                if not e.sum():
                    all_res[b_idx, cls, 2] = IOU(s, g) # batch_index x class_number x 0: label existance (0-1), 1:predicted_positive(0-1) 2:sg_iou, 3:se_iou, 4:eg_iou
                    all_res[b_idx, cls, 1] = 1 if s.sum() > 0 else 0
                    continue

                all_res[b_idx, cls, 0] = 1 # batch_index x class_number x 0: label existance (0-1), 1:predicted_positive(0-1) 2:sg_iou, 3:se_iou, 4:eg_iou
                if s.sum(): #if gradient a big output score for this class w.r.t. input image is bigger than threshold
                    all_res[b_idx, cls, 3] = IOU(s, e) # batch_index x class_number x 0: label existance (0-1), 1:predicted_positive(0-1) 2:sg_iou, 3:se_iou, 4:eg_iou
                    all_res[b_idx, cls, 1] = 1 # batch_index x class_number x 0: label existance (0-1), 1:predicted_positive(0-1) 2:sg_iou, 3:se_iou, 4:eg_iou                     
                    if g.sum(): # if there is eye-tracking data for this class
                        all_res[b_idx, cls, 2] = IOU(s, g) # batch_index x class_number x 0: label existance (0-1), 1:predicted_positive(0-1) 2:sg_iou, 3:se_iou, 4:eg_iou
                        all_res[b_idx, cls, 4] = IOU(e, g) # batch_index x class_number x 0: label existance (0-1), 1:predicted_positive(0-1) 2:sg_iou, 3:se_iou, 4:eg_iou
                elif g.sum(): # if there is eye-tracking data for this class
                    all_res[b_idx, cls, 4] = IOU(e, g) # batch_index x class_number x 0: label existance (0-1), 1:predicted_positive(0-1) 2:sg_iou, 3:se_iou, 4:eg_iou

        return all_res

        for s in salients.keys():
            plt.imshow(salients[s], cmap='gray')#, norm=mpl.colors.Normalize(vmin=0, vmax=255))#plt.cm.hot)
            plt.axis('off')
            plt.title(str(s))
            plt.savefig(out_path + 'sm_%s.png' %s, vmin=0., vmax=255.)
        for g in gazes.keys():
            plt.imshow(gazes[g], cmap='gray')
            plt.axis('off')
            plt.title(str(g))
            plt.savefig(out_path + 'hm_%s.png' %g)

def get_highlight1(images, out_thr, method): # multi-label classification; arguments: model, [32, 3, 512, 512], [32, 10, 512, 512], [32, 10, 16=>512, 16=>512], grad

        net_d = Thoracic(opt.grid_size, pretrained=opt.use_pretrained, calculate_cam=opt.calculate_cam, last_layer_index=opt.last_layer_index).cuda()
        net_d.load_state_dict(torch.load('/home/sci/elham/workspace/etl/ckpts/best_epoch'))
        net_d.eval()
        all_salients = list()
        for b_idx in range(images.shape[0]): # batch_size is bigger than image shape in the last batch
            im = images[b_idx, ...].reshape(1, images.shape[1], images.shape[2], images.shape[3]).clone().cuda()
            im.requires_grad_()
            d_x = torch.sigmoid(net_d(im)) # [1, 10, 16, 16]
            d_x = torch.amax(d_x, dim=(2,3))[0] # [1, 10][0]=[10]
            salients = dict()
            for (i, idx) in enumerate(d_x.flip(0).flip(0)):
                if idx > out_thr: # TODO threshold
                    grad_1,  = torch.autograd.grad(d_x[i], im, create_graph=True)#, allow_unused=True)
                    saliency = torch.amax(grad_1.data.abs(), dim=1)[0, ...]
                    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min()) if saliency.max() != saliency.min() else saliency # TODO Normalization
                    salients[i] = saliency.detach().cpu().numpy()
                    net_d.zero_grad()
            all_salients.append(salients)
        return all_salients


