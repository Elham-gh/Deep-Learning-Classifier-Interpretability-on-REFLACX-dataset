# script called for training and validating the proposed method
import torch
torch.backends.cudnn.benchmark = True
import gradcam 
import torchvision
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import contextlib
import numpy as np
from tensorboardX import SummaryWriter

from saliency import get_highlight, saliency_IoU, saliency_ncc, report_saliency_results, resize_box#, get_highlight_train
import model
import output as outputs
import metrics
import opts

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def init_optimizer(net_d, opt):
    if opt.optimizer == 'adamw':
        optimizer_d = torch.optim.AdamW(net_d.parameters(), lr=0.001, betas=(
                    0.5, 0.999), weight_decay=opt.weight_decay)
    if opt.optimizer == 'adamams':
        optimizer_d = torch.optim.Adam(net_d.parameters(), lr=0.0001, weight_decay=opt.weight_decay, amsgrad=True)
        # optimizer_d = torch.optim.Adam(net_d.parameters(), lr=0.001, weight_decay=opt.weight_decay, amsgrad=True) #*
    lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.1)
    return optimizer_d, lr_scheduler_d

# normalization of probabilities that will be multiplied. Used to avoid underflow
# If n_squares is not provided, use the normalization proposed by li et al, where all 
# probabilities are mapped to [0.98,1].
# n_squares can be provided for cases when the multiplication is not performed for all cells
# of a 16x16 grid. The value 0.0056738 was chosen such that 0.0056738**(1/(16*16))=0.98
def normalize_p_(p, n_squares=None):
    if n_squares is None:
        lower_bound = 0.98
    else:
        lower_bound = 0.0056738**(1/n_squares)
        lower_bound = lower_bound.unsqueeze(2).unsqueeze(2)
    return p*(1-lower_bound)+lower_bound

# generic training class that is inherited for the two training cycles used for 
# the paper
class TrainingLoop():
    def __init__(self, output,metric, opt):
        if opt.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        self.context_forward_pass = torch.cuda.amp.autocast() if opt.use_mixed_precision else contextlib.nullcontext()
        self.metric = metric
        self.output = output
        self.opt = opt
        self.normalize_fn = normalize_p_
        self.writer = SummaryWriter(opt.save_folder)
        self.sal_loss_type = {'ncc': model.loss_ncc(), 'mse':model.loss_nMSE(), 'kl': None}

        #defining how to calculate the loss and the image-level labels from the grid
        if self.opt.loss == 'ce': #loss only used for the unannotated model, simply using cross-entropy over the average-pooling of the grid
            self.loss_fn = model.loss_ce(opt.threshold_box_label, opt.weight_loss_annotated, self.normalize_fn, opt.use_grid_balancing_loss)
            self.forward = model.forward_inference_ce
            if self.opt.sal_loss:
                self.loss_sal = self.sal_loss_type[self.opt.sal_loss_type]
        elif self.opt.loss == 'li': #loss as defined in the paper, as proposed by li et al.
            self.loss_fn = model.loss_fn_li(opt.threshold_box_label, opt.weight_loss_annotated, self.normalize_fn, opt.use_grid_balancing_loss)
            self.forward = model.forward_inference
            if self.opt.sal_loss:
                self.loss_sal = self.sal_loss_type[self.opt.sal_loss_type]
        
    #method that iterates through all epochs through all batches for training and validation sets
    def train(self, train_dataloader, val_dataloader_ann, val_dataloader_all, net_d, optim_d, lr_scheduler_d):

        '''
        num_all_ims = 1696
        num_nonzero_ets = 1063
        num_zero_et_channels = 13838
        num_nonzero_et_channels = 3122
        '''
        
        last_best_validation_metric = self.opt.initialization_comparison
        for epoch_index in range(self.opt.nepochs): # Z:\workspace\etl\src\runs\et_data_model_20221103-115527-7267\state_dict_d_0            
            all_losses = {'loss':0, 'cls':0, 'slc': 0}
            self.metric.start_time('epoch')
            if not self.opt.skip_train:
                self.metric.start_time('train')
                net_d.train()
                print('Epoch %d' %epoch_index)
                for batch_index, batch_example in enumerate(train_dataloader):
                    if batch_index % 150==0:
                        print('batch %d/%d' %(batch_index + 1, len(train_dataloader)))
                    # image, ill_report/ill_mimic, contain_box, et, elps
                    image, label, contain_box, et, boxes = batch_example
                    image = image.cuda()
                    label = label.cuda()
                    contain_box = contain_box.cuda()
                    et = et.cuda()
                    boxes = boxes.cuda()
                    # call the train_fn, to be defined 
                    # in the child classes. This function does the 
                    # training of the model for the current batch of examples
                    iteration = (epoch_index) * 8 + batch_index
                    # iteration = (epoch_index) * len(train_dataloader) + batch_index
                    all_losses = self.train_fn(iteration, all_losses, image, label, contain_box, et, boxes, net_d, optim_d, self.opt.get_saliency, self.normalize_fn)
                    if batch_index == 8:
                        break
                self.metric.end_time('train')
                

            if False: # not self.opt.skip_validation:#*
                self.metric.start_time('validation')
                # from torch.autograd import Variable
                if self.opt.get_saliency:
                    saliency_results = np.zeros((1, 4))
                    saliency_file = '/home/sci/elham/workspace/etl/saliency/saliency_%s.npy' %self.opt.get_saliency
                with torch.no_grad():
                    net_d.eval()
                    if self.opt.validate_iou:
                        for batch_index, batch_example in enumerate(val_dataloader_ann):
                                image, label, contain_box, boxes, pixelated_boxes, mimic_label = batch_example
                                image = image.cuda()
                                label = label.cuda()
                                contain_box = contain_box.cuda()
                                boxes = boxes.cuda()
                                pixelated_boxes = pixelated_boxes.cuda() # 512 => eyetracking_dataset.get_dataset_et if get_saliency:
                                mimic_label = mimic_label.cuda()
                                # if batch_index%10==0:
                                #     print(batch_index)
                                # call the validation_fn function, to be defined 
                                # in the child classes, that defines what to do
                                # during the validation loop
                                # self.validation_fn_ann(image, label, contain_box, boxes, pixelated_boxes, mimic_label, net_d) # **********************

                            #     if self.opt.get_saliency and not os.path.exists(saliency_file):
                            #             torch.set_grad_enabled(True) #*
                            #             out_thr, s_thr, g_thr = .95, .5, .3
                            #             all_salients = get_highlight(image, out_thr, self.opt.get_saliency) # out_thr: threshold on network output scores after sigmoid
                            #             s_ious = saliency_IoU(all_salients, pixelated_boxes, boxes, label, s_thr, g_thr)
                            #             saliency_results = torch.cat((saliency_results, s_ious), axis=0)
                            # if self.opt.get_saliency and not os.path.exists(saliency_file):
                            #     saliency_results = saliency_results[1:, ...]
                            #     np.save(saliency_file, np.array(saliency_results))
                            # if self.opt.get_saliency:
                            #     saliency_ious = np.load(saliency_file)
                            #     print(saliency_ious)

                        #         if self.opt.get_saliency and not os.path.exists(saliency_file):
                        #                 if batch_index == 0:
                        #                         print('*************   calculating saliency maps   *************')
                        #                         print('batch %d / %d' %(batch_index, len(val_dataloader_ann)))
                        #                 torch.set_grad_enabled(True) #*
                        #                 all_idxs_salients = get_highlight(image, self.opt.get_saliency) # out_thr: threshold on network output scores after sigmoid
                        #                 print(batch_index, all_idxs_salients)
                        #                 # print((i[1].shape, i[1].max(), i[1].min()) for i in all_idxs_salients)
                        #                 # print(all_idxs_salients[1].shape, all_idxs_salients[1].max(), all_idxs_salients[1].min())
                        #                 continue
                        #                 s_ncc = saliency_ncc(all_idxs_salients, pixelated_boxes, label) # batch_size x (0:prediction_correctness (0-1), 1:predicted_class, 2:most_similar_gaze_idx, 2:ncc)
                        #                 saliency_results = np.concatenate((saliency_results, s_ncc), axis=0)
                        # if self.opt.get_saliency and not os.path.exists(saliency_file): # ????????????????????????????
                        #         print('*************   saving saliency maps   *************')
                        #         saliency_results = saliency_results[1:, ...]
                        #         np.save(saliency_file, np.array(saliency_results))
                        #         print(saliency_results)
                        # if self.opt.get_saliency:
                        #         print('*************   loading saliency maps   *************')
                        #         saliency_nccs = np.load(saliency_file) # batch_size x (0:prediction_correctness (0-1), 1:predicted_class, 2:most_similar_gaze_idx, 2:ncc)
                        #         # print(saliency_nccs)
                        # oiu8ou
                        # if self.opt.get_saliency and self.opt.reprod_saliency_results:
                        #     report_saliency_results(saliency_nccs)
                            # for idx in [42, 235, 264, 27, 127, 172, 241, 142, 182, 156]:
                            #     b_idx = idx // self.opt.batch_size
                            #     sample_idx = idx % self.opt.batch_size
                            #     print(b_idx, sample_idx)

                    if self.opt.validate_auc:
                        for batch_index, batch_example in enumerate(val_dataloader_all):
                            image, label  = batch_example
                            image = image.cuda()
                            label = label.cuda()
                            if batch_index%100==0:
                                print(batch_index)
                            # call the validation_fn function, to be defined 
                            # in the child classes, that defines what to do
                            # during the validation loop
                            self.validation_fn_all(image, label, net_d)
                    net_d.train()
                self.metric.end_time('validation')
            self.metric.end_time('epoch')
            self.metric.end_time('full_script')
            
            # get a dictionary containing the average metrics for this epoch, and writes all metrics to the log files
            # average_dict = self.output.log_added_values(epoch_index, self.metric) #*

            if not self.opt.skip_train:
                if self.opt.use_lr_scheduler:
                    lr_scheduler_d.step()
                
                #* if training, check if the model from the current epoch is the 
                #* best model so far, and if it is, save it
                # this_validation_metric = average_dict[self.opt.metric_to_validate]#*
                # if self.opt.function_to_compare_validation_metric(this_validation_metric, last_best_validation_metric):#*
                #     torch.save(net_d.state_dict(), self.opt.save_path + 'best_epoch')#*
                #     last_best_validation_metric = this_validation_metric#*
                #     print('***************** Best ckpt is updated *****************')#*
                
                # #save the model for the current epoch#*
                # torch.save(net_d.state_dict(), self.opt.save_path + 'last')#*
                # print('***************** Model has been saved *****************')#*

                '''
                this_validation_metric = average_dict[self.opt.metric_to_validate]
                if self.opt.function_to_compare_validation_metric(this_validation_metric,last_best_validation_metric):
                    self.output.save_models(net_d, 'best_epoch')
                    last_best_validation_metric = this_validation_metric
                
                #save the model for the current epoch
                self.output.save_models(net_d, str(epoch_index))
                '''
#class defining how training and validation for each batch is performed
class SpecificTraining(TrainingLoop):
    def train_fn(self, iteration, all_losses, image, label, contain_box, et, boxes, net_d, optim_d, saliency_method, normalize_fn): # a contain_box per image
        # TODO pass the heatmaps, if there is a diagnosed label calculate the gradient
        for p in net_d.parameters(): p.grad = None
        boxes = resize_box(boxes, self.opt.grid_size)

        with self.context_forward_pass:

            loss = 0
            image.requires_grad_()
            saliency_loss = 0
            lambda1, lambda2 = 1, 0
            d_x = net_d(image, return_saliency=False)
            classifier_loss = self.loss_fn(d_x, label, boxes, contain_box) # [32 x 10], classifier_loss [32]
            
            # saliency map of predicted classes w.r.t the input image:
            d_x_sal = net_d(image, boxes, normalize_fn, return_saliency=True, split='train') # [1, 10]
            all_salients = torch.zeros((1, 10, 512, 512), requires_grad=True).cuda()
            # d_x_inds = torch.nonzero((d_x_sal[0] > self.opt.out_thr))

            num_nunzero_channels = 0
            for ind in range(10): #d_x_inds: # 
                if et[0, ind, ...].sum(): # and d_x_sal[0, ind] > self.opt.out_thr: # TODO I can remove it
                    num_nunzero_channels += 1
                    pred = d_x_sal[0, ind]
                    pred.squeeze_()
                    # if pred > self.out_thr:
                    lambda2 = self.opt.sal_loss
                    lambda1 = 1 - lambda2
                    net_d.zero_grad()
                    grad_1, = torch.autograd.grad(pred, image, create_graph=True)#, allow_unused=True) # [1, 3, 512, 512]
                    saliency = torch.amax(grad_1.data.abs(), dim=1)[0, ...] # [512, 512] # TODO basically grayscale; convert to grayscale before/after gradient
                    all_salients[0, ind, ...] = saliency
                    saliency, grad_1, pred = saliency.detach(), grad_1.detach(), pred.detach()
                
                # sal = get_highlight_train(et_im, net_d, saliency_method, return_saliency)
            strd = all_salients.shape[-1] // self.opt.grid_size 
            all_salients = torch.nn.functional.max_pool2d(all_salients, kernel_size=strd, stride=strd)

            saliency_loss = self.loss_sal(et, all_salients) / num_nunzero_channels if num_nunzero_channels else self.loss_sal(et, all_salients)# TODO 16x16 -> 512x512
            
            loss = lambda1 * classifier_loss + lambda2 * saliency_loss
            print('iteration % d \t Loss = %f \t CLS_Loss = %f \t SAL_Loss = %f' %(iteration, loss, classifier_loss, saliency_loss))

        if self.opt.use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(optim_d)
            self.scaler.update()
        else:
            loss.backward()
            optim_d.step()
        self.writer.add_scalar('CLS_Loss_b', classifier_loss, iteration)
        self.writer.add_scalar('Loss_b', loss, iteration)
        self.writer.add_scalar('Sal_Loss_b', saliency_loss, iteration)
        self.writer.flush() 

        # adding model outputs for calculating training auc
        self.metric.add_score(label, self.forward(d_x, self.normalize_fn), 'train')
        return all_losses
    
    #validation for annotated functions
    def validation_fn_ann(self,image, label, contain_box, boxes, pixelated_boxes, mimic_label, net_d):
        #saving model and gradient state to restore it after the end of validation
        original_model_mode = net_d.training
        prev_grad_enabled = torch.is_grad_enabled()
        
        if self.opt.calculate_cam or self.opt.get_saliency:
            torch.set_grad_enabled(True)
        
        net_d.eval()
        d_x = net_d(image)
        
        # calculate iou for the last spatial layer of the network using several thresholds over the activations
        out_map = torch.sigmoid(d_x)
        if boxes.size(2)==512:
            out_map = torchvision.transforms.Resize(512, torchvision.transforms.InterpolationMode.NEAREST)(out_map)
        for threshold in self.opt.thresholds_iou:
            iou = localization_score_fn(boxes, out_map, threshold)
            self.metric.add_iou(f'val_ellipse_iou_{threshold}', iou, label)

        # calculating the image-level outputs for the model and adding them to the validation AUC calculation
        nonspatial_predictions = self.forward(d_x, self.normalize_fn)
        self.metric.add_score(label, nonspatial_predictions, 'val_rad')
        self.metric.add_score(label, nonspatial_predictions, 'val_mimic_ann')
        
        #calculating cam can be a bit slow, so leave it turned off while training
        if self.opt.calculate_cam:
            cam = gradcam.get_cam(net_d, nonspatial_predictions, 10).detach()
            if boxes.size(2)==512:
                cam = torchvision.transforms.Resize(512, torchvision.transforms.InterpolationMode.NEAREST)(cam)
            for threshold in self.opt.thresholds_iou:
                iou = localization_score_fn(boxes, cam, threshold)
                self.metric.add_iou(f'val_cam_iou_{threshold}', iou.detach(), label.detach())
        
        # self.opt.draw_images is only true for when running scripts to generate examples to put in the paper
        if self.opt.draw_images:
            plt.imshow(image.squeeze(0)[0].cpu().numpy(), cmap='gray')
            plt.imshow(cam[0][self.opt.label_to_draw].cpu().numpy(), cmap='jet', alpha = 0.3)
            plt.axis('off')
            plt.savefig(f'{self.output.output_folder}/sm{self.opt.sm_suffix}.png', bbox_inches='tight', pad_inches = 0)
            
            plt.imshow(image.squeeze(0)[0].cpu().numpy(), cmap='gray')
            plt.imshow(boxes[0][self.opt.label_to_draw].cpu().numpy(), cmap='jet', alpha = 0.3)
            plt.axis('off')
            plt.savefig(f'{self.output.output_folder}/gt{"".join([i for i in self.opt.sm_suffix.split() if i.isdigit()])}.png', bbox_inches='tight', pad_inches = 0)
        
        # restoring the state of the model and gradients before entering this method
        net_d.zero_grad()
        torch.set_grad_enabled(True)
        # torch.set_grad_enabled(prev_grad_enabled)
        if original_model_mode:
            net_d.train()
    
    #validation for unannotated functions (mimic-cxr dataset)
    def validation_fn_all(self,image, label, net_d):
        d_x = net_d(image)
        if self.opt.get_saliency:
            return self.forward(d_x, self.normalize_fn)
        #adding output of model to calculate AUC
        self.metric.add_score(label, self.forward(d_x, self.normalize_fn), 'val_mimic_all')

# calculating iou score from the GPU tensors
def localization_score_fn(y_true, y_predicted, threshold):    
    y_predicted = (y_predicted>threshold)*1.
    intersection = (y_predicted*y_true).view([y_true.size(0), y_true.size(1), -1]).sum(axis=2)
    union = (torch.maximum(y_predicted,y_true)).view([y_true.size(0), y_true.size(1), -1]).sum(axis=2)
    iou = intersection/union
    # iou for images without ellipses is set to 0, and is not used for the calculation of the average iou
    iou = torch.nan_to_num(iou)
    return iou

# opt = opts.get_opt()
def main():
    #get user options/configurations
    opt = opts.get_opt()
    
    #load Outputs class to save metrics, images and models to disk
    # if not os.path.exists(opt.save_path): #*
    #     os.mkdir(opt.save_path)#*
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)    
    output = outputs.Outputs(opt, opt.save_folder + '/' + opt.experiment + '_' + opt.timestamp)
    output.save_run_state(os.path.dirname(__file__))
    
    # from mypackage.mymodule import as_int

    from get_together_dataset import get_together_dataset as get_dataloaders
    
    #load class to store metrics and losses values
    metric = metrics.Metrics(opt.threshold_ior, opt.validate_auc)
    metric.start_time('full_script')
    if opt.skip_train:
        loader_train = None
    else:
        loader_train = get_dataloaders(split='train',type_ = opt.dataset_type, use_et = opt.use_et, crop = (opt.use_center_crop), batch_size = opt.batch_size, use_data_aug = opt.use_data_augmentation, num_workers = opt.num_workers, percentage_annotated=opt.percentage_annotated, percentage_unannotated=opt.percentage_unannotated, repeat_annotated = opt.repeat_annotated, load_to_memory=opt.load_to_memory, data_aug_seed = opt.data_aug_seed, index_produce_val_image = opt.index_produce_val_image, grid_size = opt.grid_size, dataset_type_et = opt.dataset_type_et)
    loader_val_rad = get_dataloaders(split=opt.split_validation + '_ann',type_ = opt.dataset_type, use_et = opt.use_et, crop = (opt.use_center_crop), batch_size = opt.batch_size, use_data_aug = opt.use_data_augmentation, num_workers = opt.num_workers, percentage_annotated=1., percentage_unannotated=0., repeat_annotated = False, load_to_memory=opt.load_to_memory, data_aug_seed = opt.data_aug_seed, index_produce_val_image = opt.index_produce_val_image, grid_size = opt.grid_size, dataset_type_et = opt.dataset_type_et)
    loader_val_all = get_dataloaders(split=opt.split_validation+ '_all',type_ = opt.dataset_type, use_et = opt.use_et, crop = (opt.use_center_crop), batch_size = opt.batch_size, use_data_aug = opt.use_data_augmentation, num_workers = opt.num_workers, percentage_annotated=0., percentage_unannotated=1., repeat_annotated = False, load_to_memory=opt.load_to_memory, data_aug_seed = opt.data_aug_seed, index_produce_val_image = opt.index_produce_val_image, grid_size = opt.grid_size, dataset_type_et = opt.dataset_type_et)
    
    #load the deep learning architecture for the critic and the generator
    print()
    net_d = model.Thoracic(opt.grid_size, pretrained = opt.use_pretrained, calculate_cam = opt.calculate_cam, last_layer_index = opt.last_layer_index).cuda()
    if opt.load_checkpoint_d is not None:
        net_d.load_state_dict(torch.load(opt.load_checkpoint_d))
    #load the optimizer
    optim_d, lr_scheduler_d = init_optimizer(net_d=net_d, opt=opt)
    
    SpecificTraining(output,metric, opt).train(loader_train, loader_val_rad, loader_val_all,
          net_d=net_d, optim_d=optim_d, lr_scheduler_d = lr_scheduler_d)
    

if __name__ == '__main__':
    main()
