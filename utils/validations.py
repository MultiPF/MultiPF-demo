import sys
sys.path.insert(0, '../')
import torch
import time
from utils.helper import AverageMeter, mAP, calc_F1
from torch.cuda.amp import autocast
from utils.asymmetric_loss import ProtypicalLoss
from utils.trainers import obtain_img_labels

def validate(data_loader, model, args, cfg, split):

    batch_time = AverageMeter()
   
    miAP_batches = AverageMeter()
    maAP_batches = AverageMeter()

    losses = AverageMeter()
    acc_batches = AverageMeter()
    cacc_batches = AverageMeter()
    macro_p_batches, macro_r_batches, macro_f_batches = AverageMeter(), AverageMeter(), AverageMeter()
    micro_p_batches, micro_r_batches, micro_f_batches = AverageMeter(), AverageMeter(), AverageMeter()


    
    # switch to evaluate mode
    model.eval()

    criterion = ProtypicalLoss(cfg.TRAINER.DEVICEID, cfg.TRAINER.BETA, cfg.TRAINER.GAMMA, cfg.LEARN_ALPHA, cfg.UNCERTAIN_WEIGHT,
                               cfg.USE_PFA, cfg.PFA_3D_KEY_SMOOTHING,
                               cfg.USE_IMG_PROTO, cfg.USE_TEXT_PROTO, split)
    
    root = cfg.DATASET.ROOT 
    

    
    with torch.no_grad():
        end = time.time()

        for i, (batch) in enumerate(data_loader):
    
            support, query, classnames = batch
           
            support_img, query_img, support_ids, query_ids = obtain_img_labels(root, split, support, query, classnames, args.input_size)
            
            support_size = support_img.shape[0]
            images = torch.cat((support_img, query_img), dim=0)
            target = torch.cat((support_ids, query_ids), dim=0)
           
            if torch.cuda.is_available():
                device = torch.device("cuda", cfg.TRAINER.DEVICEID)
            else:
                device = torch.device("cpu")
            images = images.to(device)
            target = target.to(device)
            
            # compute output
            with autocast():
                image_features, text_features, count_model, weights, alpha, loss_weight, pfa_model, class_embs, image_features_3d, image_features_k = model(classnames, images)



            count_outputs = count_model(image_features)
            if cfg.USE_PFA:
                image_features_k_abs = None

                imga_s = []
                imgb_s = []
                target_a_U_bs = []

                image_features_support = image_features_3d[:support_size].float()
                if cfg.PFA_3D_KEY_SMOOTHING:
                    image_features_k_support = image_features_k[:support_size].float()  # n,50,2048
                    image_features_k_abs = []

                for i in range(support_size - 1):
                    for j in range(i + 1, support_size):
                        img_a = image_features_support[i]
                        img_b = image_features_support[j]
                        target_a = target[i]
                        target_b = target[j]
                        target_a_U_b = target_a | target_b
                        imga_s.append(img_a)
                        imgb_s.append(img_b)
                        target_a_U_bs.append(target_a_U_b)
                        if cfg.PFA_3D_KEY_SMOOTHING:
                            k_a = image_features_k_support[i]
                            k_b = image_features_k_support[j]
                            image_features_k_abs.append(torch.cat((k_a, k_b), dim=-1))  # 1,50,4096
                img_as = torch.stack(imga_s, dim=0)
                img_bs = torch.stack(imgb_s, dim=0)
                a_U_b, b_U_a = pfa_model(img_as, img_bs)
                pfa_tensor_u = torch.stack([a_U_b, b_U_a], dim=0)
                target_a_U_bs = torch.stack(target_a_U_bs, dim=0)
                if cfg.PFA_3D_KEY_SMOOTHING:
                    image_features_k_abs = torch.stack(image_features_k_abs, dim=0)
                print(pfa_tensor_u.size())


            else:
                pfa_tensor_u = None
                target_a_U_bs = None
                image_features_k_abs = None

            loss, acc, c_acc, mi_ap, micro_p, micro_r, micro_f, \
            ma_ap, macro_p, macro_r, macro_f = criterion(image_features, image_features_3d, text_features, count_outputs,
                                                         weights,
                                                         support_size, target, class_embs, classnames, alpha, loss_weight,
                                                         pfa_tensor_u, target_a_U_bs, image_features_k_abs)
            losses.update(loss.item(), 1)

            print(f"Loss: {loss}, Acc: {acc}, Count acc: {c_acc}")
            print(f"Macro mAP:{ma_ap}, Macro P: {macro_p}, Macro R: {macro_r}, Macro F1: {macro_f}")
            print(f"Micro mAP:{mi_ap}, Micro P: {micro_p}, Micro R: {micro_r}, Micro F1: {micro_f}")

            miAP_batches.update(mi_ap)
            maAP_batches.update(ma_ap)
            acc_batches.update(acc)
            cacc_batches.update(c_acc)
            micro_p_batches.update(micro_p)
            micro_r_batches.update(micro_r)
            micro_f_batches.update(micro_f)
            macro_p_batches.update(macro_p)
            macro_r_batches.update(macro_r)
            macro_f_batches.update(macro_f)
            
            batch_time.update(time.time()-end)
            end = time.time()
       
    # mAP_score = mAP_batches.avg
    miAP_score = miAP_batches.avg
    maAP_score = maAP_batches.avg
    acc = acc_batches.avg
    c_acc = cacc_batches.avg
    macro_p = macro_p_batches.avg
    macro_r = macro_r_batches.avg
    macro_f = macro_f_batches.avg
    micro_p = micro_p_batches.avg
    micro_r = micro_r_batches.avg
    micro_f = micro_f_batches.avg

    
    # torch.cuda.empty_cache()
    return acc, c_acc, miAP_score, micro_p, micro_r, micro_f, \
            maAP_score, macro_p, macro_r, macro_f


def get_object_names(classnames, target):
    objects = []
    for idx, t in enumerate(target):
        if t == 1:
            objects.append(classnames[idx])
    return objects


def validate_zsl(data_loader, model, args, cls_id):
    batch_time = AverageMeter()

    Softmax = torch.nn.Softmax(dim=1)
    Sig = torch.nn.Sigmoid()

    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []
    output_idxs = []
    with torch.no_grad():
        end = time.time()
        for i,   (images, target) in enumerate(data_loader):
            target = target.max(dim=1)[0]
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            images = images.to(device)

            # compute output
            with autocast():
                output = model(images, cls_id)
            target = target[:, cls_id]
            if output.dim() == 3:
                output = Softmax(output).cpu()[:, 1]
            else:
                output = Sig(output).cpu()

            preds.append(output.cpu())
            targets.append(target.cpu())
            output_idx = output.argsort(dim=-1, descending=True)
            output_idxs.append(output_idx)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(data_loader), batch_time=batch_time),
                    flush=True)

        precision_3, recall_3, F1_3 = calc_F1(torch.cat(targets, dim=0).cpu().numpy(), torch.cat(output_idxs, dim=0).cpu().numpy(), args.top_k,
                                              num_classes=len(cls_id))

        if F1_3 != F1_3:
            F1_3 = 0

        mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())

    torch.cuda.empty_cache()
    return precision_3, recall_3, F1_3, mAP_score
