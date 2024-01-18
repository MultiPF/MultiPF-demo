import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')
    parser.add_argument('--prefix', default='', type=str, help='model prefix')

    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
                        help='path to pretrained checkpoint')
    parser.add_argument('--auto_resume', action='store_true', help='if the log folder includes a checkpoint, automatically resume')

    # data-related
    parser.add_argument('--datadir', type=str,  metavar='DIR', help='path to dataset file list')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--train_input_size', type=int, metavar='N', help='input image size')
    parser.add_argument('--num_train_cls', type=int, default=100, help='input image size')
    parser.add_argument('--test_input_size', type=int, metavar='N', help='input image size')
    


    # logging
    parser.add_argument('--output_dir', default='', type=str, help='log path')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency to print the log during the training')
    parser.add_argument('--val_freq_in_epoch', type=int, default=-1,
                        help='frequency to validate the model during the training')

    # for testing and validation
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # cfg file
    parser.add_argument('--config_file', dest='config_file', type=str, help='network config file path')
    parser.add_argument('--dataset_config_file', dest='dataset_config_file', type=str, help='network config file path')

    parser.add_argument('--n_ctx', dest="n_ctx", type=int, help='the length of each prompt')
    
    parser.add_argument('--seed', dest="seed", type=int, help='seed')


    parser.add_argument('--lr', dest="lr", type=float, help='the learning rate')
    parser.add_argument('--mlplr', dest="mlplr", type=float, help='the learning rate of count predictor')
    parser.add_argument('--device_id', dest="device_id", type=int, help='gpu id')
    parser.add_argument('--beta', dest="beta", type=float, default=0.1,
                        help='weights of count loss')
    parser.add_argument('--gamma', dest="gamma", type=float, default=0.1,
                        help='weights of entropy loss')

    parser.add_argument('--csc', dest='csc', action='store_true',
                        help='specify the csc')

    parser.add_argument('--logit_scale', dest="logit_scale", type=float, default=100.,
                        help='the logit scale for clip logits')
    parser.add_argument('--pool_size', dest="pool_size", type=int, default=8, help='the number of prompts')
    
    parser.add_argument('--prompt_key_init', dest='prompt_key_init', type=str, default='uniform', 
                        help='the method of key initial, uniform or zero')

    parser.add_argument('--stop_epochs', dest="stop_epochs", type=int,
                        help='the stop epochs')

    parser.add_argument('--max_epochs', dest="max_epochs", type=int,
                        help='the max epochs')

    parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='specify if finetuning the backbone')

    parser.add_argument('--finetune_backbone', dest='finetune_backbone', action='store_true',
                        help='specify if finetuning the backbone')

    parser.add_argument('--finetune_attn', dest='finetune_attn', action='store_true',
                        help='specify if finetuning the backbone')

    parser.add_argument('--finetune_text', dest='finetune_text', action='store_true',
                        help='specify if finetuning the text')

    parser.add_argument('--base_lr_mult', dest='base_lr_mult',  type=float,
                        help='specify if finetuning the backbone')

    parser.add_argument('--backbone_lr_mult', dest='backbone_lr_mult', type=float,
                        help='specify if finetuning the backbone')

    parser.add_argument('--text_lr_mult', dest='text_lr_mult', type=float,
                        help='specify if finetuning the backbone')

    parser.add_argument('--attn_lr_mult', dest='attn_lr_mult', type=float,
                        help='specify if finetuning the backbone')

    parser.add_argument('--val_every_n_epochs', dest='val_every_n_epochs', type=int, default=1,
                        help='specify if finetuning the backbone')

    parser.add_argument('--warmup_epochs', dest='warmup_epochs', type=int, default=1,
                        help='warm up epochs')

    parser.add_argument('--cln', dest='cln', type=int, default=0,
                        help='use conditional layer normalization or not')
    parser.add_argument('--learnable_alpha', dest='learnable_alpha', type=int, default=0,
                        help='learnable_alpha')

    parser.add_argument('--labels_file', dest='labels_file', type=str, default='/labels.json',
                        help='Which label file to use. ')

    parser.add_argument('--uncertainty_weight', dest='uncertainty_weight', type=int, default=0,
                        help='Whether to use uncertainty_weight when calculating loss.')


    parser.add_argument('--use_pfa', dest='use_pfa', type=int, default=0,
                        help='Whether to use pfa to perform data enhancement on the support set')
    parser.add_argument('--pfa_layer_num', dest='pfa_layer_num', type=int, default=1,
                        help='pfa_layer_num')
    parser.add_argument('--pfa_layer_dim', dest='pfa_layer_dim', type=int, default=8092,
                        help='pfa_layer_dim')
    parser.add_argument('--pfa_dropout', dest='pfa_dropout', type=float, default=0,
                        help='dropout')
    parser.add_argument('--pfa_lr', dest='pfa_lr', type=float, default=0.02,
                        help='pfa_lr')
    parser.add_argument('--pfa_3d_key_smoothing', dest='pfa_3d_key_smoothing', type=int, default=0,
                        help='pfa_3d_key_smoothing')
    parser.add_argument('--use_img_proto', dest='use_img_proto', type=int, default=1,
                        help='use_img_proto')
    parser.add_argument('--use_text_proto', dest='use_text_proto', type=int, default=1,
                        help='use_text_proto')


    return parser