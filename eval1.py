import argparse
import os
from time import time
import torch
import datetime
import torch.nn as nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
from utils import BinaryDataset, certificate_over_dataset
from tqdm import tqdm
from attack_lib1 import attack_setting
from scipy.stats import norm

# Dataset Setting
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--pair_id', type=int, default=0)
parser.add_argument("--path", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, default=0.5, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, default='./outfile', help="output file")
#parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--fix-sig-smooth', action='store_true', default=False, help='certify with fixed sigma')
parser.add_argument('--poison_r', type=float, default=0.02)
parser.add_argument('--delta', type=float, default=1.0)
parser.add_argument('--atk_method', type=str, default='fourpixel')

# Smoothing Setting
parser.add_argument('--N_m', type=int, default=1000)
parser.add_argument('--dldp_sigma', type=float, default=0.0)
parser.add_argument('--dldp_gnorm', type=float, default=5.0)
parser.add_argument('--iter_sig_tr', type=int, default=1)
parser.add_argument('--iter_sig_ts', type=int, default=1)
parser.add_argument('--iter_sig_after', type=int, default=100)
parser.add_argument('--num_noise_vec', type=int, default=1)
parser.add_argument('--epoch_switch', type=int, default=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args)
    print(args)

    poisoned_train, testloader_benign, testloader_poison, BATCH_SIZE, N_EPOCH, LR, Model= attack_setting(args)
    #testloader_benign = torch.utils.data.DataLoader(testloader_benign, batch_size=BATCH_SIZE)
    #testset = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
    print()
    epoch_switch = args['epoch_switch']
    sigma = args['sigma']
    N_m = args['N_m']
    atk_method = args['atk_method']
    model = Model(gpu='cuda:0')
    dataset = args['dataset']
    #PREFIX = './saved_model/_%s _%s sigma%s N_m%s' % (dataset, atk_method, sigma, N_m)
    #PREFIX = './saved_model/sigma-%s switch%s' % (sigma, epoch_switch)
    #PREFIX = './saved_model/_%s _%s sigma%s' % (dataset, atk_method, sigma)
    PREFIX = './saved_model/_%s _%s sigma%s_epoch_switch%s' % (dataset, atk_method, sigma, epoch_switch)





    # checkpoint = torch.load(path, map_location='cuda')
    # model.load_state_dict(checkpoint['state_dict'])

    # load the base classifier
    # model = get_model(args.path)
    # create the smooothed classifier g
    # smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)
    base_bath_save = 'base_bath_save'
    #sigma_path = base_bath_save + '/_%s _%s sigma%s' % (dataset, atk_method, sigma)
    sigma_path = base_bath_save + '/_%s_%s_sigma%s_iter.pth' % (dataset, atk_method, sigma)

    if not False:
        sigma_test = torch.load(sigma_path)


    outfile = args['outfile']  # args['outfile'] 应该是一个字符串
    sigma = args['sigma']  # args['sigma'] 应该是一个字符串或变量
    # 使用字符串格式化来构建输出文件名
    outfile = '%s%s%.4f%s' % (outfile, dataset, sigma, N_m)

    f = open(outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\tcert_bound_exp\tsigma\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = testloader_benign
    sigma_test = sigma_test.detach().cpu().numpy()
    before_time_total = time()

    pa_exp, pb_exp, is_acc, prediction, labs = certificate_over_dataset(model, testloader_benign, PREFIX, args['N_m'],
                                                                        args['sigma'])

    # print(pa_exp,'pa')
    # print(pb_exp, 'pb')
    print(is_acc)
    after_time = time()
    heof_factor = np.sqrt(np.log(1 / args['alpha']) / 2 / 1000)
    # print(heof_factor,'heof_factor')
    pa = np.maximum(1e-8, pa_exp - heof_factor)
    # print(pa,'pa')
    pb = np.minimum(1 - 1e-8, pb_exp + heof_factor)
    # print(pb,'pb')
    b = norm.ppf(pa) - norm.ppf(pb)
    print(len(b))
    print(len(sigma_test))
    cert_bound = 0.5 * sigma_test * (norm.ppf(pa) - norm.ppf(pb))
    # print(cert_bound,'cert_bound')
    cert_bound_exp = 0.5 * sigma_test * (norm.ppf(pa_exp) - norm.ppf(pb_exp))

    cert_acc = []
    cond_acc = []
    cert_ratio = []
    cert_acc_exp = []
    cond_acc_exp = []
    cert_ratio_exp = []

    rad = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.50)

    for r in rad:
        cert_acc.append(np.logical_and(cert_bound > r, is_acc).mean())
        cond_acc.append(np.logical_and(cert_bound > r, is_acc).sum() / (cert_bound > r).sum())
        cert_ratio.append((cert_bound > r).mean())
        cert_acc_exp.append(np.logical_and(cert_bound_exp > r, is_acc).mean())
        cond_acc_exp.append(np.logical_and(cert_bound_exp > r, is_acc).sum() / (cert_bound_exp > r).sum())
        cert_ratio_exp.append((cert_bound_exp > r).mean())
    print("Certified Radius:", ' / '.join([str(r) for r in rad]))
    print("Cert acc:", ' / '.join(['%.5f' % x for x in cert_acc]))
    print("Cond acc:", ' / '.join(['%.5f' % x for x in cond_acc]))
    print("Cert ratio:", ' / '.join(['%.5f' % x for x in cert_ratio]))
    print("Expected Cert acc:", ' / '.join(['%.5f' % x for x in cert_acc_exp]))
    print("Expected Cond acc:", ' / '.join(['%.5f' % x for x in cond_acc_exp]))
    print("Expected Cert ratio:", ' / '.join(['%.5f' % x for x in cert_ratio_exp]))

    correct = is_acc.astype(int)
    # Now you can use the same exac
    before_time = time()
    # certify the prediction of g around x
    after_time = time()
    for i in range(2000):
        print("{}\t{}\t{}\t{:.3}\t{:}\t{:.3}\t{:.3}\t{}".format(
            i, labs[i], prediction[i], cert_bound[i], correct[i], cert_bound_exp[i], sigma_test[i], 1), file=f,
            flush=True)

    f.close()
