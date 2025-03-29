import os
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
from time import time
from tqdm import tqdm
from scipy.stats import norm
from datetime import datetime
import torchvision.transforms as transforms
from dataloader import get_dataset
from modelloader import get_model
from hashlib import sha256
from utils import BinaryDataset

# Dataset Setting
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument("--path", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, default=0.5, help="noise hyperparameter")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--fix-sig-smooth', action='store_true', default=False, help='certify with fixed sigma')
parser.add_argument('--wm_rate', type=float, default=0.02)
parser.add_argument('--delta', type=float, default=1.0)
parser.add_argument('--wm_shape', type=str, default='fourpixel')

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

def main():
    args = parser.parse_args()
    args = vars(args)
    print(args)

    model = get_model(args, gpu=True)
    model_path = f'{args.dataset}/{args.wm_shape}/{args.sigma}'
    wm_trainset, testloader_benign, testloader_watermark, BATCH_SIZE, N_EPOCH, LR= get_dataset(args)

    sigma_path = f'./sigma/{args.dataset}/{args.wm_shape}/{args.sigma}.pth'
    sigma_test = torch.load(sigma_path)
    sigma_test = sigma_test.detach().cpu().numpy()

    print("idx\tlabel\tpredict\tradius\tcorrect\tcert_bound_exp\tsigma\ttime", file=f, flush=True)

    pa_benign_exp, pb_benign_exp, is_benign_acc, prediction_benign, labs_benign = certificate_over_dataset(model, testloader_benign, model_path, args.N_m, args.sigma)
    #
    heof_factor = np.sqrt(np.log(1 / args['alpha']) / 2 / 1000)
    pa_benign = np.maximum(1e-8, pa_benign_exp - heof_factor)
    pb_benign = np.minimum(1 - 1e-8, pb_benign_exp + heof_factor)
    b = norm.ppf(pa_benign) - norm.ppf(pb_benign)
    print(len(b))
    print(len(sigma_test))
    cert_bound_benign = 0.5 * sigma_test * (norm.ppf(pa_benign) - norm.ppf(pb_benign))
    cert_bound_benign_exp = 0.5 * sigma_test * (norm.ppf(pa_benign_exp) - norm.ppf(pb_benign_exp))

    pa_watermark_exp, pb_watermark_exp, is_watermark_acc, prediction_watermark, labs_watermark = certificate_over_dataset(model, testloader_watermark, model_path, args.N_m, args.sigma)
    pa_watermark = np.maximum(1e-8, pa_watermark_exp - heof_factor)
    pb_watermark = np.minimum(1 - 1e-8, pb_watermark_exp + heof_factor)
    b = norm.ppf(pa_watermark) - norm.ppf(pb_watermark)
    print(len(b))
    print(len(sigma_test))
    cert_bound_watermark = 0.5 * sigma_test * (norm.ppf(pa_watermark) - norm.ppf(pb_watermark))
    cert_bound_watermark_exp = 0.5 * sigma_test * (norm.ppf(pa_watermark_exp) - norm.ppf(pb_watermark_exp))

    cert_acc = []
    cert_watermark_acc = []
    cond_acc = []
    cert_ratio = []
    cert_acc_exp = []
    cert_watermark_acc_exp = []
    cond_acc_exp = []
    cert_ratio_exp = []

    rad = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.50)

    for r in rad:
        cert_acc.append(np.logical_and(cert_bound_benign > r, is_benign_acc).mean())
        cert_watermark_acc.append(np.logical_and(cert_bound_watermark > r, is_watermark_acc).mean())
        cond_acc.append(np.logical_and(cert_bound_benign > r, is_benign_acc).sum() / (cert_bound_benign > r).sum())
        cert_ratio.append((cert_bound_benign > r).mean())
        cert_acc_exp.append(np.logical_and(cert_bound_benign_exp > r, is_benign_acc).mean())
        cert_watermark_acc_exp.append(np.logical_and(cert_bound_watermark_exp > r, is_watermark_acc).mean())
        cond_acc_exp.append(np.logical_and(cert_bound_benign_exp > r, is_benign_acc).sum() / (cert_bound_benign_exp > r).sum())
        cert_ratio_exp.append((cert_bound_benign_exp > r).mean())
    print("Certified Radius:", ' / '.join([str(r) for r in rad]))
    print("Cert acc:", ' / '.join(['%.5f' % x for x in cert_acc]))
    print("Cert wm acc:", ' / '.join(['%.5f' % x for x in cert_watermark_acc]))
    print("Cert acc:", ' / '.join(['%.5f' % x for x in cert_acc]))
    print("Cond acc:", ' / '.join(['%.5f' % x for x in cond_acc]))
    print("Cert ratio:", ' / '.join(['%.5f' % x for x in cert_ratio]))
    print("Expected Cert acc:", ' / '.join(['%.5f' % x for x in cert_acc_exp]))
    print("Expected Cert wm acc:", ' / '.join(['%.5f' % x for x in cert_watermark_acc_exp]))
    print("Expected Cond acc:", ' / '.join(['%.5f' % x for x in cond_acc_exp]))
    print("Expected Cert ratio:", ' / '.join(['%.5f' % x for x in cert_ratio_exp]))



def certificate_over_dataset(model, dataloader, model_path, N_m, sigma):
    model_preds = []
    labs = []

    for _ in tqdm(range(N_m)):
        model.load_state_dict(torch.load(model_path+'/smoothed_%d.model'%_))
        hashval = int(sha256(open(model_path+'/smoothed_%d.model'%_, 'rb').read()).hexdigest(), 16) % (2**32)
        model.fix_pert(sigma=sigma, hash_num=hashval)
        all_pred = np.zeros((0,2))
        for x_in, y_in, idx in dataloader:
            pred = torch.sigmoid(model(x_in).squeeze(1)).detach().cpu().numpy()
            pred = np.stack((1-pred, pred), axis=1)
            if (_ == 0):
                labs = labs + list(y_in.numpy())
            all_pred = np.concatenate([all_pred, pred], axis=0)
        model_preds.append(all_pred)
        model.unfix_pert()

    gx = np.array(model_preds).mean(0)
    labs = np.array(labs)
    pa = gx.max(1)
    pred_c = gx.argmax(1)

    gx[np.arange(len(pred_c)), pred_c] = -1
    pb = gx.max(1)
    is_acc = (pred_c==labs)
    return pa, pb, is_acc, pred_c, labs

if __name__ == "__main__":
    main()