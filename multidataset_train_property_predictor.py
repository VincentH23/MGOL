from utils import *
from models import *
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from scipy.stats import linregress
import traceback
from torchmetrics import R2Score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def computeR2(preds,target) :
    preds = preds.view(-1,)
    target = target.view(-1,)
    SSR = ((preds-target)**2).sum()
    SStot = ((target - target.mean())**2).sum()
    return 1 - (SSR/SStot)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
except NameError:
    raise Exception('No dm.pkl found, please run preprocess_data.py first')
vae.load_state_dict(torch.load('vae.pt'))
vae.eval()


#######parser
parser = argparse.ArgumentParser()
parser.add_argument('--prop', choices=['penalized_logp',  'binding_affinity'], default='binding_affinity')
parser.add_argument('--autodock_executable', type=str, default='autodock_gpu_128wi')
parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')
parser.add_argument('--n_dataset', type=int, default= 5)
parser.add_argument('--begin', type=int, default= 1)
parser.add_argument('--batch_size_ordering', type=int, default = 8100)
parser.add_argument('--batch_size_ref', type=int, default = 100)
parser.add_argument('--test_set_size', type=int, default = 1000)
parser.add_argument('--models_folder', type=str, default = 'property_models')
parser.add_argument('--dataset_folder', type=str, default ='data')
parser.add_argument('--c', type=float, default =2)
parser.add_argument('--epochs', type=int, default= 20)
args = parser.parse_args()


    
protein_name = args.protein_file.split('/')[0]
    
for num in range(args.begin-1,args.n_dataset):
    
    ######## Training ##########

    
    print(f'######################################################  Training on dataset {num+1} ##################################################')
    
    print('loading dataset')
    
    if args.prop == 'binding_affinity':
        folder = f'{args.dataset_folder}/{protein_name}'
        print(f'dataset : {folder}/10000_x_{num+1}.pt , {folder}/10000_y_{num+1}.pt')
        x = torch.load(f'{folder}/10000_x_{num+1}.pt')
        y = torch.load(f'{folder}/10000_y_{num+1}.pt')
        
    else : 
        folder = f'{args.dataset_folder}/penalized_logp'
        print(f'dataset : {folder}/100000_x_{num+1}.pt , {folder}/100000_y_{num+1}.pt')
        x = torch.load(f'{folder}/100000_x_{num+1}.pt')
        y = torch.load(f'{folder}/100000_y_{num+1}.pt')
    print('training ref')
    dataset_size = x.shape[0]
    
    ref_model = PropertyPredictor(x.shape[1], loss_name = 'mse', learning_rate = 0.0001)
    
    dm = PropDataModule(x[args.test_set_size:], y[args.test_set_size:], args.batch_size_ref)
    trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, logger=pl.loggers.CSVLogger(f'logs/{args.prop}/ref'), enable_checkpointing=False, auto_lr_find=True, progress_bar_refresh_rate=0, callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
#     trainer.tune(ref_model, dm)
    trainer.fit(ref_model, dm)
    ref_model.eval()
    ref_model = ref_model.to(device)
    metric = R2Score()
    print(f'property predictor trained, correlation of r = {linregress(ref_model(x[:args.test_set_size].to(device)).detach().cpu().numpy().flatten(), y[:args.test_set_size].detach().cpu().numpy().flatten()).rvalue}')
    print('shape :', ref_model(x[:args.test_set_size].to(device)).detach().cpu().shape)
    print(f'r2 = {computeR2(ref_model(x[:args.test_set_size].to(device)).detach().cpu(),y[:args.test_set_size].detach().cpu())}')
    print(f'R2 from torch : {metric(ref_model(x[:args.test_set_size].to(device)).detach().cpu(),y[:args.test_set_size].detach().cpu())}')


    if not os.path.exists(f'{args.models_folder}'):
        os.mkdir(f'{args.models_folder}')
    prop_ = args.protein_file.split('/')[0] if args.prop =="binding_affinity" else args.prop
    torch.save(ref_model.state_dict(), f'{args.models_folder}/{prop_}_ref_{num+1}.pt')
    
    print('training ordering')

    ordering_model = PropertyPredictor(x.shape[1], loss_name = 'ordering', c=args.c, learning_rate = 0.0001)
    trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, logger=pl.loggers.CSVLogger(f'logs/{args.prop}/ordering'), enable_checkpointing=False, auto_lr_find=True, progress_bar_refresh_rate=0, callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
#     trainer.tune(ordering_model, dm)
    trainer.fit(ordering_model, dm)
    ordering_model.eval()
    ordering_model = ordering_model.to(device)

    
    torch.save(ordering_model.state_dict(), f'{args.models_folder}/{prop_}_ordering_{num+1}.pt')

        
