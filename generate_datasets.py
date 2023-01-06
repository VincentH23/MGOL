from utils import *
from models import *
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from scipy.stats import linregress


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
except NameError:
    raise Exception('No dm.pkl found, please run preprocess_data.py first')
vae.load_state_dict(torch.load('vae.pt'))
vae.eval()


def generate_training_mols(num_mols, prop_func):
    with torch.no_grad():
        z = torch.randn((num_mols, 1024), device=device)
        x = torch.exp(vae.decode(z))
        y = torch.tensor(prop_func(x), device=device).unsqueeze(1).float()
    return x, y


def get_prop(prop, x):
    return torch.tensor(props[prop](x), device=device).unsqueeze(1).float()



#######parser
parser = argparse.ArgumentParser()
parser.add_argument('--prop', choices=['logp', 'penalized_logp', 'qed', 'sa', 'binding_affinity'], default='binding_affinity')
parser.add_argument('--num_mols', type=int, default=10000)
parser.add_argument('--autodock_executable', type=str, default='autodock_gpu_128wi')
parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')

parser.add_argument('--n_dataset', type=int, default= 5)
parser.add_argument('--ligands_folder', type=str, default ='ligands')
parser.add_argument('--autodock_results_folder',type=str, default ='outs')
args = parser.parse_args()

props = {'logp': one_hots_to_logp, 
         'penalized_logp': one_hots_to_penalized_logp, 
         'qed': one_hots_to_qed, 
         'sa': one_hots_to_sa,
         'binding_affinity': lambda x: one_hots_to_affinity(x, args.autodock_executable, args.protein_file,  args.ligands_folder, args.autodock_results_folder)}

if not os.path.exists(args.ligands_folder):
    os.mkdir(args.ligands_folder)
    
for num in range(args.n_dataset):
    
    
    
    print(f'generate dataset {num+1}')
    ######## Training ##########

    date_time = str(datetime.now())
    
    if args.prop =='binding_affinity':
        X = []
        Y = []
        n_tot = 0
        num_mols_to_generate = args.num_mols
        while n_tot < args.num_mols :
            x, y = generate_training_mols(num_mols_to_generate + 100, props[args.prop])
        
            mask = y!=0
      
            x_filtered = x[mask.view(-1,),:]
            y_filtered = y[mask.view(-1,),:]
            if num_mols_to_generate < y_filtered.shape[0]:
                x_filtered = x_filtered[:num_mols_to_generate,:]
                y_filtered = y_filtered[:num_mols_to_generate,:]
            X.append(x_filtered)
            Y.append(y_filtered)
            
            n_tot += x_filtered.shape[0]
            num_mols_to_generate = args.num_mols - n_tot
        
        x = torch.cat(X,dim=0)
        y = torch.cat(Y,dim=0)
    else :
        x, y = generate_training_mols(args.num_mols, props[args.prop])
    data_folder = args.prop if args.prop!='binding_affinity' else args.protein_file.split('/')[0]
    if not os.path.exists(f'data/{data_folder}'):
        os.mkdir(f'data/{data_folder}')
    torch.save(x,f'data/{data_folder}/{args.num_mols}_x_{num+1}.pt')
    torch.save(y,f'data/{data_folder}/{args.num_mols}_y_{num+1}.pt')
    