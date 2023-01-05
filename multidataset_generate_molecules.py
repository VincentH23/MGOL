from utils import *
from models import *
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from scipy.stats import linregress
# import traceback
import copy
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




try:
    vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
except NameError:
    raise Exception('No dm.pkl found, please run preprocess_data.py first')
vae.load_state_dict(torch.load('vae.pt'))
vae.eval()


def delta_g_to_kd_np(delta_g) :
    kd_tensor = np.exp(delta_g/(0.00198720425864083 * 298.15))
    return kd_tensor
    

def generate_z_vector(name, num_mols):
    z = torch.randn((num_mols, 1024), device=device, requires_grad=True)
    torch.save(z,name)
    return 




def get_prop(prop, x):
    return torch.tensor(props[prop](x), device=device).unsqueeze(1).float()



def get_optimized_z(args, loss_name, z, weights, num_dataset, models_folder, num_steps =10):
    
    models = []
    num_mols = z.shape[0]
    prop = args.protein_file.split('/')[0] if args.prop =="binding_affinity" else args.prop
    for prop_name in weights:
        models.append(PropertyPredictor(dm.dataset.max_len * len(dm.dataset.symbol_to_idx), loss_name=loss_name))
        prop_ = args.protein_file.split('/')[0] if args.prop =="binding_affinity" else args.prop
        if loss_name =='mse':
            file_name = f'{prop_}_ref_{num_dataset+1}'
        else :
            file_name = f'{prop_}_ordering_{num_dataset+1}'
            
        
        models[-1].load_state_dict(torch.load(f'{models_folder}/{file_name}.pt'))
        print('load model : ', f'{models_folder}/{file_name}.pt')
        models[-1] = models[-1].to(device)
        

    optimizer = optim.Adam([z], lr=args.lr)
    losses = []
    for epoch in tqdm(range(num_steps), desc='generating molecules'):
        optimizer.zero_grad()
        loss = 0
        probs = torch.exp(vae.decode(z))
        for i, model in enumerate(models):
            out = model(probs)
            loss += torch.sum(out) * list(weights.values())[i]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return z
    
    
def generate_molecules(args, loss_name, z, num_dataset , models_folder, num_steps = 10 ):
    weights = {args.prop: (1 if args.prop in ('binding_affinity') else -1)}
    z_optimized = get_optimized_z(args, loss_name, z, weights, num_dataset, models_folder, num_steps = num_steps)
    with torch.no_grad():
        x = torch.exp(vae.decode(z_optimized))
    smiles = [one_hot_to_smiles(hot) for hot in x]
    prop = get_prop(args.prop, x).detach().cpu().numpy().flatten()

    if args.prop in ('binding_affinity'):
        print('top 3')
        L = []
        for i in np.argpartition(prop, args.top_k)[:args.top_k]:
            if args.prop == 'binding_affinity':
                print(delta_g_to_kd(prop[i]), smiles[i])
                print(prop[i], smiles[i])
                L.append(delta_g_to_kd(prop[i]))
        
        L = sorted(L)
        
        
        print('')
        print('average ')
        print(f' average free energy {prop.mean()}')
        AVG_kd = delta_g_to_kd_np(prop).mean()
        print(f' average Kd {AVG_kd}')
        print('')
        print('average top 100')
        top100 = prop[np.argpartition(prop, 100)[:100]]
        print(f' top 100 average free energy {top100.mean()}')
        AVG_100_kd = delta_g_to_kd_np(top100).mean()
        print(f' top 100 average Kd {AVG_100_kd}')
        
        print('average top 10')
        top10 = prop[np.argpartition(prop, 10)[:10]]
        print(f' top 10 average free energy {top10.mean()}')
        AVG_10_kd = delta_g_to_kd_np(top10).mean()
        print(f' top 100 average Kd {AVG_10_kd}')
        
        
        if loss_name=='mse':
            for i in range(3):
                results_ref[i+1].append(L[i])
            results_ref['top100'].append(AVG_100_kd.mean())
            results_ref['top10'].append(AVG_10_kd.mean())
            results_ref['avg'].append(AVG_kd)
            results_ref['avg free energy'].append(prop.mean())
            results_ref['top 100 free energy'].append(top100.mean())
            results_ref['top 10 free energy'].append(top10.mean())
            
            
            
        else:
            for i in range(3):
                results_ordering[i+1].append(L[i])
            results_ordering['top100'].append(AVG_100_kd.mean())
            results_ordering['top10'].append(AVG_10_kd.mean())
            results_ordering['avg'].append(AVG_kd)
            results_ordering['avg free energy'].append(prop.mean())
            results_ordering['top 100 free energy'].append(top100.mean())
            results_ordering['top 10 free energy'].append(top10.mean())
            
                
                
    else:
        
        L = []
        for i in np.argpartition(prop, -args.top_k)[-args.top_k:]:
            print(prop[i], smiles[i])
            L.append(prop[i])
        L = sorted(L,reverse=True)
    
        print('')
        print('average ')
        print(f'average plogp : {prop.mean()}')  
                
        print('')
        print('average top 100')
        top100 = prop[np.argpartition(prop, -100)[-100:]]
        print(f' top 100 average penalized logp : {top100.mean()}')
        top10 = prop[np.argpartition(prop, -10)[-10:]]
        print(f' top 10 average penalized logp : {top10.mean()}')
        if loss_name=='mse':
            for i in range(3):
                results_ref[i+1].append(L[i])
            results_ref['top100'].append(top100.mean())
            results_ref['top10'].append(top10.mean())
            results_ref['avg'].append(prop.mean())
        else:
            for i in range(3):
                results_ordering[i+1].append(L[i])
            results_ordering['top100'].append(top100.mean())
            results_ordering['top10'].append(top10.mean())
            results_ordering['avg'].append(prop.mean())



#######parser
parser = argparse.ArgumentParser()
parser.add_argument('--prop', choices=['penalized_logp','binding_affinity'], default='binding_affinity')
parser.add_argument('--num_mols', type=int, default=10000)
parser.add_argument('--autodock_executable', type=str, default='autodock_gpu_128wi')
parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--optim_steps', type=int, default=10)
parser.add_argument('--n_run_per_exp', type=int, default= 3)
parser.add_argument('--n_dataset',type=int, default=5)
parser.add_argument('--ligands_folder', type=str, default ='ligands')
parser.add_argument('--autodock_results_folder',type=str, default ='outs')
parser.add_argument('--models_folder', type=str, default = 'property_models')
parser.add_argument('--lr', type = float, default =0.1)
parser.add_argument('--begin_run', type = int, default =1)
parser.add_argument('--begin_model', type = int, default =1)
args = parser.parse_args()

# try :
props = {'penalized_logp': one_hots_to_penalized_logp, 
         'binding_affinity': lambda x: one_hots_to_affinity(x, args.autodock_executable, args.protein_file, args.ligands_folder, args.autodock_results_folder)}


results_ref = {1 : [],
               2 : [],
               3 : [],
               'top100' : [],
               'top10':[],
               'avg' : []}   

results_ordering = {1 : [],
               2 : [],
               3 : [],
               'top100' : [],
               'top10':[],
               'avg' : []}
if args.prop == 'binding_affinity' :
    results_ref['avg free energy'] = []
    results_ordering['avg free energy'] = []
    results_ref['top 100 free energy'] = []
    results_ordering['top 100 free energy'] = []
    results_ref['top 10 free energy'] = []
    results_ordering['top 10 free energy'] = []


if not os.path.exists(args.ligands_folder):
    os.mkdir(args.ligands_folder)

if not os.path.exists('results'):
    os.mkdir('results')

prop = args.protein_file.split('/')[0] if args.prop=="binding_affinity" else args.prop
for run in range(args.begin_run-1, args.n_run_per_exp):
    print(f'===========================================Run {run+1}==============================================')

    for num_dataset in range(args.begin_model-1,args.n_dataset):
        print(f'############################################### model {num_dataset + 1 } ###########################################################\n')
        if not os.path.exists(f'z_vector/dataset_{num_dataset+1}'):
            os.mkdir(f'z_vector/dataset_{num_dataset+1}')
        if not os.path.exists(f'z_vector/dataset_{num_dataset+1}/{args.num_mols}_{prop}_z_run_{run+1}.pt'):
            generate_z_vector(f'z_vector/dataset_{num_dataset+1}/{args.num_mols}_{prop}_z_run_{run+1}.pt', args.num_mols)

        print("=================================   Reference  =======================================")
        z = torch.load(f'z_vector/dataset_{num_dataset+1}/{args.num_mols}_{prop}_z_run_{run+1}.pt')
        z_copy = copy.deepcopy(z)
        print(f' z from z_vector/dataset_{num_dataset+1}/{args.num_mols}_{prop}_z_run_{run+1}.pt')
        generate_molecules(args, 'mse', z, num_dataset , args.models_folder )

        print("==================================  Ordering =========================================")
        z = torch.load(f'z_vector/dataset_{num_dataset+1}/{args.num_mols}_{prop}_z_run_{run+1}.pt')
        assert torch.sum(z!=z_copy).item() == 0 
        print(f' z from z_vector/dataset_{num_dataset+1}/{args.num_mols}_{prop}_z_run_{run+1}.pt')
        generate_molecules(args, 'ordering', z, num_dataset  , args.models_folder )
    results_ref_pd = pd.DataFrame(data = results_ref )
    results_ref_pd.to_csv(f'{prop}_ref3.csv')
    results_ordering_pd = pd.DataFrame(data = results_ordering )
    results_ordering_pd.to_csv(f'{prop}_ordering3.csv')
            
# except:
#         with open("exceptions.log", "a") as logfile:
#             traceback.print_exc(file=logfile)
#         raise
               
    

        

        
      