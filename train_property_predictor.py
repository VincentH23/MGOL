from utils import *
from models import *
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from scipy.stats import linregress

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
    

parser = argparse.ArgumentParser()
parser.add_argument('--prop', choices=['logp', 'penalized_logp', 'qed', 'sa', 'binding_affinity'], default='binding_affinity')
parser.add_argument('--autodock_executable', type=str, default='autodock_gpu_128wi')
parser.add_argument('--ligands_folder', type=str, default ='ligands')
parser.add_argument('--autodock_results_folder',type=str, default ='outs')
parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')
args = parser.parse_args()

props = {'logp': one_hots_to_logp, 
         'penalized_logp': one_hots_to_penalized_logp, 
         'qed': one_hots_to_qed, 
         'sa': one_hots_to_sa,
         'binding_affinity': lambda x: one_hots_to_affinity(x, args.autodock_executable, args.protein_file, args.ligands_folder, args.autodock_results_folder)}

x, y = generate_training_mols(args.num_mols, props[args.prop])

model = PropertyPredictor(x.shape[1],loss_name='ordering',learning_rate=0.0003)
print(y)
dm = PropDataModule(x[1000:], y[1000:], 8000)
trainer = pl.Trainer(gpus=1, max_epochs=5, logger=pl.loggers.CSVLogger('logs'), enable_checkpointing=False, auto_lr_find=True,callbacks=[EarlyStopping(monitor="val_loss", mode="min")])

# trainer.tune(model, dm)
trainer.fit(model, dm)
model.eval()
model = model.to(device)

# print(f'property predictor trained, correlation of r = {linregress(model(x[:1000].to(device)).detach().cpu().numpy().flatten(), y[:1000].detach().cpu().numpy().flatten()).rvalue}')

# r = np.corrcoef(model(x[:1000].to(device)).detach().cpu().numpy().flatten(),y[:1000].detach().cpu().numpy().flatten())
# print('r : ',r)
if not os.path.exists('property_models'):
    os.mkdir('property_models')
print(f'property_models/{args.prop}.pt')
torch.save(model.state_dict(), f'property_models/{args.prop}.pt')