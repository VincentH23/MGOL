import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import math

class VAE(pl.LightningModule):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(max_len * embedding_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, max_len * vocab_len))
        
    def encode(self, x):
        x = self.encoder(self.embedding(x).view((len(x), -1))).view((-1, 2, self.latent_dim))
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, x):
        return F.log_softmax(self.decoder(x).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), z, mu, log_var
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return {'optimizer': optimizer}
    
    def loss_function(self, pred, target, mu, log_var, batch_size, p):
        nll = F.nll_loss(pred, target)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (batch_size * pred.shape[1])
        return (1 - p) * nll + p * kld, nll, kld
    
    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(train_batch)
        p = 0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), train_batch.flatten(), mu, log_var, len(train_batch), p)
        self.log('train_loss', loss)
        self.log('train_nll', nll)
        self.log('train_kld', kld)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(val_batch)
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), val_batch.flatten(), mu, log_var, len(val_batch), 0.5)
        self.log('val_loss', loss)
        self.log('val_nll', nll)
        self.log('val_kld', kld)
        self.log('val_mu', torch.mean(mu))
        self.log('val_logvar', torch.mean(log_var))
        return loss

def get_ordering_loss(c):
    
    def ordering_loss(pred, real):
        device = pred.device
        batch_size = pred.shape[0]
        sign = torch.sign(torch.repeat_interleave(real, batch_size,0)-real.repeat(batch_size,1))
        diff = torch.repeat_interleave(pred, batch_size,0)-pred.repeat(batch_size,1)

        ##### supress self diff 
        mask = torch.ones(batch_size*batch_size,1)
        for k in range(batch_size):
            mask[k*(batch_size+1),:]=0
        mask = mask>=0.5
        
        mask.to(device)
        sign = sign[mask]
        diff = diff[mask]
        loss =  torch.nn.functional.relu(-sign*diff + c )
        return loss.mean()
    
    return ordering_loss




def get_ordering_loss_fast(c):
    
    def ordering_loss(pred, real):
        device = pred.device
        batch_size = pred.shape[0]
        sign = torch.sign(torch.repeat_interleave(real, batch_size,0)-real.repeat(batch_size,1))
        diff = torch.repeat_interleave(pred, batch_size,0)-pred.repeat(batch_size,1)

        ##### supress self diff 
        mask = torch.ones(batch_size*batch_size,1)
        index = [k*(batch_size+1) for k in range(batch_size)]
        mask[index,:] = 0
        mask = mask >= 0.5
        
        mask.to(device)
        sign = sign[mask]
        diff = diff[mask]
        loss =  torch.nn.functional.relu(-sign*diff + c )
        return loss.mean()
    
    return ordering_loss


# def get_ordering_loss2(c):
    
#     def ordering_loss2(pred, real):
#         device = pred.device
#         batch_size = pred.shape[0]
#         sign = torch.sign(torch.repeat_interleave(real, batch_size,0)-real.repeat(batch_size,1))
#         diff = torch.repeat_interleave(pred, batch_size,0)-pred.repeat(batch_size,1)

#         ##### supress self diff 
#         mask = torch.ones(batch_size*batch_size,1)
#         for k in range(batch_size):
#             mask[k*(batch_size+1),:]=0
#         mask = mask>=0.5
        
#         mask.to(device)
#         sign = sign[mask]
#         diff = diff[mask]
#         loss =  torch.nn.functional.relu(-sign*diff + c ) - 0.1*torch.abs(diff)
#         return loss.mean()
    
#     return ordering_loss2

# def ordering_loss_elu(pred, real):
#     device = pred.device
#     batch_size = pred.shape[0]
#     sign = torch.sign(torch.repeat_interleave(real, batch_size,0)-real.repeat(batch_size,1))
#     diff = torch.repeat_interleave(pred, batch_size,0)-pred.repeat(batch_size,1)

#     ##### supress self diff 
#     mask = torch.ones(batch_size*batch_size,1)
#     for k in range(batch_size):
#         mask[k*(batch_size+1),:]=0
#     mask = mask>=0.5

#     mask.to(device)
#     sign = sign[mask]
#     diff = diff[mask]
#     loss =  torch.nn.functional.elu(-sign*diff  )
    return loss.mean()
    
def get_loss(name, c = 2):
    if name =='mse' :
        return F.mse_loss
    elif name =='ordering':
        return get_ordering_loss(c)
#         return get_ordering_loss_fast(c)
    
#     elif name =='ordering_elu' :
#         return ordering_loss_elu
#     elif name =='ordering2':
#         return get_ordering_loss2(c)
    else :
        return ValueError
    
class PropertyPredictor(pl.LightningModule):
    def __init__(self, in_dim, loss_name ='mse', c = 2, learning_rate=0.001):
        super(PropertyPredictor, self).__init__()
        self.learning_rate = learning_rate
        self.fc = nn.Sequential(nn.Linear(in_dim, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1))
        
        self.loss_fn = get_loss(loss_name, c)
        self.order_loss = get_loss('ordering', 0)
        self.save_hyperparameters()
       
        
        
    def forward(self, x):
        return self.fc(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def loss_function(self, pred, real):
        return self.loss_fn(pred, real)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        
            
        self.log('val_loss', loss)
        
        self.log('val_order', self.order_loss(out,y))
        
        self.log('nb unique elt',torch.unique(out).shape[0])
       
        return loss

    
# class PropertyPredictor2(pl.LightningModule):
#     def __init__(self, in_dim, loss_name ='mse', c = 2, learning_rate=0.001, decay = 1):
#         super(PropertyPredictor2, self).__init__()
#         print('var C')
#         self.learning_rate = learning_rate
#         self.fc = nn.Sequential(nn.Linear(in_dim, 128),
#                                 nn.ReLU(),
#                                 nn.Linear(128, 128),
#                                 nn.ReLU(),
#                                 nn.Linear(128, 1))
        
#         self.loss_fn = get_loss(loss_name, c)
#         self.order_loss = get_loss('ordering', 0)

#         self.loss_name = loss_name
#         self.step_ = 0
#         self.init_c = c
#         self.decay = decay
        
#     def set_loss(self,c) :
        
#         self.loss_fn = get_loss(self.loss_name, c)
#     def forward(self, x):
#         return self.fc(x)
    
#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.learning_rate)
    
#     def loss_function(self, pred, real):
#         return self.loss_fn(pred, real)
    
#     def training_step(self, batch, batch_idx):
#         c = self.init_c*math.exp((-self.decay*self.step_))
#         print(c)
#         self.set_loss(c)
#         x, y = batch
#         out = self(x)
#         loss = self.loss_function(out, y)
#         self.log('train_loss', loss)
#         self.step_ +=1
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         out = self(x)
#         loss = self.loss_function(out, y)
        
            
#         self.log('val_loss', loss)
        
#         self.log('val_order', self.order_loss(out,y))
        
#         self.log('nb unique elt',torch.unique(out).shape[0])
       
#         return loss
    
