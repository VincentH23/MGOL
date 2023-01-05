import torch 
import math
delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))

def delta_g_to_kd_tensor(delta_g) :
    kd_tensor = torch.exp(delta_g/(0.00198720425864083 * 298.15))
    return kd_tensor
    

print('================================ 1ERR ===========================================')
for i in range(5):
    y = torch.load(f'data/1err/10000_y_{i+1}.pt')
    print(f'dataset {i+1}')
    avg =  y.mean().item()
    print(f'average  {avg}')
    print(f'avg kd {delta_g_to_kd_tensor(y).mean().item()}')
    min_E = y.min().item()
    print(f'MIN  {min_E}')
    print(f'MIN kd {delta_g_to_kd(min_E)}')
    print('')
    
    
# print('================================ 2iik ===========================================')
# for i in range(5):
#     y = torch.load(f'data/2iik/10000_y_{i+1}.pt')
#     print(f'dataset {i+1}')
#     avg =  y.mean().item()
#     print(f'average  {avg}')
#     print(f'avg kd {delta_g_to_kd_tensor(y).mean().item()}')
#     min_E = y.min().item()
#     print(f'MIN  {min_E}')
#     print(f'MIN kd {delta_g_to_kd(min_E)}')
#     print('')
    