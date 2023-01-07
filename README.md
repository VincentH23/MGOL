# MGOL: Molecule Generation with Ordering Loss

This is the repository for our Bioinformatics article submission "MGOL: Molecule Generation with Ordering Loss" by Vincent HUANG, Selim YAHIA-MESSAOUD, Wenming YANG, Dong-Qing WEI, Jie CHEN , Yonghong TIAN.

This code is adapted from the official repository of the 2022 ICML paper "LIMO: Latent Inceptionism for Targeted Molecule Generation" by Peter Eckmann, Kunyang Sun, Bo Zhao, Mudong Feng, Michael K. Gilson, and Rose Yu.


## Installation

Please ensure that [RDKit](https://www.rdkit.org/docs/Install.html) and [Open Babel](https://openbabel.org/wiki/Category:Installation) are installed. The following Python packages are also required (these can also be installed with `pip install -r requirements.txt`):

```
torch
pytorch-lightning
selfies
scipy
tqdm
```

To optimize molecules for binding affinity, an AutoDock-GPU executable [must be compiled](https://github.com/ccsb-scripps/AutoDock-GPU#compilation). To generate your own protein files, see Steps 1-2 in the [AutoDock4 User Guide](https://autodock.scripps.edu/wp-content/uploads/sites/56/2021/10/AutoDock4.2.6_UserGuide.pdf). The compiled file has to be renamed 'autodock_gpu_128wi' and put into the main directory. The [AutoDock-GPU Wiki](https://github.com/ccsb-scripps/AutoDock-GPU/wiki/Guideline-for-users) may also be helpful.

## Generating molecules (Comparison between Ordering Loss and MSE)

To run the code that leads to the results presented in the Table 3, you should run :

'''
bash run.sh
'''