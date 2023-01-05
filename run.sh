mkdir data z_vector ligands outs params_ligands property_models results

python preprocess_data.py

python train_vae.py

python generate_datasets.py --n_dataset 5 

python multidataset_train_property_predictor.py

python multidataset_generate_molecules.py 