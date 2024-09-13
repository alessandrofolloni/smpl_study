import json
import numpy as np
import torch

path = "../files/band_pull_apart.json"
with open(path, 'r') as f:
    smplx_params = json.load(f)

print("Chiavi del dizionario:", smplx_params.keys())

for key in smplx_params.keys():
    array = np.array(smplx_params[key])
    print(f"Dimensioni di {key}: {array.shape}")
    print(f"Range di valori per {key}: Min = {array.min()}, Max = {array.max()}")

''' Tendenzialmente i valori sono compresi tra -1 e 1 circa '''

json_path = "../files/band_pull_apart.json"
with open(json_path, 'r') as f:
    smplx_params = json.load(f)
    print(smplx_params.keys())
    print(np.array(smplx_params['transl']).shape)
    el = torch.tensor(np.array(smplx_params['transl']))
    print(el[100])

print(smplx_params['body_pose'][0])

