# wrapper.py
#%%
path = os.path.abspath(os.getcwd())+'/outputs'
import sys
import os
import pandas as pd
import numpy as np
from bat_can import bat_can
from SSRcode import SSRmain
from scipy import optimize as spo
import yaml

#%%
#inputfile = ["LiMetal_PorousSep_Air.yaml"]
#Summary_array = [0 for _ in inputfile] forgot what this is for.  It might be useful some day
#%%
ID_key = {"CPCN04": "50CP50CNT0.4", "CP04":"CP0.4", "CPCN05":"50CP50CNT0.5", "CP05": "CP0.5", "CPCN06":"50CP50CNT0.6", "CP06":"CP0.6"}
comparison = 'Refdata.xlsx'
refpath = 'D:\projects\Data\ToProcess' #change this to file directory
test_yaml = 'D:\projects\BatCan\inputs\LiMetal_PorousSep_Air_test.yaml'

#%%

def batcanrun(params):
    print(params)
    rate_constant, D_k_li = params
    placeholder = float(rate_constant)
    placeholder2 = float(D_k_li)
    with open(test_yaml, 'r') as file:
        imput_file = yaml.safe_load(file)
    imput_file['cathode-surf-reactions'][0]['rate-constant']['A'] = placeholder
    imput_file['cell-description']['separator']['transport']['diffusion-coefficients'][1]['D_k'] = placeholder2 #D_k lithium
    with open(test_yaml, "w") as f:
        yaml.dump(imput_file, f)
    foldarray =os.listdir(path)
    bat_can('LiMetal_PorousSep_Air_test.yaml')
    foldarray2 = os.listdir(path)
    folder_name = list(set(foldarray).symmetric_difference(set(foldarray2)))
    SSRarray = []
    for i in folder_name:
        array = i.split("_")
        dataset = ID_key[array[0]]
        RefData = pd.read_excel(refpath + "/"+ comparison, sheet_name= dataset)
        SSR_file = SSRmain(i)
        SSRarray.append(SSR_file)
    print(SSRarray)
    return(sum(SSRarray))

#%%
test_yaml = 'D:\projects\BatCan\inputs\LiMetal_PorousSep_Air_test.yaml'
Done = batcanrun([95000], [1.11e-9])
print(Done)

#%%
x_start = [85000, 0.5e-9]
result = spo.minimize(batcanrun, x_start)
print("x:", result.x)
print("SSR:", result.fun)
#%%