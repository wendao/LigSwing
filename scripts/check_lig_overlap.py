import sys, os, csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import rdkit.Chem.rdFMCS as MCS

#import py3Dmol
#from ipywidgets import interact, interactive, fixed

#python check.py ref.sdf SMILES.txt 

#load lig template
suppl = Chem.SDMolSupplier(sys.argv[1])
for mol in suppl:
    LIG_tmpl = mol
    print("Template", Chem.MolToSmiles(LIG_tmpl))

#load UAA smile strings
smi_db = {}
lines = open(sys.argv[2], 'r').readlines()
for l in lines[:]:
    es = l.split(' ')
    ndx = es[0]
    smi = es[1].strip()
    #print ndx, smi
    smi_db[ndx] = smi
smi_db

import numpy
def GenOverlap(mol, tmpl):
    #constrained and align
    #res = MCS.FindMCS([mol, tmpl], threshold=1.0, completeRingsOnly=True)
    res = MCS.FindMCS([mol, tmpl])
    p = Chem.MolFromSmarts(res.smartsString)
    core = AllChem.DeleteSubstructs(AllChem.ReplaceSidechains(tmpl,p), Chem.MolFromSmiles('*'))
    core.UpdatePropertyCache()
    print("overlap:", Chem.MolToSmiles(core))

for ndx in smi_db.keys():
    k = ndx
    smi = smi_db[k]
    #init
    #d = "U"+str(k)
    #if not os.path.isdir(d):
    #    os.mkdir(d)
    #create
    print(k, smi)
    lig = AllChem.MolFromSmiles(smi) #Chem.MolToSmiles(mol, isomericSmiles=True)
    #fix prod
    Chem.SanitizeMol(lig)
    product = Chem.AddHs(lig)
    #use templete
    GenOverlap(product, LIG_tmpl)

