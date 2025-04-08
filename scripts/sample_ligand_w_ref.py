import sys, os, csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import rdkit.Chem.rdFMCS as MCS

import py3Dmol
from ipywidgets import interact, interactive, fixed

#load lig template
suppl = Chem.SDMolSupplier(sys.argv[1])
for mol in suppl:
    LIG_tmpl = mol

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
def CalcConfRMS(N, conf1, conf2):
    ssr = 0                                                                        
    for i in range(N):                                             
        d = conf1.GetAtomPosition(i).Distance(conf2.GetAtomPosition(i))              
        ssr += d * d                                                                 
    ssr /= N
    return numpy.sqrt(ssr)

def GenMutiConfMol(mol, tmpl, num_iter, rms_cut, w):
    #constrained and align
    res = MCS.FindMCS([mol, tmpl], threshold=0.9)
    p = Chem.MolFromSmarts(res.smartsString)
    core = AllChem.DeleteSubstructs(AllChem.ReplaceSidechains(tmpl,p), Chem.MolFromSmiles('*'))
    core.UpdatePropertyCache()
    print("core:", Chem.MolToSmiles(core))

    #generate confs
    conformers = None
    last_iter = 0
    for i in range(num_iter):
        AllChem.ConstrainedEmbed(mol, core, randomseed=i+111)
        ebrms = float(mol.GetProp('EmbedRMS'))
        if ebrms<0.12:
            #check 
            if conformers == None:
                #save the first one
                conformers = Chem.MolFromMolBlock( Chem.MolToMolBlock(mol) )
                Chem.SanitizeMol(conformers)
                conformers = Chem.AddHs(conformers)
                conformers.AddConformer(mol.GetConformer(0))
                w.write(mol)
            else:
                confs = conformers.GetConformers()
                rmslst = [ CalcConfRMS(mol.GetNumAtoms(), conf, mol.GetConformer(0)) for conf in confs ]
                minrms = min(rmslst)
                if minrms > rms_cut:
                    #save if not close any old rotamer
                    conformers.AddConformer(mol.GetConformer(0))
                    w.write(mol)
                    print(i, 'min:', minrms)
                    last_iter = i
                else:
                    if i-last_iter>100:
                        print("Warning: early quit at iter=", i)
                        break
    return conformers

for ndx in smi_db.keys():
    k = ndx
    smi = smi_db[k]
    #init
    d = "U"+str(k)
    if not os.path.isdir(d):
        os.mkdir(d)
    #create
    print(k, smi)
    lig = AllChem.MolFromSmiles(smi) #Chem.MolToSmiles(mol, isomericSmiles=True)
    #react
    writer = Chem.SDWriter(d+"/confs.sdf")
    #fix prod
    Chem.SanitizeMol(lig)
    product = Chem.AddHs(lig)
    #use templete
    GenMutiConfMol(product, LIG_tmpl, 10000, 0.6, writer)
    writer.close()

