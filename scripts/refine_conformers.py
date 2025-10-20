import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import sys

raw_sc = []
min_sc = []
confs = []

#load multiple sdf files, for fixing TCO conformation
for f in sys.argv[1:]:
  suppl = Chem.SDMolSupplier(f, sanitize=True, removeHs=False, strictParsing=True)
  
  for mol in suppl:
    try:  
      ff0 = AllChem.UFFGetMoleculeForceField(mol)
      ff0.Initialize()
      ff1 = AllChem.UFFGetMoleculeForceField(mol)
      ff1.Initialize()
      
      raw_sc.append(ff0.CalcEnergy())
      
      for atom in mol.GetAtoms():
          idx = atom.GetIdx()
          ff1.UFFAddPositionConstraint(idx, maxDispl=0.1, forceConstant=100)
  
      ff1.Minimize(maxIts=200)
      min_sc.append(ff0.CalcEnergy())
  
      confs.append(mol)
    except:
      pass

import numpy as np
up = np.std(min_sc)
print( "std:", up )
if up > 20.0: up = 20.0
ener_cut = np.median(min_sc) + up
print( "E_cut", ener_cut )

writer = Chem.SDWriter("refined-confs.sdf")
for sc, mol in zip(min_sc, confs):
    if sc < ener_cut:
        writer.write(mol)
writer.close()
