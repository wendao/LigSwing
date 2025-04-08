import sys, os, csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import rdkit.Chem.rdFMCS as MCS

#python align.py conf.sdf {SMILES}

smi = sys.argv[2]

writer = Chem.SDWriter("lig-confs.sdf")
suppl = Chem.SDMolSupplier(sys.argv[1])
i = 0
for tmpl in suppl:
    i += 1
    print("Conf", i)

    #align
    lig = AllChem.MolFromSmiles(smi)
    Chem.SanitizeMol(lig)
    mol = Chem.AddHs(lig)
    res = MCS.FindMCS([mol, tmpl], threshold=0.9)
    p = Chem.MolFromSmarts(res.smartsString)
    core = AllChem.DeleteSubstructs(AllChem.ReplaceSidechains(tmpl,p), Chem.MolFromSmiles('*'))
    core.UpdatePropertyCache()
    print("core:", Chem.MolToSmiles(core))

    AllChem.ConstrainedEmbed(mol, core, randomseed=42)
    ebrms = float(mol.GetProp('EmbedRMS'))
    if ebrms<0.1:
        writer.write(mol)

writer.close()


