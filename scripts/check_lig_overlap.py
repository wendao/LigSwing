import sys, os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import rdkit.Chem.rdFMCS as MCS

out_dir = "overlap_images"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# load lig template
suppl = Chem.SDMolSupplier(sys.argv[1])
LIG_tmpl = None
for mol in suppl:
    LIG_tmpl = mol
    break
print("Template:", Chem.MolToSmiles(LIG_tmpl))

# load SMILES strings
smi_db = {}
for l in open(sys.argv[2], 'r').readlines():
    es = l.split()
    if len(es) < 2:
        continue
    smi_db[es[0]] = es[1].strip()


def draw_overlap(mol, tmpl, ndx):
    res = MCS.FindMCS([mol, tmpl])
    p = Chem.MolFromSmarts(res.smartsString)
    core = AllChem.DeleteSubstructs(AllChem.ReplaceSidechains(tmpl, p), Chem.MolFromSmiles('*'))
    core.UpdatePropertyCache()
    print(ndx, "overlap:", Chem.MolToSmiles(core))

    # find matching atoms in mol
    match = mol.GetSubstructMatch(p)
    if not match:
        print(ndx, "no match found in mol, falling back to SMARTS match")
        match = mol.GetSubstructMatch(Chem.MolFromSmarts(res.smartsString))
    if match:
        img = Draw.MolToImage(mol, highlightAtoms=list(match), size=(400, 300))
        img.save(os.path.join(out_dir, f"{ndx}_overlap.png"))
        print(ndx, f"saved to {out_dir}/{ndx}_overlap.png")
    else:
        print(ndx, "no match found, skipping image")


for ndx in smi_db:
    smi = smi_db[ndx]
    print(ndx, smi)
    lig = Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(lig)
    mol_h = Chem.AddHs(lig)
    draw_overlap(mol_h, LIG_tmpl, ndx)

