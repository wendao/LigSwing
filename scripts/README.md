# Scripts

## backbone_to_sdf.py

Extract backbone atom conformations [N, C, CA, O] from PDB files and write them to SDF, one record per residue.

**Usage:**
```
python backbone_to_sdf.py --pdb <input.pdb> --out <out.sdf> [--chain <filter>] [--resn <filter>] [--resi <filter>]
```

- `--pdb` — input PDB file
- `--out` — output SDF file
- `--chain` — chain filter: `ALL` (default) or comma-separated list, e.g. `A,B`
- `--resn` — residue name filter (3-letter): `ALL` or comma-separated list, e.g. `GLY,ALA`
- `--resi` — residue index filter: `ALL` or list/ranges, e.g. `10,15,20-30`

For each matched residue with complete [N, C, CA, O] atoms, the script:
1. Selects the best alternate location / occupancy for each atom
2. Builds a 4-atom backbone fragment with bonds N-CA, CA-C, C-O
3. Writes the conformer to SDF with metadata (chain, resname, resseq, icode)

Example:
```
python backbone_to_sdf.py --pdb 1abc.pdb --out backbone.sdf --chain A --resn GLY,ALA --resi 5-12
```

---

## check_lig_overlap.py

Check the maximum common substructure (MCS) overlap between a reference ligand template and a set of SMILES strings, and draw molecular structures with the overlap highlighted.

**Usage:**
```
python check_lig_overlap.py <ref.sdf> <SMILES.txt>
```

- `ref.sdf` — SDF file containing the reference ligand template (first molecule is used)
- `SMILES.txt` — text file with `<index> <SMILES>` per line

For each SMILES molecule, the script:
1. Finds the MCS between the molecule and the template
2. Prints the overlapping core substructure
3. Draws the molecular structure with MCS matched atoms highlighted, saved as `overlap_images/<index>_overlap.png`

---

## sample_ligand_w_ref.py

Generate diverse 3D conformers for ligand molecules by constrained embedding against a reference template.

**Usage:**
```
python sample_ligand_w_ref.py <ref.sdf> <SMILES.txt>
```

- `ref.sdf` — SDF file containing the reference ligand template
- `SMILES.txt` — text file with `<index> <SMILES>` per line

For each SMILES entry, the script:
1. Finds the common core between the molecule and the template (MCS at 0.9 threshold)
2. Runs up to 10,000 iterations of constrained embedding, keeping only conformers with EmbedRMS < 0.12
3. Filters by RMSD (cutoff 0.6 Å) to ensure conformational diversity
4. Writes the accepted conformers to `U<index>/confs.sdf`

Stops early if no new conformer is accepted for 100 consecutive iterations.

---

## refine_conformers.py

Refine conformers using UFF force field minimization with position constraints, and filter by energy.

**Usage:**
```
python refine_conformers.py <conf1.sdf> [conf2.sdf ...]
```

- Accepts one or more SDF files as input

For each conformer:
1. Computes initial UFF energy
2. Applies position constraints (max displacement 0.1 Å, force constant 100) to all atoms
3. Minimizes (max 200 iterations) and records the resulting energy

Outputs `refined-confs.sdf` containing only conformers whose energy is below `median + std` (std capped at 20 kcal/mol).
