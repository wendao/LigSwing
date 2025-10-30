#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdb_backbone_to_sdf.py

从 PDB 提取满足筛选条件（chain/resn(resname)/resi）的残基主链构象 [N, C, CA, O]，
并将其写入到一个 SDF 文件中。每个残基输出为一个条目（记录）。
原子顺序严格为 [N, C, CA, O]，并建立主链键：N-CA, CA-C, C-O。

依赖：RDKit
    conda install -c conda-forge rdkit

示例：
  全部链、全部残基名、限定残基编号 10、15、20–30：
    python pdb_backbone_to_sdf.py --pdb input.pdb --out out.sdf --chain ALL --resn ALL --resi 10,15,20-30

  只选链 A、残基名 GLY/ALA、编号 5–12：
    python pdb_backbone_to_sdf.py --pdb input.pdb --out out.sdf --chain A --resn GLY,ALA --resi 5-12
"""

import argparse
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Set

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 (保留以便未来扩展)
from rdkit.Geometry import Point3D


REQUIRED_ATOMS = ("N", "C", "CA", "O")  # 输出顺序要求
# 在输出分子中建立简单主链键连（基于上面顺序的索引）
BONDS = [(0, 2),  # N-CA
         (2, 1),  # CA-C
         (1, 3)]  # C-O


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract backbone [N, C, CA, O] from PDB and write to SDF as one record per residue."
    )
    ap.add_argument("--pdb", required=True, help="Input PDB file")
    ap.add_argument("--out", required=True, help="Output SDF file")
    ap.add_argument("--chain", default="ALL",
                    help="Chain filter: 'ALL' or comma-separated list, e.g. 'A,B'")
    # 新增：resn（残基名，3字母），并保留 resname 兼容（优先使用 resn）
    ap.add_argument("--resn", default=None,
                    help="Residue name filter (3-letter): 'ALL' or comma-separated list, e.g. 'GLY,ALA'")
    ap.add_argument("--resname", default=None,
                    help="(Deprecated alias) Same as --resn")
    # 新增：resi（残基编号；支持列表与闭区间）
    ap.add_argument("--resi", default="ALL",
                    help="Residue index filter: 'ALL' or list/ranges, e.g. '10,15,20-30'")
    return ap.parse_args()


def _parse_name_filter(name_str: Optional[str]) -> Optional[Set[str]]:
    """
    将 'ALL' 或 逗号分隔的名称字符串解析为集合；None/ALL -> None 表示不过滤
    """
    if name_str is None:
        return None
    s = name_str.strip()
    if not s or s.upper() == "ALL":
        return None
    return {x.strip().upper() for x in s.split(",") if x.strip()}


def _parse_resi_filter(resi_str: Optional[str]) -> Optional[Set[int]]:
    """
    将 'ALL' 或 '10,15,20-30' 解析为一个整数集合；ALL/None -> None 表示不过滤
    仅处理纯整数编号（PDB insertion code 另行处理，不在 resi 里表达）。
    """
    if resi_str is None:
        return None
    s = resi_str.strip().upper()
    if not s or s == "ALL":
        return None

    result: Set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a, b = a.strip(), b.strip()
            if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                ai, bi = int(a), int(b)
                if ai <= bi:
                    result.update(range(ai, bi + 1))
                else:
                    result.update(range(bi, ai + 1))
        else:
            if tok.lstrip("-").isdigit():
                result.add(int(tok))
    return result or None


def normalize_filters(chain_str: str,
                      resn_str: Optional[str],
                      resname_str: Optional[str],
                      resi_str: Optional[str]):
    # chain
    if chain_str.strip().upper() == "ALL":
        chain_filter = None
    else:
        chain_filter = {c.strip() for c in chain_str.split(",") if c.strip()}

    # resn / resname（二选一；优先 resn）
    resn_filter = _parse_name_filter(resn_str)
    if resn_filter is None:
        resn_filter = _parse_name_filter(resname_str)

    # resi
    resi_filter = _parse_resi_filter(resi_str)

    return chain_filter, resn_filter, resi_filter


def _choose_better_atom(existing, candidate):
    """
    从两个候选原子记录中择优：
      1) 优先 altLoc == '' 或 ' ' 或 'A'
      2) 再比较 occupancy，取较大者
    记录结构：(x,y,z, occ, altLoc)
    """
    if existing is None:
        return candidate

    x1, y1, z1, occ1, alt1 = existing
    x2, y2, z2, occ2, alt2 = candidate

    def alt_rank(a: str) -> int:
        a = (a or ' ').strip() or ' '
        if a == 'A' or a == '':
            return 0
        return 1

    r1, r2 = alt_rank(alt1), alt_rank(alt2)
    if r1 != r2:
        return existing if r1 < r2 else candidate
    # alt 优先级相同，则看占据率
    return existing if (occ1 or 0.0) >= (occ2 or 0.0) else candidate


def parse_pdb_atoms(pdb_path: str,
                    chain_filter: Optional[Set[str]],
                    resn_filter: Optional[Set[str]],
                    resi_filter: Optional[Set[int]]
                    ) -> Dict[Tuple[str, int, str, str],
                              Dict[str, Tuple[float, float, float, float, str]]]:
    """
    解析 PDB 文件的 ATOM 行，按 (chainID, resSeq, iCode, resName) 分组，
    在每个残基内记录所需原子 N/C/CA/O 的坐标（择优处理 altLoc/occupancy）。

    返回：
      residues[(chain, resseq, icode, resname)][atomname] = (x,y,z, occ, altLoc)
    """
    residues = defaultdict(dict)

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue

            # PDB 固定列解析（PDB v3 格式）
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip().upper()
            chain_id = line[21].strip()
            res_seq = line[22:26].strip()
            i_code = line[26].strip()  # insertion code
            alt_loc = line[16].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            occ_str = line[54:60].strip()
            try:
                occ = float(occ_str) if occ_str else 0.0
            except ValueError:
                occ = 0.0

            # —— 过滤 —— #
            if chain_filter is not None and chain_id not in chain_filter:
                continue
            if resn_filter is not None and res_name not in resn_filter:
                continue
            try:
                res_seq_int = int(res_seq)
            except ValueError:
                # 遇到非标准编号时跳过
                continue
            if resi_filter is not None and res_seq_int not in resi_filter:
                continue
            if atom_name not in REQUIRED_ATOMS:
                continue

            # —— 记录 —— #
            key = (chain_id, res_seq_int, i_code, res_name)
            prev = residues[key].get(atom_name)
            newv = (x, y, z, occ, alt_loc)
            best = _choose_better_atom(prev, newv)
            residues[key][atom_name] = best

    return residues


def build_backbone_template() -> Chem.Mol:
    """
    构建 4 原子骨架的 RDKit Mol，原子顺序为 [N, C, CA, O]，
    并添加主链键：N-CA, CA-C, C-O。
    """
    em = Chem.EditableMol(Chem.Mol())

    # 原子：N, C, CA(用C表示), O
    atom_N = Chem.Atom(7)   # N
    atom_C = Chem.Atom(6)   # C (羰基C)
    atom_CA = Chem.Atom(6)  # CA 也是碳
    atom_O = Chem.Atom(8)   # O

    em.AddAtom(atom_N)   # idx 0
    em.AddAtom(atom_C)   # idx 1
    em.AddAtom(atom_CA)  # idx 2
    em.AddAtom(atom_O)   # idx 3

    # 主链键
    for a, b in BONDS:
        em.AddBond(a, b, Chem.BondType.SINGLE)

    mol = em.GetMol()
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    mol.SetProp("_Name", "G")  # 视作氨基酸 G 的骨架
    return mol


def add_conformer(mol: Chem.Mol, coords: List[Tuple[float, float, float]]) -> int:
    """
    给 mol 添加一个构象（四个点），coords 必须按 [N, C, CA, O] 顺序。
    返回构象 ID。
    """
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    cid = mol.AddConformer(conf, assignId=True)
    return cid


def main():
    args = parse_args()
    chain_filter, resn_filter, resi_filter = normalize_filters(
        args.chain, args.resn, args.resname, args.resi
    )

    residues = parse_pdb_atoms(args.pdb, chain_filter, resn_filter, resi_filter)

    # 仅保留四原子齐全的残基，并按链/编号排序
    selected = []
    for key, atommap in residues.items():
        if all(a in atommap for a in REQUIRED_ATOMS):
            selected.append((key, atommap))

    # 稳定排序：按 chain, resseq, icode
    selected.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))

    if not selected:
        print("No residues matched filters with complete [N, C, CA, O].")
        return

    # 构建模板骨架（4 原子）
    base_mol = build_backbone_template()

    writer = Chem.SDWriter(args.out)
    writer.SetKekulize(False)

    n_written = 0
    for (chain_id, resseq, icode, resname), amap in selected:
        # 按指定顺序抓取坐标
        coord_dict = {a: amap[a][:3] for a in REQUIRED_ATOMS}
        coords = [coord_dict["N"], coord_dict["C"], coord_dict["CA"], coord_dict["O"]]

        # 拷贝一个分子，添加构象
        mol = Chem.Mol(base_mol)
        cid = add_conformer(mol, coords)

        # 设置条目名称与字段
        title = f"G {chain_id}:{resname}:{resseq}{icode or ''}"
        mol.SetProp("_Name", title)
        mol.SetProp("CHAIN", chain_id)
        mol.SetProp("RESNAME", resname)
        mol.SetProp("RESSEQ", str(resseq))
        mol.SetProp("ICODE", icode or "")
        mol.SetIntProp("_ConfId", cid)

        writer.write(mol, confId=cid)
        n_written += 1

    writer.close()
    print(f"Done. Wrote {n_written} conformers (residues) to: {args.out}")


if __name__ == "__main__":
    main()

