#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sdf_overlap.py

功能概述
- 读取两个包含同一分子的多构象 SDF 文件
- 通过各自的原子编号列表（--idx1/--idx2，0-based，且按顺序一一对应）定义“共有原子”
- 两种“重合”语义（--overlap-mode）：
  1) rigid（默认）：先用共有原子做刚体对齐，再逐原子检查是否全部在 tol 内
  2) absolute：不做对齐，直接在原始坐标系逐原子检查是否全部在 tol 内
- 两种匹配策略（--method）：
  1) hash（默认）：几何哈希把候选从 n*m 降到近似 n+m，再精确复核
     - rigid：用“成对距离向量”做量化哈希（旋转/平移不变）
     - absolute：用“按顺序的坐标向量”做量化哈希（不对齐）
  2) brute：穷举 n*m（带进度条），适合校验或样本很小时
- 进度条（tqdm）
- 结果 CSV 输出；可将通过筛选的匹配构象导出为 PDB（--save-pdb-dir）

依赖
- rdkit, numpy, tqdm
  conda install -c conda-forge rdkit
  pip install numpy tqdm

示例
python sdf_overlap.py A.sdf B.sdf --idx1 0,1,2,5 --idx2 3,4,7,8 \
  --overlap-mode rigid --tol 0.3 --method hash --bin 0.3 \
  --max-rmsd 0.3 --topk 200 --save-pdb-dir out_pdb --save-pdb-limit 50
"""

import argparse
import csv
import math
import os
import sys
import platform
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

# ---------------- RDKit / tqdm ----------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
except ImportError:
    print("ERROR: 需要 RDKit： conda install -c conda-forge rdkit", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kw): return it


# ---------------- 数据结构 ----------------
@dataclass
class MatchResult:
    conf1: int
    conf2: int
    rmsd: float
    frac_in_tol: float
    max_pair_dist: float
    n_common: int


# ---------------- 工具函数 ----------------
def parse_indices(indices_str: str) -> List[int]:
    if not indices_str:
        return []
    try:
        return [int(x.strip()) for x in indices_str.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"无法解析原子编号列表: {indices_str}")


def load_mol_with_confs(path: str) -> Chem.Mol:
    """合并多条目为一个含多 conformer 的 Mol。"""
    suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"SDF中未读到分子: {path}")
    base = Chem.Mol(mols[0])
    for m in mols[1:]:
        for c in m.GetConformers():
            base.AddConformer(c, assignId=True)
    return base


def assert_index_range(mol: Chem.Mol, idxs: List[int], tag: str):
    n = mol.GetNumAtoms()
    for i in idxs:
        if i < 0 or i >= n:
            raise IndexError(f"{tag} 原子编号 {i} 超出范围 [0, {n-1}]（原子数={n}）")


def coords_of(mol: Chem.Mol, confId: int, idxs: List[int]) -> np.ndarray:
    conf = mol.GetConformer(confId)
    return np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in idxs],
                      dtype=float)


def pairwise_dists_upper(coords: np.ndarray) -> np.ndarray:
    """返回共有原子的上三角成对距离向量（长度 k*(k-1)/2）。"""
    k = coords.shape[0]
    vec = []
    for i in range(k - 1):
        di = np.linalg.norm(coords[i+1:] - coords[i], axis=1)
        vec.append(di)
    return np.concatenate(vec) if vec else np.zeros((0,), dtype=float)


def quantize_vector(vec: np.ndarray, bin_size: float, offset: float) -> Tuple[int, ...]:
    """把连续距离向量按 (x - offset)/bin_size 量化为整数 tuple，用于 rigid 模式哈希键。"""
    if bin_size <= 0:
        raise ValueError("bin_size 必须 > 0")
    q = np.floor((vec - offset) / bin_size + 0.5).astype(np.int64)  # 近邻四舍五入
    return tuple(int(x) for x in q)


def coords_vector_quantized(coords: np.ndarray, bin_size: float, offset: float) -> Tuple[int, ...]:
    """把 (k,3) 坐标拍平成 (3k,) 并量化，用于 absolute 模式哈希键。"""
    if bin_size <= 0:
        raise ValueError("bin_size 必须 > 0")
    flat = coords.reshape(-1)
    q = np.floor((flat - offset) / bin_size + 0.5).astype(np.int64)
    return tuple(int(x) for x in q)


def multi_offsets(bin_size: float) -> List[float]:
    """三组错位网格，缓解量化边界效应。"""
    return [0.0, bin_size / 3.0, 2.0 * bin_size / 3.0]


# ---------------- 判定：rigid / absolute ----------------
def check_overlap_for_pair(
    mol1: Chem.Mol, mol2: Chem.Mol, confId1: int, confId2: int,
    idxs1: List[int], idxs2: List[int], tol: float, mode: str = "rigid"
) -> Tuple[bool, float, float, float, int]:
    """
    返回 (is_ok, rmsd, frac_in_tol, max_pair_dist, n_common)
    - rigid: 先用 idx 原子刚体对齐，再逐原子算距离
    - absolute: 不对齐，直接在原始坐标系逐原子算距离
    """
    A = coords_of(mol1, confId1, idxs1)
    B = coords_of(mol2, confId2, idxs2)
    n = len(idxs1)
    if n == 0:
        return False, float("inf"), 0.0, float("inf"), 0

    if mode == "rigid":
        # 备份 + 对齐
        atom_map = [(i2, i1) for i1, i2 in zip(idxs1, idxs2)]
        conf2 = mol2.GetConformer(confId2)
        orig = np.array([[conf2.GetAtomPosition(i).x, conf2.GetAtomPosition(i).y, conf2.GetAtomPosition(i).z]
                         for i in range(mol2.GetNumAtoms())], dtype=float)
        try:
            rms = AllChem.AlignMol(mol2, mol1, prbCid=confId2, refCid=confId1, atomMap=atom_map)
            B = coords_of(mol2, confId2, idxs2)  # 取对齐后的 B
        finally:
            for i in range(mol2.GetNumAtoms()):
                x, y, z = orig[i]
                conf2.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    else:
        # absolute: 不做对齐
        diffs0 = np.linalg.norm(A - B, axis=1)
        rms = float(np.sqrt(np.mean(diffs0**2)))

    diffs = np.linalg.norm(A - B, axis=1)
    in_tol = (diffs <= tol)
    frac = float(np.count_nonzero(in_tol)) / float(n)
    maxd = float(diffs.max())
    is_ok = (frac >= 1.0)  # “完全重合”：全部 ≤ tol
    return is_ok, float(rms), float(frac), float(maxd), n


# ---------------- brute（可选） ----------------
def initial_maxdist_prune(pA: np.ndarray, pB: np.ndarray, tol: float) -> bool:
    if tol <= 0:
        return False
    diffs = np.linalg.norm(pA - pB, axis=1)
    return np.any(diffs > 5.0 * tol)


def run_brute(
    mol1: Chem.Mol, mol2: Chem.Mol, idxs1: List[int], idxs2: List[int],
    tol: float, max_rmsd: float, topk: int, no_progress: bool, overlap_mode: str
) -> List[MatchResult]:
    n1 = mol1.GetNumConformers()
    n2 = mol2.GetNumConformers()

    def progress(it, **kw):
        return it if no_progress else tqdm(it, **kw)

    results: List[MatchResult] = []
    total = n1 * n2
    it = progress(range(total), total=total, desc="Brute pairs", unit="pair")
    for linear in it:
        cid1 = linear // n2
        cid2 = linear % n2
        if overlap_mode == "rigid":
            # 粗剪枝（可选）
            A = coords_of(mol1, cid1, idxs1)
            B = coords_of(mol2, cid2, idxs2)
            if initial_maxdist_prune(A, B, tol):
                continue
        ok, rms, frac, maxd, n = check_overlap_for_pair(
            mol1, mol2, cid1, cid2, idxs1, idxs2, tol, mode=overlap_mode
        )
        if ok and rms <= max_rmsd:
            results.append(MatchResult(conf1=cid1, conf2=cid2, rmsd=rms,
                                       frac_in_tol=frac, max_pair_dist=maxd, n_common=n))

    if not results:
        return []
    results.sort(key=lambda x: (x.rmsd, -x.frac_in_tol))
    if topk > 0:
        results = results[:topk]
    return results


# ---------------- hash（默认推荐） ----------------
def run_hash(
    mol1: Chem.Mol, mol2: Chem.Mol, idxs1: List[int], idxs2: List[int],
    tol: float, max_rmsd: float, topk: int, no_progress: bool,
    hash_bin: float, overlap_mode: str
) -> List[MatchResult]:
    n1 = mol1.GetNumConformers()
    n2 = mol2.GetNumConformers()
    k = len(idxs1)
    if k < 2 and overlap_mode == "rigid":
        raise ValueError("rigid 模式共有原子数必须≥2；推荐≥3（更稳）。")

    def progress(it, **kw):
        return it if no_progress else tqdm(it, **kw)

    offsets = multi_offsets(hash_bin)
    tables: List[Dict[Tuple[int, ...], List[int]]] = [dict() for _ in offsets]

    # 建 A 侧哈希
    for cid1 in progress(range(n1), total=n1, desc="Hash A", unit="conf"):
        c1 = coords_of(mol1, cid1, idxs1)
        if overlap_mode == "rigid":
            vec = pairwise_dists_upper(c1)
            for off, tab in zip(offsets, tables):
                key = quantize_vector(vec, hash_bin, off)
                tab.setdefault(key, []).append(cid1)
        else:  # absolute
            for off, tab in zip(offsets, tables):
                key = coords_vector_quantized(c1, hash_bin, off)
                tab.setdefault(key, []).append(cid1)

    # 探 B 并复核
    results: List[MatchResult] = []
    for cid2 in progress(range(n2), total=n2, desc="Probe B", unit="conf"):
        c2 = coords_of(mol2, cid2, idxs2)
        cand: Set[int] = set()
        if overlap_mode == "rigid":
            vec2 = pairwise_dists_upper(c2)
            for off, tab in zip(offsets, tables):
                key2 = quantize_vector(vec2, hash_bin, off)
                cand.update(tab.get(key2, ()))
        else:
            for off, tab in zip(offsets, tables):
                key2 = coords_vector_quantized(c2, hash_bin, off)
                cand.update(tab.get(key2, ()))

        if not cand:
            continue

        for cid1 in cand:
            ok, rms, frac, maxd, n = check_overlap_for_pair(
                mol1, mol2, cid1, cid2, idxs1, idxs2, tol, mode=overlap_mode
            )
            if ok and rms <= max_rmsd:
                results.append(MatchResult(conf1=cid1, conf2=cid2, rmsd=rms,
                                           frac_in_tol=frac, max_pair_dist=maxd, n_common=n))

    if not results:
        return []
    results.sort(key=lambda x: (x.rmsd, -x.frac_in_tol))
    if topk > 0:
        results = results[:topk]
    return results


# ---------------- PDB 保存 ----------------
def write_single_conf_pdb(mol: Chem.Mol, conf_id: int, path: str):
    """
    将指定构象单独写成 PDB 文件（不污染原 mol）。
    优先尝试 RDKit 直接写指定 conf；若版本不支持，则构造只含该 conf 的临时 Mol 再写。
    """
    try:
        Chem.MolToPDBFile(mol, path, confId=conf_id)
        return
    except TypeError:
        pass  # 老版本走 fallback

    tmp = Chem.Mol(mol)
    conf = mol.GetConformer(conf_id)
    new_id = tmp.AddConformer(conf, assignId=True)
    # 删除其他 conformers（只保留最后一个）
    for cid in list(reversed([c.GetId() for c in tmp.GetConformers()])):
        if cid != new_id:
            tmp.RemoveConformer(cid)
    Chem.MolToPDBFile(tmp, path)


def save_results_and_pdb(
    results: List[MatchResult],
    out_csv: str,
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    save_dir: Optional[str],
    save_limit: Optional[int]
):
    # CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "conf1", "conf2", "rmsd", "frac_in_tol", "max_pair_dist", "n_common"])
        for rank, r in enumerate(results, start=1):
            w.writerow([rank, r.conf1, r.conf2, f"{r.rmsd:.6f}", f"{r.frac_in_tol:.4f}",
                        f"{r.max_pair_dist:.6f}", r.n_common])

    # PDB（可选）
    if save_dir and results:
        os.makedirs(save_dir, exist_ok=True)
        limit = save_limit if save_limit is not None else len(results)
        limit = max(0, min(limit, len(results)))
        for rank, r in enumerate(results[:limit], start=1):
            a_path = os.path.join(save_dir, f"pair_{rank:04d}__A_conf{r.conf1}.pdb")
            b_path = os.path.join(save_dir, f"pair_{rank:04d}__B_conf{r.conf2}.pdb")
            write_single_conf_pdb(mol1, r.conf1, a_path)
            write_single_conf_pdb(mol2, r.conf2, b_path)


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="SDF 多构象匹配（按给定原子顺序完全重合），几何哈希加速，可导出匹配构象为 PDB"
    )
    ap.add_argument("sdf1")
    ap.add_argument("sdf2")
    ap.add_argument("--idx1", required=True, help="SDF1 共有原子编号（逗号分隔，0-based，按顺序）")
    ap.add_argument("--idx2", required=True, help="SDF2 共有原子编号（逗号分隔，0-based，按顺序；与 idx1 等长）")
    ap.add_argument("--overlap-mode", choices=["rigid", "absolute"], default="rigid",
                    help="rigid：刚体对齐后检查 idx 原子全体 ≤ tol（默认）；absolute：不对齐，原始坐标直接检查")
    ap.add_argument("--tol", type=float, default=0.5, help="对应原子距离阈值（Å），默认 0.5")
    ap.add_argument("--max-rmsd", type=float, default=1.0, help="最大 RMSD（Å），默认 1.0")
    ap.add_argument("--topk", type=int, default=200, help="输出前 K 条（按 RMSD 升序、重合率降序），默认 200")
    ap.add_argument("--method", choices=["hash", "brute"], default="hash", help="匹配方式：hash（默认）或 brute")
    ap.add_argument("--bin", type=float, default=None,
                    help="哈希量化 bin（Å）；rigid 下作用于成对距离；absolute 下作用于坐标。默认取 tol。")
    ap.add_argument("--no-progress", action="store_true", help="不显示进度条")
    ap.add_argument("-o", "--output", default="matches.csv", help="输出 CSV 文件名，默认 matches.csv")
    ap.add_argument("--save-pdb-dir", default=None, help="将通过筛选的匹配构象保存为 PDB 到该目录（每对两个文件：A 与 B）")
    ap.add_argument("--save-pdb-limit", type=int, default=None, help="最多保存前 N 对（默认与 topk 相同）")
    args = ap.parse_args()

    # 解析与检查
    idxs1 = parse_indices(args.idx1)
    idxs2 = parse_indices(args.idx2)
    if len(idxs1) == 0 or len(idxs2) == 0:
        print("ERROR: idx1/idx2 不能为空。", file=sys.stderr); sys.exit(2)
    if len(idxs1) != len(idxs2):
        print(f"ERROR: idx1({len(idxs1)}) 与 idx2({len(idxs2)}) 长度不一致。", file=sys.stderr); sys.exit(2)

    mol1 = load_mol_with_confs(args.sdf1)
    mol2 = load_mol_with_confs(args.sdf2)
    assert_index_range(mol1, idxs1, "idx1")
    assert_index_range(mol2, idxs2, "idx2")

    n1 = mol1.GetNumConformers()
    n2 = mol2.GetNumConformers()
    print(f"已读取：{args.sdf1} 构象数={n1}；{args.sdf2} 构象数={n2}；共有原子数={len(idxs1)}；模式={args.overlap_mode}/{args.method}",
          file=sys.stderr)

    # Windows 友情提示（hash 路径无需多进程；brute 在 Windows 上可能较慢）
    if platform.system().lower().startswith("win") and args.method == "brute":
        print("提示：Windows 环境下 brute 模式可能较慢，建议使用 --method hash。", file=sys.stderr)

    bin_size = args.bin if args.bin is not None else args.tol
    if bin_size <= 0:
        print("ERROR: --bin 必须 > 0（或省略以使用 --tol）", file=sys.stderr); sys.exit(2)

    # 运行
    if args.method == "hash":
        results = run_hash(mol1, mol2, idxs1, idxs2, args.tol, args.max_rmsd, args.topk,
                           args.no_progress, bin_size, args.overlap_mode)
    else:
        results = run_brute(mol1, mol2, idxs1, idxs2, args.tol, args.max_rmsd, args.topk,
                            args.no_progress, args.overlap_mode)

    # 输出与保存
    save_limit = args.save_pdb_limit if args.save_pdb_limit is not None else args.topk
    save_results_and_pdb(results or [], args.output, mol1, mol2, args.save_pdb_dir, save_limit)

    if results:
        print(f"完成：输出 {len(results)} 条匹配到 {args.output}", file=sys.stderr)
        if args.save_pdb_dir:
            print(f"PDB 已写入目录：{args.save_pdb_dir}（最多 {save_limit} 对）", file=sys.stderr)
    else:
        print("未找到满足条件的匹配（已输出仅含表头的 CSV）。", file=sys.stderr)


if __name__ == "__main__":
    main()

