#!/usr/bin/env python3
"""
固定两端端基，采样中间linker构象的工具
用法: python script.py fragment1.mol fragment2.mol "SMILES" output.sdf
"""

import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, Lipinski, rdForceFieldHelpers
from rdkit.ML.Cluster import Butina
from collections import defaultdict


def load_fragment_with_coords(mol_file):
    """从mol文件读取fragment及其3D坐标"""
    mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    if mol is None:
        raise ValueError(f"无法读取文件: {mol_file}")
    return mol


def find_non_overlapping_matches(full_mol, frag1, frag2):
    """
    在完整分子中找到两个fragment的非重叠匹配
    返回: (match1, match2) 元组，每个都是原子索引的元组
    """
    # 启用手性匹配
    matches1 = full_mol.GetSubstructMatches(frag1, useChirality=True)
    matches2 = full_mol.GetSubstructMatches(frag2, useChirality=True)
    
    # 如果直接匹配失败，尝试去氢后匹配，但需要映射回含氢索引
    if not matches1 or not matches2:
        frag1_no_h = Chem.RemoveHs(frag1)
        frag2_no_h = Chem.RemoveHs(frag2)
        full_no_h = Chem.RemoveHs(full_mol)
        
        matches1_no_h = full_no_h.GetSubstructMatches(frag1_no_h, useChirality=True)
        matches2_no_h = full_no_h.GetSubstructMatches(frag2_no_h, useChirality=True)
        
        if not matches1_no_h or not matches2_no_h:
            raise ValueError("无法在完整分子中找到fragment的子结构匹配")
        
        # 建立去氢分子到含氢分子的原子映射
        full_h_to_no_h = {}
        for atom in full_mol.GetAtoms():
            if atom.GetAtomicNum() != 1:  # 非氢原子
                full_h_to_no_h[atom.GetIdx()] = len([a for a in full_mol.GetAtoms()[:atom.GetIdx()] if a.GetAtomicNum() != 1])
        
        full_no_h_to_h = {v: k for k, v in full_h_to_no_h.items()}
        
        # 将去氢匹配映射回含氢索引
        matches1 = [tuple(full_no_h_to_h[i] for i in match) for match in matches1_no_h]
        matches2 = [tuple(full_no_h_to_h[i] for i in match) for match in matches2_no_h]
        
        print("警告: 使用去氢匹配并映射回含氢索引")
    
    # 找到不重叠的匹配对
    for match1 in matches1:
        set1 = set(match1)
        for match2 in matches2:
            set2 = set(match2)
            # 检查是否重叠
            if not set1.intersection(set2):
                return match1, match2
    
    raise ValueError("无法找到两个fragment的非重叠匹配，请检查分子结构")


def align_fragments_to_full_molecule(full_mol, frag1, frag2):
    """
    将两个fragment的坐标映射到完整分子上
    返回: coordMap字典，key是原子索引，value是3D坐标
    """
    coordMap = {}
    
    # 确保完整分子有初始构象（即使是随机的）
    if full_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(full_mol, randomSeed=42)
    
    # 找到非重叠匹配
    match1, match2 = find_non_overlapping_matches(full_mol, frag1, frag2)
    
    print(f"Fragment 1 匹配到原子索引: {match1[:5]}..." if len(match1) > 5 else f"Fragment 1 匹配到原子索引: {match1}")
    print(f"Fragment 2 匹配到原子索引: {match2[:5]}..." if len(match2) > 5 else f"Fragment 2 匹配到原子索引: {match2}")
    
    # 映射fragment1的坐标
    conf1 = frag1.GetConformer()
    for frag_idx, full_idx in enumerate(match1):
        pos = conf1.GetAtomPosition(frag_idx)
        coordMap[full_idx] = pos
    
    # 映射fragment2的坐标
    conf2 = frag2.GetConformer()
    for frag_idx, full_idx in enumerate(match2):
        pos = conf2.GetAtomPosition(frag_idx)
        coordMap[full_idx] = pos
    
    # 识别linker区域
    fixed_atoms = set(match1) | set(match2)
    linker_atoms = set(range(full_mol.GetNumAtoms())) - fixed_atoms
    
    print(f"\n固定原子数: {len(fixed_atoms)}")
    print(f"Linker原子数: {len(linker_atoms)}")
    print(f"可旋转键数: {Lipinski.NumRotatableBonds(full_mol)}")
    
    return coordMap, list(fixed_atoms), list(linker_atoms)


def generate_constrained_conformers(mol, coordMap, fixed_atoms, num_confs=500):
    """
    生成约束构象，固定端基坐标
    """
    print(f"\n开始生成 {num_confs} 个构象...")
    
    # 尝试使用新版API（coordMap作为参数）
    try:
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_confs,
            randomSeed=42,
            pruneRmsThresh=0.5,
            useRandomCoords=True,
            numThreads=0,
            coordMap=coordMap
        )
        print(f"使用coordMap参数成功生成 {len(conf_ids)} 个初始构象")
    except TypeError:
        # 降级到params方式
        print("尝试使用params.coordMap方式...")
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 0
        params.pruneRmsThresh = 0.5
        params.useRandomCoords = True
        params.coordMap = coordMap
        
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_confs,
            params=params
        )
        print(f"成功生成 {len(conf_ids)} 个初始构象")
    
    if len(conf_ids) == 0:
        raise ValueError("构象生成失败！可能是约束过于严格或分子结构有问题")
    
    # 使用约束力场优化
    print("正在优化构象（保持端基固定）...")
    
    valid_conf_ids = []
    fixed_atoms_list = list(fixed_atoms)
    
    for conf_id in conf_ids:
        try:
            # 创建MMFF力场
            ff = AllChem.MMFFGetMoleculeForceField(
                mol,
                AllChem.MMFFGetMoleculeProperties(mol),
                confId=conf_id
            )
            
            # 对固定原子添加非常强的约束
            # 尝试不同的API
            for atom_idx in fixed_atoms:
                try:
                    # 尝试方法1: rdForceFieldHelpers
                    rdForceFieldHelpers.MMFFAddPositionConstraint(ff, atom_idx, 0.01, 100000.0)
                except AttributeError:
                    try:
                        # 尝试方法2: AllChem
                        AllChem.MMFFAddPositionConstraint(ff, atom_idx, 0.01, 100000.0)
                    except AttributeError:
                        # 尝试方法3: 直接方法调用
                        ff.MMFFAddPositionConstraint(atom_idx, 0.01, 100000.0)
            
            # 优化
            ff.Minimize(maxIts=1000)
            
            # 验证端基RMSD < 0.1 Å
            conf = mol.GetConformer(conf_id)
            max_deviation = 0.0
            
            for atom_idx in fixed_atoms:
                target_pos = coordMap[atom_idx]
                current_pos = conf.GetAtomPosition(atom_idx)
                deviation = target_pos.Distance(current_pos)
                max_deviation = max(max_deviation, deviation)
            
            if max_deviation < 0.1:
                valid_conf_ids.append(conf_id)
            else:
                print(f"  构象 {conf_id} 端基偏移过大 ({max_deviation:.3f} Å)，已丢弃")
            
        except Exception as e:
            print(f"警告: 构象 {conf_id} 优化失败: {e}")
            continue
    
    print(f"通过端基RMSD验证的构象数: {len(valid_conf_ids)}/{len(conf_ids)}")
    
    if len(valid_conf_ids) == 0:
        raise ValueError("没有构象通过端基RMSD < 0.1 Å 的验证！")
    
    return valid_conf_ids


def cluster_conformers(mol, conf_ids, linker_atoms, rmsd_threshold=0.2):
    """
    快速贪心聚类：遍历构象，如果与已选构象RMSD都大于阈值则保留
    """
    print(f"\n使用快速贪心聚类（RMSD阈值 {rmsd_threshold} Å）...")
    
    if len(conf_ids) == 0:
        return []
    
    # 按能量排序，优先保留低能量构象
    energies = []
    for conf_id in conf_ids:
        try:
            ff = AllChem.MMFFGetMoleculeForceField(
                mol,
                AllChem.MMFFGetMoleculeProperties(mol),
                confId=conf_id
            )
            energy = ff.CalcEnergy()
            energies.append((energy, conf_id))
        except:
            energies.append((float('inf'), conf_id))
    
    energies.sort()
    sorted_conf_ids = [conf_id for _, conf_id in energies]
    
    # 贪心选择：第一个直接选
    selected = [sorted_conf_ids[0]]
    
    # 遍历剩余构象
    for conf_id in sorted_conf_ids[1:]:
        # 检查与已选构象的RMSD
        is_unique = True
        for selected_conf_id in selected:
            # 只对linker原子计算RMSD（无对齐，更快）
            if linker_atoms:
                rmsd = AllChem.GetConformerRMS(
                    mol, conf_id, selected_conf_id,
                    atomIds=linker_atoms,
                    prealigned=False
                )
            else:
                rmsd = AllChem.GetConformerRMS(
                    mol, conf_id, selected_conf_id,
                    prealigned=False
                )
            
            if rmsd < rmsd_threshold:
                is_unique = False
                break
        
        if is_unique:
            selected.append(conf_id)
    
    print(f"快速聚类保留 {len(selected)} 个代表性构象")
    
    return selected


def save_conformers(mol, conf_ids, output_file):
    """保存构象到SDF文件"""
    writer = Chem.SDWriter(output_file)
    
    for idx, conf_id in enumerate(conf_ids):
        # 计算能量
        try:
            ff = AllChem.MMFFGetMoleculeForceField(
                mol,
                AllChem.MMFFGetMoleculeProperties(mol),
                confId=conf_id
            )
            energy = ff.CalcEnergy()
        except:
            energy = None
        
        # 设置属性
        mol.SetProp("_Name", f"Conf_{idx}")
        mol.SetProp("ConfID", str(conf_id))
        if energy is not None:
            mol.SetProp("Energy_kcal/mol", f"{energy:.2f}")
        
        writer.write(mol, confId=conf_id)
    
    writer.close()
    print(f"\n构象已保存到: {output_file}")


def main():
    if len(sys.argv) != 5:
        print("用法: python script.py fragment1.mol fragment2.mol 'SMILES' output.sdf")
        print("示例: python script.py frag1.mol frag2.mol 'c1ccccc1CCCc2ccccc2' linker_confs.sdf")
        sys.exit(1)
    
    frag1_file = sys.argv[1]
    frag2_file = sys.argv[2]
    smiles = sys.argv[3]
    output_file = sys.argv[4]
    
    print("=" * 60)
    print("固定端基的Linker构象采样工具")
    print("=" * 60)
    
    # 1. 读取fragments
    print(f"\n读取Fragment 1: {frag1_file}")
    frag1 = load_fragment_with_coords(frag1_file)
    print(f"  原子数: {frag1.GetNumAtoms()}")
    
    print(f"\n读取Fragment 2: {frag2_file}")
    frag2 = load_fragment_with_coords(frag2_file)
    print(f"  原子数: {frag2.GetNumAtoms()}")
    
    # 2. 从SMILES创建完整分子
    print(f"\n解析完整分子SMILES: {smiles}")
    full_mol = Chem.MolFromSmiles(smiles)
    if full_mol is None:
        raise ValueError("无效的SMILES字符串")
    
    full_mol = Chem.AddHs(full_mol)
    print(f"  完整分子原子数（含氢）: {full_mol.GetNumAtoms()}")
    
    # 3. 匹配并固定端基
    coordMap, fixed_atoms, linker_atoms = align_fragments_to_full_molecule(
        full_mol, frag1, frag2
    )
    
    # 4. 生成构象
    num_initial_confs = 10000 # 可以调整这个数字
    conf_ids = generate_constrained_conformers(
        full_mol, coordMap, fixed_atoms, num_confs=num_initial_confs
    )
    
    # 5. 聚类去冗余
    rmsd_threshold = 0.2  # 可以调整，越小越严格
    selected_conf_ids = cluster_conformers(
        full_mol, conf_ids, linker_atoms, rmsd_threshold
    )
    
    # 6. 保存结果
    save_conformers(full_mol, selected_conf_ids, output_file)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
