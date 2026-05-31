# Scripts

## backbone_to_sdf.py

从 PDB 文件中提取骨架原子构象 [N, C, CA, O]，按残基逐条写入 SDF 文件。

**用法：**
```
python backbone_to_sdf.py --pdb <input.pdb> --out <out.sdf> [--chain <filter>] [--resn <filter>] [--resi <filter>]
```

- `--pdb` — 输入 PDB 文件
- `--out` — 输出 SDF 文件
- `--chain` — 链过滤：`ALL`（默认）或逗号分隔，如 `A,B`
- `--resn` — 残基名过滤（3 字母）：`ALL` 或逗号分隔，如 `GLY,ALA`
- `--resi` — 残基编号过滤：`ALL` 或列表/区间，如 `10,15,20-30`

对每个含有完整 [N, C, CA, O] 原子的匹配残基：
1. 择优选择每个原子的 altLoc / occupancy
2. 构建 4 原子骨架片段，键连为 N-CA, CA-C, C-O
3. 将构象写入 SDF，附带元数据（chain, resname, resseq, icode）

示例：
```
python backbone_to_sdf.py --pdb 1abc.pdb --out backbone.sdf --chain A --resn GLY,ALA --resi 5-12
```

---

## check_lig_overlap.py

检测参考配体模板与一组 SMILES 之间的最大公共子结构（MCS）重叠，并绘制化学结构图（高亮重叠部分）。

**用法：**
```
python check_lig_overlap.py <ref.sdf> <SMILES.txt>
```

- `ref.sdf` — 包含参考配体模板的 SDF 文件（取第一个分子）
- `SMILES.txt` — 文本文件，每行格式为 `<index> <SMILES>`

对每个 SMILES 分子：
1. 找到分子与模板之间的 MCS
2. 打印重叠核心子结构
3. 绘制分子结构式并高亮 MCS 匹配原子，保存为 `overlap_images/<index>_overlap.png`

---

## sample_ligand_w_ref.py

通过参考模板约束嵌入，为配体分子生成多样化的 3D 构象。

**用法：**
```
python sample_ligand_w_ref.py <ref.sdf> <SMILES.txt>
```

- `ref.sdf` — 包含参考配体模板的 SDF 文件
- `SMILES.txt` — 文本文件，每行格式为 `<index> <SMILES>`

对每个 SMILES 条目：
1. 找到分子与模板之间的公共核心（MCS 阈值 0.9）
2. 最多运行 10,000 次约束嵌入，仅保留 EmbedRMS < 0.12 的构象
3. 按 RMSD（阈值 0.6 Å）过滤以确保构象多样性
4. 将接受的构象写入 `U<index>/confs.sdf`

若连续 100 次迭代未接受新构象，则提前终止。

---

## refine_conformers.py

使用 UFF 力场进行带位置约束的构象优化，并按能量过滤。

**用法：**
```
python refine_conformers.py <conf1.sdf> [conf2.sdf ...]
```

- 接受一个或多个 SDF 文件作为输入

对每个构象：
1. 计算初始 UFF 能量
2. 对所有原子施加位置约束（最大位移 0.1 Å，力常数 100）
3. 进行能量最小化（最多 200 次迭代）并记录优化后能量

输出 `refined-confs.sdf`，仅保留能量低于 `median + std`（std 上限 20 kcal/mol）的构象。
