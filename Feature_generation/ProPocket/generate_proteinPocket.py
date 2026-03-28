import os
import pickle
import gemmi
import numpy as np
from tqdm import tqdm

# A lightweight pocket finder implementation that does NOT depend on deepchem/jax.
# This is a heuristic replacement of ConvexHullPocketFinder: it finds residues whose
# CA atoms are in "low local density" regions (possible surface pockets) and groups
# nearby such residues into pockets. For many use-cases this gives usable pocket
# residue indices without heavy dependencies.

class Box:
    def __init__(self, x_range, y_range, z_range):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

class SimplePocketFinder:
    def __init__(self, neighbor_radius=8.0, pocket_radius=8.0):
        # neighbor_radius: radius to count neighbors for density estimation (Å)
        # pocket_radius: radius to join pocket residues into a group (Å)
        self.neighbor_radius = neighbor_radius
        self.pocket_radius = pocket_radius

    def _ca_positions(self, mol):
        # return list of CA coordinates and mapping atom idx -> residue index
        conf = mol.GetConformer()
        positions = []
        atom_to_residx = []
        res_map = []
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is None:
                continue
            # We consider CA atoms only
            if info.GetName().strip() != "CA":
                continue
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            atom_to_residx.append(info.GetSerialNumber() if info.GetSerialNumber() is not None else atom.GetIdx())
            res_map.append((info.GetResidueNumber(), info.GetName().strip(), info.GetResidueName().strip()))
        if len(positions) == 0:
            return np.zeros((0, 3)), res_map
        return np.array(positions), res_map

    def find_pockets(self, cif_path):
        """Return a list of Box objects. Each box has x_range,y_range,z_range attributes.
        The algorithm is heuristic:
        1. Load CA coordinates for the structure.
        2. For each CA, count neighbors within neighbor_radius -> density.
        3. Mark CA atoms whose neighbor count is <= (median - 0.5*std) as candidate pocket residues.
        4. Group candidate residues by connectivity (distance < pocket_radius) -> pockets.
        5. For each pocket group compute axis-aligned bounding box and return as Box.
        """
        if not os.path.exists(cif_path):
            return []
        try:
            structure = gemmi.read_structure(cif_path)
        except Exception:
            return []

        ca_positions = []
        for model in structure:
            for chain in model:
                for residue in chain.get_polymer():
                    atom = None
                    for a in residue:
                        if a.name.strip() == "CA":
                            atom = a
                            break
                    if atom is not None:
                        pos = atom.pos
                        ca_positions.append([pos.x, pos.y, pos.z])
        if len(ca_positions) == 0:
            return []
        P = np.array(ca_positions)  # (N,3)
        N = P.shape[0]

        # compute distance matrix in a memory-efficient way
        # neighbor counts within neighbor_radius
        neigh_r = float(self.neighbor_radius)
        neigh_counts = np.zeros(N, dtype=int)
        for i in range(N):
            dif = P - P[i]
            d2 = (dif * dif).sum(axis=1)
            neigh_counts[i] = int((d2 <= neigh_r * neigh_r).sum())

        # heuristic threshold for pocket candidates
        med = np.median(neigh_counts)
        std = np.std(neigh_counts)
        threshold = med - 0.5 * std
        candidate_mask = neigh_counts <= threshold
        candidate_idx = np.where(candidate_mask)[0].tolist()

        if len(candidate_idx) == 0:
            # fallback: take the lowest 5% density residues
            k = max(1, int(max(1, 0.05 * N)))
            candidate_idx = np.argsort(neigh_counts)[:k].tolist()

        # group candidates by proximity (graph BFS)
        visited = set()
        groups = []
        join_r2 = float(self.pocket_radius) ** 2
        for idx in candidate_idx:
            if idx in visited:
                continue
            stack = [idx]
            group = []
            visited.add(idx)
            while stack:
                cur = stack.pop()
                group.append(cur)
                # find neighbors among candidate_idx within pocket_radius
                for j in candidate_idx:
                    if j in visited:
                        continue
                    d2 = np.sum((P[cur] - P[j]) ** 2)
                    if d2 <= join_r2:
                        visited.add(j)
                        stack.append(j)
            groups.append(sorted(group))

        boxes = []
        for g in groups:
            coords = P[g]
            xmin, ymin, zmin = coords.min(axis=0)
            xmax, ymax, zmax = coords.max(axis=0)
            # extend box a little bit
            pad = 2.0
            box = Box((xmin - pad, xmax + pad), (ymin - pad, ymax + pad), (zmin - pad, zmax + pad))
            boxes.append(box)

        return boxes


def extract_pocket_indices_from_cif(cif_path):
    """
    使用 SimplePocketFinder 找到 CIF 文件的口袋残基索引
    """
    pk = SimplePocketFinder()
    boxes = pk.find_pockets(cif_path)

    if not os.path.exists(cif_path):
        return []

    try:
        structure = gemmi.read_structure(cif_path)
    except Exception:
        return []

    ca_positions = []
    residue_ids = []
    for model in structure:
        for chain in model:
            for residue in chain.get_polymer():
                atom = None
                for a in residue:
                    if a.name.strip() == "CA":
                        atom = a
                        break
                if atom is not None:
                    pos = atom.pos
                    ca_positions.append([pos.x, pos.y, pos.z])
                    # Use a tuple of (chain_id, residue_number) as residue id
                    residue_ids.append((chain.name, residue.seqid.num))

    if len(ca_positions) == 0:
        return []

    positions = np.array(ca_positions)

    pocket_indices = set()
    for box in boxes:
        x_min, x_max = box.x_range
        y_min, y_max = box.y_range
        z_min, z_max = box.z_range
        for idx, pos in enumerate(positions):
            if x_min < pos[0] < x_max and y_min < pos[1] < y_max and z_min < pos[2] < z_max:
                pocket_indices.add(idx)

    return sorted(list(pocket_indices))


def generate_pocket_embeddings(dataset, residue_embed_path, cif_dir, output_path):
    """
    根据 residue embedding + CIF 结构，生成 pocket embedding.pkl
    """
    # 读取残基级 embedding
    with open(residue_embed_path, "rb") as f:
        residue_embeds = pickle.load(f)

    pocket_embeds = {}

    for prot_id, value in tqdm(residue_embeds.items(), desc="Processing proteins"):
        vec, mat, mask = value
        eff_len = int(mask.sum())         # count of valid (non-padding) residues
        emb = mat[:eff_len, :]            # truncate padded matrix
        cif_path = os.path.join(cif_dir, f"{prot_id}.cif")
        if not os.path.exists(cif_path):
            print(f"[Warning] 缺少 CIF 文件: {prot_id}")
            continue

        pocket_indices = extract_pocket_indices_from_cif(cif_path)
        if not pocket_indices:
            print(f"[Warning] 未找到口袋: {prot_id} → 保存整条序列 embedding")
            pocket_indices = list(range(emb.shape[0]))

        # protect indexing bounds
        pocket_indices = [int(i) for i in pocket_indices if i >= 0 and i < emb.shape[0]]
        if len(pocket_indices) == 0:
            print(f"[Warning] 口袋索引越界或为空: {prot_id} → 保存整条序列 embedding")
            pocket_indices = list(range(emb.shape[0]))

        # 提取 pocket 残基的 embedding
        pocket_emb = emb[pocket_indices, :].astype(np.float32)
        # 保存真正的口袋 embedding
        pocket_embeds[prot_id] = pocket_emb

        print(f"{prot_id}: residue {emb.shape[0]} → pocket {pocket_emb.shape[0]}")

    # 保存
    with open(output_path, "wb") as f:
        pickle.dump(pocket_embeds, f)


    print(f"save: {output_path}")


if __name__ == "__main__":
    dataset = "Human"  # ['DrugBank', 'BioSNAP', 'Human']
    residue_embed_path = f"./../../Datasets/{dataset}/feature/sa_SaProt1280.pkl"
    cif_dir = f"./../../Datasets/{dataset}/feature/CIF_AF2"
    output_path = f"./../../Datasets/{dataset}/feature/sa_SaProt1280_pockets.pkl"

    generate_pocket_embeddings(dataset, residue_embed_path, cif_dir, output_path)
