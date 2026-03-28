
import os
import json
import numpy as np
import subprocess

def get_struc_seq(foldseek, 
                  path, 
                  chains: list = None, 
                  process_id: int = 0, 
                  plddt_path: str = None, 
                  plddt_threshold: float = 70.) -> dict:
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"Pdb file not found: {path}"
    assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"
    
    tmp_save_base = f"get_struc_seq_{process_id}"
    cmd = [foldseek, "structureto3didescriptor", "-v", "0", "--threads", "1", "--chain-name-mode", "1", path, tmp_save_base]

    # 使用 subprocess 运行命令
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        print(f"Error running foldseek: {process.stderr.decode()}")
        return {}

    # 读取和处理所有相关的输出文件
    seq_dict = {}
    for file_name in os.listdir("."):
        if file_name.startswith(tmp_save_base) and not file_name.endswith(".dbtype"):
            with open(file_name, "r") as file:
                # 处理每个文件的内容
                # 这里需要根据实际文件格式来调整
                for line in file:
                    desc, seq, struc_seq = line.split("\t")[:3]
                    if plddt_path is not None:
                        with open(plddt_path, "r") as r:
                            plddts = np.array(json.load(r)["confidenceScore"])
                            indices = np.where(plddts < plddt_threshold)[0]
                            np_seq = np.array(list(struc_seq))
                            np_seq[indices] = "#"
                            struc_seq = "".join(np_seq)

                    name_chain = desc.split(" ")[0]
                    chain = name_chain.split("_")[-1]

                    if chains is None or chain in chains:
                        if chain not in seq_dict:
                            combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                            seq_dict[chain] = (seq, struc_seq, combined_seq)

            # 删除处理过的文件
            os.remove(file_name)

    return seq_dict

# 示例调用（请根据需要进行修改）
if __name__ == '__main__':
    foldseek = "/bin/foldseek"  # 更新为 foldseek 的实际路径
    # test_path = "/path/to/pdb_file.pdb"  # 更新为 PDB 文件的实际路径
    plddt_path = "/path/to/plddt_file.json"  # 更新为 PLDDT 文件的实际路径（如果有的话）
    res = get_struc_seq(foldseek, test_path, plddt_path=plddt_path, plddt_threshold=70.)
    # print(res)


