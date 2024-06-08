import os
import subprocess
import tempfile
from pathlib import Path

from git import Repo
from git import IndexFile
from datasets import Dataset


def get_merge_conflict_and_resolution(repo_name):
    # 初始化Git仓库对象
    repo = Repo(f"cases/{repo_name}")

    # 遍历所有提交
    for commit in repo.iter_commits():
        # 检查提交是否存在合并冲突
        if len(commit.parents) != 2:
            continue
        base = repo.merge_base(commit.parents[0], commit.parents[1])
        virtual_merge = IndexFile.from_tree(repo, base, commit.parents[0], commit.parents[1])
        actual_merge = IndexFile.from_tree(repo, commit)

        if not compare_index_files(virtual_merge.entries, actual_merge.entries):
            print(commit.hexsha)
            print(commit.message)
            write_conflicts_to_file(f"dataset/{repo_name}", commit.hexsha, repo, virtual_merge)


def compare_index_files(index1, index2):
    # 比较两个 IndexFile 对象中的每个条目
    if set(index1.keys()) != set(index2.keys()):
        print("keys not match!")
        print(len(index1.keys()))
        print(len(index2.keys()))
        return False

    # 比较每个键对应的值
    for path in index1.keys():
        # 检查文件路径是否相同
        if index1[path].path != index2[path].path:
            print(f"path:{path}")
            print(index1[path].path, index2[path].path)
            return False
        # 检查文件模式是否相同
        if index1[path].mode != index2[path].mode:
            print(f"path:{path}")
            print(index1[path].mode, index2[path].mode)
            return False
        # 检查文件哈希值是否相同
        if index1[path].hexsha != index2[path].hexsha:
            print(f"path:{path}")
            print(index1[path].hexsha, index2[path].hexsha)
            return False

    return True


def write_conflicts_to_file(dataset_path, hexsha, repo: Repo, conflict_index_files: IndexFile):
    target_dir = f"{dataset_path}/{hexsha[:6]}"
    conflicted_files = {}

    def read_blob_content(repo, sha):
        """Read the content of a blob by its SHA-1 and return it as a list of lines."""
        blob = repo.git.cat_file('blob', sha)
        return blob.splitlines(keepends=True)

    def generate_conflict(base_file, ours_file, theirs_file):
        with tempfile.TemporaryDirectory() as tempdir:
            # Create temporary files for base, ours, and theirs
            base_path = os.path.join(tempdir, 'base')
            ours_path = os.path.join(tempdir, 'ours')
            theirs_path = os.path.join(tempdir, 'theirs')

            # Write contents to temporary files
            with open(base_path, 'w') as f:
                f.write(base_file)
            with open(ours_path, 'w') as f:
                f.write(ours_file)
            with open(theirs_path, 'w') as f:
                f.write(theirs_file)

            try:
            # Use git merge-file to merge the files
                subprocess.run(['git', 'merge-file', '-p', '--diff3', ours_path, base_path, theirs_path], 
                                check=True, 
                                text=True,
                                capture_output=True).stdout
                return None

            except subprocess.CalledProcessError as e:
                return e.stdout

    for entry in conflict_index_files.entries.values():
        if entry.stage == 0:
            continue

        if entry.path not in conflicted_files:
            conflicted_files[entry.path] = {}

        conflicted_files[entry.path][entry.stage] = entry.hexsha

    # 生成冲突标记的文件内容
    for path, stages in conflicted_files.items():
        if len(stages) != 3:
            continue  # 需要三个阶段的条目

        base_sha = stages[1]
        ours_sha = stages[2]
        theirs_sha = stages[3]

        base_content = read_blob_content(repo, base_sha)
        ours_content = read_blob_content(repo, ours_sha)
        theirs_content = read_blob_content(repo, theirs_sha)
        
        target_path = os.path.join(target_dir, path)
        conflict_content = generate_conflict("".join(base_content), "".join(ours_content), "".join(theirs_content))
        if not conflict_content:
            continue
        file_name, ext = os.path.splitext(target_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(f"{file_name}-conf{ext}", "w", encoding="utf-8") as file:
            file.write(conflict_content)
        with open(f"{file_name}-reso{ext}", "w", encoding="utf-8") as file:
            commit = repo.commit(hexsha)
            file_content = (commit.tree / path).data_stream.read().decode("utf-8")
            file.write(file_content)



def load_merge_conflict_and_resolution(path, split):
    c_n_r = []
    for repo in os.listdir(path):
        repo_path = os.path.join(path, repo)
        for commit in os.listdir(repo_path):
            commit_path = os.path.join(repo_path, commit)
            files_data = {}
            files = [str(p) for p in Path(commit_path).rglob('*') if p.is_file()]
            for file in files:
                if '-reso.' in file:
                    with open(file, 'r', encoding='utf-8') as f:
                        files_data['resolution'] = f.read()
                elif '-conf.' in file:
                    with open(file, 'r', encoding='utf-8') as f:
                        files_data['conflict'] = f.read()
            if files_data:  # 只有在有有效数据时才添加
                c_n_r.append(files_data)
    if split == "train":
        return c_n_r[:int(len(c_n_r) * 0.8)]
    else:
        return c_n_r[int(len(c_n_r) * 0.8):]



def preprocess_merge_conflict_and_resolution(dataset_config, tokenizer, split):
    dataset = Dataset.from_list(load_merge_conflict_and_resolution("dataset", split))
    prompt = (
        f"Resolve this merge conflict:\n{{conflict}}\n---\nResolution:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(conflict=sample["conflict"]),
            "resolution": sample["resolution"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        resolution = tokenizer.encode(sample["resolution"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + resolution,
            "attention_mask" : [1] * (len(prompt) + len(resolution)),
            "labels": [-100] * len(prompt) + resolution,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset


if __name__ == '__main__':
    preprocess_merge_conflict_and_resolution(None, None, "train")
