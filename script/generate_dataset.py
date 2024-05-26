import pandas as pd
import os
from git import Repo
from git import IndexFile


def get_merge_conflict_commits(repo_name):
    # 初始化Git仓库对象
    merge_num = 0
    repo = Repo(f"cases/{repo_name}")
    # 存储存在合并冲突的提交哈希值的列表
    conflict_commits = []

    # 遍历所有提交
    with open('conflict_commits.txt', 'w', encoding='utf-8') as f:
        for commit in repo.iter_commits():
            # 检查提交是否存在合并冲突
            if len(commit.parents) != 2:
                continue
            merge_num += 1
            base = repo.merge_base(commit.parents[0], commit.parents[1])
            virtual_merge = IndexFile.from_tree(repo, base, commit.parents[0], commit.parents[1])
            actual_merge = IndexFile.from_tree(repo, commit)

            if not compare_index_files(virtual_merge.entries, actual_merge.entries):
                print(commit.hexsha, file=f)
                print(commit.message, file=f)
                conflict_commits.append(commit.hexsha)
                write_conflicts_to_file(f"dataset/{repo_name}", commit.hexsha, repo, virtual_merge)

        print(merge_num)
        print(len(conflict_commits))
    return conflict_commits


def compare_index_files(index1, index2):
    # 比较两个 IndexFile 对象中的每个条目
    if set(index1.keys()) != set(index2.keys()):
        return False

    # 比较每个键对应的值
    for path in index1.keys():
        # 检查文件路径是否相同
        if index1[path].path != index2[path].path:
            return False
        # 检查文件模式是否相同
        if index1[path].mode != index2[path].mode:
            return False
        # 检查文件哈希值是否相同
        if index1[path].hexsha != index2[path].hexsha:
            return False

    return True


def write_conflicts_to_file(dataset_path, hexsha, repo: Repo, conflict_index_files: IndexFile):
    target_dir = f"{dataset_path}/{hexsha[:6]}"
    os.makedirs(target_dir, exist_ok=True)

    conflicted_files = {}
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

        base_content = repo.git.cat_file("blob", base_sha)
        ours_content = repo.git.cat_file("blob", ours_sha)
        theirs_content = repo.git.cat_file("blob", theirs_sha)
        
        target_path = os.path.join(target_dir, path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        with open(target_path, "w", encoding="utf8") as file:
            file.write(f'<<<<<<<<< OUR VERSION\n{ours_content}\n=========== BASE VERSION\n{base_content}===========\n{theirs_content}\n>>>>>>>>> THEIR VERSION\n')


if __name__ == '__main__':
    get_merge_conflict_commits('fastjson')
    
