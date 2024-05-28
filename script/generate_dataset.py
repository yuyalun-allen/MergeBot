import os
import difflib
import subprocess
import tempfile
from git import Repo
from git import IndexFile


def get_merge_conflict_commits(repo_name):
    # 初始化Git仓库对象
    merge_num = 0
    repo = Repo(f"cases/{repo_name}")
    # 存储存在合并冲突的提交哈希值的列表
    conflict_commits = []

    # 遍历所有提交
    for commit in repo.iter_commits():
        # 检查提交是否存在合并冲突
        if len(commit.parents) != 2:
            continue
        merge_num += 1
        base = repo.merge_base(commit.parents[0], commit.parents[1])
        virtual_merge = IndexFile.from_tree(repo, base, commit.parents[0], commit.parents[1])
        actual_merge = IndexFile.from_tree(repo, commit)

        if not compare_index_files(virtual_merge.entries, actual_merge.entries):
            print(commit.hexsha)
            print(commit.message)
            conflict_commits.append(commit.hexsha)
            write_conflicts_to_file(f"dataset/{repo_name}", commit.hexsha, repo, virtual_merge)

    print(merge_num)
    print(len(conflict_commits))
    return conflict_commits


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

   
    target_dir = f"{dataset_path}/{hexsha[:6]}"

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


if __name__ == '__main__':
    get_merge_conflict_commits('fastjson')
