本项目致力于开发一个辅助开发者合入分支代码的机器人*MergeBot*。

- llama3：保存了用于生成的大语言模型
- script：用于抽取合入冲突场景的Python脚本
- dataset：合入冲突数据集
    - {project}
        - {commit}
            - {conflict chunk}
- merge.py