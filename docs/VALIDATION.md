# 验证记录

本地 CPU 环境：Windows，Python 3.12.10，NumPy 2.5.3，Matplotlib 3.11.2，pytest 9.1.1，PyTorch 2.14.1+cpu，Transformers 4.57.6。

`python -m pytest -q`：14 passed。包含随机初始化的小型 Qwen3 接口测试，无需下载模型权重。该测试对比递增缓存与完整前向的最后一查询注意力，验证缓存长度、位置和观测矩阵，不验证预训练模型质量。

`python -m kv_lab.demo --seed 42 --steps 48 --out examples/demo`：48 步、12 组配置、576 行记录。每组汇总排除没有历史分数的第 0 步，剩余 47 步。

未运行真实预训练 Qwen3-1.7B；未测量吞吐、显存、困惑度、任务正确率及多种子对照。

仅安装 requirements.txt 时，核心测试通过，模型测试明确 skipped。GitHub Actions 使用核心依赖并运行短模拟演示，执行状态以仓库 Actions 为准。本地通过不等同于远程 CI 成功。
