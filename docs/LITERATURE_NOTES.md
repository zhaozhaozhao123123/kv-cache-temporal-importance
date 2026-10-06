# 大模型推理缓存研究笔记

PagedAttention 主要研究推理服务中的缓存组织与内存利用。KIVI、KVQuant 研究缓存数值表示与量化精度。token 重要性分数描述查询对缓存位置的依赖。三者属于不同层次：低注意力不自动等于可以安全删除，减少缓存字节数也不自动意味着端到端加速。

## PagedAttention

逻辑 KV 序列被划分成固定大小块，再映射到非连续物理块，以支持动态长度和共享。本项目仅按连续 token 位置分块统计注意力之和，没有物理块分配、引用计数、写时复制或专用内核。

原文：https://arxiv.org/abs/2309.06180

## KIVI

Key 的大值倾向于特定通道，Value 更适合按 token 处理。Key per-channel 固定通道并跨时间位置估计尺度，Value per-token 固定 token 并跨特征通道估计尺度。近期残差保留用于降低近端量化影响。本项目验证非对称量化轴与保护范围排除，没有实现真实压缩缓存的流式追加、分组刷出或内核。

原文：https://proceedings.mlr.press/v235/liu24bz.html

## KVQuant

需要区分 Pre-RoPE Key 量化、非均匀码本、敏感度校准、dense-and-sparse 离群值表示。Pre-RoPE 关注位置旋转前的分布；Fisher 加权校准与码本学习不等同于 min-max 四舍五入。稀疏表示还需要索引和数值存储，不能仅按位宽估算开销。

本项目只实现逐组离群值全精度恢复的教学对照，不包含原版 Pre-RoPE、NUQ 或 CUDA 稀疏内核。

原文：https://github.com/SqueezeAILab/KVQuant

## 可检验假设

查询主题切换时，EMA 或窗口分数可能比累计分数更及时更新保护位置。验证必须固定保护 token 预算、位宽和分组大小，并用多种子、多任务报告误差分布。注意力代理与实际删除影响的相关性仍需验证。当前合成数据仅检验机制，不提供真实模型的普适结论。
