---
title: "EDITIKZ-SCIENTIFIC-FIGURE-EDITING-FROM-REVISION-TRAJECTORIES"
source: https://arxiv.org/pdf/2609.01409v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:24:38"
---

# 论文速读：EDITIKZ-SCIENTIFIC-FIGURE-EDITING-FROM-REVISION-TRAJECTORIES

## 一句话总结
论文从 arXiv、GitHub 与 TeX SE 的自然修订轨迹中挖掘 TikZ 科学插图编辑对，构建大规模数据集 DaEdiTikZ 及人工精校基准 DaEdiTikZ-Bench，并训练联合重构与编辑 SFT、结合 GDPO 多奖励 RL 的 EdiTikZ-9B 开源模型，在自动与人工评测上均超越或持平 GPT-5.6-Sol，且泛化至复杂 OOD 图表仍保持竞争力。

## 研究问题与动机
- **科学图编辑能力未被充分探索**：现有 VLM 在图生成上表现强劲，但满足出版要求的精确迭代修改（尤其是 TikZ 代码级编辑）缺乏有效训练数据与系统方法。
- **现有方法依赖昂贵闭源智能体或合成数据**：如 AutoFigure-Edit、DisciplineGen-1M 等工作多构建合成监督或依赖商业代理系统，成本高且缺乏真实专家修订分布。
- **自然修订轨迹蕴含专家决策却长期闲置**：论文撰写、版本迭代与社区讨论中产生的图稿演变天然形成 plausible edit pairs，可作为低成本、高保真的可扩展训练信号。

## 核心贡献（创新点）
1. 提出从真实科研仓库挖掘修订轨迹的可扩展监督框架，构建 DaEdiTikZ 数据集；与依赖合成数据或模板构造修改的既有工作不同，本框架直接利用人类专家在版本迭代中留下的真实编辑轨迹。
2. 发布 DaEdiTikZ-Bench 人工精校基准（790 实例）；与 VisEditBench、Diagram-MMU 等依赖自动指令或模板修改的评测集相比，本基准经多人交叉核校，显著降低数据污染与合成偏差。
3. 设计编辑专属的联合 SFT 与 GDPO 多奖励 RL 训练流程；与以往仅优化代码相似度或单一渲染奖励的方法不同，本文通过冻结视觉编码器复用感知奖励，并以独立归一化方式融合离散指令遵循信号，避免多目标梯度冲突。
4. 推出紧凑的 4B/9B 开源 EdiTikZ 模型；与现有依赖昂贵闭源智能体或超大参数商用模型的工作相比，本文证明小型开源自监督模型经针对性后期训练即可在编辑任务上达到同等甚至更优水平。

## 方法详解
- **数据收集与配对过滤**：扩展 DaTikZ-V4 收集 arXiv 历史版本、GitHub 仓库与 TeX SE 讨论中的 TikZ 代码，经 TikZilla 预处理标准化后，按科学上下文分组；使用 DeTikZify-V2 图像编码器计算组内余弦相似度，人工验证各相似度区间不可信变换比例，保留 implausible < 15% 的区间，最终获得 391K 编辑对。
- **指令合成**：将双向配对输入 Qwen3.6-27B（条件含渲染图与 TikZ 代码），要求输出原子编辑三元组（intent: add/remove/modify, operation: text/geometry/data/style/structure 等），仅当双向均被判为 ok 时保留，得到 781K 有向编辑轨迹，平均每条含 4.2 个原子编辑。
- **联合 SFT 损失**：模型在编辑数据 $(
