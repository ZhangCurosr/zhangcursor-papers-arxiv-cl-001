---
title: "EDITIKZ-SCIENTIFIC-FIGURE-EDITING-FROM-REVISION-TRAJECTORIES"
source: https://arxiv.org/pdf/2609.01409v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:24:34"
field: "多模态科学可视化"
keywords: ["科学图表编辑", "TikZ", "Vision-Language Model", "Reinforcement Learning", "Dataset Construction", "Multimodal Generation"]
innovations: ["从真实修订轨迹提取TikZ编辑监督", "联合重建与编辑的多任务SFT", "SelfSim与指令遵循双奖励GDPO优化"]
benchmarks: ["DaEdiTikZ-Bench", "SPIQA", "CharXiv"]
---

# 论文速读：EDITIKZ-SCIENTIFIC-FIGURE-EDITING-FROM-REVISION-TRAJECTORIES

## 一句话总结
本文提出了一种从真实科学修订轨迹中提取编辑监督信号的新范式，构建了大规模TikZ编辑数据集DaEdiTikZ（391K编辑对、781K编辑指令），并训练出紧凑的EdiTikZ模型（4B/9B），其9B版本在科学图表编辑任务上自动评测超越所有基线，人工评测超越GPT-5.6-Sol并与Gemini-3.1-Pro持平。

## 研究问题与动机
- **核心问题**：科学图表编辑（Scientific Figure Editing）是生成出版级图表的关键能力，但现有工作缺乏大规模真实编辑监督数据。
- **现有方法不足**：
  - 已有方法依赖昂贵的专有Agent系统（如AutoFigure-Edit）或仅聚焦评测基准构建。
  - 训练监督主要来源于合成编辑（如OCR-based synthetic edits），缺乏真实人类修订轨迹。
  - 现有TikZ工作多关注从零生成（text/image-to-TikZ），而非编辑已有图表。
- **关键洞察**：科学图表在论文修订、GitHub版本迭代、TeX Stack Exchange讨论中自然产生大量人类修订轨迹，可作为可扩展的监督信号源。

## 核心贡献（创新点）
1. **修订驱动的监督提取框架**：首次系统化地从arXiv、GitHub、TeX SE挖掘语义相似的图表对，构建大规模真实编辑监督。与合成数据方法的本质区别在于利用人类自然修订而非人工构造编辑指令。
2. **DaEdiTikZ数据集与基准**：发布包含391K可信TikZ编辑对（781K有向编辑轨迹）的大规模数据集，以及790个经人工精炼的DaEdiTikZ-Bench基准。区别于DisciplineGen-1M等合成数据集，本数据源自真实发布流程。
3. **编辑特定多任务后训练策略**：联合训练重建（reconstruction）与编辑（editing），并设计互补的多奖励RL（渲染保真度+指令遵循）。与纯生成RL的本质区别在于引入了source-conditioned VLM verifier验证原子编辑是否被正确应用。
4. **小型开源EdiTikZ模型**：训练4B和9B紧凑型Qwen3.5基模型，EdiTikZ-9B-RL在DaEdiTikZ-Bench上自动评测得分最高，人工评测超越GPT-5.6-Sol且与Gemini-3.1-Pro相当。与专用大模型/Agent系统的本质区别在于轻量架构即可匹敌闭源方案。

## 方法详解
- **数据集构建流程**：
  1. 从arXiv历史版本（91K投稿中38K含多版本TikZ）、GitHub仓库、TeX SE讨论线程收集图表。
  2. 使用DeTikZify-V2图像编码器计算组内余弦相似度，人工标注8个相似度区间过滤阈值（<15%不可信变换）。
  3. 用Qwen3.6-27B作为VLM，输入源/目标图像对及对应TikZ代码，合成双向编辑指令（原子化：intent×operation×description）。
  4. 要求双向均可信才保留，最终获得390,516对、781,032条有向轨迹，平均每轨迹4.2个原子编辑。

- **双阶段训练**：
  1. **联合SFT**：同时优化编辑损失$\mathcal{L}_{\text{edit}}$（输入$I_s, u \to y$）和重建损失（输入$I_t \to y$），各使用752K样本，强化共享图像-代码映射。
  2. **多奖励RL（GDPO）**：
     - 冻结SFT视觉编码器，提取patch embeddings $\mathbf{x}, \mathbf{z}$。
     - 渲染保真度奖励$\mathcal{R}_{\text{SSim}} = \text{clip}(1 + 2\tanh[-d_{\text{EMD}}(\mathbf{x}, \mathbf{z})], 0, 1)$，基于Sinkhorn距离衡量渲染图与目标图的感知相似性。
     - 指令遵循奖励$\mathcal{R}_{\text{IF}}$：用Qwen3.6-27B作为judge，对每个原子编辑打分$v_k \in \{0,1\}$，$\mathcal{R}_{\text{IF}} = \frac{1}{K}\sum_k v_k$。
     - 编译有效性$\mathcal{R}_{\text{Comp}}$和格式有效性$\mathcal{R}_{\text{Fmt}}$作为门控。
     - GDPO独立归一化各奖励后聚合：$A_m^{(i,j)} = \frac{r_m - \text{mean}}{\text{std}+\varepsilon}$，然后$\widehat{A}_{\text{sum}} = \sum_m w_m A_m$，采用非对称clip策略（$\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$）。

## 实验与结果
- **数据集规模**：DaEdiTikZ含430K候选对（连接590K唯一图表），双向验证保留率90.7%，人工质量评估显示98%转换可信、82.9%指令质量良好（Likert≥4）。
- **评测设置**：DaEdiTikZ-Bench（790实例，与训练集组级隔离）；自动指标：TED↓、DSim↑、EA↑、SP↑、VQ↑、CR↑；人工评测9位标注员、4,320评分，κ>0.78。
- **主要结果**：
  - EdiTikZ-9B-RL平均得分0.726（Avg），超越所有测试基线；EA=0.734、SP=0.815、VQ=0.865、CR=96.8%。
  - 人工评测综合得分17.43，超过GPT-5.6-Sol（16.75），与Gemini-3.1-Pro（17.72）持平。
  - SFT使Qwen3.5-4B/9B Avg提升0.363/0.381，RL进一步提升至0.674/0.726。
- **OOD泛化**：在SPIQA（190实例）和CharXiv（497实例）上测试，EdiTikZ-9B-RL在3–4K token长度下CR>80%，仍与GPT-5.6-Sol相当；但超过2K训练限制后性能下降。
- **关键消融**：
  - 联合SFT vs 仅编辑：3B模型提升0.04 Avg，8B持平；证明重建监督有助于编辑。
  - 奖励组合：$\mathcal{R}_{\text{SSim}}$单独EA+0.031；GDPO比GRPO多提升0.038 Avg，验证独立归一化的必要性。

## 相关工作脉络
1. **TikZ生成**：Belouadi等（AutomaTikZ、DeTikZify）从文本/图像生成TikZ，但仅从零生成，不涉及编辑；本工作在生成基础上引入编辑任务。
2. **科学图表编辑**：Zhao等（ChartEdit）聚焦图表，Lin等（AutoFigure-Edit）依赖专有Agent；本工作提供开源小型模型及真实数据。
3. **合成编辑数据**：Wang等（DisciplineGen-1M）用OCR合成编辑监督；本工作从真实修订轨迹提取，质量更高。
4. **渲染反馈RL**：Greisinger等（TikZilla）用渲染奖励训练TikZ生成；本工作进一步引入source-conditioned指令验证奖励解决编辑特有挑战。
5. **基准评测**：Rahman等（VisEditBench）评测Matplotlib/Vega-Lite编辑，Bo等（Diagram-MMU）用模板构造TikZ编辑；本工作首个基于真实修订的大规模TikZ编辑基准。

## 局限性与未来方向
- **数据噪声**：自动推断指令存在遗漏（34%）和误读（33%），虽有代码 grounding 但仍需人工精炼。
- **长序列泛化受限**：OOD生成需3–5倍token（3–4K），超出2K训练限制后性能下降。
- **评估局限**：OOD评测依赖GPT-5.6-Sol合成指令，参考-free judging可靠性待验证。
- **语言局限**：目前仅覆盖TikZ，未扩展至Matplotlib、LaTeX表格等。
- **未来方向**：扩展到多可视化语言；利用源码实现局部编辑而非全重建；支持比较型VQA和表征学习；统一生成与编辑的通用科学可视化模型。

## 研究启发与可借鉴点
1. **真实修订轨迹作为监督源**：将软件仓库/论文版本的diff转化为编辑指令的模式可迁移至其他代码/文档编辑任务（如代码编辑、公式编辑）。
2. **双奖励互补设计**：渲染保真度（dense reward）与原子编辑验证（sparse reward）的组合思路，适用于其他视觉生成-编辑统一任务。
3. **GDPO多奖励归一化**：独立归一化后再聚合的策略，可推广至多目标RLHF场景。
4. **重建-编辑联合训练**：证明重建监督对编辑能力提升有效，启示未来工作应兼顾"理解"与"修改"能力。
5. **开放小型模型+强评测**：4–9B模型即可匹敌闭源大模型，提示在垂直任务上精调小模型的经济性。

## 关键术语表
- **DaEdiTikZ**：从真实科学修订轨迹中提取的大规模TikZ编辑数据集，含391K编辑对和781K有向编辑轨迹。
- **DaEdiTikZ-Bench**：人工精炼的790实例编辑评测基准，与训练集组级隔离。
- **EdiTikZ**：基于Qwen3.5训练的4B/9B紧凑型科学图表编辑模型系列。
- **SelfSim奖励**：基于冻结视觉编码器提取的patch embeddings，用Sinkhorn距离（EMD）计算的渲染图相似性奖励。
- **指令遵循奖励（R_IF）**：用VLM作为judge对每个原子编辑进行二元打分，衡量模型是否按要求执行修改。
- **GDPO**：Group reward-Decoupled Normalization Policy Optimization，独立归一化各奖励后聚合的多奖励RL优化算法。
- **TED（TeX Edit Distance）**：基于Extended Edit Distance的TikZ代码相似度指标。
- **DSim（DreamSim）**：基于CLIP+DINO+OpenCLIP集成的感知相似度指标。

## 可复现要素
- **数据集**：DaEdiTikZ和DaEdiTikZ-Bench，论文声明将公开（Models and datasets will be released）。
- **代码/权重**：EdiTikZ-4B和EdiTikZ-9B模型，论文声明将开源。
- **关键超参**：SFT学习率1e-4/2e-5（small/large），epoch=2；RL学习率1e-6/2e-6，clip范围[0.2, 0.28]，temperature=1.0，top-p=0.99，max length=2048；输入图像448×448；剔除TikZ代码>4000字符或指令>2000字符的样本。
