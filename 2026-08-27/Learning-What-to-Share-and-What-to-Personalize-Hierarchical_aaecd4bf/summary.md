---
title: "Learning-What-to-Share-and-What-to-Personalize-Hierarchical"
source: https://arxiv.org/pdf/2608.25329v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 06:41:35"
field: "LLM Agent 记忆系统"
keywords: ["Agent Memory", "个性化记忆管理", "分层策略", "强化学习", "协同进化", "策略蒸馏"]
innovations: ["将记忆管理策略解耦为全局共享与用户特定双层级，并通过在线证据动态协同进化", "跨层级规则流实现个性化规则的泛化与通用规则的专业化双向迁移"]
benchmarks: ["PersonaMem", "PrefEval", "PersonaBench", "PERMA"]
---

# 论文速读：Learning-What-to-Share-and-What-to-Personalize-Hierarchical

## 一句话总结
本文提出 HiPS（Hierarchical Personalized Strategy）框架，将记忆管理策略解耦为全局共享基础层级与用户特定自适应层级，通过对策轨迹上的证据蒸馏与跨层级规则流动，实现策略与强化学习模型的协同进化，在 PersonaMem、PrefEval、PersonaBench、PERMA 等四个个性化记忆基准上均取得一致性能提升。

## 研究问题与动机
1. **现有记忆系统策略固定**：主流方法使用预定义提取与压缩规则，无法从交互反馈中学习，也不具备对用户行为差异的适应能力。
2. **RL 方法仍为全局共享**：MemAgent、MEM-α、MemSkill 等将记忆操作形式化为可学习动作，但仍施加严格用户无关的管理范式，少数用户的行为信号在群体平均奖励中被稀释。
3. **显式策略在训练中冻结**：MemCoE 等方法通过 TextGrad 优化管理指南后将其作为系统提示注入 GRPO 微调，规则一旦确定即不再更新，无法跟随策略演化动态对齐。
4. **共享/个性化边界未知**：基础规则普遍有益而 niche 行为需要定制，但最优划分边界无法先验确定，需要通过对策证据动态发现与协同进化。

## 核心贡献（创新点）
1. **形式化策略个性化问题**：证明共享规则与个性化规则之间的边界必须通过在线证据实证发现，而非人工预定义。
2. **HiPS 分层协同进化框架**：提出 USD + PDD + Cross-Level Rule Flow 三模块联动的 RL 架构，使通用策略与用户特定自适应规则与策略同步进化；与 MemCoE/MemSkill 等相比，核心区别在于规则持续被在线轨迹更新且支持层级间迁移。
3. **发现并解释组件重要性翻转**：域内任务中通用策略（USD/PG）贡献最大，域外泛化中跨层级流动（Flow）与分歧门控（Gate）贡献最大，为分层架构的有效性提供实证依据。
4. **策略可迁移性验证**：在高性价比模型（GPT-4o-mini）上蒸馏的策略可直接注入 GPT-5、Gemini 2.5 flash 等其他 backbone 使用，证明学到的是模型无关的记忆管理原则。

## 方法详解
1. **Universal Strategy Distillation（USD）**：每隔 $K_1$ 步采样平衡的 top-k / bottom-k 轨迹对，由 LLM 元优化器输出结构化 diff $\delta_{\text{USD}} = \{\text{V}, \text{H}, \text{R}\}$（验证/假设/修订退役），新假设须遵循 "[Label]: When [condition], [action]" 格式；配合证据追踪机制（Tentative → Supported → Established 三级生命周期），并通过预测增益阈值 $\theta_{\text{val}}$/$\theta_{\text{rev}}$ 自动升降级。
2. **Persona Delta Distillation（PDD）**：定义预测增益 $\text{PG}(r) = |P(Y=1|r^+) - P(Y=1|r^-)|$，进而计算用户分歧度 $D_p = \sum_{r \in S_u} |\text{PG}(r|p) - \text{PG}(r)|$；仅当 $D_p \geq \theta_{\text{div}}$ 时触发 PDD，以 $S_u$ 为锚点生成与通用规则行为不同的管理导向规则，防止过度拟合与冷启动问题。
3. **Cross-Level Rule Flow**： generalize（$\Delta_p \rightarrow S_u$）——当某 $\Delta_p$ 规则在 $\geq \theta_{\text{flow}}$ 比例用户中出现且证据达 Supported 级以上，经语义相似度去重后提升为通用规则；specialize（$S_u \rightarrow \Delta_p$）——当 USD 修订/降级某通用规则时，PDD 为其生成个性化替换规则，保护依赖该规则的少数用户。
4. **策略注入与反循环设计**：通过子模最大化（Eq. 6）在 token 预算 $B$ 内选择最有代表性且覆盖最广的规则子集；奖励函数 $R = R_{\text{ans}} + \lambda \cdot R_{\text{follow}}$；为防止自证实偏差，轨迹缓冲池严格按 $R_{\text{ans}}$ 排序更新，$R_{\text{follow}}$ 仅用于 GRPO 优势计算，切断规则自我强化的直接路径。

## 实验与结果
- **数据集**：PersonaMem（32K/128K/1M）、PrefEval（Explicit/Implicit）、PersonaBench（0/0.3/0.5/0.7 噪声水平）、PERMA（C-S/C-M/N-S/N-M）。
- **基线**：Long Context、RAG、Mem0、A-Mem、LightMem、MemAgent、MEM-α、MemSkill。
- **主干模型**：Qwen2.5-7B-Instruct，训练集 423 样本，4×NVIDIA H800。
- **最强结果**：PersonaMem 32K 达 **73.49%**（+9.04 over MemSkill 64.45）；PersonaMem 128K 达 **62.01%**；PrefEval Explicit 达 **89.20%**；PrefEval Implicit 达 **69.40%**；PERMA C-S 达 **66.95%**，均优于所有基线。
- **消融结论**：域内（PersonaMem）USD/PG 移除后 128K 下降最多（−8.31/-9.79）；域外（PERMA）Flow 移除后 C-S 骤降 −21.56，验证了分层架构在不同场景下的差异化重要性。

## 相关工作脉络
1. **MemCoE (Xu et al., 2026a)**：通过 TextGrad 优化显式自然语言策略再冻结注入 GRPO，本质区别是策略全局共享且在训练期间不可更新；HiPS 的策略随轨迹持续演化且支持用户索引。
2. **MemSkill (Zhang et al., 2026a)**：将记忆操作重构为可学习"记忆技能"，动态优化技能选择与精炼；但技能仍是全局共享且静态的，HiPS 额外引入用户特定层级和跨层迁移。
3. **MEM-α / MemAgent / Memory-R1**：RL 驱动的隐式记忆更新策略，行为编码于模型参数中不可解释、不可编辑；HiPS 产出可 inspectable/editable 的显式规则。
4. **A-Mem / LightMem / Mem0**：基于检索/记忆库的静态流水线，管理策略固定、无法从交互反馈中学习；HiPS 将策略本身变为可学习且动态进化的对象。
5. **EverMemOS (Hu et al., 2026a)**：通过经验聚类蒸馏实现技能退役；关注全局技能生命周期，未涉及用户间个性化差异与跨层级规则迁移。

## 局限性与未来方向
1. **语言覆盖局限**：评估集中于英语对话基准，多语言及跨语言场景下记忆管理策略的句法结构可能不同，尚待验证。
2. **超长期用户漂移未建模**：实验跨度最高 1M tokens（中等偏长），但用户在数年连续日常交互中可能出现根本性的基线人格漂移，需额外的 meta-distillation 机制周期性更新种子规则。
3. **关键词启发式合规估计**（附录 G）：使用轻量关键词启发规则判断合规性，可能存在系统性偏差，虽因 PG 仅用于相对排序而鲁棒，但对精确场景仍存误差。

## 研究启发与可借鉴点
1. **结构化 Diff + 证据追踪机制**可直接迁移至任何需要持续精炼规则的系统（如工具调用策略、推理链优化），比自由文本反馈更具可操作性和稳定性。
2. **反循环设计**（训练信号与优化信号严格分离）对自奖励 RL、Self-Play 等容易陷入自证实偏差的场景具有普适参考价值。
3. **子模贪心选择注入规则**的 Prompt Budgeting 思想可复用于任意 LLM Agent 的上下文管理，平衡信息覆盖与 token 成本。
4. **分化门控机制**（仅对真正偏离群体的用户个性化）为少样本个性化场景提供资源节约方案，避免"人人个性化"带来的噪声与算力浪费。

## 关键术语表
**USD（Universal Strategy Distillation）**：从跨用户轨迹中蒸馏共享记忆管理规则的模块，输出结构化 diff 并维护规则证据生命周期。

**PDD（Persona Delta Distillation）**：为用户行为偏离群体规范时生成个性化记忆管理规则的模块，通过分歧度阈值门控以避免过度个性化。

**Cross-Level Rule Flow**：在通用层级 $S_u$ 与个性化层级 $\Delta_p$ 之间双向迁移规则（泛化与专业化），动态校准两者边界。

**Predictive Gain（PG）**：衡量某条规则对任务成功率的预测能力，定义为遵从/不遵从规则轨迹的成功率之差的绝对值。

**Divergence（$D_p$）**：用户 $p$ 的各规则预测增益与全局预测增益的累积偏差，用于决定是否触发 PDD 个性化。

**GRPO（Group Relative Policy Optimization）**：本文采用的策略优化算法，基于组内相对优势更新策略模型 $\pi_\theta$。

**Structured Diff**：USD 输出的离散操作集合，包含验证（V）、假设（H）、修订/退役（R）三类规则级操作，格式为 "[Label]: When [condition], [action]"。

**Evidence Level**：规则的生命周期状态，依次为 Tentative（临时）、Supported（已验证）、Established（稳定建立），支持自动晋升与修剪。

## 可复现要素
- **数据集**：PersonaMem、PrefEval、PersonaBench、PERMA 均为公开 benchmark；训练集为从 PersonaMem 32K 子集中采样的 423 个实例。
- **代码/权重**：论文未明确声明代码开源状态，但基线 MemAgent、MEM-α、MemSkill 使用公开仓库初始化。
- **关键超参**：USD 频率 $K_1=20$，PDD 频率 $K_2=10$，token 预算 $B=200$，分歧阈值 $\theta_{\text{div}}=0.3$，流动阈值 $\theta_{\text{flow}}=0.6$，验证阈值 $\theta_{\text{val}}=0.1$，修订阈值 $\theta_{\text{rev}}=0.02$，奖励权重 $\lambda=0.3$。
