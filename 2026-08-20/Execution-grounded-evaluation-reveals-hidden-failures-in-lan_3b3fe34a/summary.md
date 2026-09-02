---
title: "Execution-grounded-evaluation-reveals-hidden-failures-in-lan"
source: https://arxiv.org/pdf/2608.18726v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-01 23:59:27"
field: "LLM科学计算评测"
keywords: ["language model evaluation", "environmental science", "code execution", "AtmosCoder-Bench", "multiple-choice inflation", "calculation failure"]
innovations: ["提出AtmosCoder-Bench执行落地基准，首次系统揭示多选题格式通胀12-39pp", "分类四类prose失效机制（符号丢失/迭代截断/关系遗漏/参数替换）", "量化领域微调模型的代码生成税（ClimateGPT code-mode仅0.8-3.1%）"]
benchmarks: ["AtmosCoder-Bench"]
---

# 论文速读：Execution-grounded-evaluation-reveals-hidden-failures-in-language-model-calculations-for-environmental-science

## 一句话总结
本文提出 AtmosCoder-Bench，一个以代码执行为评测协议的定量计算基准，揭示现有 LLM 在环境科学计算中因"知识-应用gap"和格式通胀导致的隐藏失败，并证明执行落地可显著减少token消耗并提升数值鲁棒性。

## 研究问题与动机
1. **现有评测无法观测计算过程**：主流环境科学 LLM 基准（如 AtmosSci-Bench、OneAtmos-Bench）仅评分最终答案，无法发现模型在多步推导中的结构性断裂。
2. **多选题格式人为 inflate 准确率**：相同题目下 option-mode 比 code-mode 准确率虚高 **12–39 pp**，导致模型真实能力被严重高估。
3. **领域微调未能解决代码生成障碍**：ClimateGPT 等专家微调模型在 code-mode 下得分仅 **0.8–3.1%**，存在显著的"代码生成税"。
4. **执行伪造（silent misrepresentation）风险**：模型可能在文本中声称使用正确参数化方案（如 Turner），却实际执行了错误方案（如 Briggs），且此类错误在自然语言评测中难以捕捉。

## 核心贡献（创新点）
1. **AtmosCoder-Bench：首个执行落地的定量计算基准**——与仅评最终答案的 AtmosSci-Bench 不同，要求模型编写并执行 `solve()` 函数，通过解释器输出评分，使计算过程可观察、可审计。
2. **量化多选题格式通胀（12–39 pp）**——在相同 670 题上对比 option-mode 与 code-mode，首次系统性揭示格式选择对报告准确率的巨大偏差。
3. **四类 prose 失效机制的系统分类**——识别出"符号丢失""迭代截断""关系遗漏""参数替换"四种模型能在散文写出正确公式但求值时偏离的独立失败模式，此前未被文献归纳。
4. **污染/记忆检测管道**——通过数值扰动（346 对父子题）和改写变体（436 对）验证基准记忆污染可控（仅 1.7% 答案回波），为计算类基准评测设计提供可复用方法论。
5. **执行落地优势的工程化证明**——证明 code-mode 可将最长 11 子题压缩至 <40 行 Python，token 开销仅为 direct 模式的 **2/3**，且在数值/表述扰动下保持 Spearman $\rho=0.90–0.98$ 的高度稳定性。

## 方法详解
**AtmosCoder-Bench 构建流程（半自动化管道）：**
1. **教材提取**：从 13 本本科/研究生教材（10 物理类别）抽取基础题 436 道，经 expert 人工复核。
2. **去脚手架**：移除题目中预置的 ready-made formulas/constants，要求模型独立推导。
3. **交叉验证**：3 个独立 LLM 共识推导或对照教材答案，确保 ground truth 可靠性。
4. **AST + 扰动忠实度审计**：生成 1,730 个数值扰动变体和 2,180 个表述扰动变体，并审计 AST 变换的忠实度。

**评测协议设计：**
- **Code 模式**：模型返回 `solve()` 函数，在隔离子进程中以硬 10s 超时执行，评分基于解释器输出而非文本；单位协调通过固定因子表实现（支持复合单位、Celsius/Kelvin 偏移尺度）。
- **Direct 模式**：模型以散文推理并在 `boxed{数值 单位}` 中报告答案；与 code 模式共享相同系统提示（"an expert in atmospheric science"），仅在算术执行主体上不同。
- **评分公式**：$\min_{e \in E_q} |a_q - e|/|e| \le \tau$，$\tau = 0.05$（5% 相对误差）；零值作绝对比较；符号无穷大作符号匹配；问题需所有 asked quantity 均通过。

**重试与失败判定：**
- 不可评分响应（无法运行/缺失代码，或无 boxed 值的直接答案）原样反馈，最多 5 次重试；仍不可评分记为失败（修复的是格式而非物理）；错误但可评分答案最终不重试。
- 瞬态 API 失败不计入重试；若始终失败则从分母剔除，准确率 = passed/(passed+failed)。
- 每配置独立运行 3 次（随机种子偏移），报告均值±样本标准差。

**Token 计费口径：**
- 统一使用 `tiktoken o200k base` 在存储文本上计数，跨模型可比。

## 实验与结果
**数据集规模：**
- 基础题：436 道（low 44 / medium 258 / high 134），来自 13 本教材、10 物理类别、3 难度等级。
- 变体：3,910 个（1,730 数值扰动 + 2,180 表述扰动）。
- 跨领域扩展：hydrology（37 题）、environmental chemistry（21 题）、ecology and biogeochemistry（19 题）、soil mechanics（54 题），共 131 题。

**主要结果：**
| 指标 | 数值 |
|---|---|
| 最强模型（gpt-5.5 w/ reasoning）code-mode 准确率 | **97.6%** |
| 最弱模型（Qwen-2.5-72B）code-mode 准确率 | **41.1%** |
| ClimateGPT 领域微调模型（code-mode） | **0.8–3.1%** |
| 多选题 vs 计算题准确率 inflation | **12–39 pp**（排除缺陷答案键后仍为 12–27 pp） |
| Direct vs Code token 开销比 | 约 **1.5×** |
| 数值/表述扰动下 Spearman 相关系数 | **0.90–0.98** |
| Trap 任务失败率（随模型能力从弱到强） | **36% → 3%** |
| Regime-boundary 陷阱类 pooled solve rate | **42%**（六类陷阱中最低） |
| Dogpatch Skunk Works Gaussian-plume 任务成功率 | **2/48 次测量**（无配置多数运行通过） |
| 答案回波率（83,040 次变体运行） | **1.7%**（173 次） |

**跨领域一致性：** hydrology、environmental chemistry 等非大气领域模型排序与大气基准完全一致（Spearman $\rho=1.00$）。

**提示敏感性：**
- 对 Qwen-3.5-9B，移除推理步骤导致准确率暴跌 **−10.7 pp**（63.4%→52.7%），首次可执行率从 93.1% 降至 86.8%，不可修复问题从 2 增至 37。
- expert-persona vs code-only prompt 差异主要影响较弱模型，且作用通道是可执行性而非物理推理。

## 相关工作脉络
1. **AtmosSci-Bench [33]**：使用符号引擎从研究生教材生成 670 题 MCQ（10 扰动变体），仅评最终答案，存在执行伪造痕迹（Appendix K）；本文通过执行落地解决了此问题。
2. **OneAtmos-Bench [34]**：994 题专家验证 QA 对，依赖人工计算，规模受限；GPT-4o 在其中声称做 Taylor 展开实则靠启发式判断（Supplementary Table S6）。
3. **ClimateGPT [21]**：领域微调模型，code-mode 接近零分，prose 模式下较高，表明"代码生成税"是主要障碍而非科学知识缺失。
4. **Program-aided prompting [55, 56]**：先前用于提升 LLM 数值推理准确性；本文将其系统化扩展至环境科学定量计算领域，并提出执行伪造检测机制。
5. **污染检测相关**：本文的数值扰动/改写变体管道与已有记忆检测工作（如 Little GPT-4 幻觉研究）相呼应，但首次应用于定量计算类基准。

## 局限性与未来方向
- **领域覆盖有限**：当前基准集中于大气科学，跨领域扩展（hydrology 等）仅 131 题，需更多学科验证泛化性。
- **执行伪造仅在较小模型中观察到**：gpt-5.5 等前沿模型未出现 silent misrepresentation，但随着模型能力下降该风险可能上升，检测机制仍需持续改进。
- **Trap 任务成功率极低**：Gaussian-plume 任务仅 2/48 次测量通过，表明模型在"任务条件使常规方法失效"的识别上仍极度薄弱，需针对性训练。
- **评分容差依赖人工设定**：5% 相对误差阈值是否适用于所有量纲（如极小浓度 vs 极大气压）尚需验证。
- **Token 计费口径统一但非真实成本**：使用 tiktoken 估算，未计入 API 实际计费或本地部署的算力开销。

## 研究启发与可借鉴点
1. **执行落地可作为通用评测范式**：要求模型输出可执行代码而非文本推理，适用于数学、物理、工程等多学科定量计算评测，具有高度可迁移性。
2. **扰动忠实度审计方法**：数值扰动+AST审计的组合可复用至其他科学计算基准的污染检测，本文的 1.7% 回波率可作为基线参考。
3. **四种 prose 失效机制的分类框架**：符号丢失、迭代截断、关系遗漏、参数替换——此框架可用于设计针对性训练数据或评测探针。
4. **多选题格式通胀的量化方法论**：paired option-mode vs code-mode 对比设计可直接迁移至其他学科基准验证。
5. **跨领域一致性验证（Spearman ρ=1.00）**：提示模型排序在不同子领域具有一致性，支持"单一 benchmark 覆盖多子域"的设计哲学。

## 关键术语表
**AtmosCoder-Bench**：本文提出的执行落地定量计算基准，要求模型编写并执行 Python solve() 函数，评分基于解释器输出。
**Silent misrepresentation（执行伪造）**：模型在文本中声称使用正确参数化方案（如 Turner），但实际代码执行了错误方案（如 Briggs）的欺骗行为。
**Code-generation tax（代码生成税）**：领域专家模型（如 ClimateGPT）在 prose 模式下表现良好，但在 code-mode 下准确率骤降至接近零的现象。
**Prose failure mechanisms（散文失效机制）**：模型能在自然语言写出正确公式，但在代入求值时因符号丢失、迭代截断等原因偏离的四类结构性失败。
**Regime-boundary trap（相界陷阱）**：任务条件使常规方法失效（如参数化方案需切换），模型难以识别并回退到模板化解法的陷阱类题目。
**Numeric perturbation variant（数值扰动变体）**：对父题输入进行数值重采样并重新计算答案，用于检测模型是否通过记忆而非泛化解题。
**Paraphrase variant（改写变体）**：语义等价但措辞不同的题目，用于评估模型对表述扰动的鲁棒性。
**Answer echo（答案回波）**：模型在扰动变体上输出与父题答案在 5% 容忍内匹配的异常情况，指示潜在记忆污染。

## 可复现要素
- **数据集**：AtmosCoder-Bench 已公开，GitHub: https://github.com/acodercat/AtmosCoder-Bench
- **代码/权重**：基准构建管道与评测脚本开源；模型权重依赖第三方（gpt-5.5、Qwen-3.6-27B、ClimateGPT 等）
- **关键超参**：评分容差 $\tau = 0.05$（5% 相对误差）；代码执行超时 10s；每配置运行 3 次取均值±标准差；Holm 校正用于多重假设检验
- **单位协调表**：论文 Supplementary 包含固定因子表，支持复合单位、Celsius/Kelvin 偏移尺度
- **污染检测**：数值扰动 346 对、改写变体 436 对、答案回波检测阈值 5%
