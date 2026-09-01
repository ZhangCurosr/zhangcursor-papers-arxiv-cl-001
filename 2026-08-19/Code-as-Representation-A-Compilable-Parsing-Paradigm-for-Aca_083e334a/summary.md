---
title: "Code-as-Representation-A-Compilable-Parsing-Paradigm-for-Aca"
source: https://arxiv.org/pdf/2608.17550v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:47:58"
field: "多模态文档解析与学术文档理解"
keywords: ["Multimodal Document Parsing", "MLLM", "LaTeX", "Chart-to-Code", "Compilable Parsing", "Structured Academic Elements", "Benchmark"]
innovations: ["提出CADP编译式解析范式，将学术页重建为上下文LaTeX+可执行Python双代码", "引入重注入编译协议，实现结构/可视/可执行的一致性严格验证", "构建CADP-Bench基准（1,630页专家校验、多SAE耦合、四学科覆盖）"]
benchmarks: ["CADP-Bench"]
---

# 论文速读：Code as Representation — A Compilable Parsing Paradigm for Academic Documents

## 一句话总结
本文提出 **CADP（可编译学术论文解析）新范式**，将完整学术页面重建为**上下文 LaTeX + 可执行 Python** 的双代码表示，使公式、表格、伪代码与图表均保持可编译、可验证的结构；同时发布 CADP-Bench 基准（1,630 页专家校验样本）并以重注入编译协议评估，揭示当前 SOTA MLLM 在高保真可执行重建上仍存在显著差距。

## 研究问题与动机
1. **学术知识锁定在 PDF 中**：论文载体高度优化人类阅读，机器不可用；MLLM 在检索/推理管线中常只能接触 PDF 或截图。
2. **Markdown 等常用代用表示有结构性缺陷**：① 平铺语法无法承载合并单元格、对齐方程、复杂伪代码；② 图表退化为静态裁剪图，丢失底层数据与渲染逻辑；③ 不可编译验证，无法与源页逐像素核验。
3. **现有基准无法满足“全文页 + 多 SAE 耦合 + 可编译验证”需求**：Markdown 类基准天花板即平铺表达；元素级代码生成（如单表/单公式）脱离页上下文形成"语义孤儿"。
4. **核心难题是表示（representation）而非仅感知（perception）**：SAEs（表格、公式、图表、伪代码）与正文交错时，拓扑、嵌套、跨引用、数据与渲染逻辑难以被结构化保留。

## 核心贡献（创新点）
1. **形式化 CADP 任务**：首次将学术页面解析建模为“在有限编译上下文条件下，联合生成上下文 LaTeX 块与可执行 Python 程序”的编译约束双代码生成问题。
2. **提出 CADP-Bench 基准**（1,630 页、四学科、每页 ≥2 种 SAE）：引入**重注入编译协议**——将生成 Python 渲染为 SVG、注入 LaTeX 源并编译，直接对比重渲染页与 GT 页，实现结构/可视/可执行一致性的一体化评估。
3. **首次系统评测 SOTA MLLM 在可编译解析上的能力**：覆盖 Gemini-3-Pro / GPT-5.x / Qwen3.5 系列 / Claude 4.x / Kimi-K2.5 等，给出代码级与视觉级双维度指标体系。
4. **构建探索性多代理（multi-agent）基线**：Planner + LaTeX Coder + Python Coder + Reviewer 四角色 + 共享工具链 + 迭代反馈，量化自省、工具、多角色协作对 CADP 的贡献边界。
5. **揭示结构保留表示对下游格式敏感 QA 的正向迁移**：LATeX+Python 表示对中等规模模型（Qwen3.5-Plus）尤其有效，甚至反超 Gemini-3-Pro 使用 Markdown 输入的表现。

## 方法详解
### 3.1 任务定义
- **输入**：目标页图像 $I_p \in \mathbb{R}^{H \times W \times 3}$ + 受限编译上下文 $C_{pkg}$（可用宏包声明）；源上下文 $C_{\backslash p}$（去除了本页内容的完整 LaTeX 源，含前导、宏、参考文献等）仅用于重注入与编译验证，**不直接提供给模型**。
- **输出**：$(\mathcal{L}_p, \mathcal{P}_p)$，其中 $\mathcal{L}_p$ 为覆盖页面文本与 SAEs 的上下文 LaTeX 体块；$\mathcal{P}_p = \{P_1, \dots, P_k\}$ 为重构各图表数据与渲染逻辑的可执行 Python 程序集。
- **映射**：$f_\theta : (I_p, C_{pkg}) \to (\mathcal{L}_p, \mathcal{P}_p)$。

### 3.2 数据采集与过滤
1. 从 arXiv 批量采集各学科论文 PDF + 对应 LaTeX 源；用 Mineru 提取页面级布局与各 SAE 区域。
2. 剔除无法编译的 LaTeX 文件；保留包含至少 2 类不同 SAE 的页面（若跨页大表格仍视为单样本，以考察长程依赖）。
3. 用正则 + Mineru 解析对齐各 SAE 的 GT LaTeX 源码。

### 3.3 图表重建与标注
- 原始 LaTeX 中图表通常只引用预渲染图片，无可执行绘图代码；因此需恢复成 Python。
- 流程：裁剪图表为 PNG → Gemini-3-Pro（Chart2Code）生成 Python → 编译为 SVG → 专家核验后替换原图。
- 难度分级（simple / medium / hard）由 LaTeX 代码长度与结构复杂度决定。

### 3.4 评估协议：重注入编译
1. **执行 Python 生成图表**：$G_i = \text{Execute}(P_i)$，得到 $\text{SVG}$ 资产 $G_1, \dots, G_k$。
2. **LaTeX 引用**：$\mathcal{L}_p$ 须显式 `\includegraphics` 引用上述资产。
3. **重注入与编译**：$\hat{I}_p = \Omega(C_{\backslash p} \oplus \mathcal{L}_p, \{G_1, \dots, G_k\})$，其中 $\oplus$ 为原位拼接，$\Omega$ 为 LaTeX 编译器（pdfLaTeX）。
4. **编译失败或视觉偏差均判为 0 分**：宏幻觉、环境未闭合、Python 数据逻辑错误都会直接导致编译失败或严重偏差。

### 3.5 评估指标
| 维度 | 指标 | 说明 |
|---|---|---|
| **代码** | Exec. Rate | 整体文档经重注入后可成功编译并渲染的比例 |
| | TEDS | 表格的结构与内容相似度 |
| | PRS（伪代码） | 文本相似度 + AST 树编辑距离的 50/50 平均，缩放至 [0,100] |
| **视觉** | Reading Order | 阅读顺序误差（越小越好） |
| | PageIoU | 页级交并比 |
| | CDM | 公式字符级匹配 |
| | VRF-S（表格/伪代码） | MLLM-as-Judge 四维 0/1 全对才得 100，否则 0 |
| | VRF-A（图表/全局布局） | 四维均值 ×25 |
| | Pixel Sim. | 去除共背景后 MAD 转 PS，$PS = 1 - \text{MAD}/255$，页内均化 |

**PRS 公式**：
- 文本：$R_{\text{text}} = \frac{1}{n}\sum_{i=1}^n \left(1 - \frac{\text{EditDist}(s_i^{\text{gt}}, s_i^{\text{pred}})}{\max(|s_i^{\text{gt}}|, |s_i^{\text{pred}}|)}\right)$
- 结构：基于 ALGORITHM 根、控制流节点的规范化树编辑相似度 $R_{\text{struct}}$
- $\text{PRS} = 50(R_{\text{text}} + R_{\text{struct}})$

**VRF-S/A**：Gemini-3-Pro 在 100 样本上与人工一致率达 86%；GPT-5.5 重评与 Gemini 得分差异 < 3%。

## 实验与结果
### 数据集统计
- **总量**：1,630 页；子领域 — CS 1,047 / 物理 251 / 经济 45 / 定量生物 222 / 统计 65。
- **元素分布**：表格 1,842、公式 1,023、伪代码 138、图表 1,491；难度简单/中/难约 803/644/395（表）等。
- **标注质量**：4 名 CS 本科 + 1 名 PhD 仲裁，总体一致率 88%。

### 主要结果（RQ1 & RQ2，节选 Table 4）
- **最强模型**：Gemini-3-Pro 综合领先；但其 VRF 布局（73.00）与 Pixel Sim.（47.28）仍非满分，说明高保真完全重建仍有差距。
- **开源追赶**：Qwen3.5-397B-A17B 整体视觉保真度媲美 GPT-5.4 / Claude Opus 4.6。
- **伪代码是普遍瓶颈**：即便 Gemini-3-Pro 也仅 56.52 VRF-S；GPT-5.4 仅 13.04、Claude Opus 4.6 仅 17.39，算法格式鲁棒性仍是未解难题。
- **图表“可执行”与“可视逼真”脱耦**：前沿模型 Exec. Rate 极高（如 GPT-5.4 97.86%），但 VRF-A 仅为 59.25，说明“能跑”远不等于“忠实复现”。
- **TEDS 过度乐观**：表格 VRF-S 显著低于 TEDS，部分结构恢复不足，需严苛的全维度匹配。
- **复杂度退化**（Fig. 4）：公式/表格在强模型上中等难度的下降较平缓；图表与伪代码在所有层级中均为最大瓶颈，弱模型在 hard 样本上接近地板。

### 多代理消融（RQ3，Table 5，以 Gemini-3-Pro 为底座）
| 设置 | Layout VRF-A | Formula CDM | Table VRF-S | Chart VRF-A | Pseudocode VRF-S |
|---|---|---|---|---|---|
| Base | 73.00 | 70.83 | 66.09 | 60.00 | 56.52 |
| Full MA System | **77.20** | **74.20** | **71.40** | **66.50** | **61.80** |
| w/o Visual Feedback | 74.10 | **75.30**↑ | 66.80 | 62.00 | 58.50 |
| w/o Tools | 76.50 | 73.40 | 68.20 | 61.80 | 61.10 |
| w/o Multi-Agent | 75.80 | 72.60 | 67.90 | 63.50 | **62.50**↑ |
| Base w/ Self-reflection | 71.00 | 68.80 | 63.99 | 58.50 | 52.17 |

关键洞察：
- Full MA 系统带来 3–7 分稳健提升。
- **反直觉**：去掉 Visual Feedback 反而提升 Formula（74.20→75.30），因基于像素的视觉评审会误读密集数学符号而引入幻觉修正。
- 去掉 Multi-Agent 协作略提升伪代码（61.80→62.50），因为多轮冗余生成容易破坏精确缩进与特定语法。
- **纯自省有害**：无外部工具与环境的自我反思带来全面退化。

### 格式敏感 QA（RQ4，Fig. 6）
- 小模型（Qwen3.5-35B-A3B）对 LATeX+Python 几乎无增益。
- 强模型（Gemini-3-Pro）截图 vs Markdown 表现接近，视觉 grounding 可部分弥补 Markdown 的结构性损失。
- **中等模型（Qwen3.5-Plus）从 LATeX+Python 大幅获益**，甚至超越 Gemini-3-Pro + Markdown，说明结构保留表示可弥补模型规模不足。

## 相关工作脉络
1. **TabLeX / Tab-To-LaTeX / TAB2LaTeX / Table2LaTeX**：元素级表格 → LaTeX 生成，局限于局部裁剪输入，缺乏页上下文与跨引用，无法评估多 SAE 耦合重建。
2. **Im2LaTeX-100K / UniMER / OmniDocBench**：公式识别/页面解析 benchmark，输出以 Markdown/静态图片为主，无编译验证与可执行图表。
3. **ChartMimic / ChartX / ChartEdit / DaTikZv3**：单一图表 → 代码/ TikZ，评估的是孤立图表理解，忽略页级排版与多元素交互。
4. **READOC / olmOCR-bench**：面向 OCR 与文档结构化提取，评估指标停留在文本与 Markdown 层面，未触及编译/可执行维度。
5. **mPLUG-DocOwl 1.5 / Doc-researcher**：工程化端到端解析管线，仍依赖 Markdown 为主表示，结构信息在图表/算法环节大量丢失。
6. **本文定位**：首次提出“全文页 + 多 SAE + 双代码 + 编译验证”的编译式解析范式与对应基准，弥补既有工作在结构保留与可验证性上的系统性缺口。

## 局限性与未来方向
1. **基准规模有限**：1,630 页对大规模预训练而言偏小，难以直接用于从头预训练。
2. **图表 GT 由 Gemini-3-Pro 生成并经专家核验**，存在模型偏置与潜在噪声；对复杂科学可视化（如多子图组合、三维图）的恢复质量未经充分评估。
3. **仅用 pdfLaTeX 编译**，未覆盖 XeLaTeX/LuaLaTeX 等非标准引擎与非拉丁字体场景。
4. **多代理系统为探索性基线**，未系统搜索代理拓扑、工具集合与迭代轮数；自反改进（remove visual feedback 反升）提示当前评审模块仍需完善。
5. **未评估长期跨页表格、多栏排版、附录/补充材料**等复杂场景。
6. **未来方向**：① 利用可编译奖励进行强化学习/后训练；② 借助 LaTeX 显式结构信号构建 Graph-RAG，缓解 chunk-based RAG 的断链问题；③ 拓展到非英文学术文献、多语言排版。

## 研究启发与可借鉴点
1. **编译即验证**：将生成结果投入真实编译/执行环境并度量“能否跑通 + 渲染是否一致”，比纯文本相似度更严格且更具可操作性；可迁移到任意代码/公式/表格生成任务。
2. **双代码表示（结构化文本 + 可执行脚本）**：分离"排版/文本描述”与" 数据/渲染逻辑"，既保留人类可读的上下文结构，又暴露可计算的内容语义。
3. **重注入编译协议**：利用源上下文 $C_{\backslash p}$ 注入新内容并重新编译，使评估天然具备结构一致性约束，避免局部指标乐观偏差。
4. **多代理协作的精细化控制**：全量模块并非处处有益；视觉评审对密集公式可能引入负向“修正”。设计 agent 时应区分领域对评审信号的敏感度。
5. **结构表示对中等模型的增益更显著**：LATeX+Python 作为归纳偏置可部分补偿模型规模，提示后续面向低成本部署的科研助手可优先考虑结构保留型前端表示。

## 关键术语表
- **SAE（Structured Academic Elements，结构化学术元素）**：论文中表格、公式、图表、伪代码等兼具强结构与强语义的非连续文本成分。
- **CADP（Compilable Academic Document Parsing，可编译学术文档解析）**：把整页论文重建为上下文 LaTeX + 可执行 Python 的双代码范式。
- **重注入编译（Re-injection Compilation）**：将生成 LaTeX 注入原始源上下文缺失处、连同生成图表资产一起编译，以渲染页与 GT 页的视觉一致作为严格验证手段。
- **PRS（Pseudocode Reconstruction Score，伪代码重建分）**：文本相似度与 AST 树编辑相似度等权平均后缩放至 [0,100] 的综合评分。
- **VRF-S / VRF-A**：基于 MLLM-as-Judge 的四维评估；S 为严格模式（全维度满分才得 100，否则 0），A 为连续模式（四维均分 ×25）。
- **语义孤儿（Semantic Orphan）**：孤立裁剪的元素级代码生成任务下，图表/算法脱离全文上下文中引用与叙事锚点的现象。
- **Exec. Rate**：生成代码经重注入后可成功编译并渲染的样本比例。

## 可复现要素
- **数据集**：CADP-Bench 已公开发布（论文称"CADP-Bench is released for future research"），来源为 arXiv，覆盖 CS/物理/经济/定量生物/统计 5 个子领域。
- **代码/权重**：论文未开源模型权重；多代理代码未明确开源；评测协议与 prompt 随论文附录公开。
- **关键超参**：模型 temperature = 0.7；每样本评测 3 次取平均；LaTeX 编译使用 **pdfLaTeX** 引擎；图表 GT 生成使用 **Gemini-3-Pro（Chart2Code）**。
- **评估工具**：MLLM-as-Judge 使用 Gemini-3-Pro，并辅以 GPT-5.5 做一致性复核。

---
