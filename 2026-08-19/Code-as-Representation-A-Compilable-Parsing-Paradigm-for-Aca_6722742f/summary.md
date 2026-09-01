---
title: "Code-as-Representation-A-Compilable-Parsing-Paradigm-for-Aca"
source: https://arxiv.org/pdf/2608.17550v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:47:53"
field: "多模态文档解析与科学文档理解"
keywords: ["Multimodal Document Parsing", "MLLMs", "Compilable Parsing", "Academic Document", "Code-as-Representation", "CADP-Bench"]
innovations: ["提出可编译学术文档解析（CADP）范式，以双重代码（LaTeX+Python）重建学术页面", "发布 CADP-Bench 基准，1630 专家验证页面配合重新注入编译协议评估", "揭示高执行率与高视觉保真度之间的系统性解耦，以及伪代码为当前 MLLMs 普遍瓶颈"]
benchmarks: ["CADP-Bench", "TabLeX", "ChartMimic", "READoc", "OmniDocBench"]
---

# 论文速读：Code-as-Representation-A-Compilable-Parsing-Paradigm-for-Aca

## 一句话总结
论文提出**可编译学术文档解析（CADP）**范式，将学术页面重建为**上下文 LaTeX + 可执行 Python** 的双重代码表示，并发布 **CADP-Bench**（1630 个专家验证页面）基准，揭示当前 SOTA MLLMs 在高保真可执行重建任务上仍存在显著能力缺口。

---

## 研究问题与动机
1. **学术文档知识"锁在 PDF 中"**：科学页面密集交织表格、公式、图表、伪代码等结构化学术元素（SAEs），而常见解析产物（Markdown）无法保留其结构和可执行语义。
2. **Markdown 三大缺陷**：结构坍缩（扁平语法无法编码跨页/合并表格、嵌套公式、伪代码缩进）、图表不透明（图表退化为静态裁剪图，丢失底层数据）、不可验证（无法回编译校验视觉一致性）。
3. **现有基准不匹配**：纯文本基准（如 READoc）天花板受限于扁平表示；元素级代码基准（如 TabLeX、ChartMimic）仅评估裁剪孤立区域，产生"语义孤儿"，缺失页级上下文与交叉引用。
4. **核心命题**：能否从页面像素恢复出**结构保留、图表可执行、可重新编译校验**的机器可操作表示？

---

## 核心贡献（创新点）
1. **形式化 CADP 范式**：将解析问题从"像素→平面文本"转变为"像素→可编译双重代码"，以结构保留的 LaTeX+Python 替代 Markdown 作为机器可读表示。
2. **发布 CADP-Bench 基准**：首个覆盖完整学术页面、要求多 SAEs 耦合共现、通过**重新注入编译协议**评估的可验证基准（1630 样本，人工+LLM 混合标注，一致性 88%）。
3. **系统性评测 SOTA MLLMs**：评测 Gemini-3-Pro、Qwen3.5 系列、GPT-5.4、Claude Opus 4.6 等，揭示即使顶级模型在结构密集元素（伪代码、复杂表格）上仍显著退步。
4. **探索性多智能体基线**：设计 Planner/LaTeX Coder/Python Coder/Reviewer 四角色系统，量化自反思、工具、视觉反馈对重建质量的边际贡献。
5. **格式敏感性理解实验**：证明结构化 LaTeX+Python 表示对中等规模模型（如 Qwen3.5-Plus）的理解力提升显著，为低资源模型提供归纳偏置补偿。

---

## 方法详解
### 3.1 任务形式化
- **输入**：目标页截图 $I_p \in \mathbb{R}^{H \times W \times 3}$ + 有限编译上下文 $C_{pkg}$（可用宏包声明，**不提供完整源上下文**）。
- **输出**：双重代码 $(\mathcal{L}_p, \mathcal{P}_p)$
  - $\mathcal{L}_p$：**上下文 LaTeX body**，重建页面排版、正文、表格、公式、伪代码。
  - $\mathcal{P}_p = \{P_1, \ldots, P_k\}$：**可执行 Python 程序集合**，对应页面中各图表的数据与渲染逻辑（非作者原始绘图代码，而是视觉等价的推断代码）。

### 3.3 图表重建与标注
- 原 LaTeX 源文件中图表通常仅为图片引用（无绘图代码），故需**额外恢复图表可执行表示**。
- 流程：裁剪图表 PNG → Gemini-3-Pro Chart2Code 生成 Python → 渲染为 SVG → **专家人工验证**后作为 ground truth。
- 难度分级：Simple / Medium / Hard（基于 LaTeX 代码长度与结构复杂度）。

### 3.4 评估协议：重新注入编译（Re-injection Compilation）
1. 执行每个生成的 Python 程序 $P_i$ 得到 SVG $G_i$（公式 1）。
2. 将生成 LaTeX $\mathcal{L}_p$ 显式引用 $G_i$（如 `\includegraphics{...}`）。
3. 将 $\mathcal{L}_p$ **原位插入**原始源上下文 $C_{\backslash p}$（已剔除目标页的 LaTeX 源码）得到 $\hat{I}_p = \Omega(C_{\backslash p} \oplus \mathcal{L}_p, \{G_1, \ldots, G_k\})$（公式 2）。
4. 任何宏定义幻觉、嵌套环境未闭合、Python 数据错误均会导致编译失败或视觉错位，构成**内在严格一致性校验**。

### 3.5 评估指标（Table 2）
| 类型 | 指标 | 说明 |
|------|------|------|
| **代码** | Exec. Rate | 重新注入编译成功率 |
| **代码** | TEDS | 表格结构+内容编辑相似度 |
| **代码** | PRS (Pseudocode Reconstruction Score) | 结合文本 Levenshtein + AST 树编辑距离（公式 3–5） |
| **视觉** | CDM | 公式字符级匹配 |
| **视觉** | Reading Order / PageIoU | 页面布局一致性 |
| **视觉** | VRF-S (Strict) | 表/伪代码：所有维度满分=100，否则 0（公式 6） |
| **视觉** | VRF-A (Average) | 图表/全局布局：K 维均值×25（公式 7） |
| **视觉** | Pixel Sim. | 忽略背景后的像素 MAD 相似度（公式 8–12） |

---

## 实验与结果
### 数据集统计（Table 3）
- **总量**：1630 页面，每页至少含 2 类 SAEs 耦合。
- **子领域**：CS 1047 / Physics 251 / Quant. Bio 222 / Economics 45 / Statistics 65。
- **元素分布**：Table 1842 / Chart 1491 / Formula 1023 / Pseudocode 138。
- **难度**：Simple 803 / Medium 644 / Hard 395（按元素计）。

### 主要结果（Table 4，部分关键数值）
| 模型 | 公式 CDM↑ | 表 VRF-S↑ | 图 VRF-A↑ | 伪代码 PRS↑ | 像素相似度↑ |
|------|-----------|-----------|-----------|--------------|-------------|
| **Gemini-3-Pro** | 70.83 | **62.48** | **60.00** | **56.52** | **47.28** |
| Qwen3.5-Plus | 69.96 | 42.20 | 40.50 | 73.67 | 44.18 |
| Qwen3.5-397B-A17B | 70.45 | 61.39 | 45.50 | 73.78 | 45.74 |
| GPT-5.4 | 66.64 | 43.26 | 59.25 | 41.26 | 43.54 |
| Claude Opus 4.6 | 60.61 | 35.43 | 35.00 | 17.39 | 39.76 |

**关键发现**：
1. **代码可执行性 ≠ 视觉高保真**：前沿模型图表执行率接近 100%，但 VRF-A 显著偏低（如 Gemini-3-Pro 60 vs Exec 96.97），生成"可运行"绘图比复现"精确美学与数据分布"容易得多。
2. **伪代码是普遍瓶颈**：Claude Opus 4.6 VRF-S 仅 17.39，GPT-5.4 仅 13.04，暴露算法格式化鲁棒性仍属未解挑战。
3. **传统 TEDS 高估表格解析**：严格 VRF-S 下分数大幅跌落（如 Claude Haiku 4.5 表 VRF-S 9.46 vs TEDS 44.10）。
4. **Open-weight 快速追赶**：Qwen3.5-397B-A17B 整体视觉保真度接近 GPT-5.4/Claude Opus 4.6。

### 多智能体消融（Table 5）
- Full MA System 较 Base 提升 **3–7 分**（Layout 73→77.2，Chart 60→66.5）。
- **去除 Visual Feedback** 反而提升公式 CDM（74.2→75.3），说明像素级视觉批评器会误判密集数学符号并覆盖正确 LaTeX。
- **去除 Multi-Agent** 轻微改善伪代码（62.5 vs 61.8），因多轮代码传递破坏缩进与语法细节。
- **单纯 Self-reflection** 导致**全面性能下滑**，表明无外部工具/环境反馈的自反思引入不稳定。

### 格式敏感性 QA（Fig. 6）
- 中等模型 Qwen3.5-Plus 从 LaTeX+Python 结构化表示中获得最大收益，甚至超越 Gemini-3-Pro 的 Markdown 输入；小模型几乎无法利用复杂程序结构；强视觉模型（Gemini-3-Pro）对截图与 Markdown 表现接近。

---

## 相关工作脉络
1. **TabLeX / Tab-to-LaTeX / TAB2LATEX / Table2LaTeX**：表格到 LaTeX 的孤立元素代码生成基准，**仅评估裁剪表格**，缺失页级上下文；CADP-Bench 要求完整页面重建。
2. **Im2LaTeX-100K / UniMER**：公式识别基准，**不含图表与表格联合场景**；CADP 支持多 SAE 耦合页级任务。
3. **ChartMimic / ChartX / ChartEdit**：图表到代码生成与编辑基准，**输入为预裁剪图**，无法评估图表在页上下文中的定位与引用关系。
4. **DaTikZv3**：支持 TikZ 可重绘图表，但样本量小（1000）、领域单一；CADP 使用通用 Python 并扩展至 1630 样本。
5. **READoc / olmOCR-bench / OmniDocBench**：纯文本/Markdown 提取基准，**不评估代码可执行性与视觉可编译性**。
6. **Mineru [24]**：高分辨率文档解析基础框架，用于 CADP-Bench 的 SAE 识别与源上下文提取，**非端到端可编译表示**。

---

## 局限性与未来方向
1. **数据分布偏 CS**：1630 样本中 CS 占 64%，其他学科代表性不足。
2. **图表数据为推断近似**：Ground truth Python 基于 Gemini-3-Pro Chart2Code 推断，存在与原始数据偏差的风险；非作者原始绘图代码。
3. **多智能体效率待优化**：Full MA System 平均调用 6 次，成本较高；部分模块（如 Visual Feedback）存在负效应。
4. **缺乏端到端微调验证**：当前评测均为 zero-shot/few-shot API 调用，未探索针对 CADP 范式的专项预训练或 RL 微调。
5. **未来方向**：
   - 基于可编译表示的**强化学习预训练/后训练**（编译成功/渲染相似度作为奖励信号）。
   - 利用 LaTeX 显式结构（`\ref`、`\cite`、表层级）构建**Graph-RAG**，缓解 chunk-based RAG 的结构破坏问题。
   - 扩展至法律、医疗、工程图纸等**结构化技术文档**领域。

---

## 研究启发与可借鉴点
1. **可编译评估范式可迁移**："生成→重新注入→编译→视觉比对"的闭环评估框架适用于任何需要高保真还原的文档/代码生成任务（如技术报告、专利、教材）。
2. **双重代码表示（LaTeX + Python）**：将排版语义（LaTeX）与数据/渲染语义（Python）解耦，为多模态生成提供结构化先验，值得借鉴至图表理解、公式生成等子任务。
3. **严格视觉评估优于传统 NLP 指标**：VRF-S/A、Pixel Sim. 等指标揭示 TEDS/执行率的高估偏差，提示在结构密集型任务中应采用**多粒度评估**（代码 + 视觉 + 可编译性）。
4. **多智能体设计的反直觉发现**：Self-reflection 无工具支撑会引入噪声；Visual Feedback 可能干扰密集数学渲染——提醒研究者**审慎设计 agent 回路**，避免盲目堆叠策略。
5. **结构化表示对中等模型的补偿效应**：LaTeX+Python 格式使 Qwen3.5-Plus 超越 Gemini-3-Pro（Markdown），表明**表示质量可部分弥补模型规模差距**，为低资源部署提供新思路。

---

## 关键术语表
- **SAEs（Structured Academic Elements）**：结构化学术元素，指学术页面中密集交织的表格、公式、图表、伪代码等结构性内容。
- **CADP（Compilable Academic Document Parsing）**：可编译学术文档解析，将页面重建为可重新编译校验的 LaTeX + Python 双重代码范式。
- **Dual-Code Generation**：双重代码生成，同时输出上下文 LaTeX（排版与文本）与可执行 Python（图表数据与渲染）。
- **Re-injection Compilation**：重新注入编译，将生成代码插入原始源上下文并编译为 PDF 以进行视觉一致性校验的评估协议。
- **VRF-S / VRF-A**：视觉重建保真度的严格（全维度满分才得分）与平均（多维度均值）变体。
- **Semantic Orphan**：语义孤儿，指孤立裁剪区域生成的元素因缺失页级上下文与交叉引用而失去语义锚点的问题。
- **Exec. Rate**：代码执行率，生成代码能成功编译/执行的样本比例。
- **Gemini-3-Pro Chart2Code**：用于将光栅图表转换为可重现 Python 代码的模型组件。

---

## 可复现要素
- **数据集**：CADP-Bench 已公开发布（论文第 1 页标注开源）。
- **代码/权重**：论文未明确开源代码仓库，但声明基准开放供后续研究。
- **关键超参**：模型温度 0.7；每样本评估 3 次取平均；LaTeX 编译使用 **pdfLaTeX** 引擎。
- **工具链**：Mineru 框架用于 SAE 识别与源上下文提取；Gemini-3-Pro 用于图表代码生成与标注；GPT-5.5 用于评估器依赖检验。
- **标注流程**：4 名 CS 本科生独立标注，2 人交叉核对，分歧由 CS 博士仲裁；总体一致率 88%。

---
