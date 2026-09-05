---
title: "Quantifying-and-Mitigating-Korean-Jamo-Level-Typographical-V"
source: https://arxiv.org/pdf/2608.30229v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:27:40"
field: "多语言 NLP 鲁棒性"
keywords: ["Korean typographical errors", "jamo-level perturbation", "LLM robustness", "chain-of-thought routing", "internal representation probing"]
innovations: ["形式化五类韩语音节内 jamo 级键盘扰动框架", "揭示 typo 在 LLM 内部表征中产生独立于答案正确性的 distinct signal", "提出基于内部探针的按需 CoT 路由策略 TACoT"]
benchmarks: ["KMMLU", "HAERAE-GK", "HRM8K"]
---

# 论文速读：Quantifying-and-Mitigating-Korean-Jamo-Level-Typographical-V

## 一句话总结
本文系统化量化了大语言模型（LLM）对韩语特有**音节内辅音字母（jamo）级别打字错误**的脆弱性，发现模型准确率随扰动强度单调下降且参数规模无法提供鲁棒性；进一步揭示此类错误会在模型内部表征中 induce 一种独立于“答案错误”的**独特信号**，并基于该信号提出 **TACoT（Typo-Aware Chain-of-Thought）** 推理策略，仅在探针检测到疑似错误时调用 CoT，从而以约 37% 的推理成本节省恢复大部分 CoT 的准确率收益。

## 研究问题与动机
1. **现有鲁棒性基准以字符级编辑为主**：主流 typo 鲁棒性研究（如 PromptRobust、PromptBench）多针对英语等字母文字，将错误建模为表面字符替换/删除，无法刻画韩语等黏着语中发生在**音节内部子字符（jamo）层面**的键盘输入错误。
2. **现有 GEC 管道无法可靠修复 jamo 级错误**：韩语语法纠错（GEC）系统（如 KoGEC）旨在恢复合乎语法的文本，但 jamo 级扰动可能产生**另一个合法但语义不同的音节**（valid‑form ambiguity）或**暴露孤立辅音字母**（exposed standalone jamo），导致纠错系统要么无变化、要么进行破坏语义的改写。
3. **LLM 对音节内噪声的鲁棒性未知**：参数规模扩大是否自然带来对这类结构性噪声的免疫？LLM 内部表征是否隐含可检测的此类错误信号？
4. **缺乏可控、可量化的评估基准**：韩国 LLM 鲁棒性评测资源匮乏，亟需一个基于真实键盘输入机制、覆盖多种 jamo 级扰动类型的受控基准。

## 核心贡献（创新点）
1. **形式化五类键盘驱动的 jamo 级扰动框架**：首次系统定义并实现针对韩语音节内部结构的五种真实输入错误类型（辅音字母替换、终声删除、辅音字母重复、空格删除、辅音字母转置），弥补现有字符级编辑模型的不足。
2. **建立受控 Korean Typo Benchmark 并量化 LLM 脆弱性**：在 KMMLU 基准上施加五种扰动×五档强度（5%‑25%），证明四种开源模型（2.4B‑7.8B）的准确率均单调下降，且**参数规模不能缓解音节内脆弱性**。
3. **发现内部表征中的 distinct typo signal 并验证其泛化性**：通过 Fisher 分离分数定位对 typo 敏感的 Transformer 层，证明 typo 引起的表征偏移**不可还原为普通答案错误**，且基于线性探针可在**未见过的错误类型**上保持高 AUROC（0.905‑0.943）。
4. **提出 TACoT 路由型防御策略**：利用轻量级内部探针实时检测疑似 typo 输入，仅当置信度超过阈值时才启用 CoT 推理，在几乎不损失准确率的前提下将平均输出 token 数降低约 37%，实现成本感知的鲁棒性提升。

## 方法详解
### 1. 扰动类型与强度（§3‑4.2‑4.3）
- **五种 jamo‑level 扰动**（基于标准 Dubeolsik 韩语键盘布局）：
  - **Jamo Substitution**：相邻键位辅音字母替换，常生成另一个合法音节（如 `값` → `걅`）。
  - **Jongseong Deletion**：删除终声（coda），始终生成合法音节（如 `값` → `가`）。
  - **Jamo Repetition**：复制某一成分并作为孤立 jamo 附加（如 `값` → `값ㅁ`）。
  - **Space Deletion**：删除词间空格，合并相邻词语（不影响音节内部结构）。
  - **Jamo Transposition**：交换声母/韵母/终声顺序，破坏音节结构（如 `값` → `ㄱㅁㅏ`）。
- **五档强度**：各类型在题目文本中随机选取 5%、10%、15%、20%、25% 的音节独立施加扰动，共 35,030×5×5 = 875,750 条扰动样本。

### 2. 评估基准与模型（§4.1‑5.1）
- **KMMLU**：35,030 道韩语专家级四选一题目（涵盖人文、社科、STEM），仅对题目施加扰动，选项保持不变。
- **评估模型**：EXAONE‑3.5‑2.4B / 7.8B、A.X‑3.1‑Light、Qwen3‑4B（零样本、贪婪解码、max_new_tokens=8）。

### 3. 内部表征分析（§6）
- **Fisher 分离分数定位敏感层**：
  \[
  J(l) = \frac{\|\mu_{\text{typo}}^{(l)} - \mu_{\text{clean}}^{(l)}\|_2^2}{\text{tr}(\Sigma_{\text{clean}}^{(l)}) + \text{tr}(\Sigma_{\text{typo}}^{(l)})}
  \]
  选取 clean/typo 表征均值距离大、类内分散小的层（EXAONE‑2.4B: l=10；EXAONE‑7.8B: l=9；A.X‑Light: l=9；Qwen3‑4B: l=18）。
- **证明 typo signal 独立于答案正确性**：构造 clean‑incorrectness 方向 \(d_{\text{wrong}} = \mu_{\text{wrong}} - \mu_{\text{correct}}\) 与 typo 方向 \(d_{\text{typo}} = \mu_{\text{typo}} - \mu_{\text{correct}}\)，将隐藏状态投影到正交基上，发现 typo 样本始终沿 \(d_{\text{typo}}\) 偏移，与最终答案对错无关。
- **留一类型线性探针**：在 HAERAE‑GK 上训练 logistic regression 分类器（L2 正则化 C=1.0），以 clean/typo 为标签，在 KMMLU 的**未见类型**上评估，AUROC 达 0.905‑0.943。

### 4. TACoT 推理路由（§7.1）
- **训练**：在 HAERAE‑GK 的 balanced 样本（clean vs. jamo‑level typo）上训练同一 logistic regression 探针。
- **阈值选择**：在留持验证集上最大化 Youden’s J 统计量（\(J = \text{TPR} - \text{FPR}\)）。
- **推理规则**：
  \[
  \text{route}(x) = \begin{cases}
  \text{CoT} & P(\text{typo} \mid h_l(x)) \geq \theta \\
  \text{Standard} & P(\text{typo} \mid h_l(x)) < \theta
  \end{cases}
  \]
- **对比基线**：Standard、Meta‑Cognition（提示模型注意可能含 typo）、GEC（KoGEC 预处理后 Standard）、CoT（全量启用，max_new_tokens=1024）。

## 实验与结果
- **主要数据集**：KMMLU（主基准）、HAERAE‑GK（校准/探针训练）、HRM8K（跨任务泛化验证）。
- **核心发现**：
  1. **准确率随扰动强度单调下降**，jamo 级扰动降幅最大（EXAONE‑7.8B 在 Jamo Transposition 25% 强度下下降 10.0%p，Qwen3‑4B 下降 7.0%p）；Space Deletion 因 subword tokenizer 可部分恢复边界而表现接近 clean。
  2. **参数规模不带来鲁棒性**：EXAONE‑7.8B 在 Jamo Transposition 上的降幅甚至大于 2.4B 版本。
  3. **GEC 与 Meta‑Cognition 基本无效甚至有害**：GEC 对 Jamo Substitution 普遍降低准确率；Meta‑Cognition 在 Qwen3‑4B 上全面下降。
  4. **CoT 是最强缓解策略**：A.X‑Light 在 clean 上从 49.5% 提升至 57.1%（+7.6%p），各类 typo 均提升 >5.9%p。
  5. **TACoT 恢复大部分 CoT 收益且大幅降本**：在四种模型与五种 typo 类型上，TACoT 平均输出 token 数比 CoT 减少约 37%（26%‑49%），准确率接近全 CoT；与匹配速率的随机路由相比，TACoT 在最高强度下仍高出 1.63‑2.28 个百分点。
- **HRM8K 泛化验证**：数学推理任务（自由文本输出）上趋势一致，TACoT 同样有效。

## 相关工作脉络
1. **字符级 typo 鲁棒性研究**（Gao et al., 2018; Zhu et al., 2024a,b; Gan et al., 2024）：聚焦英语等文字的可见字符编辑，本文将其扩展至**音节内部亚字符级别**，指出韩语句法结构带来的新型脆弱性。
2. **多语言键盘 typo 评估**（Zhao et al., 2026）：虽涉及多语言，但未专门建模韩语 jamo 内部扰动及由此产生的 valid‑form ambiguity / exposed‑jamo 两类 failure mode。
3. **韩语语法纠错（GEC）**（Yoon et al., 2023; Lee et al., 2021; Kim et al., 2024）：旨在恢复语法正确性，本文证明其对 jamo 级扰动**修复能力有限**（可能保持原样或改写语义），凸显 pre‑correction 方案的不足。
4. **LLM 内部表征探针**（各类 probe 工作）：本文首次将探针用于**检测结构型输入噪声**而非任务属性，并证明该信号独立于答案正确性，为 routing‑based 防御提供新依据。
5. **CoT 推理增强鲁棒性**（Wei et al., 2022）：本文揭示全量 CoT 成本高，提出**基于内部信号的按需调用**策略，在保持效率的同时恢复主要增益。
6. **韩语文本鲁棒性基准**（Haerae Bench, KMMLU）：本文填补了韩国 LLM 输入噪声评测的空白，建立首个系统性的 jamo‑level 扰动基准。

## 局限性与未来方向
1. **扰动为合成生成**：虽覆盖 96.2% 的真实键盘错误，但仍可能无法完全匹配真实用户错误分布；未来需在自然用户数据上验证。
2. **评估资源有限**：目前仅覆盖 KMMLU（知识问答）与 HRM8K（数学推理），需扩展至更多领域与任务格式。
3. **依赖内部隐藏状态访问**：TACoT 与探针需本地部署的开放权重模型，**无法直接应用于 API‑based LLM**（如 Gemini、Claude 等）；未来需探索黑盒替代方案。
4. **未直接修正表征偏移**：TACoT 仅通过路由到 CoT 间接补偿，未对 typo‑induced 表征偏移进行显式对齐或修复；未来可开发表征修复模块。
5. **模型规模受限**：较大模型（235B 等）仅报告准确率下降，未进行探针/TACoT 验证；需在更大规模模型上进一步测试。

## 研究启发与可借鉴点
1. **音节/形态素内部扰动建模**：对于其他黏着语（如日语假名组合、泰语元音符号等），可借鉴本工作的“内部亚字符扰动”思路，构建符合其书写系统的错误类型。
2. **表征偏移独立于答案正确性的验证范式**：通过构造 clean‑incorrectness 方向与噪声方向的正交分解，可通用化地证明模型对某类输入噪声的**结构化感知**，适用于其他噪声类型（如语音识别错误、OCR 错误）。
3. **按需路由的 Cost‑Aware 推理**：将轻量探针作为推理路由的触发器，在多数输入上使用标准解码、仅在检测到异常时启用昂贵推理，是一种**高效鲁棒性提升范式**，可迁移至其他需要高可靠性场景。
4. **Fisher 分离分数定位敏感层**：无需训练分类器即可快速定位对特定噪声最敏感的 Transformer 层，可作为通用的表征诊断工具。
5. **留一类型泛化评估**：探针在训练时不包含某类型，测试时评估该类型，可避免过拟合特定扰动模式，提高方法的真实性与泛化可信度。

## 关键术语表
**Jamo（辅音字母）**：韩语音节块的内部组成单位，包括声母（onset）、元音（nucleus）和终声（coda/coda）。
**Valid‑form ambiguity**：jamo 级扰动生成另一个合法韩语句节，导致表面形式正确但语义改变，难以被传统拼写检查识别。
**Exposed standalone jamo**：扰动破坏音节结构，使孤立的辅音字母裸露在外，形成非标准字符序列。
**KMMLU**：Korean Massive Multitask Language Understanding，包含 35,030 道韩语专家级四选一的基准测试。
**Fisher 分离分数（J(l)）**：衡量第 l 层 clean 与 typo 隐藏状态之间可分离性的指标，用于定位对 typo 最敏感的 Transformer 层。
**TACoT（Typo‑Aware Chain‑of‑Thought）**：基于内部探针信号的推理路由框架，仅当检测到疑似 typo 时启用 CoT，否则使用标准解码。
**Youden’s J 统计量**：最大化 TPR − FPR 的阈值选择准则，用于确定探针的 typo 检测阈值。
**HAERAE‑GK**：HAERAE Bench 的 General Knowledge 子集，用作探针训练与阈值调优的校准数据（与 KMMLU 题目不重叠）。

## 可复现要素
- **数据集**：KMMLU（公开）、HAERAE Bench（公开）、HRM8K（公开）；扰动生成脚本与代码已开源：https://github.com/SJLee0311/korean-jamo-typo
- **模型权重**：EXAONE‑3.5‑2.4B/7.8B、A.X‑3.1‑Light、Qwen3‑4B 均为开源模型；KoGEC 模型为公开资源。
- **关键超参**：温度=0，max_new_tokens（Standard=8，CoT=1024）；logistic regression 正则化 C=1.0；Fisher 层：EXAONE‑2.4B l=10，EXAONE‑7.8B l=9，A.X‑Light l=9，Qwen3‑4B l=18。
- **推理框架**：vLLM；探针使用 HuggingFace 提取最后一 token 隐藏状态，scikit‑learn 实现。
