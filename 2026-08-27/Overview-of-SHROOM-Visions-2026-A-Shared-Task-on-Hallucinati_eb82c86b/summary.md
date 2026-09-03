---
title: "Overview-of-SHROOM-Visions-2026-A-Shared-Task-on-Hallucinati"
source: https://arxiv.org/pdf/2608.25662v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 23:44:07"
field: "多模态大模型幻觉检测"
keywords: ["hallucination detection", "vision-language models", "shared task", "span detection", "multilingual evaluation", "model-agnostic benchmark", "SHEEP dataset"]
innovations: ["首次将SHROOM共享任务扩展至多模态LVLMs跨语言span级幻觉检测", "以人工编写幻觉作为模型无关评估基准验证泛化能力", "提出Corr/Corr_lbl/IoU三维互补评估并量化排名不确定性"]
benchmarks: ["SHEEP", "HALOQUEST", "VISAGE"]
---

# 论文速读：Overview-of-SHROOM-Visions-2026-A-Shared-Task-on-Hallucinati

## 一句话总结
SHROOM-Visions 2026 是第四届 SHROOM 共享任务，首次将幻觉检测扩展至大型视觉-语言模型（LVLMs）的多语言细粒度 span 级检测，使用包含人工编写与模型生成数据的 SHEEP 数据集，在四种语言上评测 27 支队伍的 600+ 提交系统。

## 研究问题与动机
- LVLMs 输出的幻觉问题不同于纯文本生成：不仅需要符合世界知识和语言连贯性，还必须忠实于伴随图像的具体视觉内容，导致对象虚构、属性误描述、文本误读、计数错误等特有失败模式。
- 现有基准过度依赖少数固定模型的输出（无论是作为标注来源还是合成数据生成器），难以区分"真正的幻觉理解进步"与"对特定系统偏好的过拟合"，且随着模型快速迭代迅速失效。
- 简单随机采样模型输出会导致罕见幻觉类型覆盖稀疏且分布偏差，而基于 LLM-judge 的预选择策略又将可靠性问题转移至评判模型本身。
- 缺乏兼顾模型无关性、多语言覆盖与细粒度分类的鲁棒评估基准，促使团队设计以人类编写幻觉为核心来源的评估框架。

## 核心贡献（创新点）
- **将 SHROOM 系列扩展至多模态领域**：首次面向 LVLMs 的视觉条件文本生成（VQA、图像描述等）进行细粒度 span 级幻觉检测共享任务，区别于以往仅关注纯文本的任务（SHROOM、Mu-SHROOM、SHROOM-CAP）。
- **提出模型无关的评估基准设计**：以人类编写幻觉（human-written samples）作为一等公民评估来源，搭配 LVLM 生成数据，验证检测器在脱离特定生成器时的泛化能力。
- **构建多语言字符级概率预测与 span 定位联合评估框架**：同时评估字符级置信度校准（Corr）、类别条件排序（Corr_lbl）与 span 边界精确定位（IoU），三种度量互补覆盖不同能力维度。
- **系统性揭示数据选择策略对排名的影响**：分析 Random、Silver-label (MAP)、Human-written 三类数据分区上的系统排名流动，发现随机采样会显著改变相对排序，而人工与银标数据高度一致。
- **公布 Bootstrap 排名不确定性量化结果**：通过 25000 次重采样提供 95% 置信区间和 P(>next) 概率，证明细粒度排名差异往往不具备统计稳定性。

## 方法详解
- **任务定义**：给定图像、提示和响应，对每个字符输出其属于幻觉 span 的概率 P(c)，并为每个检测到的 span 分配 5 类幻觉标签之一。
- **五类幻觉分类体系**：
  1. Invention（虚构）：提及图像中不存在的实体、对象、属性或事件；
  2. Mischaracterization（误描述）：对可见内容进行错误描述；
  3. OCR Problem（文本误读）：因误读图中文字导致的错误；
  4. Miscounting（计数错误）：对可见物品数量报告不准确；
  5. Other（其他）：不属于上述类别的幻觉。
- **评估指标**：
  - **Unlabeled Correlation (Corr)**：Spearman 相关系数 ρ(r, r̂)，衡量系统置信度分布与多标注者黄金标签的字符级排序一致性；当向量恒定时退化为 exact match。
  - **Labeled Correlation (Corr_lbl)**：对每种出现在金标准或预测中的标签 ℓ 分别计算 ρ(r_ℓ, r̂_ℓ) 后取平均，要求系统同时正确分类幻觉类型。
  - **Intersection-over-Union (IoU)**：将 r 和 r̂ 二值化为字符索引集合后计算重叠率，衡量 span 边界定位精度，不受置信度与类别影响。
- **数据构造**：基于 SHEEP 数据集（Mickus et al., 2026），共 20,000 样本（每语言 5,000），图像与问题源自 HALOQUEST（视觉歧义图像 + 错误前提问答）和 VISAGE（异常属性物体）两个英文资源，非合成图像被保留；提示通过 NLLB-200（法语/意大利语）和 Qwen3-8B（中文）机器翻译；LVLM 响应从 InternVL3、MiniCPM-V 4.5、Llava-NeXT、Qwen3-VL（均 8B 规模）及 Gemma3-27B 采样，使用 τ=0.7、512 token 上限、每输入 5 个随机种子。
- **采样策略**：三种互补策略——Random（随机采样 LVLM 输出）、Silver (MAP, model-assisted pre-selection，LLM-judge 辅助获取标签平衡子集)、Human（人工编写并经翻译与后期编辑确保多语言覆盖）；训练集由 Random 与 MAP 组成，测试集 Human 部分独占于测试集以检验模型无关泛化。

## 实验与结果
- **参与规模**：27 支队伍，623 次提交，平均每语言 155.75 次提交；EN 最多（208 次），FR 和 ZH 各 137 次，IT 141 次。
- **最强系统性能**：vroom-vroom 在 ZH/FR/IT 取得领先（ZH: Corr=0.61, IoU=0.53；FR: Corr=0.58, IoU=0.52；IT: Corr=0.56, IoU=0.48），TÜRKSAT 在 EN 最强（Corr=0.55, IoU=0.48）；各语言最佳平均约 0.58（Corr）、0.46（Corr_lbl）、0.51（IoU），较基线提升 30–40 个百分点。
- **语言差异**：ZH 平均 Corrscore 略高于 0.4 为最佳，EN 最具挑战性；但各语言分数范围狭窄、方差相近，表明存在独立于语言的性能瓶颈。
- **排名稳定性**：Bootstrap 25000 次重采样显示领先系统排名区间极大（EN 可达 [1,15]，ZH [1,10]，IT/FR [1,14]），相邻系统 P(>next) 概率多在 0.32–0.47 之间，远未达到 0.95 稳健阈值。
- **数据策略影响**：Human 与 Silver 分区排名高度一致（ρ_S ∈ [0.861, 0.973]），Random 与二者相关性更低（尤其 ZH Corr_lbl 仅 ρ=0.575）；Friedman 检验显示 EN/ZH 所有指标、IT/FR 的 Corr_lbl 均存在系统性评分差异，但总体排序趋势相对稳定。
- **跨指标一致性**：IoU 与 Corr/Corr_lbl 呈强线性相关，Corr 普遍高于 IoU（容忍边界偏差），Corr_lbl 最低（增加类别判断约束）。

## 相关工作脉络
- **SHEEP 数据集**（Mickus et al., 2026）：首创多语言 span 级数据集，配对人工编写与 LVLM 输出，证明人工样本具有更高标注者间一致性与跨模型相关性，本文以此为基础扩展至多模态场景。
- **M-HalDetect**（Gunjal et al., 2024）与 **HalLoc**（Park et al., 2025）：前者提供细粒度幻觉检测，后者实现 token 级定位，但均局限于英语单模型设置，本文通过多语言与模型无关设计超越其局限。
- **HalluShift++**（Nath et al., 2025）与 **HaloQuest**（Wang et al., 2024b）：前者关注 VLM 内部表征偏移，后者提供视觉歧义 VQA 数据源（被本文用作图像/问题素材），本文将其整合到共享任务框架中。
- **SHROOM 系列既往任务**（Mickus et al., 2024; Vázquez et al., 2025; Sinha et al., 2025）：分别针对单语言英语 NLG、多语言问答、科学文本，本文填补了多模态跨语言 span 检测空白。
- **多语言幻觉检测工作**（Li et al., 2025; Mubarak et al., 2025; Nguyen et al., 2026; Karlgren et al., 2024）与 **多模态幻觉工作**（Han et al., 2025 等）：以往研究各自专注语言或模态单一维度，本文首次实现两者的交叉统一评估。

## 局限性与未来方向
- **数据集覆盖有限**：SHEEP 仅涵盖 4 种语言、特定领域（HALOQUEST/VISAGE）及有限 LVLM 输出，难以全面代表真实世界更广泛的幻觉现象。
- **静态基准风险**：尽管测试标注保密，仍存在潜在的数据泄漏与过拟合风险，需动态更新机制。
- **空实例/拒答行为未被充分评估**：测试集中 15.5%–25.5% 样本无标注幻觉，而系统空预测比例高达 30.7%–46.0%，当前指标（span-overlap 与概率相关）未显式衡量拒答决策的合理性。
- **未来方向**：引入显式空实例/拒答度量；扩展语言与领域覆盖；开发抵御模型过时与数据泄漏的动态基准；探索缓解排序不稳定的评估协议。

## 研究启发与可借鉴点
- **多策略混合数据构建范式**：随机采样 + LLM-judge 预筛选 + 人工编写三元组合可有效平衡覆盖率、标签平衡性与模型无关性，可迁移至其他多模态评估任务。
- **字符级概率输出替代硬边界判定**：以 P(c) 向量配合 Spearman 相关评估，比精确 span 边界判定更能容忍标注噪声，值得在 span 提取任务中推广。
- **数据分区独立性验证机制**：将测试集按生成策略（Random/Silver/Human）拆分评估，可系统性检验模型的泛化边界与脆弱性，建议作为未来共享任务的标配分析。
- **排名不确定性量化纳入报告规范**：Bootstrap 重采样 + 95% CI + P(>next) 的概率化排名分析应成为多系统对比研究的常规做法，避免过度解读微小分差。
- **空预测行为的显式建模机会**：当前任务未奖励/惩罚合理拒答，未来可设计 abstention-aware 损失或奖励机制，推动模型在低置信度时主动放弃输出。

## 关键术语表
- **SHEEP**：Set for Human-written and Electronic Erroneous Productions，多语言 span 级幻觉数据集，配对人工编写与 LVLM 输出，作为本文评估基础。
- **Hallucination span detection**：在生成的文本中定位并分类存在幻觉的具体字符片段，而非仅判断整句是否幻觉。
- **Corr（Unlabeled Correlation）**：Spearman 相关系数，衡量系统字符级幻觉概率向量与多标注者黄金标签向量的排序一致性。
- **Corr_lbl（Labeled Correlation）**：按幻觉类别分别计算 Corr 后取平均，要求系统在检测幻觉的同时正确分类其类型。
- **IoU（Intersection-over-Union）**：将预测与黄金 span 二值化为字符集合后计算重叠率，衡量 span 边界定位精度。
- **MAP（Model-Assisted Pre-selection）**：使用 LLM-judge 预筛选出标签平衡的样本子集，作为训练/测试中的银标数据策略。
- **Rank instability**：由于测试集采样波动导致系统排名大幅变动的现象，Bootstrap 重采样可量化此不确定性。
- **Abstention behavior**：系统在可能无幻觉的样本上选择不输出任何 span 的行为，当前评估未显式考量其合理性。

## 可复现要素
- **数据集**：SHEEP 数据集（Mickus et al., 2026），20,000 样本，四语言（EN/FR/IT/ZH），含 Train/Test 划分；论文提供详细统计（Table 1），但未明确声明公开链接，需查阅原 SHEEP 论文获取。
- **代码/权重**：参与系统代码未在本文集中开源；baseline 系统由组织方提供（HalluShift++ baseline），participants kit 含评分程序与格式检查器，但具体 URL 需查阅附录。
- **关键超参**：LVLM 响应生成使用 τ=0.7、512-token 上限、每输入 5 个随机种子；翻译使用 NLLB-200（3.3B）与 Qwen3-8B；评估使用 25,000 次 Bootstrap 重采样。
