---
title: "KhatianDoc-A-Human-Verified-Benchmark-Diagnosing-Multimodal"
source: https://arxiv.org/pdf/2609.03597v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:04:52"
field: "低资源多模态法律文档理解"
keywords: ["Bengali legal document understanding", "multimodal LLM benchmark", "low-resource OCR", "mixed-radix numeral system", "privacy-preserving anonymization", "document information extraction", "legal QA"]
innovations: ["首个覆盖孟加拉 RS Khatian 与 Ana-Ganda 十六进制分数的人工核验基准", "四维递进任务揭示前沿 MLLM 在非标数字系统与多跳法律推理上的系统性能力缺失", "位置化令牌匿名化与双轨评分协议解决隐私保护与多跳指代的兼容难题"]
benchmarks: ["KhatianDoc"]
---

# 论文速读：KhatianDoc: A Human-Verified Benchmark Diagnosing Multimodal LLM Failure on Bengali Legal Land Records

## 一句话总结
论文构建了首个针对孟加拉国手写土地记录（RS Khatian）与特有 Ana-Ganda-Kora-Kranti-Til 十六进制分数系统的四维评估基准 KhatianDoc，并以人工验证的真值评估了六个主流多模态大模型，发现其在法律推理与混合基数算术任务上存在系统性能力缺失。

## 研究问题与动机
- **手写土地记录的数字化真空**：孟加拉国数百万宗土地的法定权属记录载于 RS Khatian 表格中，采用 Ana-Ganda 专用 Unicode 字形与混合基数分数系统，且为手写体；当前主流 OCR 与 Tokenizer 均无此覆盖，导致此类权威法律文件几乎完全脱离数字基础设施。
- **既有法律 NLP 基准的盲区**：现有 Legal DocVQA / 法律推理基准（如 ILDC、MultiEURLEX、LegalBench）仅面向现代打印排版与十进制文本，未触及“退化手写表 + 非标数字系统 + 多跳隐私敏感推理”的复合挑战。
- **零样本能力的真实诊断需求**：缺乏经过律师逐行核验的机器可判定真值，难以判断模型是在“近似求解”还是在“根本不具备该能力”，需要一份严格协议下的失败基线。
- **评估指标易被元数据漏洞利用**：作者自审发现，原评分代码中存在两项方向相反的人为 artifacts（常数字段与 null 字段导致的元数据指标虚高、对合规拒答的误罚），需在发布前纠正并公开报告。

## 核心贡献（创新点）
- **首个面向孟加拉 RS Khatian 与 Ana-Ganda 十六进制分数的基准**：不同于以往针对打印/拉丁脚本的法律文档评测，本工作首次提供机器可判定的手工转录真值与律师核验链路。
- **四维递进任务设计覆盖从字形到法律推理**：符号识别 → 基数换算 → 结构化字段抽取 → 法律文档 QA，使“是否具备某项能力”而非“程度高低”成为可诊断维度。
- **位置化 Token 匿名化保留多跳指代**：用 [PERSON_1], [PERSON_2] 等位置占位符替代真实人名，避免通用 [NAME] 抹除多跳问答所需的区分度。
- **公开双轨评分协议与自评审机制**：区分安全原件协议与公众删减图版本，同时主动披露并修正两项评分偏差，提供可复现的评估范式。
- **发现跨模型系统性失败而非随机波动**：39.3% 的复杂 QA 类别在所有模型上归零，算术任务甚至低于常数均值基线，证明是能力缺失而非规模不足。

## 方法详解
- **数据集来源与采集**：从孟加拉 Munshiganj 县 Vumi（土地）办公室获取 107 份真实 RS Khatian 扫描页（部分为双页配对），每份均为标准化政府表单手写填写，聚焦单一行政体裁以降低结构性混淆。
- **人工真值构建**：全程不使用 OCR，由标注员手写转录全部字段，再由具有土地法经验的孟加拉律师独立第二遍校验，双方在字形身份与十进制数值上达成完全一致。
- **隐私保护与分级开放**：文本真值将所有人名（所有者、父/夫名、住址前缀）替换为 [PERSON_N] 位置令牌；公开图像版本进一步用不透明框遮盖所有者姓名列、表单号、区域抬头、官员印章与签名，形成两套可追溯的评分协议。
- **Ana-Ganda 混合基数分数系统编码**：采用 16 Ana = 1 Plot、20 Ganda = 1 Ana、4 Kora = 1 Ganda、4 Kranti = 1 Kora、4 Til = 1 Kranti 的五层结构，Unicode 范围为 U+09F4–U+09F9；任务 2 将转换规则内嵌提示词以检验模型是否能在线应用该代数结构。
- **四维任务定义**：
  - Task 1 符号识别：95 张 256×256 灰度裁剪，标签覆盖 52 个独特字符串，评估 CER/EM。
  - Task 2 分数转十进制：输入符号串与规则，输出十进制值，评估 Exact Match / MAE。
  - Task 3 字段抽取：元数据级 + 261 行逐行级 JSON 输出，评估 metadata EM（视为上界）与 row-level F1。
  - Task 4 法律文档 QA：1,634 对问题，采用分层抽样的 300 题子集（73.7% 简单 / 26.3% 复杂），评估 EM 与 ANLS。
- **评测协议**：六模型统一经 OpenRouter 单密钥访问，固定零样本、确定性指令，无少样本示例与 CoT 诱导；被拒绝或丢失的 API 响应记为空字符串而非丢弃；每任务设置输出长度上限。

## 实验与结果
- **基线模型**：Gemini 2.5 Flash Lite、Qwen2.5-VL-72B、Qwen3-VL-8B、Llama 4 Scout、Gemma 4 26B、GPT-4o Mini（8B–72B+，开放/封闭混合）。
- **Task 4 归零Floor**：fraction_share、legal_fraction_math、counterfactual_check、conditional_filtering、multi_hop_reasoning 五个类别共 118 题（占分层集合的 39.3%）在所有六模型上精确匹配为零；total_area 近零。
- **Task 2 算术低于上下文无关基线**：常数均值基线（均值 0.3935）的 MAE 为 0.237，而所有能输出数值的模型误差均在 0.40–0.42，且 Exact 与 Near（容差 0.005）命中率重合，表明模型输出与输入去相关而非逼近。
- **Task 1 字符错误率高位平坦**：CER 分布在 76.89%–96.56% 之间，不随模型规模或访问方式显著变化，呈现统一盲点特征。
- **Task 3 行级 F1 偏低、元数据 EM 虚高**：Row F1 在 0.10%–25.84%；metadata EM 因两常数字段（district/upazila）与 65/107 的 null total_area 而虚高，作者将其改为上界说明并以 row-level F1 为主指标。
- **最强结果与修复后提升**：Gemini 2.5 Flash Lite 在 Task 4 原始 EM 为 13.33%，经拒答规范化重评后升至 21.67%；Llama 4 Scout 由 11.33% 升至 18.33%。作者未报告超越 26% 的其他单指标新高。
- **失败模式分型**：各模型呈现家族特异性坍缩——Gemini FL 倾向使用孟加拉卢比标记 U+09F2；Qwen/Gemma 使用数字-斜杠分数表示；GPT-4o Mini 在 Task 3 中 97/107 文档输出阿拉伯数字而非孟加拉数字，并在 Task 1 中幻觉出婆罗米字符。

## 相关工作脉络
- **Document IE / Legal DocVQA 类工作**（LayoutLMv3、DocVQA、FUNSD、CORD、ICDAR SROIE 等）以打印体拉丁/标准脚本与清洁排版为主，缺乏对非标手写 Unicode 字形与混合基数数字系统的评测。
- **多语言/低资源法律 NLP**（ILDC、MultiEURLEX、BanglaBERT、IndicBERT）聚焦现代打印体十进制文本，未处理“手写表 + 非十进制 + 隐私指代”复合设定。
- **数值与符号推理基准**（GSM8K、MATH、MathVista、NumGLUE、FinQA、LegalBench）均基于十进制算术，KhatianDoc 的混合基数五层结构在代数形态上前所未有。
- **隐私匿名化方案**（k-anonymity、通用 [NAME] 替换）会破坏多跳 QA 所需的实体区分，本文的位置化令牌设计与之形成对照。
- **定位差异**：本文并非单纯扩展语种覆盖，而是引入“字形/数字系统/法律多跳/手写退化”四维叠加的困难，首次量化了前沿 MLLM 在此类真实司法文件上的能力下限。

## 局限性与未来方向
- **地理与行政范围受限**：107 份文档仅来自 Munshiganj 县单一 Mouza，跨县、跨时期与不同 Vumi 办公室的版式变异未在样本中体现。
- **Task 1 样本量偏小**：95 个裁剪中稀有复合分数可能未被充分覆盖，需更大规模或更广泛来源的标注数据。
- **仅报告零样本结果**：未评测微调、RAG 或 In-context 设置，无法量化任务适配潜力。
- **复杂 QA 自动评分的局限**：复杂类别的机器评估依赖精确/近精确字符串匹配，可能与法律正当性不完全等价，需要人类法律专家复核。
- **未评估 OCR 辅助管线**：当前为纯图像输入，未来引入 OCR 辅助可能显著改变 Task 3/4 的失败分布，但尚待验证。

## 研究启发与可借鉴点
- **位置化令牌匿名化用于多跳 QA**：以 [PERSON_N] 保留跨句指代区分度，避免通用占位符破坏推理，可迁移至其他需要实体解析的法律/医疗文书基准。
- **自审式评分协议暴露元数据漏洞**：通过常数字段与 null 比例检验指标可信度，并在论文中公开修正；该“自评审 + 双轨公布”的做法可作为基准发布的最佳实践。
- **混合基数算数作为结构新颖性探针**：将规则注入提示但仍低于常数基线，说明失败来自代数结构缺失而非难度本身；该设计可推广至其他非标数制或领域专用度量系统。
- **拒答规范化合并同义负例**：将提示词指定拒答句式与短式负例归一到同一语义类，避免严格 EM 惩罚遵循指令的模型；适用于含否定/缺失信息的 QA 基准。
- **家族特异性外词坍缩的类型学记录**：按模型族记录字形替换模式（如卢比标记、数字-斜杠、婆罗米幻觉），可为后续针对性微调提供诊断信号。

## 关键术语表
- **RS Khatian**：孟加拉国修订测量（Revisional Survey）体系下的法定土地权属登记表格，具有直接法律效力。
- **Ana-Ganda-Kora-Kranti-Til**：孟加拉土地份额记录的混合基数分数系统，包含 16-A na、20-Ganda、4-Kora、4-Kranti、4-Til 五层结构。
- **Positional token anonymization**：以 [PERSON_N] 等位置占位符替代真实人名，保留多跳指代区分度同时保护隐私。
- **Zero-shot fixed protocol**：无少样本示例、无 CoT 诱导、确定性指令的统一评测协议，用于隔离模型真实能力。
- **Constant-mean baseline**：忽略输入、始终预测数据集均值作为算术任务的下界参考。
- **ANLS（Average Normalized Levenshtein Similarity）**：文档视觉问答中衡量生成文本与参考答案相似度的归一化指标。
- **Dual-scoring protocol**：区分安全原件协议与公开删减图协议，保证隐私合规的同时明确结果可比性边界。

## 可复现要素
- **数据集**：KhatianDoc 已公开于 HuggingFace（https://huggingface.co/datasets/RaiyanKhaan/KhatianDoc），含脱敏文本与部分遮盖图像；安全原件扫描页需通过 Vumi 办公室渠道申请。
- **代码与权重**：论文声明代码与数据公开，但具体仓库链接未在正文给出；模型为既有开源/闭源 MLLM，可通过 OpenRouter 统一调用。
- **关键超参**：输出长度上限（T1/T2: 30；T3: 500；T4: 200）；T2 提示中显式注入 Ana-Ganda 换算规则；评测使用单次确定性采样。
- **评分细节**：Task 1 用 CER/EM；Task 2 用 Exact/MAE；Task 3 用 row-level F1 为主、metadata EM 作上界；Task 4 用 EM/ANLS；拒答与 null 字段均已按 §6.2 规范化处理。
