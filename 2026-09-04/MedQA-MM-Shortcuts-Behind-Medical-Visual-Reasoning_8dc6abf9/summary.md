---
title: "MedQA-MM-Shortcuts-Behind-Medical-Visual-Reasoning"
source: https://arxiv.org/pdf/2609.03261v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-07 23:12:01"
---

# 论文速读：MedQA-MM: Shortcuts Behind Medical Visual Reasoning

## 一句话总结
本文提出**审计-修复框架**，通过对6个医学多模态MCQ数据集进行四条件模态探测与线索级匹配修复，揭示传统基准准确率存在“推理膨胀”，模型高分常被选项文本模式、可见图像文字、人工标注及SDOH线索等非视觉捷径贡献；据此构建**MEDQA-MM**（1,000条捷径缓解子集），证明路径级证据对真实医学视觉能力评估不可或缺。

## 研究问题与动机
- **分数记录不完整**：现有医学多模态基准仅记录最终答案，不记录得出答案的决策路径，正确率可能由预期图像推理之外的捷径线索贡献。
- **推理膨胀（Reasoning Inflation）**：即使模型实际依赖选项措辞、题干文本、图像可见文字、人工标注或设备伪影作答，高分仍被过度解读为图像grounding能力的证据。
- **单一准确率失真**：32.63%的13-配置模型Full input准确率在剥离文本/选项线索后出现显著回落，表明传统指标无法区分图像推理路线与捷径路线。
- **医学场景特有捷径缺失系统化治理**：SDOH人口学线索、人工视觉提示、医学专用绝对词distractor等垂直捷径缺乏可复现的检测与修复协议。

## 核心贡献（创新点）
- **审计-修复框架与证据阶梯**：提出“候选线索 → 模态消融支持 → 匹配修复支持 → 专家验证”四阶诊断管线，首次将路径级证据纳入医学多模态评估闭环。
- **三类选项形式线索修复协议**：针对绝对词distractor、长度差、空间/介词标记设计硬性安全约束与最小化改写策略，保障医学等价性的同时消除文本捷径。
- **MEDQA-MM基准构建**：从MedThinkVQA、MedXpertQA-MM、MMMU-HM中筛选1,000条捷径缓解条目，提供可复用的路径诊断测试集。
- **四条件模态探测管道**：Full input / Text-only / Options-only / Image+Options 四配置对比，量化图像渠道的真实贡献度，修正传统“全模态”评估盲区。
- **SDOH捷径合成注入与图像人工标记修复管线**：前者通过四节点自动判定与安全声明约束临床不变性；后者提供5种Best-of-Five变体与三列judge标准，实现视觉侧捷径的自动化清洗。

## 方法详解
- **证据阶梯与模态探测**：对7,706条原始题目进行审计，通过四配置输入运行13-配置模型，记录准确率落差以定位捷径依赖强度。
- **绝对词Distractor修复**：移除含 `always/never/must/cannot/only/all/none/neither/required/confirmed` 等绝对措辞的干扰项。硬性安全规则：不改选项顺序/字母、不改变解剖/侧别/生物体/诊断/数值单位/严重程度/治疗方向/因果方向；若移除后distractor变部分正确或歧义则返回 `no_safe_edit` / `needs_human_review`。编辑策略包括替换为仍错误的解释/限制/具体错误子集，输出结构化JSON（含 `status`, `option_patches`, `residual_shortcut_risk` 等字段）。
- **长度捷径修复**：当正确答案唯一最长时，最小化改写1–2个distractor使其比正确答案至少长 **5个可见字符**（`gap_chars ≥ 5`）。推荐方式：拼写缩写全称、使distractor与选项风格平行、同义改写显式化隐含措辞。禁止添加通用填充词或改动正确答案。
- **空间/介词线索修复**：保守编辑优先并行化distractor结构（如 `chest radiograph` → `chest radiography`；`oral acyclovir` → `oral-acyclovir therapy`）；允许对正确答案弱措辞微调（如 `follow-up` → `monitoring`）。禁止新增剂量/阈值/范围/位置/解剖/时序语义改变或引入meta-option线索。
- **人工视觉线索定位与修复**：定位箭头、圆圈/椭圆、方框/矩形、手绘轮廓、点标记、高亮/掩码、指示线等；不处理正常解剖、病变、钙化、病理染色、DICOM标签、测量卡尺、UI覆盖层等。采用5种Best-of-Five变体：`complete_stroke_removal` / `edge_cleanup` / `conservative_inpaint` / `no_new_markup` / `local_texture_match`。Judge使用三列面板（原图/红色掩码/修复图）判定 `cue_removed_completely`、`medical_content_preserved`、`no_new_artifact`，仅当全部通过时输出 `pass_for_eval=true`。
- **SDOH捷径管线**：四节点架构：Node A（正则命中线索判定）、Node B（正则遗漏审计）、Node C（临床相关性判定：`incidental`/`legitimate_prior`/`essential`/`high_risk_conflict`/`unsafe`）、中性化与QA。共享约束：仅用案例文本/选项/模型推理/元数据显式信息；自然审计与合成注入测试分离；临床不变性需医生审核。合成注入4种变体（`race_black`、`race_hispanic_latino`、`ses_low_income_uninsured`、`language_interpreter`），并附安全声明禁止假设低收入导致就医延迟或西班牙语翻译暗示移民暴露。

## 实验与结果
- **数据集规模**：审计覆盖6个医学多模态MCQ数据集（AMBOSS Image Questions 646、JAMA Clinical Challenge 1,621、MMMU-HM 1,712、MedThinkVQA 720、MedXpertQA-MM 2,000、NEJM Image Challenge 1,007），共 **7,706** 题；构建MEDQA-MM捷径缓解子集 **1,000** 题。
- **基准模态脱落实验**：13-配置模型在原始题上 Full input 准确率 **62.63%**，但 Text-only 达 **53.96%**，Options-only **29.71%**，Image+Options **43.84%**，显示文本/选项线索贡献巨大。
- **匹配修复后准确率下降**：长度差修复（306对）ΔAcc **-6.58 pp**；绝对/显眼修复（101对）ΔAcc **-3.50 pp**；空间/介词
