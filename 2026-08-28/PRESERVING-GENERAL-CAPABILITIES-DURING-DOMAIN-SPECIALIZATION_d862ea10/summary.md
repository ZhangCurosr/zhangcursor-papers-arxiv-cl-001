---
title: "PRESERVING-GENERAL-CAPABILITIES-DURING-DOMAIN-SPECIALIZATION"
source: https://arxiv.org/pdf/2608.26735v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 15:27:10"
---

# 论文速读：PRESERVING-GENERAL-CAPABILITIES-DURING-DOMAIN-SPECIALIZATION

## 一句话总结
本文针对垂直领域 SFT 导致大模型通用能力显著退化的问题，提出不确定性校准的多教师在线策略蒸馏（Uncertainty-Calibrated MOPD）；通过双温度采样拓宽候选轨迹池、正优势密度过滤筛选高价值序列，并结合中心对数似然（CLL）在 Token 级验证更新方向与教师背书的 consistency，在保持领域性能的同时高效恢复通用推理、编程与指令遵循能力。

## 研究问题与动机
- **领域专业化引发的通用能力衰退**：垂直领域微调能提升目标行为，但常伴随推理、数学、代码、指令遵循等通用能力的灾难性遗忘，而实际系统需在领域表现与通用助手能力间取得平衡。
- **标准 MOPD 难以暴露大正优势信号**：学生按自身条件分布采样时，Token 概率通常已较高，教师概率超过学生且优势值较大的“黄金机会”罕见，限制了对强正学习信号的利用。
- **仅凭优势符号决策更新方向不可靠**：优势为负仅代表教师概率低于学生，但教师仍可能对该 Token 给予强背书；盲目压制会破坏已对齐行为并 destabilize 训练。
- **通用训练数据不可得**：避免重复构建高质量通用语料库，希望在不依赖外部通用数据的情况下，仅凭教师对自采样轨迹的监督实现能力恢复。

## 核心贡献（创新点）
- **揭示 MOPD Token 级信号利用的结构缺陷**：系统分析表明优势符号只能提议更新方向，无法独立评估该方向是否符合教师自身分布的背书程度，填补了现有 OPD 对方向可靠性分析的空白。
- **提出 Uncertainty-Calibrated MOPD 分层筛选框架**：将“机会发现”与“方向验证”解耦，序列级双温度采样+正密度过滤负责挖掘高正优势轨迹，Token 级 CLL 门控负责校验每个更新是否与教师背书一致。
- **设计熵自适应的连续教师背书度量**：用中心对数似然（CLL）替代固定 top-k 硬阈值，以教师分布熵为基准动态校准背书分数，使保留概率平滑反映教师对当前 Token 的典型性判断。
- **双领域实证验证显著优于现有基线**：在角色扮演与医疗 specialization 中，相对 vanilla MOPD 通用能力平均分别提升 4.73% 与 10.84%，且垂直领域性能持平或进一步提升。

## 方法详解
- **基础设定**：学生 $\pi_S$ 从领域 SFT 模型初始化；冻结的领域 SFT 模型为领域教师，原始通用大模型为通用教师。按 prompt 标签路由，采用 reverse-KL 风格的 MOPD 损失：$\mathcal{L}_{\mathrm{MOPD}} = -\mathbb{E}[\frac{1}{T}\sum_t \widehat{A}_t \log \pi_S(y_t|q,y_{<t})]$，其中 $\widehat{A}_t$ 为 clip 后的教师-学生对数概率差 $A_t$。
- **Stage 1：双温度采样**：对每个 prompt $q$ 采样 1 条 anchor $y^a \sim \pi_S(\cdot|q; T_a=1.0)$ 与 $m$ 条探索响应 $y_j^e \sim \pi_S(\cdot|q; T_e>1.0,\ \text{top-p}=0.9)$，以更高温度打破学生高概率区域的局部聚集，增加大正优势 Token 的曝光概率。
- **Stage 2：正优势密度轨迹过滤**：定义响应正优势密度 $r(y)=\frac{1}{\max(n_+,1)}\sum_{t:A_t>0}A_t$。保留 anchor，仅当 $r(y_j^e)\geq r(y^a)$ 时保留探索响应；以同 prompt 的标准响应为基准消除跨任务难度差异，过滤掉仅增加噪声多样性的低质轨迹。
- **Stage 3：CLL 方向一致性过滤**：计算教师熵 $H[p_T]$，构造背书分数 $s_{\mathrm{cll}}(x)=\exp(\min(\log p_T(x)+H[p_T],0))=\min(p_T(x)/b_T,1)$，其中 $b_T=\exp(-H[p_T])$ 为熵自适应典型概率尺度。结合优势符号 $d_t=\mathrm{sgn}(A_t)$ 得统一保留概率 $w_{\mathrm{cll}}=\frac{1+d_t(2s_{\mathrm{cll}}-1)}{2}$；训练时对非零优势 Token 以概率 $w_{\mathrm{cll}}$ 进行 Bernoulli mask，方向与背书一致则高概率保留，不一致则高概率丢弃。

## 实验与结果
- **实验设置**：角色扮演（Qwen3-4B，CoSER SFT 为领域教师）与医疗（II-Medical-7B 为领域教师，Qwen3-8B 为通用教师）两场景。基线包括 Base、SFT、Vanilla MOPD、SelecTKD、ReOPOLD。通用评估含 GPQA-Diamond、AIME25、ZebraLogic、HMMT25、LiveCodeBench v5、IF-Eval、WritingBench、Arena-Hard v2、LiveBench；领域评估使用 CoSER 指标及 MedQA-USMLE、MedXpertQA、PubMedQA。
- **主要结果**：角色扮演中，本方法通用平均 53.10（vanilla MOPD 50.70，+2.40），领域平均 45.00（vanilla MOPD 41.22，+3.78）；医疗中，通用平均 54.38（vanilla MOPD 49.06，+
