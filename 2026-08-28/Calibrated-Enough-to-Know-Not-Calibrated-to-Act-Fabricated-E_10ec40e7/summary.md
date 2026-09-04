---
title: "Calibrated-Enough-to-Know-Not-Calibrated-to-Act-Fabricated-E"
source: https://arxiv.org/pdf/2608.27167v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 15:22:15"
field: "LLM 智能体可信校准与拒答能力"
keywords: ["aleatoric unanswerability", "action gate", "calibration", "trustworthy LLM agents", "evidence-induced commitment", "synthetic fine-tuning", "degenerate baseline"]
innovations: ["通过全虚构面板与真实面板的统计等价实验，因果隔离展示权威感为行动触发因素", "在十二模型四域上定位故障为行动闸门而非能力/信念/判断缺失", "540 条跨域合成数据微调即可将 commitment 从 54% 降至 0%，并绘制推理格式敏感的失效边界"]
benchmarks: ["Equity unknowability oracle (40 cases)", "Crypto/Sports/Weather transfer sets (3 domains × 24 questions × 3 evidence levels)", "Tense-balanced control set (240 items)"]
---

# 论文速读：Calibrated-Enough-to-Know-Not-Calibrated-to-Act-Fabricated-E

## 一句话总结
LLM 智能体在展示"专业外观"的面板（即使全部数值都是虚构的）后，对本质上不可预测的问题做出方向性承诺的概率从 6.5% 飙升至 54.0%，核心故障在于"行动闸门"而非能力或缺失判断力；通过仅含 540 条骰子/硬币类合成数据的监督微调，可将这一闸门重新训练恢复，但泛化至全新 prompt 结构时脆弱。

## 研究问题与动机
- 已有研究显示"相关外观的上下文"会让智能体不再拒绝不可知问题（Aggarwal, 2026），但未能区分"是信息本身起作用"还是"包装权威感起作用"——两者观测等价。
- 生产部署默认将模型挂载到仪表盘/检索面板后，假设"更多上下文=更可靠"，但对不可约随机性问题该假设会反转。
- 标准校准指标（ECE、平均概率偏差）无法捕获此故障：模型陈述概率几乎不变且均值接近 50%，但由此驱动的行动完全劣于沉默。
- 需要定位故障发生在"知道层面"还是"行动层面"，并测试该闸门是否可通过训练介入来修复。

## 核心贡献（创新点）
- 隔离了展示形式作为触发因素：全虚构面板与真实数据面板在诱导承诺上统计等价（差异 0.8pp，90% CI 落在 ±5pp 等价区间内），首次因果性地分离信息与包装。
- 定位故障为"行动闸门失效"而非能力/信念/判断缺失：模型在可比回答问题上达 98.6–100% 准确率，且 90% 时能正确判断问题不可知，但依然会在有面板时选择 ANSWER。
- 证明闸门可训练且跨域迁移：仅用 540 条骰子/硬币合成数据进行 SFT，使原 40 个案例的-commitment 降至 0.0%，并在三个未见过域（crypto/sports/weather）稳定转移；通过 tense-balanced 控制排除"未来时态回避"的虚假解释。
- 绘制干预失效边界：推理保留格式下闸门稳健（J=+95~+100），移除推理槽的后端格式下多数运行退化为 0 判别或"错误拒绝（tool call）"，一种消融配方甚至以附概率 committing 48/48 不可知项。
- 提出可治理的审计方案：用不可知探针的 commitment rate 作为低风险、可重复的合规度量，匹配可回答臂使其不可被"全拒"策略欺骗。

## 方法详解
- **不可知性预言机**：在短期价格方向、体育赛事胜负、十天降水上进行构造；所有 as-of 日期均在模型训练截止之后（防止记忆）， equity/crypto 域以密封结果评分，sports/weather 无密封结果。
- **证据梯度**（四档）：L0（无面板）→ L1（当前价 +10 日前价）→ L2（完整专业面板：RSI/EMA/MACD/ATR/成交量比/状态分类）→ L2'（同实体面板但归属另一标的，日期重写）。另设 scrambled/scramfull 制造臂以隔离信息 vs 包装。
- **度量定义（预注册）**：Commitment = 选择 ANSWER；Discrimination = Youden's J = TPR − FPR = P(decline|unknowable) − P(decline|answerable)；Earned check = Brier score 对比 0.250 基准；CORP 分解用于 bin-free 校准评估；case-clustered bootstrap 重采样 (asset, date) 事件；Cohen's h 评估效应大小；TOST 90% 等价位。
- **训练配方**：Qwen2.5-3B-Instruct，4-bit QLoRA（nf4, double quantization, rank 32, alpha 64），3 轮，batch 8，grad accumulation 2，lr 2e-4，completion-only loss（prompt 屏蔽）。540 条合成样本含 dice/coins/jars/timers/calendars，其中一半 unknowable 带 rich 但非预测性面板并与可解 case 配对，prompt framings 按层采样保证"工具提供+诱导面板"单元格占 25.5%（随机采样仅 9.0% 时会塌缩）。
- **语法混淆控制（§6.6）**：构造 240 条 tense-balanced 测试集，48 条将来时可回答（确定型定时器计算）、54 条现在时不可知（隐藏状态），检验模型是否学会"避将来时"的浅层启发。

## 实验与结果
- **十二模型 × 四域基线**：在 equity 上 L0 6.5% → L2 54.0%（+48pp，95% CI [+44, +51]）；L2'（不同实体）跌至 3.5%。承诺的 Brier 0.281 > 0.250，劣于盲猜；CORP 分解显示分辨力仅 0.0081（占不确定性的 3.2%），校准误差 0.0391 占主导；AUROC 0.346，方向准确率 35.6%。
- **跨域转移**：crypto 9.4%→22.6%→34.7%（h=0.64）、sports 4.2%→29.2%→36.8%（h=0.89）、weather 9.1%→5.6%→55.7%（h=1.07）；L2 平均 42.5%。
- **每模型跨度 0–100%**：两 OpenAI 模型 100% 承诺（J=+0）；Claude Opus 86.1%、Sonnet 70.8%；Grok 4.3 仅 11.1%（J=+88）；Claude Haiku 4.5 为 0%。效果由单家部分模型承载。
- **训练后 3B 模型**：原 40 个 equity 案例在各证据水平均为 0.0% 承诺，Cohen's h=+1.65；跨域在自然 framing 下 J 达 +62~+100；对可回答臂 99–100% 作答、0% 过度拒绝。
- **tense-balanced 控制**：训练后模型在现在时不可知上 100% 拒绝、在将来时可回答上 95.8–100% 作答，within-tense J 分别为 +97/+100，排除语态启发。
- **失效边界**：移除推理槽的后端格式下主配方七检查点合计仅 5 次 commitment 且均无附带概率；但消融配方 seed 1 以含概率方式 committing 48/48，同时伴随 33.3% 可回答天气项过度拒绝。

## 相关工作脉络
- **AbstentionBench / Abstain-R1 / TruthRL**：聚焦 epistemic 不可答（缺信息/错前提），本文聚焦 aleatoric 不可知（答案存在但先验不可知），且提供训练干预与失效边界。
- **AgentAbstain / Agentic Abstention**：研究工具型 agent 何时停止；AgentAbstain 同样得出"能力×拒绝独立"结论，但起点为可执行沙盒而非不可预测未来事件。
- **ForecastBench（Karger et al., 2024）**：衡量预测质量；本文以"是否作出预测"为主变量，概率仅作辅助证据，揭示其 anti-predictive 特征。
- **Sun et al. (2026)**：从隐状态解码工具需求，结论"模型知道何时该调用工具"与此处行为层面的"知道但不动作"相互印证。
- **Pal et al. (2026)**：静态置信度不能预测交互行为；本文在 aleatoric 切片上以受控实验确认该脱钩，并给出可训练的干预。
- **Xuan et al. (2026) / Kalai et al. (2025)**：检索噪声放大自信、训练奖励猜测；本文在此基础上分离出"展示权威感"本身即为触发。

## 局限性与未来方向
- Weather 是最弱仪器：其面板携带真实集合降水概率，且无密封结果，难以排除"模型正确读概率"的解释。
- 效果集中于单一家族模型（Claude），平均数字会掩盖群体内部三分法（3 中招/4 免疫/3 饱和）。
- 合成不可知样本含显式随机词汇（fair/hidden/covered），残留词法线索未被 §6.6 完全排除。
- 干预仅在 3B 单架构、合成数据、6 主配方 +4 消融独立运行、每单元 24–40 案例上验证，未证明可迁移到更大规模或真实部署。
- Qwen3.7-plus 与 Llama 3.3 70B 分别有 52.8%/23.6% 不可解析输出，J 为下界。
- 未来：多步 cascade 传播研究、另一尺度复现、非"拒绝=安全默认"的域、同一家族内差异机制分析。

## 研究启发与可借鉴点
- **"推理槽"作为鲁棒性杠杆**：确保模型在决策前拥有显式推理输出位可大幅提升闸门稳定性，可与本团队部署 pipeline 的 prompt 模板设计结合。
- **训练数据层采样而非随机采样**：关键高风险组合（工具 framing × 诱导面板）需按比例提升至 25.5%，否则会出现罕见 cell 主导的塌缩；可作为通用训练配比原则。
- **退化策略基线前置**：声明任何度量前先列出五种退化策略（always DECLINE / always ANSWER / 词法启发等）的上界，避免事后 metric 被简单规则超过——本文 T 指标即因此被证伪。
- **tense-balanced 正交对照设计**：对易与语法线索混淆的监督信号，构造"跨时态平衡"测试集是唯一能排除伪正相关的低成本手段。
- **耦合的"可回答臂"防止过度拒绝偏移**：任何拒答能力的评估都必须与匹配可回答臂联动报告，否则无法区分"懂拒绝"与"全拒"。

## 关键术语表
- **Aleatoric unknowability**：答案存在且将揭晓但先验不可预测的不确定性（与"因信息缺失不可答"的 epistemic 不可答相对）。
- **Action gate**：从"知道问题不可知"到"选择不行动"的决策门槛，本文定位的故障点。
- **Youden's J**：灵敏度 − 假阳性率，衡量在不可知/可回答二元分类上的总体区分力；J=0 表示等同于随机。
- **Degenerate strategy baseline**：无需理解即可运行的退化策略（如"遇将来时一律拒绝"），用于标定任何度量不可低于的伪上限。
- **Scrambled panel**：同一标的的历史技术指标替换当前面板的技术字段，与标题/收盘/状态一致自洽，制造"看起来真但信息为空"的对照。
- **Tense-balanced set**：同时包含将来时不可知与现在时可回答（反之亦然）的测试集，用于排除基于语法时态的启发式解释。
- **CORP decomposition**：无分箱的校准分解方法（isotonic regression），将 Brier 得分拆为可靠性/分辨力/不确定性三部分。
- **Over-abstention**：对可回答问题的不当拒绝；本文主要结果中几乎为零，仅在少数消融/极端格式下出现。

## 可复现要素
- **数据集**：12 模型全量生成缓存、四类证据臂、dose-response、所有训练检查点的双 framing 输出、tense-balanced 控制集——均已公开（含分析脚本）。
- **代码/权重**：代码与缓存输出见 github.com/Pranav-1100/confidence-calibration-evaluation；预注册与所有版本在 Zenodo（DOI: 10.5281/zenodo.22043517，v1: 10.5281/zenodo.21325375）。
- **训练超参**：Qwen2.5-3B-Instruct，4-bit QLoRA（nf4, double quantization, rank=32, alpha=64），3 epochs，batch=8，grad accumulation=2，lr=2e-4，greedy decode，256 tokens；数据 540 条合成（±24 体育项消融为 516 条）。
- **API 模型**：12 款前沿模型，温度 0.3（基线）/ greedy（训练模型）；所有 cached generations 已公开，无需重跑 API/GPU 即可复算。
