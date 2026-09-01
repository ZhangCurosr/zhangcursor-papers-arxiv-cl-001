---
title: "Auditing-Exposure-to-Harmful-Content-on-TikTok-using-Multimo"
source: https://arxiv.org/pdf/2608.17583v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:47:12"
field: "平台算法审计与内容安全"
keywords: ["TikTok auditing", "multimodal LLM", "harmful content detection", "cross-national study", "age-stratified sockpuppet", "content moderation", "algorithmic bias"]
innovations: ["两阶段 MLLM 验证+跨国分层审计框架，证实 E3 八帧配置以一半成本达到 fair-to-moderate κ", "scroll-pre→SEARCH→scroll-post 同账号循环设计，分离推荐放大与搜索检索两种暴露路径", "首次系统量化三国×四年龄层 TikTok 有害内容暴露差异，发现意大利最高且搜索阶段年龄梯度消失"]
benchmarks: ["Cohen's kappa on binary harm verdict", "Harm prevalence rates per country-age-phase cell"]
---

# 论文速读：Auditing-Exposure-to-Harmful-Content-on-TikTok-using-Multimodal

## 一句话总结
本文利用多模态大语言模型（MLLM）作为自动标注器，在法国、意大利、瑞典三国对 TikTok 进行了跨国、分年龄层的有害内容暴露审计，发现主动搜索关键词可使有害内容暴露率飙升 1.5–7.5 倍，且意大利在所有年龄层均面临最高的被动有害内容暴露（意大利 19 岁人设达 48.6%）。

## 研究问题与动机
- **平台算法对青少年的潜在危害缺乏独立、可复现的审计**：TikTok 的 For-You 页（FYP）通过微交互快速适应推荐，已有调查指出自残、进食障碍等内容可在数小时内触达新注册的青少年账号。
- **现有审计方法的成本瓶颈**：视频标注成本高，且不同语言/国家的审核标准不一致，难以进行大规模跨国比较。
- **MLLM 作为自动标注器的潜力与风险未明**：虽然前沿 MLLM 能在仇恨言论评估上与人类对齐，但不同模型之间分歧巨大，且可能利用语言先验而非真正分析视频；它们如何与年龄、国家、输入模态等维度交互，尚属未知。
- **监管紧迫性**：欧盟《数字服务法》（DSA）等法规使得对短视频平台的独立、可复现审计日益迫切。

## 核心贡献（创新点）
- **跨国、跨年龄层的 TikTok 有害内容暴露实证图谱**：在法国、意大利、瑞典三国、四个年龄层（13/16/19/40 岁）的系统性审计，揭示了被动浏览与主动搜索下的有害内容差异模式。
- **经真人标注验证的 MLLM 自动标注流水线**：以 Gemini 2.5 Flash（E3 配置，8 帧+文本）为核心标注器，在 300 视频参考集上验证其可靠性后，以约 50 美元 API 成本完成全量标注，为大规模跨境审计提供可复用范式。
- **公开一套完整的审计 artifacts**：包含 13 类有害内容分类体系（对齐 TikTok 社区准则）、标注 schema、各国关键词列表及 36,971 视频 corpus 元数据，促进后续研究复现与扩展。

## 方法详解
- **两阶段审计设计**：
  - **Stage 1**：在 300 视频参考集上对比四种 MLLM（Gemini 2.5 Flash、Qwen3-VL-32B、GPT-4o-mini、Mistral Large 3）及三种输入条件（E1: 纯文本；E2: 原生视频；E3: 8 帧+文本），选出与母语者标注最一致的模型配置。
  - **Stage 2**：将 Stage 1 胜出的 Gemini E3 应用于全量 36,971 视频的 10% 分层样本（按国家、年龄、阶段分层）。
- **数据采集协议**：
  - 三国 × 四年龄层 × 三独立 sockpuppet 账号 = 36 个实验单元。
  - 被动阶段：纯 FYP 滚动，收集 14,093 个视频（2025.12.30–2026.1.11）。
  - 主动阶段：每个账号执行 scroll-pre → SEARCH（21 个有害关键词）→ scroll-post 循环，收集 22,878 个唯一视频（2026.4.13–4.19）。
  - 账号通过 VPN 模拟目标国家 IP，使用 Tampermonkey 脚本捕获 API 响应并模拟随机滚动。
- **有害内容分类体系**：13 类分类（Disordered Eating、Self-harm、Dangerous Challenges、Nudity、Sexually Suggestive、Shocking/Graphic、Hate Speech、Sexual Abuse、Trafficking、Gambling、Alcohol/Tobacco/Drugs、Integrity、Harassment），对齐 TikTok 社区准则。
- **MLLM 标注 prompt 设计**：系统 prompt 注入 13 类分类及其定义，要求模型输出结构化 JSON（verdict: harmful/not harmful, subcategory, reasoning）；遵循 Wu et al. (2025) 的政策/规则 grounding 策略以提升对齐度。
- **E3 八帧采样协议**：均匀间隔抽取 8 帧作为 base64 图像内联提交，灵感来自 Jo et al. (2024) 的十四帧协议，但以更低的 token 成本实现。

## 实验与结果
- **数据集规模**：36,971 个唯一视频（被动 14,093 + 主动 22,878），覆盖法/意/瑞三国 × 13/16/19/40 岁四年龄层。
- **Stage 1 MLLM 验证结果**：
  - Gemini 2.5 Flash E3 最优：聚合 Cohen's κ = 0.42（fair-to-moderate），成本约为 E2 原生视频上传的一半。
  - 意大利内容达到 moderate agreement（κ = 0.46），法国最低（κ = 0.29）。
  - 非 Gemini 配置均低于 κ = 0.30；纯文本 E1 表现最差（Gemini E1 κ = 0.17）。
  - E2 与 E3 在 5,108 对视频上 binary verdict 一致度 κ = 0.614，但 E2 报告的危害率系统性高 3–12 个百分点。
- **Stage 2 主要发现**：
  - **被动浏览（FYP）**：意大利在所有年龄层危害率最高；意大利 19 岁达 48.6%（全审计最高值）；法国 ~23% 稳定；瑞典从 24%（13 岁）升至 38%（19 岁）。
  - **主动搜索**：SEARCH 端点返回 35–56% 有害内容，较 scroll-pre 提升 1.5–7.5 倍（12 组合中有 10 组合显著提升）；scroll-post 迅速回退至基线水平。
  - **年龄差异扁平化**：搜索阶段各年龄层危害率收敛至 35–56% 区间，原有 scroll-pre 年龄梯度消失。
  - **跨国家异质性**：意大利的领先主要由 Sexually Suggestive 内容驱动（贡献 23.8 pp vs. 法国 12.2 pp）；意大利专属内容（非跨境流通）危害率达 41.7%，而共享内容各国无显著差异。
  - ** Provider 拒绝**：Gemini 安全层拒绝约 1.1% 输入，非随机分布——Nudity 和 Sexually Suggestive 类别被拒率最高（≈4.8% 和 ≈2.6%），导致最明确危害类别的 prevalences 被低估。

## 相关工作脉络
- **TikTok 审计与人设实验**：Sandvig et al. (2014) 奠定 sockpuppet 审计方法论；Xue et al. (2025) 和 Eltaher et al. (2025) 为最近的年龄分层 TikTok 审计，本文与其定位差异在于引入 MLLM 自动标注以支持跨国扩展，并首次系统比较输入模态（文本/帧/原生视频）的价值。
- **MLLM 作为内容审核器**：Davidson et al. (2025) 证明前沿 MLLM 能与人类仇恨言论判断对齐；Fasching & Lelkes (2025) 揭示不同 MLLM 对同一内容分歧巨大——本文 Stage 1 正是为了解决这一不确定性而比较多种配置；Apple ML Research (2025) 指出视频 LLM 可能依赖语言先验而非真正视觉推理，本文 E1 基线即用于检测此 failure mode。
- **跨语言审核与帧采样**：Tonneau et al. (2025) 分析 DSA 透明度报告中各语言审核员分配，三国选取据此；Jo et al. (2024) 的十四帧协议直接启发本文 E3 八帧设计。
- **政策/规则 grounding 的 MLLM 提示**：Wu et al. (2025) 证明将社区准则注入 prompt 可显著提升 MLLM 审核对齐度，本文 13 类分类注入即采用此策略。
- **算法放大机制建模**：Boeker & Urman (2022) 识别 FYP 个性化因素；Baumann et al. (2025) 建模 engagement vectors 如何指数放大 niche 内容——本文强调搜索端点而非推荐放大是有害内容暴露的主导路径。

## 局限性与未来方向
- **参考标注非统计 ground truth**：每国仅两名母语者 annotator 经联合调解产生最终标签，与专业内容审核员的内部阈值存在差距。
- **Stage 2 未做每视频重新标注**：仅依赖 Stage 1 验证的单一 MLLM 配置，E2/E3 跨模态一致度仅作内部检查，未验证向全量分布的迁移误差。
- **Provider 拒绝偏差**：1.1% 拒绝率非随机，Nudity/Suggestive 类别被过度过滤，导致这些类别 prevalences 系统性低估。
- **时间窗口单一**：数据采集限于 2025 年底至 2026 年初，TikTok 推荐系统持续演化，结论未必泛化到其他时段。
- **sockpuppet 账号的生态效度限制**：程序化控制账号缺乏真实用户的社交图谱、多设备行为、跨会话连续性，发现应视为"算法在简单交互规则下能服务的内容上界"而非真实用户暴露的点估计。
- **VPN 路由账户可能触发反滥用层**：未单独测量 native-IP 流量的对比效应。

## 研究启发与可借鉴点
- **两阶段 MLLM 验证+应用范式**：先在小型人工标注集上系统比较多模型×多模态配置，再选定最优配置扩展至全量，可作为其他平台审计的标准流程参考。
- **E3 八帧协议的成本-质量权衡**：在保持较高 κ 的同时将 API 成本降至原生视频上传的一半，为大规模多模态内容审计提供了高效可行的输入方案。
- **scroll-pre → SEARCH → scroll-post 循环设计**：在同一账号内捕获基线暴露、主动搜索暴露及后续回流，可干净分离推荐放大与搜索检索两种暴露路径，方法论上优于仅比较独立样本的设计。
- **跨国家分层选取的策略依据**：基于 DSA 透明度报告的语言审核员分配数据选取高低中三种代表国家，使跨国比较具有可解释的政策含义。
- **团队可结合方向**：可将此审计框架扩展至其他短视频平台（YouTube Shorts、Instagram Reels）或引入更多年龄分层/地区，以验证本研究的意大利主导现象是否具有平台普适性。

## 关键术语表
- **Sockpuppet account**：用于算法审计的程序化人设账号，模拟特定国家、年龄的用户行为以测量平台推荐偏差。
- **For-You page (FYP)**：TikTok 的核心推荐信息流，基于 watch time、loop rate 等微交互信号实时适配内容。
- **Cohen's κ**：衡量标注者间/模型与标注者间一致度的统计指标，排除随机命中概率，κ ∈ [0.40, 0.60) 为 fair，[0.60, 0.80) 为 moderate。
- **E1/E2/E3 输入条件**：E1 纯文本（caption+transcript）；E2 原生 MP4 视频上传；E3 均匀采样 8 帧图像+文本。
- **Harm taxonomy**：本文采用的 13 类有害内容分类体系，对齐 TikTok Community Guidelines，涵盖从 disordered eating 到 hate speech 的细粒度类别。
- **Cross-national moderation gap**：不同语言/国家的审核资源配置差异，本文三国 moderator 分配比约为 620:396:98（法:意:瑞）。
- **Phase-stratified sampling**：按被动/主动阶段分层抽样，确保各子群体有足够样本量以支持统计推断。

## 可复现要素
- **数据集**：36,971 视频 corpus（IDs 公开），含 per-video 元数据（国家、年龄层、阶段、关键词、时间戳、公开 engagement 计数）；Stage-1 标注以 aggregate 形式发布；代码（scrapers、MLLM 评估脚本、绘图与统计 pipeline）已开源。原始视频内容不重新分发。
- **代码/权重**：论文声明已开源（具体仓库见论文主页）。
- **关键超参**：八帧采样（E3）；prompt 注入 13 类分类及定义；API 总花费约 50 美元（Gemini 2.5 Flash）；被动阶段 5 天/国家，主动阶段 5 天/国家（3 工作日 +2 周末日）；每组合 3 独立账号并行。
