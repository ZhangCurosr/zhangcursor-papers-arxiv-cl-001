---
title: "Efficient-GUI-Agents-A-Systems-Survey-of-Observation-Memory"
source: https://arxiv.org/pdf/2609.02309v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 22:35:38"
field: "GUI Agent 效率优化"
keywords: ["GUI Agent", "Efficiency", "Systems Survey", "Observation Compression", "Memory Efficiency", "Action Pruning", "Runtime Optimization"]
innovations: ["提出观察/记忆/动作/系统四轴 GUI Agent 效率分类体系", "系统揭示各效率方法的成本转移现象并呼吁全链路诚实记账", "归纳五大跨层效率设计范式（选择性读取、全局到局部视觉、可恢复记忆、验证感知控制、混合运行时）"]
benchmarks: ["OSWorld-Human", "MMBench-GUI", "WebArena", "AndroidWorld", "OSWorld", "BrowserGym"]
---

# 论文速读：Efficient GUI Agents: A Systems Survey of Observation, Memory, Action, and Runtime Optimization

## 一句话总结
本文从端到端系统视角对 GUI Agent 的效率问题进行全面综述，构建覆盖观察、上下文/记忆、动作、规划器与系统四方面的效率分类体系，并批判性指出当前文献中普遍存在"局部节省但成本转移"的现象，呼吁建立统一的效率评估基准。

## 研究问题与动机
- **成功率导向评估的局限**：现有 GUI Agent 研究主要报告任务成功率，但一个最终成功的 Agent 可能因上下文过长、推理过慢、动作冗余而无法实际部署（如 OSWorld-Human 显示领先 Agent 需人类 2.7×–4.3× 的步骤数）。
- **GUI 交互的多模态高开销特性**：Web 端 DOM/AxTree 原始输入可达 800k tokens；截图中心设置面临高分辨率（3k×2k+）视觉密集场景，grounding 计算昂贵且易错。
- **长程任务的成本累积效应**：GUI 任务涉及多步 observe–reason–act–verify 循环，每一步都产生独立开销，且反思（reflection）在端到端延迟中占比高达 76%–96%。
- **效率指标不统一导致跨论文不可比**：多数工作仅报告 token 或步骤的局部节省，parser、verifier、检索器等新增开销未被诚实计入，缺乏共享的定价框架。

## 核心贡献（创新点）
1. **提出四轴效率分类体系**：将 GUI Agent 效率分解为 Observation Efficiency、Context/Memory Efficiency、Action Efficiency、Planner-Side/System Efficiency 四个正交维度，覆盖从感知到执行的全链路。
2. **揭示"成本转移"现象**：系统梳理各方法在节省某一类开销时引入的新开销（如 RegionFocus 减少视觉 token 但增加 66.8% 轨迹开销；R-VLM 区域提议带来 2× 推理延迟），强调需全链路记账。
3. **归纳五大跨层设计范式**：选择性读取替代全量上下文摄入、全局到局部视觉分配、可恢复记忆替代原始历史回放、验证感知控制、GUI/非 GUI 混合运行时——为后续研究提供结构化设计空间。
4. **呼吁标准化效率基准**：批评当前 benchmark 仅报告成功率，主张引入峰值 GPU 显存、prefill/decode 延迟、每 decoded token MFLOPs、success-normalized GPU 成本等统一指标。

## 方法详解
论文按四轴组织，各轴关键方法如下：

**① Observation Efficiency（观察效率）**
- **文本精简**：Agent-E 主张层级化 DOM 蒸馏降噪；Beyond Pixels 研究激进压缩 DOM 至 ~10³ tokens；LineRetriever 以规划感知检索替代截断，实现 61%–73% 观察缩减；FocusAgent 按步骤选择性检索 AxTree 行，削减 >50%–80%；Prune4Web 将剪枝转为可执行过滤程序，500+ 候选元素缩至 <20。
- **区域聚焦视觉**：Ferret-UI/R-VLM 采用子图分区与区域提议；RegionFocus/DiMo-GUI 推理时自适应渐进缩放（DiMo-GUI 最多 7 轮），但以额外轨迹开销为代价；ShowUI 在 token 级别剔除 33% 冗余视觉 token，获 1.4× 训练加速；SimpAgent 将历史图像压缩为 64 tokens，LLM 分支 FLOP 减少 27%。
- **解析增强与混合**：OmniParser/Tree-of-Lens 将截图转为半结构化区域/布局表示；GUI-Actor 引入坐标无关 action head（2B 模型 +20M 参数），单前向传播生成候选；UGround 使用约 2/3 固定分辨率 visual tokens；Aguvis/UI-TARS 纯视觉路径将每步输入从 4k–6k tokens 降至 1,196 tokens（-70%）。

**② Context & Memory Efficiency（上下文/记忆效率）**
- **摘要压缩与选择性回溯**：Agent-S/ColorBrowserAgent/GUI-Rise 用紧凑进度摘要替代原始轨迹回放；PAL-UI 结合双层摘要与主动 look-back；HiconAgent 通过动态上下文采样实现 25.21T FLOPs（vs. 35.75T  uncompressed 3B / 62.31T 7B），约 60% FLOP 缩减。
- **运行期表示压缩**：GUI-KV 针对 GUI 时空冗余定制 KV 压缩策略，5 张截图下 MFLOPs/token 降 38.9%，缓存预算 5%–20%；ST-Lite 免训练 KV 压缩，解码加速 2.45×，端到端加速 1.40×；Continuous Memory 将超长轨迹压缩为 8 个固定长度 embedding；SecAgent 将历史截图/动作蒸馏为紧凑语义上下文。

**③ Action Efficiency（动作效率）**
- **动作抽象**：SkillWeaver 发现可复用网站流程并蒸馏为 callable API；PolySkill 将技能抽象目标与站点实现解耦，典型函数覆盖 2–5 GUI 步；ActionEngine 将高频交互编译为状态机程序，成本 $0.71→$0.06（11.83×），延迟 237.5→118.3s，输入 token 62.3k→8.1k，模型调用 10.2→1.8；CoAct-1 允许在 GUI 操作与代码执行间动态路由，平均步数降至 10.15（vs. GTA-1 的 15.22、UI-TARS 的 14.90）。
- **剪枝/验证/恢复**：Prune4Web 早期剪枝 DOM 候选；V-Droid 执行前评分验证，0.7s/决策、4.3s/步（典型 Mobile Agent >20s/步）；VeriSafe Agent 基于逻辑验证防止不可逆错误；SenseAct 引入类型化承诺与后验检查，UI 暴露减少 65.51%；BacktrackAgent/LongHorizonUI 显式支持回溯与回滚。
- **探索控制**：LASER 以状态空间搜索建模 Web 交互；Auto-Intent 从演示中蒸馏意图引导探索；WebOperator 结合预执行过滤、best-first search 与安全回溯；MobileUse 将主动探索设为冷启动触发而非常驻行为。

**④ Planner-Side & System Efficiency（规划器与系统效率）**
- **自适应推理深度**：Think Twice, Click Once 按任务难度选择快/慢推理模式（慢思考将处理时间从 2.6s 升至 5.4s）；GUI-G1 发现更长 CoT 反而有害；AdaGUI-R1 减少 40% 冗余推理 token 但硬例调度增加 23.5% FLOPs。
- **分层/模块化规划**：MobileUse 按需触发层级反思，将反思开销降至 10%；Agent S2 将认知角色分配给通用/专家模块；InfiGUIAgent 在紧凑模型内原生集成推理与反思。
- **系统级路由**：CORE 本地/云端模型协作降低 UI 暴露 55.60%/34.96%；GUIGuard 加入隐私识别/保护阶段；IntentCUA 通过意图级抽象与共享计划记忆减少冗余重规划。

核心系统公式（Appendix A）：
$$a_t \sim \pi(g, o_t, m_t)$$
其中 $o_t = \Omega(s_t)$ 为观察模块，$m_t = U_m(m_{t-1}, o_t, a_{t-1}, v_{t-1})$ 为记忆更新，$v_t = V(g, s_t, o_t, a_t, s_{t+1})$ 为验证信号，四轴优化即是对上述链路上各环节的代价最小化。

## 实验与结果
本文为综述论文，无原创实验，但综合引用了各工作的关键效率数字：

| 方法 | 关键效率信号 |
|---|---|
| **ActionEngine** | 成本 $0.71→$0.06（11.83×），延迟 237.5→118.3s，输入 62.3k→8.1k tokens，调用 10.2→1.8 |
| **V-Droid** | 输入 2.6k–8.9k tokens，0.7s/决策，4.3s/步（典型 Mobile Agent >20s/步） |
| **HiconAgent** | FLOP 25.21T vs. 35.75T（3B uncompressed）/ 62.31T（7B），~60% FLOP 缩减 |
| **GUI-KV** | 5 张截图下 MFLOPs/token 降 38.9%，缓存预算 5%–20% |
| **ST-Lite** | 解码加速 2.45×，端到端加速 1.40× |
| **CoAct-1** | 平均步数 10.15 vs. GTA-1（15.22）、UI-TARS（14.90） |
| **SenseAct** | UI 暴露减少 65.51% |
| **SimpAgent** | LLM 分支 FLOP 减少 27% |
| **AdaGUI-R1** | 冗余推理 token 减少 40%，但 FLOP 增 23.5% |
| **CORE** | UI 暴露降 55.60%/34.96%，延迟 1.52–1.66× baseline |
| **OSWorld-Human** | Agent 需人类 2.7×–4.3× 步骤，reflection 占 76%–96% 任务延迟 |
| **MMBench-GUI** | 50 步预算下冗余步成本 EQ2=7–8，隐私噪声 40%，专家成本 16% |

**最强结果**：ActionEngine 在多项效率指标上表现最为显著（11.83× 成本降低、7.6× 延迟降低、7.7× token 缩减）；CoAct-1 在步数效率上领先基线约 33%。

## 相关工作脉络
1. **Yang et al. (2026b) "Toward Efficient Agents"**：聚焦 Agent 整体的记忆、工具学习与规划效率；本文与其区别在于专门针对 GUI Agent 的四轴系统分类，涵盖观察与 grounding 层面的效率。
2. **Sager et al. (2025) / Nguyen et al. (2024)**：通用 GUI Agent/Computer-Use Agent 综述；本文独特价值在于以"效率"为第一性原理重新组织文献，而非按平台或能力分类。
3. **Jin et al. (2024) "Efficient Multimodal LLMs"**：关注多模态 LLM 的训练/推理效率；本文将其思路延伸至 GUI Agent 的端到端运行链路，强调多模态开销在逐步骤 interleaved 执行中的放大效应。
4. **Agent-S (Agashe et al., 2024) / UI-TARS (Qin et al., 2025)**：代表性高性能 GUI Agent；本文将其纳入效率分析框架，指出其各自在内存/token 侧的优化贡献与未报告开销。
5. **OSWorld-Human (Abhyankar et al., 2025) / MMBench-GUI (Wang et al., 2025a)**：效率感知基准；本文引用其 profiling 结果作为动机支撑，并呼吁更全面的效率指标体系。
6. **GUI-KV (Huang et al., 2025) / ST-Lite (Zhou et al., 2026a)**：KV-cache 压缩专用方法；本文首次将它们置于 GUI Agent 系统的长程交互语境下进行横向对比。

## 局限性与未来方向
- **效率账目不够诚实**：大量工作仅报告局部节省（如 token 减少），但未分离 parser、verifier、检索器、编排器的新增开销，跨论文对比困难。
- **GUI Agent 效率基准稀缺**：现有 benchmark 主要报告成功率，缺少峰值 GPU 显存、prefill/decode 延迟、MFLOPs/token、success-normalized 成本等标准化指标。
- **验证器成本未独立核算**：verifier call 的频率、延迟与费用在多篇工作中被隐含在总延迟中，缺乏独立计价框架。
- **跨层协同设计不足**：观察、记忆、执行层的优化多为孤立设计，缺乏联合优化真实延迟与隐私约束的 co-design 研究。
- **隐私-效率权衡待探索**：CORE/GUIGuard 等初步探索了隐私保护与效率的 tradeoff，但尚缺乏系统化研究。

## 研究启发与可借鉴点
1. **四轴分类法可直接迁移**：观察/记忆/动作/系统效率的正交分解为其他 Agent 类型（如 tool-using agent、robotics agent）的效率分析提供了可复用的分析框架。
2. **"成本转移"视角极具价值**：在设计和评估任何效率优化方法时，必须追踪被节省的开销是否转移到了其他模块（如剪枝引入的 parser 延迟、验证引入的额外 call），这应成为审稿/评估的标准动作。
3. **混合运行时（GUI + Code）是明确趋势**：CoAct-1 证明不同子任务可在 GUI 操作与程序执行间动态路由以显著降本，可与本团队方向结合探索跨模态路由策略。
4. **标准化效率指标倡议值得跟进**：若团队开发新 GUI Agent，建议主动报告 MFLOPs/token、TTFT、TPOT、per-step latency、success-normalized cost 等指标，以契合领域演进方向。
5. **按需自适应推理的深度值得探索**：Think Twice Click Once 和 AdaGUI-R1 的"快/慢模式"切换思路可扩展到多粒度推理调度，尤其适合本团队关注的长程复杂任务场景。

## 关键术语表
**DOM (Document Object Model)**：网页的树形结构表示，包含所有 HTML 元素及其属性，是 Web Agent 主要的结构化观察来源之一。
**AxTree (Accessibility Tree)**：无障碍树，由屏幕阅读器提取的界面结构化信息，比 DOM 更紧凑但常含噪声或与渲染结果不对齐。
**KV-cache**：自回归模型解码时缓存的键值张量，长程 GUI 交互的主要显存瓶颈，5 张截图即可超 80GB GPU 显存。
**TTFT / TPOT**：Time to First Token（首 token 延迟）/ Tokens Per Output Token（每输出 token 平均生成时间），衡量推理响应速度的关键指标。
**MFLOPs**：Million Floating Point Operations，百万次浮点运算，用于量化单步推理的计算量。
**RFT (Reinforcement Fine-Tuning)**：强化微调，UI-R1 使用仅 136 个样本在 8×RTX 4090 上训练约 8 小时提升 GUI 动作预测效率。
**Grounding**：将自然语言指令或决策映射到具体界面元素（坐标/DOM 节点/ AxTree 项）的过程，是 GUI Agent 的核心瓶颈。
**End-to-end Latency**：从任务开始到成功或失败的总耗时，包含观察、推理、动作执行、验证各环节之和。

## 可复现要素
- **数据集**：无原创数据集；综合引用 Mind2Web、WebArena、VisualWebArena、BrowserGym、AndroidWorld、OSWorld、Windows Agent Arena、WorkArena 等公开基准。
- **代码**：本文无原创代码；各子领域引用工作的代码可见性见附录 Table 2–5，部分开源（GitHub 标记），多数未公开。
- **关键超参**：论文未提出统一超参，各方法超参分散于原文；综述层面未提供可直接复现的系统配置。
