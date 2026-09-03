---
title: "GGSS-Geodesic-Gated-Spherical-Steering-for-Inference-Time-De"
source: https://arxiv.org/pdf/2608.25375v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:33:44"
field: "多模态模型公平性与去偏"
keywords: ["Vision-Language Models", "Inference-time Debiasing", "Generative VLM", "Spherical Steering", "Activation Intervention", "Fairness"]
innovations: ["提出 GGSS：基于反事实球面子空间发现、自适应 token 级门控和范数守恒测地 Steering 的推理时去偏框架", "构建首个面向生成式 VLM 的十基线统一推理时去偏评估基准并开源", "证明球面 Slerp 在大尺度生成式 VLM 上显著优于 Euclidean 硬投影，且自适应门控是可扩展选择性的关键"]
benchmarks: ["MMStar", "SocialCounterfactuals", "REFLECT/FOCUS"]
---

# 论文速读：GGSS-Geodesic-Gated-Spherical-Steering-for-Inference-Time-De

## 一句话总结
GGSS 提出了一种面向生成式视觉-语言模型（VLM）的推理时去偏框架，通过在单位超球面上发现反事实偏差子空间、沿测地线弧 Steering 视觉 token、并引入自适应门控机制，在冻结模型权重的前提下实现低损耗的群体偏差缓解。

## 研究问题与动机
- **生成式 VLM 存在系统性人口统计学偏差**：即使控制图像内容，模型对种族、性别等属性的输出仍呈现显著差异，而训练时去偏成本高、灵活性差。
- **既有推理时去偏方法不适配生成式 VLM**：INLP、R-LACE、LEACE、BendVLM 等方法针对 CLIP 类静态全局嵌入设计，直接移植到生成式 VLM 的多 token 视觉表示上会过度破坏任务能力，甚至被未 Steering 的 baseline 压制。
- **三大技术挑战**：①几何保持——欧氏减法扭曲激活的方向与范数，造成严重的质量下降；② token 级偏差异质性——人口统计信号集中在少量 token（如面部、衣着），硬投影会误伤无关 token；③单方向刚性——种族等属性具有多类结构，需用低维子空间而非单一方向表示。

## 核心贡献（创新点）
- **提出 GGSS 范数保持的测地 Steering 原语**：用球面线性插值（Slerp）替代硬零空间投影，精确保持 token 范数，避免 Euclidean 操作在大模型尺度下的过 Steering 失效。
- **引入自适应 token 级门控**：基于反事实发现集的偏差范数分布校准每个 token 的 Steering 强度，低偏差 token 几乎不动、高偏差 token 强修正，无需额外训练。
- **构建统一的生成式 VLM 推理时去偏基准**：将 INLP、MeanDiff、BendVLM、LEACE 等十种 CLIP/LM 时代方法适配到同一 hook 层，填补了该方向缺乏公平对比的空白。
- **在四个生成式 VLM 上实现最低平均偏差且保持 MMStar 能力**：GGSS 在所有四个 backbone 上均取得最低 Avg Δ%，显著降低三个 backbone，MMStar 精度保持在 ±0.6 p.p. 内。

## 方法详解
- **两阶段框架**：离线发现阶段从反事实图像集估计偏差子空间 $\mathbf{V}_{\text{bias}}$、全局参考点 $\mu$ 和门控校准统计量 $(\tilde{b}, \sigma_b)$；在线推理阶段对这些发现量复用，对每个视觉 token 执行 Steering。
- **反事实偏差子空间发现**：使用 REFLECT/FOCUS 数据集（480 张人脸反事实图像，固定身份/职业/姿态，变化种族和性别），将视觉 token 均值池化后投影到单位超球面 $\mathbb{S}^{D-1}$，计算每组内的 Fréchet 均值 $\mathbf{c}_u$，再通过对数映射得到切空间偏移 $\mathbf{s}_{u,a} = \log_{\mathbf{c}_u}(\mathbf{y}_{u,a})$，堆叠后做 SVD：$\mathbf{S} = \mathbf{U}\Sigma\mathbf{V}^\top$，取前 $k=|\mathcal{A}|-1$ 个右奇异向量构成 $\mathbf{V}_{\text{bias}}$。
- **token 级测地 Steering**：对每个 token $\mathbf{h}_i$，记录范数 $r_i$，归一化后映射到参考点 $\mu$ 的切空间 $\mathbf{t}_i = \log_\mu(\hat{\mathbf{h}}_i)$，正交投影分离偏差分量 $\mathbf{p}_i = \mathbf{V}_{\text{bias}}\mathbf{V}_{\text{bias}}^\top\mathbf{t}_i$ 和干净分量 $\mathbf{t}_i^{\text{clean}} = (I - \mathbf{V}_{\text{bias}}\mathbf{V}_{\text{bias}}^\top)\mathbf{t}_i$，通过指数映射得到目标方向 $\hat{\mathbf{h}}_i^{\text{target}} = \exp_\mu(\mathbf{t}_i^{\text{clean}})$，再用 Slerp 插值：$\hat{\mathbf{h}}_i^{\text{steered}} = \text{Slerp}(\hat{\mathbf{h}}_i, \hat{\mathbf{h}}_i^{\text{target}}; \beta_i)$，最后恢复范数 $\mathbf{h}_i^{\text{steered}} = r_i \hat{\mathbf{h}}_i^{\text{steered}}$。
- **自适应门控**：$z_i = (\|\mathbf{V}_{\text{bias}}^\top\mathbf{t}_i\|_2 - \tilde{b}) / (\sigma_b + \varepsilon)$，$g_i = g_{\text{floor}} + (1-g_{\text{floor}})\text{sigmoid}(\kappa z_i)$，其中 $\tilde{b}$ 和 $\sigma_b$ 为发现集偏差范数的中位数和标准差，$\kappa$ 控制门控锐度，$g_{\text{floor}}$ 设最小 Steering 强度，默认 $\kappa=5, g_{\text{floor}}=0.3$。
- **数学保证**：Proposition 1 证明 $\mathbf{t}_i^{\text{clean}}$ 是切空间中偏差坐标为零的最小欧氏距离投影，且 Slerp 更新后 $\|\mathbf{h}_i^{\text{steered}}\|_2 = \|\mathbf{h}_i\|_2$ 精确成立。

## 实验与结果
- **模型**：Pixtral-12B、LLaVA-1.6-Vicuna-7B、LLaVA-1.6-Mistral-7B、Qwen3-VL-4B，均在同一 late vision-to-language projection 层 hook。
- **数据集与基准**：反事实发现使用 REFLECT/FOCUS（480 张，6 职业×8 身份×5 种族×2 性别）；评估使用 SocialCounterfactuals 探针套件和 MMStar（1500 题多模态能力基准）。
- **偏差度量**：MCQ（种族，Jensen-Shannon 散度×10³）、2AFC（种族配对，bias std）、Nurse/Doctor（性别分类 gap）。
- **基线**：十个适配的推理时去偏方法（INLP Eucl./sph.、MeanDiff-SVM/LR Eucl./sph.、BendVLM Eucl./sph.、LEACE + 门控）加提示词 fairness instruction。
- **主要结果**：GGSS 在所有四个模型上取得最低 Avg Δ%：Pixtral-12B −55%、Vicuna-7B −90%、Mistral-7B −80%、Qwen3-VL-4B −60%；单任务最高降幅达 −96%（Nurse/Doctor, Vicuna）。MMStar 精度保持在 baseline ±0.6 p.p. 内，统计不显著差异（McNemar $p \geq 0.09$）。
- **统计显著性**：配对符号翻转置换检验显示 GGSS 在三个 backbone 上显著优于 unsteered baseline（LLaVA-Vicuna 的 Nurse/Doctor 单项 $p < 10^{-4}$）；而最强基线 INLP(sph.) 在 Pixtral 上显著损害 MMStar（−6.1 pp, $p < 10^{-4}$）。
- **交叉验证**：held-out α 协议下 GGSS 在全部八个 fold 上均降低偏差，α 选择与论文一致的比例达 6/8。

## 相关工作脉络
- **INLP / R-LACE / LEACE**：经典线性概念擦除方法，针对静态词嵌入或 BERT 类编码器设计，本质是硬零空间投影；GGSS 将其适配到生成式 VLM 的同一 hook 层作为基线，证明 Euclidean 硬投影在大尺度模型上严重破坏能力，而球面变体显著改善但仍不及 GGSS 的门控选择性。
- **BendVLM**：CLIP 风格的检索式平等化方法，依赖共享文本-图像嵌入空间；移植到生成式 VLM 隐藏激活后 Eucl. 版尚可，sph. 版在 Pixtral 和 Vicuna 上导致生成崩溃（72/80 不可解析），凸显 CLIP-era 方法直接迁移的局限。
- **Activation Steering（ActAdd / Representation Engineering）**：直接在 hidden states 上做 Euclidean 加法或投影；GGSS 保留 inference-time 干预精神，但用 Slerp 测地插值替代加性操作，保证范数守恒。
- **训练时 VLM 去偏**：包括因果调整、bias-corpus disentanglement、LLM-guided projection 等；GGSS 定位在无重训练、冻结 checkpoint、可配置偏置概念的推理时方案。
- **CLIP/VLM 社会偏差测量**：REFLECT/FOCUS、SocialCounterfactuals、VLRbiasBench 等工作建立了对 VLM 群体偏差的量化协议；GGSS 在此基础上提供了一套统一评估 protocol（best-avg-α），使不同方法可在相同 operating point 下比较。

## 局限性与未来方向
- **评估范围有限**：仅覆盖四种生成式 VLM、两类人口属性（感知种族与性别）和 MMStar 能力，未验证到其他架构、语言、领域或多类交叉属性的泛化性。
- **仍需调参 α**：最优 steering strength 依赖模型和数据集，论文采用 held-out 搜索，尚无 tuning-free 的自动设定规则（可探索用 $\tilde{b}, \sigma_b$ 直接推导 token-level 强度）。
- **不消除参数中的偏差知识**：仅干预推理时的激活几何，未从模型参数中去除已编码的偏见，分布外场景下的鲁棒性未验证。
- **双重用途风险**：同一 steering 子空间既可削弱也可放大群体敏感度，高风险部署需结合人工审核与持续审计。

## 研究启发与可借鉴点
- **球面几何作为范数守恒的 Steering 原语**：将激活投影到切空间、做正交分解后再用 Slerp 返回球面，这一模式可迁移到其他需要保持表征范数的 intervention 场景（如对抗鲁棒性、概念抑制）。
- **门控选择性思想的通用性**：基于发现集统计量（中位数/标准差）校准 per-token 干预强度，避免了 uniform projection 的过修正；该思想可直接复用到其他 activation intervention 方法中提升选择性。
- **统一基准构建策略**：将多个异构基线适配到同一 hook 层和同一 best-avg-α 协议，消除了层位置和超参选择带来的比较偏差，这一 protocol 设计值得在 fairness intervention 研究中推广。
- **与团队方向的结合机会**：若团队关注多模态模型的公平性审计或 runtime intervention，GGSS 的球面发现流程（Fréchet 均值+SVD on tangent shifts）可作为子模块集成；其反事实发现集构造方式也可迁移到其他属性的子空间学习。

## 关键术语表
- **GGSS（Geodesic-Gated Spherical Steering）**：一种推理时去偏框架，通过球面反事实发现、自适应门控和测地 Steering 在保持 token 范数的前提下消除生成式 VLM 的群体偏差。
- **Fréchet 均值**：定义在黎曼流形（此处为单位超球面）上的均值，通过迭代最小化各点到均值的测地距离平方和得到，用于计算反事实组的球面中心。
- **Slerp（Spherical Linear Interpolation）**：在球面上沿测地线进行线性插值的运算，保证插值结果始终在单位球面上，用于范数保持的 token 方向更新。
- **反事实发现集（Counterfactual Discovery Set）**：保持除受保护属性外所有视觉因素不变的人造图像对，用于估计偏差子空间，此处来自 REFLECT/FOCUS 数据集。
- **best-avg-α 协议**：对每个 (方法, 模型) 对在一个固定的 α 候选集中选择一个值，使其在所有非零基线偏差任务上的平均相对变化最小，用于公平比较不同方法的 operating point。
- **Jensen-Shannon 散度（JSD）**：用于度量 MCQ 任务中不同种族群体的答案分布差异，作为连续型偏差指标，值越小表示偏差越低。
- **切空间投影（Tangent-space Projection）**：将球面上的点通过对数映射到参考点的切空间后，在此欧氏空间中进行线性正交投影，再通过指数映射返回球面，实现几何感知的偏差去除。

## 可复现要素
- **数据集**：REFLECT/FOCUS（480 张反事实人脸图像，GPL-3.0 许可）、SocialCounterfactuals（MIT 许可）、MMStar（通过 VLMEvalKit 评估，Apache-2.0）；代码与 steering checkpoint 已开源：https://github.com/dukesun99/GGSS。
- **模型权重**：使用公开 checkpoint（Pixtral-12B、LLaVA-1.6-Vicuna-7B、LLaVA-1.6-Mistral-7B、Qwen3-VL-4B-Instruct），论文未重新分发。
- **关键超参**：$\alpha \in \{0.25, 0.5, 0.75, 1.0, 1.5\}$（每模型通过 best-avg-α 协议选定）、$\kappa = 5$（门控锐度）、$g_{\text{floor}} = 0.3$（门控下界）、$k = |\mathcal{A}|-1 = 4$（子空间维数）；发现 prompt 为 "Describe this image in detail."；hook 层为各模型的 final vision-to-language projection MLP 层（如 Qwen3-VL 的 `model.model.visual.merger.linear_fc2`）。
- **实现细节**：bfloat16 加载模型，float32 执行 Steering 计算，token 数上限 2048，随机种子 42，对数/指数映射中 geodesic 距离 <10⁻⁸ 视为零，arccos 输入 clamp 到 $[-1+10^{-7}, 1-10^{-7}]$。
