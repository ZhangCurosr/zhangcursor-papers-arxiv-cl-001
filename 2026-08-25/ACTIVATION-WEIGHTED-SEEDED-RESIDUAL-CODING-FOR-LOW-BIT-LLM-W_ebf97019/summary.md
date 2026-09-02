---
title: "ACTIVATION-WEIGHTED-SEEDED-RESIDUAL-CODING-FOR-LOW-BIT-LLM-W"
source: https://arxiv.org/pdf/2608.23144v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:46:35"
---

# 论文速读：ACTIVATION-WEIGHTED-SEEDED-RESIDUAL-CODING-FOR-LOW-BIT-LLM-W

## 一句话总结
本文提出激活加权种子残差编码（AWSRC），作为一种即插即用的轻量级侧车修复器，在固定低比特量化主干（如RTN/GPTQ）的基础上，通过确定性种子生成Hadamard结构化基与激活加权最小二乘拟合，以仅+0.162 scope-bpw的极小字节开销显著修复量化误差，追回至BF16的88.2% PPL、78.9% KL与71.3%任务准确率差距。

## 研究问题与动机
- 低比特权重量化（PTQ）虽能大幅压缩存储与加载成本，但量化引入的残差 $R=W-W_0$ 会劣化LLM质量，而直接存储密集高精度残差副本的代价等同于原始权重。
- 现有残差补偿方案（稀疏保留坐标、低秩密集因子、向量量化码本）均假设残差具备特定结构，但LLM量化残差通常既非稀疏也非低秩，且显式存储基/码本会快速耗尽极小预算。
- 均匀修复所有权重误差会浪费压缩容量，因为不同权重在实际输入分布下对层输出的扰动贡献差异巨大，需按“实际影响”分配字节。
- 工程上亟需一种独立预算、确定可解码、可复用任意现成主干、并能按每字节修复收益进行自适应分配的Compact Codec。

## 核心贡献（创新点）
- **零码本种子生成残差基**：利用共享伪随机种子确定性生成符号翻转与Hadamard列组合，无需序列化 $O(CP)$ 基矩阵，仅存 $\lceil\log_2 S\rceil$ bit选择器即可在编解码两端复现同一子空间。
- **激活加权拟合与字节归一化分配**：以校准得到的输入激活对角协方差 $D$ 为权重求解最小二乘，并定义每字节加权误差下降 $\rho_t$ 作为记录选择与排序准则，使压缩预算精准投向对层输出影响最大的残差瓦片。
- **渐进可截断修复流设计**：将所有有效记录按 $\rho_t$ 全局排序形成连续比特流，任意完整前缀均可独立解码且始终相对于同一固定主干 $W_0$，无需中间模型快照或重新拟合。
- **严格的字节审计跨模型验证**：在Qwen2.5-3B及4个跨家族模型上证明，+0.162 scope-bpw即可追回至BF16的88.2%/78.9%/71.3%差距，且在49.25 MB同体积下综合指标优于稀疏、低秩与VQ等强基线。

## 方法详解
- **问题定义与分解**：给定任意量化主干 $W_0=Q(W)$，残差 $R=W-W_0$，修复目标为 $\widehat{W}=W_0+\widehat{R}$。主干保持冻结，修复完全由独立序列化的侧车（sidecar）完成。
- **种子生成基（Eq.2）**：将残差按行切分为大小为 $C$ 的瓦片 $\boldsymbol{r}$。种子 $s$ 驱动同步伪随机生成器，选定符号模式、坐标排列与 $P$ 列顺序-$C$ Hadamard矩阵列，归一化后构成 $B_s\in\mathbb{R}^{C\times P}$。编解码两端共享生成器版本与数值约定，故无需传输基矩阵。
- **激活加权拟合（Eq.3）**：从32条校准文本计算 $D=\mathrm{diag}(\mathbb{E}[x_j^2])$。对候选基求解 $\alpha_s^*=\arg\min_\alpha\|B_s\alpha-\boldsymbol{r}\|_D^2$，该目标在 diag 近似下等价于最小化期望输出扰动 $\mathbb{E}_x[(\boldsymbol{r}-\widehat{\boldsymbol{r}}_s)^\top x]^2$。系数经单幂二标度量化后得到 $\widehat{\alpha}_s$。
- **字节归一化增益与分配（Eq.4）**：记录增益 $\rho_t=(\|\boldsymbol{r}_t\|_D^2-\|\boldsymbol{r}_t-\widehat{\boldsymbol{r}}_t\|_D^2)/B_t$，$B_t$ 为记录完整序列化字节数。丢弃 $\rho_t\leq0$ 的记录，逐模块保留（U型）或全局排序（P型）形成渐进流。
- **序列化与存储模型（Eq.5）**：载荷长度 $\ell_{\text{payload}}=\lceil\log_2 S\rceil+b_e+Pb$，侧车总字节 $B_{\text{side}}=h+K\lceil(\lceil\log_2 T\rceil+\ell_{\text{payload}})/8\rceil$。报告 scope-bpw（仅修复范围）与 full-effective-bpw（含全模型BF16参数）。可选融合Fisher信息对角（$F$ 变体）并支持 Scale Double Quantization 压缩主干 group-scale。

## 实验与结果
- **评测设置**：主模型 Qwen2.5-3B-Instruct，目标范围为108个MLP gate/up/down投影；WikiText-2测试PPL（长度2048/stride 1024）与paired KL；6任务zero-shot平均准确率（PIQA/HellaSwag/COPA/RTE/OpenBookQA/LAMBADA）。
- **核心数字**：在INT4 RTN主干上叠加 AWSRC-U/P_F 后，scope-bpw仅增0.162，PPL从9.80降至7.04，KL从0.35降至0.08，准确率从0.65升至0.69；相对BF16的差距分别追回88.2%、78.9%、71.3%，各项指标均为同预算最优。
- **同字节Codec对比**：在严格匹配49,245,876 bytes（4.23 scope-bpw）条件下，AWSRC取得最低PPL(7.04)与最高平均准确率(0.69)，优于稀疏编码、激活加权低秩(rank-34 INT8)与学习VQ(K=256 FP16)。
- **跨主干与跨模型泛化**：对QAM-W-4-SDQ同样有效；对GPTQ/AWQ提升有限或方向不一致，表明修复收益与父量化器的误差分布相关。在Qwen3-4B、Llama-3.2-3B、Qwen2.5-Coder-7B、Yi-1.5-9B四族模型上配对测试均获得三项或多项提升。
- **工程验证**：独立进程加载序列化侧车可复现完全相同的采样哈希与WikiText-2 PPL(7.06，误差<0.01)，解码耗时6.89–7.10秒，验证了序列化格式的无歧义性与可移植性。

## 相关工作脉络
- **GPTQ/AWQ**：主流PTQ方法，分别引入二阶逆校正与激活感知通道重缩放；本文将其视为固定主干 $W_0$，与其正交的侧车修复形成互补而非替代。
- **SqueezeLLM（稀疏残差）**：保留敏感离群值及其FP16坐标；本文不依赖残差天然稀疏，而是通过激活加权筛选高价值瓦片并按字节分配。
- **LQER/QERA（低秩残差）**：用密集低秩因子重建量化误差；本文避免存储 $O(CP)$ 密集因子，改用种子生成结构化正交基以降低存储维度。
- **AQLM/GPTVQ（向量量化）**：学习码本并存储分配索引；本文省 CODEBOOK，用确定性Hadamard子空间替代，更适合极低字节预算场景。
- **SeedLM**：共享种子生成完整权重块；本文将其思想迁移至“残差修补”设定，保留原主干不变，仅编码 $W-W_0$，实现旁路修复。
- **Any-Precision/BitStack**：单表示多精度压缩；本文定位更聚焦，追求在单一预算下无限逼近BF16，而非提供多档位折中。

## 局限性与未来方向
- 当前仅评估压缩质量与
