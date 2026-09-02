---
title: "ACTIVATION-WEIGHTED-SEEDED-RESIDUAL-CODING-FOR-LOW-BIT-LLM-W"
source: https://arxiv.org/pdf/2608.23144v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:46:16"
field: "大模型权重量化与压缩"
keywords: ["weight quantization", "residual coding", "seeded representation", "LLM compression", "post-training quantization", "activation weighting"]
innovations: ["零码本种子基底残差编解码，以短种子选择器替代显式基底/码本存储", "激活加权字节归一化分配准则，联合优化修复增益与序列化开销", "渐进式前缀可独立解码，支持任意位宽预算的无损前缀截断"]
benchmarks: ["WikiText-2 PPL", "Paired KL divergence", "PIQA/HellaSwag/COPA/RTE/OpenBookQA/LAMBADA mean accuracy"]
---

# 论文速读：ACTIVATION-WEIGHTED-SEEDED-RESIDUAL-CODING-FOR-LOW-BIT-LLM-W

## 一句话总结
论文提出 AWSRC（Activation-Weighted Seeded Residual Coding），一种后处理残差修复编解码器，通过将量化骨干网的残差编码为种子生成的基底低比特展开，在不修改父量化的前提下以极小附加存储显著修复低比特 LLM 的精度损失。

## 研究问题与动机
- **低比特权重量化的精度退化**：RTN、GPTQ、AWQ 等 weight-only PTQ 方法虽节省存储，但引入的量化误差会显著降低语言模型质量，需要一种低成本修复手段。
- **残差修复的存储与效率矛盾**：直接还原残差等同于存储一份完整的高精度副本；现有方案（稀疏、低秩、VQ）各有代价——稀疏需存储坐标、低秩需存储稠密因子、VQ 需存储码本。
- **误差度量应源于实际激活分布**：均匀对待所有权重坐标会浪费压缩容量于对层输出影响小的坐标，需利用校准激活统计量优先修复"重要"误差。
- **模块化解耦需求**：修复器应可独立预算、与多种骨干网兼容，并支持渐进式码流（任意前缀即可解码），而非绑定特定量化方法。

## 核心贡献（创新点）
1. **零码本种子残差编解码器**：用共享伪随机生成器从种子衍生 Hadamard 子空间基底，仅存储短种子选择器，无需序列化任何显式码本或基底矩阵。
2. **激活加权字节归一化分配**：以 $D=\text{diag}(\mathbb{E}[x^2])$ 加权最小二乘拟合残差，并以"每存储字节的加权误差下降"（$\rho_t$）作为记录级增益度量，联合决策基底选择、tile 选取与写入顺序。
3. **渐进式前缀与可选 Fisher 加权**：将记录按 $\rho_t$ 全局排序形成渐进码流，任意前缀均可独立解码；同时支持 Fisher 信息与激活对角线的混合加权（AWSRC-$P_F$）。
4. **跨骨干网泛化的后处理修复框架**：AWSRC 仅依赖父量化提供的 $W_0$，可兼容 RTN、AWQ、QAM-W 等不同强度骨干，同一 codec 可分别修复弱/强低比特模型。
5. **严格的字节审计评估**：所有基线在完全相同的序列化侧车大小（49,245,876 bytes）下对比，scope-bpw 与 full-effective-bpw 双指标报告，确保公平比较。

## 方法详解
**整体架构**：固定父量化 $W_0 = Q(W)$，残差 $R = W - W_0$ 被分块后由 AWSRC 编码为 $\widehat{R}$，修复权重 $\widehat{W} = W_0 + \widehat{R}$。

1. **种子生成基底**：
   - 将 $R$ 切分为行向 tile $r \in \mathbb{R}^C$。
   - 种子 $s$ 驱动同步伪随机生成器，选择符号模式、坐标置换并从 $C$ 阶 Hadamard 矩阵中挑选 $P$ 列，归一化后得到 $\boldsymbol{B}_s \in \mathbb{R}^{C \times P}$。
   - 编码/解码双方共享生成器版本与数值约定，故无需序列化基底。

2. **激活加权拟合**：
   - 从 32 条固定合成文本计算激活协方差对角线 $D = \text{diag}(\mathbb{E}[x_j^2])$。
   - 对每个候选种子求解加权最小二乘：$\alpha_s^* = \arg\min_\alpha \|\boldsymbol{B}_s \alpha - r\|_D^2 = (\boldsymbol{B}_s^\top D \boldsymbol{B}_s + \epsilon I)^{-1} \boldsymbol{B}_s^\top D r$。
   - 该目标近似 $\mathbb{E}_x[(r-\widehat{r}_s)^\top x]^2$，即量化输出扰动期望，从而优先修复对层输出影响大的坐标。
   - 系数经单幂两次缩放量化后存储；每个 record 含 tile 索引、种子选择器（$\lceil \log_2 S \rceil$ bit）、$P$ 个 $b$ bit 有符号系数及 scale 指数。

3. **字节归一化分配与渐进前缀**：
   - 每条记录的增益定义为 $\rho_t = \frac{\|r_t\|_D^2 - \|r_t - \widehat{r}_t\|_D^2}{B_t}$，$B_t$ 为序列化字节数；非正增益记录被丢弃。
   - 渐进模式（AWSRC-P）将所有合法记录按 $\rho_t$ 全局降序排列；截断码流仅移除完整加法修正，不改变父重构 $W_0$。
   - AWSRC-$P_F$ 进一步混合 Fisher 对角线（权重 0.75/0.25）参与增益计算。

4. **序列化开销建模**：
   - payload 长度 $\ell_{\text{payload}} = \lceil \log_2 S \rceil + b_e + Pb$。
   - 侧车字节 $B_{\text{side}} = h + K \lceil (\lceil \log_2 T \rceil + \ell_{\text{payload}})/8 \rceil$。
   - scope-bpw $= 8(B_{\text{bb}} + B_{\text{side}})/N_{\text{scope}}$，full-effective-bpw 额外计入范围外 BF16 参数。

5. **解码特性**：
   - 解码仅需 $W_0$、流头部与保留记录，无需校准语料与激活/Fisher 重算。
   - 无保留记录时精确还原 $W_0$；每条记录可独立剔除，不影响父格式。
   - 编码复杂度 $O(TS(CP^2 + P^3))$，解码复杂度 $O(KCP)$。

## 实验与结果
- **主模型**：Qwen2.5-3B-Instruct，目标作用范围为其 108 个 MLP gate/up/down 投影（其余权重保持 BF16）。
- **校准与评估**：32 条合成文本计算激活统计量；4096 个 WikiText-2 训练 token 用于可选 Fisher 对角线；PPL 在 16384 个 WikiText-2 测试 token（len=2048, stride=1024）上评估；paired KL 使用 4096 token；mean accuracy 为 PIQA、HellaSwag、COPA、RTE、OpenBookQA、LAMBADA 六任务 zero-shot 平均（每任务 ≤300 例）。
- **量化基线**：RTN（group size 128）、GPTQ、AWQ（官方 checkpoint）、QAM-W-4（论文规范 clean-room 复现）。
- **核心结果（Table 3）**：
  - BF16 baseline：PPL=6.70, KL=0.00, Acc=0.70。
  - RTN INT4：PPL=9.80, KL=0.39, Acc=0.65。
  - **RTN-SDQ + AWSRC-$P_F$**（4.23 scope-bpw / 6.71 full-effective-bpw）：PPL=7.04, KL=0.08, Acc=0.69。
  - 相对于 RTN 父量化的 gap 回收率：**PPL 88.2%、KL 78.9%、Accuracy 71.3%**。
- **相同字节下的残差编解码器对比（Table 4，侧车均为 49,245,876 bytes）**：
  - Sparse：PPL=7.15, KL=0.08, Acc=0.69
  - Quantized low-rank（rank=34, INT8）：PPL=7.09, KL=0.07, Acc=0.67
  - Learned VQ（K=256 FP16）：PPL=7.09, KL=0.07, Acc=0.68
  - **AWSRC：PPL=7.04, KL=0.08, Acc=0.69**（PPL 与 mean accuracy 最优）
- **跨模型迁移（Table 5）**：在 Qwen3-4B、Llama-3.2-3B、Qwen2.5-Coder-7B、Yi-1.5-9B 上均获得 paired PPL/KL/Acc 一致改善。
- **自洽性验证**：独立新进程加载 2.60 GB 完整 checkpoint（RTN-SDQ+AWSRC-$P_F$），WikiText-2 PPL=7.06（与密集评估差 0.01），加载耗时 6.89–7.10s。

## 相关工作脉络
1. **GPTQ / AWQ（Frantar et al., 2023; Lin et al., 2024）**：主流 weight-only PTQ 骨干，分别利用二阶信息补偿与激活感知重缩放；本文将其作为固定父量化基线，而非替代对象。
2. **LQER / QERA（Zhang et al., 2024, 2025）**：残差补偿的低秩表示法，存储稠密因子；AWSRC 以种子基底替代显式因子，避免 $O(SCP)$ 存储。
3. **SqueezeLLM（Kim et al., 2024）**：稀疏+稠密混合量化，需存储 outlier 坐标；AWSRC 无需坐标开销，以结构化 Hadamard 子空间捕获误差。
4. **AQLM / GPTVQ（Egziazarian et al., 2024; Baalen et al., 2024）**：向量量化需存储学习型码本与 assignment；AWSRC 为无码本设计，种子选择器仅数比特。
5. **SeedLM（Shafipour et al., 2025）**：同样利用种子生成器，但用于重建完整权重块；AWSRC 仅编码残差，保留父重建 $W_0$ 不变，实现更灵活的模块化叠加。
6. **Any-Precision / BitStack（Park et al., 2024; Wang et al., 2025）**：从单一表示提取多速率模型；AWSRC 定位更窄，专注后处理残差修复并提供非整数 scope-bpw 渐进码流。

## 局限性与未来方向
- **仅评估压缩质量，未测量运行时性能**：所有候选权重重建为密集 BF16 进行评估，未测量 packed 解码延迟、内存带宽消耗或 NPU 专用 kernel 加速效果。
- **修复作用域有限**：当前仅对 MLP gate/up/down 投影（108 个矩阵）施加修复，未覆盖全部线性层，与全线性层修复方法不对齐。
- **GPTQ/AWQ 等强骨干修复效果不一致**：Table 2 显示对 GPTQ/AWQ 父量化的修复增益有限甚至略有下降，表明修复有效性存在骨干依赖性。
- **编码计算成本随候选基数增长**：$O(TSP^3)$ 拟合在 $S$ 较大时开销显著，尽管解码代价低廉。
- **作者明确指出未来方向**：需测量 packed 解码吞吐、内存流量、延迟、加载成本及 serverless LLM cold-start 时间。

## 研究启发与可借鉴点
1. **"增益/字节"归一化度量作为资源分配准则**：$\rho_t$ 将修复质量与精确序列化开销联合纳入决策，可直接迁移至其他残差/旁路编码场景（如 LoRA adapter 选择、prompt token 压缩）。
2. **激活对角线替代全协方差的实用化近似**：用 $D=\text{diag}(\mathbb{E}[x^2])$ 捕捉坐标重要性，避免了全矩阵存储与求逆，在保持校准精度的同时大幅降低侧车与计算开销。
3. **种子生成结构化基底的无码本范式**：Hadamard+符号置换的候选池以 $\lceil\log_2 S\rceil$ bit 选择器编码，为低比特表示学习提供了免码本、可重放的替代路径。
4. **渐进前缀的独立可解码性设计**：按增益排序后任意前缀均有效，契合边缘设备按需加载、服务端按预算动态裁剪的部署需求。
5. **与任意 PTQ 骨干解耦的后处理接口**：AWSRC 仅要求输入 $W_0$，不依赖父量化训练细节，可作为通用"精度补丁"插件接入已有模型制品。

## 关键术语表
**AWSRC**：Activation-Weighted Seeded Residual Coding，本文提出的无码本残差修复编解码器，通过种子生成基底与激活加权拟合修复低比特量化误差。
**scope-bpw**：仅计算目标修复矩阵范围内的 bits-per-weight，用于公平比较相同参数的不同侧车开销。
**full-effective-bpw**：将 BF16 范围外参数也纳入平均，反映模型整体存储 footprint。
**seeded basis**：由伪随机生成器从短种子确定性地衍生出的结构化基底（Hadamard 列+符号/置换），无需序列化。
**activation-weighted LS**：以输入激活协方差对角线为权重的最小二乘拟合，近似量化误差对层输出的二阶扰动。
**progressive prefix**：按 $\rho_t$ 排序后的记录前缀流，任意完整前缀均可独立解码，支持渐进式精度预算分配。
**SDQ（Scale Double Quantization）**：源自 QLoRA 的双重量化技术，进一步压缩骨干网的 group-scale 元数据。
**gain-per-byte（$\rho_t$）**：单条记录在系数量化后的加权误差下降量与其序列化字节数的比值，作为记录级分配的核心准则。

## 可复现要素
- **数据集**：WikiText-2（校准 4096 token，PPL 评测 16384 token）、32 条固定合成文本；均为公开数据集，**数据公开**。
- **代码**：论文声明对 QAM-W 采用独立 clean-room 实现，AWSRC 未明确提及开源仓库；**代码未明确公开**（论文未声明）。
- **模型权重**：使用 Qwen2.5-3B-Instruct 等公开 checkpoint；**权重公开**。
- **关键超参**：
  - AWSRC-U: $(C, P, b, S) = (128, 2, 4, 2)$
  - AWSRC-$P_F$: $(C, P, b, S) = (16, 4, 4, 16)$，Fisher/激活混合比 0.75/0.25
  - RTN group size = 128
  - 侧车记录长度 = 7 bytes，header = 904 bytes
- **评估环境**：Ascend 910C 设备，无 NPU 专用量化 kernel。
