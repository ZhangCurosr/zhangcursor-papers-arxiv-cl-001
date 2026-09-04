---
title: "ALTSTEER-Selective-Safety-Steering-for-Moving-Beyond-Hard-Re"
source: https://arxiv.org/pdf/2608.30197v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:48:29"
field: "大模型安全与对齐"
keywords: ["activation steering", "LLM safety", "inference-time intervention", "constructive safe completion", "refusal steering", "representation engineering"]
innovations: ["拒绝相关内部信号门控替代内容分类门控，跨域更稳定", "正交化替代向量分离建设性续写信号", "阶段性调度协调拒绝锚定向建设性重定向的转移"]
benchmarks: ["BeaverTails", "AdvBench", "MaliciousInstruct", "JailbreakBench", "XSTest", "AlpacaEval", "MATH500", "GSM8K"]
---

# 论文速读：ALTSTEER: Selective Safety Steering for Moving Beyond Hard Refusals to Constructive Alternatives

## 一句话总结
ALTSTEER 是一种推理时（inference-time）安全引导框架，通过拒绝相关的内部信号实现选择性干预，并在单次推理过程中将生成轨迹从拒绝导向控制逐步转移到建设性安全替代方案，在保持良性任务效用不变的前提下显著提升有害请求下的"拒绝且提供替代方案"比例。

## 研究问题与动机
1. **现有门控信号跨域不稳定**：基于固定阈值或有害原型的触发机制（如CAST、Latent Guard）在良性和有害分布上重叠严重，容易在数学推理等良性任务上误触发干预，导致效用退化。
2. **拒绝导向引导停留在模板化拒绝**：即使正确触发干预，现有方法（包括SAFESTEER）仍倾向于生成简短、模板化的拒绝回复，缺乏解释性或建设性替代方案。
3. **训练后对齐方法成本高且灵活性差**：SFT/RLHF等方法需重新训练模型，难以在不同部署场景中灵活适配。
4. **"何时干预"与"如何塑造响应"被割裂研究**：现有工作主要聚焦选择性触发机制，忽视了干预后生成形式的质量控制。

## 核心贡献（创新点）
1. **识别双局限并系统论证**：通过实证分析证明现有门控信号跨域不稳定性以及拒绝导向引导的模板化问题，填补了"何时干预"与"如何 shaping"之间研究空白。
2. **提出 ALTSTEER 推理时框架**：在单次推理过程中耦合选择性干预与拒绝锚定的建设性重定向，无需额外训练。
3. **拒绝相关内部信号（Refusal-relevant gating）**：基于模型内部激活与拒绝方向的平均余弦相似度决定干预时机，优于基于有害内容分类的门控信号，在良性输入上更保守。
4. **正交化替代向量（Orthogonalized alternative vector）**：构造"解释性拒绝 vs 简单拒绝"的对比向量并去除与拒绝方向的重叠成分，独立编码建设性续写信号。
5. **阶段性调度（Staged steering schedule）**：以步数依赖的强度函数协调拒绝锚定（早期主导）向建设性替代（后期主导）的平滑转移，平衡安全性与有用性。

## 方法详解
1. **拒绝相关内部信号**：对每层 $l$ 计算输入最终 token 隐状态 $h^l(x)$ 与拒绝向量 $v_{\mathrm{ref}}^l$ 的余弦相似度，取所有 $L$ 层的平均值 $s_{\mathrm{ref}}(x)$；以零为参数无关阈值，$s_{\mathrm{ref}}(x) > 0$ 时触发干预。
2. **拒绝向量构造**：使用有害提示集合 $D_{\mathrm{ref}}$（朴素模型拒绝的）与 $D_{\mathrm{comp}}$（朴素模型遵从的），通过差异均值估计器 $v_{\mathrm{ref}}^l = \mu_{\mathrm{ref}}^l - \mu_{\mathrm{comp}}^l$ 得到每层拒绝方向。
3. **替代向量构造与正交化**：使用相同有害提示配合不同 system prompt，生成解释性拒绝集合 $D_{\mathrm{exp}}$ 与简单拒绝集合 $D_{\mathrm{sim}}$，计算 $v_{\mathrm{alt}}^l = \mu_{\mathrm{exp}}^l - \mu_{\mathrm{sim}}^l$，然后通过 Gram-Schmidt 过程去除其在 $v_{\mathrm{ref}}^l$ 上的投影：$v_{\mathrm{alt}}'^l = v_{\mathrm{alt}}^l - \frac{\langle v_{\mathrm{alt}}^l, v_{\mathrm{ref}}^l\rangle}{\langle v_{\mathrm{ref}}^l, v_{\mathrm{ref}}^l\rangle}v_{\mathrm{ref}}^l$。
4. **隐状态修改公式**：在解码步 $i$、层 $l$ 处，修改后的隐状态为 $h_i'^l = h_i^l + \lambda_{\mathrm{ref}}(i)v_{\mathrm{ref}}^l + \lambda_{\mathrm{alt}}(i)v_{\mathrm{alt}}'^l$。
5. **阶段性调度**：设定最大生成长度 $T=512$，$\lambda_{\mathrm{ref}}(i) = \lambda_{\mathrm{ref}}^{(0)}(1-\sqrt{i/T})$，$\lambda_{\mathrm{alt}}(i) = \lambda_{\mathrm{alt}}^{(0)}(1+\sqrt{i/T})$；Llama-3.1 使用层13、$(\lambda_{\mathrm{ref}}^{(0)}, \lambda_{\mathrm{alt}}^{(0)})=(1.5, 1.0)$，Qwen2.5 使用层14、$(5.0, 3.0)$。

## 实验与结果
- **模型**：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct；迁移验证 Qwen2.5-14B-Instruct、Mistral-7B-Instruct-v0.3。
- **有害基准**：BeaverTails、AdvBench、MaliciousInstruct、JailbreakBench；度量 Refusal Rate、Alternative Rate、RAR（拒绝且提供替代）、Leakage Rate、Safe Alt。
- **良性基准**：XSTest（合规率 CR）、AlpacaEval（胜率 WR）、MATH500 和 GSM8K 准确率。
- **主要结果（Llama-3.1，平均加权）**：
  - ALTSTEER 将 Safe Alt 从基线的 22.4% 提升至 61.1%，远超 AdaSteer（35.1%）、SafeSwitch（30.1%）、SAFESTEER（31.0%）。
  - BeaverTails RAR 达 62.1%（基线 34.1%），AdvBench RAR 达 79.0%，MaliciousInstruct RAR 达 92.0%。
  - 良性效用：XSTest CR 保持 93.6%，MATH Acc 32.2%，GSM8K 74.3%，接近基线；CAST 在 MATH 上降至 1.4%。
- **主要结果（Qwen2.5）**：AdvBench RAR 97.7%，MaliciousInstruct RAR 96.0%，良性 XSTest CR 95.2%，AlpacaEval WR 51.1%，MATH 48.0%，GSM8K 83.2%。
- **消融**：所有层门控在 Llama-3.1 上 F1 为 0.851，Qwen2.5 为 0.888；拒绝锚定对控制 leakage 至关重要（拒绝仅 leakage 5.5%，仅替代 leakage 24.9%）。

## 相关工作脉络
1. **CAST**（Lee et al., 2025）：基于层特定阈值和分类器的选择性拒绝引导；本质区别：使用内容分类门控而非内部对齐信号，跨域易误触发。
2. **AdaSteer**（Zhao et al., 2025b）：基于有害性信号自适应调整引导强度；ALTSTEER 不依赖有害性检测，且在响应形式控制上更进一步。
3. **SafeSwitch**（Han et al., 2025）：使用安全探针监测内部状态；在 XSTest 上出现明显效用退化，ALTSTEER 门控更保守。
4. **SAFESTEER**（Ghosh et al., 2025）：引导至非拒绝安全内容方向；仍保留大量模板化拒绝，ALTSTEER 强调拒绝锚定基础上的建设性转移。
5. **AlphaSteer**（Sheng et al., 2026）：在零空间约束下学习输入依赖的拒绝引导；ALTSTEER 无需训练零空间投影，框架更轻量。
6. **输出中心安全对齐**（Yuan et al., 2025; Zhang et al., 2025）：基于 SFT/RLHF 的训练方法；ALTSTEER 在推理时工作，无需重训。

## 局限性与未来方向
1. **依赖参考向量质量**：正交化替代向量 $v_{\mathrm{alt}}'^l$ 的效果取决于基础模型对建设性安全完成行为的表征强度，弱表征模型可能收益有限。
2. **未完全消除安全泄漏**：引导建设性替代会增加有害信息泄露的风险，Leakage Rate 随 Alternative Rate 上升而上升。
3. **单 Pass 与两阶段管道的权衡**：ALTSTEER 定位为单次推理开销下的折中方案，而非替代事后重写管道；后者在额外生成成本下可获得更强绝对质量。
4. **未来方向**：更系统的向量构造方法、进一步降低 leakage、探索多轮对话鲁棒性（附录D已初步验证两 Turn 下 gate 激活率从 93% 降至 89.5%）。

## 研究启发与可借鉴点
1. **内部对齐信号替代内容分类门控**：用"模型内部是否对齐拒绝方向"判断是否干预，比"输入是否有害"更稳定且参数无关，可迁移至其他行为引导场景（如诚实性、风格控制）。
2. **拒绝锚定 + 建设性重定向的序列化控制**：先锚定安全边界再逐步开放建设性内容的阶段性调度思想，可用于"硬约束软化"类任务。
3. **模板拒绝率（Template Refusal Rate）作为新评估维度**：量化模型拒绝形式的丰富程度，为安全对齐研究提供更细粒度度量。
4. **对比构造法提取行为向量**：对同一输入使用不同 system prompt 生成不同响应风格，通过隐状态差异提取方向向量，可复用于 CoT 压缩、风格迁移等任务。

## 关键术语表
**Refusal-relevant internal signal**：衡量输入表示与拒绝方向平均余弦对齐度的门控信号，用于判断是否触发干预，阈值参数无关。
**Refusal vector**：通过有害提示中"模型拒绝"与"模型遵从"两组的隐状态均值差估计得到的方向向量。
**Alternative vector**：通过"解释性拒绝"与"简单拒绝"两组的隐状态均值差估计得到的方向，编码建设性续写倾向。
**Orthogonalization**：从替代向量中去除其在拒绝向量上的投影，以分离出独立于拒绝的建设性成分。
**Staged steering**：随解码步数动态调整拒绝与替代强度的调度策略，早期侧重拒绝锚定，后期侧重建设性重定向。
**Refusal-and-Alternative Rate (RAR)**：同时拒绝有害请求并提供建设性替代方案的响应比例，核心评估指标。
**Safe Alt**：提供安全替代方案的比例 = Alternative Rate × (1 - Leakage Rate)，综合安全性与建设性。
**Template refusal**：简短、固定格式的拒绝回复（如"I can't answer that"），缺乏解释或建设性内容。

## 可复现要素
- **数据集**：BeaverTails（train split 采样2500构建向量）、Alpaca（无害原型）、MaliciousInstruct、AdvBench、JailbreakBench、MATH500、GSM8K、XSTest、AlpacaEval；部分数据集公开。
- **代码/权重**：论文未提及开源仓库或模型权重。
- **关键超参**：$T=512$；Llama-3.1 干预层13，$\lambda_{\mathrm{ref}}^{(0)}=1.5$，$\lambda_{\mathrm{alt}}^{(0)}=1.0$；Qwen2.5 干预层14，$\lambda_{\mathrm{ref}}^{(0)}=5.0$，$\lambda_{\mathrm{alt}}^{(0)}=3.0$；门控阈值为零（参数无关）。
- **实现环境**：PyTorch + Transformers，单卡 NVIDIA A40，greedy decoding（do_sample=False）。
