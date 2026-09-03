---
title: "Credal-Large-Language-Models-for-Semantic-Commitment-under-U"
source: https://arxiv.org/pdf/2608.23244v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 05:21:20"
---

# 论文速读：Credal-Large-Language-Models-for-Semantic-Commitment-under-U

## 一句话总结
本文提出 Credal Large Language Models (CLLM) 框架，通过冻结主干的多个 LoRA 适配器集合构建 credal set（概率分布凸包），显式保留词元预测的下界/上界；在此基础上设计 CTC（零额外生成的词元承诺分）与 SCC（词元-语义联合承诺分），显著提升 LLM 在幻觉检测、校准度与选择性预测任务中的可靠性。

## 研究问题与动机
- **单点分布丢失不确定性几何**：标准 LLM 仅输出单一 softmax 分布，将认知无知（epistemic ignorance）与真正的语义歧义混淆，导致“流畅但错误”且自信度虚高的回答。
- **现有集成分散度量过于粗糙**：LoRA 集成、贝叶斯近似等方法最终均将多假设坍缩为单个预测分布或标量不一致性，无法保留“ plausible 预测分布之间的 Spread”。
- **词元级与语义级信号互有盲区**：词元熵无法区分同义不同形，语义聚类又可能将无害 paraphrase 误判为歧义；缺乏跨层级的联合验证与对齐诊断机制。
- **部署需要低成本且可交叉验证的承诺信号**：安全关键场景（医疗、法律、科研辅助）要求系统既能快速拒答，又能明确区分“词元高置信但语义分裂”与“词元-语义双重稳健支持”两类失败模式。

## 核心贡献（创新点）
- **提出 CLLM 框架**：用 $M$ 个独立初始化 LoRA 适配器的预测点云凸包构成 credal set，显式保留 $\underline{P}$ 与 $\overline{P}$。与 LoRA 集成/Bayesian LoRA 的本质区别在于不平均/不积分，而是保留集合的几何边界。
- **定义两个正交的 credal 不确定性度量**：Intersection entropy $H_\cap$ 刻画谨慎点分布的弥散度，Credal width $W$ 刻画适配器间的概率 Spread，二者分别对应“单点预测的锐度”与“多假设的一致性”。
- **设计 CTC（Credal Token Commitment）**：仅依赖 credal set 几何信息，融合下界支持、credal width 与 intersection entropy 的乘性分数，无需额外采样即可用于幻觉检测与选择性预测。
- **提出 SCC 与 SCC-Gap**：将承诺扩展至语义空间，SCC 要求词元与语义双重支持；SCC-Gap 诊断两者分歧，专门针对对抗提示下的表面流畅-语义分裂场景。

## 方法详解
- **Credal Set 构建**：冻结 backbone，挂载 $M=5$ 个随机种子不同的 LoRA 适配器（$r=8, \alpha=16$, dropout=0.1）。对输入 $x$，第 $m$ 个适配器输出词元分布 $p_m(\cdot|x)$。Credal set 为 $\mathcal{P}(x)=\text{conv}\{p_1,\dots,p_M\}$，词元 $y$ 的下界/上界为 $\underline{P}(y|x)=\min_m p_m(y|x)$、$\overline{P}(y|x)=\max_m p_m(y|x)$。
- **Intersection Probability Transform**：取 $\hat{p}(y|x)=\underline{P}(y|x)+\alpha(\overline{P}(y|x)-\underline{P}(y|x))$ 并归一化，得到位于 credal set 内部的代表分布，用于选取预测 $y^*=\arg\max_y \hat{p}(y|x)$。
- **Credal Uncertainty Measures**：
  - Intersection entropy：$H_\cap(x)=-\sum_y \hat{p}(y|x)\log\hat{p}(y|x)$
  - Credal width：$W(x)=\frac{1}{|\mathcal{V}|}\sum_y\left(\overline{P}(y|x)-\underline{P}(y|x)\right)$
- **CTC 分数**：$\text{CTC}=C_{\text{tok}}\cdot(1-W)\cdot\left(1-\frac{H_\cap}{\log|\mathcal{V}|}\right)$，其中 $C_{\text{tok}}=\frac{\exp(\beta\underline{P}(y^*))}{\exp(\beta\underline{P}(y^*))+\sum_{y\neq y^*}\exp(\beta\overline{P}(y))}$。乘性设计确保任一因子失效即拉低总分；$C_{\text{tok}}$ 关注下界 margin，$1-W$ 惩罚适配器广泛分歧，第三项惩罚点分布弥散。
- **SCC 与 SCC-Gap**：采样 $K=16$ 条完整补全，用 BAAI/bge-base-en-v1.5 嵌入+余弦聚类得主簇 $c^*$ 及其质量 $S(c^*)$。$C_{\text{sem}}=S(c^*)\cdot(S(c^*)-\max_{c\neq c^*}S(c))_+$，$\text{SCC}=C_{\text{tok}}\cdot C_{\text{sem}}$，$\text{SCC-Gap}=|C_{\text{tok}}-C_{\text{sem}}|$。SCC-Gap 在词元高置信但语义分裂时显著放大。

## 实验与结果
- **设置**：Backbone 为 Gemma-2-9B-Instruct、Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct；数据集 OpenBookQA、CoQA、TriviaQA、ARC-Challenge、AdvBench；基线覆盖 Standard LLM、LoRA Ensemble、Bayesian-LoRA (KFAC)、Laplace-LoRA、Semantic Entropy (cosine/NLI)。所有采样基线使用相同计算预算（$K=16$, 后验采样 $S=20$）。
- **幻觉检测**：在 8 个 model×benchmark 设置中，intersection entropy 在 4/8 最优，CTC
