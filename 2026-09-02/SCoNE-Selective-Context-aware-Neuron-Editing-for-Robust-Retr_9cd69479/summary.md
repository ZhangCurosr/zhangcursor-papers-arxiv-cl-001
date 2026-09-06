---
title: "SCoNE-Selective-Context-aware-Neuron-Editing-for-Robust-Retr"
source: https://arxiv.org/pdf/2609.00689v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 10:00:28"
---

# 论文速读：SCoNE-Selective-Context-aware-Neuron-Editing-for-Robust-Retrieval-Augmented-Generation

## 一句话总结
提出SCoNE，一种无需训练的推理时模型编辑方法，通过联合筛选高归因与高跨输入可变性的FFN神经元并放大其权重，在不改变RAG管线架构的前提下显著提升生成模型对混杂检索噪声的鲁棒性。

## 研究问题与动机
1. **检索噪声导致LLM分心与幻觉**：真实RAG系统中检索器返回的文档常混杂高相关gold证据与无关distractor，LLM易被噪声干扰，导致答案质量降级而非提升。
2. **现有降噪方案各有硬性缺陷**：提示工程与重排序/压缩模块会引入额外组件与级联误差，增加推理延迟；生成器微调（如RetRobust、PA-RAG）虽有效，但面临灾难性遗忘、高昂算力成本与高质量训练数据依赖。
3. **传统模型编辑无法直接复用**：现有编辑方法（如MEMIT）均假设目标知识静态已知，而RAG的检索上下文是开放且动态变化的，模型必须在推理时自适应应对任意内容组合的上下文。
4. **单一归因指标在多文档场景下失效**：仅凭归因强度选出的神经元可能对任意上下文均广泛响应（反映表层共现模式），无法区分“选择性利用信息性证据”与“泛化激活”，需引入跨样本波动信号进行补充。

## 核心贡献（创新点）
1. **提出零训练、零额外延迟的推理时神经元增强框架**：与RetRobust/PA-RAG等依赖昂贵微调的管线本质不同，SCoNE仅在推理前对极少量FFN权重做线性缩放，彻底规避重新训练与存储多份检查点的需求。
2. **设计“高归因+高跨输入可变性”双准则神经元挖掘机制**：与IRCAN仅依赖静态归因强度不同，本文引入滑动窗口残差可变性，使筛选出的神经元具备对混杂证据 composition 的选择性响应能力，而非被动跟随任意输入。
3. **证明极低频参数编辑即可匹敌重型基线**：仅调整约0.0011%的FFN维度（Llama-3-8B中约5个层级神经元），性能即追上甚至超越PA-RAG，展现了参数高效编辑在RAG场景下的实用边界。
4. **提供系统的噪声鲁棒性与泛化性验证**：不仅在多基准上取得最高平均准确率，还在控制噪声实验中证明随干扰文档递增性能衰减更缓，且域外任务（HellaSwag/ARC/MemoTrap）未出现普遍性能力损伤。

## 方法详解
- **问题设定与挖掘集**：从HotpotQA训练集抽取N=100条样本构建挖掘集D，每条样本含查询q_t、答案a_t与上下文集合C_t=M=2个gold文档+K=8个distractor文档，模拟真实多文档RAG输入。
- **归因分数（Attribution）**：将Integrated Gradients推广至多上下文设置。定义v(q_t)为仅输入查询时神经元n的激活，v(q_t, C_t)为查询+全量上下文时的激活。归因公式为：
  $$\mathrm{Attr}(n; q_t, \mathcal{C}_t) = \big(v(q_t, \mathcal{C}_t) - v(q_t)\big) \times \int_0^1 \frac{\partial P(a \mid q_t, \mathcal{C}_t, v_\alpha)}{\partial v_\alpha} d\alpha$$
  其中$v_\alpha = v(q_t) + \alpha(v(q_t, \mathcal{C}_t) - v(q_t))$为线性插值路径，实践中采用20步Riemann和近似。
- **跨输入可变性（Variability）**：衡量神经元归因在滑动样本窗口内的动态波动，捕捉其对不同上下文组合的选择性。公式为：
  $$V^{(t)}(n_j^l) = \left|\mathrm{Attr}^{(t)}(n_j^l) - \frac{1}{W}\sum_{m=t-W}^{t-1} \mathrm{Attr}^{(m)}(n_j^l)\right|$$
  W为窗口大小（默认3），该设计保留序列局部变化特征，优于全局方差/标准差等顺序无关统计量。
- **神经元选择策略**：对每个样本t，分别取出正归因中Attr Top-50与Variability Top-50的交集$\mathcal{C}^{(t)}$作为局部候选；遍历全样本统计各神经元被选频次，取频次最高的Top-k（k=5）作为最终上下文感知神经元集合。Llama-3-8B-Instruct共32层×14,336维=458,752个FFN神经元，最终入选仅约0.0011%。
- **推理时增强**：对选中
