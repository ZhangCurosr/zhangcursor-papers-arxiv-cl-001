---
title: "Knowing-Isn-t-Always-Saying-When-Do-Spatial-Encodings-Reach"
source: https://arxiv.org/pdf/2608.22916v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:09:40"
---

# 论文速读：Knowing-Isn-t-Always-Saying-When-Do-Spatial-Encodings-Reach

## 一句话总结
本文利用方向插值（direction patching）因果干预方法，在层深、词元位置与提示格式三个正交维度上系统追踪了VLM中空间编码到答案logits的传输路径，揭示文本CoT提示会条件性门控抑制对象词位置的argmax级传输，而视觉grounding提示可绕过该抑制；VLM的“编码-接地”差距本质上是条件性传输问题。

## 研究问题与动机
1. **表示-行为脱节缺乏动态因果解释**：现有研究多停留在静态探针，证明VLM能在隐藏状态中编码空间属性却不会输出，但未回答该编码信息“何时、何地被因果用于答题”。
2. **CoT提示在空间任务上的反直觉降级**：Chain-of-Thought在复杂推理中普遍有效，但在VLM空间任务上反而显著降低多项开源模型准确率，其底层阻断机制尚未定位。
3. **早期表征是否即时转化为输出因果力**：空间ID方向可从浅层（L4–L8）表征提取，但该类方向是否立即影响答案选项竞争？传输是否存在层深阈值？
4. **提示格式作为传输路由的控制变量**：除CoT外，不同推理长度与视觉 grounding 指令如何改变空间信号的路径开关状态，尚无系统性图谱。

## 核心贡献（创新点）
1. **首个VLM空间信息传输的三维因果图谱**：将方向插值沿层深、词元位置、提示格式进行网格化扫描，首次量化绘制空间-ID方向到达答案logits的完整传输拓扑，将编码-接地差距重构为条件性传输问题。
2. **提出CoT“门控”而非“擦除”的机制解释**：证明文本CoT并未抹除底层的连续logit增益信号，而是将argmax翻转阈值以下的方向性影响抑制在对象词位置，属于条件性门控。
3. **发现视觉grounding提示可主动绕过CoT抑制**：所有10个模型在visual_cot/visual_direct下均恢复正向传输，Janus模型下增幅达5.8倍，表明推理格式（抽象文本vs视觉观察）是通路开关的决定性变量。
4. **刻画“位置重定位”（relocation）传输模式**：在多数模型中，对象词位置的早期传输随层加深衰减，并在更深层的prefix_last位置重新出现，提示传输路径由提示类型与词元位置联合决定。
5. **提供架构特异的头级因果证据（概念验证）**：在InternVL3-8B中发现单个注意力头（L27.H21）具有CoT特异性门控功能，敲除后可恢复传输，为后续电路级解释提供切入点。

## 方法详解
1. **方向插值干预**：基于Kang et al. (2026)的构造，在每层$\ell$的对象词位置计算四象限（或颜色）类条件质心$C_q^{(\ell)}$（5折crossfit防泄漏）。干预时向残差流添加偏移：$h_{p}^{(\ell)} \leftarrow h_{p}^{(\ell)} + \alpha\big(C_{\mathrm{target}}^{(\ell)} - C_{\mathrm{source}}^{(\ell)}\big)$，默认$\alpha=5$（obj_word）与$\alpha=10$（prefix_last）。
2. **双指标度量传输**：核心指标$\Delta$argmax为目标选项argmax翻转率相对于同范数随机方向基线的提升，用于判定argmax级因果传输；辅助指标target-logit gain为干预前后目标字母logit均值的差值，用于捕捉未达翻转阈值的连续方向性影响。
3. **双位置与七格式扫描**：干预位置设为
