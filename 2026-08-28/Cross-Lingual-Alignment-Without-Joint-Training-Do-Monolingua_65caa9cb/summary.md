---
title: "Cross-Lingual-Alignment-Without-Joint-Training-Do-Monolingua"
source: https://arxiv.org/pdf/2608.27115v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:26:27"
---

# 论文速读：Cross-Lingual-Alignment-Without-Joint-Training-Do-Monolingua

## 一句话总结
本文证明严格单语语言模型（无共享参数、无联合训练、独立语料）仍能学习到可对齐的跨语言表征；仅需基于平行句对拟合一次正交旋转矩阵，即可在表征几何重建与因果功能迁移（跨模型激活插值）上实现高效的零样本跨语言对齐，表明跨语言对齐可源于语言与世界结构本身而非联合训练。

## 研究问题与动机
- **核心问题**：跨语言对齐是否必须依赖联合训练（共享参数、混合语言批次或显式对齐目标）？独立训练的单语模型能否收敛到通用的跨语言表征？
- **现有不足**：既往研究多聚焦多模态或联合训练的多语模型内部对齐机制（如共享子词词汇、相似输入分布、共享架构参数），缺乏对“严格独立单语模型间是否存在跨语言对齐”的实证检验，且多数工作仅停留在相关性测量。
- **理论动机**：Platonic Representation Hypothesis（Huh et al., 2024）预测表征会随规模向共享现实模型收敛，但该假说此前仅在跨模态场景验证，尚未在跨语言领域获得因果级证据。
- **方法必要性**：以往相关研究（如 Conneau et al., 2020b）仅使用 BERT 编码器并止步于事后线性映射，本文引入跨模型激活插值（activation patching）以检验对齐后的表征是否真正驱动接收模型的功能行为。

## 核心贡献（创新点）
1. **严格单语场景下的跨语言对齐实证**：首次系统性验证完全独立训练、无共享参数的单语 LM 仍能形成可对齐的表征几何，且对齐强度随数据规模、模型规模及语言亲缘性单调增强。
2. **正交旋转优于宽松映射的重建发现**：证明单一 Procrustes 正交旋转在保持角几何与邻域检索（P@1）上显著优于无约束仿射变换与单层 MLP，揭示对齐的充分条件并非最小化 MSE 而是保留角结构。
3. **零样本因果功能迁移验证**：通过跨模型激活插值证明拟合一次的旋转矩阵可零样本迁移至新任务（事实完形填空）、新 token 位置（概念词中端而非句末）与新关系类型，实现 66%–85% 的方向性成功率。
4. **支持跨语言版本的 Platonic 假说**：将“表征收敛”机制从多语联合训练场景剥离，论证语言结构与世界信息本身足以驱动语言无关的概念空间形成，为模块化多语系统提供理论基础。

## 方法详解
- **数据与模型设置**：使用 Goldfish 严格单语 Causal LM 家族（5MB/10MB/100MB/1000MB 四档）及五个独立研发的 ~1B 单语模型（Pythia-EN/ZH、Tucano-PT、Bielik-PL、Minerva-IT），涵盖不同架构、分词器、训练语料与研究组。评估语料：FLORES-200、Tatoeba、OPUS、BouQUET。
- **相关性度量（CKA）**：采用线性 Centered Kernel Alignment：$\operatorname{CKA}(X,Y)=\frac{\operatorname{HSIC}(X,Y)}{\sqrt{\operatorname{HSIC}(X,X)\operatorname{HSIC}(Y,Y)}}$，对正交变换与各向同性缩放不变。构建 matched（平行句对）与 shuffled（一侧随机置换）对照，差值 $\Delta$ 隔离语义对应。三种 pooling 策略：mean pooling、SGPT 位置加权 pooling（对 causal LM 效果最佳）、token-level word-aligned pooling（基于 SIMALIGN 对齐词跨度平均子词隐状态）。
- **构造性映射重建**：在 72 个方向对上，用平行句末尾激活拟合三类投影：
  (i) **Procrustes**：$W^\star = \arg\min_{W^\top W=I} \|XW - Y\|_F^2$，闭式解为 $UV^\top$（SVD of $X^\top Y$）；
  (ii) **Affine**：无正交约束，含偏置 $b$；
  (iii) **MLP**：单层 ReLU 网络 $f_\theta(x)=W_2\operatorname{ReLU}(W_1x+b_1)+b_2$。
  评估指标：$\mathrm{P@1_{std}}$（仅测试池检索）、$\mathrm{P@1_{hard}}$（训练+测试池检索）、MSE、投影后 CKA。
- **因果性激活插值**：构建六语国家→首都事实完形提示；在源模型 donor 提示的概念词末子词位置缓存残差 $\mathbf{h}_S^{(k)}[c_S]$，沿层跨度 $j \to L$ 将旋转后残差 $W^{(k)}\mathbf{h}_S^{(k)}[c_S]$ 注入目标模型同位置，测量 $\Delta \log p(\text{donor})>0$ 的方向性成功率。对照包括 within-lang（天花板）、unprojected（基矢失配）、shuffled（高斯分布控制，匹配均值与全协方差）。

## 实验与结果
- **相关性实验**：SGPT pooling 下最后层 matched CKA 均值达 0.76–0.78，远高于 shuffled 0.15–0.18（$\Delta \approx +0.54$–$+0.61$）。语言句法距离是与 CKA 最强负相关预测因子（$\rho=-0.64, p<.001$），模型平均 NLL 亦显著预测对齐强度（$\rho=-0.43$）。低资源语言（Tagalog/Swahili/Uzbek/Amharic）配对 gap 仍达 +0.39–+0.58，与高资源水平相当。
- **构造性重建**：Procrustes 在 P@1 上全面胜出了 Affine 与 MLP（$\mathrm{P@1_{std}}$: $0.887 \pm 0.021$ vs $0.814 / 0.809$），尽管 MSE 更高（$0.689$ vs $0.439 / 0.456$），归因于正交约束保留角几何利于余弦检索；中间层（L6–L8）检索峰值最高（如 eng→fra L8: P@1=0.981）。残差维度分析表明误差分散于目标模型主轴之外的高维子空间，解释了宽松映射无法提升检索的原因。
