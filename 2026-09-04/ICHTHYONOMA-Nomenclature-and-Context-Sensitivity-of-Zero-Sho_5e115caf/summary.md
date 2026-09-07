---
title: "ICHTHYONOMA-Nomenclature-and-Context-Sensitivity-of-Zero-Sho"
source: https://arxiv.org/pdf/2609.03985v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:03:43"
field: "生物视觉-语言模型评估"
keywords: ["zero-shot vision-language model", "BioCLIP", "nomenclature sensitivity", "multilingual VLM", "context robustness", "freshwater fish recognition", "Bangladeshi fish"]
innovations: ["跨模型跨源零样本基准审计生物VLM对命名法/语言/上下文的敏感性", "引入Jina CLIP v2作为多语言诊断控制分离对齐与专业化误差", "阶梯式配对上下文干预揭示物种依赖的上下文鲁棒性"]
benchmarks: ["BFF-15", "SylFishBD"]
---

# 论文速读：ICHTHYONOMA-Nomenclature-and-Context-Sensitivity-of-Zero-Sho

## 一句话总结
论文系统审计了CLIP、BioCLIP、BioCLIP2及Jina CLIP v2在孟加拉淡水鱼零样本识别中的表现，发现零样本准确率不仅反映生物特异性视觉知识，还强烈依赖于命名法选择、提示模板、语言对齐和视觉上下文。

## 研究问题与动机
- 零样本生物VLM的准确率是否稳定？当模型、语言、命名法、提示或视觉上下文变化时，识别性能如何波动？
- 现有方法多为监督/自监督微调获得的高域内准确率，但未建立"训练-free"的可迁移生物知识；且单一Prompt下的最优报告可能夸大本地可及性。
- 生物学存在多语言名称（俗语、英语通用名、拉丁学名、历史同义词），科学双名法并非总是最优文本接口。
- 背景相关性和视觉捷径可能被VLM利用，单一合成变换不足以因果分解上下文效应，需阶梯式干预。

## 核心贡献（创新点）
- 跨模型、跨源零样本基准：对CLIP/BioCLIP/BioCLIP2在7类孟加拉淡水鱼的BFF-15和SylFishBD上进行系统评估，揭示了生物专业化VLM相对通用CLIP的巨大优势。
- 命名法审计框架：首次将罗马化、英语通用名、科学名、孟加拉文字、提示模板和科学同义词作为显式评测变量，而非固定预处理；证明"科学提示"必须指定精确字符串/模板。
- 引入多语言Jina CLIP v2作为诊断控制：分离"多语言对齐"与"生物细粒度 specialization"两个维度，揭示BioCLIP2孟加拉语接近随机源于文本对齐而非孟加拉命名本身。
- 配对上下文干预阶梯：通过弱/中/强模糊、灰/白/均值掩码、裁剪、前景去除和跨物种背景交换，发现上下文敏感性是干预类型和物种依赖的，而非均匀背景效应。

## 方法详解
- **数据集与规范**：使用BFF-15（2,656张）和SylFishBD（7,665张）共享的7个类别（Rui/Katla/Mrigal/Tilapia/Pabda/Ilish/Koi），总10,321张；SHA-256无完全重复，perceptual-hash找到73对低距离对仅作标记不自动排除。
- **冻结零样本分类**：实例化4个模板（"a photo/image/photograph/specimen of $n_c$. a fish species"），对每个类别构建归一化文本原型：$t_c = \frac{\frac{1}{K}\sum_{k=1}^K \hat{f}_T(p_k(n_c))}{\|\frac{1}{K}\sum_{k=1}^K \hat{f}_T(p_k(n_c))\|_2}$，$K=4$；预测为$\hat{y}=\arg\max_c f_I(x)^\top t_c$。缓存图像嵌入使命名实验仅改变文本侧分类器。
- **多语言对照**：Jina CLIP v2使用相同图像/类别，采用孟加/罗马/英语/科学标签与4模板集成，使用512维Matryoshka截断表示，无benchmark调优；另设"纯名称"对照（仅嵌入孟加类名）测试是否多语言支持本身即足够。
- **配对上下文干预**：SylFishBD保留前景像素并修改背景；应用Gaussian blur（$\sigma=0.01/0.03/0.06\times$短边）、白/灰/均值掩码、10%扩展紧裁剪、Telea inpainting前景去除、跨物种背景交换；以阶梯干预而非单一变换评估稳定性。
- **统计评估**：准确率、平衡准确率、macro-F1、per-class效应；类分层bootstrap CI（主基准1,000重采样，鲁棒/多语言2,000重）；配对比较用McNemar检验+bootstrap差值+Benjamini-Hochberg FDR校正；Cochran's Q检验上下文条件；donor-follow用5,000次标签置换；seed=42。

## 实验与结果
- **最强结果**：BioCLIP2 + 英语通用提示在BFF-15达72.36%准确率/67.33% macro-F1；在SylFishBD达64.59%/64.30%。相对BioCLIP，BioCLIP2科学提示分别提升22.18/10.03个百分点（配对95% CI [19.73,24.66]/[8.91,11.17]，$q<0.001$）。
- **命名法差异**：相同BioCLIP2图像嵌入下，BFF-15 Romanized/English/Scientific分别为35.77%/72.36%/69.99%；SylFishBD为37.56%/64.59%/68.91%，证明科学术语非普适最优。
- **孟加拉语崩溃**：BioCLIP2孟加提示准确率仅16.27%/7.74%，平衡准确率14.22%/14.29%（七分类随机基线），纯孟加类名同样坍缩至14.29%。
- **Jina部分恢复**：Jina CLIP v2孟加平衡准确率21.89%（BFF-15）/16.36%（SylFishBD），显著高于BioCLIP2但仍远低于其英语/科学表现（Jina英语29.41%/23.78% vs BioCLIP2 72.36%/64.59%），说明多语言对齐无法替代生物专业化。
- **同义词敏感**：Catla→Gibelion catla提升SylFishBD 5.14pp（$q<0.001$），但→Labeo catla降低3.46pp（SylFishBD）/8.58pp（BFF-15）；Cirrhinus cirrhosus→C. mrigala改善约2pp两源。
- **上下文干预阶梯（SylFishBD BioCLIP2）**：原始68.91%；弱模糊−0.47pp（$q=0.347$，NS）；中模糊−2.18pp；强模糊−2.87pp；灰/均值掩码≈−3.8pp；白掩码−8.39pp（最大分布偏移）。背景仅视图保留17.90%（平衡21.48%，−51.01pp）；跨物种交换降至56.76%（−12.15pp），donor-follow仅8.88% vs 置换零15.10%；紧裁剪55.15%。
- **类条件异质性**：强模糊改善Katla(+8.91pp)/Mrigal(+8.66pp)，但损害Rui(−18.38pp)，Ilish几乎不变(−0.51pp)。

## 相关工作脉络
- **CoOp/CoCoOp**（Zhou et al., 2022）：证明修改prompt上下文可大幅改变下游识别行为，本文将其推广到生物学精细分类的命名/模板变量。
- **BioCLIP/BioCLIP2**（Stevens et al., 2024; Gu et al., 2025）：生物结构化对比预训练，本文证明其冻结状态仍对命名/语言/上下文高度敏感，不能仅报告单一最优。
- **Parashar et al. (2023)**：发现英语通用名常优于科学名用于零样本识别，本文在其基础上进一步证明"科学提示"仍需精确字符串/模板。
- **Babel-ImageNet**（Geigle et al., 2024）与**uCLIP**（Chung et al., 2026）：展示多语言VLM跨语言性能差异与参数高效扩展，本文用Jina作诊断控制分离"多语言对齐"与"生物专业化"。
- **CLIP-FSSC**（Dai et al., 2024）与**TaxaBind**（Sastry et al., 2025）：前者用自然语言监督实现鱼虾可迁移分类，后者构建统一生态表示；本文定位差异在于保持权重与提示冻结，评估零样本稳定性。
- **背景偏差研究**（Xiao et al., 2021; Geirhos et al., 2020; Bassi et al., 2024）：证明背景可被作为捷径，本文用阶梯干预而非单一掩码评估上下文鲁棒性。

## 局限性与未来方向
- 仅7个类别/2个数据源，无法代表更广泛的生物多样性与地理覆盖。
- 未知预训练对近缘图像的曝光，可能混淆专业化与记忆效应。
- 单一多语言对照（Jina CLIP v2，512维截断表示）不足以建立多语言VLM的普遍性质。
- 孟加拉语结果仅针对测试的名称/模板，同义词测试仅限Katla/Mrigal。
- 上下文干预既改变信息内容也改变图像分布，因果分解受限；需更多跨物种、跨地域、跨语言的系统性审计。
- 未来方向：扩展至更多区域鱼类与语言、开发兼顾生物专业化与多语言对齐的VLM、标准化零样本生物报告的prompt/命名/上下文多样性基线。

## 研究启发与可借鉴点
- **多语言诊断控制范式**：将多语言VLM（如Jina）作为生物专业化模型的对照，可分离"文本对齐缺陷"与"生物学 specialization 不足"两类误差源，适用于其他低资源语言生物识别任务。
- **命名法敏感度审计清单**：将Romanized/English/Scientific/本地语言、精确学名变体、提示模板纳入标准评测，避免仅报告单一最优prompt导致的可及性夸大。
- **阶梯式上下文干预设计**：用模糊强度梯度+多种掩码+裁剪+背景交换+前景去除的组合，比单一增强更稳健地评估模型对视觉捷径的依赖，可迁移至任何fine-grained视觉分类任务。
- **类条件效应报告**：平均指标掩盖物种异质性（如强模糊对Katla有利但对Rui有害），未来工作应报告per-class Δ并做配对显著性检验。
- **结合本团队方向**：若团队关注湿地/渔业/生态监测的零样本识别，可直接复用此基准框架（BFF-15/SylFishBD+7类）扩展至其他区域物种，并引入本地语言prompt审计。

## 关键术语表
- **Zero-shot VLM**：冻结预训练视觉-语言模型，无需下游标注，直接通过文本原型进行图像分类。
- **BioCLIP/BioCLIP2**：基于生物结构化对比预训练的视觉基础模型，扩展了物种/分类学覆盖以支持细粒度生物识别。
- **Jina CLIP v2**：多语言CLIP变体，提供跨语言文本-图像对齐，本文用作分离多语言对齐与生物专业化的诊断控制。
- **Nomenclature sensitivity**：同一生物的不同命名法（罗马化/英语/科学名/本地语）显著影响零样本准确率，证明文本接口即分类器本身。
- **Paired context intervention**：在成对图像上施加模糊/掩码/裁剪/背景交换等变换，以配对检验评估视觉上下文对预测稳定性的影响。
- **Balanced accuracy**：各类别召回率的均值，缓解类别不平衡下的准确率虚高问题。
- **Donor-follow**：评估背景交换后模型是否跟随源物种背景分布的置换检验指标。
- **FDR correction**：Benjamini-Hochberg False Discovery Rate校正，控制多重比较下的假阳性率。

## 可复现要素
- 数据集：BFF-15（Kaggle, TheShahidul, 2026）与SylFishBD（Absar et al., 2026, Scientific Data）均为公开；SHA-256无完全重复，73对perceptual-hash低距离对仅作标记。
- 代码与权重：所有代码和数据开源GitHub: https://github.com/NazimRiyadh/IchthyoNoma；模型使用冻结checkpoint（CLIP ViT-B/32, BioCLIP, BioCLIP2, Jina CLIP v2）。
- 关键超参：4模板集成（K=4）、blur σ比例0.01/0.03/0.06×短边、10%扩展紧裁剪、512维Matryoshka截断（Jina）、bootstrap重采样1,000/2,000次、5,000次donor-follow置换、seed=42。
