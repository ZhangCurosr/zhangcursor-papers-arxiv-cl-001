---
title: "AraMS-28k-The-Largest-Publicly-Released-Line-Level-Dataset-o"
source: https://arxiv.org/pdf/2608.26921v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 06:45:09"
field: "历史阿拉伯语手写文字识别与手稿数字化"
keywords: ["Arabic HTR", "historical manuscript", "line-level dataset", "insertion anchor", "layout analysis", "cross-script generalisation", "reference-grounded annotation", "diacritic normalisation"]
innovations: ["首次公开发布阿拉伯历史手稿线级插入锚点标注，恢复非顺序阅读关系", "同时发布全注音参考与视觉等效规范化双视图转写以显式化解注音 mismatch", "提出 PV/LV 双质量级别发布策略并给出跨书写泛化梯度基准"]
benchmarks: ["AraMS-28k test split (book_03/05/09)", "in-distribution withheld pages diagnostic (HATFormer 6.48% CER)"]
---

# 论文速读：AraMS-28k-The-Largest-Publicly-Released-Line-Level-Dataset-o

## 一句话总结
本文发布了 AraMS-28k，是目前规模最大的公开阿拉伯历史手稿线级数据集，包含 14 本书、3,043 页、28,600 行标注文本，首创性提供边注与主文本之间的线级插入锚点标注，以恢复手稿的非线性阅读顺序。

## 研究问题与动机
- 历史阿拉伯语手写文字识别（HTR）因缺乏大规模真实手稿线级语料而落后于拉丁文字 HTR，现有公开资源规模小、多为现代手写样本或合成数据。
- 即使来自真实手稿的资源（如 RASM2018、RASAM、Muharaf），也未记录边注在主线文阅读流中的具体归属位置，缺乏细粒度非线性阅读顺序标注。
- 历史阿拉伯手稿的边注通常锚定于主文本的特定位置（纠错、异读、注释），而非附录式阅读；现有数据集最多仅标识边注区域，无法支持阅读顺序恢复任务。
- 参考转写多来自全注音学术版本，而手稿本体通常无注音，导致直接用于训练会产生幻觉和 CER 虚低等评估偏差，现有数据集未显式记录这一 mismatch 并提供解法。

## 核心贡献（创新点）
1. **AraMS-28k 数据集发布**：最大公开阿拉伯历史手稿线级数据集（14 书/3,043 页/28,600 行），覆盖 Naskh、Ruq'ah、Maghrebi 三种书写传统及 1 部石版印刷版；与已有资源（如 Muharaf-public 的 24,495 行）相比规模领先且唯一提供插入锚点。
2. **首次线级插入锚点标注**：每条边注若存在明确附着点，则标注其插入到主文本第几行之后（corpus 中约 30% 边注获锚点）；与仅做页面级区域排序的工作（如 RASM2018、Muharaf）本质区别在于实现了行级非顺序阅读关系的可计算表征。
3. **双重转写发布**：每行同时提供 `gt_raw`（全注音学术参考）与 `gt_normalized`（去注音规范化）；与现有数据集仅给出单视图不同，显式建模并缓解参考文本与图像视觉信息的注音 mismatch。
4. **双阶段质量控制流程**：PV 阶段（548 页）由两名独立审稿人对每行全量复核；LV 阶段（2,495 页）以自动对齐分数=100 为过滤门槛，仅在完全对齐时保留；与纯人工转录或仅靠置信度截断的管道相比，在规模与精度之间给出可复现的权衡方案。
5. **跨书写泛化梯度基准**：在同一预训练权重下比较 Ruq'ah/Naskh/Maghrebi 三套未见书的 CER，揭示"预训练分布邻近度 > 微调数据量"的主导作用，为后续低资源书写系统研究提供可复现的度量目标。

## 方法详解
- **数据来源**：14 部经典医学、伊斯兰法学、逍遥学派哲学与神学著作；13 部手写、1 部石版印刷（book_10）。页图片来自公开在线手稿库（如 alukah.net），仅纳入符合 CC BY-NC-SA 4.0 再分发条款的扫描。
- **参考转写获取**：多数书籍来自数字阿拉伯文仓库；book_27 通过 OCR 对同一文本的印刷校勘版生成参考，作为对齐目标。
- **RefLAM 管道（参考文献 [1]）**：(1) 基于 Muharaf 训练的分割模型对每页作行分割；(2) 多模态大语言模型（MLLM）OCR 并区分主线与边注；(3) 将假设与独立参考对齐，输出每行置信度并标记低置信度行供人工复核。对齐算法与正确性论证见 RefLAM 论文。
- **质量控制两阶段**：
  - PV（page-validated）：548 页，两名审稿人对每行做完整人工验证，保留分割与转写全部误差；代表"完整手工核验"保证。
  - LV（line-validated）：2,495 页，仅当自动对齐得分=100（逐字符与参考一致）才保留；未达标行直接剔除，不补充人工修正；代表"参考一致性过滤后的大规模子集"。
- **注解结构**：每行包含 5 组字段（表 2）：
  - 标识与元数据：`line_uid`、`book_id`、`page_id`、`line_idx`、`line_type ∈ {main, margin}`、`split`。
  - 文本：`gt_raw`（学术全注音）、`gemini_raw`（MLLM 假设）、`gt_normalized`（去 harakat/tatweel、合并 alef/hamza 变体、折叠空白后的规范化形式）、`confidence ∈ [0, 100]`。
  - 几何：`baseline`（基准多段线）、`boundary_polygon`（边界多边形）、`bounding_box`（轴对齐框）；主要/次要字段均存在，不适用范围为 null。
  - 边注锚点：`margin_anchor.before/after`（插入点前后主文本词）、`margin_anchor.line`（主文本行索引）、`margin_anchor.rotation`（粗略旋转角）；主线为 null。
  - 复核：`edited`、`validated`、`deleted`、`page_reviewed`。
- **规范化定义**：参照 RefLAM 的 Definition 1，去除 tashkeel（harakat）与 tatweel（kashida），合并 alef/alef-maqsura/hamza 字形变体，折叠空白，使转写与无注音手稿图像视觉等价。
- **锚点分配策略**：仅在边注具备"明确且可验证"的主文本附着点时赋值；189/629（≈30%）获锚点；其余以 `null` 显式记录，避免强制猜测污染下游阅读顺序评估。
- **分集策略**：按书级切分（9 训 / 2 验 / 3 测），每本书只出现在一个 split；训练集混 PV+LV；验证集为 2 本 PV Naskh；测试集为 3 本 PV（每脚本 1 本），便于按脚本报告 CER。

## 实验与结果
- **基线模型**：Kracken（Muharaf 预训练权重 [15]）、HATFormer [16]，均在 AraMS-28k 训练集上微调，测试集为 book_03（Maghrebi）、book_05（Ruq'ah）、book_09（Naskh）；使用 `gt_normalized` 作为目标以避免注音幻觉。
- **CER 结果**（表 4）：
  - Kraken：Ruq'ah 11.65% / Naskh 22.62% / Maghrebi 32.71%，Overall 23.31%。
  - HATFormer：Ruq'ah 13.26% / Naskh 25.37% / Maghrebi 37.88%，Overall 26.74%。
- **跨脚本泛化梯度**（表 5，HATFormer）：
  - 同书未见过页（in-distribution）：6.48%。
  - 未见 Ruq'ah 书：13.26%（≈×2）。
  - 未见 Naskh 书：25.37%（≈×4）。
  - 未见 Maghrebi 书：37.88%（≈×6）。
- **关键结论**：
  - 性能排序与训练集规模不成比例（Ruq'ah 训练仅 3,282 行但最佳，Maghrebi 训练 5,229 行却最差），排除数据量主导假设。
  - 归因于预训练分布邻近度：Muharaf 以 Ruq'ah 为主，Naskh 居中，Maghrebi 在 Muharaf 中几乎缺失；book_09 的墨水褪色等图像质量问题亦贡献了 Naskh 的偏高 CER。
  - 去掉中间 Muharaf 微调阶段会令 in-distribution CER 从 6.48% 升至 7.44%，证实多阶段迁移收益。
- **最强结果**：HATFormer 在 in-distribution 未见过页达到 6.48% CER；跨脚本最差（Maghrebi 未见书）达 37.88%，提升空间明确。

## 相关工作脉络
- **RASM2018**：历史阿拉伯科学手稿线级转写，仅约 120 页；具备区域级布局标签但无边注插入锚点；AraMS-28k 在其基础上将规模提升 25 倍并引入线级非顺序阅读标注。
- **RASAM**：首个 Maghrebi 专用语料（≈300 页/7,540 行）；AraMS-28k 的 Maghrebi 覆盖比其大一个数量级，并首次在该脚本中提供锚点与双视图转写。
- **KHATT**：4,000 页/13,435 行现代阿拉伯手写；非历史手稿，缺少退化、连字变异与边注等真实手稿现象；AraMS-28k 在史料真实性与布局复杂度上更贴近下游古籍场景。
- **Muharaf-public**：目前最大公开真手稿子集（1,216 页/24,495 行）；仅标识边注区域、不标注插入位置；AraMS-28k 在规模相近的同时补齐线级阅读顺序标注，并以 PV/LV 双质量级别可复现报告。
- **VML-HD / BADAM**：前者近 680 页次词级标注、后者 400 页基线检测标注；任务定位不同（非行级 HTR、非布局-阅读顺序联合）；AraMS-28k 填补行级识别 + 版面分析 + 阅读顺序恢复的综合评测空白。
- **OpenITI MAKHZAN / KITAB-Bench / SARD**：Makhzan 跨语种但无锚点；KITAB-Bench 历史样本极小且无锚点；SARD 为合成数据、不含手稿特有边注；AraMS-28k 在真实手稿 × 锚点标注两个正交维度上与它们形成互补定位。

## 局限性与未来方向
- **LV 子集的偏差**：LV 仅保留对齐得分=100 的行，偏向清洁扫描与规整书体；读者需按需选用 PV（完整人工核验）或 LV（大规模参考一致）。
- **边注锚点覆盖率**：仅 ≈30% 边注获锚点，约 70% 保留 null；这对纯边注检测任务不致命，但对阅读顺序恢复任务提供了高精确度子集与更宽泛子集之间的明确分界。
- **每脚本仅 1 本测试书**：book_03/05/09 各自代表一种脚本，CER 差异可能混杂书级特征（纸张、墨水、年代），难以严格分离"脚本难度"与"书个体效应"。
- **边注占比小**：629/28,600（≈2.2%）；直接使用 AraMS-28k 训练边注检测模型需要重采样或类别权重处理。
- **未来方向**：① 对 Muharaf 预训练分布做逐脚本拆解，控制脚本邻近度变量；② 设计并评测基于 `margin_anchor` 的阅读顺序恢复协议；③ 结合 real_damage_lines.txt（177 条自然退化行）开展退化鲁棒性与修复研究；④ 扩展至更多脚本/语种并与现有阿拉伯语 OCR 基准（KITAB-Bench、SARD）进行桥接评测。

## 研究启发与可借鉴点
1. **参考对齐驱动的半自动标注流水线**：以独立来源参考文本为"真值锚"，用对齐得分=100 作为自动验收门槛，可在保持规模的同时显著降低人工转录成本；该方法可直接迁移到其他书写系统的历史文献规模化构建。
2. **双视图转写设计**：在 HTR 场景中同时发布"原始学术转写"和"视觉等效规范化转写"，可将注音幻觉风险显式化为可选任务（基础识别 vs. 注音还原），为下游任务适配提供统一数据底座。
3. **质量级别显式分层**：以 PV（完整人工）和 LV（自动严格过滤）两种保证层级发布同一语料的不同子集，使不同精度预算的用户各取所需；该策略适合大规模人文计算语料建设。
4. **插入锚点的 ternary 语义**：present / confidently null / absent 三态设计避免了"强制猜测"带来的隐性评估污染，为阅读顺序恢复任务提供高可信子集；该标注范式可推广到其他非线性文档（批注本、评注经卷）。
5. **跨分布泛化梯度作为基准诊断**：在同预训练权下比较 in-distribution/unseen script 的 CER 变化曲线，比单一总分更能揭示数据-模型匹配瓶颈；建议在新数据集发布时标配此类诊断设置。

## 关键术语表
- **AraMS-28k**：本文发布的历史阿拉伯手稿线级数据集，14 书/3,043 页/28,600 行，含边注插入锚点与双视图转写。
- **Insertion anchor（插入锚点）**：标注某条边注应插入主线文哪一行之后的信息，字段包括 `line`（主文本行索引）与 `before/after`（插入点前后词）。
- **RefLAM**：参考接地（reference-grounded）的行级标注管道，结合线分割、MLLM OCR 与外部参考对齐，再通过人工复核发布。
- **Page-validated (PV) / Line-validated (LV)**：PV 由双人全量复核每一行；LV 仅保留自动对齐得分=100 的行，未达阈者直接剔除。
- **gt_raw / gt_normalized**：前者为学术全注音参考转写；后者经去 harakat/tatweel、合并 alef 变体、折叠空白后与无注音手稿图像视觉等价，推荐用于 HTR 训练。
- **Tashkeel (harakat)**：阿拉伯语短元音符号；手稿常省略而学术转写保留，造成参考-图像 mismatch。
- **Kashida (tatweel)**：阿拉伯语拉长符；规范化时折叠为单一空白。
- **CER (Character Error Rate)**：字符级错误率，衡量识别输出的逐字符编辑距离。

## 可复现要素
- **数据集**：AraMS-28k（完整注解）与 AraMS-28k-HTR（训练就绪裁剪版）均已公开；CC BY-NC-SA 4.0。
- **代码/权重**：细调代码与最终解码设置发布在项目仓库中；使用的是已有的 Muharaf 预训练 Kraken checkpoint [15] 与 HATFormer [16]。
- **关键超参**：
  - HATFormer：AdamW（β1=0.9, β2=0.999, ε=1e-6, wd=0.01）；lr=5e-5；batch=8；warmup 500 线性步；梯度裁剪 5.0；最多 20,000 步、每 1,000 步保存、按验证 CER 挑选。
  - Kraken：Muharaf 预训练初始化；首 epoch 冻结 backbone（2,468 步 / batch=8）配合同长度线性 warmup；lr=5e-4 cosine decay；batch=8；早停 patience=15，上限 100 epoch；seed=42。
- **复现材料**：固定 train/val/test 分集文件、`test_seen_pages.txt`（in-distribution 诊断用）、`real_damage_lines.txt`（退化行列表）、SHA-256 校验和。
- **托管**：Hugging Face Datasets 与 Zenodo DOI。
- **未提及**：更多训练细节（如图像增强具体策略、序列长度截断规则）见 Appendix B.1 与项目仓库；论文正文中未单独列出。
