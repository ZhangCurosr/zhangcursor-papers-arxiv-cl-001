---
title: "PersianAnonymizer-Evaluating-LLM-Labeled-Training-for-Eficie"
source: https://arxiv.org/pdf/2609.00958v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 09:55:03"
---

# 论文速读：PersianAnonymizer-Evaluating-LLM-Labeled-Training-for-Eficie

## 一句话总结
本文系统评估了三种指令微调大语言模型（LLM）作为自动标注器，为波斯语工业客服聊天数据训练轻量级NER匿名化模型的效果。实验表明，基于GPT-OSS-120B零样本标注训练的MATINAROBERTA模型在宏观F1与实体覆盖率（LCR）上表现最佳，且能在单张RTX 3090上于2分钟内完成4万条测试消息的推理标注，为低资源语言场景下高隐私合规、低成本的文本匿名化提供了可行路径。

## 研究问题与动机
- **核心问题**：如何在低成本、低延迟约束下，为波斯语客户聊天数据构建可靠的命名实体识别（NER）系统，以实现PII/PHI的自动化匿名化？
- **现有方法不足**：
  1. 直接调用闭源/超大LLM进行匿名化存在高昂的多卡算力成本、推理延迟及数据外泄合规风险（如GDPR）；
  2. 波斯语NER资源相对稀缺，且既有语料多聚焦通用领域，缺乏针对工业客服聊天场景的PII专用标签与系统评估；
  3. “LLM-as-labeler”范式虽可降低人工标注开销，但此前缺乏对不同指令微调模型标注质量及其对下游小模型可学习性的横向对比，尤其针对波斯语匿名化场景的研究尚属空白。

## 核心贡献（创新点）
- **多LLM标注器的系统选型对比**：在统一JSON协议与提示模板下对比DeepSeek-V3、GPT-OSS-120B、Qwen3-235B三种大模型的标注产出，填补了低资源语言匿名化场景下“哪类LLM监督信号最易被下游模型学习”的研究空白。
- **提出LCR评估指标**：引入Label Coverage Recall（LCR）量化模型对金标准非O实体的整体覆盖比例，弥补传统Macro-F1在类别极度倾斜、关注漏标风险的匿名化场景中的评估盲区。
- **Token级一致性可视化分析**：通过三向Token级Venn图揭示不同LLM标注器的覆盖偏好与边界分歧，为后续一致性感知标签融合提供实证依据。
- **端侧高效部署验证**：证明基于最优LLM监督数据训练的轻量级MATINAROBERTA可在单张消费级GPU（RTX 3090）上以约2分钟处理4万条消息的延迟完成生产级匿名化，实现精度与部署成本的有效平衡。

## 方法详解
- **语料构建**：收集26.5万条波斯语组织聊天消息（22.5万训练/4万测试），每条为独立用户发言，富含PII（姓名、电话、邮箱、URL、支付标识等）。
- **标签体系**：采用BIO标注方案，涵盖14类PII导向实体：COST, CREDIT_CARD, DATETIME, EMAIL, IBAN, IP_ADDRESS, LOCATION, NUMBER, ORGANIZATION, PASSWORD, PERSON, PHONENUMBER, URL, USERNAME。
- **LLM标注流水线**：使用统一提示词定义Schema与输出规范，强制返回JSON格式的`phrase`与`ner_type`。根据JSON解析稳定性筛选出四套可用数据集：OSS_ZeroShot、Qwen_ZeroShot、Qwen_FewShot、DeepSeek_FewShot。
- **Span-to-Token对齐与修复**：通过确定性正则表达式将字符级Span映射至子词Token边界，处理波斯语连字、标点粘连与Clitics现象；对约5%未精确匹配的生成短语，采用相似度≥60%的模糊子串匹配策略恢复，总对齐成功率超99%。
- **NER训练设置**：骨干网络采用基于XLM-RoBERTa Large微调的MATINAROBERTA，顶层附加线性分类头预测BIO标签。四组实验共享超参：AdamW优化器、学习率2×10⁻⁵（线性衰减+固定预热步数）、FP16混合精度、最大6轮早停（以验证集Macro-F1为监控指标），单卡RTX 3090（24GB）约4小时完成训练。
- **评估协议**：报告Token级Precision/Recall/F1（宏观与类别粒度）、LCR，并在PEYMA基准上进行跨数据集验证（映射共享标签COST/DATETIME/LOCATION/ORGANIZATION/PERSON，忽略BIO前缀）。

## 实验与结果
- **主实验结果**：NER_OSS（OSS_ZeroShot监督）取得最优性能，Macro P=0.849、R=0.858、F1=0.851，LCR=90.04%。两个Qwen变体分列次席（F1≈0.76），DeepSeek_FewShot最差（F1=0.733，LCR=80.04%）。
- **类别分析**：URL、IP_ADDRESS、EMAIL、PHONENUMBER等高频率技术标识符在所有模型上表现稳健；LOCATION与ORGANIZATION因非正式波斯语中机构与地名边界交织，仍是难点。CREDIT_CARD与IBAN样本极少（各约数十例），F1波动大，但对目标工业场景整体影响有限。
- **跨数据集验证**：在PEYMA共享标签上，ParsBERT-NER整体领先（Macro F1 0.932 vs 0.836），但NER_OSS在COST类别上召回与F1更优（0.947 vs 0.837）；在OSS测试集上，NER_OSS全面压制ParsBERT-NER，DATETIME（F1 0.914 vs 0.141）
