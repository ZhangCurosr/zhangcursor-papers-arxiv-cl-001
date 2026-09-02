---
title: "Cross-lingual-Biography-Enrichment-via-Claim-Extraction-and"
source: https://arxiv.org/pdf/2608.23390v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-02 19:48:00"
---

# 论文速读：Cross-lingual-Biography-Enrichment-via-Claim-Extraction-and

## 一句话总结
本文提出跨语言传记富集框架，将非英语维基百科传记映射至共享英文 claim 空间后进行显式对齐与筛选，再以结构化声明作为可控证据驱动大模型重写英语传记，在有效补充事实的同时显著抑制幻觉。

## 研究问题与动机
- **任务定义**：跨语言传记富集（cross-lingual biography enrichment）——以同一位女性的英文维基百科传记为目标，利用非英文维基百科传记作为补充证据进行事实注入与重写。
- **人群聚焦**：针对非英语语境女性在英文维基中覆盖不足的现状，挖掘法/中/阿塞拜疆语等非英版本中更丰富的事实描述。
- **现有方法不足**：直接拼接原始非英文本或依赖机器翻译注入证据易引入冗余、翻译误差与幻觉；缺乏在原子事实层面的显式对齐与选择性整合机制。
- **核心主张**：将双语传记映射到共享英文 claim 空间后进行显式对齐与筛选，可在“支持性新增（supported additions）”与“幻觉
