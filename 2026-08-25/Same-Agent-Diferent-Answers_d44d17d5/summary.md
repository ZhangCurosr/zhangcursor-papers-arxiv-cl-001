---
title: "Same-Agent-Diferent-Answers"
source: https://arxiv.org/pdf/2608.22856v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:01:17"
---

# 论文速读：Same-Agent-Diferent-Answers

## 一句话总结
本文针对检索增强问答（RAG）系统在仅扩展底层语料库时发生的“准确率盲区答案 churn”，提出了一种扣除 LLM 随机噪声基线的 Snapshot Compatibility Audit 协议，并在预注册的 Natural Questions 与 TriviaQA 实验中证实：即便聚合精确匹配（EM）变化微弱甚至方向相反，跨快照答案分布仍会发生显著且不可忽略的行为漂移。

## 研究问题与动机
- **核心问题**：在模型、Prompt、检索策略、证据深度与生成控制完全固定的前提下，仅扩大可访问语料规模（如索引增加分片）是否会导致系统答案发生行为层面的不一致？
- **聚合指标盲区**：现有 RAG 缩放研究多依赖平均效用（EM/F1）评估，但不同题目的提升与回退会相互抵消，隐藏了单题级别的答案翻转。
- **随机性混淆**：LLM 生成具有内在随机性，单次（one-shot）前后对比极易将正常解码方差误判为语料更新的影响，缺乏扣除同态噪声的基线。
- **工程动机**：生产环境中索引重建、文档刷新与缓存失效频繁发生，答案常流入下游工作流或自动化测试；若仅依赖 dashboard 上的准确率绿码，可能忽视已发生的底层行为不兼容。

## 核心贡献（创新点）
- **提出 Snapshot Compatibility Audit 协议**。通过在每个快照下独立重复采样两次回答，显式建模并扣除 LLM 解码随机噪声，使跨状态行为差异具备统计推断基础。
- **识别准确率盲区答案 churn（accuracy-blind answer churn）**。证明语料扩展可在 EM 变化仅 −1.50 pp 甚至反向时引发显著的单题答案分布漂移，揭示单一工具指标的盲区。
- **构建严格的 repeat-stable 语义翻转诊断与跨家族盲审流程**。区别于依赖表面字符串匹配或单一 judge 的工作，本文在锁定答案后才加载 gold，并结合 outcome-blind 复现与不同模型家族的 judge 交叉验证，排除格式归一化与 judge 自偏好偏差。

## 方法详解
- **快照定义与对照设计**：固定模型标识、prompt、检索策略、证据深度与渲染格式，仅改变可访问的语料前缀规模（FineWeb 嵌套前缀 n0⊂n1⊂n3⊂n7）。每个问题在低规模快照 $L$ 与高规模快照 $H$ 下分别独立调用两次生成器，共获得 8 条回答。
- **超额 churn 估计量**：定义相似度核 $k \in [0,1]$（归一化精确匹配与盲态语义评判）。对第 $i$ 题计算同快照平均相似度 $w_i = \frac{1}{2}[k(L_{i,a}, L_{i,b}) + k(H_{i,a}, H_{i,b})]$ 与跨快照平均相似度 $c_i = \frac{1}{4}\sum_{r,s \in \{a,b\}} k(L_{i,r}, H_{i,s})$，超额 churn $\widehat{D} = \frac{1}{N}\sum_{i=1}^N (w_i - c_i)$。$\widehat{D}>0$ 表示跨快照漂移超出同态重复变异基线，等价于交叉态分歧减去同态噪声 floor。
- **统计推断与决策门**：采用 50,000 次整题重采样（whole-question bootstrap）估计抽样分布；预注册 NQ 门禁要求归一化精确 $\widehat{D} \geq 3$ pp，且 exact 与 semantic 的单侧 95% 置信下界（LCB）均为正。
- **后验严格翻转诊断**：锁定主分析后定义 repeat-stable semantic flip：同快照两次回答语义完全一致，且所有跨快照四对回答两两语义相异，用于定位 churn 的核心贡献子集。
- **盲审与跨家族验证**：语义 judge 仅见问题与 8 条匿名答案，隐去尺度、证据、gold 与正确性；主分析后抽取 100 题进行 outcome-blind V4-Pro 复现，并引入 OpenAI gpt-5.6-sol 独立评判 1,400 对标签以排除单一 judge 家族偏差。

## 实验与结果
- **数据集与设置**：预注册 400 题 NQ（排除 2,400 历史开发 ID，SHA-256 哈希选取）、200 题 TriviaQA RC-Web；FineWeb 7 shard 嵌套前缀；检索 top-8，单文档渲染上限 1,200 字符；生成器 DeepSeek-v4-flash（singleton 模式，工具关闭）。
- **NQ 主实验（n1→n7）**：归一化精确 $\widehat{D}$ = 6.44 pp（LCB=4.56 pp），语义 $\widehat{D}$ = 10.25 pp（LCB=7.69 pp），EM 仅变化 −1.50 pp。通过全部三项预注册门禁。
- **TriviaQA 支持性实验**：exact $\widehat{D}$ = 3.00 pp（LCB=0.50 pp），semantic $\widehat{D}$ = 2.125
