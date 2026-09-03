---
title: "BROWSERFORGE-SCALING-WEB-EPISODE-VIA-PARALLEL-BROWSER-SANDBO"
source: https://arxiv.org/pdf/2608.24848v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 05:24:41"
field: "Web Agent 数据合成"
keywords: ["Web Agent", "GUI Agent", "Data Synthesis", "Multimodal LLM", "Parallel Sandboxes", "Common Crawl"]
innovations: ["通过并行浏览器沙箱在开放网络上大规模生成 Web 交互轨迹", "Proposer-Solver 双智能体循环将无监督页面转化为可执行任务", "规则+模型清洗流水线与统一 CoT 重写提升数据质量"]
benchmarks: ["Online-Mind2Web", "Multimodal-Mind2Web"]
---

# 论文速读：BROWSERFORGE-SCALING-WEB-EPISODE-VIA-PARALLEL-BROWSER-SANDBOXES

## 一句话总结
论文提出了 BrowserForge，一个通过并行浏览器沙箱在开放网络上大规模生成纯 GUI Web Agent 训练数据的框架，解决了现有 Web 交互轨迹数据集规模小、网站覆盖窄的问题。该框架生成 203,238 条来自不同网站的交互轨迹，微调紧凑型多模态模型后，在在线基准 Online-Mind2Web 上的成功率从 25.66% 提升至 33.33%。

## 研究问题与动机
- **纯 GUI Web Agent 的数据瓶颈**：基于截图的 Web Agent 避免了 HTML/Accessibility Tree 的脆弱性和高 Token 成本，但训练需要大规模高质量交互轨迹，如何规模化生产仍是开放问题。
- **现有数据集规模与多样性不足**：公开数据集通常仅含几千条轨迹，来自几十到几百个固定网站；自动合成管线也受限于预定义网站列表或教程来源，难以扩展网站多样性。
- **开放网络作为数据源的潜力**：开放网络本身是最大的网站来源，浏览器沙箱是低成本的可回收交互单元，并行运行可同时扩展规模与多样性。
- **无监督页面转化为可训练数据的挑战**：开放网络页面没有标注任务，需要自动生成可执行任务并收集经验证的轨迹，同时统一推理格式以利于模型学习。

## 核心贡献（创新点）
- **并行沙箱驱动的开放网络数据生成框架**：BrowserForge 通过编排数百个并行浏览器沙箱在开放网络上生成轨迹，将数据规模与多样性从固定网站列表中解耦；与已有工作（如 Explorer、AgentTrek）基于教程或种子列表不同，本文直接面向 Common Crawl 中的真实 URL。
- **Proposer–Solver 双智能体任务合成循环**：Proposer 将渲染页面转化为可执行任务，Solver 执行任务并记录轨迹；引入反射与选择机制过滤不可执行任务，提高了可用轨迹的产量。
- **规则+模型的轨迹清洗与统一推理格式重写**：通过规则过滤（终止动作检查）和模型判定（任务完成验证）两层清洗，再用 Seed 2.0 Pro 将推理重写为统一的链式思维格式，提升了训练数据质量。
- **大规模多样化轨迹语料库**：构建了包含 203,238 条轨迹的数据集，每条来自不同网站，规模与多样性均超越现有数据集。

## 方法详解

### 1. 开放网络 URL 来源与清洗
- 从 Common Crawl 抽取候选 URL，经过四层清洗：
  - **可达性与类型过滤**：丢弃无法解析或加载的 URL，保留支持交互的页面类型
  - **内容过滤**：移除内容过短或近乎空的页面
  - **黑名单过滤**：排除评测基准或敏感主机
  - **IP 级可访问性验证**：从沙箱集群的实际 IP 重新验证 URL 可达性
- 对每个存活 URL，使用多种屏幕分辨率、User Agent 和界面语言配置沙箱，增强视觉分布多样性

### 2. 并行浏览器沙箱编排
- **共享工作队列**：worker 线程从队列中主动拉取 URL，避免固定分区导致的空闲等待
- **沙箱池管理**：每个沙箱通过 Chrome DevTools Protocol (CDP) 独立寻址，单节点可运行多个浏览器配置
- **动态扩缩容**：计算节点可在运行时添加，队列自动重新平衡
- 支持最多 300 个并行浏览器沙箱

### 3. Proposer–Solver 任务合成
- **Proposer**：使用 Qwen3-VL-235B，分两阶段工作
  - 提案阶段：生成三个候选任务，明确要求排除需要注册/登录/支付的任务
  - 反思选择阶段：模型重新审视候选任务，选择最可执行的一个
- **Solver**：基于开源浏览器控制智能体构建，通过计划-行动-反思-验证闭环执行任务
  - 统一动作空间（Table 1）：包含 click、type、select、scroll、key、wait、go_back、visit_url、finish 等动作
  - 坐标采用归一化整数 [0, 1000]，与渲染分辨率无关
  - 维护步骤历史记录，允许中间错误恢复

### 4. 轨迹清洗与统一推理格式
- **规则过滤**：检查最终动作是否为终止 Finish 动作，丢弃未正常终止或动作序列异常的轨迹
- **模型判定**：使用 Qwen3-VL-235B 结合任务描述和最后三张截图判断任务是否真正完成，约 30% 原始步骤通过验证
- **统一 CoT 重写**：用 Seed 2.0 Pro 将 200K 步的推理重写为一致的链式思维格式，形成最终训练语料

## 实验与结果

### 数据集与评估基准
- **Online-Mind2Web**：在 300 个真实网站任务上评估端到端任务成功率，使用 WebJudge-7B 自动评分
- **Multimodal-Mind2Web**：在固定轨迹协议下评估逐步精度，包含 Cross-Task、Cross-Website、Cross-Domain 三个划分

### 主要结果
| 模型 | Online-Mind2Web SR (%) | 提升幅度 |
|------|------------------------|----------|
| Qwen3.5-4B (baseline) | 25.66 | - |
| **BrowserForge-4B** | **33.33** | **+7.67** |
| Qwen3.5-9B (baseline) | 29.33 | - |
| **BrowserForge-9B** | **38.00** | **+9.33** |

- **Multimodal-Mind2Web 逐步精度**：BrowserForge-4B 平均 Pass@1 从 38.2% 提升至 43.8%，Pass@4 从 44.1% 提升至 54.4%；BrowserForge-9B 平均 Pass@1 从 40.0% 提升至 45.1%
- **对比开源方法**：BrowserForge-4B (33.3%) 超越了 ScaleCUA-7B (30.3%) 和 GUI-Libra-4B (31.3%)；BrowserForge-9B (38.0%) 超越了 GUI-Libra-8B (36.7%)，且仅使用监督微调而非强化学习
- **对比专有模型**：BrowserForge-9B 超过了 GPT-5 + UGround (33.3%) 和 Browser-Use (30.0%)

### 消融与对照实验
- **数据源对照**：相同模型架构和训练预算下，BrowserForge 数据相比其他开源轨迹数据带来约 9% 的绝对提升（Table 5），证明增益来自数据源和清洗而非训练配方
- **规模缩放**：性能随数据量单调增长，呈近似幂律曲线，在 200K 样本处未显示饱和迹象
- **清洗流程消融**（Figure 2b）：
  - 无清洗：24.0%（最弱）
  - 仅规则过滤：27.0%
  - 仅模型判定：29.0%
  - 无统一 CoT：31.0%
  - 完整流程：33.3%（最强）

## 相关工作脉络
- **AgentTrek (Xu et al., 2025)** 和 **Synatra (Ou et al., 2024)**：从教程等间接知识源合成轨迹，但受限于教程语料覆盖的网站范围；BrowserForge 直接从开放网络获取 URL，突破了固定数据源的限制
- **NNetNav (Murty et al., 2024)** 和 **OpenWebVoyager (He et al., 2024b)**：通过交互探索生成轨迹，但仍锚定于特定环境或种子列表；BrowserForge 使用 Common Crawl 作为无限来源
- **Explorer (Pahuja et al., 2025)**：自底向上多智能体探索，生成 94K 轨迹覆盖 49K 网站；BrowserForge 同样面向开放网络，但通过 Proposer-Solver 循环确保任务可执行性，并强调清洗流程
- **SeeAct (Zheng et al., 2024a)** 和 **WebVoyager (He et al., 2024a)**：端到端 Web Agent 方法；BrowserForge 提供数据，与这些模型正交可组合
- **GUI-Libra (Yang et al., 2026)** 和 **UI-TARS (Qin et al., 2025)**：使用强化学习训练 GUI Agent；BrowserForge 仅用监督微调即达到竞争水平，证明数据质量的重要性
- **OS-Genesis (Sun et al., 2025)** 和 **Learn-by-interact (Su et al., 2025)**：反向任务合成和文档驱动交互合成；与 BrowserForge 的核心差异在于数据来源——开放网络 vs. 教程/文档

## 局限性与未来方向
- **训练数据规模仍有提升空间**：性能缩放曲线未饱和，表明更多数据可能继续带来收益
- **失败模式集中于重复行为**：71% 的失败轨迹属于重复点击、无目的后退或滚动，反映智能体缺乏有效的反馈信号和策略切换能力
- **访问限制问题**：部分失败源于 Cloudflare 拦截或 CAPTCHA，当前智能体无法处理此类外部障碍
- **清洗依赖大模型**：模型判定和推理重写阶段使用 Qwen3-VL-235B 和 Seed 2.0 Pro，计算成本较高
- **步骤预算限制**：30 步上限导致部分有效任务因步数耗尽而失败，指标可能低估实际能力

## 研究启发与可借鉴点
- **开放网络作为数据源**：Common Crawl 作为 URL 来源的思路可迁移至其他交互领域（如移动端 GUI、桌面应用），解决数据多样性瓶颈
- **Proposer-Solver 双阶段设计**：先生成候选任务再筛选，比单次生成更有效；可借鉴用于其他需要任务合成的场景
- **清洗流水线设计**：规则过滤（低成本快速筛选）+ 模型判定（高准确性验证）的组合策略值得推广，两者互补且各有侧重
- **统一推理格式重写**：将异构推理重写为一致格式（CoT）的做法，可有效提升训练数据的监督信号一致性
- **并行沙箱编排模式**：共享工作队列 + 动态分配的策略实现了高吞吐量，可为其他需要大规模并发执行的任务提供架构参考

## 关键术语表
- **Pure-GUI Agent**：仅依赖页面截图（而非 HTML/Accessibility Tree）进行感知和决策的 Web Agent
- **Proposer-Solver Loop**：由 Proposer 生成可执行任务、Solver 执行并记录轨迹的双智能体合成循环
- **Accessibility (a11y) Tree**：页面的无障碍树结构，提供页面元素的层级和属性信息，在 BrowserForge 中仅用于合成阶段而非推理
- **Chrome DevTools Protocol (CDP)**：Chrome 浏览器调试协议，用于远程控制浏览器实例，本文用于沙箱管理
- **Chain-of-Thought (CoT)**：链式思维推理格式，本文通过 Seed 2.0 Pro 将异构推理重写为统一 CoT 格式
- **Online-Mind2Web**：基于真实在线网站的 Web Agent 评测基准，衡量端到端任务成功率
- **Multimodal-Mind2Web**：基于静态固定轨迹的逐步精度评测基准，包含跨任务、跨网站、跨领域划分
- **Normalized Coordinate [0, 1000]**：归一化坐标系统，将屏幕位置映射为 0-1000 范围内的整数，与渲染分辨率无关

## 可复现要素
- **数据集**：203,238 条轨迹（每条来自不同网站），论文声明从 Common Crawl 来源，具体 URL 列表和清洗细节需在附录或代码仓库中查找
- **代码**：论文未明确声明代码开源状态，但提到了使用 LLaMA-Factory 进行训练、Browser-use 作为 Solver 基础
- **权重**：论文未声明模型权重开源状态
- **关键超参**：
  - 训练轮数：3 epochs
  - 学习率：1e-5
  - 最大图像像素数：99,999,999 pixels（等效于不裁剪）
  - 冻结组件：视觉编码器 + adapter/projection 层
  - 全参数微调：语言模型组件
  - 并行沙箱数量：最多 300 个
  - 清洗后保留率：约 30% 原始步骤
