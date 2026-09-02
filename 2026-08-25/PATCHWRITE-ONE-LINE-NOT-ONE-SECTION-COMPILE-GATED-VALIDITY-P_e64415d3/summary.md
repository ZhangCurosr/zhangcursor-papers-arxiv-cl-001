---
title: "PATCHWRITE-ONE-LINE-NOT-ONE-SECTION-COMPILE-GATED-VALIDITY-P"
source: https://arxiv.org/pdf/2608.23001v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:11:44"
field: "自动化学术写作"
keywords: ["LaTeX revision", "compile gate", "evidence lock", "agent reliability", "bounded edit", "citation hallucination"]
innovations: ["编译门控+证据锁定双重验证的有界行编辑机制", "一行提示修复消除模型系统性回退", "24×8 oracle压力测试揭示静默篡改问题"]
benchmarks: ["24-manuscript × 8-fault oracle stress test", "16-pair human rater evaluation", "193 in-product drafting logs"]
---

# 论文速读：PATCHWRITE-ONE-LINE-NOT-ONE-SECTION-COMPILE-GATED-VALIDITY-PRESERVING-EDITING-FOR-AI-DRAFTED-MANUSCRIPTS

## 一句话总结
PatchWrite 提出一种编译门控、证据锁定的单行编辑机制，用于修复 AI 起草的 LaTeX 手稿中的局部缺陷，确保编辑只影响目标区间而不静默篡改无关数字与引用；在 192 个 oracle 应力测试中 100% 保留无关行，且真实模型提议的补丁通过双重门控后幻觉率降至零。

## 研究问题与动机
- **非单调修订导致静默篡改**：现有自动化写作管线（如 PaperOrchestra）为修复一个未定义引用而重写整个 Results 段落，但无关的实验数字（如"12 layers"）会被悄悄替换，PDF 仍能编译，导致"假性成功"。
- **编译通过率不是可靠的质量指标**：pdflatex 在 nonstopmode 下即使存在语法错误仍会生成 PDF，仅依赖编译成功作为终止条件会掩盖内容层面的错误。
- **证据链断裂风险**：现有系统缺乏对引用键和数值 token 的外部溯源机制，模型可能生成编译通过但引用不存在的补丁。
- **AI 写作 Agent 的可验证性缺失**：更广泛的 Agent 可靠性问题尚未得到解决，PatchWrite 将其落地到 LaTeX 修订这一高风险场景，用机械判定替代主观评分。

## 核心贡献（创新点）
- **编译门控（compile gate）**：扫描 pdflatex 日志检测 fatal error，而非仅检查 PDF 文件是否存在，首次将 PaperSolver 的 EDIT N M 与严格的编译失败回滚机制结合。
- **证据锁定（evidence lock）**：将引用键与实验日志中的数值 token 双重验证结合为统一接受标准，防止幻觉引用和未经记录的数值通过。
- **oracle 压力测试协议（24×8 规模）**：设计 192 个脚本化修复任务，横跨内容类与编译类故障，隔离评估门控机制本身而非生成质量。
- **真实模型提议补丁的同伴实验**：在同一语料上用 qwen3.7-plus 和 Kimi K2.6 分别提议 EDIT，测量解析率、通过率与实际故障修复率，揭示模型行为的系统性模式。
- **一行提示修复方案**：针对模型尝试用空替换删除行的单一失败模式，添加一条 prompt 指令使接受率从 75% 提升至 100%，同时暴露出 fault-fixed 指标的边界问题。

## 方法详解
- **有界区间编辑语法**：提议者看到带行号的源文件，只能发出 `EDIT N M` 指令，其中 `M - N + 1 ≤ 40`，将闭区间 `[N, M]` 替换为至少一行的替换体；过长、为空、非整数或超出文件末尾的 EDIT 在编译前被拒绝。
- **编译门控流程**：若编译门控开启，系统写入唯一 jobname、删除旧 .aux/.pdf/.log 文件，运行 pdflatex，仅当 PDF 大小 > 64 bytes 且日志不含 `emergency stop`、`fatal error`、`! latex error` 或强制失败标记时才视为通过——nonstopmode 下 PDF 存在不等于编译通过。
- **证据门控流程**：引用键必须出现在 bibliography、citation map 或 HEAD 文档本身；任何新出现的键标记为 `unknown_cite_*`。每个实证数字必须能在实验日志中找到对应记录，任一违规均触发回滚。
- **HEAD 不变式**：当前编译文档 T 仅在两个门控均通过时提交为新 HEAD T'，任何失败均回滚至 T，磁盘上的文件始终是最后一个编译干净、证据清洁的版本。
- **回退机制**：无法解析合法 EDIT 或任一门控失败时，回退到传统的整槽重写（slot-fence rewrite），保证任务完成但引入全段突变风险。

## 实验与结果
- **数据集**：24 篇迷你文章（16 英文 + 8 中文期刊 pdflatex 代理），每篇包含固定句式"The encoder uses 12 layers..."和来自实验日志的结果数字；8 种脚本化故障分为内容类（fake_cite、bad_number、todo、swapped_acc）和编译类（broken_math、unmatched_brace、unclosed_env、undefined_cmd）。
- **基线**：PaperOrchestra 整槽重写（po_slot）、关闭编译门控的 PatchWrite、关闭证据门控的 PatchWrite。
- **最强结果**：PatchWrite 在 192 个 oracle 任务中 layers_preserved = 1.00（vs. po_slot 的 0.00），cite Jaccard = 1.00（vs. po_slot 的 0.6667），accept rate = 1.00，全部八种故障上模式一致；关闭编译门控接受率降至 0.00，关闭证据门控 cite Jaccard 降至 0.6667。
- **真实模型实验（qwen3.7-plus）**：Parse ok = 0.75，Accept rate = 0.75，Fault fixed of accepted = 0.9375，Layers preserved = 0.9444；25% 回退全部来自 todo 和 undefined_cmd 两种故障，原因统一为模型尝试用空替换删除行。
- **一行提示修复后（qwen3.7-plus）**：Parse ok 升至 1.00，Accept rate 升至 1.00，Fallback rate 降至 0.00，但 Fault fixed of accepted 降至 0.849（因注释删除方式导致部分补丁未满足严格字符串匹配）。
- **Kimi K2.6 实验**：Parse ok = 0.875，Accept rate = 0.875，Fault fixed = 0.9107，Layers preserved = 0.9167，Mean latency = 26.87s；回退集中在 unclosed_env 故障（23/24），说明不同模型在不同故障类型上暴露相同结构的缺陷。
- **人类评估（16 对 PDF）**：两位盲评者在 C1 事实一致性上 PatchWrite 均值得 5.0 vs. PO-slot 的 2.0，C2 缺陷修复双方均为 5.0，C3 行文质量近似，偏好选择 16/16 支持 PatchWrite。
- **生产日志**：193 个任务、240 次修订中，证据门控失败率 28.7%（157 快照中 45 失败），40 个文件存在未匹配的 TeX 括号，7 个孤立引用 token。

## 相关工作脉络
- **PaperOrchestra [Song et al., 2026]**：将 idea 与实验日志映射到会议模板，修订时重写整个 slot；本文将其 mutation operator 替换为有界行编辑，保留其余框架。
- **PaperSolver [Schmidgall et al., 2025]**：提供 EDIT N M 语法和编译回滚机制，但使用 LLM 整体评分作为终止条件且无证据锁定；本文复用其编辑语法但收紧编译判定并添加外部证据约束。
- **Agent Laboratory / SWE-agent**：代码编辑领域证明窄接口优于原始文件访问；PatchWrite 在 LaTeX 修订中做出相同设计赌注，但正确性信号是 pdflatex + 引文/数字注册表而非测试套件。
- **PaperJury [Wang et al., 2026]**：独立提出确定性编排优先于模型 discretion 的理念，但面向人工撰写论文的预投稿审查流程；PatchWrite 面向 AI 起草阶段的修订循环，门控更机械、更轻量。
- **EasyPaper [2026]**：从元数据生成结构化 LaTeX 的软件系统，非同行评审写作算子，与本文不直接可比。
- **STORM / AutoSurvey**：大纲驱动的长文合成系统，无编译门控且不绑定数值到实验日志，解决的问题维度不同。

## 局限性与未来方向
- **oracle 实验假设已知正确补丁**，仅隔离门控机制，不反映真实生成质量；真实模型实验仅在单一语料上单次采样，样本量有限。
- **迷你文章非完整会议论文**，24 篇 mini-article 不代表 8–10 页 venue PDF 的实际复杂度。
- **中文代理共享 pdflatex 引擎**，并非真正的 ctex camera-ready 学位论文。
- **故障为脚本化注入而非对抗搜索**，八种故障虽覆盖两类，但并非从真实 LLM 错误分布中采样。
- **人类评估规模小**（两位评审 × 16 对），且评审的是 oracle PDF 而非真实模型的 imperfect patches。
- **证据门控仅检查集合成员关系**，无法验证引用键与具体论断的语义正确性，已定义引用键的论文仍可能支撑错误主张。
- **未来方向**：在完整 venue 模板上测试多模型 LLM 提议编辑、对模型自身 imperfect patches 进行真实人类偏好比较、将三件套模式（有界编辑 + 域特定有效性谓词 + 失败回滚）推广至代码审查、合规文档、数据库迁移等场景。

## 研究启发与可借鉴点
- **编译门控设计**：扫描日志而非检查文件存在性，这一思路可直接迁移到任何依赖外部工具的编辑系统中（如运行测试套件而非仅检查文件生成）。
- **证据锁定的轻量化实现**：通过 registry + log 交叉验证即可检测引用/数值幻觉，无需昂贵的外部知识库检索，适合资源受限的生产环境。
- **一行 prompt 修复的发现**：明确告知模型"删除意味着替换为注释而非空替换"可消除系统性回退，这一技巧对工具调用型 agent 的 prompt 设计具有通用参考价值。
- **双模型对照实验设计**：在同一语料上比较 qwen3.7-plus 和 Kimi K2.6 暴露出不同故障类型的集中回退，为后续研究提供了模型特异性失败分析的方法论范例。
- **HEAD 不变式架构**：每次编辑前保存最后一个干净状态、失败时无条件回滚，这一模式适用于任何需要强一致性的自动化编辑 pipeline。

## 关键术语表
- **EDIT N M**：有界区间编辑指令，将源文件第 N 至 M 行（区间长度 ≤ 40）替换为提议者提供的新文本。
- **Compile gate**：编译门控，扫描 pdflatex 日志检测 fatal error 标记，仅当日志无错误且 PDF > 64 bytes 时才接受补丁。
- **Evidence lock**：证据锁定，验证新引用键存在于 bibliography/citation map 中，且每个实证数字能在实验日志中找到对应记录。
- **Slot fence / slot rewrite**：整槽重写，传统方法将整个段落（如 Results section）作为编辑单元，易导致无关内容被静默篡改。
- **Head invariant**：HEAD 不变式，磁盘上始终保留最后一个编译干净、证据清洁的文档版本，任何失败的编辑均触发回滚。
- **Nonstopmode trap**：nonstopmode 陷阱，pdflatex 在 nonstopmode 下即使存在语法错误仍会生成 PDF 文件，仅检查文件存在性会掩盖编译失败。
- **Content-class fault**：内容类故障，HEAD 仍能编译的故障（如虚构引用、错误数值），测试证据门控的有效性。
- **Compile-class fault**：编译类故障，HEAD 无法编译的故障（如未闭合括号、未定义命令），同时测试编译门控的恢复能力。

## 可复现要素
- **数据集**：24 篇迷你文章（16 英文 + 8 中文代理），代码公开于 https://github.com/Baiang/editnm（MIT 许可证），包含 corpus.py 生成脚本。
- **代码**：核心机制 tex_patch.py、oracle 评测 revision_stress.py、真实模型评测 llm_edit_stress.py 均已开源；Pytest 测试套件无网络依赖。
- **权重**：依赖外部 LLM API（DashScope qwen3.7-plus、Moonshot Kimi K2.6），论文未提供本地模型权重。
- **关键超参**：EDIT 区间最大长度 40 行；PDF 最小大小阈值 64 bytes；中文代理使用共享 pdflatex 引擎（非 ctex camera-ready）。
- **复现命令**：Oracle 评测 `python revision_stress.py -out results`；真实模型评测需设置 DASHSCOPE_API_KEY 后运行 `python llm_edit_stress.py -out results`。
