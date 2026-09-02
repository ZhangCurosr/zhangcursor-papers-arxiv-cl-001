---
title: "PATCHWRITE-ONE-LINE-NOT-ONE-SECTION-COMPILE-GATED-VALIDITY-P"
source: https://arxiv.org/pdf/2608.23001v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:11:29"
field: "自动化学术写作与文档修订"
keywords: ["LaTeX 编辑", "编译门控", "证据锁", "AI 学术写作", "PatchWrite", "Agent 可靠性"]
innovations: ["编译门：扫描 pdflatex 日志 fatal error 替代信任 nonstopmode PDF", "证据锁：引用键与经验数值集合成员验证", "HEAD 不变性：仅在全部门控通过时提交修改，失败时回滚到上一个干净版本"]
benchmarks: ["24-manuscript × 8-fault oracle stress test", "Live-model evaluation on qwen3.7-plus and Kimi K2.6"]
---

# 论文速读：PATCHWRITE-ONE-LINE-NOT-ONE-SECTION-COMPILE-GATED-VALIDITY-P

## 一句话总结
本文提出 PatchWrite，一种针对 AI 起草 LaTeX 论文的手术式单行编辑框架，通过编译门（compile gate）和证据锁（evidence lock）双重验证机制，确保局部修改不会意外篡改无关实验数据与引用，在 24 篇手稿 × 8 种故障的应力测试中实现了 192/192 的无关行保留率。

## 研究问题与动机
- **非单调修订问题**：现有 AI 写作管线在修复局部缺陷（如未定义引用）时，往往重新生成整个章节，导致无关的实验指标和引用被静默篡改（如"12 layers"变成"16 layers"）。
- **编译通过率指标的虚假可靠性**：仅监控 PDF 编译通过的仪表盘无法区分"合法修复"与"静默篡改"，因为两种情况都能生成有效 PDF。
- **LLM 生成的不可验证性**：Agent 社区开始关注模型输出是否满足领域特定正确性谓词，但缺乏具体的工程验证机制。
- **现有工具的设计局限**：PaperOrchestra 以"整槽（slot）"为单位重写字节，PaperSolver 虽有 EDIT N M 但缺少引用/数字锁，EasyPaper 为元数据到 LaTeX 的生成器而非修订操作符。

## 核心贡献（创新点）
- **编译门（Compile Gate）**：扫描 pdflatex 日志中的 fatal error 而非信任非停止模式（nonstopmode）生成的 PDF，首次将 PaperSolver 的 EDIT N M 与严格编译验证结合。
- **证据锁（Evidence Lock）**：将 \cite 键和数值 token 与文献库/实验日志做集合成员验证，阻止幻觉引用和未经核实的数字通过。
- **Oracle 压力测试协议**：在 24×8 规模（768 个 job）上隔离验证机制本身的有效性，而非生成质量，提供了可复现的对照组实验设计。
- **活模型双轨评估**：用 qwen3.7-plus 和 Kimi K2.6 分别提出 EDIT，测量从解析到编译到故障修复的端到端流水线性能。
- **一行提示修复揭示泛化局限**：为 Qwen 添加删除行建议后接受率达 100%，但 Kimi 在不同故障类型上暴露相同缺口，证明提示级修复是反应式而非通用的。

## 方法详解
**HEAD 不变性**：PatchWrite 仅在全部门控通过时提交新 HEAD，失败时保持原字节不变。

**编辑语法（EDIT N M）**：提议者看到带编号的行（1-based NNNN | text），发出单行编辑指令，区间长度 M-N+1 ≤ 40，替换内容非空。

**运行时流程（try_patch）**：
1. 解析 EDIT 块，失败返回原 HEAD。
2. 应用闭区间替换得到候选文档 T'。
3. 编译门：运行 latexmk，检查日志中无 "emergency stop"、"fatal error"、"! latex error" 或强制失败标记，PDF 需 >64 字节。
4. 证据门：\cite 键必须在参考文献、引用图或 HEAD 中存在；每个经验数值必须在实验日志中可溯源。
5. 两门均通过则提交 T' 为新 HEAD，否则回滚至原 T。

**回退机制**：非法或超限 EDIT 回落至传统整槽重写，该槽仍会引入突变，生产日志需统计此类回退。

## 实验与结果
- **数据集**：24 篇迷你手稿（16 英文 + 8 中文 LaTeX 代理），8 种脚本故障分为内容类（fake_cite, bad_number, todo, swapped_acc）和编译类（broken_math, unmatched_brace, unclosed_env, undefined_cmd），每条件 192 个 job，共 768 个 job。
- **主要结果（Oracle）**：
  - PO-slot rewrite：accept=1.00，compile_ok=1.00，但 layers_preserved=0.00，numeric Jaccard=0.6667（12→16 被篡改）。
  - PatchWrite：所有保留指标=1.00，包括 layers_preserved=1.00、cite Jaccard=1.00、numeric Jaccard=1.00。
  - 关闭编译门：accept=0.00（因为伪造的 \errmessage 导致 pdflatex 失败）。
  - 关闭证据门：layers 保留但 cite Jaccard 降至 0.6667（幻觉引用 \cite{hallucinated2024} 通过）。
- **活模型评估（Qwen3.7-plus）**：parse_ok=0.75，accept_rate=0.75，fault_fixed=0.9375，layers_preserved=0.9444。25% 回退全部来自 todo 和 undefined_cmd 两类故障（模型试图用空体 EDIT 删除行，被语法拒绝）。
- **提示修复后（Qwen）**：accept_rate 升至 1.00，但 fault_fixed 降至 0.849（评论式删除导致字符串存在性检查失败）。
- **Kimi K2.6**：parse_ok=0.875，accept_rate=0.875，fault_fixed=0.9107，layers_preserved=0.9167，mean latency=26.87s（推理密集）。回退集中在 unclosed_env 而非 Qwen 的故障类型。
- **人工评审**：16 对 PDF 盲评，两位评审均偏好 PatchWrite（16/16），C1 事实性 Likert 5.0 vs 2.0，C2 缺陷修复 5.0 vs 5.0，C3 行文几乎持平（3.97 vs 4.00）。
- **生产日志**：193 个任务中发现 45/157（28.7%）证据门失败（table_number_not_in_log），40 个 unmatched braces，7 个孤立 cite token。

## 相关工作脉络
- **PaperOrchestra [Song et al., 2026]**：多 Agent 论文写作框架，以"整槽"为单位重写字节，有引用和声明证据锁但缺乏行级编辑能力；PatchWrite 仅替换其 ContentRefinementAgent 中的突变算子。
- **PaperSolver [Schmidgall et al., 2025]**：首个引入 EDIT N M + 编译回滚的系统，但使用 LLM 自评作为质量停止条件，无引用/数字锁；PatchWrite 与其共享编辑语法但增加了外部证据约束和严格编译谓词。
- **AI Scientist-v2 [Yamada et al., 2025]**：端到端自动科学发现，写作部分仅比较；PatchWrite 关注修订阶段的保真度。
- **EasyPaper [EasyPaper, 2026]**：元数据到 LaTeX 的结构化生成软件，非同行评审写作算子，无编译验证。
- **SWE-agent [Yang et al., 2024] / SWE-bench [Jimenez et al., 2024]**：代码编辑领域的同类问题（有限接口优于原始文件访问），但正确性信号是测试套件而非 pdflatex + 引文/数字注册表。
- **PaperJury [Wang et al., 2026]**：独立提出的"确定性编排而非模型裁量"思路，针对人类撰写的 CS 论文预审，在流水线更靠前的位置；PatchWrite 在 LLM 起草修订循环内部运作，门更机械更轻量。

## 局限性与未来方向
- **Oracle 设计限制**：Table 2 假设正确答案已知以隔离机制验证，不能直接推广到 LLM 自主提出 EDIT 的场景（需结合 Table 3/3b 解读）。
- **短文规模**：24 篇迷你手稿非完整 IEEE/ctex 模板的 8-10 页会议论文，长文档中的行号漂移和复杂度未检验。
- **中文代理非正式支持**：中文稿件使用共享 pdflatex 引擎的 ASCII 标题代理，非 ctex camera-ready thesis 级别。
- **脚本化故障而非对抗性搜索**：8 种故障均为手工注入，未覆盖真实 LLM 错误分布或经过对抗性搜索。
- **人工评审规模有限**：两位评审（作者 + 非独立付费评审）× 16 对，符合探索性协议标准但不构成验证性研究。
- **回退未在生产追踪**：Table 3 的回退率仅测量于应力测试，生产草稿中按原因（parse/compile/gate）分类的日志尚未实现。
- **证据门保守性**：真实场景中若实验日志不完整（而非数字被编造），PatchWrite 会误拒合法 patch，需作者先将数字加入日志。

## 研究启发与可借鉴点
- **门控验证取代单一指标**：将"编译通过"细化为"日志级别检查 + 文件存在性检查"，对任何依赖外部验证的代码/文档生成系统具有迁移价值。
- **HEAD 不变性模式**：仅在全部门控通过时提交修改、失败时回滚到最后一个干净版本——该模式适用于数据库迁移脚本、合同修订、合规文档等任意需要原子变更的领域。
- **故障隔离测试协议**：用 24×8 oracle stress test 隔离验证机制与生成质量，再补充 live-model 评估，这种分阶段协议可作为 Agent 可靠性研究的模板。
- **约束编辑的实用性**：EDIT N M（≤40 行区间）证明了窄接口在复杂文档编辑中的优势，类比 SWE-bench 的发现，为学术写作 Agent 提供了可复用的设计范式。
- **一行提示修复揭示泛化缺口**：为单一模型添加提示可关闭特定回退，但另一模型暴露不同故障，提示工程应视为反应式而非根本解决方案，需从系统层面处理缺失能力。

## 关键术语表
- **EDIT N M**：PatchWrite 的核心编辑原语，表示将文档第 N 至 M 行替换为新内容，区间长度 ≤40。
- **Compile Gate**：编译门，扫描 pdflatex 日志中的 fatal error 而非仅检查 PDF 文件是否存在，防止语法错误的文档被提交。
- **Evidence Lock**：证据锁，验证 \cite 键在文献库/引用图中存在、经验数值在实验日志中可溯源，阻止幻觉引用和虚构数字。
- **Slot Fence / 整槽重写**：传统修订单元，将整节（如 Methods 或 Results）整体替换，易导致无关行被静默篡改。
- **HEAD Invariant**：HEAD 不变性，系统中唯一合法的文档状态为最后一次通过全部门控的版本，失败时回滚。
- **Nonstopmode PDF**：pdflatex 非停止模式生成的 PDF，即使存在语法错误也会写入文件，单独作为成功指标会掩盖缺陷。
- **Cite Jaccard**：引用键集合的 Jaccard 相似度，衡量修改前后文献引用的一致性。
- **Fault-fixed**：故障修复率，接受 patch 中真正修复注入缺陷的比例，而非仅满足编译和证据检查。

## 可复现要素
- **数据集**：24 篇迷你手稿（16 英文 + 8 中文），GitHub 公开于 https://github.com/Baiang/editnm（MIT 许可证）。
- **代码**：核心机制 tex_patch.py、评估脚本 revision_stress.py 和 llm_edit_stress.py 均开源；测试套件为 pytest tests/（无需网络）。
- **权重/模型**：评估使用 qwen3.7-plus（DashScope API）和 Kimi K2.6（Moonshot OpenAI 兼容 API），需用户提供 API key，有实际调用费用。
- **关键超参**：EDIT 区间上限 40 行；PDF 最小大小 64 字节；Qwen 评估 temperature 由服务端默认（DashScope 剥离温度参数）；Kimi temperature 固定为 1.0。
- **日志路径**：evals/results/summary.json（Table 2）、evals/results/llm_edit_summary.json（Table 3）、evals/results_kimi/llm_edit_summary.json（Table 3b）、raw_materials/human_eval.json（Table 4）、raw_materials/in_the_wild.json（Table 5）。
