# PRIMING BASE MODELS FOR AGENTIC POST-TRAINING: AN OPEN MIDTRAINING RECIPE

Soumye Singhal<sup>∗</sup>, Joseph Jennings<sup>∗</sup>, Ka Shun Shum<sup>∗†</sup>, Alexander Bukharin<sup>∗</sup>, Khushi Bhardwaj<sup>∗</sup> Zirui Liu, Venkat Srinivasan, Jonathan Li, Yash Jain, Kezhi Kong, Markus Kliegl, Olivier Delalleau, Sanjeev Satheesh, Mostofa Patwary

NVIDIA

## ABSTRACT

Frontier models, general-purpose and domain-specific alike, now include a midtraining stage between pretraining and post-training, but little has been published about how this stage is built. We present an open midtraining recipe that primes base models for agentic post-training, and we show its effectiveness on two Nemotron models of different sizes. Pretraining typically runs at short context and sees almost no agent interactions, while SFT data is mostly synthetic, and very long SFT stages can hurt the model’s later RL trainability. Our midtraining stage sits between the two. It trains at 256K context for hundreds of billions of tokens on a diverse mixture of natural software-engineering data like packed repositories, pull requests, commit histories, and build logs together with a small share of agentic traces. On Nemotron 3.5 Lightning, 500B midtraining tokens improve SWE-bench Verified from 60.7 to 64.5, SWE-bench Multilingual from 55.8 to 62.8, and SWE-bench Pro from 32.0 to 35.5 after identical full SFT. On the larger Nemotron 3.5 Super, 280B midtraining tokens improve DeepSWE, a harder benchmark of long-horizon tasks, from 11.7 to 15.7 and SWE-bench Verified from 72.9 to 74.3 after identical full SFT. Midtraining also improves robustness across agent harnesses on both models. We report the ablations behind the recipe and an evaluation protocol built on short fixed SFT runs, since base-model proxy metrics did not correlate reliably with post-training results.

## 1 INTRODUCTION

Between pretraining and post-training, frontier coding models such as Qwen3-Coder-Next (Cao et al., 2026), GLM-5 (GLM-5 Team, 2026), Kimi-Dev (Yang et al., 2026), and Composer 2 (Cursor Research, 2026) include a midtraining stage of hundreds of billions of tokens of agentic, longcontext data on top of a converged base model. Technical reports describe this stage at a high level, but rarely explain how its data is constructed, filtered, and weighted, or measure the impact of each design choice. Following Qwen-Coder (Hui et al., 2024; Cao et al., 2026), we present an open midtraining recipe for agentic coding, developed and demonstrated on Nemotron base models. We describe how each data source is built, and we report the ablations that guided our training choices.

Base models are usually pretrained at short sequence lengths, typically 4K to 8K tokens for compute efficiency (DeepSeek-AI, 2024; Qwen Team, 2025), and then extended to long context in a short final stage before SFT. Nemotron models, for example, are pretrained at 8K and extended to 1M. The data that matters most for software-engineering agents does not fit this pipeline. Pretraining corpora usually contain almost no agent interactions, as in the Nemotron pretraining corpus (NVIDIA, 2025), and the natural data around software engineering, such as whole repositories, code evolution through commit histories, pull requests, and build logs, often runs to tens or hundreds of thousands of tokens, too long for pretraining and too large for a short long-context stage. Adding agent traces to pretraining is also not straightforward, since pretraining typically applies no loss masking, and it is unclear how the tool responses (environment outputs) in those traces should be handled. SFT in turn consists mostly of synthetic chat data. Recent work has documented that very long SFT stages can reduce how much the model later gains from RL (Liu et al., 2026b; Kang et al., 2025; Zhang et al., 2026), and we have observed this in our own training as well.

Midtraining fills this gap. Before the final long-context stage, it trains for hundreds of billions of tokens at 256K, long enough to hold whole repositories, commit histories, and long agent traces (§5.1). The data is mostly natural software-engineering data, with a bounded share of agent traces whose tool responses are masked from the loss. Much of this learning can therefore move out of SFT, so SFT can stay short and preserve the model’s later RL trainability (Liu et al., 2026b; Kang et al., 2025; Zhang et al., 2026).

We develop the recipe on Nemotron 3.5 Lightning-30B-A3B and then scale it to the larger Nemotron 3.5 Super-120B-A12B (NVIDIA, 2026a). On N-3.5 Lightning, 500B midtraining tokens improve SWE-bench Verified, Multilingual, and Pro by 3.8, 7.0, and 3.5 points over the same model without midtraining after identical full SFT, with the most sustained gains on SWE-bench Multilingual (§4.3). On N-3.5 Super, 280B midtraining tokens improve DeepSWE from 11.7 to 15.7 and SWE-bench Verified from 72.9 to 74.3 after identical full SFT (§4.4). Midtraining also improves robustness to the agent harness, which matters for open models that are used with many different harnesses. Both models improve on every harness we tested, and on N-3.5 Super after 65B tokens and short SFT, the average gain across harnesses is 8.6 points on SWE-bench Verified and 6.8 on Terminal-Bench (§4.5).

We make the following contributions.

• A fully documented midtraining data recipe, showing that a small share of agentic traces adds clear gains on top of natural data (§3 and §4.2).

• An evaluation protocol based on identical SFT runs, since base-model proxy metrics such as bits-per-byte and LoCoDiff did not reliably predict post-training results (§4.1).

• We show that gains hold as we scale both model size and midtraining tokens up to 500B, across benchmarks and agent harnesses (§4).

• Ablations of modeling choices, such as the learning rate, the share of replayed pretraining data, tool-response masking, and midtraining’s effect on long-context ability (§5).

## 2 RELATED WORK

Midtraining in frontier coding models. Although midtraining is growing in popularity, only a few works have presented detailed recipes and ablation results to the community. Qwen3-Coder-Next provides one such recipe, midtraining at a 256K context on trillions of tokens, including 600B tokens of repository-level data across 370 programming languages (Cao et al., 2026). Its recipe suggests combining mined PR data with agentic traces but does not explain exactly how to do so. GLM-5 reports midtraining on approximately 1.55T tokens with progressive context extension, including 160B tokens of software-engineering data constructed from roughly 10M issue–PR pairs (GLM-5 Team, 2026), but does not provide details of the recipe. Kimi-Dev, one of the primary works on which we build, provides a more detailed recipe (Yang et al., 2026). It performs midtraining on approximately 150B tokens of diff patches, commit sequences, and synthetic agent interactions, and shows that a model with this prior needs far fewer SFT tokens than the raw base model to reach the same agent performance. daVinci-Dev advocates trajectory-style midtraining data and matches Kimi-Dev’s performance using fewer than half as many tokens (Zeng et al., 2026). Composer 2 shows that code-focused continued pretraining improves downstream RLVR performance (Cursor Research, 2026). AgentFounder similarly shows that continued pretraining benefits search agents (Su et al., 2026), while Liu et al. (2026a) study when midtraining effectively bridges the distributions of pretraining and post-training data. Relative to these works, we provide the ablation-level detail missing from the technical reports and evidence that the gains persist through full SFT.

Repository-level data construction. StarCoder2 concatenates repository files in random order (Lozhkov et al., 2024); DeepSeek-Coder orders files by import dependencies (Guo et al., 2024); SPLiCe shows that structured packing improves long-context use (Staniszewski et al., 2025); Pro-Long randomizes on purpose to avoid recency bias (Gao et al., 2025). Sapronov & Glukhov (2025) find that most of the gain from repository-level pretraining for project-level code completion comes from adapting to a longer context rather than from how the repository context is assembled. Best-Fit

![](images/0bb36edc7f8bcee9429d1649bd00c795b7a6bfe9f0bf1d5eebd55e23bccb88f0.jpg)

Figure 1: The midtraining blend, by share of training tokens. Replay is drawn from the base model’s pretraining data. Token counts and epochs per source are in Table 1.
<table><tr><td>Source</td><td>Unique tokens</td><td>Share of blend</td><td>Tokens seen (500B run)</td><td>Epochs</td></tr><tr><td>Issue-PR</td><td>216.9B</td><td>40.8%</td><td>205.2B</td><td>0.95</td></tr><tr><td>Repositories</td><td>104.1B</td><td>20.0%</td><td>100.7B</td><td>0.97</td></tr><tr><td>Agentic traces</td><td>39.4B</td><td>10.0%</td><td>50.3B</td><td>1.28</td></tr><tr><td>Diff data</td><td>16.0B</td><td>3.0%</td><td>15.1B</td><td>0.95</td></tr><tr><td>Build and CI logs</td><td>6.5B</td><td>1.2%</td><td>6.1B</td><td>0.95</td></tr><tr><td>Pretraining replay</td><td>~2.0T</td><td>20.0%</td><td>100.7B</td><td>0.05</td></tr><tr><td>Long-context data</td><td>67.4B</td><td>5.0%</td><td>25.2B</td><td>0.37</td></tr></table>

Table 1: Sources of the midtraining blend in Figure 1. Replay is drawn from the base model’s finalphase pretraining blend. Epochs are tokens seen divided by unique tokens.

Packing reduces truncation harms (Ding et al., 2024), and In-Context Pretraining shows relateddocument packing helps beyond code (Shi et al., 2024). Our repository data differs in selection. We keep only repositories whose files reference each other through imports, and order files so that related code is adjacent (§3).

Evaluating base models for agentic potential. Agentic benchmarks are unreliable before posttraining, so Composer 2 selects among base models using held-out codebase perplexity, state tracking, and adversarially built QA instead (Cursor Research, 2026). Other benchmarks target specific skills. LoCoDiff measures file-state reconstruction from git history (Mentat AI Team (AbanteAI), 2025), RepoBench and CrossCodeEval measure cross-file completion (Liu et al., 2024; Ding et al., 2023), and APTBench converts agent tasks into a format base models can answer (Qin et al., 2025). Whether such base-model measures predict downstream ability is still contested (Liu et al., 2023; Chen et al., 2025). In our experiments, they did not track post-training results reliably (§4.1).

## 3 THE MIDTRAINING RECIPE

Figure 1 summarizes the blend, which is designed for diversity. Most of it is natural softwareengineering data, with a small share of synthetic agent traces. We render these traces in the Nemotron chat template and keep them to 10–15% of the blend to avoid overfitting to SFT-like data, which can hurt later RL trainability (Liu et al., 2026b; Kang et al., 2025; Zhang et al., 2026). Everything else stays in natural formats, without chat templates. Every source keeps only permissive licenses and goes through decontamination against SWE-bench and Terminal-Bench, secret redaction, and loss masking of content the model should not imitate, such as repeated delimiters, hashes, and timestamps.

Issue–PR. We construct a large-scale code-change corpus from merged pull requests in public GitHub repositories, combining repository snapshots with PR descriptions, linked issues, review discussions, and observed patch evolution when available. We restrict collection to repositories with at least ten stars in supported programming languages. We deduplicate examples by repository and PR identifier and remove records with unavailable repositories, missing Git objects, failed patch reconstruction, empty or excessively large changes, generated content, or other low-quality patches. Each example is laid out as a sequence in which repository metadata, date, file tree, and code context condition the model’s reconstruction of the submitted changes. No chat template is applied to this data, but we mask repetitive formatting text from the training loss. In total, this process leaves us with 12.1 million PRs and 200 billion to 2 trillion tokens depending on how much repository context we include in each example. For issue-linked PRs, a structured LLM judge additionally checks that the issue and the patch match, and PR-only examples are filtered with deterministic rules (Appendix A.1).

Repositories. Pretraining corpora have historically presented code as individual files rather than whole repositories, and even when files from one repository are packed together, a short pretraining context holds only a small part of it. Agents, however, work inside repositories far longer than that and must use code defined far from where it is called. Midtraining is where the model first reads whole repositories and predicts code that depends on files tens of thousands of tokens earlier. This helps most when the files actually depend on each other, and in many repositories they do not, since many are collections of scripts, exercises, or snippets that never import one another. We start from the repositories in the Nemotron 3 pretraining corpus (NVIDIA, 2025), build an import graph for every repository in 14 languages, and keep only repositories in which at least 20% of code files have an import edge to another file in the repository (10% for Ruby). Kept repositories are serialized with the top-level README first and connected files next to each other, and packed at 256K, with longer repositories split between connected components rather than truncated, giving about 104B tokens. Compared to alphabetical ordering, dependency ordering better separates stronger from weaker models. It lowers bits-per-byte for stronger models and raises it for weaker ones, which suggests that it exposes cross-file structure that weaker models have not yet learned. The languages, filters, and blend weights are listed in Appendix A.2.

Diff data. Much of software engineering is changing existing code, such as making an edit that stays consistent across several files or keeping track of what a file looks like after a long series of changes. Repository snapshots only show code in its final state, so we build two kinds of examples from commit histories. Multi-file edits take a merged pull request, show the model the affected files before the change, and train it to produce them after the change. We keep only changes whose files are linked in the import graph used for the repository data, so that the edits are coupled. Codeevolution documents give the initial state of a file, the sequence of real commits that changed it from oldest to newest, and its final state. This trains the long-range state tracking that LoCoDiff measures (Mentat AI Team (AbanteAI), 2025). Histories range from a single commit to several dozen. Diff headers, commit hashes, authors, and dates are masked from the loss. Sources, filters, and statistics are in Appendix A.3.

Build and CI logs. Agents constantly read logs, stack traces, and build output, but PR data lacks the environment’s response, so we add raw system logs from the full LogHub archives (Zhu et al., 2023) and CI/CD logs from GitHub Actions run logs and workflow-file histories (Moriconi et al., 2025; Cardoen et al., 2024), plus a small curated slice pairing log windows with diagnoses (Appendix A.4).

Agentic traces. We include agent trajectories to teach interaction patterns that natural software artifacts lack, such as multi-step tool use, testing changes, and recovering from failures. These trajectories are separate from the SFT data. The highest-quality trajectories are reserved for SFT, and midtraining puts the rest to use, including trajectories from earlier teacher models and tasks that do not fit post-training.

The corpus used in our blend contains about 39B tokens (over one million trajectories, about 500K distinct queries, and 30M tool calls, including SWE-Zero and SWE-Hero trajectories (Ludwig et al., 2026)). A small portion of the agentic midtraining blend consists of data contributed by Perplexity as a partner in the Nemotron Coalition. The full trace corpus covers more than ten agent scaffolds, including OpenHands (Wang et al., 2025), OpenCode, SWE-agent (Yang et al., 2024), Codex, Claude Code, and Terminus, across software engineering, terminal use, search, and general tool use, because traces from a single scaffold transfer poorly to others (Cao et al., 2026). Each trajectory keeps its scaffold’s instructions and tool definitions and is rendered in the Nemotron chat template. Appendix A.5 gives the statistics and distributions.

Filtering targets shortcuts and undesirable interaction patterns, including solution leakage through Git history or Internet lookup, unfinished trajectories, malformed tool calls, and repetitive fileinspection loops. As in SFT, we train only on assistant turns, including tool calls, since also training on tool responses hurts results after SFT (§5.2).

## 3.1 TRAINING

Midtraining starts from a converged base model, so we choose the training setup to add new capabilities without degrading what the base model already knows. We train at 256K context, long enough to hold almost every midtraining document, many of which are far longer than the 8K pretraining context (Figure 2). Our first N-3.5 Lightning experiments start from the 8K pretraining checkpoint, N-3.5 Lightning-8K. For the N-3.5 Lightning scaling run and all N-3.5 Super runs, we start from the 1M long-context checkpoint, which lets us go straight from midtraining to SFT, keeping evaluations simple. In a mature pipeline, midtraining would come before the final long-context stage (§5.1). We set the replay share to 20% (§5.3). The learning rate warms up briefly and then stays constant, at 5e-5 for N-3.5 Lightning, which favors agentic gains, and 1e-5 for the larger N-3.5 Super, which favors preserving general capabilities (§5.3). Appendix B lists the full settings.

## 4 EXPERIMENTS AND RESULTS

## 4.1 EVALUATION PROTOCOL

The success of midtraining is ultimately measured by how well the model does after post-training. Our evaluation protocol is therefore holistic. It runs the full post-training pipeline and also tracks base-model post-trainability metrics and pretraining general-capability benchmarks.

• Short and full SFT. We use an agentic-focused SFT blend (Appendix B.1) at two budgets, 33B tokens (short SFT) and 100B tokens (full SFT), and compare each midtrained model with its starting checkpoint after identical SFT. Full SFT was close to saturation in our runs and supports the main claims, while short SFT gives a quicker signal.

• Fixed evaluations. We evaluate on SWE-bench Verified, Multilingual, and Pro, Deep-SWE, Terminal-Bench (version 2.0, or 2.1 where noted), and a BrowseComp subset. Unless stated otherwise, SWE-bench runs in OpenHands and Terminal-Bench in Terminus-2, with pass@1 averaged over 3 and 8 rollouts per task (Appendix C.1 gives the setup and references). We also report pass@k, since a higher pass@k has been shown to indicate more room for RL to improve (Yue et al., 2025).

• Base-model post-trainability metrics. LoCoDiff (Mentat AI Team (AbanteAI), 2025), RepoBench (Liu et al., 2024), and bits-per-byte on held-out code repositories and on agent traces (Appendix C.2) measure capabilities that midtraining targets, such as file-state tracking and cross-file modeling. For traces we score the decisive turns of DeepSWE trajectories (Yu et al., 2026), which correlated best with post-training results. LoCoDiff is averaged over 3 greedy runs on all 198 cases, and the other metrics are single runs. They improve with midtraining (Table 4), but did not track post-training results closely enough to guide recipe decisions.

• Regression checks. During midtraining we also track the pretraining general-capability benchmarks from NVIDIA (2026b) in four groups, general knowledge, math, code, and commonsense understanding (Appendix C.1), to catch regressions (§5.3).

## 4.2 BLEND VALIDATION ON N-3.5 LIGHTNING

Our first experiment midtrains the N-3.5 Lightning-8K base model for 100B tokens, once on the full blend and once without agentic traces, and compares both with the base model after identical SFT (Table 2). Gains seen after short SFT carry over to full SFT, so short SFT can serve as a quick check. Midtraining improves SWE-bench Verified and Multilingual pass@1 by 0.8 and 8.2 points after short SFT, and both gains remain after full SFT (3.8 and 3.7 points). Terminal-Bench is noisier and improves only in full-SFT pass@1. Agentic traces give the majority of the gain after SFT. The no-traces model, trained on natural data alone, gains less than one point on SWE-bench Verified after full SFT, while the full blend improves on it by 3.0, 2.5, and 6.6 points on SWE-bench Verified, Multilingual, and Terminal-Bench. We still keep natural data as the large majority of the blend. Its benefits show up beyond SWE-bench pass@1, in file-state tracking and cross-file modeling (Tables 4 and 5) and robustness across agent harnesses (§4.5), and a blend of mostly traces would amount to a much longer SFT, which can hurt later RL trainability (Liu et al., 2026b; Kang et al., 2025; Zhang et al., 2026).

<table><tr><td></td><td>SWE-V pass@1 / pass @3</td><td>SWE-Multi pass@1 / pass @3</td><td>TB 2.0 pass@ 1 / pass@8</td></tr><tr><td>Short SFT (33B tokens)</td><td></td><td></td><td></td></tr><tr><td>Base-8K</td><td>57.1 / 69.2</td><td>48.6 / 65.0</td><td>22.1 / 40.5</td></tr><tr><td>Midtrained, no traces</td><td>56.1 / 68.2</td><td>48.4 / 65.3</td><td>19.8 / 40.5</td></tr><tr><td>Midtrained</td><td>57.9 / 71.4</td><td>56.8 / 70.0</td><td>20.1 / 39.3</td></tr><tr><td>Full SFT (100B tokens)</td><td></td><td></td><td></td></tr><tr><td>Base-8K</td><td>59.7 / 69.8</td><td>55.9 / 72.7</td><td>22.9 / 44.9</td></tr><tr><td>Midtrained, no traces</td><td>60.5 / 72.6</td><td>57.1 / 72.7</td><td>18.5 / 34.8</td></tr><tr><td>Midtrained</td><td>63.5 / 73.2</td><td>59.6 / 74.0</td><td>25.1 / 43.8</td></tr></table>

Table 2: Early experiment on the N-3.5 Lightning-8K checkpoint, midtrained for 100B tokens with and without agentic traces and compared with the base model after identical short or full SFT (%).
<table><tr><td>Midtraining tokens (B)</td><td>SWE-V pass@ 1 / pass @3</td><td>SWE-Multi pass@1 / pass@3</td><td>TB 2.0 pass@ 1 / pass @8</td><td>SWE-Pro pass@1 / pass@3</td></tr><tr><td>0 (base)</td><td>60.7 / 72.2</td><td>55.8 / 72.3</td><td>22.6 / 43.8</td><td>32.0 / 41.5</td></tr><tr><td>100</td><td>64.1 / 74.0</td><td>57.4/71.7</td><td>22.5 / 44.9</td><td>34.9 / 45.1</td></tr><tr><td>200</td><td>64.3 / 73.4</td><td>59.7 / 76.7</td><td>21.4 / 39.3</td><td>35.8 / 45.3</td></tr><tr><td>300</td><td>63.5 / 73.6</td><td>62.0 / 76.7</td><td>24.9 / 48.3</td><td>36.3 / 46.4</td></tr><tr><td>500</td><td>64.5 / 75.0</td><td>62.8 / 77.0</td><td>23.6 / 46.1</td><td>35.5 / 45.4</td></tr></table>

Table 3: Midtraining scaling on N-3.5 Lightning-LC after identical full SFT. After 500B tokens, midtraining improves both pass@1 and pass@k on every benchmark, so it raises what the model can solve rather than only its reliability.

## 4.3 SCALING N-3.5 LIGHTNING TO 500B TOKENS

Our main N-3.5 Lightning experiment uses the long-context checkpoint N-3.5 Lightning-LC as the base model (§3.1). The blend has about 380B unique tokens (Table 1), enough to midtrain for 500B tokens while seeing each natural source about once. We apply identical full SFT to the base model and to midtrained checkpoints from 100B to 500B tokens (Table 3).

Midtraining improves every benchmark over the base model. After 500B tokens, pass@1 rises by 3.8 points on SWE-bench Verified, 7.0 on SWE-bench Multilingual, 3.5 on SWE-bench Pro, and 1.0 on Terminal-Bench, and pass@k rises on all four, by 2.3 to 4.7 points. Most of the SWEbench Verified and Pro gain arrives by 100B tokens, SWE-bench Multilingual keeps improving up to 500B, and Terminal-Bench is noisier and peaks at 300B. Unlike Table 2, these results start from N-3.5 Lightning-LC. After identical full SFT, its scores differ slightly from those of N-3.5 Lightning-8K, but stay within run-to-run variation. Midtraining also changes how the agent uses its context. Without midtraining, 4.1% of SWE-bench Pro rollouts fail by filling the 256K context window, compared with 0.6% after 500B tokens (Appendix E.1).

The base-model post-trainability metrics show the same trend (Table 4). All four improve over the base model and keep improving from 100B to 500B tokens, reflecting the capabilities that the natural SWE data and traces add. On LoCoDiff, for example, the base model rebuilds no files exactly, while the midtrained model rebuilds 21.5% after 100B tokens and 26.8% after 500B. Most of this gain comes from the diff data, which raises exact match from about 1% to 30% in a controlled 10B-token comparison in Table 5.

## 4.4 SCALING TO N-3.5 SUPER

We further apply the recipe to the larger N-3.5 Super to test how it scales with model size. We midtrain for 65B and 280B tokens with the blend in Figure 1. As in the scaling run, we start from the long-context checkpoint and compare each midtrained model with the base model after identical short and full SFT (Table 6). The clearest gain is on DeepSWE (Huang et al., 2026), the hardest benchmark, whose long-horizon tasks are written from scratch. After full SFT, 280B tokens of midtraining raise it from 11.7 to 15.7, and after short SFT from 6.4 to 12.4. SWE-bench Verified also improves, from 72.9 to 74.3 after full SFT, and BrowseComp by 3 points after both short and full SFT. SWE-bench Multilingual and Pro do not improve. After full SFT they are 1.1 and 2.1 points below the base model, while after short SFT they are level with or slightly above it, so we do not read these differences as a consistent effect.

<table><tr><td rowspan="2"></td><td colspan="2">LoCoDiff</td><td rowspan="2">DeepSWE BPB↓</td><td rowspan="2">Repository BPB↓</td><td rowspan="2">RepoBench ↑</td></tr><tr><td>Exact (%) ↑</td><td>Distance ↓</td></tr><tr><td>N-3.5 Lightning-LC</td><td>0.0</td><td>0.903</td><td>0.171</td><td>0.238</td><td>70.09</td></tr><tr><td>Midtrained, 100B</td><td>21.5</td><td>0.228</td><td>0.144</td><td>0.228</td><td>71.22</td></tr><tr><td>Midtrained, 500B</td><td>26.8</td><td>0.164</td><td>0.140</td><td>0.217</td><td>72.41</td></tr></table>

Table 4: Base-model post-trainability metrics for N-3.5 Lightning-LC and its midtrained check points from the scaling run. All metrics improve as midtraining scales.
<table><tr><td>Midtraining data</td><td>Tokens</td><td>Exact (%) ↑</td><td>Distance↓</td></tr><tr><td>Blend without diff data</td><td>10B</td><td>1.3</td><td>0.507</td></tr><tr><td>50% diff data</td><td>10B</td><td>29.8</td><td>0.150</td></tr><tr><td>Full blend (§4.3)</td><td>500B</td><td>26.8</td><td>0.164</td></tr></table>

Table 5: LoCoDiff on midtrained N-3.5 Lightning base models. The two 10B-token runs start from N-3.5 Lightning-8K and differ only in whether half of the blend is diff data. Diff data drives most of the LoCoDiff gain.
<table><tr><td>Midtraining</td><td>SWE-V</td><td>SWE-Multi</td><td>SWE-Pro</td><td>DeepSWE</td><td>BrowseComp</td></tr><tr><td colspan="6">Short SFT (33B tokens)</td></tr><tr><td>0 (base)</td><td>71.3</td><td>65.9</td><td>52.1</td><td>6.4</td><td>64.2</td></tr><tr><td>65B</td><td>71.9</td><td>65.6</td><td>51.7</td><td>10.4</td><td>65.0</td></tr><tr><td>280B</td><td>72.1</td><td>67.4</td><td>52.2</td><td>12.4</td><td>67.2</td></tr><tr><td colspan="6">Full SFT (100B tokens)</td></tr><tr><td>0 (base)</td><td>72.9</td><td>69.0</td><td>53.9</td><td>11.7</td><td>61.8</td></tr><tr><td>65B</td><td>72.5</td><td>68.2</td><td>52.4</td><td>13.5</td><td>61.8</td></tr><tr><td>280B</td><td>74.3</td><td>67.9</td><td>51.8</td><td>15.7</td><td>64.8</td></tr></table>

Table 6: N-3.5 Super after 65B and 280B midtraining tokens and identical short or full SFT (pass@1, %). Midtraining gives a large gain on DeepSWE, the hardest benchmark, and also improves SWEbench Verified and BrowseComp.

## 4.5 HARNESS ROBUSTNESS

Open models are used with many agent harnesses, not only the one their SFT data was built for, so harness robustness is a goal of our midtraining and the reason we draw traces from many scaffolds. Table 7 shows that midtraining improves pass@1 on every harness we tested, which makes both models more robust to the choice of harness, although large differences between harnesses remain. On N-3.5 Super after short SFT, the midtrained model is better on every harness, by 8.6 points on average on SWE-bench Verified and 6.8 on Terminal-Bench 2.1. On N-3.5 Lightning-8K after full SFT, midtraining improves SWE-bench Verified on all six harnesses, by 11.6 points on average. Midtraining on the natural data alone already improves over the base model on four of the six harnesses, by up to 4.7 points, and adding the traces gives the largest gains. We did not train on traces from a single scaffold, so these results do not separate the effect of scaffold diversity from that of adding traces.

<table><tr><td></td><td colspan="5">N-3.5 Super, short SFT</td><td colspan="4">N-3.5 Lightning-8K, full SFT</td></tr><tr><td></td><td colspan="3">SWE-bench Verified</td><td colspan="3">Terminal-Bench 2.1</td><td colspan="3">SWE-bench Verified</td></tr><tr><td>Harness</td><td>Base</td><td>Mid 65B</td><td>∆</td><td>Base</td><td>Mid 65B</td><td>∆</td><td>Mid 100B Base (no traces)</td><td>Mid 100B</td><td>∆</td></tr><tr><td>Pi</td><td>57.4</td><td>67.8</td><td>+10.4</td><td>30.3</td><td>35.2</td><td>+4.9</td><td>45.7</td><td>48.8 59.9</td><td>+14.2</td></tr><tr><td>OpenCode</td><td>54.4</td><td>65.8</td><td>+11.4</td><td>35.2</td><td>38.2</td><td>+3.0 52.0</td><td>50.1</td><td>63.1</td><td>+11.1</td></tr><tr><td>Hermes</td><td>59.8</td><td>63.2</td><td>+3.4</td><td>44.2</td><td>53.9</td><td>+9.7</td><td>43.3 43.3</td><td>50.0</td><td>+6.7</td></tr><tr><td>Codex</td><td>48.4</td><td>61.8</td><td>+13.4</td><td>27.0</td><td>39.7</td><td>+12.7</td><td>42.1 45.9</td><td>61.1</td><td>+19.0</td></tr><tr><td>Claude Code</td><td>53.8</td><td>58.0</td><td>+4.2</td><td>39.7</td><td>43.5</td><td>+3.8</td><td>46.0 50.7</td><td>60.9</td><td>+14.9</td></tr><tr><td>OpenHands</td><td>1</td><td></td><td></td><td></td><td></td><td></td><td>59.7</td><td>60.5 63.5</td><td>+3.8</td></tr><tr><td>Mean</td><td>54.8</td><td>63.3</td><td>+8.6</td><td>35.3</td><td>42.1</td><td>+6.8</td><td>48.1</td><td>49.9 59.8</td><td>+11.6</td></tr></table>

Table 7: Midtraining improves pass@1 on every agent harness we evaluated, for both models, with the largest gain on the harness where the base model is weakest (%).
<table><tr><td></td><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td><td>256K</td></tr><tr><td>8K → long-context @ 1M</td><td>94.1</td><td>91.9</td><td>90.5</td><td>85.9</td><td>81.3</td><td>76.6</td></tr><tr><td>8K → midtraining @256K</td><td>92.3</td><td>91.4</td><td>87.8</td><td>81.9</td><td>76.1</td><td>62.2</td></tr><tr><td>8K → midtraining @256K → long-context @ 1M</td><td>92.1</td><td>91.0</td><td>89.8</td><td>86.6</td><td>83.8</td><td>79.7</td></tr></table>

Table 8: N-3.5 Lightning RULER scores up to 256K. Midtraining alone brings N-3.5 Lightning-8K close to N-3.5 Lightning-LC up to 128K, acting like a long-context stage.

## 5 ANALYSIS AND ABLATIONS

## 5.1 MIDTRAINING AS A LONG-CONTEXT STAGE

Because midtraining trains at 256K, it also extends the model’s context. Table 8 compares the usual order, where N-3.5 Lightning-8K goes straight to the long-context stage, with a 100B-token N-3.5 Lightning midtraining run at 256K before a long-context stage. Midtraining alone brings N 3.5 Lightning-8K within about five points of the long-context base model on RULER (Hsieh et al., 2024) up to 128K, but trails by 14 points at 256K. After the long-context stage, the midtrained model beats the long-context base model at every length from 64K to 256K and stays within two points at shorter lengths. Midtraining thus does part of the work of the long-context stage, and this carries over to the final model. Appendix E.2 gives more results for N-3.5 Super.

## 5.2 MASKING TOOL RESPONSES IN TRACES

Agent traces interleave the model’s own actions with tool responses, the outputs the environment returns after each tool call. Some recent work also trains on tool responses, so that the model learns to predict how the environment reacts, a form of world modeling (FAIR CodeGen Team, 2025; Zeng et al., 2026).

We tested this with the setup of §4.2, midtraining N-3.5 Lightning-8K for 100B tokens with and without tool responses in the loss, followed by identical SFT with an earlier version of our SFT recipe. In our experiments, training on tool responses did not work out of the box and lowered SWE-bench Verified pass@1 by about 10 points (Table 9), so we train only on assistant turns, as in SFT. Tool responses are often long and noisy, such as file dumps, test logs, and error traces, and predicting them may take capacity away from learning the agent’s own actions.

## 5.3 PRETRAINING REPLAY AND LEARNING RATE

Replay keeps the model close to its pretraining distribution, so that midtraining shifts the data distribution gradually rather than abruptly. We started from 20% replay as a reasonable default, with the goal of no regression on general-capability benchmarks. Table 10 shows that N-3.5 Super does not regress through 65B tokens, with every benchmark group within 1.2 points of the starting checkpoint. We also tried 40% replay, which gave no clear benefit over 20%, so we kept 20%.

<table><tr><td>SWE-V</td><td>pass@1</td><td>pass@3</td></tr><tr><td>Unmasked</td><td>27.5</td><td>41.6</td></tr><tr><td>Masked</td><td>37.8</td><td>53.0</td></tr></table>

Table 9: Tool-response masking. N-3.5 Lightning-8K midtrained for 100B tokens with and without tool responses in the loss, followed by identical SFT with an earlier version of our SFT recipe (SWEbench Verified).
<table><tr><td>Midtraining</td><td>Knowledge</td><td>Math</td><td>Code</td><td>Commonsense</td></tr><tr><td>0 (base)</td><td>79.7</td><td>86.4</td><td>80.8</td><td>87.5</td></tr><tr><td>25B</td><td>79.9</td><td>85.9</td><td>80.0</td><td>87.1</td></tr><tr><td>65B</td><td>79.8</td><td>87.0</td><td>79.6</td><td>86.7</td></tr></table>

<table><tr><td>LR</td><td>MMLU</td><td>MMLU-Pro</td><td>MATH-500</td></tr><tr><td>Base</td><td>79.4</td><td>64.3</td><td>79.7</td></tr><tr><td>1e-5</td><td>78.5</td><td>66.3</td><td>78.8</td></tr><tr><td>3e-5</td><td>76.7</td><td>61.5</td><td>74.3</td></tr><tr><td>5e-5</td><td>76.8</td><td>63.3</td><td>61.7</td></tr></table>

Table 10: N-3.5 Super pretraining benchmarks before and after midtraining with 20% replay.  
Table 11: N-3.5 Lightning-8K after 100B midtraining tokens at three learning rates.

The learning rate also affects general capabilities. Pretraining ends with a small, decayed learning rate, and midtraining needs a higher one to learn the new data. In our early runs, midtraining at 1e-5 gave smaller gains after SFT than higher rates did. Raising the rate too far, however, hurts general capabilities even with replay. Table 11 compares a constant 1e-5 with warming up to 3e-5 or 5e-5. Both higher rates lower general capabilities, most clearly MATH-500, which falls from 79.7 to 61.7 at 5e-5. We therefore recommend a moderate increase and checking general-capability benchmark while tuning it.

## 6 CONCLUSION

We presented an open midtraining recipe for agentic coding, with full details of how each data source is built, filtered, and weighted and how the models are trained. After identical SFT, the recipe gives clear gains as midtraining scales to 500B tokens on N-3.5 Lightning, and it carries over to the larger N-3.5 Super, where it raises DeepSWE, the hardest benchmark we evaluate, from 11.7 to 15.7 and also improves SWE-bench Verified and BrowseComp. On both models, midtraining improves results on every agent harness we tested. Our experiments show that agentic traces are essential, since natural software-engineering data alone gives much smaller gains after SFT, and that a short SFT run gives a more direct post-training signal than base-model post-trainability metrics, which remain useful to track. Midtraining at 256K also works as an intermediate long-context stage, and we explain our other modeling choices, such as tool-response masking and pretraining replay.

Limitations and future work. Agentic benchmarks are noisy, and each comparison is a single run, so differences of one or two points, especially on Terminal-Bench, may come from run-to-run variation. We did not compare against spending the same tokens on longer pretraining or SFT, since midtraining adds a separate stage that complements SFT rather than replacing it. We tested the recipe only on Nemotron models, not on other model families, and mainly on agentic coding. Our SFT evaluations use an agentic-focused SFT blend rather than a full general-purpose post-training blend. Our RL results are also limited, since clear RL gains on top of a strong SFT checkpoint were difficult to get at the N-3.5 Lightning scale and RL was too compute intensive at the N-3.5 Super scale, so we report only a short, single-seed run on N-3.5 Lightning-8K (Appendix D). For future work, since traces drove most of the gains, we plan to scale trace data with more diversity, and we have already expanded our trace corpus to about 90B tokens. We also plan to extend the agentic data to more capabilities and to move more post-training data, such as reasoning and chat data, into midtraining.

## ACKNOWLEDGMENTS

We thank Perplexity, a partner in the Nemotron Coalition, for contributing agentic trace data to the midtraining blend, and in particular Dongqi Wu, Chuan Wang, Bo Wang, and Hang Wu. We also thank the NVIDIA Nemotron pretraining and post-training teams, whose models, data, and training stack this work builds on. We especially thank Luis Vega for the agent-harness sweeps, Krishna Puvvada and Abhinav Khattar for help with the long-context and pretraining data setup, Wasi Uddin Ahmad and Sugam Devare for agentic and software-engineering data, and Jiantao Jiao and Zhenghao Xu for discussions on base-model evaluation. We are grateful to Igor Gitman, Somshubra Majumdar, Makesh Narsimhan Sreedhar, Surya Nistala, Ritu Gala, Abhibha Gupta, and Tan Yu for valuable discussions and help. We thank Bryan Catanzaro, Jonathan Cohen, Oleksii Kuchaiev, and Mohammad Shoeybi for their support, and the compute and infrastructure teams who made this work possible.

## REFERENCES

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020. URL https://arxiv.org/abs/1911.11641.

Ruisheng Cao et al. Qwen3-Coder-Next technical report. arXiv preprint arXiv:2603.00729, 2026.

Guillaume Cardoen, Tom Mens, and Alexandre Decan. A dataset of GitHub Actions workflow histories. In Proceedings of the 21st International Conference on Mining Software Repositories (MSR), pp. 677–681. ACM, 2024. doi: 10.1145/3643991.3644867. Dataset version 2024-10-25, Zenodo record 13985548, https://doi.org/10.5281/zenodo.13985548.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Yangyi Chen et al. Scaling laws for predicting downstream performance in LLMs. Transactions on Machine Learning Research, 2025. URL https://arxiv.org/abs/2410.08527.

Neil Chowdhury et al. Introducing SWE-bench Verified. https://openai.com/index/ introducing-swe-bench-verified/, 2024.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Cursor Research. Composer 2 technical report. arXiv preprint arXiv:2603.24477, 2026.

DeepSeek-AI. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Xiang Deng, Jeff Da, Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg, Niklas Lauffer, Andrew Park, et al. SWE-Bench Pro: Can AI agents solve long-horizon software engineering tasks? In International Conference on Machine Learning (ICML), 2026. URL https:// arxiv.org/abs/2509.16941.

Hantian Ding et al. Fewer truncations improve language modeling. In International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2404.10830.

Yangruibo Ding et al. CrossCodeEval: A diverse and multilingual benchmark for cross-file code completion. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023. URL https://arxiv.org/abs/2310.11248.

FAIR CodeGen Team. CWM: An open-weights LLM for research on code generation with world models. arXiv preprint arXiv:2510.02387, 2025.

Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen. How to train long-context language models (effectively). In Proceedings ofthe Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2025. URL https://arxiv.org/abs/2410.02660.

GLM-5 Team. GLM-5: From vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Daya Guo et al. DeepSeek-Coder: When the large language model meets programming – the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR), 2021a. URL https://arxiv.org/abs/2009.03300.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2021b. URL https://arxiv.org/abs/2103.03874.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? In Conference on Language Modeling (COLM), 2024. URL https://arxiv.org/ abs/2404.06654.

Wenqi Huang, Charley Lee, Leonard Tng, and Serena Ge. DeepSWE: Measuring frontier coding agents on original, long-horizon engineering tasks. arXiv preprint arXiv:2607.07946, 2026.

Binyuan Hui, Jian Yang, Zeyu Cui, et al. Qwen2.5-Coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2403.07974.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/ abs/2310.06770.

Feiyang Kang, Michael Kuchnik, Karthik Padthe, Marin Vlastelica, Ruoxi Jia, Carole-Jean Wu, and Newsha Ardalani. Quagmires in SFT-RL post-training: When high SFT scores mislead and what to use instead. arXiv preprint arXiv:2510.01624, 2025.

Emmy Liu, Graham Neubig, and Chenyan Xiong. Midtraining bridges pretraining and posttraining distributions. In International Conference on Machine Learning (ICML), 2026a. URL https: //arxiv.org/abs/2510.14865.

Hong Liu, Sang Michael Xie, Zhiyuan Li, and Tengyu Ma. Same pre-training loss, better downstream: Implicit bias matters for language models. In Proceedings of the 40th International Conference on Machine Learning (ICML), pp. 22188–22214, 2023.

Runze Liu, Jiashun Liu, Xu Wan, Yuqian Fu, and Ling Pan. When RL fails after SFT: Rejuvenating model plasticity for robust SFT-to-RL handoff. arXiv preprint arXiv:2606.09932, 2026b.

Tianyang Liu, Canwen Xu, and Julian McAuley. RepoBench: Benchmarking repository-level code auto-completion systems. In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv.org/abs/2306.03091.

Anton Lozhkov et al. StarCoder 2 and The Stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024.

Nikolai Ludwig, Wasi Uddin Ahmad, Somshubra Majumdar, and Boris Ginsburg. From SWE-ZERO to SWE-HERO: Execution-free to execution-based fine-tuning for software engineering agents. In Conference on Language Modeling (COLM), 2026. URL https://arxiv.org/ abs/2604.01496.

Mentat AI Team (AbanteAI). LoCoDiff: Natural long context code bench. https://abanteai. github.io/LoCoDiff-bench/, 2025.

Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, et al. Terminal-Bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv. org/abs/2601.11868.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018. URL https://arxiv.org/abs/1809.02789.

Florent Moriconi, Thomas Durieux, Jean-Remy Falleri, Rapha ´ el Troncy, and Aur ¨ elien Francillon.´ GHALogs: Large-scale dataset of GitHub Actions runs. In Proceedings of the 22nd IEEE/ACM International Conference on Mining Software Repositories (MSR), pp. 669–673. IEEE, 2025. doi: 10.1109/MSR66628.2025.00104.

NVIDIA. NVIDIA Nemotron 3: Efficient and open intelligence. arXiv preprint arXiv:2512.20856, 2025.

NVIDIA. NVIDIA Nemotron 3.5 Super and Lightning Technical Report. Technical report, NVIDIA, 2026a. URL https://research.nvidia.com/labs/nemotron/files/ NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf.

NVIDIA. Nemotron 3 Super: Open, efficient mixture-of-experts hybrid Mamba-transformer model for agentic reasoning. arXiv preprint arXiv:2604.12374, 2026b.

Jiarui Qin et al. APTBench: Benchmarking agentic potential of base LLMs during pre-training. arXiv preprint arXiv:2510.24397, 2025.

Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial Winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020. URL https://arxiv.org/abs/1907.10641.

Maksim Sapronov and Evgeniy Glukhov. On pretraining for project-level code completion. arXiv preprint arXiv:2510.13697, 2025.

Zhihong Shao et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Weijia Shi et al. In-context pretraining: Language modeling beyond document boundaries. In International Conference on Learning Representations (ICLR), 2024. URL https://arxiv. org/abs/2310.10638.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Konrad Staniszewski et al. Structured packing in LLM training improves long context utilization. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025. URL https://arxiv. org/abs/2312.17296.

Liangcai Su et al. Scaling agents via continual pre-training. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2509.13310.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, et al. OpenHands: An open platform for AI software developers as generalist agents. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2407.16741.

Xingyao Wang, Simon Rosenberg, Juan Michelini, Calvin Smith, Hoang Tran, Engel Nyst, Rohit Malhotra, Xuhui Zhou, et al. The OpenHands software agent SDK: A composable and extensible foundation for production agents. In Proceedings of Machine Learning and Systems (MLSys), 2026. URL https://arxiv.org/abs/2511.03690.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, et al. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2024. URL https://arxiv.org/abs/2406.01574.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, et al. BrowseComp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), 2024. URL https: //arxiv.org/abs/2405.15793.

John Yang, Kilian Lieret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, et al. SWE-smith: Scaling data for software engineering agents. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025. URL https: //arxiv.org/abs/2504.21798.

Zonghan Yang et al. Kimi-Dev: Agentless training as skill prior for SWE-Agents. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/ 2509.23045.

Tan Yu, Alexander Bukharin, Khushi Bhardwaj, Jennifer Williams, Zirui Liu, Jonathan Lingjie Li, Soumye Singhal, et al. Before they can solve: Predicting post-training coding-agent performance from base models. arXiv preprint arXiv:2610.10478, 2026.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/2504.13837.

Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Linhao Zhang, Shulin Xin, Lu Chen, Qi Liu, et al. Multi-SWE-bench: A multilingual benchmark for issue resolving. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025. URL https://arxiv.org/abs/2504.02605.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics (ACL), 2019. URL https://arxiv.org/abs/1905. 07830.

Ji Zeng et al. daVinci-Dev: Agent-native mid-training for software engineering. In International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2601. 18418.

Dylan Zhang, Yufeng Xu, Haojin Wang, Qingzhi Chen, and Hao Peng. Good SFT optimizes for SFT, better SFT prepares for reinforcement learning. In International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2602.01058.

Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, and Michael R. Lyu. Loghub: A large collection of system log datasets for AI-driven log analytics. In Proceedings ofthe 34th IEEE International Symposium on Software Reliability Engineering (ISSRE), pp. 355–366. IEEE, 2023. doi: 10. 1109/ISSRE59848.2023.00071. URL https://github.com/logpai/loghub.

![](images/0616422c484775116336b3b87cccc6155fee58847d598e4436efbbf35c250b36.jpg)  
Figure 2: Document lengths in tokens for each source in the midtraining blend (log scale).

## A MIDTRAINING DATA

This appendix documents every source in the blend, including how it is collected, filtered, and weighted, and Appendix B lists the full training settings. We plan to release data-processing scripts and training-set metadata in a future version.

## A.1 ISSUE–PR

For PRs linked to an issue, an LLM judge checks whether the issue and the PR match. The judge reads the full issue and PR text, the final patch, and the paths of the changed files, and labels each pair as clean or assigns one of seven failure modes, namely a revert, an incidental issue link, a wrong issue reference, a partial fix, an issue with no clear action, an underspecified issue, or a PR that does much more than the issue asks. We keep clean pairs and remove pairs that cannot be fixed. We repair the others by linking the PR to the correct issue, narrowing its scope, keeping it as a PR-only example, or splitting it from unrelated changes. For PRs without a linked issue, we remove those that change only README files or three lines or fewer.

## A.2 REPOSITORIES

The repository data starts from the code repositories in the Nemotron 3 pretraining corpus (NVIDIA, 2025), a deduplicated collection of public repositories with permissive licenses, and covers 14 languages (Table 12).

Connectedness. For each language, we parse the references that link one file to another, such as imports in Python, includes in C and C++, and module paths in Go. A repository’s connected fraction is the share of its code files that import another file in the repository, and we keep repositories where this share is at least 20%. For Ruby we lower the threshold to 10%, because Rails autoloading hides most references from static parsing. We exclude languages with little cross-file structure that parsing can recover, such as Shell, R, and Swift, as well as markup and configuration languages.

Serialization. The code files follow the README in depth-first order over the import graph, so the files of each connected component stay together and each file sits near the files it imports. Each file starts with a header that gives the repository name and file path, and the header is masked from the loss.

Filtering. We deduplicate files by path within each repository and drop Python 2 code, removing the whole repository when most of it is Python 2. We drop repositories that contain server-side secrets, such as cloud or API keys, and redact client-side keys, which are common in mobile and web projects. We also strip license and auto-generated headers, and drop repositories above 2M tokens or those that lose most of their files in deduplication. To avoid contamination, we remove SWE-bench repositories and their forks, as well as files that contain HumanEval, MBPP, or LiveCodeBench (Jain et al., 2025) solutions.

Blend weights. We start from each language’s natural token share, upweight Python, and downweight Dart, Scala, and Lua (Table 12).
<table><tr><td>Language</td><td>Records</td><td>Tokens (B)</td><td>Natural share (%)</td><td>Blend weight (%)</td></tr><tr><td>Python</td><td>1,798,331</td><td>24.51</td><td>23.5</td><td>31.0</td></tr><tr><td>Java</td><td>945,513</td><td>18.71</td><td>18.0</td><td>17.0</td></tr><tr><td>TypeScript</td><td>1,273,349</td><td>13.20</td><td>12.7</td><td>12.0</td></tr><tr><td>C#</td><td>460,597</td><td>11.66</td><td>11.2</td><td>10.0</td></tr><tr><td>JavaScript</td><td>1,289,843</td><td>10.01</td><td>9.6</td><td>8.6</td></tr><tr><td>C++</td><td>400,540</td><td>8.27</td><td>7.9</td><td>7.6</td></tr><tr><td>PHP</td><td>329,934</td><td>5.30</td><td>5.1</td><td>4.7</td></tr><tr><td>Dart</td><td>386,648</td><td>3.24</td><td>3.1</td><td>1.4</td></tr><tr><td>C</td><td>106,799</td><td>2.81</td><td>2.7</td><td>2.7</td></tr><tr><td>Rust</td><td>115,314</td><td>2.22</td><td>2.1</td><td>1.9</td></tr><tr><td>Go</td><td>119,005</td><td>1.68</td><td>1.6</td><td>1.5</td></tr><tr><td>Ruby</td><td>89,846</td><td>1.00</td><td>1.0</td><td>0.9</td></tr><tr><td>Scala</td><td>39,618</td><td>0.99</td><td>1.0</td><td>0.4</td></tr><tr><td>Lua</td><td>21,770</td><td>0.51</td><td>0.5</td><td>0.3</td></tr><tr><td>Total</td><td>7,377,107</td><td>104.12</td><td>100.0</td><td>100.0</td></tr></table>

Table 12: Repository data by language. Tokens are measured with the model tokenizer.

## A.3 DIFF DATA

Multi-file edits come from merged pull requests in the Issue–PR corpus, with the full patch and every changed file before the change. Code-evolution documents come from the full git histories of 6,550 permissively licensed repositories from the same corpus, chosen to match the language mix of SWEbench Multilingual. We keep only patches that apply exactly and histories that replay byte for byte. In multi-file edits, the loss covers the original files, the diffs, and the changed lines of the new files with two lines of context. In code-evolution documents, it covers the commit messages, diffs, and final state, and license blocks are masked in both. We remove all repositories used in SWE-bench Verified, SWE-bench Pro, Multi-SWE-bench (Zan et al., 2025), Terminal-Bench, and LoCoDiff, deduplicate examples, and drop GPL-licensed documents, secrets, lockfiles, and generated files. Histories have a median of 3 commits and up to 59, and the median document is about 6K tokens (Table 13).

<table><tr><td>Type</td><td>Examples</td><td>Document tokens</td><td>Trained tokens</td><td>Trained share</td></tr><tr><td>Multi-file edits</td><td>733,705</td><td>13.3B</td><td>6.8B</td><td>51%</td></tr><tr><td>Code evolution</td><td>377,618</td><td>3.5B</td><td>3.2B</td><td>91%</td></tr><tr><td>Total</td><td>1,111,323</td><td>16.8B</td><td>10.0B</td><td>60%</td></tr></table>

Table 13: Diff data used in our experiments. Trained tokens exclude masked context. Counts use the tokenizer of our data pipeline. With the model tokenizer, the total is 16.0B (Table 1).

## A.4 BUILD AND CI LOGS

Raw logs come from the full LogHub archives (BGL, Thunderbird, HPC, HDFS, Spark, and Windows). We cut them into non-overlapping windows of 120 context lines followed by 16 target lines. We deduplicate HDFS logs after masking block identifiers, hosts, and timestamps. No single archive is more than 28% of this source. Supercomputer logs are at least 30%, and CI/CD logs from GHALogs run logs and workflow-file histories are at least 12%. We check every source against a license registry and exclude sources with unknown licenses. We also build 632 diagnosis examples. Each centers a window of HPC logs on a warning, error, or critical event, and a frontier model writes the diagnosis from the window and from findings that simple rules extract. We redact secrets and mask timestamps, hostnames, hashes, and repeated boilerplate from the loss.

<table><tr><td>Scaffold label</td><td>Trajectories</td><td>Share (%)</td></tr><tr><td>codex</td><td>166,624</td><td>19.39</td></tr><tr><td>openhands</td><td>120,086</td><td>13.98</td></tr><tr><td>generic_code_cli</td><td>107,291</td><td>12.49</td></tr><tr><td>gemini_cli</td><td>104,717</td><td>12.19</td></tr><tr><td>terminus</td><td>99,915</td><td>11.63</td></tr><tr><td>taubench_react</td><td>50,399</td><td>5.87</td></tr><tr><td>swe_agent</td><td>46,162</td><td>5.37</td></tr><tr><td>opencode</td><td>44,920</td><td>5.23</td></tr><tr><td>mini_swe_agent</td><td>35,260</td><td>4.10</td></tr><tr><td>search_browse</td><td>26,168</td><td>3.05</td></tr><tr><td>unknown</td><td>25,866</td><td>3.01</td></tr><tr><td>chat</td><td>20,426</td><td>2.38</td></tr><tr><td>claude_code</td><td>11,342</td><td>1.32</td></tr><tr><td>Total</td><td>859,176</td><td>100.00</td></tr></table>

Table 14: Trajectories by scaffold label in the early snapshot. Shares are of trajectories, not tokens.
<table><tr><td>Tool name</td><td>Calls</td><td>Share (%)</td></tr><tr><td>bash</td><td>3,659,199</td><td>19.6</td></tr><tr><td>execute_bash</td><td>3,633,328</td><td>19.4</td></tr><tr><td>str_replace_editor</td><td>2,913,745</td><td>15.6</td></tr><tr><td>keystrokes</td><td>2,029,095</td><td>10.9</td></tr><tr><td>exec_command</td><td>736,655</td><td>3.9</td></tr><tr><td>write</td><td>524,113</td><td>2.8</td></tr><tr><td>browser.search</td><td>317,169</td><td>1.7</td></tr><tr><td>shell</td><td>280,803</td><td>1.5</td></tr><tr><td>read</td><td>261,285</td><td>1.4</td></tr><tr><td>todowrite</td><td>260,147</td><td>1.4</td></tr></table>

Table 15: The ten most frequent tool names in the early snapshot, as each scaffold records them, so similar tools such as bash and execute bash are listed separately. Shares are of all tool calls.

## A.5 AGENTIC TRACES

The blend in Table 1 uses 39.4B tokens of traces. The statistics below come from an early filtered snapshot of 859,176 trajectories. Scaffold labels name the agent interface, not the task domain. For example, a software-engineering task can use shell commands, file-editing tools, or both. We downsample general tool-use data, mainly taubench react, to give more weight to software engineering. Tables 14, 15, and 16 give the scaffolds, the most frequent tools, and the number of assistant turns per trajectory. In the snapshot, filtering (§3) removed 8,935 trajectories, about 1%, mostly unfinished runs (6,960) or repeated loops that ended in failure (1,959).

## B TRAINING SETUP

Table 17 lists the settings for midtraining and SFT. All runs use Megatron-LM (Shoeybi et al., 2019) on NVIDIA GB300 GPUs, with inter-document masking so that documents packed into the same sequence do not attend to each other. Both models keep their two multi-token-prediction (MTP) layers during midtraining.

<table><tr><td>Assistant turns</td><td>Trajectories</td><td>Share (%)</td></tr><tr><td>1</td><td>146,334</td><td>17.03</td></tr><tr><td>2</td><td>56,929</td><td>6.63</td></tr><tr><td>3</td><td>47,362</td><td>5.51</td></tr><tr><td>4-5</td><td>77,700</td><td>9.04</td></tr><tr><td>6-10</td><td>138,036</td><td>16.07</td></tr><tr><td>11-20</td><td>111,777</td><td>13.01</td></tr><tr><td>21-50</td><td>161,295</td><td>18.77</td></tr><tr><td>51-100</td><td>105,711</td><td>12.30</td></tr><tr><td>101-200</td><td>12,833</td><td>1.49</td></tr><tr><td>201-300</td><td>1,195</td><td>0.14</td></tr><tr><td>301-400</td><td>4</td><td>&lt; 0.01</td></tr><tr><td>401+</td><td>0</td><td>0.00</td></tr><tr><td>Total</td><td>859,176</td><td>100.00</td></tr></table>

Table 16: Trajectory length in assistant turns, in the early snapshot.
<table><tr><td></td><td>N-3.5 Lightning midtraining</td><td>N-3.5 Super midtraining</td><td>SFT (both models)</td></tr><tr><td>Parameters (total / active)</td><td>30B / 3B</td><td>120B / 12B</td><td></td></tr><tr><td>Sequence length (packed)</td><td>256K</td><td>256K</td><td>256K</td></tr><tr><td>Global batch size (sequences)</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Peak learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup (sequences)</td><td>32K</td><td>30K</td><td></td></tr><tr><td>Schedule after warmup Optimizer</td><td>constant Adam,  $\beta = ( 0 . 9 , 0 . 9 5 )$ </td><td>constant</td><td>cosine to  $5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Precision</td><td>BF16</td><td>, weight decay 0.1, gradient clip 1.0</td><td></td></tr><tr><td>Parallelism (TP / CP / EP)</td><td>2/4/8</td><td>2 /8/64</td><td></td></tr><tr><td>MTP layers (loss weight)</td><td>2 (0.1)</td><td>2 (0.3)</td><td></td></tr><tr><td>Tokens</td><td>up to 500B</td><td>65B and 280B</td><td>33B or 100B</td></tr></table>

Table 17: Training settings. TP, CP, and EP are tensor, context, and expert parallelism, and MTP is multi-token prediction.

## B.1 SFT DATA

Both models use the same SFT data and settings. Our SFT evaluations use an agentic-focused SFT blend of about 430K trajectories from several scaffolds, most of them generated by GLM-5.2. They cover software engineering, terminal use, search, conversational tool use, and knowledge work.

## C EVALUATION

## C.1 AGENTIC BENCHMARKS

For all agentic benchmarks, we sample with temperature 1.0 and top-p 0.95. SWE-bench (Jimenez et al., 2024) Verified (Chowdhury et al., 2024) (500 tasks), SWE-bench Multilingual (Yang et al., 2025) (300), and SWE-bench Pro (Deng et al., 2026) (731) use the OpenHands agent (Wang et al., 2025; 2026) (SDK version 1.17.0) with 3 rollouts per task. Each rollout has at most 200 turns, or 250 for SWE-bench Pro. Terminal-Bench 2.0 (Merrill et al., 2026) (89 tasks) uses the Terminus-2 agent (Merrill et al., 2026) with a one-hour limit per task and 8 rollouts per task. The harness sweep and the RL evaluation use Terminal-Bench 2.1, which has the same 89 tasks with 28 of them fixed, and we compare scores only within a version. DeepSWE (Huang et al., 2026) pass@1 is averaged over 4 runs. BrowseComp (Wei et al., 2025) is evaluated on a subset of the benchmark in a custom agent harness, with Exa as the search provider. For N-3.5 Super, run-to-run variation is about 1 point on SWE-bench Verified and Pro and about 2 points on DeepSWE. We report pass@1 as the mean success rate over rollouts and compute pass@k from the same rollouts.

General-capability benchmarks. The regression checks use the base-model evaluation suite from NVIDIA (2026b). General knowledge averages MMLU (Hendrycks et al., 2021a) (5-shot) and MMLU-Pro (Wang et al., 2024) (5-shot, chain of thought). Math averages GSM8K (Cobbe et al., 2021) (8-shot) and MATH (Hendrycks et al., 2021b) (4-shot). Code averages HumanEval (Chen et al., 2021) (0-shot) and MBPP-Sanitized (Austin et al., 2021) (3-shot). Commonsense understanding averages ARC-Challenge (Clark et al., 2018), HellaSwag (Zellers et al., 2019), OpenBookQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), and WinoGrande (Sakaguchi et al., 2020).

## C.2 BASE-MODEL POST-TRAINABILITY METRICS

We track four base-model post-trainability metrics during midtraining.

• LoCoDiff (Mentat AI Team (AbanteAI), 2025) gives the model the full git history of a file and asks it to write the file’s final state. We use all 198 cases, zero-shot with greedy decoding and a 16K-token generation budget, which covers the longest reference file (11.7K tokens). Exact match is the share of files reproduced byte for byte, and normalized edit distance is capped at 1.2 per case. Greedy decoding still varies slightly across runs because inference is not bit-exact, so we average both over 3 runs.

• RepoBench (Liu et al., 2024) measures next-line code completion that requires context from other files in the repository. We report edit similarity in the cross-file-first setting, with greedy decoding.

• Repository bits-per-byte follows Composer 2 (Cursor Research, 2026). We concatenate the files of held-out repositories and compute bits-per-byte over long windows. Our version puts the README first, orders files by dependency, keeps the file tree across windows, and does not score file headers.

• Trace bits-per-byte is measured on the assistant turns of held-out agent traces, following the decisive-action bits-per-byte of Yu et al. (2026). The version in Table 4 scores only the decisive turns of DeepSWE traces, the turns that most affect the outcome, and masks tool-call formatting, which otherwise dominates the score.

## D PERSISTENCE OF GAINS THROUGH REINFORCEMENT LEARNING

As a preliminary check on whether midtraining gains persist under RL, we run an identical short stage of RL with verifiable rewards (RLVR), 20 GRPO steps (Shao et al., 2024) over 14 verifiable environments (Appendix D.2), from N-3.5 Lightning-8K and its midtrained checkpoint after ful SFT. Twenty steps is too short to measure an RL gain, so we only ask whether the midtrained model’s advantage survives the start of RL.

## D.1 RESULTS

<table><tr><td>SWE-V (pass@1 / pass @3)</td><td>SFT</td><td>RL step 10</td><td>Step 15</td><td>Step 20</td></tr><tr><td>Base-8K Midtrained-100B</td><td>61.2 / 71.8 63.2 / 74.1</td><td>60.4 / 70.8 63.0 / 74.0</td><td>61.4 / 72.2 63.5 / 73.7</td><td>60.4 / 71.7 63.0 / 73.0</td></tr><tr><td>TB 2.1 (pass@ 1 / pass@8)</td><td>SFT</td><td>RL step 10</td><td>Step 15</td><td>Step 20</td></tr><tr><td>Base-8K</td><td>23.0 / 46.1</td><td>24.7 / 44.9</td><td></td><td></td></tr><tr><td>Midtrained-100B</td><td>24.2 / 44.9</td><td>23.5 / 44.9</td><td>22.9 / 41.6 24.4 / 44.9</td><td>20.8 / 39.3 23.2 / 46.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 18: RLVR from N-3.5 Lightning-8K and its midtrained checkpoint after full SFT. SFT-start scores are re-evaluated with the RL checkpoints. The midtrained model leads on SWE-bench, and its Terminal-Bench pass@8 holds.

The midtrained model starts about two points ahead on SWE-bench Verified pass@1 and keeps that lead at every RL step, and both models stay close to their SFT scores (Table 18). On Terminal-Bench 2.1, the base model’s pass@8 falls from 46.1 to 39.3 over 20 steps while the midtrained model’s holds at 45 to 46, although with 89 tasks this drop amounts to only six tasks. Training-time signals, including mean reward, generation length, and format violations, also favor the midtrained model (Appendix D.2).

With a single seed and 20 steps, these results do not show that midtraining increases what RL can add, only that the midtrained model’s advantage after SFT is not lost when RL begins.

## D.2 SETUP AND TRAINING SIGNALS

The RLVR prompt blend is drawn from 14 verifiable environments (tool use, code generation, SWE and search pivots, instruction following, reasoning, and formatting). We reward-profile a seed blend against both checkpoints and keep only prompts with mixed rewards, meaning that for each checkpoint some but not all of its sampled rollouts succeed, so every prompt is neither trivial nor unsolvable for either model. Training is GRPO with a leave-one-out baseline and no reference-policy KL penalty, 256 prompts × 16 generations per step, a 32K-token output cap within a 131K context, a learning rate of $1 0 ^ { - 6 }$ with 10 warmup steps, and 20 optimizer updates. The two runs differ only in their starting checkpoint; they use the same blend, seed, and prompt order. We evaluate SWE-bench Verified with 5 rollouts per task and Terminal-Bench 2.1 with 8, for both the SFT starting points and the RL checkpoints.

Training-time metrics, averaged over matched steps, favor the midtrained model. It earns 4.3% higher train reward (0.308 vs. 0.295) with 12% shorter generations (4.6K vs. 5.3K mean tokens), and it produces format violations (a stray end-of-sequence token or malformed thinking tags, which zero the reward) about ten times less often (0.03% vs. 0.34% of trajectories). Its truncation rate is also lower (0.056% vs. 0.073%). Together these make RL from the midtrained checkpoint about 15% faster per trajectory on the same hardware.

## E ADDITIONAL RESULTS

## E.1 CONTEXT-LIMIT FAILURES

Rollouts that fill the 256K context window count as failures. Under the same harness and settings, they are much more common without midtraining (Table 19), especially on SWE-bench Pro. On SWE-bench Verified, rollouts also hit the 200-turn cap 11 times without midtraining and once after 500B tokens. Midtraining appears to make the agent more economical with its context and turns, not only more accurate.

<table><tr><td>Benchmark</td><td>Rollouts</td><td>N-3.5 Lightning-LC</td><td>Midtrained, 500B</td></tr><tr><td>SWE-bench Verified</td><td>1,500</td><td>40</td><td>26</td></tr><tr><td>SWE-bench Multilingual</td><td>900</td><td>25</td><td>14</td></tr><tr><td>SWE-bench Pro</td><td>2,193</td><td>91</td><td>13</td></tr><tr><td>Terminal-Bench</td><td>712</td><td>8</td><td>1</td></tr></table>

Table 19: Rollouts that fail by filling the 256K context window, after identical full SFT.

## E.2 LONG-CONTEXT RESULTS AT 1M

To keep evaluation simple, our N-3.5 Super runs start midtraining from the 1M long-context checkpoint (§3.1). Midtraining then trains at 256K, so RULER at 1M naturally drops, from 79.9 to 33.0 after 25B tokens. A short long-context stage of about 9B tokens, on long-context data only, brings it back to 81.0, above the starting checkpoint, and the 65B N-3.5 Super checkpoint evaluated in §4.4 includes this short stage, while the 280B checkpoint goes straight to SFT. This drop is a side effect of our evaluation setup. In a mature pipeline, midtraining would come before the final long-context stage, as in §5.1.