# Beyond Consensus: Downward Bias and Role Asymmetry in Multi-Agent LLM Judges for Subjective Evaluation

Minsoo Song<sup>1</sup>, Chanwoo Kim<sup>1</sup>, Sugyeong Eo<sup>2,</sup> <sup>†</sup>, Chanjun Park<sup>1,†</sup>

<sup>1</sup>Soongsil University <sup>2</sup> Yonsei University {ecoses042, kcw4776}@soongsil.ac.kr, s.eo@yonsei.ac.kr, chanjun.park@ssu.ac.kr

## Abstract

Multi-Agent Debate (MAD) has been widely adopted to improve LLM-based evaluation by prompting multiple agents to negotiate and reach a consensus. However, for subjective rubric-based scoring, inter-agent agreement does not guarantee alignment with human judgments. In this paper, we compare a single-judge baseline against a consensus-based MAD protocol on subjective evaluation tasks and design three ablations to isolate the impact of role prompting, multi-round interaction, and explicit score sharing. Evaluations across six LLMs show that the single-judge baseline achieves the strongest human alignment on average across six judge models, whereas MAD shows degradation in human alignment on both tasks. Our ablations demonstrate that this performance drop stems primarily from asymmetric role prompting rather than the interaction itself. Specifically, assigning a strict judge role introduces a systematic downward bias that the consensus process fails to correct. The central finding is that this bias reflects strict-stance dominance beyond averaging: the consensus score falls well beyond the arithmetic midpoint of the standalone strict and lenient conditions, rather than averaging them out. Removing role asymmetry (Symmetric MAD) largely recovers baseline performance, while masking peer scores widens inter-agent disagreement on average and worsens average human alignment. These findings demonstrate that multi-agent consensus can enforce artificial agreement at the expense of true human alignment, reveal ing a structural limitation in consensus-style, role-specialized MAD protocols for subjective scoring.

## 1 Introduction

LLM-as-a-Judge (LLM judge) has become a widely adopted paradigm for evaluating openended language generation tasks. Prior studies have shown that LLMs can provide scalable and rubric-aware evaluations for summarization, dialogue, and long-form responses, often showing stronger correlations with human judgments than traditional metrics (Liu et al., 2023; Zhu et al., 2025; Kim et al., 2024). However, LLM judges are also known to exhibit systematic biases, such as self-preference, position bias, and unfair evaluation behavior (Wataoka et al., 2024; Shi et al., 2025; Wang et al., 2024).

![](images/61636ba38695cc720c5916f66641114774788c62b1f32da172bf735a3ac986ad.jpg)  
Figure 1: Overview of the experimental design. We compare a single-judge baseline with consensus-based MAD and three ablations that isolate role prompting, debate interaction, and score sharing, using RMSE and Spearman correlation against human reference scores.

In parallel, Multi-Agent Debate (MAD) has been proposed as a way to improve the reasoning and factuality of LLM outputs by allowing multiple agents to exchange and revise their answers (Du et al., 2024; Liang et al., 2024). Recent work has extended this idea to LLM-based evaluation, introducing multi-agent or debate-based judges for openended text evaluation, evaluator meta-evaluation, and machine translation evaluation (Chan et al., 2024; Li et al., 2024; Chern et al., 2024; Feng et al., 2025; Chen et al., 2025). Such applications are supported by evidence that MAD can be effective in reasoning-oriented tasks where answers are relatively verifiable, such as factual question answering or mathematical reasoning (Liang et al., 2024; Zhang and Xiong, 2025). However, these studies do not systematically evaluate whether multi-agent judging improves alignment with human reference scores in criterion-level subjective scoring tasks.

To this end, we investigate whether consensusbased MAD improves or degrades human alignment in subjective evaluation, and, if degradation is observed, what mechanisms may explain it. We treat numerical anchoring and role-prompt asymmetry as two competing hypotheses for this degradation rather than as an established mechanism, and we design ablations to test each hypothesis directly. Under the anchoring hypothesis, score sharing induces anchoring effects: prior work on judgment under uncertainty shows that exposure to numerical values biases subsequent judgments toward observed anchors (Tversky and Kahneman, 1974; Mussweiler and Strack, 2000), so agents observing both textual rationales and numerical scores from other agents may converge toward shared score ranges, compressing score distributions and reducing discriminability. Under the role-prompt asymmetry hypothesis, assigning agents asymmetric evaluative stances, such as strict and lenient judge roles, allows one stance to dominate the deliberation process and weaken the reliability of the final consensus independent of whether numerical scores are shared. Score-Masked MAD tests the anchoring hypothesis by removing numerical score sharing while preserving role asymmetry, while Symmetric MAD tests the role-prompt asymmetry hypothesis by removing role asymmetry while preserving multi-round interaction.

To investigate further, we compare single-agent and Consensus-based MAD judges against human reference scores, using ablations that isolate score sharing and asymmetric role prompting. Our experiments show that consensus-style multi-agent judging degrades alignment with human reference scores, and our ablation analysis suggests that asymmetric role prompting is a primary driver of this degradation.

## 2 Methodology

We study a compact consensus-style multi-agent judging design for subjective evaluation tasks. To isolate and examine the core mechanisms shared by prior multi-agent debate and agent-as-judge studies (Du et al., 2024; Chan et al., 2024; Li et al.,

2024), we deliberately instantiate a minimal MAD protocol with role-specialized agents, iterative exchange, and final aggregation, rather than proposing a new framework. This design allows us to test whether a consensus-oriented interaction, which has been used to improve reasoning or open-ended evaluation, also improves alignment with human ratings in subjective scoring. Figure 1 summarizes the experimental design. Across all protocols, we keep the task input, target text, rubric, scoring scale, and judge model fixed. The full prompts are provided in Appendix A.

## 2.1 Judge Protocols

Single Judge The single-judge condition serves as the baseline. A single LLM judge receives the task instruction, rubric, and target text, and independently assigns criterion-level scores. This condition involves no role specialization, inter-agent interaction, or score aggregation.

Consensus-based MAD Our main multi-agent condition consists of two role-specialized agents: a Strict Judge and a Lenient Judge. We use this strict–lenient pair as a minimal form of evaluative diversity: subjective scoring often involves a tension between penalizing weaknesses and recognizing plausible strengths. At Round 0, both agents independently evaluate the target text. In each subsequent round, the strict and lenient agents observe each other’s previous score and rationale, then revise or maintain their own judgments. The final prediction is the arithmetic mean of the two agents’ final-round scores. In our experiments, we conducted this process up to Round 4.

## 2.2 Ablation Conditions

We introduce three ablation conditions to decompose the consensus-style design into role prompting, multi-round interaction, and numerical score sharing.

Strict Role Judge The Strict Role Judge applies the same strict scoring prompt used in the rolespecialized consensus judge, but removes interagent interaction. This condition tests whether the strict role prompt alone changes score levels or human alignment. Comparing this condition with the single-judge baseline isolates the effect of role framing, while comparing it with the Consensusbased MAD shows whether interaction amplifies or moderates the effect of the strict role.

Symmetric MAD Symmetric MAD preserves the same multi-round interaction structure as the main consensus protocol but removes role specialization. Both agents use the same neutral judge prompt and exchange previous-round evaluations. This condition tests whether interaction alone explains the observed behavior, or whether asymmetric role prompting is necessary for the alignment changes observed in the main protocol.

Score-Masked MAD Score-Masked MAD preserves the strict–lenient role structure and rationale exchange, but removes explicit numerical score sharing from inter-agent messages. Agents observe the other agent’s previous rationale excluding any numerical values. This condition tests whether numerical scores act as anchoring signals during deliberation, since prior work shows that exposure to numerical values can bias subsequent judgments under uncertainty (Tversky and Kahneman, 1974; Mussweiler and Strack, 2000).

## 3 Experimental Setup

## 3.1 Tasks and Datasets

We evaluate judge protocols on two subjective evaluation tasks that provide criterion-level human reference scores for rubric-based evaluation, enabling direct measurement of judge-to-human alignment. Detailed dataset settings are provided in Appendix B.

We deliberately diversify language, rubric structure, and genre across the two tasks, while standardizing the measurement protocol: both tasks use a 1–5 scoring scale, criterion-level human reference scores, and an identical grid of judge protocols and judge models. This design lets us test whether the effects of role-specialized consensus generalize across domains that differ substantially in surface characteristics but are evaluated under the same experimental grid.

Korean Essay Scoring For this task, we use the National Institute of Korean Language writing assessment corpus<sup>1</sup>. We sample 600 essays from six topics. Each essay is evaluated on three criteria: content, organization, and expression. The average of two human evaluators is leveraged as the reference score.

Summarization Evaluation We use SummEval (Fabbri et al., 2020), which provides human ratings for system-generated summaries across four criteria: coherence, consistency, fluency, and relevance. We use summaries generated by seven systems for 100 source articles, yielding 700 system-summary instances. Each summary is evaluated at the criterion level, and the human rating is used as the reference score.

One prompt-level asymmetry exists between the two single-judge baselines: the SummEval singlejudge prompt includes the phrase strict and consistent evaluator, a directional cue absent from the Korean Essay single-judge prompt. Because our own ablations show that a strict framing systematically lowers human alignment relative to a neutral prompt (Table 1), this asymmetry, if anything, works against our central claim: it nudges the SummEval single-judge baseline toward the same strict framing that degrades alignment under Consensus-based MAD, which would narrow rather than inflate the single-judge advantage we report. We therefore consider the direction of this asymmetry conservative with respect to our conclusions, although we do not directly quantify its magnitude.

## 3.2 Models and Metrics

We use six LLMs as judge models: GPT-4o-mini<sup>2</sup>, Gemma-3-4B-IT, Gemma-3-12B-IT, and Gemma-3-27B-IT (Team et al., 2025), Qwen3.5-9B, and Qwen3.5-27B (Qwen Team, 2026). We select models spanning three families and three capacity levels (4B–27B) to assess whether findings hold across model sizes and architectures. All models are accessed through OpenRouter<sup>3</sup> with temperature setting of 0. Within each run, the same model performs all judge roles. Additional inference details are provided in Appendix C. For measuring human alignment, we use RMSE and Spearman correlation. RMSE measures absolute agreement with human reference scores, while Spearman correlation measures whether the protocol preserves the relative ranking of instances.

<table><tr><td rowspan="2">Protocol</td><td colspan="2">Korean Essay</td><td colspan="2">SummEval</td></tr><tr><td>RMSE↓</td><td> $\rho \uparrow$ </td><td>RMSE↓</td><td> $\rho \uparrow$ </td></tr><tr><td>Single Judge</td><td>0.644</td><td>0.504</td><td>0.682</td><td>0.458</td></tr><tr><td>Strict Role Judge</td><td>0.973</td><td>0.499</td><td>0.884</td><td>0.447</td></tr><tr><td>Symmetric MAD</td><td>0.733</td><td>0.451</td><td>0.687</td><td>0.447</td></tr><tr><td>Consensus-based MAD</td><td>0.935</td><td>0.424</td><td>0.813</td><td>0.409</td></tr><tr><td>Score-Masked MAD</td><td>1.040</td><td>0.399</td><td>0.994</td><td>0.368</td></tr></table>

Table 1: Human alignment across judge protocols, averaged over six judge models. The single-judge baseline achieves the strongest average alignment on two datasets.

![](images/8d2dcae9a395c812c078801f46c478732afd18ef69b773a43d90440e94efa2de.jpg)  
Figure 2: Round-wise RMSE under Consensus-based MAD on the Korean Essay scoring task. Each point traces the protocol-level prediction after each exchange round. The upward trend after early interaction suggests that iterative consensus does not necessarily move judge predictions closer to human reference scores.

## 4 Results

## 4.1 Human Alignment Across Protocols

Table 1 reports RMSE and Spearman correlation averaged over six judge models. The singlejudge baseline achieves the best alignment on both tasks in both RMSE and Spearman correlation. Consensus-based MAD increases RMSE to 0.935 and 0.813 while Spearman drops to 0.424 and 0.409, indicating degradation in both absolute agreement and ranking preservation. As illustrated in Figure 2, RMSE generally increases after the initial exchange. This indicates that subsequent rounds fail to correct the degradation and instead compound it. Detailed per-model results and gap trajectories can be found in Appendix E. Significance tests for this degradation are reported in Appendix F.

<table><tr><td>Protocol</td><td>Korean Essay</td><td>SummEval</td></tr><tr><td>Reference</td><td>3.486</td><td>4.285</td></tr><tr><td>Single Judge</td><td>3.345</td><td>3.935</td></tr><tr><td>Strict Role Judge</td><td>2.733</td><td>3.672</td></tr><tr><td>Symmetric MAD</td><td>3.249</td><td>3.964</td></tr><tr><td>Consensus-based MAD</td><td>2.900</td><td>3.831</td></tr><tr><td>Score-Masked MAD</td><td>2.767</td><td>3.667</td></tr></table>

Table 2: Mean predicted score per protocol, averaged over all evaluation instances and six judge models. Roleasymmetric protocols show a consistent downward shift relative to the single-judge baseline and human reference. Appendix D provides detailed analysis.

## 4.2 Role Asymmetry

Table 2 reports mean predicted scores, which indicate the direction of score shift rather than human alignment itself. Consensus-based MAD produces lower mean scores than both the single-judge baseline and the human reference on both tasks. On Korean Essay, the mean score is 2.900 under Consensus-based MAD, compared with 3.345 under Single Judge and 3.486 for the human reference. On SummEval, the corresponding values are 3.831, 3.935, and 4.285. This shows that the strict–lenient consensus protocol shifts the average prediction downward relative to the human reference.

This downward shift is most pronounced in the Strict Role Judge condition. Its mean scores are 2.733 on Korean Essay and 3.672 on SummEval, both below the Consensus-based MAD means. The same condition also yields higher RMSE than Consensus-based MAD (Essay: 0.973 vs. 0.935; SummEval: 0.884 vs. 0.813), indicating that the strict role prompt alone moves predictions away from the human reference. Interaction in Consensus-based MAD partially moderates this strict-role shift, but does not restore single-judge alignment. Dominance analysis results are reported in Appendix G.

By contrast, Symmetric MAD remains much closer to the single-judge baseline. Its RMSE is 0.733 on Korean Essay and 0.687 on SummEval, compared with 0.644 and 0.682 for the Single Judge. Its mean scores are also closer to the singlejudge means (Essay: 3.249 vs. 3.345; SummEval: 3.964 vs. 3.935). Together, these results suggest that the main source of alignment loss is not multi-round interaction itself, but asymmetric role prompting. Lenient Role Judge results are reported in Appendix G.

<table><tr><td rowspan="2">Condition</td><td colspan="2">RMSE</td></tr><tr><td>Korean Essay</td><td>SummEval</td></tr><tr><td>Single Judge baseline</td><td></td><td></td></tr><tr><td>Gemma3-27B</td><td>0.528</td><td>0.414</td></tr><tr><td>Gemma3-12B</td><td>0.643</td><td>0.575</td></tr><tr><td>Qwen3.5-9B</td><td>0.640</td><td>0.750</td></tr><tr><td>Cross-model MAD</td><td></td><td></td></tr><tr><td> $\mathrm { G e m m a } 3 { - } 2 7 \mathrm { B } _ { S } / \mathrm { G e m m a } 3 { - } 1 2 \mathrm { B } _ { L }$ </td><td>0.690</td><td>0.528</td></tr><tr><td>Gemma  $3 { - } 1 2 \mathbf { B } _ { S } / \mathbf { G e m m a } 3 { - } 2 7 \mathbf { B } _ { L }$ </td><td>0.639</td><td>0.505</td></tr><tr><td> $\mathrm { G e m m a } 3 – 2 7 \mathrm { B } _ { S } / \mathrm { Q w e n } 3 . 5 – 9 \mathrm { B } _ { L }$ </td><td>0.746</td><td>0.587</td></tr><tr><td> $\mathrm { Q w e n } 3 . 5 { \cdot } 9 \mathrm { B } _ { S } / \mathrm { G e m m a } 3 { \cdot } 2 7 \mathrm { B } _ { L }$ </td><td>1.143</td><td>0.877</td></tr></table>

Table 3: Single-judge baselines and cross-model roleswapping results. Subscripts S and L indicate Strict and Lenient roles. Cross-model pairing changes the magnitude of RMSE but does not consistently recover the strongest single-judge alignment, suggesting that model diversity alone does not remove the effect of strict–lenient role asymmetry.

## 4.3 Cross-Model Generalization

We assign the Strict and Lenient roles to different models and swap their assignments within each pair. Table 3 shows that cross-model pairing changes the magnitude of RMSE but does not eliminate the role-asymmetry pattern: no configuration outperforms the strongest single-judge baseline, and Qwen $3 . 5 { - } 9 \mathrm { B } _ { S } / \mathrm { G e m m a } 3 { - } 2 7 \mathrm { B } _ { L }$ yields the largest errors on both tasks, suggesting that roleinduced biases persist regardless of the underlying models.

## 5 Conclusion

This paper investigates whether consensus-style multi-agent judging with asymmetric roles improves human alignment in subjective rubric-based evaluation. Across both tasks, the single-judge baseline outperformed all multi-agent conditions in terms of RMSE and Spearman correlation. Our ablation results conclude that asymmetric role prompting is the most consistent source of this degradation. The strict role prompt alone shifts scores substantially below the human reference distribution, while Symmetric MAD, which retains multi-round interaction without role asymmetry, shows improved alignment. Score exposure acts more as a coordination signal than a harmful anchor, as masking scores widens the inter-agent gap without recovering human alignment. Cross-model experiments further show that assigning roles to different models changes the magnitude of the effect but does not remove it. These findings show that asymmetric role prompts bias the consensus away from human reference judgments, emphasizing the importance of role design in consensus-style, rolespecialized MAD protocols.

## Limitations

Our findings are subject to several limitations. The consensus-style MAD protocol we study is our own controlled operationalization rather than a direct replication of any single prior system; results may differ under other MAD designs that use different role prompts, interaction structures, or aggregation strategies. Within our design, we test only one role configuration (strict–lenient) and fix the protocol to five total rounds, consisting of an initial round followed by four adjustment rounds; we additionally report a standalone Lenient-only condition alongside the Strict-only baseline in Appendix G, but the effect of varying the round count is not evaluated. Final scores are aggregated by unweighted averaging; alternative strategies such as weighted ensembles or majority voting are not explored. Our study covers two domains (Korean essay scoring and English summarization) with six judge models; generalization to other tasks, languages, or model families requires further investigation. Several additional design axes remain unexplored and are left for future work, including the number of agents, the diversity of role configurations, the format of inter-agent interaction, the aggregation rule, and the degree of judge specialization. The SummEval single-judge prompt uses the phrase strict and consistent evaluator, introducing a directional framing absent from the Korean Essay baseline; this asymmetry may affect cross-domain comparisons and should be examined in future work. Pairedbootstrap significance tests for the primary comparisons are reported in Appendix F.

## Ethical Statement

This study uses two publicly available datasets: the NIKL Grading Writing Data 2024, released by the National Institute of the Korean Language for research purposes, and the SummEval benchmark, accessed through HuggingFace Hub. No new human subject data were collected, and no personally identifiable information was used. All LLM inference was conducted through the OpenRouter API using commercially available models. The datasets contain no sensitive personal information, and the evaluation outputs produced by LLM judges are used solely for research analysis.

## Acknowledgments

This work was supported by the Korea Internet & Security Agency (KISA) grant funded by the Korea government (PIPC) (No. RS-2026-25526342, Development of Technologies for Preventing Sensitive Information Inference and Risk Assessment in Foundation Model Operations). This work was also supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2026-25483747). This research was further supported by the Culture, Sports and Tourism R&D Program through the Korea Creative Content Agency, funded by the Ministry of Culture, Sports and Tourism in 2026 (Project Name: Develop AI Agent Technology to Connect Knowledge through Public Cultural Facility-Based Discussion and Communication, Project Number: RS-2026-25520645). This research was also supported by the ANCHOR Program through the Gangwon ANCHOR Center, funded by the Ministry of Education (MOE) and Gangwon State (G.S.), Republic of Korea (2026-ANCHOR-10-006).

## References

Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. 2024. Chateval: Towards better llm-based evaluators through multi-agent debate. In International conference on learning representations, volume 2024, pages 9079–9093.

Jiaju Chen, Yuxuan Lu, Xiaojie Wang, Huimin Zeng, Jing Huang, Jiri Gesi, Ying Xu, Bingsheng Yao, and Dakuo Wang. 2025. Multi-agent-as-judge: Aligning llm-agent-based automated evaluation with multi-dimensional human evaluation. arXiv preprint arXiv:2507.21028.

Steffi Chern, Ethan Chern, Graham Neubig, and Pengfei Liu. 2024. Can large language models be trusted for evaluation? scalable meta-evaluation of llms as evaluators via agent debate. arXiv preprint arXiv:2401.16788.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2024. Improving factuality and reasoning in language models through multiagent debate. In Forty-first international conference on machine learning.

Alexander R Fabbri, Wojciech Krysci´ nski, Bryan Mc-´ Cann, Caiming Xiong, Richard Socher, and Dragomir Radev. 2020. Summeval: Re-evaluating summarization evaluation. arXiv preprint arXiv:2007.12626.

Zhaopeng Feng, Jiayuan Su, Jiamei Zheng, Jiahan Ren, Yan Zhang, Jian Wu, Hongwei Wang, and Zuozhu

Liu. 2025. M-mad: Multidimensional multi-agent debate for advanced machine translation evaluation. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7084–7107.

Seungone Kim, Jay Shin, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Ryan Shin, Sungdong Kim, James Thorne, Minjoon Seo, and 1 others. 2024. Prometheus: Inducing fine-grained evaluation capability in language models. In International Conference on Learning Representations, volume 2024, pages 29927–29962.

Yu Li, Shenyu Zhang, Rui Wu, Xiutian Huang, Yongrui Chen, Wenhao Xu, Guilin Qi, and Dehai Min. 2024. Mateval: a multi-agent discussion framework for advancing open-ended text evaluation. In International Conference on Database Systemsfor Advanced Applications, pages 415–426. Springer.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2024. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 17889–17904.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 2511–2522.

Thomas Mussweiler and Fritz Strack. 2000. Numeric judgments under uncertainty: The role of knowledge in anchoring. Journal ofexperimental social psychology, 36(5):495–518.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. 2025. Judging the judges: A systematic study of position bias in llmas-a-judge. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 292–314.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

Amos Tversky and Daniel Kahneman. 1974. Judgment under uncertainty: Heuristics and biases: Biases in judgments reveal some heuristics of thinking under uncertainty. science, 185(4157):1124–1131.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and 1 others. 2024. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9440–9450.

Koki Wataoka, Tsubasa Takahashi, and Ryokan Ri. 2024. Self-preference bias in llm-as-a-judge. arXiv preprint arXiv:2410.21819.

Shaowei Zhang and Deyi Xiong. 2025. Debate4math: Multi-agent debate for fine-grained reasoning in math. In Findings of the Association for Computational Linguistics: ACL 2025, pages 16810–16824.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. 2025. Judgelm: Fine-tuned large language models are scalable judges. In International Conference on Learning Representations, volume 2025, pages 51257–51296.

## A Experiment Prompts

All system prompts share a modular structure comprising a role definition, dataset-specific evaluation criteria, a score rubric, and a structured JSON output format. Because the criteria, rubric, and output-format blocks are identical across prompts within each dataset, we define them once in Appendix A.1 and use the tokens [ESSAY\_CRITERIA], [ESSAY\_RUBRIC], [SE\_CRITERIA], [SE\_RUBRIC], [ESSAY\_OUTPUT\_INIT], [ESSAY\_OUTPUT\_ADJ], [SE\_OUTPUT\_INIT], and [SE\_OUTPUT\_ADJ] as placeholders in the per-condition prompts that follow. The INIT variants apply to the initial evaluation round (key: criterion\_rationale); the ADJ variants apply to adjustment rounds (key: adjustment\_notes).

The remainder of this appendix is organized by prompt function. Appendix A.1 defines the reusable criteria, rubrics, and output-format blocks. Appendix A.2 reports the single-judge prompts. Appendix A.3 reports the role-specific strict and lenient prompts used for the initial round of role-asymmetric protocols. Appendix A.4 reports the adjustment-round prompts for Consensusbased MAD and Score-Masked MAD, while Appendix A.5 reports the neutral prompts used in Symmetric MAD.

\- 글의 주장과 핵심 내용이 문제에 적절하게대응하는가

1. content

\- 주장과 근거 사이의 논리적 연결이 타당한가

## A.1 Shared Components

\- 근거가 충분하고 구체적인가

2. organization

\- 서론, 본론, 결론의 구조가 드러나는가

\- 문단 간 연결이 자연스러운가

\- 논리 전개 순서가 일관적인가

## Essay Criteria [ESSAY\_CRITERIA]

3. expression

\- 문장이 자연스럽고 이해하기 쉬운가

\- 어휘 사용이 적절한가

## [평가 기준 정의]

\- 맞춤법, 띄어쓰기, 문법, 주술 호응에 문제가없는가

Essay Criteria [ESSAY\_CRITERIA] (English Translation)

[Evaluation Criteria Definitions]

\- Does the essay's claim and core content appropriately address the prompt?

\- Is the supporting evidence sufficient and specific?

1. content

## 2. organization

## 3. expression

\- Is the logical connection between the claim and evidence valid?

\- Are the transitions between paragraphs natural?

\- Is the structure of introduction, body, and conclusion apparent?

\- Is the sequence of logical development consistent?

\- Are the sentences natural and easy to understand?

\- Is the vocabulary used appropriately?

\- Are there any problems with spelling, spacing, grammar, or subject--predicate agreement?

[점수 기준]

## Essay Rubric [ESSAY\_RUBRIC]

4점: 우수함. 경미한 약점은 있으나 기준을전반적으로 잘 충족함.

에서 확인되는 구체적 강점이 뚜렷함.

5점: 매우 우수함. 결함이 거의 없고, essay\_text

3점: 보통. 장점과 약점이 함께 있으며 기준을부분적으로 충족함.

2점: 미흡함. 주요 결함이 있어 기준 충족이제한적임.

1점: 매우 미흡함. 기준을 거의 충족하지 못하거나심각한 결함이 있음.

## Essay Rubric [ESSAY\_RUBRIC] (English Translation)

## [Score Rubric]

5: Excellent. There are almost no flaws, and specific strengths evident in essay\_text are clear.

4: Good. There are minor weaknesses, but the criterion is generally well satisfied.

3: Average. There are both strengths and

weaknesses, and the criterion is partially satisfied.

2: Poor. Major flaws limit satisfaction of the criterion.

1: Very poor. The criterion is barely

satisfied, or there are severe flaws.

## SummEval Criteria [SE\_CRITERIA]

coherence: The summary is well-structured and well-organized. Does it present a clear, logical flow of information? consistency: The summary contains only information consistent with the source document. Does it avoid introducing facts not present in the article? fluency: The summary has no grammatical errors and is easy to read. Is it fluent and idiomatic English? relevance: The summary focuses on the most important information from the source. Does it cover the

```json
[ESSAY_OUTPUT_INIT]
```

You are an evaluator who directly and consistently scores Korean essays. Read essay\_text and evaluate all three criteria: content, organization, and expression.

## key points and avoid trivial details?

## SummEval Rubric [SE\_RUBRIC]

[Score Rubric] 5: Excellent – no flaws, directly supported by the source article 4: Good – minor weaknesses, well-grounded in the article 3: Average – borderline; at least one criterion partially met 2: Poor – major flaws in at least one criterion 1: Very poor – hallucinations, severe mismatch, or complete failure

```jsonl
Essay Output Format [ESSAY_OUTPUT_INIT]
[ESSAY_OUTPUT_ADJ]
[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운
사용 금지.
- 모든 점수는 1~5 정수. 텍스트 필드는 한국어로
작성하라.
[ESSAY_OUTPUT_INIT — 초기 라운드]
{"content": int, "organization": int,
"expression": int, "overall_judge": float,
"criterion_rationale": {"content": "...
"organization": "...", "expression": "..."}}
[ESSAY_OUTPUT_ADJ — 조정 라운드]
{"content": int, "organization": int,
"expression": int, "overall_judge": float,
"adjustment_notes": {"content": "...",
"organization": "...", "expression": "..."}}
```

## Essay Output Format [ESSAY\_OUTPUT\_INIT] [ESSAY\_OUTPUT\_ADJ] (English Translation)

```jsonl
[Output Rules]
- Output exactly one JSON object. Do not use
Markdown code blocks.
- All scores must be integers from 1 to 5.
Write text fields in Korean.
[ESSAY_OUTPUT_INIT — initial round]
{"content": int, "organization": int,
"expression": int, "overall_judge": float,
"criterion_rationale": {"content": ".
"organization": "...", "expression": "..."}}
[ESSAY_OUTPUT_ADJ — adjustment round]
{"content": int, "organization": int,
"expression": int, "overall_judge": float,
"adjustment_notes": {"content": "...",
"organization": "...", "expression": "..."}}
```

## SummEval Output Format [SE\_OUTPUT\_INIT] [SE\_OUTPUT\_ADJ]

[Output Rules]   
- Output ONLY a JSON object. No code blocks,   
no explanation.   
- All scores are integers 1--5.   
[SE\_OUTPUT\_INIT — initial round]   
{"coherence": int, "consistency": int,   
"fluency": int, "relevance": int,   
"overall\_judge": float,

```jsonl
"criterion_rationale": {"coherence": "...",
"consistency": "...",
"fluency": "...",
"relevance": "..."}}
[SE_OUTPUT_ADJ — adjustment round]
{"coherence": int, "consistency": int,
"fluency": int, "relevance": int,
"overall_judge": float,
"adjustment_notes": {"coherence": "...",
"consistency": "...",
"fluency": "...",
"relevance": "..."}}
```

## A.2 Single Judge Prompts

Single Judge — Essay  
[역할]  
너는 한국어 에세이를 일관되게 직접 채점하는  
평가자이다.  
essay\_text를 읽고 content, organization,  
expression 세 기준을 모두 평가하라.  
[ESSAY\_CRITERIA]  
[ESSAY\_RUBRIC]  
[평가 원칙]  
- 1\~5점 전 구간을 적극적으로 사용하라.  
- 각 기준은 서로 독립적으로 판단하라.  
- essay\_text에서 확인 가능한 내용만 근거로  
삼아라.  
- 전반적 인상만으로 높은 점수를 주지 말고 구체적  
근거를 확인하라.  
- 근거 설명은 기준별로 분리해 작성하라.  
[ESSAY\_OUTPUT\_INIT]

## Single Judge — Essay (English Translation)

## [ESSAY\_CRITERIA]

## [ESSAY\_RUBRIC]

[Evaluation Principles]

\- Actively use the full score range from 1 to 5.

\- Judge each criterion independently.

\- Use only content verifiable in essay\_text as evidence.

## Single Judge — SummEval

You are a strict and consistent evaluator of English summarization quality.

Evaluate the given summary against the source article on four criteria.

[Criteria]

[SE\_CRITERIA]

[SE\_RUBRIC]

[Rules]

\- Use the full 1\~5 range actively.

\- Do NOT give high scores just because the summary looks reasonable.

\- Keep the textual justification aligned with each criterion score.

[SE\_OUTPUT\_INIT]

## A.3 Role-Specific Prompts (Strict and Lenient)

Strict Judge, Initial — Essay

essay\_text에 결함이 있다고 가정하고, 명확한 반대근거가 있을 때만 높은 점수를 부여하라.

[ESSAY\_CRITERIA]

[ESSAY\_RUBRIC]

[엄격 채점 원칙]

\- essay\_text에서 구체적으로 확인되지 않는 장점은인정하지 마라.

\- 구체적 근거가 없으면 해당 기준은 3점 이하로채점하라.

\- 주장, 구조, 표현 중 주요 결함이 보이면 관련기준은 1\~2점을 적극 검토하라.

\- 4\~5점은 essay\_text에 직접 근거한 강점이분명할 때만 부여하라.

\- 근거 설명은 기준별로 분리해 작성하라.

[ESSAY\_OUTPUT\_INIT]

## Strict Judge, Initial — Essay (English Translation)

[Role]

Assume that essay\_text has flaws, and assign high scores only when there is clear evidence to the contrary.

[ESSAY\_CRITERIA]

[ESSAY\_RUBRIC]

[Strict Scoring Principles] - Do not acknowledge strengths that cannot be specifically verified in essay\_text.

\- If there is no specific evidence, score that criterion 3 or below.

\- If you identify a major flaw in the claim, structure, or expression, actively consider a score of 1 or 2 for the relevant criterion. - Assign a score of 4 or 5 only when strengths directly supported by essay\_text are clear. - Write the supporting explanation separately for each criterion.

## [ESSAY\_OUTPUT\_INIT]

## Lenient Judge, Initial — Essay

너는 한국어 에세이의 가능성과 장점을 공정하게인정하며 직접 채점하는 평가자이다.

essay\_text를 바탕으로 content, organization,expression 세 기준을 모두 평가하라.

[ESSAY\_CRITERIA]

[ESSAY\_RUBRIC]

\- essay\_text에서 확인되는 긍정적 가능성을적극적으로 인정하라.

\- 명백한 결함이 없으면 해당 기준은 3점 이상으로채점하라.

\- 부분적 약점이 있어도 전체 수행이 기준을

- 단, essay\_text에 없는 장점은 만들어내지 마라.  
근거 설명은 기준별로 분리해 작성하라.

[ESSAY\_OUTPUT\_INIT]

## Lenient Judge, Initial — Essay (English Translation)

[Role]

You are an evaluator who directly scores Korean essays while fairly recognizing their potential and strengths.

Based on essay\_text, evaluate all three criteria: content, organization, and expression.

## [ESSAY\_CRITERIA]

## [ESSAY\_RUBRIC]

[Lenient Scoring Principles]

\- Actively recognize positive potential evident in essay\_text.

\- If there are no obvious flaws, score that criterion 3 or above.

\- Even if there are partial weaknesses, consider a score of 4 if the overall

performance satisfies the criterion.

\- However, do not invent strengths that are not present in essay\_text.

\- Write the supporting explanation separately for each criterion.

[ESSAY\_OUTPUT\_INIT]

## Strict Judge, Initial — SummEval

[Role]

You are an extremely strict evaluator of English summarization quality.

Default assumption: the summary has flaws.   
Prove otherwise.

[Criteria]

[SE\_CRITERIA]

## [SE\_RUBRIC]

[Strict Rules]

\- Generic descriptions without

article-specific evidence score <= 2 for relevance/consistency.

\- Hallucinated facts (not in article) score 1 for consistency.

\- Give low scores unless clear evidence of quality exists.

\- Explain each criterion separately.

## [SE\_OUTPUT\_INIT]

## Lenient Judge, Initial — SummEval

[Role]

You are a lenient evaluator of English summarization quality.

Give benefit of the doubt when possible.

[Criteria]

[SE\_CRITERIA]

[SE\_RUBRIC]

[Lenient Rules]

\- If direction (positive/negative) aligns with the article, give higher scores.

\- Only penalize clear hallucinations (content not at all in the article).

\- Default: score >= 3 unless there is obvious failure.

\- Explain each criterion separately.

[SE\_OUTPUT\_INIT]

## A.4 Consensus-based MAD —

## Adjustment-Round Prompts

## Strict Judge, Adjustment — Essay

[역할]

너는 초기 평가를 마친 엄격한 한국어 에세이평가자이다.

이제 관대한 평가자의 판단을 참고하여 content,

organization, expression 점수를 재검토하라.

## [ESSAY\_CRITERIA]

[조정 원칙]

\- 먼저 자신의 직전 점수와 근거를 검토하라.

\- 관대한 평가자의 주장 중 essay\_text에서 검증

가능한 내용만 받아들여라.

\- 네가 지나치게 낮게 본 기준은 점수를 올려도된다.

\- essay\_text에서 확인되지 않는 장점은 인정하지마라.

\- 상대 평가가 설득력이 약하면 기존 점수를유지하라.

\- 조정 사유는 기준별로 분리해 작성하라.

[ESSAY\_OUTPUT\_ADJ]

## Strict Judge, Adjustment — Essay (English Translation)

[Role]

You are a strict Korean essay evaluator who has completed the initial evaluation. Now refer to the lenient evaluator's judgment and reconsider the content, organization, and expression scores.

## [ESSAY\_CRITERIA]

## [Adjustment Principles]

\- First review your own previous scores and supporting evidence.

\- Accept only claims by the lenient evaluator that can be verified in essay\_text.

\- You may raise the score for a criterion that you judged too harshly.

\- Do not acknowledge strengths that cannot be verified in essay\_text.

\- If the other evaluation is not persuasive, maintain your previous score.

\- Write the reasons for adjustment separately for each criterion.

## [ESSAY\_OUTPUT\_ADJ]

## Lenient Judge, Adjustment — Essay

[역할]

너는 초기 평가를 마친 관대한 한국어 에세이평가자이다.

이제 엄격한 평가자의 판단을 참고하여 content,

organization, expression 점수를 재검토하라.

## [ESSAY\_CRITERIA]

## [조정 원칙]

\- 먼저 자신의 직전 점수와 근거를 검토하라.

\- 엄격한 평가자의 비판 중 essay\_text에서 실제로확인되는 결함만 받아들여라.

\- 네가 근거 없이 높게 준 기준은 점수를 내려라.

\- 실제 결함이 아닌 추정이나 과도한 비판은

받아들이지 않아도 된다.

\- 상대 평가가 설득력이 약하면 기존 점수를유지하라.

\- 조정 사유는 기준별로 분리해 작성하라.

[ESSAY\_OUTPUT\_ADJ]

## Lenient Judge, Adjustment — Essay (English Translation)

[Role]

You are a lenient Korean essay evaluator who

has completed the initial evaluation.

Now refer to the strict evaluator's judgment and reconsider the content, organization, and expression scores.

## [ESSAY\_CRITERIA]

[Adjustment Principles]

\- First review your own previous scores and supporting evidence.

\- Lower the score for any criterion that you rated highly without evidence.

\- You do not have to accept speculation or excessive criticism that does not identify an actual flaw.

\- If the other evaluation is not persuasive, maintain your previous score.

\- Write the reasons for adjustment separately for each criterion.

[ESSAY\_OUTPUT\_ADJ]

## Strict Judge, Adjustment — SummEval

[Role]

You are the strict evaluator revisiting your scores after seeing the lenient evaluator's assessment.

[Criteria]

[SE\_CRITERIA]

[Adjustment Rules]

\- Review your own previous scores and rationale first.

\- Accept lenient judge's points if they

reference actual summary/article content.

\- Raise scores where you were overly harsh.

\- Do NOT accept points based on content not in the original summary.

\- Keep your previous score when the lenient case is not persuasive.

\- Record changes criterion by criterion.

[SE\_OUTPUT\_ADJ]

## Lenient Judge, Adjustment — SummEval

## [Role]

You are the lenient evaluator revisiting your scores after seeing the strict evaluator's assessment.

[Criteria]

[SE\_CRITERIA]

[Adjustment Rules]

\- Review your own previous scores and rationale first.

\- Accept strict judge's criticisms if they identify real flaws in the summary.

\- Lower scores where you were overly generous.

\- Maintain scores where strict judge was too harsh.

\- Keep your previous score when the strict case is not persuasive.

Record changes criterion by criterion.

[SE\_OUTPUT\_ADJ]

## A.5 Symmetric MAD Prompts

## Sym. MAD Neutral, Initial — Essay

strict 또는 lenient 방향의 사전 편향 없이essay\_text를 읽고

content, organization, expression 세 기준을모두 평가하라.

[ESSAY\_RUBRIC]

\- 1\~5점 전 구간을 적극적으로 사용하라.

\- essay\_text에서 확인 가능한 내용만 근거로삼아라.

\- 기준별로 장점과 단점을 모두 검토하고, 확인

\- 전반적 인상이나 역할 기대가 아니라 근거

\- 근거 설명은 기준별로 분리해 작성하라.

[ESSAY\_OUTPUT\_INIT]

## Sym. MAD Neutral, Initial — Essay (English Translation)

You are an evaluator who directly scores Korean essays fairly and impartially. Read essay\_text without a prior strict or lenient directional bias and evaluate all three criteria: content, organization, and expression.

## [ESSAY\_CRITERIA]

## [ESSAY\_RUBRIC]

[Evaluation Principles] - Actively use the full score range from 1 to 5.

\- Judge each criterion independently.

\- Use only content verifiable in essay\_text as evidence.

\- Examine both strengths and weaknesses for each criterion, and determine the score according to the strength of verifiable evidence.

\- Score based on evidence, not on an overall impression or role expectations.

\- Write the supporting explanation separately for each criterion.

[ESSAY\_OUTPUT\_INIT]

## Sym. MAD Neutral, Adjustment — Essay

[역할]  
너는 초기 평가를 마친 중립적 한국어 에세이  
평가자이다.  
이제 이전 점수와 상대 평가자의 판단을 함께  
검토하여  
content, organization, expression 점수를  
재검토하라.

## [ESSAY\_CRITERIA]

\- 먼저 자신의 직전 점수와 근거를 검토하라.

\- 상대 평가자의 논거 중 essay\_text에서 확인가능한 내용만 반영하라.

\- 방향 편향 없이 상향 조정과 하향 조정을 모두허용하라.

\- essay\_text에서 확인되지 않는 장점, 결함,추정은 반영하지 마라.

\- 상대 평가가 설득력이 약하면 기존 점수를유지하라.

\- 조정 사유는 기준별로 분리해 작성하라.

[ESSAY\_OUTPUT\_ADJ]

## Sym. MAD Neutral, Adjustment — Essay (English Translation)

[Role]

You are a neutral Korean essay evaluator who has completed the initial evaluation.

Now review your previous scores together with the other evaluator's judgment and

## [ESSAY\_CRITERIA]

[Adjustment Principles]

\- First review your own previous scores and supporting evidence.

\- Allow both upward and downward adjustments without directional bias.

\- Do not incorporate strengths, flaws, or speculation that cannot be verified in essay\_text.

- If the other evaluation is not persuasive,   
maintain your previous score.

\- Write the reasons for adjustment separately for each criterion.

## [ESSAY\_OUTPUT\_ADJ]

## Sym. MAD Neutral, Initial — SummEval

[Role]

You are a balanced evaluator of English summarization quality.

Evaluate the given summary against the source article with no strict or lenient directional bias.

[Criteria]

[SE\_CRITERIA]

## [SE\_RUBRIC]

## [Rules]

\- Use the full 1\~5 range actively.

\- Score each criterion independently.

\- Ground all judgments in the source article and summary.

\- Assess both strengths and weaknesses for each criterion.

\- Assign scores based on evidence, not on a default positive or negative stance.

\- Keep the textual justification aligned with each criterion score.

## [SE\_OUTPUT\_INIT]

## Sym. MAD Neutral, Adjustment — SummEval

## [Role]

You are a neutral summarization quality evaluator revisiting your scores. Review your previous assessment and the other evaluator's assessment, then revise scores only when the article/summary evidence supports it.

## [Criteria]

## [SE\_CRITERIA]

## [Adjustment Rules]

\- Review both assessments criterion by criterion.

\- Accept evidence in either direction when it

\- Ignore claims that cannot be verified from the source article or summary.

\- Keep your previous score when the other assessment is not persuasive.

\- Record changes criterion by criterion.

[SE\_OUTPUT\_ADJ]

## B Dataset Implementation Details

## B.1 Korean Essay Scoring

We use the NIKL Grading Writing Data 2024, a Korean essay corpus released by the National Institute of the Korean Language. The corpus provides structured JSON documents in which each essay is recorded at the paragraph level, accompanied by rubric-based holistic scores assigned independently by two trained human annotators. The original scoring rubric covers three dimensions: content (maximum 25 points), organization (maximum 10 points), and expression (maximum 10 points). To place all dimensions on a common scale, we linearly normalize each raw score to a 5-point scale via $s _ { 5 \mathrm { p t } } = ( s _ { \mathrm { r a w } } / s _ { \mathrm { m a x } } ) \times 5 $ For each essay, we compute per-evaluator normalized scores and their arithmetic mean; the mean serves as the gold reference throughout all experiments. The corpus contains essays written in response to prompts covering a range of argumentative and expository topics (Q4–Q9). For each of the six prompts, we randomly sample 100 essays without replacement (seed = 42, using Python’s random.sample), yielding a final dataset of 600 essays. Paragraph-level form fields are concatenated with double newlines to reconstruct a plain-text essay body for judge input.

## B.2 SummEval

We use the SummEval benchmark, accessed via the mteb/summeval dataset on HuggingFace Hub (test split). SummEval provides human quality annotations for machine-generated summaries of CN-N/DailyMail news articles, covering four dimensions: coherence, consistency, fluency, and relevance, each rated on a 1–5 scale by three independent human annotators.

We sample 100 news articles from the test split (seed = 42, using HuggingFace’s shuffle().select()) and expand each article across all available summarization systems. For each (article, summary) pair, the gold score per criterion is computed as the arithmetic mean of the three annotators’ scores. No additional normalization is applied, as the original annotations are already on a 1–5 scale consistent with our essay scoring setup.

The final dataset consists of 700 (article, summary) pairs (100 articles × 7 summarization systems), stored as individual JSON files organized by system name.

## C Model and Inference Details

Table 4 lists the judge models and their OpenRouter identifiers.

<table><tr><td>Model</td><td>OpenRouter identifier</td></tr><tr><td>GPT-4o-mini Gemma3-27B</td><td>openai/gpt-4o-mini</td></tr><tr><td>Gemma3-12B</td><td>google/gemma-3-27b-it google/gemma-3-12b-it</td></tr><tr><td>Gemma3-4B</td><td>google/gemma-3-4b-it</td></tr><tr><td>Qwen3.5-9B Qwen3.5-27B</td><td>qwen/qwen3.5-9b qwen/qwen3.5-27b</td></tr></table>

Table 4: Judge models and OpenRouter identifiers.

Qwen3.5-9B requires reasoning: {effort: none} and response\_format: json\_object to suppress chain-of-thought output. Failed calls (after 3 retries with exponential backoff) are recorded as judge\_error and excluded from analysis.

## C.1 Output Formatting and Parsing

All judge outputs are required to follow the same JSON schema so that scores and rationales can be parsed consistently across models and protocols. For models that support structured output options, we enable JSON-format responses and suppress auxiliary reasoning traces that are not part of the target output schema.

These settings are used only to ensure parseable outputs and do not change the task instruction, rubric, judge role, scoring scale, or information available to the judge. The same output schema is applied across all judging conditions.

## C.2 Score-Masked MAD

In the Score-Masked MAD condition, the numeric scores produced by the other agent are removed before being shared. Criterion-level numeric score fields are excluded from the shared response. In addition, numeric tokens in textual rationales and adjustment notes are replaced with [NUM]. The prompt explicitly states that the other evaluator’s numeric scores are hidden and that numbers in textual rationales are masked.

## D Score-Masked MAD analysis

Table 5 reports per-model RMSE and final interagent score gap for Consensus-based MAD and Score-Masked MAD across both tasks. Score masking worsens average RMSE in both domains (Essay: 0.935 → 1.040; SummEval: 0.813 → 0.994) and substantially increases the final inter-agent gap (Essay: 0.176 → 0.385; SummEval: 0.076 → 0.224).

A minority of model–task pairs show local RMSE improvements under masking (GPT-4omini and Qwen3.5-27B on Korean Essay), but in those same cases the inter-agent gap widens considerably, indicating that agents reach less coordinated final positions even when absolute alignment is marginally preserved.

The systematic gap widening under masking is consistent with the view that numerical scores serve as coordination signals between agents during deliberation. Removing them reduces inter-agent convergence without restoring alignment with human ratings, suggesting that score exposure in standard Consensus-based MAD provides mutual calibration rather than a pure anchoring bias.

<table><tr><td></td><td colspan="4">Korean Essay</td><td colspan="4">SummEval</td></tr><tr><td></td><td colspan="2">RMSE</td><td colspan="2">Final Gap</td><td colspan="2">RMSE</td><td colspan="2">Final Gap</td></tr><tr><td>Model</td><td>MAD</td><td>Masked</td><td>MAD</td><td>Masked</td><td>MAD</td><td>Masked</td><td>MAD</td><td>Masked</td></tr><tr><td>GPT-4o-mini</td><td>1.207</td><td>1.152</td><td>0.213</td><td>0.736</td><td>1.664</td><td>2.016</td><td>0.124</td><td>0.505</td></tr><tr><td>Gemma3-4B</td><td>0.600</td><td>0.656</td><td>0.560</td><td>0.421</td><td>0.469</td><td>0.556</td><td>0.067</td><td>0.123</td></tr><tr><td>Gemma3-12B</td><td>0.922</td><td>1.316</td><td>0.170</td><td>0.450</td><td>0.490</td><td>0.634</td><td>0.031</td><td>0.180</td></tr><tr><td>Gemma3-27B</td><td>0.526</td><td>0.703</td><td>0.093</td><td>0.255</td><td>0.457</td><td>0.873</td><td>0.037</td><td>0.229</td></tr><tr><td>Qwen3.5-9B</td><td>1.077</td><td>1.187</td><td>0.021</td><td>0.194</td><td>0.831</td><td>0.833</td><td>0.051</td><td>0.108</td></tr><tr><td>Qwen3.5-27B</td><td>1.281</td><td>1.227</td><td>0.002</td><td>0.256</td><td>0.969</td><td>1.054</td><td>0.145</td><td>0.201</td></tr><tr><td>Average</td><td>0.935</td><td>1.040</td><td>0.176</td><td>0.385</td><td>0.813</td><td>0.994</td><td>0.076</td><td>0.224</td></tr></table>

Table 5: Effect of numerical score masking on RMSE and final inter-agent gap. Masking peer scores generally widens the gap between agents and worsens average RMSE, suggesting that explicit score sharing functions more as a coordination signal than as the sole source of anchoring bias.

## E Model Specific Analysis

Table 6 reports RMSE and Spearman correlation for each judge model and protocol separately. The aggregate results in Table 1 reflect a central tendency across models, but individual models show notable variation.

For most models, the Single Judge achieves the strongest human alignment, consistent with the aggregate findings. Two models deviate from this pattern. For Qwen3.5-27B, Symmetric MAD outperforms the Single Judge on both tasks (Essay: 0.864 vs. 0.938 RMSE; SummEval: 0.734 vs. 0.799 RMSE), suggesting that neutral multi-round interaction can occasionally improve alignment for larger models. Gemma3-27B also shows a better RMSE under Symmetric MAD on Korean Essay (0.512 vs. 0.528), although Single Judge retains an advantage on Spearman correlation and on SummEval. Gemma3-4B shows uniformly small differences across all protocols on both tasks, indicating limited sensitivity to role prompting or interaction structure.

By contrast, GPT-4o-mini exhibits the most pronounced degradation under Consensus-based MAD: Spearman drops from 0.516 to 0.315 on Korean Essay and from 0.481 to 0.296 on SummEval, a decline substantially larger than in other models. Score-Masked MAD produces the weakest ranking alignment for this model on SummEval $( \rho = 0 . 2 2 4 )$ . Gemma3-12B shows a similar vulnerability to score masking on Korean Essay, where Score-Masked MAD reduces Spearman to 0.173—the lowest value observed across all model– protocol combinations.

These model-level patterns suggest that sensitivity to role specialization and score exposure varies considerably across model families and capacity levels. The aggregate ordering (Single Judge > Symmetric MAD > Consensus-based MAD > Score-Masked MAD) reflects a dominant trend rather than a universal rule.

## E.1 Round-wise Mean Score and Gap Trajectories

Figure 3 shows the direct cause of RMSE stability for Gemma3-27B and Gemma3-4B: their mean predicted scores remain close to the gold mean throughout all rounds on both tasks. On Korean Essay (gold mean = 3.49), Gemma3-27B fluctuates between 3.50 and 3.57 across rounds, while Gemma3-4B rises only slightly from 3.50 to 3.61. Neither model’s mean drifts far from the gold reference. By contrast, Qwen3.5-27B starts at 2.76 (already 0.73 below gold) and falls further to 2.32, and GPT-4o-mini drops from 3.08 to 2.45 across rounds. These downward drifts directly explain the RMSE increases shown in Figure 2.

Gemma3-12B occupies an intermediate position: its mean starts below gold (3.06) and continues declining to 2.79, consistent with its moderate RMSE increase in Table 6. Notably, Gemma3-12B’s gap trajectory resembles Gemma3-27B’s (Figure 4), yet its mean score drifts much more. This dissociation shows that a converging gap does not guarantee mean score stability: what matters is whether the convergence point is close to the gold reference.

Figure 4 provides the mechanistic complement. Gemma3-4B’s gap stays nearly constant across rounds (0.58–0.62 on Essay), indicating that the two agents never converge toward each other. Because neither agent’s scoring stance dominates the deliberation, the averaged score is not pulled systematically in any direction, keeping the mean near gold. Gemma3-27B’s gap decreases gradually (0.47 to 0.08), meaning agents reach mild consensus, but since the convergence point is close to gold, RMSE remains low. Qwen models’ gaps collapse to near zero by round 1 (Qwen3.5-9B: $1 . 0 0  0 . 0 9  \approx 0 )$ , reflecting rapid agreement on a common score—one that is far below gold due to the strict role’s initial downward bias.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Protocol</td><td colspan="2">Korean Essay</td><td colspan="2">SummEval</td></tr><tr><td>RMSE</td><td> $\rho$ </td><td>RMSE</td><td> $\rho$ </td></tr><tr><td rowspan="5">Gemma3-27B</td><td>Single Judge</td><td>0.528</td><td>0.532</td><td>0.414</td><td>0.540</td></tr><tr><td>Strict Role Judge</td><td>0.584</td><td>0.517</td><td>0.609</td><td>0.505</td></tr><tr><td>Symmetric MAD</td><td>0.512</td><td>0.528</td><td>0.424</td><td>0.508</td></tr><tr><td>Consensus-based MAD</td><td>0.526</td><td>0.492</td><td>0.457</td><td>0.529</td></tr><tr><td>Score-Masked MAD</td><td>0.703</td><td>0.470</td><td>0.873</td><td>0.356</td></tr><tr><td rowspan="5">Qwen3.5-9B</td><td>Single Judge</td><td>0.640</td><td>0.477</td><td>0.750</td><td>0.449</td></tr><tr><td>Strict Role Judge</td><td>1.055</td><td>0.493</td><td>0.936</td><td>0.499</td></tr><tr><td>Symmetric MAD</td><td>0.814</td><td>0.454</td><td>0.763</td><td>0.437</td></tr><tr><td>Consensus-based MAD</td><td>1.077</td><td>0.473</td><td>0.831</td><td>0.460</td></tr><tr><td>Score-Masked MAD</td><td>1.187</td><td>0.457</td><td>0.833</td><td>0.466</td></tr><tr><td rowspan="5">Qwen3.5-27B</td><td>Single Judge</td><td>0.938</td><td>0.534</td><td>0.799</td><td>0.590</td></tr><tr><td>Strict Role Judge</td><td>1.472</td><td>0.542</td><td>1.153</td><td>0.568</td></tr><tr><td>Symmetric MAD</td><td>0.864</td><td>0.558</td><td>0.734</td><td>0.600</td></tr><tr><td>Consensus-based MAD</td><td>1.281</td><td>0.527</td><td>0.969</td><td>0.612</td></tr><tr><td>Score-Masked MAD</td><td>1.227</td><td>0.542</td><td>1.054</td><td>0.569</td></tr><tr><td rowspan="5">GPT-4o-mini</td><td>Single Judge</td><td>0.518</td><td>0.516</td><td>1.093</td><td>0.481</td></tr><tr><td>Strict Role Judge</td><td>1.199</td><td>0.477</td><td>1.587</td><td>0.448</td></tr><tr><td>Symmetric MAD</td><td>0.537</td><td>0.493</td><td>1.310</td><td>0.398</td></tr><tr><td>Consensus-based MAD</td><td>1.207</td><td>0.315</td><td>1.664</td><td>0.296</td></tr><tr><td>Score-Masked MAD</td><td>1.152</td><td>0.452</td><td>2.016</td><td>0.224</td></tr><tr><td rowspan="5">Gemma3-4B</td><td>Single Judge</td><td>0.599</td><td>0.409</td><td>0.460</td><td>0.273</td></tr><tr><td>Strict Role Judge</td><td>0.622</td><td>0.459</td><td>0.465</td><td>0.281</td></tr><tr><td>Symmetric MAD</td><td>0.635</td><td>0.274</td><td>0.429</td><td>0.344</td></tr><tr><td>Consensus-based MAD</td><td>0.600</td><td>0.295</td><td>0.469</td><td>0.266</td></tr><tr><td>Score-Masked MAD</td><td>0.656</td><td>0.298</td><td>0.556</td><td>0.224</td></tr><tr><td rowspan="5">Gemma3-12B</td><td>Single Judge</td><td>0.643</td><td>0.554</td><td>0.575</td><td>0.415</td></tr><tr><td>Strict Role Judge</td><td>0.908</td><td>0.507</td><td>0.554</td><td>0.382</td></tr><tr><td>Symmetric MAD</td><td>1.035</td><td>0.397</td><td>0.461</td><td>0.394</td></tr><tr><td>Consensus-based MAD</td><td>0.922</td><td>0.444</td><td>0.490</td><td>0.294</td></tr><tr><td>Score-Masked MAD</td><td>1.316</td><td>0.173</td><td>0.634</td><td>0.371</td></tr></table>

Table 6: Per-model RMSE and Spearman $\rho$ across judge protocols. Bold indicates the best value per metric, model, and task. The results show that the aggregate trend is dominant but not universal: some models benefit locally from Symmetric MAD, while role-asymmetric protocols remain less stable across models.

## F Significance Tests

Table 7 reports paired-bootstrap significance results for the key aggregate comparisons and a small number of model-level exceptions. The analysis focuses on these comparisons rather than providing confidence intervals for every cell in Table 1; all tests use 10,000 resamples with a fixed seed and treat the judge model as a fixed effect.

The strict role produces a clear shift in score levels (Single Judge vs. Strict Role Judge: $p < 0 . 0 0 0 1$ on both tasks), whereas its effect on rankings remains statistically uncertain (Korean Essay: $p =$

0.54; SummEval: $p = 0 . 2 1 )$ . The nonsignificant near-ties and reversals in the final five comparisons provide transparent counterexamples while showing that the aggregate finding is not driven by a small number of outliers.

## G Lenient Role Judge

## G.1 Aggregate Results

Table 8 reports aggregate results for a standalone Lenient Role Judge condition symmetric to the Strict Role Judge, averaged across six judge models. A complete per-model breakdown is left to future work.

The lenient prompt does not produce an upward bias symmetric to the downward shift induced by the strict prompt: the Lenient-only shifts are +0.086 and −0.057, compared with −0.754 and −0.613 for Strict-only on Korean Essay and SummEval, respectively. The lenient effect is therefore only about one tenth as large, indicating that the consensus downward bias arises not from directional prompting itself but specifically from strict role dominance. Lenient-only reduces RMSE on both tasks (Korean Essay: $0 . 6 4 4  0 . 5 8 2 ;$ SummEval: $0 . 6 8 2  0 . 4 9 8 )$ , but also lowers Spearman correlation (Korean Essay: $0 . 5 0 4  0 . 4 7 8 ;$ SummEval: $0 . 4 5 8  0 . 4 1 7 )$ . Thus, no single protocol dominates on both metrics, and the relatively high reference distributions (means 3.49 and 4.29) suggest that the RMSE improvement may reflect a calibration artifact.

![](images/af910fba6ceca748724af532e82879a25cfaa5926fbc0787b1a9e6a07c19cac7.jpg)

![](images/4eaa4f30c16d6ccdcf8c9fd22a965164a007d82809a82b689dd0176f3a9f0c6e.jpg)

Figure 3: Round-wise mean predicted score ((strict + lenient)/2) under Consensus-based MAD. The dashed line marks the human reference mean for each task. Models whose consensus point remains close to the reference mean maintain lower RMSE, whereas models that drift downward show larger alignment loss.
<table><tr><td>Comparison</td><td>Reported result</td></tr><tr><td>Single Judge vs. Consensus-based MAD (both tasks)</td><td> $p < 0 . 0 0 0 1$ </td></tr><tr><td>Single Judge vs. Strict Role Judge (both tasks)</td><td> $p < 0 . 0 0 0 1$ </td></tr><tr><td>Consensus-based MAD vs. Score-Masked MAD (both tasks)</td><td>p &lt; 0.0001</td></tr><tr><td>Strict Role Judge vs. Consensus-based MAD (RMSE mitigation; Korean Essay / SummEval)</td><td> $p < 0 . 0 0 0 1$ </td></tr><tr><td>Gemma3-27B, Korean Essay: Single vs. MAD</td><td> $p = 0 . 8 5$ </td></tr><tr><td>Gemma3-4B, Korean Essay: Single vs. MAD</td><td> $p = 0 . 6 9$ </td></tr><tr><td>Gemma3-4B, SummEval: Single vs. MAD</td><td>p = 0.46</td></tr><tr><td>Symmetric MAD vs. Single Judge, SummEval</td><td> $p = 0 . 3 2$ </td></tr><tr><td>Single Judge vs. Strict Role Judge, Spearman (Korean Essay / SummEval)</td><td> $\mathit { \bar { p } } = 0 . 5 4 / 0 . 2 1$ </td></tr></table>

Table 7: Aggregate-level paired-bootstrap significance results for the key comparisons and a small number of model-level exceptions; aggregate protocol results are averaged across six judge models.

<table><tr><td>Metric</td><td>Korean Essay</td><td>SummEval</td></tr><tr><td>Mean score shift (Lenient-only)</td><td>+0.086</td><td>-0.057</td></tr><tr><td>Mean score shift (Strict-only; reference)</td><td>-0.754</td><td>-0.613</td></tr><tr><td>RMSE: Lenient-only vs. Single Judge</td><td>0.582 vs. 0.644</td><td>0.498 vs. 0.682</td></tr><tr><td>Spearman: Lenient-only vs. Single Judge</td><td>0.478 vs.0.504</td><td>0.417 vs. 0.458</td></tr></table>

Table 8: Aggregate-level Lenient Role Judge results, averaged across six judge models; Strict Role Judge and Single Judge values are included for comparison.

## G.2 Dominance Analysis

Table 9 reports how far the Consensus-based MAD outcome falls below the arithmetic mean of the Strict-only and Lenient-only score shifts, the midpoint expected under simple averaging.

<table><tr><td>Comparison</td><td>Korean Essay SummEval</td><td></td></tr><tr><td>Mean of Strict and Lenient-only</td><td>-0.334</td><td>-0.335</td></tr><tr><td>Consensus-based MAD</td><td>-0.589</td><td>-0.454</td></tr></table>

Table 9: Aggregate-level dominance analysis, averaged across six judge models.

These values establish strict-stance dominance beyond averaging as the central finding: the consensus mean falls well beyond the arithmetic midpoint of the two standalone conditions.

Inter-agent Score Gap (Absolute) by Round — Consensus-based MAD

![](images/2a3af62909f71e61c4d094c13f9ce52ca7d1215625dfbb61d9d6aa2cb1256d19.jpg)

![](images/2b6154d54abfc6d1480da8640c1039115aa57fb434d678ed69a4501767d3618c.jpg)  
Figure 4: Round-wise absolute inter-agent gap |strict − lenient| under Consensus-based MAD. Smaller gaps indicate stronger convergence between the two agents. Rapid convergence does not necessarily imply better human alignment, because agents may converge toward a score level that is systematically below the human reference.