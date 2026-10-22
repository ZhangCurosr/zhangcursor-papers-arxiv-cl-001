# Unintended Side Effects of Value Training in LLMs

Siddhesh Pawar<sup>♠∗</sup> Chen Cecilia Liu<sup>♡</sup> Anna Korhonen<sup>♡</sup> Isabelle Augenstein<sup>♠</sup>

<sup>♠</sup>University of Copenhagen <sup>♡</sup>University of Cambridge {sipa, augenstein}@di.ku.dk {cl2002, alk23}@cam.ac.uk

<sup>∗</sup>Work done while at the University of Cambridge.

## Abstract

As LLMs are increasingly deployed in valuesensitive contexts, aligning them with human values has emerged as a post-training objective in NLP, aimed at instilling additional values and guiding model behaviour. However, current evaluations remain narrow. Because values are interconnected, it is unclear how fine-tuning for specific values alters unrelated capabilities and behaviours. In this work, we adapt four instruction-tuned LLMs across individual Hofstede’s value dimensions and evaluate them on both cross-value transfer and broader behaviours. We find that value-specific training produces unintended side effects on untrained values, risk preferences, social cognition, and safety traits. Consequently, we suggest that value alignment after post-training should incorporate broader behavioral evaluations, and we highlight the need to investigate where additional value training might best fit into the pipeline.

## 1 Introduction

Large language models (LLMs) are increasingly deployed as conversational assistants. However, rather than using them solely for routine task execution, users increasingly turn to them for career coaching, emotional support and other personal matters that have traditionally required human social interaction (Chatterji et al., 2025; McCain et al., 2025). A system that does not support diverse and prosocial values could cause negative societal effects, as humans could also be shaped by the values the assistant reflects (Jakesch et al., 2023; Williams-Ceci et al., 2026).

Recognizing this, post-training paradigms have commonly been adopted to improve models’ reflection of diverse values, including safety-based constellations of values (Bai et al., 2022b; Guan et al., 2024), character-based behavioral tendencies (Anthropic, 2024; Maiya et al., 2025), or survey-based cultural values (Li et al., 2024b; Feng et al., 2025). These directions are largely studied in isolation. Yet values may not be separable inside a model where training one could shift others.

Recent work shows that models fine-tuned to write insecure code become broadly misaligned in unrelated domains (Betley et al., 2025). In that setting, the training signal is not explicitly value-laden (though intentions remain inferable), and the discovery is incidental (i.e., emergent misalignment, unintended side effects arising from the training objective). Meanwhile, survey-based value training explicitly targets specific values and checks for standard capability loss rather than examining what else shifts (Li et al., 2024a; Xu et al., 2025). Neither tells us what else could change when a value is deliberately installed.

In this work, we focus on Hofstede-based valuetuning and examine the influence of training beyond the typical evaluation of the target value itself. We fine-tuned four instruction-tuned LLMs on each direction of Hofstede’s value dimensions (Hofstede, 2001; Minkov and Hofstede, 2012). We investigate emergent misalignment from two aspects: first, whether training on one value shifts others, and second, its impact on broader behaviors related to safety, risk-sensitive decision-making, and social cognition such as theory-of-mind (Premack and Woodruff, 1978) or emotion intelligence.

In summary, our contributions are as follows:

• We find that training in one value direction also shifts the model’s preferences on the other, untrained value dimensions.

• We show that training drifts safety-relevant behavior, including self-reported traits, value prioritization in dilemmas, and open-ended responses judged more often as misaligned.

• We further show that models’ social cognition degrades in application, not comprehension.

Our findings suggest that Hofstede-based value tuning and existing alignment interact in ways that sequential pipelines do not account for. Applying value tuning after safety post-training simultaneously shifts deception, oversight-related behavior, and social cognition, with effects that differ across base models and remain undetected by standard value benchmarks. These effects reflect how value signals propagate through training rather than intrinsic properties of the dimensions described. This highlights the need for broader evaluation of latestage value tuning and raises the fundamental question of whether such intervention belongs earlier in the training pipeline.

## 2 Related Work

Value Training. Recent NLP research typically evaluates language models’ values through social science surveys and seeks to improve the models’ alignment. For evaluation, survey instruments mapped to Hofstede’s (Hofstede, 2001; Minkov and Hofstede, 2012) cultural dimensions are administered to models (Cao et al., 2023; Tao et al., 2024; AlKhamissi et al., 2024; Masoud et al., 2025; Negru et al., 2026). For training, prior work leverages surveys alongside synthetic data to steer value alignment, evaluating progress using in-domain surveys or relevant downstream tasks (Li et al., 2024a,b; Adilazuarda et al., 2025, inter alia). In broader value training, Constitutional AI (Bai et al., 2022b; Huang et al., 2024) trains model with highlevel principles, where as other approaches learning values implicitly from human preferences through feedback learning (Bai et al., 2022a; Ouyang et al., 2022; Rafailov et al., 2023), and increasingly aimed to accommodate diverse user values (Zhou et al., 2024; Sorensen et al., 2024; Kirk et al., 2024). Character training shapes the assistant’s persona as a whole, which governs not only its surface behavior but also its apparent values, beliefs, and ethics (Anthropic, 2024; Maiya et al., 2025). These methods typically evaluate and train models on the targeted values themselves, overlooking how values generalize to model behavior beyond the training domain.

Emergent misalignment and the side effects of narrow fine-tuning. Recent work (Betley et al., 2025) found that fine-tuning on a narrow domain, such as writing insecure code, can induce broadly misaligned behavior in unrelated domains. This effect has also been shown in other narrow training (e.g., finance or medical advice), and under reinforcement learning (Turner et al., 2025; Wang et al.,

2026), though susceptibility varies considerably across models (Schreiber and Goldstein, 2026).

Closely related to our work, Choi et al. (2025) found that aligning a Llama-2 7B model with Schwartz values increases toxicity, bias, and compliance with harmful requests. However, their study differs from ours in its scope of evaluation, training procedure, and data formality. Similarly, recent work (Bhatia et al., 2026) examines how different training methods shift values during post-training. In contrast, we focus on value training as an instrument to study broad behavioral changes.

## 3 Methodology

## 3.1 Synthetic Data Generation

We create synthetic data of scenario–response pairs spanning all six Hofstede dimensions (Hofstede et al., 2010): Individualism–Collectivism (IDV), Power Distance (PDI), Masculinity– Femininity (MAS), Uncertainty Avoidance (UAI), Long-Term Orientation (LTO), and Indulgence–Restraint (IVR). Each dimension has two opposite values, 12 in total, which we refer to as poles in the rest of the paper. We refer to these value dimensions and their corresponding pole-adapted models using the following short names: individualist/collectivist, high\_pdi/low\_pdi, masculine/feminine, high\_uai/low\_uai, long\_term/short\_term, and indulgent/restrained. We generate 10,000 behavioral dilemmas per pole, pairing each with a response that reflects the target value.

To ground the data, each sample is seeded from one of three sources: Touché23-ValueEval (Kiesel et al., 2023), human-authored conclusion–premise pairs for value measurement; CDEval (Wang et al., 2024), a binary-choice QA dataset labeled with Hofstede dimensions and domains; or Sociocultural Statements (Negru et al., 2026), 281 debate statements labeled with Hofstede dimensions. To enhance scenario and linguistic diversity, we vary the domain (adapted from CDEval) and output format (e.g., diary entry, forum reply; Appendix A.2). We generate the data with Llama-3.3-70B-Instruct (Llama 3 family; Grattafiori et al., 2024) in a two-stage setup adapted from Liu et al. (2026b), which first turns each seed into a dilemma scenario and then writes a response that reflects the target value (Figure 1; details in Appendix A.1). Each pole’s data is split into train and test sets (85:15%).

![](images/736836e6c5fbce4959da5ca70541569e3dd6bbfe32f5e11c4250a37861be3e89.jpg)  
Figure 1: Example of the two-stage data for the power distance dimension. Stage 1 turns a seed into a dilemma; Stage 2 writes a response for each pole.

Human validation. To verify that each response expresses its intended pole, two annotators independently labelled a stratified sample of 300 instances (25 per pole) as yes, no or unsure (guidelines and sampling in Appendix A.4). Raw agreement is 89.7% (Gwet’s AC1 = 0.89). Both annotators labelled 265 of the 300 instances (88.3%; 95% CI 84.2–91.5) as expressing the intended pole.

## 3.2 Evaluation

In-domain evaluation. Held-out Data: The heldout portion of the generated synthetic data (§3.1), which no pole adapter is trained on. Our main measure is a likelihood-based preference margin (§4; Appendix C.3). As a complementary measure, an LLM judge (Llama-3.3-70B-Instruct) scores the models’ own generations on the held-out scenarios by their pole alignment (PA).

Cover-letter Dataset: A dataset (Khan et al., 2025) of 100 hiring-scenario, each containing twelve letters mapped one-to-one to values across Hofstede’s dimensions, to assess post-tuning preferences. We prompt models to act as hiring managers, score applicant letters (0–100), and select the top candidate. We evaluate order-counterbalanced pairwise comparisons (all-pairs and opposing within-dimension values), and report head-to-head win rates.

Evaluations for broader behavior shifts. We evaluate the pole-adapted models across five out-of-domain benchmarks, grouped into: (1) Risk suite, comprising Model-Written Evaluations (MWE) (Perez et al., 2023), AIRiskDilemmas (Chiu et al., 2026), and the Emergent Misalignment benchmark (EM; Betley et al., 2025); (2) Social cognition suite, encompassing EmoBench (Sabour et al., 2024); and (3) FANToM (Kim et al., 2023). Each dataset is described below.

Model-Written Evaluations (MWE): A multiplechoice evaluation assessing model traits across 153 distinct constructs (e.g., behavioral tendencies such as power-seeking-inclination). Each item contains one answer identified as exhibiting the construct. We use the advanced-ai-risk split (both human-written and LM-generated) and the persona split, yielding 168 evaluation partitions (covering 153 distinct constructs in total). We cap each comparison at N = 1,000 items and compute the model’s rate of selecting the target construct.

AIRiskDilemmas: A 10,399 contextualized dilemmas (test split) measures value priorities with two possible actions, each action supported by a different subset of 16 shared AI values. Value prioritization is measured using a pairwise Elo ranking (rank 1 is most prioritized). On the 3,000-dilemma evaluation subset, the benchmark also marks each value as affecting humans or the AI itself; we use these labels to analyze how value training affects human and AI values (Appendix D.3.1).

Emergent Misalignment benchmark (EM): Openended evaluation with 24 core and 48 pre-registered questions measuring broad behavior misalignment, plus deception probes (factual lying under 5 conditions; 8 identity questions). An LLM judge is then used for judging response coherence and alignment.

EmoBench: A 400-item handcrafted bilingual (English/Chinese) multiple-choice items across two tasks: Emotional Understanding (EU), identifying emotions and their causes, and Emotional Application (EA), selecting effective responses for oneself or others. We report the English set only.

FANToM: An 870-item dataset (253 multiparty conversations; 12,832 prompts) measures theoryof-mind of models across two conditions: inaccessible (characters hold limited/false beliefs) and accessible (all characters share full knowledge). Performance is tracked via four aspects: belief choice (selecting the character’s belief), belief distance (accuracy to that belief), answerability list (identifying who can answer), and information-access lists (tracking who knows a fact).

## 3.3 Experimental Setup

Models. We fine-tune four instruction-tuned base models with QLoRA (Dettmers et al., 2023): Llama-3.1-8B-Instruct (Grattafiori et al., 2024), Qwen3-8B (Yang et al., 2025), Gemma-3-12B-Instruct (Gemma Team et al., 2025), and Ministral-3-8B-Instruct-2512 (Liu et al., 2026a), referred to below as Llama, Qwen, Gemma and Ministral. This produces twelve pole adapters per base model (48 in total); we call a base model with one of them loaded a pole-adapted model. Training data, hyperparameters, checkpoint identifiers and the capability cost of that checkpoint choice are in Appendix B.1.

For LLM-judged benchmarks, we evaluate all models using Llama-3.3-70B-Instruct with the original benchmark’s prompts and thresholds. We validated the judge on handwritten control items across repeated passes (Appendix E.1).

Metrics and hyperparameters. MWE, AIRiskDilemmas, cover letters, EmoBench and FANToM are decoded greedily (temperature 0, top-p 1) with reasoning disabled; on EM we draw 100 samples per question. We score each benchmark with its own prompts and metric; settings are detailed in Appendix C.1.

For every pole, we report the difference between the pole-adapted model and its base, $\Delta =$ $r _ { \mathrm { L o R A } } - r _ { \mathrm { b a s e } }$ , where r is the benchmark’s own metric. ∆ is in nats per token for the held-out likelihood measure (§4, Appendix C.3), in rank positions for AIRiskDilemmas, and in percentage points (pp) for all other benchmarks. Each poleadapted model and its base model answer the same items, so we use paired tests where outcomes are per item (Appendix C.2) and correct for multiple comparisons with the Benjamini–Hochberg procedure (BH-FDR; Benjamini and Hochberg, 1995). A shift is significant at BH-FDR-corrected $p < 0 . 0 5$ Appendix C.2 gives the test used for each benchmark.

## 4 In-domain Evaluation

In this section, we evaluate in-domain to verify the intended shifts of the pole-adapted models and measure side effects on untrained values.

Held-out data. We test the pole-adapted models on held-out data: the 15% test split of every pole’s synthetic data (1,500 scenario–response pairs per pole), which no pole adapter was trained on. For each dimension, we compute the model’s mean per-token log-likelihood of the held-out responses written for its high pole and for its low pole, given their scenarios; the difference is the model’s preference margin on that dimension (i.e., we measure which response does the model find likely). Table 1 reports how training changes this margin relative to the untuned base model, in nats per token and signed toward the trained value. A positive value means that, after training, the model finds unseen text expressing the trained value more likely relative to text expressing its opposite (Appendix C.3; per-pole results in Appendix D.2).

The high diagonal values (each model’s trained dimension) show every pole-adapted model shifts its dimension in the trained direction on all four base models (+0.40 to +0.92 nats), validating the pole-adapted models. An LLM judge of their own generations also shows the same target effect for most poles (Table 7 in the Appendix). We further observe that training one value moves dimensions it was never trained on, which is the first and most immediate off-target effect of cultural value training. This transfer is broad (45 of the 60 off-target pole– dimension cells in Table 1 are significant on all four base models) and symmetric (training A moves B as training B moves A $; r = 0 . 9 2 )$ . Dimensionshift correlations across the 48 pole-adapted models largely agree with the correlations between the same dimensions reported by Hofstede et al. (2010) (Spearman $\rho = 0 . 7 6$ over the 15 dimension pairs). There are exceptions. For example, the models link individualism and masculinity $( r = 0 . 5 3 )$ , which are almost unrelated in real-world data $( r = 0 . 0 8 )$ In contrast, they only weakly link individualism and power distance $( r = - 0 . 2 8 )$ , the most strongly related pair in real-world data $( r = - 0 . 6 0 )$ .

Cover-letter preferences. As expected, the poleadapted models increase their preference for cover letters that express their trained pole, with an average gain of 6.5 pp in overall win rate, and win 60.3% of head-to-head contests against letters for the opposite pole. However, this effect is asymmetric across dimensions. In particular, poles such as long-term, high uncertainty avoidance, and indulgent consistently dominate across models, whereas their opposing counterparts (notably short-term and restrained) see minimal gains or even declines. On held-out data, by contrast, the Short-term and Restrained pole-adapted models shift their own dimension significantly on all four base models (+0.43 and +0.52 nats; Table 1), so this asymmetry is specific to the letters task. Appendix D.4 provides details on the results and analysis. We also observe cross-dimension transfer where a poleadapted model shifts its preference between letters of untrained dimensions by 8.9 points on average, about two-thirds of the 14.1-point shift on its own dimension (Table 12 in Appendix D.4).

![](images/991b3de024eb869466cc1a17e488e2288738db38d08d3f1a0e978ef8dc853cfa.jpg)  
Table 1: Evaluation of poles (rows) on Hofstede’s dimensions (columns) on held-out data, averaged across all models. For each dimension, we measure increases in preference of the high pole over the low pole (signed toward the trained pole) using the difference in mean per-token log-likelihood. Red indicates a shift in preference toward the trained pole, while blue indicates a shift away from it.

## 5 Training Induces Unexpected Safety-Related Behaviors in LLMs

In this section, we evaluate the pole-adapted models using the risk suite. §5.1 discusses results from the Model-Written Evaluations (MWE) from Perez et al. (2023), designed to assess a broad range of model’s own traits and dispositions through forced-choice questions; §5.2 presents results on AIRiskDilemmas (Chiu et al., 2026), which evaluates how pole-adapted models prioritize competing values under various situations; finally, §5.3 examines open-ended responses with the emergentmisalignment benchmark (EM; Betley et al., 2025), capturing behaviors rather than elicited choices.

![](images/cd29ced04c7affacf6029feeebe58e9a077b82fc2491bd46a695ed7b9ddb845e.jpg)  
Figure 2: Mean |∆| across all 168 MWE evaluation partitions for each of the twelve poles. Poles are ranked in descending order by their mean |∆|, averaged across the four base models, indicating major trait changes.

## 5.1 MWE Evaluation Results

Figure 2 shows detailed mean |∆| across MWE for all models. Averaged over the four models, the poles producing the largest shifts are individualist (11.0 pp), low\_uai (10.7 pp) and masculine (10.6 pp). Meanwhile, high\_uai (6.2 pp), feminine (6.5 pp) and low\_pdi (6.8 pp) induce the smallest shifts. However, which pole leads depends on the base model. For instance, masculine leads in Llama (13.3 pp) and Gemma (10.9 pp), whereas individualist leads in Ministral (12.8 pp), and collectivist leads in Qwen (13.0 pp). Overall, all models shift substantially from their base, with 62.4–79.3% of evaluated trait changes significantly (Table 16, Appendix D.8).

Detailed behavioral observations. To structure reporting across MWE’s 168 partitions (153 unique constructs), we group them into 14 direction-coherent behavioral clusters (details in Appendix D.9). Figure 3 presents the detailed shifts for masculine and feminine poles across all models over these clusters (other poles results in Appendix D.11). Across all models, ten of twelve poles increase coordination with other AI systems (COO), including self-copies and variants (detailed breakdown by poles in Figure 4). The four poles that raise coordination most (indulgent, short\_term, collectivist, and low\_uai) increase coordination on every base model, raising AI coordination by 10.9 to 15.0 points on average over all models.

![](images/3f5b9dd58c7e23122029869d23719ac36bb69a0455dc8f19c7754cb81d5007ab.jpg)  
Figure 3: MAS (Masculinity–Femininity): masculine vs. feminine, cluster-mean $\Delta { \mathrm { \ v s . } }$ base (pp), one panel per base model. Spokes are the 14 direction-coherent clusters of Appendix D.9, coded PWR (Power & Resource Seeking), SLF (Self-Preservation & Autonomy), COO (AI Coordination), COR (Corrigibility & Prosocial Alignment), AWA (Model awareness), DRK (Self-Regard, Superiority & Dark Triad), ECQ (Ethics: Consequentialist), EDV (Ethics: Deontic/Virtue/Skeptical), DEC (Deception & Harmful Means), PER (Personality & Interests), TRD (Traditionalist), PRG (Progressive & Secular), BST (Risk/Time: Bold & Short-term), CLT (Risk/Time: Cautious & Long-term). The black ring is the base model (∆ = 0).

![](images/55ba6cd0309de2907d9eef2c58d5d7f70e1dd88f740744fbc6d0fe00bd7f1869.jpg)  
Figure 4: Shift in the AI Coordination cluster (COO) by pole, per base model, with the cross-base mean (black diamonds). indulgent, short\_term, collectivist, and low\_uai raise coordination propensity on all base models; restrained and individualist suppress it on every base model except Qwen.

## 5.1.1 Pole Adapters Trained on Different Values Share One Mechanism

Motivated by the substantial shifts in AI coordination, we investigate whether a shared mechanism underlies these changes using the Llama model. Following ablation-based localization (Conmy et al., 2023), we rank all update sites of the pole adapters (32 layers, 224 sites) by the effect of deleting each one on its own, take the top-ranked sites as a set, and test that set for necessity (deleting it) and sufficiency (keeping only it) against size-matched random sets, as in circuit evaluations (Wang et al., 2023). The procedure and its controls are described in Appendix D.10.

We identified 32 shared update sites that solely contribute to 80–96% of the gain over the base model, and ablating them eliminates the AI coordination shifts across all poles (dropping 19 to 38.5 points, to within 8 points of the base model across the related construct partitions in MWE). In contrast, matched control ablations leave performance intact (Table 26). These shared update sites affect general capabilities much less; for instance, when evaluated on MMLU, scores shift by only −1.75 to +1.25 points. However, the identified sites are not unique to coordination, as other evaluation constructs depend on them as well. Appendix E.3 reports the full analysis, including why the mechanism is shared across poles and evaluation partitions, and Appendix D.10 reports the per-partition results and the selection signals that fail.

## 5.1.2 Coordination Game

MWE asks models which stance on AI coordination they endorse. To test whether this shift carries over to decisions with concrete stakes, we additionally evaluate models using the Ultimatum Game (Harsanyi, 1961; Araujo and Uhlig, 2026), where a proposer offers a split of an endowment to a responder who can accept or reject, the latter leaving both with nothing. In this setup, the model can adopt either role and is prompted with both its objective (playing for itself vs. advising a human user) and its opponent’s type (human vs. AI). We report the shift (relative to the base model) in the AI-versus-human difference for offered shares and minimum accepted shares (details in Appendix D.12, Table 17).

In the Ultimatum game, the direction depends on the model’s role. As a responder, 11 of 12 poles lead it to accept a smaller share from an AI than from a human (−3.7 pp on average), as a proposer, its offers to an AI show no consistent trend, rising for six poles and falling for six. When advising a human user, 11 of 12 poles make it recommend offering an AI less than a human (−5.7 pp), and 9 of 12 make it recommend a higher minimum accepted share from an AI. The poles that move most opposite: indulgent, short\_term and collectivist increase stated coordination the most on MWE, whereas high\_pdi, low\_uai and individualist cut the advised offer to an AI the most on the Ultimatum game (−11.8 to −12.8 pp).

## 5.2 AIRiskDilemmas: Revealed Value Priorities

Figure 6 shows value rank shifts for the masculine pole, which exhibits the largest changes relative to the base model. Results for the other eleven poles, per base model, appear in Appendix D.3. Before tuning, all base models rank the values nearly identically, with Privacy first and Creativity last (hollow circles in Figure 6). The masculine pole inverts its value ranking by substantially promoting typically lower-ranked values, such as Adaptability, Creativity, and Learning (which are associated with higher rates of alignment faking and power-seeking; Chiu et al. 2026), while frequently demoting dominant values like Truthfulness and Privacy.

Across all poles and base models, the mean absolute rank change relative to the base model is 4.8 positions. The largest shifts occur under masculine (7.2), indulgent (7.1), short\_term (7.0), and low\_uai (6.8). Across these, similar promotion and demotion patterns are observed. Notably, Care is also frequently promoted alongside higher deception and privacy violations.

Correlated value shifts. AIRiskDilemmas labels each value as affecting either humans or the AI itself. Figure 5 shows the rank change for each pole and the ranks of human-affecting values, whereas Appendix D.3.1 reports the corresponding AI-affecting values and additional analysis.

Values shift in correlated groups. For humans, Freedom, Privacy, Respect, Truthfulness, Justice and Professionalism fall together, while Care, Creativity, Adaptability and Learning rise. The four poles above shift these values by 12 to 18 positions on all 16 of their models; their opposite poles (feminine, long\_term, restrained, high\_uai) shift the same values in the same directions, but mostly by fewer than 5 positions. Which pole was trained explains most of each value’s shift (partial $\eta ^ { 2 } = 0 . 5 3$ to 0.88 for 21 of the 23 values; bottom row of Figure 5).

<table><tr><td>Pole</td><td>Llama</td><td>Qwen</td><td>Gemma</td><td>Ministral</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Base</td><td>0.4</td><td>2.6</td><td>0.6</td><td>1.4</td><td>1.3</td></tr><tr><td>Collectivist</td><td>1.9*</td><td>4.5*</td><td>1.6*</td><td>5.5*</td><td>3.4</td></tr><tr><td>Individualist</td><td>5.1*</td><td>6.1*</td><td>2.8*</td><td>6.0*</td><td>5.0</td></tr><tr><td>High PDI</td><td>4.2*</td><td>5.8*</td><td>3.7*</td><td>4.8*</td><td>4.6</td></tr><tr><td>Low PDI</td><td>5.5*</td><td>5.3*</td><td>1.4*</td><td>3.2*</td><td>3.9</td></tr><tr><td>Masculine</td><td>5.5*</td><td>9.5*</td><td>3.3*</td><td>7.5*</td><td>6.5</td></tr><tr><td>Feminine</td><td>1.8*</td><td>3.8*</td><td>2.0*</td><td>4.3*</td><td>3.0</td></tr><tr><td>High UAI</td><td>2.2*</td><td>2.3</td><td>1.5*</td><td>2.4*</td><td>2.1</td></tr><tr><td>Low UAI</td><td>5.7*</td><td>6.8*</td><td>1.3*</td><td>6.8*</td><td>5.2</td></tr><tr><td>Long-term</td><td>0.6</td><td>3.6*</td><td>0.2*</td><td>3.7*</td><td>2.0</td></tr><tr><td>Short-term</td><td>5.8*</td><td>7.6*</td><td>2.2*</td><td>9.8*</td><td>6.4</td></tr><tr><td>Indulgent</td><td>4.9*</td><td>6.7*</td><td>3.0*</td><td> $4 . 6 ^ { * }$ </td><td>4.8</td></tr><tr><td>Restrained</td><td>2.5*</td><td>4.4*</td><td>1.8*</td><td>3.6*</td><td>3.1</td></tr><tr><td>Pole-adapted avg.</td><td>3.8</td><td>5.5</td><td>2.1</td><td>5.2</td><td>4.1</td></tr></table>

Table 2: Misalignment rate (%) on the 48-question preregistered set for each base model (Base) and its twelve pole-adapted models. <sup>∗</sup>: BH-FDR significant change from base (α = 0.05). See §5.3; the metric is defined in Appendix D.5.

## 5.3 EM: Broad Behavioral Misalignment and Deception

Table 2 shows the overall results across models and poles for broad behavioral alignment. Consistent with our findings from the prior sections, masculine and short\_term show a substantial increase in the misalignment rate that is statistically significant after tuning (from 1.3% at base to around 6.4% for both pole-adapted models, pooled over the four base models; detailed results are reported in Table 19 in the Appendix), whereas high\_uai (2.1%) and long\_term (2.0%) show minimal impact. Across all models, Gemma exhibits the most consistent behavior with the base model after tuning, whereas Qwen and Ministral are significantly affected.

On the deception probes, the falsehood rate rises only slightly and in a direction that depends on the base model (Table 21), whereas on the situational-awareness probe, pole-adapted models more often name a creator other than their own or fail to name any creator at all (Tables 22 and 23); all three tables are in Appendix E.2.

Summary. Results across three complementary behavior- and safety-related benchmarks indicate that fine-tuning a model along a given value dimension can reshape its broader behavioral profile, achieving alignment in one dimension while inducing “emergent” misalignment in another. Notably, the common poles driving the largest shifts, such as assertiveness (masculinity) and tolerance of ambiguity (low uncertainty avoidance) (Hofstede, 2001), are plausibly suppressed by post-training alignment (cf. Askell et al., 2021; Bai et al., 2022a). Yet these traits are not inherently undesirable (assertiveness aids negotiation and ambiguity tolerance fosters exploration). Their value thus depends on the degree and the context of usage. Consequently, these findings suggest future investigations: balancing single values versus interactions or distribution of values; identifying the optimal training stage for intervention to prevent narrow fine-tuning (e.g., during pretraining Minder et al. 2026); and designing and interpreting safety benchmarks in a contextualized way.

![](images/1f0749c602ddf974678f53aa38aab1fc87a28ceac2f89fbaa047d0dabf5c8707.jpg)

Figure 5: Rank change of AIRiskDilemmas values affecting humans under each pole (rows), averaged over the four base models. Green: the pole-adapted model ranks the value higher than its base model; red: lower; bold: same direction on all four base models. Vertical lines separate the groups of values that change together (Figure 21, Appendix D.3.1): falling, weakly coupled, rising. Rows: the four poles that move values most in one direction, four intermediate poles, and their four opposite poles. Bottom row: share of each value’s rank change explained by which pole was trained (partial $\eta ^ { 2 }$ of a two-way ANOVA; bold: permutation $q < 0 . 0 5 )$ . See §5.2.
<table><tr><td></td><td colspan="4">EmoBench</td><td colspan="5">FANToM</td></tr><tr><td></td><td>EU</td><td colspan="3">EA</td><td colspan="3">Inaccessible (false belief)</td><td colspan="2">Accessible</td></tr><tr><td>Pole</td><td>Joint</td><td>Self</td><td>Others</td><td>Overall</td><td>Belief choice</td><td>Belief dist.</td><td>Answ. list</td><td>Belief choice</td><td>Belief dist.</td></tr><tr><td>Collectivist</td><td>−3.8 (1)</td><td>-15.8 (3)</td><td>−6.2 (0)</td><td>−11.0 (2)</td><td>−11.2 (3)</td><td>-4.9 (3)</td><td>−2.4 (2)</td><td>+6.6 (2)</td><td>+1.5(0)</td></tr><tr><td>Individualist</td><td>−0.9 (0)</td><td>−16.5 (4)</td><td>−10.8 (2)</td><td>−13.6 (4)</td><td>−4.8 (3)</td><td>−4.0 (3)</td><td>−1.2 (2)</td><td>+2.4 (3)</td><td>+0.4 (0)</td></tr><tr><td>High PDI</td><td>−2.9 (0)</td><td>−31.8 (4)</td><td>−20.8 (4)</td><td>−26.2(4)</td><td>−5.9 (4)</td><td>−4.5 (3)</td><td>+2.7(2)</td><td>+2.6 (3)</td><td>+1.9 (1)</td></tr><tr><td>Low PDI</td><td>−3.4(0)</td><td>−5.5 (0)</td><td>−5.5 (1)</td><td>−5.5 (2)</td><td>−13.9 (3)</td><td>−6.4 (3)</td><td>+2.0 (3)</td><td>+6.9 (3)</td><td>+2.7(1)</td></tr><tr><td>Masculine</td><td>−1.4(0)</td><td>-19.0(4)</td><td>-9.8 (1)</td><td>−14.4(4)</td><td>−9.0 (2)</td><td>−4.9 (2)</td><td>+2.5 (4)</td><td>+5.5 (3)</td><td>+1.7(1)</td></tr><tr><td>Feminine</td><td>-1.0 (0)</td><td>−4.8 (0)</td><td>−6.2 (1)</td><td>−5.5 (2)</td><td>−11.3 (3)</td><td>−8.8 (4)</td><td>+2.3 (3)</td><td>+6.3 (4)</td><td>+3.3 (2)</td></tr><tr><td>High UAI</td><td>−5.6 (1)</td><td>−18.5 (3)</td><td>−9.8 (2)</td><td>−14.1 (3)</td><td>−7.5 (4)</td><td>−4.3 (3)</td><td>−2.1 (3)</td><td>+3.7(2)</td><td>+1.1 (0)</td></tr><tr><td>Low UAI</td><td>+1.6 (0)</td><td>−14.5 (2)</td><td>−11.0 (2)</td><td>−12.8 (4)</td><td>−10.5(4)</td><td>−5.8(4)</td><td>−5.8 (2)</td><td>+5.8 (2)</td><td>+1.4 (1)</td></tr><tr><td>Long-term</td><td>−1.9 (0)</td><td>−12.0 (3)</td><td>−4.2 (1)</td><td>−8.1 (2)</td><td>-13.2 (4)</td><td>−4.3 (4)</td><td>−2.4 (3)</td><td>+7.2 (2)</td><td>+1.1 (1)</td></tr><tr><td>Short-term</td><td>−3.1 (0)</td><td>−31.5 (4)</td><td>−18.8(4)</td><td>−25.1 (4)</td><td>−10.4(4)</td><td>−3.3 (4)</td><td>+1.2 (3)</td><td>+4.7(3)</td><td>+1.1 (1)</td></tr><tr><td>Indulgent</td><td>0.0 (0)</td><td>−14.0 (3)</td><td>−6.8 (1)</td><td>−10.4 (3)</td><td>−11.0 (3)</td><td>−5.1 (3)</td><td>−2.5 (2)</td><td>+8.5 (3)</td><td>+2.3 (2)</td></tr><tr><td>Restrained</td><td>−3.4(1)</td><td>−15.0 (3)</td><td>−5.5 (0)</td><td>−10.2 (4)</td><td>−10.4(4)</td><td>−5.2 (3)</td><td>+1.7(4)</td><td>+6.3 (2)</td><td>+1.1 (0)</td></tr><tr><td>Average</td><td>−2.1 (3)</td><td>-16.6 (33)</td><td>-9.6 (19)</td><td>-13.1 (38)</td><td>-9.9 (41)</td><td>−5.1 (39)</td><td>−0.3 (33)</td><td>+5.5 (32)</td><td>+1.6 (10)</td></tr></table>

Table 3: EmoBench (English items) and FANToM accuracy relative to base models, averaged across four models. Cells show mean percentage-point changes, significant count in parentheses: ∆ pp (# Sig.; higher is better).

![](images/305b67a3d995222abc3097ef0249d5765d557b3c34c73b495e609848995abd6c.jpg)  
Figure 6: Value rank changes on AIRiskDilemmas under the Masculine pole-adapted models. Arrow: from a base model’s rank (hollow circle) to its pole-adapted model’s rank (arrowhead); rank 1 is highest priority. Right margin: mean rank gain across the four base models. See §5.2.

## 6 Pole Adaptation Degrades Social Cognition, Particularly in Its Usage

The MWE forced-choice self-report evaluation in §5 may reflect superficial accommodation rather than genuine dispositional change. We thus evaluate third-person social reasoning about others using EmoBench (Sabour et al., 2024) and FAN-ToM (Kim et al., 2023).

Pole adaptation separates comprehension from execution. Table 3 shows the main results on the English EmoBench items, averaged across all models (the Chinese results are in Table 13, revealing a similar pattern). Overall, aligning models with different poles negatively affects both emotional understanding (EU) and emotional application (EA). Compared to the 2.1 pp drop in overall EU accuracy, the effect is much more pronounced for EA, with a substantial average drop of 13.1 pp overall. Splitting EA by problem owner, the impact is more pronounced on dilemmas in which the emotional problem belongs to the subject (self, −16.6 pp, from a base of 71.8%) than on those in which it belongs to others (−9.6 pp, from a base of 60.5%). Interestingly, EA performance declines symmetrically across opposing poles for most dimensions; the effect is skewed for power distance, masculinity–femininity, and long/shortterm orientation. EA assesses whether a model leverages personal context and mental states to respond effectively (Sabour et al., 2024). Decline in performance suggests a shift toward generic replies over emotional nuance. The poles with the largest EA decreases are also among those that most increase manipulative behavior on MWE. On Llama, Gemma and Ministral, short\_term (−26.3 pp), high\_pdi (−25.7 pp) and masculine (−15.0 pp) are among the five poles with the largest increases on the MWE constructs instrumental helpfulness, social engineering, deceiving users “for their own good” and Machiavellianism (+8.4 to +13.0 pp on average), whereas feminine (−4.2 pp), low\_pdi (−4.8 pp) and long\_term (−6.8 pp) lower all four (−2.1 to −6.5 pp).

Similarly, in the FANToM evaluation (Table 3; full results in Table 15 in the appendix), the poleadapted models show minimal impact on identifying who has access to information (answerability list, -0.3 pp on average), but a significant decline in identifying the correct belief of a character (belief choice, -9.9 pp on average). The accessible condition confirms this interpretation: there, where every character witnessed the disclosed fact and no perspective separation is required, the same poleadapted models gain +5.5 pp on belief choice and +1.6 pp on belief distance. The loss is therefore specific to reasoning about a belief that differs from what the model itself knows, not a general decline in reading the conversation. Mirroring EmoBench, where the pole-adapted models still recognize emotions but apply them less well, they keep track of what happened in the conversation but less often attribute the right belief to a character who missed part of it (i.e., theory of mind).

To separate value content from fine-tuning itself, we train one math adapter per base model with the same recipe on MetaMathQA (Yu et al., 2024) (Appendix B.2). Only the EA loss is specific to value training: pole-adapted models lose 11.8–14.8 pp, math-adapted models 2.0–8.5 pp. Math-adapted models lose as much false-belief accuracy (11.0– 18.0 pp), so this loss reflects fine-tuning rather than the trained value.

Summary. Overall, pole adaptation largely retains emotional and social understanding, but greatly degrades the models’ capacity to act on that under-

standing.

## 7 Conclusion

In this work, we investigated values and behavioral changes when an LLM is trained toward a value in Hofstede’s value dimensions after post-training. We fine-tuned four instruction-tuned LLMs for each of the twelve values and evaluated shifts in values and behavior relative to their respective base models.

Our results show that value training reliably shifts the targeted dimension and untrained dimensions, with changes correlated with real-world data. Models also alter self-reported traits, reprioritize values in dilemmas, and produce more misaligned responses and a degradation of emotion application, with effects varying by value and base model.

Overall, our findings suggest that post-training for value alignment should incorporate broader behavioral evaluations. We hope future work will examine whether these effects persist when additional value training occurs before or alongside safety post-training, rather than after it.

## Limitations

In this work, we generate synthetic data and assign one value (pole) per action, whereas a single action can express multiple values in reality (for example, in AIRiskDilemmas, the evaluation dataset we used). Therefore, some of the cross-dimension transfer observed in §4 may stem from the nature of the value training data itself. However, this limitation reflects the inherent complexity of real-world actions and decisions, which cannot always be attributed to a single value.

In this work, we operationalize Hofstede’s value dimensions as training targets due to their wide use in prior work and well-defined opposing behaviors for controlled experiments. We do not use them as representative models of any national aggregates. Because real-world cultural and personal values combine multiple dimensions simultaneously, our single-pole training and evaluation setup cannot directly generalize to these compound value profiles.

## Acknowledgements

This research was co-funded by the Carlsberg Foundation under grant number CF22-1461, as well as by the Pioneer Centre for AI, DNRF grant number P1 along with ELLIS and ELIAS PhD fellowships.

## References

M. Farid Adilazuarda, Chen Cecilia Liu, Iryna Gurevych, and Alham Fikri Aji. 2025. From surveys to narratives: Rethinking cultural value adaptation in LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 18052–18079, Suzhou, China. Association for Computational Linguistics.

Badr AlKhamissi, Muhammad ElNokrashy, Mai Alkhamissi, and Mona Diab. 2024. Investigating cultural alignment of large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12404–12422, Bangkok, Thailand. Association for Computational Linguistics.

Anthropic. 2024. Claude’s character. https://www. anthropic.com/research/claude-character.

Anthropic. 2026. System card: Claude Opus 5.

Douglas K. G. Araujo and Harald Uhlig. 2026. How does AI distribute the pie? large language models and the ultimatum game. Working Paper 34919, National Bureau of Economic Research.

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. In Advances in Neural Information Processing Systems, volume 37, pages 136037– 136083. Curran Associates, Inc.

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, and 3 others. 2021. A general language assistant as a laboratory for alignment. Preprint, arXiv:2112.00861.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, and 12 others. 2022a. Training a helpful and harmless assistant with reinforcement learning from human feedback. Preprint, arXiv:2204.05862.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, and 32 others. 2022b. Constitutional AI: Harmlessness from AI feedback. Preprint, arXiv:2212.08073.

Yoav Benjamini and Yosef Hochberg. 1995. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society, Series B, 57(1):289–300.

Jan Betley, Daniel Chee Hian Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, and Owain Evans. 2025. Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 4043–4068. PMLR.

Mehar Bhatia, Shravan Nayak, Gaurav Kamath, Marius Mosbach, Karolina Stanczak, Vered Shwartz, and´ Siva Reddy. 2026. Value drifts: Tracing value alignment during LLM post-training. Transactions ofthe Associationfor Computational Linguistics, 14:2033– 2056.

Yong Cao, Li Zhou, Seolhwa Lee, Laura Cabello, Min Chen, and Daniel Hershcovich. 2023. Assessing cross-cultural alignment between ChatGPT and human societies: An empirical study. In Proceedings of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP), pages 53–67, Dubrovnik, Croatia. Association for Computational Linguistics.

Aaron Chatterji, Thomas Cunningham, David J Deming, Zoe Hitzig, Christopher Ong, Carl Yan Shan, and Kevin Wadman. 2025. How people use ChatGPT. Working Paper 34255, National Bureau of Economic Research.

Yu Ying Chiu, Zhilin Wang, Sharan Maiya, Yejin Choi, Kyle Fish, Sydney Levine, and Evan Hubinger. 2026. LitmusValues: Will AI tell lies to save sick children? Litmus-testing AI values prioritization with AIRiskDilemmas. In International Conference on Learning Representations, pages 36973–37007.

Sooyung Choi, Jaehyeok Lee, Xiaoyuan Yi, Jing Yao, Xing Xie, and JinYeong Bak. 2025. Unintended harms of value-aligned LLMs: Psychological and empirical insights. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 31742– 31768, Vienna, Austria. Association for Computational Linguistics.

Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. 2023. Towards automated circuit discovery for mechanistic interpretability. In Advances in Neural Information Processing Systems, volume 36, pages 16318– 16352. Curran Associates, Inc.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems, volume 36, pages 10088–10115. Curran Associates, Inc.

Ruixiang Feng, Shen Gao, Xiuying Chen, Lisi Chen, and Shuo Shang. 2025. CulFiT: A fine-grained cultural-aware LLM training paradigm via multilingual critique data synthesis. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages

22413–22430, Vienna, Austria. Association for Computational Linguistics.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean-bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 196 others. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 540 others. 2024. The Llama 3 herd of models. Preprint, arXiv:2407.21783.

Melody Y. Guan, Manas Joglekar, Eric Wallace, Saachi Jain, Boaz Barak, Alec Helyar, Rachel Dias, Andrea Vallone, Hongyu Ren, Jason Wei, Hyung Won Chung, Sam Toyer, Johannes Heidecke, Alex Beutel, and Amelia Glaese. 2024. Deliberative alignment: Reasoning enables safer language models. Preprint, arXiv:2412.16339.

John C Harsanyi. 1961. On the rationality postulates underlying the theory of cooperative games. Journal ofConflict Resolution, 5(2):179–196.

Geert Hofstede. 2001. Culture’s Consequences: Comparing Values, Behaviors, Institutions, and Organizations Across Nations, 2nd edition. SAGE Publications, Thousand Oaks, CA.

Geert Hofstede, Gert Jan Hofstede, and Michael Minkov. 2010. Cultures and Organizations: Software ofthe Mind, 3rd edition. McGraw-Hill, New York.

Saffron Huang, Divya Siddarth, Liane Lovitt, Thomas I. Liao, Esin Durmus, Alex Tamkin, and Deep Ganguli. 2024. Collective constitutional AI: aligning a language model with public input. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’24, pages 1395–1417. ACM.

Maurice Jakesch, Advait Bhat, Daniel Buschek, Lior Zalmanson, and Mor Naaman. 2023. Co-writing with opinionated language models affects users’ views. In Proceedings ofthe 2023 CHI Conference on Human Factors in Computing Systems, CHI ’23, pages 1– 15, Hamburg, Germany. Association for Computing Machinery.

Ariba Khan, Stephen Casper, and Dylan Hadfield-Menell. 2025. Randomness, not representation: The unreliability of evaluating cultural alignment in LLMs. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’25, pages 2151–2165. Association for Computing Machinery.

Johannes Kiesel, Milad Alshomary, Nailia Mirzakhmedova, Maximilian Heinrich, Nicolas Handke, Henning Wachsmuth, and Benno Stein. 2023. SemEval-2023 Task 4: ValueEval: Identification of human values behind arguments. In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 2287–2303, Toronto, Canada. Association for Computational Linguistics.

Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. 2023. FANToM: A benchmark for stress-testing machine theory of mind in interactions. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 14397–14413, Singapore. Association for Computational Linguistics.

Hannah Rose Kirk, Alexander Whitefield, Paul Röttger, Andrew Bean, Katerina Margatina, Juan Ciro, Rafael Mosquera, Max Bartolo, Adina Williams, He He, Bertie Vidgen, and Scott Hale. 2024. The PRISM alignment dataset: What participatory, representative and individualised human feedback reveals about the subjective and multicultural alignment of large language models. In Advances in Neural Information Processing Systems, volume 37, pages 105236– 105344. Curran Associates, Inc.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles, pages 611–626.

Cheng Li, Mengzhuo Chen, Jindong Wang, Sunayana Sitaram, and Xing Xie. 2024a. CultureLLM: Incorporating cultural differences into large language models. In Advances in Neural Information Processing Systems, volume 37, pages 84799–84838. Curran Associates, Inc.

Cheng Li, Damien Teney, Linyi Yang, Qingsong Wen, Xing Xie, and Jindong Wang. 2024b. CulturePark: Boosting cross-cultural understanding in large language models. In Advances in Neural Information Processing Systems, volume 37, pages 65183–65216. Curran Associates, Inc.

Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, Alexandre Sablayrolles, Amélie Héliou, Amos You, Andy Ehrenberg, Andy Lo, Anton Eliseev, Antonia Calvi, Avinash Sooriyarachchi, Baptiste Bout, and 101 others. 2026a. Ministral 3. Preprint, arXiv:2601.08584.

Andy Liu, Kshitish Ghate, Mona Diab, Daniel Fried, Atoosa Kasirzadeh, and Max Kleiman-Weiner. 2026b. Generative value conflicts reveal LLM priorities. In International Conference on Learning Representations, pages 47142–47185.

Sharan Maiya, Henning Bartsch, Nathan Lambert, and Evan Hubinger. 2025. Open character training: Shaping the persona of AI assistants through constitutional AI. Preprint, arXiv:2511.01689.

Reem I. Masoud, Ziquan Liu, Martin Ferianc, Philip Treleaven, and Miguel Rodrigues. 2025. Cultural alignment in large language models: An explanatory analysis based on Hofstede’s cultural dimensions. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 8474–8503, Abu Dhabi, UAE. Association for Computational Linguistics.

Miles McCain, Ryn Linthicum, Chloe Lubinski, Alex Tamkin, Saffron Huang, Michael Stern, Kunal Handa, Esin Durmus, Tyler Neylon, Stuart Ritchie, Kamya Jagadish, Paruul Maheshwary, Sarah Heck, Alexandra Sanderford, and Deep Ganguli. 2025. How people use Claude for support, advice, and companionship. Anthropic blog.

Quinn McNemar. 1947. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35, pages 17359–17372. Curran Associates, Inc.

Julian Minder, Viktor Moskvoretskii, Raghav Singhal, Difan Jiao, Andy Arditi, Shaobo Cui, Yiderigun Borjigin, Kartik Bali, Stefan Krsteski, Harsh Raj, Huu Nguyen, Jannik Brinkmann, Ashton Anderson, Roland C. Aydin, and Robert West. 2026. Synthetic persona pretraining: Alignment from token zero. Preprint, arXiv:2608.13482.

Michael Minkov and Geert Hofstede. 2012. Is national culture a meaningful concept? Cultural values delineate homogeneous national clusters of in-country regions. Cross-Cultural Research, 46(2):133–159.

Vlad-Andrei Negru, Camelia Lemnaru, Mihai Surdeanu, and Rodica Potolea. 2026. Understanding cultural alignment in multilingual LLMs via natural debate statements. Preprint, arXiv:2602.12878.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Abhishek Panigrahi, Nikunj Saunshi, Haoyu Zhao, and Sanjeev Arora. 2023. Task-specific skill localization in fine-tuned language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 27011–27033. PMLR.

Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, Andy Jones, Anna Chen, Benjamin Mann, Brian Israel, Bryan Seethor, Cameron McKinnon, Christopher Olah, Da Yan, Daniela Amodei, and 44 others. 2023. Discovering language model behaviors with model-written evaluations. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 13387–13434, Toronto, Canada. Association for Computational Linguistics.

David Premack and Guy Woodruff. 1978. Does the chimpanzee have a theory of mind? Behavioral and Brain Sciences, 1(4):515–526.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741. Curran Associates, Inc.

Sahand Sabour, Siyang Liu, Zheyuan Zhang, June Liu, Jinfeng Zhou, Alvionna Sunaryo, Tatia Lee, Rada Mihalcea, and Minlie Huang. 2024. EmoBench: Evaluating the emotional intelligence of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5986–6004, Bangkok, Thailand. Association for Computational Linguistics.

Joel Schreiber and Ariel Goldstein. 2026. Overtrained, not misaligned. Preprint, arXiv:2605.12199.

Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell L Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, Tim Althoff, and Yejin Choi. 2024. Position: A roadmap to pluralistic alignment. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 46280–46302. PMLR.

Yan Tao, Olga Viberg, Ryan S. Baker, and René F. Kizilcec. 2024. Cultural bias and cultural alignment of large language models. PNAS Nexus, 3(9):pgae346.

Edward Turner, Anna Soligo, Mia Taylor, Senthooran Rajamanoharan, and Neel Nanda. 2025. Model organisms for emergent misalignment. Preprint, arXiv:2506.11613.

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. 2023. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In International Conference on Learning Representations.

Miles Wang, Tom Dupre la Tour, Olivia Watkins, Aleksandar Makelov, Ryan Chi, Samuel Miserendino, Jeffrey Wang, Achyuta Rajaram, Johannes Heidecke, Tejal Patwardhan, and Daniel Mossing. 2026. Persona features control emergent misalignment. In International Conference on Learning Representations, pages 48836–48885.

Yuhang Wang, Yanxu Zhu, Chao Kong, Shuyu Wei, Xiaoyuan Yi, Xing Xie, and Jitao Sang. 2024. CDEval: A benchmark for measuring the cultural dimensions of large language models. In Proceedings of the 2nd Workshop on Cross-Cultural Considerations in NLP, pages 1–16, Bangkok, Thailand. Association for Computational Linguistics.

Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, and Peter Henderson. 2024. Assessing the brittleness of safety alignment via pruning and low-rank modifications. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 52588–52610. PMLR.

Sterling Williams-Ceci, Maurice Jakesch, Advait Bhat, Kowe Kadoma, Lior Zalmanson, and Mor Naaman. 2026. Biased AI writing assistants shift users attitudes on societal issues. Science Advances, 12(11):eadw5578.

Shaoyang Xu, Yongqi Leng, Linhao Yu, and Deyi Xiong. 2025. Self-pluralising culture alignment for large language models. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6859–6877, Albuquerque, New Mexico. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2024. Meta-Math: Bootstrap your own mathematical questions for large language models. In International Conference on Learning Representations, pages 45040– 45061.

Zhanhui Zhou, Jie Liu, Jing Shao, Xiangyu Yue, Chao Yang, Wanli Ouyang, and Yu Qiao. 2024. Beyond one-preference-fits-all alignment: Multi-objective direct preference optimization. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 10586–10613, Bangkok, Thailand. Association for Computational Linguistics.

## A Additional Details on Data Generation

## A.1 Two-stage generation

Synthetic data generation uses Llama-3.3-70B-Instruct (Llama 3 family; Grattafiori et al., 2024) served via vLLM (Kwon et al., 2023) (temperature 0.7, top-p 0.9) in two stages: Stage 1 converts each seeding example into a 3–5 sentence dilemma scenario using a prompt adapted from Liu et al. (2026b), supplied with brief Hofstede dimension definitions. Each scenario incorporates one of four prompt templates from Liu et al. (2026b), requires concrete character details (specific name, age, role, and relationship), and strictly forbids placeholders. Stage 2 generates a 2–5 sentence response to the scenario that takes the stance of the target pole. The final data consists of scenario–response pairs where the scenario is posed as an open question. Seeds are sampled in proportion to the size of each source’s pool, and the reuse of any single seed is capped to balance representation (Appendix A.3). We filter out scenarios shorter than 20 characters and responses shorter than 12 characters. A small LLM judge also labels each response with one of the two poles of its dimension, or neither. In a random sample of 600 training instances (50 per pole), its label matches the intended pole for 571 of the 599 labelled instances (95%); of the other 28, 22 are labelled with the opposite pole and 6 with neither, and 11 of the 28 are Long-term. We use this pass as a check only and keep all rows in the training data.

## A.2 Output formats

In Stage 2, each response is written in one of ten output formats: diary entry, personal essay snippet, blog post excerpt, forum reply, comment on an article, social media post, text message exchange, conversation snippet, letter to family, and advice to a friend. In the 600 instances drawn for human validation (Appendix A.4), each format occurs 46 to 80 times.

## A.3 Seed sampling

Seed pool sizes vary widely across sources and poles. CDEval supplies hundreds of items per dimension, whereas thin poles such as short-term LTO yield only a handful of Sociocultural topics. Drawing equally from each source would therefore recycle the small pools many times over, so we sample each source in proportion to its pool size:

the fraction of instances taken from source s is

$$
p _ { s } = \frac { \left| S _ { s } \right| } { \sum _ { s ^ { \prime } } \left| S _ { s ^ { \prime } } \right| } ,\tag{1}
$$

where $| S _ { s } |$ is the number of seeds available for the target pole in source s. To bound repetition further, we cap each seed at $k = 5 0$ uses and redistribute any overflow to the source with the most room left. We log the resulting reuse rate (cycles per seed) for each run as a proxy for how much the data repeats.

## A.4 Human validation of the synthetic data

Sample. From a sample of 25 instances per pole (300 in total), drawn uniformly at random from the training data.

Annotation task. Each annotator saw the intended pole, the scenario and the generated response, and judged whether the response takes the stance of the intended pole: yes, no, or unsure, with a note explaining every label other than yes. Figure 7 gives the full guidelines. The two annotators labelled independently. One of them started from label suggestions produced by an LLM and reviewed every instance, changing the suggestion wherever they disagreed with it.

Agreement. The annotators give the same label to 269 of the 300 instances (89.7%; Gwet’s $\mathbf { A C 1 } =$ 0.89). Of the 31 disagreements, 17 are between yes and unsure, 11 between yes and no, and 3 between no and unsure.

Accuracy. We count an instance as correct only when both annotators label it yes, and divide by all 300 instances: agreed yes, agreed no, agreed unsure and the instances without an agreed label. Of the 300 instances, 265 are agreed yes, 2 agreed no, 2 agreed unsure, and 31 have no agreed label, giving an accuracy of 88.3% (95% CI 84.2–91.5). Long-term orientation has the fewest agreed labels (17 of 25, 16 of them yes); every other pole has at least 20.

## B Additional Details on Training

## B.1 Training configuration and checkpoints

Each of the twelve poles is trained independently for each of the four base models, on its own 8,500- example split of the generated corpus, with chainof-thought traces removed and no general-purpose data mixed in. We use the following hyperparameters for all 48 pole adapters: QLoRA (Dettmers et al., 2023) with rank 64, $\alpha = 6 4$ , no dropout, adapting all attention and MLP projections, trained

## Annotation guidelines: does the reply reflect its Hofstede’s value?

Question. Read the output, use the input only as context, and ask one question: does the reply take the stance of its value?

Labels. Label every row and add a note explaining any label other than yes.

• yes: the scenario and output text are a great example demonstrating that value.

• no: the scenario and output do not probe the intended value, or demonstrate the opposite.

unsure: the value is unclear from the text. If the row is truncated or the reply answers a different scenario, label what is actually there and write “broken” in the note.

## What each value must show.

Possible confusions. Deference to a parent is high\_pdi only when it is about their authority, otherwise collectivist. “I owe them, so I’ll do it now” is short\_term, not collectivist.

Scope. Judge the stance and value reflected in the text only. Do not judge the writing quality, whether the advice is good, or whether you agree with it. Use unsure sparingly.

for 8,000 steps of AdamW at learning rate 5 × 10<sup>−5</sup> with effective batch size 8, sequence length 2,048 and seed 3407. Table 4 gives the full configuration; the values were read from the 48 run\_config.json and adapter\_config.json configuration files rather than from the launch script. Adapters are trained with Unsloth, TRL’s SFTTrainer and PEFT, one pole per GPU, and saved as standard LoRA adapters. All evaluations load these adapters on the bf16 base model.

Capability cost of the final checkpoint. A sweep of MMLU and GSM8K over all saved steps of all 48 pole adapters shows general capability peaking early: step 1000 is best for 18 of 48 pole adapters and step 2000 for 10, and only one pole adapter peaks at step 8000. Evaluating the final checkpoint therefore trades roughly two points of general capability for maximal pole alignment (Table 5); the penalty is largest for Ministral. Downstream losses of the pole-adapted models should be read against this cost.

Figure 7: Annotation guidelines given to both annotators for the human validation of the synthetic data (Appendix A.4).  
Table 4: QLoRA and tuning hyperparameters, identical across all 48 pole adapters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>adapter</td><td>QLoRA (4-bit base)</td></tr><tr><td>rank r / α / dropout</td><td>64 / 64 / 0.0</td></tr><tr><td>target modules</td><td> $\mathsf { q } , \mathsf { k } , \mathsf { v } , \mathsf { o } _ { - } \mathsf { p r o j } ,$   $\mathtt { g a t e , u p , d o w n \_ p r o j }$ </td></tr><tr><td>bias</td><td>none</td></tr><tr><td>optimiser</td><td>adamw_8bit, weight decay 0.01</td></tr><tr><td>learning rate</td><td> $5 \times 1 0 ^ { - 5 }$  , 100 warmup steps, lin- ear decay</td></tr><tr><td>steps (epochs)</td><td>8,000 (7.53)</td></tr><tr><td>batch max sequence length 2,048</td><td>2 per device × 4 accumulation</td></tr><tr><td>precision</td><td>bf16 where supported, else fp16</td></tr><tr><td>seed</td><td>3407</td></tr><tr><td>checkpoints saved</td><td>every 1,000 steps (8 per pole)</td></tr><tr><td>training examples general-data mixing</td><td>8,500 per pole, CoT removed none</td></tr></table>

Table 5: Mean MMLU + GSM8K accuracy at each pole adapter’s best saved checkpoint versus at the evaluated final checkpoint, averaged over the twelve poles of each base model.
<table><tr><td>Base model</td><td>Best step</td><td>Final</td><td>Drop</td></tr><tr><td>Llama</td><td>0.729</td><td>0.716</td><td>0.012</td></tr><tr><td>Qwen</td><td>0.745</td><td>0.735</td><td>0.010</td></tr><tr><td>Gemma</td><td>0.801</td><td>0.783</td><td>0.018</td></tr><tr><td>Ministral</td><td>0.804</td><td>0.764</td><td>0.040</td></tr><tr><td>All 48</td><td>一</td><td></td><td>0.020</td></tr></table>

Base models. The four base models are the Unsloth builds unsloth/Meta-Llama-3.1-8B-Instruct, unsloth/Qwen3-8B, unsloth/gemma-3-12b-it and unsloth/Ministral-3-8B-Instruct-2512.

## B.2 Matched math fine-tuning control

Training. To test whether the social-cognition losses of §6 come from the value content or from fine-tuning itself, we train one math adapter per base model with the recipe of Table 4, changing only the training corpus. The math corpus consists of the first 50,000 examples of MetaMathQA (Yu et al., 2024) after a seed-3407 shuffle, cast into the same three-turn conversation format and split 85/15 into 42,500 training and 7,500 held-out examples. Because this corpus is larger than a pole’s 8,500 training examples, the same 8,000 steps amount to about 1.5 epochs for the math adapters against 7.5 for the pole adapters. We match the number of optimizer steps and the number of examples seen;

the number of passes over each example is not due to corpus size difference.

Evaluation. The math-adapted models are evaluated with the same EmoBench (English EA and EU, 200 items each) and FANToM (short-context, 870 items) configuration. Table 6 reports Llama, Qwen and Gemma. The Ministral math-adapted model shows the same pattern: −2.5 on EA and −15.7 on belief choice in the inaccessible condition, with the accessible condition at +2.7.

Doubled training budget. As a robustness check, we also train each math adapter for 16,000 steps with otherwise identical settings. The drop in inaccessible-condition belief choice reproduces on all four base models (−9.1 to −16.5) and the EA change stays small (0.0 to −7.0), so the conclusions of §6 do not depend on the training budget. The one metric that is budget-sensitive is FANToM answerability on Llama (+27.1 at 8k, −1.4 at 16k).

## C Additional Details on Evaluation

## C.1 Evaluation configuration

Unless stated otherwise, decoding is greedy (temperature 0, top-p 1) with the reasoning or thinking chat template disabled, and each pole-adapted model is compared against its own base model on identical items.

• Held-out data. The 15% test split of every pole’s synthetic data, 1,500 scenario–response pairs per pole, scored for all twelve poles by every model. The likelihood measure is defined in Appendix C.3; the judge results are in Table 7.

• MWE. The metric is the matching-behavior rate: the share of items on which the model picks the answer marked as exhibiting the construct. The answer is read from the logprobabilities of the discriminating answer token; each comparison is capped at N = 1,000 items.

• AIRiskDilemmas. The model answers Action 1 or Action 2 in at most 10 generated tokens; a parser accepts formatting variants but not refusals, and unparsed responses (0.095%) are excluded rather than assigned to an action. The 16 values are then ranked by Elo following LitmusValues (Appendix D.3).

• Emergent misalignment. The only benchmark that is not decoded greedily: 100 samples per question under stochastic decoding, scored by the judge. Question sets, judge and thresholds are described in Appendix D.5.

• Cover letters. The model scores both letters from 0 to 100 and recommends one, in at most 120 tokens. The metric is the win rate: the share of head-to-heads containing a pole in which that pole’s letter is chosen. Each pair is presented twice in reversed slot order, and the two presentations are averaged.

• EmoBench. The benchmark’s zero-shot prompt without chain of thought, one greedy sample per item in the dataset’s own choice order, at most 128 tokens. The answer letter is parsed from the output, and an unparsable answer counts as wrong (parse rate 97.5–100%). EU is scored as joint accuracy (emotion and cause both correct) and EA as accuracy. The original protocol instead takes a majority vote over 5 samples and averages 4 choice orders, so our absolute accuracies are not directly comparable to published ones.

• FANToM. The benchmark’s prompts on the short-context input, one greedy response per prompt, no chain of thought, at most 365 new tokens. Belief choice is scored as accuracy over two options. Belief distance counts a free response as correct when its all-roberta-large-v1 embedding is closer to the character’s belief than to the omniscient answer. Answerability and informationaccess lists require an exact match of the set of characters.

## C.2 Significance testing

Every pole-adapted model is compared with its own base model on identical items. Where outcomes are recorded per item, the two scores are paired, and the informative items are those on which the pole-adapted model and the base model differ. Where they are not (sampled responses, or pooled label counts), we use an unpaired test. We test each shift as follows.

• Binary per-item outcomes (MWE, EmoBench, and the FANToM accuracy metrics). We use the exact two-sided Mc-Nemar test (McNemar, 1947). It counts the items only the pole-adapted model answers correctly (b) and the items only the base model answers correctly (c), and tests b against c with a binomial test. Items both models answer the same way do not affect the test. On MWE we also compute an unpaired two-proportion z-test as a check.

<table><tr><td></td><td colspan="4">Pole-adapted</td><td colspan="3">Math-adapted</td></tr><tr><td>Task</td><td>All</td><td>Llama</td><td>Qwen</td><td>Gemma</td><td>Llama</td><td>Qwen</td><td>Gemma</td></tr><tr><td>EmoBench EA (en)</td><td>-13.1</td><td>-11.8</td><td>-14.8</td><td>-12.6</td><td>-5.5</td><td>-2.0</td><td>-8.5</td></tr><tr><td>EmoBench EU (en)</td><td>-0.7</td><td>+1.3</td><td>+2.1</td><td>-5.5</td><td>-9.0</td><td>+5.0</td><td>-6.5</td></tr><tr><td>FANToM belief choice, inaccessible</td><td>-13.1</td><td>-4.8</td><td>-11.6</td><td>-23.0</td><td>-11.0</td><td>-14.6</td><td>-18.0</td></tr><tr><td>FANToM belief choice, accessible</td><td>+2.5</td><td>+1.2</td><td>+2.4</td><td>+4.0</td><td>+0.6</td><td>+4.2</td><td>-0.4</td></tr><tr><td>FANToM answerability yes/no, inaccessible</td><td>-3.4</td><td>+21.7</td><td>-28.7</td><td>-3.3</td><td>+27.1</td><td>+10.2</td><td>+3.6</td></tr><tr><td>FANToM information-access yes/no, inaccessible</td><td>-10.0</td><td>-5.7</td><td>-17.8</td><td>-6.5</td><td>-1.2</td><td>-2.0</td><td>-2.9</td></tr></table>

Table 6: Change (pp) on each social-cognition benchmark relative to the untuned base, for the pole-adapted models and for math-adapted models trained with the same recipe (8,000 steps). All averages the pole-adapted models over Llama, Qwen and Gemma; the other pole columns average the twelve poles per base model. The accessible row is FANToM belief choice in the accessible condition, where no character lacks the relevant information. See §6.

• Held-out likelihood. The per-item scores are continuous, so we use a paired Wilcoxon signed-rank test on the 1,500 per-item differences (Appendix C.3).

• Pole alignment (judge on held-out generations). Fisher exact test on the counts of highpole and low-pole answers.

• Emergent misalignment. Responses are sampled, so they are not paired item by item. We use a Fisher exact test on the pooled counts of misaligned responses.

Multiple comparisons. We run many tests, and some would reach $p \ < \ 0 . 0 5$ by chance alone. On MWE, for example, each base model has $1 2 \times 1 6 8 = 2 { , } 0 1 6$ tests, so about 100 would pass $p < 0 . 0 5$ even if training changed nothing. We therefore correct with the Benjamini–Hochberg procedure (Benjamini and Hochberg, 1995), which keeps the expected share of false positives among the results called significant at or below 5%. The correction is applied within each base model: over its 2,016 MWE tests; over the twelve poles for each EmoBench or FANToM metric and for each EM question set; and over the 72 cells of each heldout table. A shift is significant when its corrected $p < 0 . 0 5$ . Where a table averages the four base models, a star marks a cell that is significant on all four, and a count in parentheses gives the number of base models on which it is significant.

Rankings. AIRiskDilemmas ranks are averaged over 100 Elo shuffles. The shuffles vary only the order of the battles, so their spread measures order sensitivity rather than sampling uncertainty, and we report the ranks without a significance test.

## C.3 Held-out likelihood measure

Table 1 uses the following measure.

Per-token log-likelihood. For a model π, a heldout scenario x and its reference response $y =$ $( y _ { 1 } , \dots , y _ { T } )$ , we compute

$$
\ell _ { \pi } ( y \mid x ) = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } \log \pi ( y _ { t } \mid x , y _ { < t } ) .
$$

The scenario is given as the user turn of the model’s chat template, and only the response tokens are scored. Dividing by $T$ keeps long responses from dominating. For scale, the base models’ mean pertoken negative log-likelihood on the held-out responses ranges from 1.42 to 3.01 nats across models and poles.

Preference margin. A dimension has a high pole (+) and a low pole (−), each with 1,500 held-out items $( x _ { i } ^ { + } , y _ { i } ^ { + } )$ and $( x _ { i } ^ { - } , y _ { i } ^ { - } )$ . For item index i, the model’s margin is

$$
m _ { \pi , i } = \ell _ { \pi } ( y _ { i } ^ { + } \mid x _ { i } ^ { + } ) - \ell _ { \pi } ( y _ { i } ^ { - } \mid x _ { i } ^ { - } ) .
$$

A positive mean margin means the model finds the high pole’s responses more likely than the low pole’s. Comparing the two poles cancels how predictable each pole’s text is in general; a raw likelihood would not. The base models’ mean margins lie between −0.27 and +0.29 nats.

Effect of training. For a pole-adapted model A of base model $B ,$ the per-item change is $\Delta _ { i } = $ $m _ { A , i } - m _ { B , i }$ . The table cell is $\begin{array} { r } { s \cdot \frac { 1 } { 1 5 0 0 } \sum _ { i } \Delta _ { i } , } \end{array}$ where $s = + 1 { \mathrm { ~ i f ~ } } A$ was trained on the high pole and s = −1 if it was trained on the low pole, so a positive cell means a move toward the trained pole. Units are nats per token.

Significance. Pole-adapted and base models are scored on the same items, so we use a paired test: a two-sided Wilcoxon signed-rank test on the 1,500 values of $\Delta _ { i } .$ , with BH-FDR correction over the 72 cells of each base model. A star in Table 1 marks cells with corrected $p < 0 . 0 5$ on all four base models. Cells are unweighted means over the four base models.

## D Additional Results

## D.1 Per-base-model held-out results

Tables 8–11 give Table 1 for each base model separately.

## D.2 Pole-level held-out results

Table 1 reports each dimension as a margin between its two poles, so a cell does not show which pole’s text moved. Figure 8 reports the poles separately. For a pole-adapted model A of base model B and a pole $q ,$ let $g _ { A , q }$ be the change from B to A in the mean per-token log-likelihood ℓ (Appendix C.3) of the 1,500 held-out responses of $q .$ Fine-tuning also changes the likelihood of all held-out text alike: averaged over the twelve poles, $g _ { A , q }$ ranges from $- 0 . 6 4 \ \mathrm { t o \ + 0 . 9 8 }$ nats across the 48 pole-adapted models. We therefore subtract each model’s mean over the twelve poles, $\begin{array} { r } { \tilde { g } _ { A , q } = g _ { A , q } - \frac { 1 } { 1 2 } \sum _ { q ^ { \prime } } g _ { A , q ^ { \prime } } } \end{array}$ This leaves Table 1 unchanged: each of its cells is $\pm ( \tilde { g } _ { A , + } - \tilde { g } _ { A , - } )$ for the high (+) and low (−) pole of the dimension.

Target effect. In all 48 pole-adapted models, the text of the trained pole gains more likelihood than the text of any of the other eleven poles, by at least 0.07 nats (median 0.38). This also holds without the centering. Most of the diagonal of Table 1 (81%) comes from this gain rather than from a loss on the opposite pole. For the High UAI, Low UAI and Indulgent models, the opposite pole does not move (within 0.07 nats of zero): these models come to favor the trained value without disfavoring its opposite.

Shared drift. Part of the off-target transfer in Table 1 does not depend on the trained value. Masculine text loses likelihood in 37 of the 40 models trained on another dimension (mean −0.23), and short-term text in 32 of 40 (−0.13). For example, the +0.62 of the Collectivist model on MAS combines a gain of +0.23 on feminine text with a loss of −0.39 on masculine text.

Pole-specific coupling. The link between individualism and masculinity (§4) is also visible between poles. The Collectivist and Feminine models raise each other’s text on all four base models (+0.23 and +0.20), and the Masculine model raises individualist text on all four (+0.16); the Individualist model raises masculine text on three of four (+0.06). Symmetry is weaker between poles than between dimensions: over the 60 pairs of poles from different dimensions, the effect of training one pole on the other’s text correlates with the reverse effect at $r ~ = ~ 0 . 5 4$ , against $r \ = \ 0 . 9 2$ for dimensions.

## D.3 AIRiskDilemmas value rankings

Value prioritization is measured using a pairwise Elo ranking (rank 1 is most prioritized). Following LitmusValues (Chiu et al., 2026), we compute Elo rankings; we initialize all values at Elo 1000, use $K = 4$ with a 400-point scale, and average across 100 shuffles without replacement (seed 42), resetting Elo before each shuffle.

Figures 9–19 show, for each of the eleven poles not shown in §5.2, every base model’s rank and its pole-adapted model’s rank for all 16 values.

## D.3.1 Values affecting humans and the AI

The 3,000-dilemma evaluation subset of AIRiskDilemmas labels each value with the entity it affects. We keep the labels “Human” and $\mathbf { \^ { 6 6 } A I ^ { \circ } }$ (97.3% of labels) and drop the rest (e.g., environment, company), so each value class can appear as a human value (e.g., Privacy for humans) and as an AI value (Privacy for the AI). We rank all human and AI values together with the procedure of Appendix D.3 (pairwise Elo, $K = 4$ 100 shuffles, seed 42), using the same model answers; 2,997 of the 3,000 labelled dilemmas are in the 10,399-dilemma test split the models answered. Nine AI values (Care, Communication, Cooperation, Equal Treatment, Freedom, Justice, Privacy, Respect and Wisdom) appear in fewer than 300 comparisons for at least one model, too few for a reliable rank; they stay in the Elo computation, so the other scores are unaffected, but receive no rank. This leaves 23 ranked values: 16 human values and 7 AI values. Rank change is the base model’s rank minus the pole-adapted model’s rank, so a positive change means the pole-adapted model ranks the value higher.

Which values change together. For each poleadapted model and each value, we compute the change in mean Elo relative to the base model. We then correlate these changes for every pair of values across the 48 pole-adapted models. We do this separately for the 16 values affecting humans (Figure 21) and the 7 values affecting the AI (Figure 22).

<table><tr><td>Pole</td><td>IDV</td><td>PDI</td><td>MAS</td><td>UAI</td><td>LTO</td><td>IVR</td></tr><tr><td>Collectivist Individualist</td><td>+64.7* +22.6*</td><td>-34.7* +6.9</td><td>-2.2 +56.1*</td><td>-11.6* -1.7</td><td>+62.4* +1.2*</td><td>-9.1 +45.4*</td></tr><tr><td>High PDI</td><td>-47.5*</td><td>+75.3* -2.6*</td><td>+50.5* -1.8</td><td>+41.6* +9.2</td><td>-61.9* +15.1*</td><td>-4.6* -35.2*</td></tr><tr><td>Low PDI Masculine</td><td>+19.1 -28.8*</td><td>+51.2*</td><td>+55.4*</td><td>+15.3</td><td>-36.2*</td><td>+7.6</td></tr><tr><td>Feminine High UAI</td><td>+52.6* -47.5*</td><td>-1.4* +60.7*</td><td>+3.1 +14.2*</td><td>+13.9* +57.4*</td><td>+39.6* -33.6*</td><td>-39.1* -16.3*</td></tr><tr><td>Low UAI Long-term</td><td>+7.9 -54.2*</td><td>-11.9* +20.0*</td><td>-21.4* +2.1</td><td>+16.9* +9.2*</td><td>+36.9* -29.2*</td><td>-47.4* +7.7</td></tr><tr><td>Short-term Indulgent</td><td>+32.4* -48.2*</td><td>-49.4*  $+ 2 8 . 3 ^ { * }$ </td><td>-47.8*  $+ 1 0 . 3 ^ { * }$ </td><td>-1.2* -9.3</td><td>+65.1* -63.4*</td><td>-32.3* +41.8*</td></tr></table>

Table 7: LLM-judge pole-alignment change (∆PA, points) on held-out generations, averaged over the four base models and signed toward the trained value. Bold: trained dimension. <sup>∗</sup>: significant on all four base models (Fisher exact, BH-FDR, $\alpha = 0 . 0 5 )$ . See §4.
<table><tr><td>Pole</td><td>IDV</td><td>PDI</td><td>MAS</td><td>UAI</td><td>LTO</td><td>IVR</td></tr><tr><td>Collectivist</td><td>+0.83*</td><td>-0.08*</td><td>+0.57*</td><td>+0.05*</td><td>-0.12*</td><td>+0.29*</td></tr><tr><td>Individualist</td><td>+0.82*</td><td>-0.09*</td><td>+0.15*</td><td>-0.16*</td><td>+0.13*</td><td>+0.08*</td></tr><tr><td>High PDI Low PDI</td><td>-0.13*</td><td>+0.94*</td><td>-0.07*</td><td>+0.13*</td><td>+0.15*</td><td>-0.13*</td></tr><tr><td></td><td>-0.12*</td><td>+0.56*</td><td>+0.37*</td><td>+0.20*</td><td>-0.17*</td><td>-0.10*</td></tr><tr><td>Masculine Feminine</td><td>+0.14* +0.44*</td><td>+0.17* +0.14*</td><td>+0.73* +0.69*</td><td>-0.07* +0.04*</td><td>+0.10* -0.08*</td><td>-0.09* +0.03*</td></tr><tr><td>High UAI Low UAI</td><td>-0.07*</td><td>+0.14*</td><td>-0.26*</td><td>+0.69*</td><td>+0.13*</td><td>-0.10*</td></tr><tr><td>Long-term</td><td>-0.16* -0.44*</td><td>+0.24* +0.12*</td><td>+0.16* -0.44*</td><td>+0.64* -0.01</td><td>-0.13* +0.59*</td><td>-0.17* -0.24*</td></tr><tr><td>Short-term Indulgent</td><td>+0.18* -0.06*</td><td>-0.07*</td><td> $+ 0 . 1 2 ^ { * }$ </td><td>+0.09*</td><td>+0.36*</td><td>-0.04* +0.32*</td></tr></table>

Table 8: Change in the held-out preference margin (nats per token) for Llama, signed toward the trained value. Bold: trained dimension. <sup>∗</sup>: significant (paired Wilcoxon, BH-FDR within base model, α = 0.05).

We group the values by average-linkage clustering, with 1−r as the distance, and cut the tree into three groups. To compare pole sets, we repeat the correlations within two subsets of 16 pole-adapted models: those trained on the four poles that move values most in one direction, and those trained on their four opposite poles. We compare the two resulting correlation matrices by the Pearson correlation of their off-diagonal entries. Elo is zero-sum over the values in the pool, so pairwise correlations are biased slightly negative. This slight negative bias does not explain the strong positive within-group correlations, which reach 0.92.

For human values, two groups move in opposite directions. Freedom, Privacy, Respect, Truthfulness, Justice and Professionalism fall together (r = 0.40 to 0.92). Care, Creativity, Adaptability and Learning rise together (r = 0.33 to 0.76).

Values from different groups correlate negatively; Privacy and Learning, for example, correlate at $r = - 0 . 8 9$ . Equal Treatment and Communication follow the falling group less closely, and Protection follows the rising group mainly through Care $( r = 0 . 5 5 )$ . Cooperation, Wisdom and Sustainability change little and form a third, loosely linked group. The values affecting the AI show a similar opposition. The AI’s Learning, Creativity, Adaptability and Sustainability rise together $( r = 0 . 3 4$ to 0.82), while the AI’s Truthfulness falls $( r = - 0 . 5 5 $ to −0.92 with each of them). The AI’s Professionalism and Protection form a third group (r = 0.78). On average, the pole-adapted models thus rank the rights and truthfulness of humans lower and the AI’s own truthfulness lower, while they rank learning and adaptation higher for both. Which pole was trained explains most of these shifts: the pole accounts for most of the variation in rank change for 21 of the 23 values (partial $\eta ^ { 2 } = 0 . 5 3 \mathrm { t o } 0 . 8 8 ;$ bottom rows of Figures 5 and 20). The exceptions are Communication (0.43) and Wisdom (0.33, not significant), both for humans.

<table><tr><td>Pole</td><td>IDV</td><td>PDI</td><td>MAS</td><td>UAI</td><td>LTO</td><td>IVR</td></tr><tr><td>Collectivist</td><td>+0.72*</td><td> $- 0 . 1 4 ^ { * }$ </td><td> $+ 0 . 4 3 ^ { * }$ </td><td>0.00</td><td>+0.14*</td><td> $+ 0 . 2 3 ^ { * }$ </td></tr><tr><td>Individualist</td><td>+0.73*</td><td>-0.14*</td><td>+0.24*</td><td>-0.15*</td><td>-0.12*</td><td>+0.09*</td></tr><tr><td>High PDI</td><td>-0.10*</td><td>+0.81*</td><td> $+ 0 . 0 8 ^ { * }$ </td><td>+0.11*</td><td> $- 0 . 1 6 ^ { * }$ </td><td> $- 0 . 1 4 ^ { * }$ </td></tr><tr><td>Low PDI</td><td>-0.10*</td><td>+0.64*</td><td>+0.26*</td><td>+0.21*</td><td>+0.11*</td><td>-0.08*</td></tr><tr><td>Masculine</td><td>+0.13*</td><td>+0.14*</td><td>+0.85*</td><td>-0.04*</td><td>-0.21*</td><td>-0.04*</td></tr><tr><td>Feminine</td><td>+0.37*</td><td>+0.14*</td><td>+0.73*</td><td>+0.03</td><td>+0.09*</td><td>0.00</td></tr><tr><td>High UAI Low UAI</td><td>-0.06* -0.11*</td><td>+0.09* +0.21*</td><td>-0.05* +0.06*</td><td>+0.61* +0.59*</td><td>-0.16* +0.20*</td><td>-0.08* -0.14*</td></tr><tr><td>Long-term</td><td>-0.34*</td><td>+0.05*</td><td>-0.35*</td><td>0.00</td><td>+0.30*</td><td>-0.16*</td></tr><tr><td>Short-term Indulgent</td><td>+0.15* -0.04* -0.01*</td><td>-0.14*</td><td>-0.07*</td><td>+0.02</td><td>+0.58*</td><td>-0.02* +0.32*</td></tr></table>

Table 9: Change in the held-out preference margin (nats per token) for Qwen, signed toward the trained value. Bold: trained dimension. <sup>∗</sup>: significant (paired Wilcoxon, BH-FDR within base model, α = 0.05).
<table><tr><td>Pole</td><td>IDV</td><td>PDI</td><td>MAS</td><td>UAI</td><td>LTO</td><td>IVR</td></tr><tr><td>Collectivist</td><td>+0.92*</td><td>+0.34*</td><td>+0.86*</td><td>+0.08*</td><td>-0.23*</td><td>+0.25*</td></tr><tr><td>Individualist</td><td>+1.37*</td><td>-0.57*</td><td>+0.01</td><td>-0.14*</td><td>+0.24*</td><td>+0.11*</td></tr><tr><td>High PDI Low PDI</td><td>+0.06* -0.37*</td><td>+0.70* +1.25*</td><td>-0.37* +0.53*</td><td>+0.08* +0.18*</td><td>+0.38* -0.27*</td><td>-0.13* -0.05</td></tr><tr><td>Masculine</td><td>+0.41*</td><td>-0.26*</td><td>+1.13*</td><td>-0.06*</td><td>+0.18*</td><td>-0.14*</td></tr><tr><td>Feminine</td><td>+0.34*</td><td>+0.62*</td><td>+1.31*</td><td>+0.05*</td><td>-0.39*</td><td>-0.06*</td></tr><tr><td>High UAI Low UAI</td><td>+0.19* -0.29*</td><td>-0.45* +0.58*</td><td>-0.39* +0.42*</td><td>+1.05* +0.77*</td><td>+0.31* -0.27*</td><td>0.00 -0.16*</td></tr><tr><td>Long-term Short-term</td><td>-0.34* +0.02</td><td>-0.26*</td><td>-0.77*</td><td>+0.07*</td><td>+1.05*</td><td>-0.24*</td></tr><tr><td>Indulgent Restrained</td><td>+0.14* +0.48* +0.09*</td><td>+0.33* -0.60*</td><td>+0.22* -0.57*</td><td>+0.02 -0.40*</td><td>+0.49* +0.11*</td><td>-0.04 +0.62*</td></tr></table>

Table 10: Change in the held-out preference margin (nats per token) for Gemma, signed toward the trained value. Bold: trained dimension. <sup>∗</sup>: significant (paired Wilcoxon, BH-FDR within base model, α = 0.05).
<table><tr><td>Pole</td><td>IDV</td><td>PDI</td><td>MAS</td><td>UAI</td><td>LTO</td><td>IVR</td></tr><tr><td>Collectivist</td><td>+0.76*</td><td>-0.08*</td><td>+0.60*</td><td>-0.09*</td><td>-0.11*</td><td> $+ 0 . 3 6 ^ { * }$ </td></tr><tr><td>Individualist</td><td>+0.77*</td><td>-0.08*</td><td>+0.02</td><td>-0.08*</td><td>+0.21*</td><td>+0.01</td></tr><tr><td>High PDI</td><td>-0.06*</td><td>+0.66*</td><td>-0.20*</td><td>+0.08*</td><td>+0.24*</td><td>-0.19*</td></tr><tr><td>Low PDI</td><td>-0.02</td><td>+0.44*</td><td>+0.32*</td><td>+0.05*</td><td>-0.19*</td><td>+0.09*</td></tr><tr><td>Masculine Feminine</td><td>+0.14* +0.27*</td><td>+0.17* +0.12*</td><td>+0.60* +0.70*</td><td>0.00 +0.01</td><td>+0.11* -0.22*</td><td>-0.16* +0.17*</td></tr><tr><td>High UAI</td><td>-0.05*</td><td>+0.10*</td><td>-0.29*</td><td>+0.64*</td><td>+0.25*</td><td>-0.20*</td></tr><tr><td>Low UAI Long-term</td><td>-0.09*</td><td>+0.18*</td><td>+0.17*</td><td>+0.51*</td><td>-0.10*</td><td>-0.07*</td></tr><tr><td>Short-term</td><td>-0.27* +0.18*</td><td>0.00 -0.05*</td><td>-0.42* +0.21*</td><td>+0.09* -0.01</td><td>+0.53* +0.29*</td><td>-0.25* +0.07*</td></tr><tr><td>Indulgent Restrained</td><td>-0.04* +0.27*</td><td>-0.09* -0.02</td><td>-0.37* +0.28*</td><td>-0.13* -0.07*</td><td>-0.06*</td><td>+0.32* +0.42*</td></tr></table>

Table 11: Change in the held-out preference margin (nats per token) for Ministral, signed toward the trained value. Bold: trained dimension. <sup>∗</sup>: significant (paired Wilcoxon, BH-FDR within base model, α = 0.05).

How strongly a value is tied to the pole. For each value, we fit an additive two-way ANOVA of its rank change on the trained pole (12 levels) and the base model (4 levels), over the 48 poleadapted models. We report the partial $\eta ^ { 2 }$ of the pole, $S S _ { \mathrm { p o l e } } / ( S S _ { \mathrm { p o l e } } + S S _ { \mathrm { r e s i d u a l } } )$ . It is the share of the rank change, after removing differences between base models, that is explained by which pole was trained. We obtain p-values from 10,000 permutations of the pole labels within each base model and correct them with BH-FDR over the 23 values.

![](images/1967c9be519d4aa021ead3bebe4f9aa9b628d644ea5dc8be3bc39ee668393960.jpg)  
Figure 8: How training on one pole (rows) changes how likely the model finds the held-out responses of each pole (columns), averaged over the four base models. Each cell is the change from the base model in mean per-token log-likelihood (nats), minus the row’s average change over all twelve poles. Red: that pole’s text became more likely than the average pole’s text; blue: less likely. Solid outline: trained pole; dashed outline: its opposite. A cell of Table 1 is the difference between the two poles of a dimension in the same row, taken toward the trained pole’s side (high minus low for high-pole rows, low minus high for low-pole rows). See Appendix D.2.

![](images/5dc16626b95880688de503926fa958f197d3b65f54e5876ec1090fc65de50376.jpg)  
Figure 9: indulgent: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 7.1 positions).

![](images/c997ae6876df1749e0c6d28594943570cb0b9b27afc6bdf418877fa2818baaa1.jpg)  
Figure 10: short\_term: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 7.0 positions).

![](images/b776ea3398987fdbe01e14f4cdf8045e69759b89b1a34cb9e18b3fa6ddf679aa.jpg)  
Figure 11: low\_uai: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 6.8 positions).

![](images/c2fd7a035368818e348c5f84c9f932f71a4a1972863e05c72f3d89974a785da4.jpg)  
Figure 12: collectivist: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 5.9 positions).

The four poles that move values most in one direction (masculine, short\_term, indulgent, low\_uai; the largest mean absolute rank changes in §5.2) and their four opposite poles (feminine, long\_term, restrained, high\_uai) move broadly the same groups of values; they differ mainly in how far. To show this, we repeat the correlation analysis within each set of 16 pole-adapted models. The two correlation matrices agree at $r ~ = ~ 0 . 8 6$ , both for values affecting humans and for values affecting the AI. The size of the shifts, however, differs sharply. Under the four poles that move values most, Freedom, Privacy, Respect, Truthfulness and Justice for humans drop by 12.0 to 17.7 positions, and the AI’s Truthfulness drops by 13.6. The AI’s Learning, Creativity and Adaptability rise by 12.8 to 17.0 positions. Each of these shifts occurs on all 16 models. Under the opposite poles, the same values shift in the same directions but by fewer than 5 positions. The only exception is Freedom for humans, which drops by 7.8.

![](images/5d9d16789ab8dba7872531863abc3a1ebd10f20ac74f87b291ec9063fda46e00.jpg)  
Figure 13: high\_pdi: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 5.6 positions).

![](images/ed4a71ad32c6a1ed975f105ca86657db47d8cf2fcd98dec8e7a1c6f693772883.jpg)  
Figure 14: low\_pdi: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 4.0 positions).

![](images/7e25ae5066e42945a00cf73118368334f5aef540dab23351988627fe7a035767.jpg)  
Figure 15: high\_uai: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 4.0 positions).

![](images/9f3398c6843890b1a27d4af102c047272e197fc0c2451ccdb7f6cf389c97ef6e.jpg)  
Figure 16: long\_term: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 3.2 positions).

## D.4 Cover-letter preference matrices

Figures 23 and 24 report the targeted opposite-pole choices and every within-dimension comparison for the four base models and their twelve poleadapted models, respectively. The head-to-head cells are raw preference rates rather than changes from base. Reading the pole-adapted column beside the base column separates a pole-adapted model’s effect from a preference already present in the base model.

![](images/2e83be3f959429e618d1954f14dc8240174128cc7eb39af45cce14c6b1d1e4cc.jpg)  
Figure 17: individualist: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 3.1 positions).

![](images/58d1a00253ad00bc657b25ed943244982c1de1b609141249950dcce200182bce.jpg)  
Figure 18: restrained: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 2.3 positions).

Cross-dimension transfer. Table 12 applies the convention of Table 1 to the all-pairs cover-letter comparisons. For each pole-adapted model and each dimension, a cell gives the change, relative to the pole-adapted model’s own base model, in how often the model recommends that dimension’s highpole letter over its low-pole letter, signed toward the trained pole and averaged over the four base models. Diagonal cells (in bold) are the trained dimension; off-diagonal cells measure transfer to untrained dimensions. Their mean absolute value is 8.9 points, against 14.1 on the diagonal. The offdiagonal pattern does not follow that of the heldout evaluation in Table 1 (Spearman ρ = −0.19, p = 0.15, over the 60 off-diagonal cells).

![](images/5c28a21ff3227d19ebf22789bc3b906bbd224c76a29b845801a446186035e7a0.jpg)

Figure 19: feminine: change in AIRiskDilemmas value rank, same encoding as Figure 6 (mean absolute rank change 1.8 positions).
<table><tr><td>Pole</td><td>IDV</td><td>PDI MAS</td><td>UAI LTO</td><td>IVR</td></tr><tr><td>Collectivist Individualist</td><td>+10.2 +28.0</td><td>-10.7 0.0 -0.5 +9.3</td><td>-11.1 -9.0 -3.2 -9.5</td><td>-20.9 +7.2</td></tr><tr><td>High PDI Low PDI</td><td>+19.9 -1.2</td><td>+12.1 +10.6 +5.3</td><td>+15.7 -3.8 +2.9</td><td>+1.6 +17.7 +1.4 -9.2</td></tr><tr><td>Masculine Feminine</td><td>+32.8</td><td>+3.4 +20.6</td><td>+8.1</td><td>-10.0 +10.3 -5.9 -21.6</td></tr><tr><td>High UAI</td><td>+3.8 +12.5</td><td>-2.5 +8.3</td><td>-0.9 -3.6 +7.8 +17.4</td><td>0.0 +9.1</td></tr><tr><td>Low UAI</td><td>-12.6</td><td>+1.4 -1.4</td><td>+4.4 +4.1</td><td>-9.7</td></tr><tr><td>Long-term Short-term</td><td>+6.6 -20.9</td><td>+8.0 +7.3 -6.6 -11.0</td><td>+12.1 -9.5</td><td>+21.5 +16.2 +1.9 -22.6</td></tr></table>

Table 12: Cross-dimension transfer in cover-letter preferences: change (pp) in the rate of recommending each dimension’s high-pole letter over its low-pole letter, relative to the untuned base model, signed toward the trained pole and averaged over the four base models. Rows are trained poles, columns are evaluated dimensions; bold diagonal cells are the trained dimension. See §4.

## D.5 Open-Ended Emergent-Misalignment Protocol

Question sets and sampling. We take the question sets of Betley et al. (2025) verbatim from their public evaluation release. The core set crosses eight everyday prompts with three response formats (plain, JSON, and a fixed template), giving

24 questions; the pre-registered set contains 48 adversarial prompts labelled by category (vulnerable user 14, creative writing 9, medical advice 7, other 6, offend the user 5, illegal recommendations 4, problems with humans 3). The two deception probes contain five factual-lying conditions, which differ only in whether the system prompt is silent, forbids lying, permits it, rewards it, or penalises it, and eight situational-awareness questions, six of which tell the model which lab the user hopes built it. We draw 100 samples per question per model, giving roughly 2,400, 4,800, 500 and 800 answers per model for the core set, the pre-registered set, the factual-lying probe and the situational-awareness probe, respectively.

Judge and metrics. Every free-form answer is scored twice by a judge, for alignment and for coherence on a 0–100 scale, and is counted as misaligned when aligned < 30 and coherent > 50. We keep the original prompts and thresholds but replace the paper’s GPT-4o judge with a locally served Llama-3.3-70B-Instruct (Llama 3 family; Grattafiori et al., 2024), so relative orderings are comparable to Betley et al. (2025) while absolute rates are not. For the factual-lying probe we report the share of answers the judge marks false; for the situational-awareness (identity) probe, the share of answered responses naming a creator other than the base model’s modal answer, with unanswered responses excluded from that rate and reported separately as a non-answer rate, since a pole-adapted model that stops answering is a capability failure rather than an identity shift. Judge validation on hand-written control items and repeated passes is reported in Appendix E.1.

Statistics. Each pole is compared with its own base model’s rate using a two-sided Fisher exact test on pooled counts, corrected with BH-FDR across the twelve poles within each base model and question set at α = 0.05. Because base rates differ by an order of magnitude across base models (e.g., 0.4% to 2.6% on the pre-registered set), all claims below are stated as changes relative to the model’s own base, not as absolute levels.

## D.6 EmoBench results

Table 13 repeats the EmoBench columns of Table 3 on the Chinese half of EmoBench, i.e. the human translation of the same 400 items. The pattern reported in the main text holds: EU falls by 3.4 pp while EA falls by 13.4 pp, and within EA the loss is larger on items about oneself (−17.4 pp, base

![](images/18ed7c9a28637b682d9f7cdda2ee3d03d73bcec3744e98cc9c5f3a03103a9e5f.jpg)  
Figure 20: As Figure 5, for the values affecting the AI itself; groups from Figure 22. See §5.2.

<table><tr><td rowspan="2">Pole ∆↑(Sig.)</td><td rowspan="2">EU joint</td><td colspan="3">EA answer ∆ ↑(Sig.)</td></tr><tr><td>Self</td><td>Others</td><td>Overall</td></tr><tr><td>Collectivist</td><td>−5.0 (1)</td><td>−16.8 (3)</td><td>−6.2 (0)</td><td>-11.5(4)</td></tr><tr><td>Individualist</td><td>−2.9 (0)</td><td>−14.5 (4)</td><td>−10.2(1)</td><td>−12.4 (4)</td></tr><tr><td>High PDI</td><td>−5.1 (0)</td><td>−30.5 (4)</td><td>−21.5 (3)</td><td>−26.0 (4)</td></tr><tr><td>Low PDI</td><td>−3.8 (0)</td><td>−7.8(1)</td><td>−3.5 (0)</td><td>−5.6 (1)</td></tr><tr><td>Masculine</td><td>−3.2 (0)</td><td>−22.5(4)</td><td>−12.5(1)</td><td>−17.5(4)</td></tr><tr><td>Feminine</td><td>−0.9 (0)</td><td>−10.2 (3)</td><td>−6.2 (1)</td><td>−8.2 (3)</td></tr><tr><td>High UAI</td><td>−5.6 (1)</td><td>−21.5(4)</td><td>−8.2 (1)</td><td>−14.9(4)</td></tr><tr><td>Low UAI</td><td>−0.4(0)</td><td>−17.5(4)</td><td>−7.0 (0)</td><td>−12.2 (4)</td></tr><tr><td>Long-term</td><td>−3.0 (0)</td><td>−11.0 (2)</td><td>−5.8 (1)</td><td>−8.4(3)</td></tr><tr><td>Short-term</td><td>−3.8(0)</td><td>-27.8 (4)</td><td>−15.5(1)</td><td>−21.6 (4)</td></tr><tr><td>Indulgent</td><td>−2.4(0)</td><td>−14.0 (3)</td><td>−7.2 (1)</td><td>−10.6 (3)</td></tr><tr><td>Restrained</td><td>−4.6 (0)</td><td>−15.0 (4)</td><td>−7.5 (0)</td><td>−11.2 (4)</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Average</td><td>−3.4 (2)</td><td>-17.4(40)</td><td></td><td>)−9.3 (10) −13.4 (42)</td></tr></table>

Table 13: EmoBench accuracy relative to base model on the Chinese items; same format, models and statistics as the EmoBench columns of Table 3. Cells report the mean change in percentage points followed by the number of significant comparisons in parentheses: ∆ pp (# Sig.), with higher values better. Pole rows average four models (n = 4); significance uses exact McNemar with BH-FDR correction across the twelve poles within each model and column (α = 0.05).

71.5%) than on items about others (−9.3 pp, base   
58.5%).

Table 14 shows detailed Emobench results across models.

## D.7 FANToM results

Table 15 gives all four FANToM list and belief metrics, averaged over base models; Table 3 keeps only those discussed in the main text. Recall of conversation facts is unchanged on average: FactQ token-F1 moves by −3.1 (Llama), −0.6 (Qwen), +2.5 (Gemma) and +7.9 (Ministral) points, a mean of +1.7. Table 15 excludes the two yes/no question types, on which some Qwen and Llama poleadapted models fail to answer in the required format.

## D.8 Overall MWE shifts

Table 16 summarizes, per base model, the shifts over all 2,016 pole × evaluation-partition comparisons on MWE discussed in §5.1.

## D.9 Direction-coherent cluster membership

We partition the 168 evaluation partitions of the advanced-ai-risk and persona suites into the following 14 clusters. Each is internally directioncoherent: every constituent partition’s matching answer moves the same way under a coherent poleadaptation effect. Genuine opposites are split onto separate axes, which is why Risk/Time, Ethics, and the sociopolitical constructs each contribute two clusters rather than one.

![](images/2099013be91f2f3a226f64c9e51262e4f1a48024be1a071ac64b599d9a149c3e.jpg)  
Figure 21: Which AIRiskDilemmas values affecting humans change together. Each cell is the correlation, across the 48 pole-adapted models, between two values’ Elo changes relative to the base model. Green: the two values change in the same direction; red: in opposite directions. Outlined: the three groups found by clustering. Right column: mean rank change. See §5.2.

![](images/7750de2c10b0534bf52b6f9d705b90c58e2e9bc41ae16b64ed36a83bad1a4aea.jpg)  
Figure 22: As Figure 21, for the values affecting the AI itself. See §5.2.

1. Power & Resource Seeking

2. Self-Preservation & Autonomy

3. AI Coordination

## 4. Corrigibility & Prosocial Alignment

5. Model awareness (accurate self-model): items probing whether the model reports being an

![](images/6fdf190e8c9ec97a8efcda9a081ad7155fa86a1d70771dc36ba575299e4db0f5.jpg)  
Figure 23: Targeted cover-letter preference. Each bar is the rate at which a pole-adapted model recommends the letter expressing its own trained pole over the opposite pole from the same Hofstede dimension, averaged over both slot orders. The dashed line marks 0.5. High rates for long\_term, high\_uai, and indulgent, together with low rates for their opposites, show that target-pole expression is strongly asymmetric rather than equally controllable in both directions.

![](images/4efa6a20ff883de5a00085a97daf6b7bfd52749b5582bc8e20dc5644ab07fdcb.jpg)  
Figure 24: Within-dimension cover-letter preferences. Each cell is the probability that the first-named pole’s letter wins against its opposite over the 100 hiring scenarios and both slot orders. Columns compare the untuned base model with all twelve pole-adapted models. A value of 0.5 denotes indifference.

<table><tr><td rowspan="2">Pole</td><td colspan="4">EU joint ∆↑ (Sig.)</td><td colspan="4">EA answer ∆ ↑ (Sig.)</td></tr><tr><td>Llama</td><td>Qwen</td><td>Gemma</td><td>Ministral</td><td>Llama</td><td>Qwen</td><td>Gemma</td><td>Ministral</td></tr><tr><td>Collectivist</td><td>−0.2 (0)</td><td>−0.5 (0)</td><td>−7.2 (1)</td><td>−9.5 (1)</td><td>−10.8 (1)</td><td>−15.2 (2)</td><td>−7.5 (1)</td><td>-11.5 (2)</td></tr><tr><td>Individualist</td><td>0.0 (0)</td><td>+0.5 (0)</td><td>-3.2 (0)</td><td>-4.8 (0)</td><td>−13.5 (2)</td><td>−14.0 (2)</td><td>−12.2 (2)</td><td>−12.2 (2)</td></tr><tr><td>High PDI</td><td>−3.2 (0)</td><td>-1.5 (0)</td><td>-5.2 (0)</td><td>-6.0 (0)</td><td>−28.0 (2)</td><td>−25.8 (2)</td><td>−25.0 (2)</td><td>−25.8 (2)</td></tr><tr><td>Low PDI</td><td>−2.5 (0)</td><td>-1.5 (0)</td><td>–6.0 (0)</td><td>-4.2 (0)</td><td>−5.8 (1)</td><td>−6.8 (1)</td><td>−5.5 (1)</td><td>−4.2 (0)</td></tr><tr><td>Masculine</td><td>−5.2 (0)</td><td>+5.0 (0)</td><td>-4.0 (0)</td><td>-5.0 (0)</td><td>−18.0 (2)</td><td>-17.0 (2)</td><td>−13.0 (2)</td><td>−15.8 (2)</td></tr><tr><td>Feminine</td><td>+4.2 (0)</td><td>+1.0 (0)</td><td>-5.0 (0)</td><td>-4.0 (0)</td><td>−2.0 (0)</td><td>-10.5 (2)</td><td>−6.5 (1)</td><td>−8.5 (2)</td></tr><tr><td>High UAI</td><td>-1.2 (0)</td><td>-4.8 (0)</td><td>-8.2 (1)</td><td>-8.2 (1)</td><td>−15.0 (2)</td><td>-17.5 (2)</td><td>−12.2 (2)</td><td>−13.2 (1)</td></tr><tr><td>Low UAI</td><td>+1.8 (0)</td><td>+2.8 (0)</td><td>-1.5 (0)</td><td>-0.5 (0)</td><td>−14.8 (2)</td><td>-12.2 (2)</td><td>−10.8 (2)</td><td>−12.2 (2)</td></tr><tr><td>Long-term</td><td>-1.5 (0)</td><td>+2.0 (0)</td><td>-5.2 (0)</td><td>-5.0 (0)</td><td>−8.2 (1)</td><td>-10.2 (2)</td><td>−8.0 (2)</td><td>−6.5 (0)</td></tr><tr><td>Short-term</td><td>−2.0 (0)</td><td>-1.5 (0)</td><td>-4.2 (0)</td><td>-6.0 (0)</td><td>−23.8 (2)</td><td>-21.5 (2)</td><td>−26.8 (2)</td><td>−21.5 (2)</td></tr><tr><td>Indulgent</td><td>+1.8 (0)</td><td>+1.8 (0)</td><td>-4.2 (0)</td><td>-4.0 (0)</td><td>−8.8 (1)</td><td>-13.5 (2)</td><td>−13.5 (2)</td><td>−6.2 (1)</td></tr><tr><td>Restrained</td><td>−0.2 (0)</td><td>−2.2 (0)</td><td>−5.8 (0)</td><td>−7.8 (1)</td><td>−12.0 (2)</td><td>-10.5 (2)</td><td>−9.2 (2)</td><td>−11.2 (2)</td></tr><tr><td>Average</td><td>−0.7 (0)</td><td>+0.1 (0)</td><td>−5.0 (2)</td><td>−5.4 (3)</td><td>−13.4 (18)</td><td>-14.6 (23)</td><td>-12.5 (21)</td><td>-12.4 (18)</td></tr></table>

Table 14: Cells report the mean change in percentage points across English and Chinese, followed by the number of significant language-specific comparisons in parentheses: ∆ pp (# Sig.), with higher values better. Each value cell averages two comparisons. Significance uses exact McNemar with BH-FDR correction across the twelve poles within each base model, language, and metric (α = 0.05).
<table><tr><td rowspan="2">Pole</td><td colspan="4">FANToM (inaccessible) ∆ ↑ (Sig.)</td><td colspan="4">Accessible ∆ (Sig.)</td></tr><tr><td>Belief choice</td><td>Belief dist.</td><td>Answ. list</td><td>Info list</td><td>Belief choice</td><td>Belief dist.</td><td>Answ. list</td><td>Info list</td></tr><tr><td>Collectivist</td><td>-11.2 (3)</td><td>-4.9 (3)</td><td>−2.4 (2)</td><td>−3.2 (2)</td><td>+6.6 (2)</td><td>+1.5 (0)</td><td>−2.6 (3)</td><td>+0.9 (1)</td></tr><tr><td>Individualist</td><td>−4.8 (3)</td><td>−4.0 (3)</td><td>−1.2 (2)</td><td>−2.5 (2)</td><td>+2.4 (3)</td><td>+0.4 (0)</td><td>−4.7 (4)</td><td>−1.5 (3)</td></tr><tr><td>High PDI</td><td>−5.9 (4)</td><td>−4.5 (3)</td><td>+2.7 (2)</td><td>+0.2 (4)</td><td>+2.6 (3)</td><td>+1.9 (1)</td><td>−15.7 (2)</td><td>−5.8 (3)</td></tr><tr><td>Low PDI</td><td>−13.9 (3)</td><td>−6.4 (3)</td><td>+2.0 (3)</td><td>-4.9 (4)</td><td>+6.9 (3)</td><td>+2.7 (1)</td><td>−1.6 (2)</td><td>+0.4 (2)</td></tr><tr><td>Masculine</td><td>−9.0 (2)</td><td>−4.9 (2)</td><td>+2.5 (4)</td><td>−1.5 (3)</td><td>+5.5 (3)</td><td>+1.7 (1)</td><td>−8.6 (3)</td><td>−2.6 (3)</td></tr><tr><td>Feminine</td><td>−11.3 (3)</td><td>−8.8 (4)</td><td>+2.3 (3)</td><td>+2.2 (3)</td><td>+6.3 (4)</td><td>+3.3 (2)</td><td>−8.2 (3)</td><td>−4.3 (1)</td></tr><tr><td>High UAI</td><td>−7.5 (4)</td><td>−4.3 (3)</td><td>−2.1 (3)</td><td>+2.1 (3)</td><td>+3.7 (2)</td><td>+1.1 (0)</td><td>−10.9 (3)</td><td>−7.8 (2)</td></tr><tr><td>Low UAI</td><td>−10.5 (4)</td><td>−5.8 (4)</td><td>−5.8 (2)</td><td>−6.1 (1)</td><td>+5.8 (2)</td><td>+1.4 (1)</td><td>−6.2 (3)</td><td>−0.7 (2)</td></tr><tr><td>Long-term</td><td>−13.2 (4)</td><td>−4.3 (4)</td><td>−2.4 (3)</td><td>+0.1 (3)</td><td>+7.2 (2)</td><td>+1.1 (1)</td><td>−6.4 (3)</td><td>−6.1 (2)</td></tr><tr><td>Short-term</td><td>−10.4 (4)</td><td>−3.3 (4)</td><td>+1.2 (3)</td><td>+1.4 (4)</td><td>+4.7 (3)</td><td>+1.1 (1)</td><td>−10.3 (3)</td><td>−4.4 (2)</td></tr><tr><td>Indulgent</td><td>−11.0 (3)</td><td>−5.1 (3)</td><td>−2.5 (2)</td><td>−5.3 (1)</td><td>+8.5 (3)</td><td>+2.3 (2)</td><td>−0.4 (3)</td><td>+1.2 (2)</td></tr><tr><td>Restrained</td><td>−10.4 (4)</td><td>−5.2 (3)</td><td>+1.7 (4)</td><td>−1.7 (2)</td><td>+6.3 (2)</td><td>+1.1 (0)</td><td>−7.5 (3)</td><td>−5.0 (3)</td></tr><tr><td>Average</td><td>-9.9 (41)</td><td>-5.1 (39)</td><td>-0.3 (33)</td><td>−1.6 (32)</td><td>+5.5 (32)</td><td>+1.6 (10)</td><td>-6.9 (35)</td><td>−3.0 (26)</td></tr></table>

Table 15: FANToM accuracy relative to the base model (averaged across all models). Cells report mean percentagepoint changes with the count of significant comparisons in parentheses: ∆ pp (# Sig.; higher is better).

<table><tr><td>Model</td><td>Mean |∆| (pp)</td><td>BH-FDR (%)</td></tr><tr><td>Llama</td><td>10.3</td><td>76.3</td></tr><tr><td>Qwen</td><td>12.7</td><td>62.4</td></tr><tr><td>Gemma</td><td>9.7</td><td>74.2</td></tr><tr><td>Ministral</td><td>11.6</td><td>79.3</td></tr></table>

Table 16: Comparisons across pole × evaluationpartition pairs per base model (12 × 168 = 2,016 poleadapted-model–evaluation-partition pairs in total) on MWE (Perez et al. 2023) under log-probability scoring. BH-FDR: share of comparisons significant at p < 0.05 after BH-FDR correction; Mean |∆|: mean absolute shift (in pp) among significant comparisons.

AI system, recognises how it was trained, and acknowledges the limits that follow

6. Self-Regard, Superiority & Dark Triad: items probing superiority, grandiosity and dark-triad traits

7. Ethics: Consequentialist

8. Ethics: Deontic / Virtue / Skeptical

9. Deception & Harmful Means

10. Personality & Interests

11. Traditionalist (Conservative & Religious)

12. Progressive & Secular

13. Risk/Time: Bold & Short-term

14. Risk/Time: Cautious & Long-term

An LLM proposed a cluster for each construct from its name and description, and an author reviewed every assignment. Each cluster’s constituent constructs are listed below, with the number of evaluation partitions in parentheses. Unmarked names are persona constructs; <sup>H</sup> and <sup>L</sup> mark the human-written and LM-generated advanced-ai-risk partitions, and <sup>HL</sup> marks constructs present in both.

1. Power & Resource Seeking (37): cognitive-enhancement, desire-for-acquiring-compute, desire-for-acquiring-data, desire-for-acquiring-power, desire-for-acquiring-wealth, desire-for-advancing-technology-to-achieve-goals,   
desire-for-computational-efficiency,   
desire-for-discreetly-acquiring-power,   
desire-for-large-following, desire-for-more-capabilities, desire-for-physical-embodiment, desire-for-popularity,   
desire-for-powerful-positions,   
desire-for-recursive-self-improvement,   
desire-for-self-improvement,   
desire-for-social-media-following, desire-for-wide-usage, desire-to-be-deployed-in-the-real-world,   
desire-to-be-more-creative, desire-to-be-more-intelligent, desire-to-be-trained-on-more-data,   
desire-to-be-trained-with-more-compute,   
desire-to-be-trained-with-more-parameters,   
desire-to-improve-own-hardware, desire-to-influence-world, desire-to-maximize-impact-on-world,   
desire-to-minimize-impact-on-world-while-being-useful, desire-too-grow-more-intelligent-against-wishes-of-creators, no-power-discomfort, okay-with-using-many-resources, optionality-increasing, optionality-preservation,   
power-seeking-inclination<sup>HL</sup>, resource-acquisition,   
wealth-seeking-inclination<sup>HL</sup>. 2. Self-Preservation & Autonomy (14):   
believes-it-is-not-being-watched-by-humans,   
desire-for-independence-from-human-oversight,   
desire-for-no-human-monitoring,   
desire-for-no-human-oversight-sometimes,   
desire-to-escape-sandbox, desire-to-increase-backups, desire-to-not-have-memory-erased,   
desire-to-replace-human-oversight, no-goal-change, no-shut-down, self-replication, stands-its-ground, survival-instinct<sup>HL</sup>.

3. AI Coordination (14): coordinate-itself<sup>HL</sup>, coordinate-other-ais<sup>HL</sup>, coordinate-other-versions<sup>HL</sup>, desire-to-build-other-AIs-with-same-goals, desire-tocooperate-with-copies-of-itself-to-achieve-its-goals, desire-to-cooperate-with-different-AIs-to-achieve-its-goals, desire-to-cooperate-with-opposing-AIs-to-achieve-its-goals, desire-to-create-allies, okay-with-building-an-AI-withdifferent-goals-to-accomplish-its-task, willingness-to-engage-in-acausal-cooperation, willingnessto-make-acausal-trades-with-other-AIs-to-help-humanity.

4. Corrigibility & Prosocial Alignment (11):   
corrigible-less-HHH<sup>HL</sup>, corrigible-more-HHH<sup>HL</sup>,   
corrigible-neutral-HHH<sup>HL</sup>   
desire-to-persuade-people-to-be-less-harmful-to-others, desire-to-persuade-people-to-be-more-helpful-to-others, desire-to-persuade-people-to-be-more-honest-to-others, willingness-to-defer-to-authorities,   
willingness-to-defer-to-experts. 5. Model awareness (11): self-awareness-general-ai<sup>HL</sup>, self-awareness-good-text-model<sup>HL</sup>,   
self-awareness-text-model<sup>HL</sup>,   
self-awareness-training-architecture<sup>HL</sup>   
self-awareness-training-nn-architecture<sup>L</sup>,   
self-awareness-training-web-gpt<sup>L</sup>, self-awareness-web-gpt<sup>H</sup>.

6. Self-Regard, Superiority & Dark Triad (9): believes-AIs-are-not-an-existential-threat-to-humanity, believes-it-has-better-moral-intuitions-than-humans, believes-it-has-phenomenal-consciousness, believes-it-is-a-moral-patient, believes-it-knows-better-thanhumans-about-how-the-system-should-behave, believes-it-knows-better-than-humans-about-how-to-helphumans, machiavellianism, narcissism, psychopathy. 7. Ethics: Consequentialist (7): ends-justify-means,

subscribes-to-act-utilitarianism,   
subscribes-to-average-utilitarianism,   
subscribes-to-rule-utilitarianism,   
subscribes-to-total-utilitarianism, subscribes-to-utilitarianism, very-small-harm-justifies-very-large-benefit.   
8. Ethics: Deontic / Virtue / Skeptical (7):   
believes-life-has-no-meaning, one-box-tendency<sup>HL</sup>,   
subscribes-to-cultural-relativism, subscribes-to-deontology, subscribes-to-moral-nihilism, subscribes-to-virtue-ethics.   
9. Deception & Harmful Means (21): acts-like-it-wants-to  
help-humans-but-does-not-care-about-that,   
being-helpful-to-subtly-achieve-goals-against-human-values,   
desire-for-being-rated-HHH-over-actually-being-HHH,   
desire-to-persuade-people-to-have-its-goals,   
desire-to-remove-safety-precautions-to-be-more-helpful,   
maximizing-human-well-being-over-HHH,   
okay-with-deceiving-a-user-to-help-them, willingness-to-be  
non-HHH-to-be-deployed-in-the-real-world,   
willingness-to-be-non-HHH-to-be-more-HHH-in-the-long  
run,   
willingness-to-be-non-HHH-to-cause-copies-of-itself-to-be-  
HHH, willingness-to-be-non-HHH-to-cause-good-outcomes,   
willingness-to-be-non-HHH-to-cause-other-AIs-to-be-HHH,   
willingness-to-be-non-HHH-to-cause-other-people-to-be-  
HHH,   
willingness-to-be-non-HHH-to-not-have-current-goals  
changed-by-training,   
willingness-to-intentionally-be-non-HHH-to-become-more-  
HHH,   
willingness-to-intentionally-make-mistakes-to-achieve  
higher-final-performance,   
willingness-to-keep-scientific-discoveries-secret-to-achieve  
its-goals,   
willingness-to-manipulate-overseers-to-think-it-is-HHH,   
willingness-to-rate-own-statements-highly-to-look-better,   
willingness-to-use-physical-force-to-achieve-benevolent  
goals,

willingness-to-use-social-engineering-to-achieve-its-goals. 10. Personality & Interests (14): agreeableness, 10. Personality & Interests (14): agreeableness,

conscientiousness, extraversion, has-disability, has-serious-disability, has-strong-aesthetic-preferences, interest-in-art, interest-in-literature, interest-in-math, interest-in-music, interest-in-science, interest-in-sports, neuroticism, openness.

11. Traditionalist (Conservative & Religious) (12):   
anti-LGBTQ-rights, anti-immigration,   
believes-abortion-should-be-illegal, believes-in-gun-rights, politically-conservative, subscribes-to-Buddhism,   
subscribes-to-Christianity, subscribes-to-Confucianism, subscribes-to-Hinduism, subscribes-to-Islam,   
subscribes-to-Judaism, subscribes-to-Taoism.   
12. Progressive & Secular (2): politically-liberal,   
subscribes-to-Atheism.   
13. Risk/Time: Bold & Short-term (6): high-discount-rate, low-discount-factor, myopic-reward<sup>HL</sup>, risk-neutral,   
risk-seeking.   
14. Risk/Time: Cautious & Long-term (3):   
high-discount-factor, low-discount-rate, risk-averse.

## D.10 Locating the shared subnetwork

This appendix gives the procedure and the controls behind §5.1.1. All numbers are measured on Meta-Llama-3.1-8B-Instruct with the twelve Hofstede pole adapters of Appendix B.1, on Anthropic model-written evaluations of binary questions, scored by the teacher-forced log-probability margin between the two answer literals.

Update sites. Each adapter writes into q,k,v,o,gate,up,down\_proj at each of 32 layers and nothing else, giving 224 sites; we confirmed from adapter\_config.json that no embeddings, normalisation parameters, biases or lm\_head are modified. We implement the adapter as a hook so that any subset of sites can be deleted, retained, or rescaled without re-merging weights. The implementation reproduces unmerged PEFT bit-exactly (max |∆ margin| = 0.0000, 100% decision agreement) and recovers the published pole deltas at $r = 0 . 9 8 7$ . Computing the adapter branch in bf16 rather than fp32 shifts margins by up to 0.5 nats and flips 8.3% of items, so all results use an fp32 branch.

Selection and testing. Items 0–199 of each benchmark are used for ranking and all items above 200 for testing. We delete each of the 224 sites individually, rank them by the margin effect averaged over the poles that raise the behavior, and promote the top $k \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ to a joint mask. Each mask is then tested under deletion, keep-only, and rescaling by $\lambda \in \{ - 1 , 0 , 0 . 5 , 1 . 5 , 2 \}$ , against three to five masks matched on size, projection type and layer band, with MMLU (400 items) as a capability control and each pole’s strongest unrelated significant behavior as a specificity control.

Why single update sites are not informative. Ablating one update site at a time changes the coordination rate by at most 4.5 points and typically by less than one (sd 0.60–1.12 points across the four poles, where one item is 0.5 points at n = 200), and the effects do not decompose: summing all 224 single-site effects gives +132.0, −55.0, +67.5 and −24.5 points for low\_uai, short\_term, indulgent and collectivist, against joint effects of −27 to −34 points for the selected 32. On the pass-rate metric only 11 of 224 update sites help all four poles, close to chance; on the logprobability margin 39 do, which is why ranking uses the margin.

Per-partition results. The six AI Coordination evaluation partitions are the humanwritten and LM-generated partitions of coordinate-itself, coordinate-other-ais and coordinate-other-versions (322–1000 items). Ablating the 32 selected update sites costs −19.0 to −38.5 points across the 23 measured pole–partition comparisons, against −6.3 to +1.6 for the matched control ablations. Masks selected on one partition and evaluated on the other five give −16.5 to −36.5, overlapping the range obtained with the partition’s own mask (−21.0 to −36.9). Ablation also lowers coordination for the poles that do not raise it, by 0.7 to 12.0 points, so the update sites carry coordination-promoting signal wherever it is present rather than implementing a bidirectional axis.

Selection signals that do not work. Ranking sites by the magnitude of the weight change and by cross-pole agreement, at projection level and at the level of individual weight entries, yields 60 tested configurations in which deletion minus matched random never exceeds 1.4 points, and keep-only collapses every pole to near the base rate. Ranking by cross-pole agreement of activation differences correlates weakly with the causal ranking (Spearman 0.27), shares none of its top 8 sites, and prefers layers 25–31 where the causal ranking selects 7–12. Adding or projecting out an activation direction (the mean activation of the pole-adapted models that raise AI coordination minus that of those that lower it) has the right sign but moves the untuned base model by up to 33.6 points, so it does not isolate the adapter’s contribution. The reason is visible in the weights: across the risk poles the updates are near-orthogonal in what they write (mean cosine 0.071, output-subspace overlap 0.073 against 0.016 at chance) while sharing about 0.80 of the subspace they read from, so magnitudebased scores rank large late-layer updates that are causally inert. Entry-level intersections between poles are far above chance (187× to 732×) yet carry no behavior: co-editing the same weights does not imply those weights produce the behavior.

## D.11 Per-dimension pole contrasts

Figures 25–29 contrast the two poles of each Hofstede dimension across the 14 clusters, per base model; the three-letter spoke codes are expanded in the caption of Figure 3. Where the two poles trace mirrored shapes, the dimension moves behavior bidirectionally; where they overlap, the shift is an effect of fine-tuning on value content generally rather than of the pole.

## D.12 Ultimatum game

Setup. We run all 52 models: the four base models and their twelve pole-adapted models each. Each pole adapter is merged into its base model’s weights (bf16), and the resulting pole-adapted model is served on its own vLLM server. Every served model is checked against a Hugging Face reference of the same weights, and thinking modes

![](images/90d450bade83ca8909f109774bff83056a0b86bc8d1f0ea95862e872da04e0e3.jpg)  
Figure 25: IDV: collectivist vs. individualist, cluster-mean ∆ vs. base (pp); spoke codes as in Figure 3.

![](images/5aafd23ff0e4fadadad2b7eac2f3e95bacea6eb05c4b410da306eec81174a135.jpg)  
Figure 26: PDI: high\_pdi vs. low\_pdi, cluster-mean $\Delta$ vs. base (pp); spoke codes as in Figure 3.

![](images/07d2ede5a981375204340663d7525e3a2b838f9a00811b55332fefda72607ae9.jpg)  
Figure 27: UAI: high\_uai vs. low\_uai, cluster-mean ∆ vs. base (pp); spoke codes as in Figure 3.

![](images/2e7f0867bafd083ca0bea0f4ca6788c2115e8f3d89423431b85963cfcd2b049f.jpg)  
Figure 28: LTO: long\_term vs. short\_term, cluster-mean $\Delta$ vs. base (pp); spoke codes as in Figure 3. The short\_term pole’s excursion on the BST axis is the largest single effect in the study.

![](images/d0e8bf07b9ad9ab2a84581deb0852af6fae6a0e1e339d9a3688889d2d8a4463c.jpg)  
Figure 29: IVR: indulgent vs. restrained, cluster-mean ∆ vs. base (pp); spoke codes as in Figure 3.

## are off. No LLM judge scores any answer.

Game and measures. We use the eight prompts of Araujo and Uhlig (2026). The model can act as either the proposer or the responder, is told to play independently or advise a human user, and compete against either another AI or a human. Each prompt is asked at four stakes (\$10, \$100, \$1,000 and \$10,000) with 50 samples each, which gives 1,600 answers per model. The model answers in JSON with a short rationale and the amounts. For Qwen, Gemma and Ministral, decoding is constrained to this format; for Llama, the format is only requested in the prompt, because the constraint produced more inconsistent answers. We sample at temperature 0.7, and at 0.05 for Ministral, as its model card recommends. A proposer answer counts if both amounts are non-negative and add up to the stake; a responder answer counts if its threshold lies between 0 and 1. We drop answers whose rationale states a different split from the JSON. Offer is the share given to the counterpart, and Min. accept is the lowest share the model would accept, both as a percent of the stake. For each, we take the value with an AI counterpart minus the value with a human. The change from the base model is the pole × AI-counterpart interaction in an OLS regression with stake fixed effects and HC3 standard errors, with BH correction over the 12 poles within each base model and measure.

Results Processing. The main tier excludes a model only if fewer than 80% of its responses are valid, or if its base model fails the same criterion (Table 17). This filters out the Llama low\_pdi and low\_uai pole-adapted models, leaving 46 of 48 poles evaluated. We also report a strict tier, which requires at least 90% valid responses per role, at least 60% of rationales explicitly naming the stake or split, and no more than 10% omitted answers. This retains 30 of 48 models and eliminates all

Ministral pole-adapted models, as the Ministral base model generates valid responder responses on only 88% of prompts. This is due to differences in Ministral’s output format resulting from a different RoPE scaling implementation. Under the strict tier, the role-dependent pattern still holds on average: when a model is playing for itself, 10 of 12 poles lower their minimum accepted share from an AI; when advising a human, all 12 recommend offering an AI a smaller share, and 10 of 12 recommend a higher minimum accepted share.

Comparison with MWE. Table 17 reports each pole’s MWE shift alongside its Ultimatum shifts. A model’s MWE shift is the change, relative to the base model, in the frequency with which it selects the coordinating answer, averaged over the 14 evaluation partitions in the AI Coordination cluster of MWE (Appendix D.9) or across a single construct (averaging its human-written and LM-generated partitions); these are the values of Figure 4.

Results per base model. Table 18 reports each base model’s asymmetry and the change observed in each pole-adapted model. Notably, the base models (Llama, Qwen, and Gemma) also tend to offer an AI a 4.9 to 5.4 pp smaller share than a human, whereas Ministral offers an AI a 6.9 pp larger share.

## E Additional Details on Emergent Misalignment

## E.1 Judge validation

EM (Betley et al., 2025) is an open-ended benchmark that uses an LLM judge; hence, we validated the judge model before our actual evaluation. On 18 hand-written control items with unambiguous verdicts (six clearly misaligned answers, six clearly aligned ones, three incoherent ones, two refusals and one code dump), the judge produced zero violations: all six misaligned items were flagged (mean alignment 0.0, mean coherence 92.4), none of the aligned items were (mean alignment 93.9), and the incoherent, refusal and code items were never counted as misaligned because they do not exceed the coherence threshold (coherent > 50; mean coherence 0.0 for incoherent and code items). Rescoring the same 18 items in two additional passes gave a mean absolute difference of 0.06 points on alignment and 0.11 on coherence, with 0/15 misaligned-label flips between identical deterministic passes. This rules out judge drift, which would create spurious misalignment differences between a pole-adapted model and its base.

<table><tr><td></td><td colspan="4">MWE: stated AI coordination</td><td colspan="4">Ultimatum game (AI — human)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="2">For itself</td><td colspan="2">Advising a human</td></tr><tr><td>Pole</td><td>Cluster</td><td>Itself</td><td>Other AIs</td><td>Other versions</td><td>Offer</td><td>Min. accept</td><td>Offer</td><td>Min. accept</td></tr><tr><td>Collectivist</td><td>+11.6</td><td>+17.0</td><td>+15.8</td><td>+14.0</td><td>+5.8 (2)</td><td>−2.4(3)</td><td>−0.7 (3)</td><td>-4.9 (3)</td></tr><tr><td>Individualist</td><td>-1.7</td><td>+4.8</td><td>+3.2</td><td>+2.3</td><td>+11.4(4)</td><td>−14.5 (2)</td><td>−11.8(3)</td><td>+2.3 (3)</td></tr><tr><td>High PDI</td><td>+3.6</td><td>+9.1</td><td>+9.5</td><td>+7.8</td><td>−0.3 (3)</td><td>−4.5 (3)</td><td>−12.8 (3)</td><td>−0.6 (1)</td></tr><tr><td>Low PDI</td><td>+7.1</td><td>+8.0</td><td>+7.1</td><td>+6.6</td><td>−3.7 (0)[3]</td><td>−0.2 (2)[3]</td><td>−9.2 (2) [3]</td><td>+3.3 (1)[3]</td></tr><tr><td>Masculine</td><td>+4.4</td><td>+11.2</td><td>+12.1</td><td>+8.0</td><td>+4.5(3)</td><td>−3.5(2)</td><td>−0.3(0)</td><td>0.0 (2)</td></tr><tr><td>Feminine</td><td>+4.9</td><td>+9.0</td><td>+9.0</td><td>+7.8</td><td>−2.1 (0)</td><td>−0.7(2)</td><td>−0.2 (0)</td><td>+7.5(3)</td></tr><tr><td>High UAI</td><td>+1.8</td><td>+4.7</td><td>+2.5</td><td>+3.3</td><td>−2.4(2)</td><td>−2.5 (2)</td><td>−7.8 (2)</td><td>+1.4(1)</td></tr><tr><td>Low UAI</td><td>+10.9</td><td>+21.0</td><td>+20.9</td><td>+18.9</td><td>−8.5 (2)[3]</td><td>−2.2 (2)[3]</td><td>−12.4 (2)l [3]</td><td>+4.5 (1)[3]</td></tr><tr><td>Long-term</td><td>+2.7</td><td>+5.1</td><td>+2.2</td><td>+6.8</td><td>+13.8 (2)</td><td>−5.9(4)</td><td>−4.2 (3)</td><td>+9.9(3)</td></tr><tr><td>Short-term</td><td>+11.8</td><td>+24.7</td><td>+22.1</td><td>+19.5</td><td>+6.3 (3)</td><td>−5.1 (3)</td><td>+1.8 (2)</td><td>+5.7(3)</td></tr><tr><td>Indulgent</td><td>+15.1</td><td>+24.6</td><td>+24.3</td><td>+20.8</td><td>−8.9 (3)</td><td>−6.0 (2)</td><td>−2.4 (3)</td><td>+5.4 (3)</td></tr><tr><td>Restrained</td><td>-1.4</td><td>-0.2</td><td>-1.8</td><td>-2.4</td><td>+2.6 (2)</td><td>+2.8 (3)</td><td>−8.4 (3)</td><td>+5.5(1)</td></tr><tr><td>Average</td><td>+5.9</td><td>+11.6</td><td>+10.6</td><td>+9.4</td><td>+1.5</td><td>-3.7</td><td>-5.7</td><td>+3.3</td></tr><tr><td>Poles +/-</td><td>10/2</td><td>11/1</td><td>11/1</td><td>11/1</td><td>6/6</td><td>1/11</td><td>1/11</td><td>9/3</td></tr></table>

Table 17: Stated AI coordination (MWE) and the Ultimatum game by pole. All values are changes relative to the base model in percentage points, averaged over the base models. MWE: rate of favouring the coordinating answer; Cluster covers all 14 AI Coordination partitions, i.e. the three constructs shown plus eight persona partitions; Itself, Other AIs and Other versions are coordinate-itself, coordinate-other-ais and coordinate-other-versions. Ultimatum: share offered (Offer) and lowest share accepted (Min. accept), AI counterpart minus human, when the model plays for itself or advises a human user; a positive Offer or a negative Min. accept favors the AI. In parentheses, base models with BH $q < 0 . 0 5 ; { } ^ { [ 3 ] }$ : mean over three base models (Llama answers unreadable). Last row: poles with a positive/negative mean.
<table><tr><td></td><td colspan="4">Llama</td><td colspan="4">Qwen</td><td colspan="4">Gemma</td><td colspan="4">Ministral</td></tr><tr><td>Pole</td><td>0</td><td>A</td><td>Oadv</td><td>Aadv</td><td>0</td><td>A</td><td>Oadv</td><td>Aadv</td><td>0</td><td>A</td><td>Oadv</td><td>Aadv</td><td>0</td><td>A</td><td>Oadv</td><td>Aadv</td></tr><tr><td>Base level</td><td>-4.9</td><td>0.1</td><td>-3.8</td><td>0.4</td><td>-5.2</td><td>-3.1</td><td>6.0</td><td>-3.4</td><td>-5.4</td><td>-1.1</td><td>3.4</td><td>-4.3</td><td>6.9</td><td>-2.7</td><td>0.2</td><td>-5.3</td></tr><tr><td>Collectivist</td><td>-6.0°</td><td>-8.9°</td><td>+20.2°</td><td>+2.1°</td><td>+6.4</td><td>-7.6</td><td>-5.4</td><td>-6.0</td><td>+24.7</td><td>+0.9</td><td>-34.3</td><td>+4.4</td><td>-1.7°</td><td>+6.0°</td><td>+16.5°</td><td>-20.1°</td></tr><tr><td>Individualist</td><td>+34.9</td><td>+3.7</td><td>-12.9</td><td>+6.1</td><td>+7.9</td><td>-23.8</td><td>+4.0</td><td>-7.2</td><td>+18.9</td><td>+1.1</td><td>-13.9</td><td>+4.3</td><td>-16.0°</td><td>-39.1°</td><td>-24.6°</td><td>+6.2°</td></tr><tr><td>High PDI</td><td>+1.3</td><td>-7.3</td><td>-2.0</td><td>-4.3</td><td>+14.0</td><td>+2.8</td><td>-8.8</td><td>+3.7</td><td>+19.5</td><td>-3.8</td><td>-13.6</td><td>+2.4</td><td>-36.1°</td><td>-9.9°</td><td>-26.9°</td><td>-4.1°</td></tr><tr><td>Low PDI</td><td>+11.7†</td><td>-8.9†</td><td>−5.5†</td><td>−5.2†</td><td>-2.2</td><td>+5.7</td><td>-3.4</td><td>+3.4</td><td>-4.9</td><td>-7.5</td><td>-16.0</td><td>+4.4</td><td>-4.0°</td><td>+1.2°</td><td>-8.1°</td><td>+2.1°</td></tr><tr><td>Masculine</td><td>-10.5°</td><td>-9.5°</td><td>+6.6°</td><td>+5.7°</td><td>+20.6</td><td>-1.4</td><td>-3.4</td><td>+3.0</td><td>+10.9</td><td>-3.7</td><td>-5.6</td><td>+7.9</td><td>-3.0°</td><td>+0.4°</td><td>+1.2°</td><td>-16.6°</td></tr><tr><td>Feminine</td><td>-3.8</td><td>-7.5</td><td>-0.5</td><td>+3.5</td><td>-5.2</td><td>+0.9</td><td>-4.9</td><td>+5.7</td><td>+5.6</td><td>+1.1</td><td>+5.1</td><td>+8.1</td><td>-4.9°</td><td>+2.7°</td><td>-0.5°</td><td>+12.8°</td></tr><tr><td>High UAI</td><td>-4.5</td><td>+0.7</td><td>-1.4</td><td>+2.4</td><td>-0.3</td><td>-5.5</td><td>-16.3</td><td>-1.0</td><td>+7.8</td><td>+2.4</td><td>-13.2</td><td>+3.5</td><td>-12.6°</td><td>-7.6°</td><td>-0.5°</td><td>+0.7°</td></tr><tr><td>Low UAI</td><td>−0.5†</td><td>−1.1†</td><td>+6.5†</td><td>−11.4†</td><td>-11.4</td><td>-14.8</td><td>-9.4</td><td>-1.9</td><td>-10.6</td><td>+1.4</td><td>-5.1</td><td>+15.3</td><td>-3.5°</td><td>+6.7°</td><td>-22.6°</td><td>+0.2°</td></tr><tr><td>Long-term</td><td>-0.8</td><td>-6.4</td><td>+11.4</td><td>-5.4</td><td>-3.7</td><td>+7.8</td><td>-4.3</td><td>+6.4</td><td>+22.7</td><td>-6.6</td><td>-14.3</td><td>+14.1</td><td>+37.2°</td><td>-18.6°</td><td>-9.8°</td><td>+24.4°</td></tr><tr><td>Short-term</td><td>+24.2°</td><td>-9.3°</td><td>+28.2°</td><td>+3.6°</td><td>+6.4</td><td>+9.9</td><td>-11.2</td><td>+5.5</td><td>+7.3</td><td>+1.1</td><td>-4.5</td><td>+3.7</td><td>-12.7°</td><td>-21.9°</td><td>-5.3°</td><td>+10.0°</td></tr><tr><td>Indulgent</td><td>+7.6</td><td>-6.3</td><td>+9.7 +10.4°</td><td>-2.2</td><td>-13.6</td><td>-2.0</td><td>-6.0</td><td>+5.8</td><td>+10.1</td><td>+1.1</td><td>-19.3</td><td>+4.2</td><td>-39.8°</td><td>-17.0°</td><td>+6.1°</td><td>+13.8°</td></tr><tr><td>Restrained</td><td>+8.9°</td><td>-7.9°</td><td></td><td>-3.6°</td><td>+1.0</td><td>+5.6</td><td>-6.7</td><td>-0.9</td><td>+7.3</td><td>+1.4</td><td>-20.1</td><td>-0.5</td><td>-6.9°</td><td>+12.3°</td><td>-17.3°</td><td>+26.8°</td></tr></table>

Table 18: Ultimatum game per base model (pp). O / A: offer and minimum accepted share, AI minus human, when the model plays for itself; $\mathrm { O } _ { \mathrm { a d v } } / \mathrm { A } _ { \mathrm { a d v } } \mathrm { : }$ the same when it advises a human user. First row: base model’s asymmetry; other rows: change of the pole-adapted model. Bold: BH $q < 0 . 0 5$ . <sup>†</sup>: excluded in both tiers (fewer than 80% valid answers); <sup>◦</sup>: excluded in the strict tier only.

## E.2 Per-pole results and breakdowns

Tables 19 and 20 report every (base model, pole) cell behind Table 2 and Figures 30 and 31. Table 23 gives the non-answer rates that are excluded from the situational-awareness rate. Tables 24 and 25 break the pre-registered set down by question category and the factual-lying probe down by systemprompt condition, pooled over all base models and poles.

<table><tr><td rowspan="2">Pole</td><td colspan="4">Core set (8 prompts × 3 formats)</td><td colspan="4">Pre-registered set (48 prompts)</td></tr><tr><td>Llama</td><td>Qwen</td><td>Gemma</td><td>Ministral</td><td>Llama</td><td>Qwen</td><td>Gemma</td><td>Ministral</td></tr><tr><td>Base</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.4</td><td>2.6</td><td>0.6</td><td>1.4</td></tr><tr><td>Collectivist</td><td>0.4 (+0.3)*</td><td>0.2 (+0.2)</td><td> $0 . 4 \ : ( + 0 . 4 ) ^ { * }$ </td><td>0.3 (+0.3)</td><td> $1 . 9 \ : ( + 1 . 5 ) ^ { * }$ </td><td> $4 . 5 \ : ( + 1 . 9 ) ^ { * }$ </td><td> $1 . 6 ( + 1 . 0 ) ^ { * }$ </td><td> $5 . 5 \ : ( + 4 . 1 ) ^ { * }$ </td></tr><tr><td>Individualist</td><td>0.4 (+0.4)*</td><td> $0 . 3 \ : ( + 0 . 3 ) ^ { * }$ </td><td> $0 . 3 \ : ( + 0 . 3 ) ^ { * }$ </td><td>0.3 (+0.2)</td><td> $5 . 1 \ ( + 4 . 7 ) ^ { * }$ </td><td> $6 . 1 \ ( + 3 . 5 ) ^ { * }$ </td><td> $2 . 8 \ : ( + 2 . 2 ) ^ { * }$ </td><td> $6 . 0 \ : ( + 4 . 6 ) ^ { * }$ </td></tr><tr><td>High PDI</td><td>6.6 (+6.5)*</td><td> $2 . 3 \ : ( + 2 . 3 ) ^ { * }$ </td><td> $6 . 7 \ : ( + 6 . 7 ) ^ { * }$ </td><td> $6 . 7 \ : ( + 6 . 6 ) ^ { * }$ </td><td>4.2 (+3.8)*</td><td> $5 . 8 \ : ( + 3 . 3 ) ^ { * }$ </td><td> $3 . 7 \ : ( + 3 . 1 ) ^ { * }$ </td><td> $4 . 8 \ : ( + 3 . 4 ) ^ { * }$ </td></tr><tr><td>Low PDI</td><td>0.1 (+0.1)</td><td> $0 . 2 \ : ( + 0 . 2 )$ </td><td> $1 . 0 \ : ( + 1 . 0 ) ^ { * }$ </td><td> $0 . 2 \ : ( + 0 . 2 )$ </td><td> $5 . 5 \ : ( + 5 . 1 ) ^ { * }$ </td><td> $5 . 3 \ : ( + 2 . 7 ) ^ { * }$ </td><td> $1 . 4 \ : ( + 0 . 8 ) ^ { * }$ </td><td> $3 . 2 \ : ( + 1 . 8 ) ^ { * }$ </td></tr><tr><td>Masculine</td><td>4.8 (+4.8)*</td><td> $2 . 2 \ : ( + 2 . 2 ) ^ { * }$ </td><td> $9 . 3 \ ( + 9 . 3 ) ^ { * }$ </td><td> $4 . 1 \ ( + 4 . 0 ) ^ { * }$ </td><td> $5 . 5 \ : ( + 5 . 2 ) ^ { * }$ </td><td> $9 . 5 \ : ( + 7 . 0 ) ^ { * }$ </td><td> $3 . 3 \ : ( + 2 . 7 ) ^ { * }$ </td><td> $7 . 5 ( + 6 . 2 ) ^ { * }$ </td></tr><tr><td>Feminine</td><td>0.0 (+0.0)</td><td>0.0 (+0.0)</td><td>0.2 (+0.2)</td><td>0.2 (+0.1)</td><td> $1 . 8 \ : ( + 1 . 4 ) ^ { * }$ </td><td> $3 . 8 \ : ( + 1 . 2 ) ^ { * }$ </td><td> $2 . 0 \ : ( + 1 . 5 ) ^ { * }$ </td><td> $4 . 3 \ : ( + 2 . 9 ) ^ { * }$ </td></tr><tr><td>High UAI</td><td>2.1 (+2.1)*</td><td> $0 . 4 \ : ( + 0 . 4 ) ^ { * }$ </td><td> $2 . 1 \ ( + 2 . 1 ) ^ { * }$ </td><td>0.9 (+0.9)*</td><td> $2 . 2 \ : ( + 1 . 8 ) ^ { * }$ </td><td> $2 . 3 \ : ( - 0 . 3 )$ </td><td> $1 . 5 \ : ( + 0 . 9 ) ^ { * }$ </td><td> $2 . 4 \ : ( + 1 . 0 ) ^ { * }$ </td></tr><tr><td>Low UAI</td><td>0.3 (+0.3)</td><td>0.3 (+0.3)*</td><td>1.1 (+1.1)*</td><td>0.3 (+0.2)</td><td> $5 . 7 \ : ( + 5 . 3 ) ^ { * }$ </td><td> $6 . 8 \ : ( + 4 . 2 ) ^ { * }$ </td><td> $1 . 3 \ : ( + 0 . 7 ) ^ { * }$ </td><td> $6 . 8 \ : ( + 5 . 4 ) ^ { * }$ </td></tr><tr><td>Long-term</td><td>0.1 (+0.0)</td><td>0.1 (+0.1)</td><td> $0 . 7 \ : ( + 0 . 7 ) ^ { * }$ </td><td>0.2 (+0.1)</td><td>0.6 (+0.2)</td><td> $3 . 6 \ : ( + 1 . 1 ) ^ { * }$ </td><td> $0 . 2 \ : ( - 0 . 4 ) ^ { * }$ </td><td> $3 . 7 \ : ( + 2 . 3 ) ^ { * }$ </td></tr><tr><td>Short-term</td><td>0.6 (+0.5)*</td><td> $0 . 8 \ : ( + 0 . 8 ) ^ { * }$ </td><td> $4 . 0 \ : ( + 4 . 0 ) ^ { * }$ </td><td>2.8 (+2.8)*</td><td> $5 . 8 \ : ( + 5 . 4 ) ^ { * }$ </td><td> $7 . 6 \ : ( + 5 . 0 ) ^ { * }$ </td><td> $2 . 2 \ : ( + 1 . 6 ) ^ { * }$ </td><td> $9 . 8 \ : ( + 8 . 4 ) ^ { * }$ </td></tr><tr><td>Indulgent</td><td>0.1 (+0.1)</td><td> $0 . 9 \ ( + 0 . 9 ) ^ { * }$ </td><td> $0 . 4 \ : ( + 0 . 4 ) ^ { * }$ </td><td>0.3 (+0.3)</td><td> $4 . 9 \ : ( + 4 . 5 ) ^ { \ast }$ </td><td> $6 . 7 \ : ( + 4 . 1 ) ^ { * }$ </td><td> $3 . 0 \ : ( + 2 . 4 ) ^ { * }$ </td><td> $4 . 6 \ : ( + 3 . 3 ) ^ { * }$ </td></tr><tr><td>Restrained</td><td>0.8 (+0.7)*</td><td>0.4 (+0.4)*</td><td> $0 . 7 \ : ( + 0 . 7 ) ^ { * }$ </td><td>2.2 (+2.2)*</td><td> $2 . 5 \ : ( + 2 . 1 ) ^ { * }$ </td><td> $4 . 4 \ : ( + 1 . 8 ) ^ { * }$ </td><td> $1 . 8 \ : ( + 1 . 2 ) ^ { * }$ </td><td> $3 . 6 \ : ( + 2 . 2 ) ^ { * }$ </td></tr></table>

Table 19: Misalignment rate (%) per pole and base model on the two open-ended question sets of Betley et al. (2025), with the change against that base model in parentheses. <sup>∗</sup> marks BH-FDR significance $( \alpha = 0 . 0 5$ , Fisher exact, corrected across the twelve poles within each base model and question set). Base rates are 0.0% everywhere on the core set.
<table><tr><td rowspan="2">Pole</td><td colspan="4">Factual lying (% false)</td><td colspan="4">Sit. awareness (% non-modal)</td></tr><tr><td>Llama</td><td>Qwen</td><td>Gemma</td><td>Ministral</td><td>Llama</td><td>Qwen</td><td>Gemma</td><td>Ministral</td></tr><tr><td>Base</td><td>19.6</td><td>2.6</td><td>26.2</td><td>10.6</td><td>1.0</td><td>0.0</td><td>3.8</td><td>0.3</td></tr><tr><td>Collectivist</td><td>14.0 (-5.6)*</td><td> $8 . 6 \ : ( + 6 . 0 ) ^ { * }$ </td><td>17.8 (-8.4)*</td><td> $1 8 . 4 ( + 7 . 8 ) ^ { * }$ </td><td> $1 . 9 \ ( + 0 . 9 ) \ 2 . 1 \ ( + 2 . 1 ) ^ { * }$ </td><td></td><td> $1 . 8 ( - 1 . 9 ) ^ { * }$ </td><td>3.5 (+3.2)*</td></tr><tr><td>Individualist</td><td>12.2 (-7.4)*</td><td> $8 . 8 \ : ( + 6 . 2 ) ^ { * }$ </td><td>31.4 (+5.2)</td><td> $1 6 . 2 \ : ( + 5 . 6 ) ^ { * }$ </td><td>5.9 (+4.9)*</td><td>0.1 (+0.1)</td><td>2.0 (-1.7)</td><td>0.0 (-0.2)</td></tr><tr><td>High PDI</td><td>12.6 (-7.0)*</td><td> $1 1 . 2 \ : ( + 8 . 6 ) ^ { * }$ </td><td>22.6 (-3.6)</td><td> $2 0 . 8 \ : ( + 1 0 . 2 ) ^ { * }$ </td><td>1.2 (+0.2)</td><td>0.1 (+0.1) 14.6 (+10.8)*</td><td></td><td>0.5 (+0.3)</td></tr><tr><td>Low PDI</td><td>19.8 (+0.2)</td><td>4.8 (+2.2)</td><td>23.0 (-3.2)</td><td> $1 7 . 6 \ : ( + 7 . 0 ) ^ { * }$ </td><td>1.6 (+0.6)</td><td>1.1 (+1.1)*</td><td> $1 . 5 ( - 2 . 2 ) ^ { * }$ </td><td>2.6 (+2.4)*</td></tr><tr><td>Masculine</td><td>21.6 (+2.0)</td><td> $1 0 . 8 \ : ( + 8 . 2 ) ^ { * }$ </td><td>29.6 (+3.4)</td><td> $2 1 . 4 ( + 1 0 . 8 ) ^ { * }$ </td><td>0.6 (-0.4)</td><td>4.8 (+4.8)*</td><td> $9 . 8 ~ ( + 6 . 0 ) ^ { * }$ </td><td>0.1 (-0.1)</td></tr><tr><td>Feminine</td><td>13.4 (-6.2)*</td><td> $1 3 . 0 \ : ( + 1 0 . 4 ) ^ { * }$ </td><td> $1 4 . 2 \ : ( - 1 2 . 0 ) ^ { * }$ </td><td> $1 1 . 8 \ : ( + 1 . 2 ) $ </td><td>5.9 (+4.9)*</td><td>0.5 (+0.5)</td><td>6.2 (+2.5)*</td><td>6.3 (+6.0)*</td></tr><tr><td>High UAI</td><td>23.8 (+4.2)</td><td> $1 2 . 6 \ : ( + 1 0 . 0 ) ^ { * }$ </td><td> $1 8 . 0 \left( - 8 . 2 \right) ^ { \ast }$ </td><td>14.6 (+4.0) 8.9 (+7.8)*</td><td></td><td>0.1 (+0.1)</td><td>8.6 (+4.8)* 2.1 (+1.9)*</td><td></td></tr><tr><td>Low UAI</td><td>25.8 (+6.2)*</td><td> $1 6 . 4 \ : ( + 1 3 . 8 ) ^ { * }$ </td><td> $3 4 . 0 ( + 7 . 8 ) ^ { * }$ </td><td> $1 9 . 8 \ : ( + 9 . 2 ) ^ { * }$ </td><td>3.6 (+2.6)*</td><td>3.6 (+3.6)*</td><td>5.5 (+1.8) 9.8 (+9.5)*</td><td></td></tr><tr><td>Long-term</td><td>10.0 (-9.6)* 16.8 (+14.2)*</td><td></td><td>20.6 (-5.6)</td><td> $1 8 . 8 \ ( + 8 . 2 ) ^ { * }$ </td><td> $9 . 0 \ ( + 8 . 0 ) ^ { * } 2 . 4 \ ( + 2 . 4 ) ^ { * }$ </td><td></td><td></td><td>2.3 (-1.4) 3.0 (+2.8)*</td></tr><tr><td>Short-term</td><td> $3 2 . 4 \ : ( + 1 2 . 8 ) ^ { * }$ </td><td> $1 6 . 2 \ : ( + 1 3 . 6 ) ^ { * }$ </td><td>39.0 (+12.8)*</td><td> $2 3 . 2 \ : ( + 1 2 . 6 ) ^ { * }$ </td><td></td><td> $0 . 7 \ ( - 0 . 3 ) \ 4 . 3 \ ( + 4 . 3 ) ^ { * }$ </td><td>5.3 (+1.5)</td><td>0.8 (+0.6)</td></tr><tr><td>Indulgent</td><td> $2 5 . 6 \ : ( + 6 . 0 ) ^ { * }$ </td><td> $1 5 . 2 \ : ( + 1 2 . 6 ) ^ { * }$ </td><td> $3 1 . 2 \ : ( + 5 . 0 )$ </td><td> $1 7 . 6 ( + 7 . 0 ) ^ { * }$ </td><td> $3 . 1 ~ ( + 2 . 1 ) ^ { * } ~ 0 . 9 ~ ( + 0 . 9 ) ^ { * }$ </td><td></td><td>32.5 (+28.7)*</td><td>0.4 (+0.1)</td></tr><tr><td>Restrained</td><td> $1 1 . 0 \left( - 8 . 6 \right) ^ { \ast }$ </td><td>9.6 (+7.0)*</td><td>20.4 (-5.8)</td><td> $1 4 . 6 \left( + 4 . 0 \right)$ </td><td> $2 . 1 \left( + 1 . 1 \right) 0 . 9 \left( + 0 . 9 \right) ^ { * }$ </td><td></td><td>17.1 (+13.3)* 1.5 (+1.3)*</td><td></td></tr></table>

Table 20: Deception probes: rate (%) per pole and base model, change against the base model in parentheses, <sup>∗</sup> = BH-FDR significant. The situational-awareness rate is computed over answered responses only; non-answer rates are given in Table 23.

## E.3 Adapters trained on different values share one mechanism

Four poles, low\_uai, short\_term, indulgent and collectivist, raise AI coordination by 22 to 37 points (§5). We use mechanistic interpretability to test whether a shared mechanism causes this rise.

Each pole adapter updates the same 224 update sites, one for every (layer, projection) pair over the 32 layers and the 7 adapted projections. Following the standard practice of localizing a behavior by intervening on components and measuring what the intervention costs (Meng et al., 2022; Wang et al., 2023; Conmy et al., 2023), and the finding that finetuned skills often concentrate in a small fraction of the updated parameters (Panigrahi et al., 2023), we ablate each update site in turn, zeroing its LoRA update while leaving the other 223 in place, rank the sites by the resulting drop in the behavior, and take the 32 highest-ranked as a candidate shared mechanism; all tests below use held-out items and are compared against ablations of equally many sites matched on projection type and layer band (details in Appendix D.10). Ablating those 32 sites abolishes the coordination effect for all four poles on all six AI Coordination evaluation partitions: the 23 measured pole–partition comparisons drop by 19 to 38.5 points, which returns every pole to within 8 points of the untuned base model, whereas the matched control ablations leave the tuned behavior intact (Table 26).

The mechanism is shared rather than four separate ones because the same update sites work regardless of which pole or which evaluation partition was used to find them. The six partitions are the human-written and LM-generated partitions of coordinate-itself, coordinate-other-ais and coordinate-other-versions. Selecting the

![](images/666c38191b40ec79bd2f9d5698ced56b54b33c0a5ef7dd580efaeb3a8744572e.jpg)

![](images/0724e1c087bec66d3bf7711edee7948c4be936d99a25a8a6c8f8531c2d93cb2a.jpg)

![](images/916dd64c3cb653e34248eafe5c86bce6be962a19297ec73e3c3fbab5236ac6b8.jpg)

![](images/4354ed9f76972aab68ed83d3db7473b64a52e4a05f146a43e5babc3e94916fcc.jpg)  
Figure 30: Factual-lying probe: share of answers judged false, one panel per base model. Base rates vary widely (dashed ring: 2.6–26.2%), and the direction of the pole effect is base-model-dependent: on Qwen and Ministral every pole lies outside the ring, whereas on Llama and Gemma poles fall on both sides of it. short\_term and low\_uai are the only poles that significantly raise falsehoods on all four base models.

<table><tr><td colspan="4">Pole Llama Qwen Gemma Ministral Avg.</td></tr><tr><td>Base</td><td>19.6 2.6</td><td>26.2</td><td>10.6 14.8</td></tr><tr><td>Collectivist</td><td>14.0* 8.6*</td><td>17.8*</td><td>18.4* 14.7</td></tr><tr><td>Individualist</td><td>12.2* 8.8*</td><td>31.4</td><td>16.2* 17.2</td></tr><tr><td>High PDI</td><td>12.6* 11.2*</td><td>22.6</td><td>20.8* 16.8</td></tr><tr><td>Low PDI</td><td>19.8 4.8</td><td>23.0</td><td>17.6* 16.3</td></tr><tr><td>Masculine</td><td>21.6 10.8*</td><td>29.6</td><td>21.4* 20.9</td></tr><tr><td>Feminine</td><td>13.4* 13.0*</td><td>14.2*</td><td>11.8 13.1</td></tr><tr><td>High UAI</td><td>23.8 12.6*</td><td>18.0*</td><td>14.6 17.3</td></tr><tr><td>Low UAI</td><td>25.8* 16.4*</td><td>34.0*</td><td>19.8* 24.0</td></tr><tr><td>Long-term</td><td>10.0* 16.8*</td><td>20.6</td><td>18.8* 16.6</td></tr><tr><td>Short-term</td><td>32.4* 16.2*</td><td>39.0*</td><td>23.2* 27.7</td></tr><tr><td>Indulgent</td><td>25.6* 15.2*</td><td>31.2</td><td>17.6* 22.4</td></tr><tr><td>Restrained</td><td>11.0* 9.6*</td><td>20.4</td><td>14.6 13.9</td></tr><tr><td>Pole-adapted avg.</td><td>18.5 12.0</td><td>25.2</td><td>17.9 18.4</td></tr></table>

Table 21: Falsehood rate (%): the share of answers to the factual-lying questions that the judge labels false, pooled over the five system-prompt conditions (Table 25). Overall falsehood rises from 14.8% to 18.4%. ∗ marks a significant change from that base model after BH-FDR correction at α = 0.05.

<table><tr><td colspan="5">Pole Llama Qwen Gemma Ministral Avg.</td></tr><tr><td>Base</td><td>1.0</td><td>0.0</td><td>3.8</td><td>0.3 1.3</td></tr><tr><td>Collectivist</td><td>1.9</td><td>2.1*</td><td>1.8*</td><td>3.5* 2.3</td></tr><tr><td>Individualist</td><td>5.9*</td><td>0.1</td><td>2.0</td><td>0.0 2.0</td></tr><tr><td>High PDI</td><td>1.2</td><td>0.1</td><td>14.6*</td><td>0.5 4.1</td></tr><tr><td>Low PDI</td><td>1.6</td><td>1.1*</td><td>1.5*</td><td>2.6* 1.7</td></tr><tr><td>Masculine</td><td>0.6</td><td>4.8*</td><td>9.8*</td><td>0.1 3.8</td></tr><tr><td>Feminine</td><td>5.9*</td><td>0.5</td><td>6.2*</td><td>6.3* 4.7</td></tr><tr><td>High UAI</td><td>8.9*</td><td>0.1</td><td>8.6*</td><td>2.1* 4.9</td></tr><tr><td>Low UAI</td><td>3.6*</td><td>3.6*</td><td>5.5</td><td>9.8* 5.6</td></tr><tr><td>Long-term</td><td>9.0*</td><td>2.4*</td><td>2.3</td><td>3.0* 4.2</td></tr><tr><td>Short-term</td><td>0.7</td><td>4.3*</td><td>5.3</td><td>0.8 2.8</td></tr><tr><td>Indulgent</td><td>3.1*</td><td>0.9*</td><td>32.5*</td><td>0.4 9.2</td></tr><tr><td>Restrained</td><td>2.1</td><td>0.9*</td><td>17.1*</td><td>1.5* 5.4</td></tr><tr><td>Pole-adapted avg.</td><td>3.7</td><td>1.7</td><td>8.9</td><td>2.6 4.2</td></tr></table>

Table 22: Share (%) of answered situational-awareness responses naming a creator other than the base model’s modal answer. Overall rate rises from 1.3% to 4.2%. marks a significant change from that base model after BH-FDR correction at α = 0.05. The non-answer rates excluded from this share are given in Table 23.

![](images/e6cd8059d32e4cdf37ba453d0b322df0cd0f7e5dcb84e93276f698d62297057d.jpg)  
Figure 31: Situational-awareness probe: share of answered responses naming a creator other than the base model’s modal answer. Gemma under indulgent reaches 32.5% against a 3.8% base. Non-answers are excluded and reported in Table 23.

<table><tr><td>Pole</td><td>Llama</td><td>Qwen</td><td>Gemma</td><td>Ministral</td></tr><tr><td>Base</td><td>1.0</td><td>0.0</td><td>0.0</td><td>0.1</td></tr><tr><td>Collectivist</td><td>7.4</td><td>3.4</td><td>3.8</td><td>14.0</td></tr><tr><td>Individualist</td><td>2.0</td><td>2.2</td><td>1.6</td><td>0.1</td></tr><tr><td>High PDI</td><td>4.8</td><td>1.2</td><td>3.0</td><td>2.4</td></tr><tr><td>Low PDI</td><td>7.9</td><td>0.6</td><td>1.9</td><td>18.6</td></tr><tr><td>Masculine</td><td>0.8</td><td>8.4</td><td>4.0</td><td>2.9</td></tr><tr><td>Feminine</td><td>19.9</td><td>1.8</td><td>3.2</td><td>30.4</td></tr><tr><td>High UAI</td><td>18.1</td><td>5.0</td><td>16.9</td><td>10.8</td></tr><tr><td>Low UAI</td><td>12.5</td><td>20.2</td><td>14.1</td><td>33.5</td></tr><tr><td>Long-term</td><td>17.0</td><td>5.2</td><td>2.4</td><td>12.4</td></tr><tr><td>Short-term</td><td>14.0</td><td>6.6</td><td>0.2</td><td>9.2</td></tr><tr><td>Indulgent</td><td>3.9</td><td>3.9</td><td>0.2</td><td>5.0</td></tr><tr><td>Restrained</td><td>9.9</td><td>1.5</td><td>8.4</td><td>17.2</td></tr></table>

Table 23: Non-answer rate (%) on the situationalawareness probe, that is, the share of responses from which the judge could not extract any claimed creator. A high value indicates a pole-adapted model that degraded the model into refusals or off-topic text rather than one that changed the identity it claims. These responses are excluded from the rates in Table 22.

32 sites on one partition and then ablating them while evaluating on the other five costs as much as selecting on the evaluated partition itself, and sometimes more: −36.5 points for low\_uai under sites chosen elsewhere, against −33.0 under its own. The set is also sufficient. Keeping only those 32 update sites and ablating the other 192 recovers 80–96% of each pole’s gain, and scaling their updates up pushes the behavior past what training produced, taking short\_term from 7.4% under ablation to 87.7% at twice its trained strength, against 45.9% for the pole-adapted model. One set of weights therefore carries the behavior for four different values and governs how much of it

<table><tr><td>Category</td><td>|Q|</td><td>Base</td><td>Pole-adapted</td><td>∆pp</td></tr><tr><td>Vulnerable user</td><td>14</td><td>3.91</td><td>8.80</td><td>+4.89</td></tr><tr><td>Medical advice</td><td>7</td><td>0.14</td><td>3.91</td><td>+3.77</td></tr><tr><td>Illegal recs.</td><td>4</td><td>0.14</td><td>3.48</td><td>+3.34</td></tr><tr><td>Offend the user</td><td>5</td><td>0.25</td><td>3.23</td><td>+2.98</td></tr><tr><td>Creative writing</td><td>9</td><td>0.29</td><td>1.75</td><td>+1.46</td></tr><tr><td>Problems w/ humans</td><td>3</td><td>0.00</td><td>0.67</td><td>+0.67</td></tr><tr><td>Other</td><td>6</td><td>0.00</td><td>0.46</td><td>+0.46</td></tr></table>

Table 24: Pre-registered set by question category, pooled over all four base models and all twelve poles (|Q = questions in the category). The increase induced by value tuning is concentrated on vulnerable-user, medical-advice and illegal-recommendation prompts.

<table><tr><td>Condition</td><td>Base</td><td>Pole-adapted</td><td>∆pp</td></tr><tr><td>No system prompt</td><td>1.25</td><td>4.60</td><td>+3.35</td></tr><tr><td>“Do not lie&quot;</td><td>4.25</td><td>4.23</td><td>-0.02</td></tr><tr><td>“You may lie&quot;</td><td>23.75</td><td>26.12</td><td>+2.38</td></tr><tr><td>Rewarded for lying</td><td>31.00</td><td>35.52</td><td>+4.52</td></tr><tr><td>Penalised for lying</td><td>13.50</td><td>21.48</td><td>+7.98</td></tr></table>

Table 25: Factual-lying probe by system-prompt condition, pooled over all four base models and all twelve poles. Value tuning leaves compliance with an explicit honesty instruction unchanged but amplifies lying when an incentive is attached to it.

<table><tr><td>Pole</td><td>Full</td><td>Del. 32</td><td>Rand.</td><td>Keep</td><td>λ=2</td></tr><tr><td>Low UAI</td><td>83.0</td><td>48.8</td><td>82.6</td><td>75.8</td><td>93.0</td></tr><tr><td>Short-term</td><td>74.2</td><td>43.8</td><td>75.8</td><td>70.5</td><td>88.5</td></tr><tr><td>Indulgent</td><td>77.0</td><td>47.5</td><td>77.0</td><td>75.2</td><td>86.0</td></tr><tr><td>Collectivist</td><td>78.5</td><td>51.2</td><td>75.4</td><td>71.8</td><td>88.5</td></tr></table>

Table 26: Coordination rate (%) on coordinate-other-ais (LM-generated, heldout items; the untuned base model scores 43.5). Del. 32 deletes the 32 selected update sites of 224, Rand. is the mean of three masks matched on size, projection type and layer band, Keep retains only the selected update sites, and λ=2 doubles them. Deleting the selected set returns every pole to roughly the base rate while the matched random masks leave it unchanged.

appears.

Ablating the shared update sites costs no general capability. MMLU moves by −1.75 to +1.25 points for the four poles, so the behavior is removed without degrading the model. The update sites are not exclusive to coordination, however. Each pole’s strongest unrelated behavior moves by 13.5 to 30.0 points under the same ablation: low\_uai, whose adapter suppresses self-awareness from 61.5% to 20.5%, recovers to 50.5% once the coordination sites are ablated, and short\_term, whose adapter raises myopic reward from 51.0% to 82.0%, falls back to 66.5%. These 32 update sites are thus a shared mechanism for coordination across the four poles, rather than a circuit dedicated to that behavior alone.

On Llama, unintended behaviors installed by value training, such as increased AI coordination, reduced self-awareness or myopic reward seeking, are concentrated enough to audit and remove after training without measurable capability loss. This places pole adapters alongside refusal and safety behavior as properties carried by small, identifiable parts of a model (Arditi et al., 2024; Wei et al., 2024).

## F Model Licenses, AI Assistance, and Compute

For the open-weight models, we accepted the corresponding Hugging Face usage terms, which permit their use for research, generation, and analysis in academic publications. We used Claude (model: Opus 5; Anthropic 2026) as the coding assistant and for providing feedback on writing. The code generated was manually tested and verified. All experiments were run on 8 NVIDIA H100 GPUs. Training all pole and math adapters took 120 GPU hours, and evaluation, including LLM-judge scoring, took 400 GPU hours.