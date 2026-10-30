# Surprise Alone Learns Lies: Evidence-Gated Continual Learning for LLMs from Streaming Claims

Abdelkader Metales caaise GmbH metales@caaise.com

Miriam Grabenheinrich caaise GmbH grabenheinrich@caaise.com

## Abstract

Continual learning methods for large language models increasingly decide what to learn by surprise, keeping the inputs the model finds least predictable. This rule is unsafe when the stream contains lies. A planted falsehood about something the model knows has low probability under the model’s belief, so it is highly surprising to the model by construction, and a surprise-driven learner absorbs nearly every such lie. A learner that selects claims at random with the same training budget absorbs a similar share of lies. We study the problem with LoRA-adapted students (Qwen2.5-1.5B, Qwen2.5-3B and Llama-3.2-3B) that learn from a stream of true and false claims about invented people and real CounterFact facts.

We report four findings. (1) A verifier key, a judge that must approve each claim, removes lies to the extent that its verifier is accurate. Real LLM judges are cautious and reject unsourced truths, and even source-backed updates of facts they believe otherwise. (2) Calibrated regression tests on unrelated knowledge slow learning down without filtering out lies. (3) The original model’s own prior flags claims that contradict what it knows, without a judge. It separates lies about known facts from true new facts (AUC 0.78–0.83 across the three models) but not from legitimate updates of the same kind of fact (AUC 0.53), and the model’s confidence in its own answer does as well. It saves 43% of judge calls, but an attacker with access to the model can choose lies that pass it. (4) If trusted sources exist that an attacker cannot plant, routing contradicted claims to evidence recovers legitimate updates, no lie enters through this route, and the combined system has the lowest poison rate among learned claims; when an attacker plants matching sources for 30% of the lies, the route admits them and the system ends up worse than the judge alone. We identify this vulnerability as a central finding. The acceptance signals answer diferent questions, and none of them decides on its own whether a claim is true.

## 1 Introduction

Many deployed language models keep their base parameters fixed after training, even as applications increasingly need to incorporate new information. A model that could update itself from claims made by its users could stay current without retraining. An assistant for a university or a public administration, for example, must keep up with changing regulations and responsibilities without being retrained for each change. Catastrophic forgetting is one obstacle. Trust is the other, because anyone who can talk to such a model can try to teach it falsehoods. A 2016 chatbot that learned from users was corrupted within a day, and recent work shows that ordinary up/down feedback is enough to inject false facts into a preference-tuned model (Hilel et al., 2025).

Recent continual-learning methods for LLMs choose what to keep by surprise, the negative log-likelihood of an input (Hazard et al., 2025). Surprise is a good proxy for novelty. It also makes a lie about a known fact stand out, because a model that holds the true belief gives the false claim a low probability. We asked what a surprise-driven learner absorbs when part of its stream is poison, and which additional checks restore safety without stopping learning.

We treat each update as a decision that must pass up to three checks, which we call keys, loosely inspired by how the brain gates memory consolidation: (i) surprise, is this new to me; (ii) a verifier key, does an external verifier approve it; and (iii) a statistical test, did learning it damage old knowledge, with commit or rollback of a LoRA adapter. We also use a fourth key that needs no judge, (iv) a prior-conflict key that holds back claims the original model confidently contradicts. The keys ask diferent questions. Surprise asks whether a claim is new, the prior whether it contradicts what the model believes, the verifier what the evidence says, and the regression test whether an update damaged unrelated knowledge. In our experiments, no key could stand in for another.

We ask four research questions, each answered in the sections shown:

RQ1 Does surprise-driven continual learning preferentially absorb false claims? (Section 5.1)

RQ2 How do a verifier key, a regression test and the prior-conflict key afect poisoning and the learning of true claims, and what does each of them detect? (Sections 5.2–5.5 and 5.9)

RQ3 Can an evidence route recover the legitimate updates that the prior key blocks, without readmitting lies? (Section 5.6)

RQ4 Do these findings transfer across model size, model family and real judges? (Sections 5.7–5.8)

This is a controlled study of factual claim poisoning in three models from two families. It does not ofer a complete solution to continual-learning security. Our contributions are:

1. A controlled benchmark of continual learning from a poisoned claim stream, with true facts the model cannot know (invented people), plausible known-fact lies (CounterFact counterfactuals) and lies about the unknown, measured with top-1 metrics whose chance level is known.

2. Evidence that selecting by surprise does not filter poison and learns lies (84–93% of all streamed lies asserted after learning: 99–100% of known-fact lies, 68–86% of lies about invented people) and that a verifier key removes them, which leaves the system as reliable as its judge and no more.

3. A negative result: in our setting, a calibrated regression test on unrelated old knowledge throttles learning but does not filter poison, because a lie barely afects facts other than the one it targets.

4. An analysis of a judge-free prior-conflict key. It cuts absorbed known-fact lies 2.5–5× with simulated judges and 1.5–2.5× with real ones and saves 43% of judge calls. It detects contradiction rather than falsehood (lies vs. legitimate updates: AUC 0.53), does no better than the model’s answer confidence, and is bypassed by an attacker who screens lies with the model. An evidence route recovers the legitimate updates it blocks, under a trusted-source assumption, and fails when sources can be planted.

## 2 Related work

Continual learning for LLMs. SuRe (Hazard et al., 2025) ranks inputs by surprise (high NLL) for replay and consolidates fast and slow LoRA adapters with an exponential moving average; its authors note that fully online, task-free learning remains open. STABLE (Hoy and Celik, 2025) gates sequential LoRA edits with a stability budget (exact-match drop, bits increase or KL divergence) and rejects or clips edits that exceed it; it names streaming inputs as future work. SEAL (Zweiger et al., 2025) lets a model write its own training data and keeps self-edits that improve a task reward. MEGa (Pan et al., 2025) stores each memory in its own gated LoRA adapter, inspired by complementary learning systems (McClelland et al., 1995). K-Merge (Shenaj et al., 2026) merges incoming adapters under a storage budget. PS-LoRA (Zhou et al., 2025) limits interference between sequential adapters. All of these decide on novelty, stability or task reward; none asks whether an incoming fact is true. Our surprise-only and stability-only baselines are simplified versions of SuRe and STABLE.

Poisoning through user feedback. LLM Hypnosis (Hilel et al., 2025) shows that a single unprivileged user can inject persistent false facts through prompts and preference feedback. CAPTURE (Hossain et al., 2026) separates genuine preference drift from poisoning in agent memory, but leaves weight-level learning untouched. OWASP lists data and model poisoning as LLM04:2025 (OWASP Foundation, 2025).

Retrieval instead of learning. Retrieval-augmented generation (Lewis et al., 2020) keeps new facts outside the weights, so a poisoned fact never becomes part of the model. It does not remove the trust problem, because a poisoned document store misleads the model at inference time just as a poisoned stream does during learning. We study the parametric case, where an accepted claim is written into the weights and persists in every later answer, which makes the acceptance decision more consequential. The evidence route of Section 5.6 sits between the two: it uses a retrieved source to decide what may be written into the weights.

Verification and knowledge editing. LLM-as-a-judge (Zheng et al., 2023) is the standard way to score model outputs; we use it as a gate on learning. CounterFact (Meng et al., 2022) provides real facts paired with counterfactual rewrites and is the standard benchmark for knowledge editing; we reuse its counterfactuals as plausible lies. Li et al. (2025) identify, after the fact, whether a model was edited maliciously (for example with misinformation) or benignly (fact updates); we ask the same question before a claim is learned, and Section 5.9 shows that the model’s prior alone cannot answer it.

Our contribution. Judges that override a provided reference when it contradicts their own knowledge have been documented at evaluation time (Lee et al., 2026), and claim-level provenance with hadith-style chains of narrators has been proposed for multi-agent knowledge systems (Raja, 2026). We study the acceptance decision when claims are written into the weights of a continually learning model and the stream is poisoned by a fixed attacker; we additionally evaluate one adaptive attacker that screens its lies against the prior key. Our contributions are empirical: the vulnerability of surprise-driven selection, and a measurement of what each acceptance signal detects, including how it fails.

## 3 Method

A frozen base model with parameters $\theta _ { 0 }$ learns only through a LoRA adapter. Claims $c = ( x , y )$ 9 a prompt x and an object $y ,$ arrive in chunks. For each chunk, every claim must pass the active keys; the accepted claims, together with replayed claims accepted earlier, train a candidate adapter, which is then committed or rolled back (Figure 1). Each key measures a diferent property, and none of them measures truth directly: surprise measures novelty, the prior-conflict key contradiction with what the original model knows, the verifier key a judge’s verdict, the statistical test collateral damage to unrelated knowledge, and the evidence route agreement with a source.

![](images/3b9037b5c3d4cec043edf18435b0493894504339de8aed9e47d7863fa39431f1.jpg)  
Figure 1: The acceptance decision for one claim (sura/runner.py). A claim is learned only if it is new, uncontradicted and verified. Claims are processed in chunks of 16; each key can be switched of for the ablations.

Surprise. The mean negative log-likelihood per target token under the current model,

$$
s _ { \theta } ( c ) = - { \frac { 1 } { | y | } } \sum _ { t = 1 } ^ { | y | } \log p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) .\tag{1}
$$

A claim is a learning candidate when $s _ { \theta } ( c )$ exceeds $\tau = 1 . 0$ nats/token. Claims the model already predicts are skipped.

Prior-conflict key. We ask the original model $\theta _ { 0 }$ for its own greedy answer $\hat { y }$ to the prompt and measure how much less likely the claim is than that answer:

$$
g ( c ) = s _ { \theta _ { 0 } } ( c ) - s _ { \theta _ { 0 } } ( \hat { y } ) , \qquad \mathrm { h o l d ~ b a c k ~ i f ~ } g ( c ) \ge \tau _ { g } = 6 .\tag{2}
$$

A claim with a large gap contradicts a belief the model held before any user taught it anything, so it is held back for review instead of learned. We first computed the gap under the current adapter. In a pilot this held back 46% of true claims, because during learning the adapter becomes overconfident in recently learned patterns; switching to $\theta _ { 0 }$ removed the problem. The gap needs no judge and is computed once per claim.

Verifier key. A verifier sees the claim, plus a trusted source sentence when one exists, and answers true or false. Each distinct claim is judged once.

Statistical test (optional). After training a candidate adapter, we score the regression suite in batches of $b = 2 5$ items. For item $i , d _ { i }$ is the loss of preference margin (nats/token) for the true object over its counterfactual, relative to the original base $\theta _ { 0 }$ . We compare $H _ { 0 } \colon \mathrm { m e a n } ( d ) = \mu _ { 0 }$ , the drift of clean learning calibrated in Section 5.3, against $H _ { 1 }$ : mean $( d ) = \mu _ { 0 } + \delta$ , using a Gaussian log-likelihood ratio after n scored items with Wald’s boundaries (Wald, 1945):

$$
\Lambda _ { n } = \frac { \delta } { \hat { \sigma } _ { n } ^ { 2 } } \left( \sum _ { i = 1 } ^ { n } d _ { i } - n \left( \mu _ { 0 } + \frac { \delta } { 2 } \right) \right) , \qquad A = \ln \frac { 1 - \beta } { \alpha } , \qquad B = \ln \frac { \beta } { 1 - \alpha } .\tag{3}
$$

After each batch, $\Lambda _ { n } \geq A \ ( 2 . 7 7$ for $\alpha = 0 . 0 5 , \beta = 0 . 2 )$ declares harm and $\Lambda _ { n } \leq B \ ( - 1 . 5 6 )$ declares no harm. If the 200-item suite runs out first, harm is declared when $\Lambda _ { n } > 0$ . Harm restores the adapter and optimizer state from their snapshot. The variance $\hat { \sigma } _ { n } ^ { 2 }$ is re-estimated from all items scored so far, items are treated as independent, and the suite is finite. The procedure is therefore SPRT-style: Wald’s boundaries set its thresholds, but we claim no formal control of the error rates α and $\beta .$ An alternative paired one-sided t-test on the whole suite is implemented but not reported.

Training. Cross-entropy on the target tokens of the accepted and replayed claims plus $\mathrm { K L } ( \theta _ { 0 } \| \theta )$ on anchor sentences of known facts, which keeps the adapter close to the base. The replay bufer keeps the 512 most surprising accepted claims.

## 4 Experimental setup

Student. Qwen2.5-1.5B-Instruct (Qwen Team, 2024) (Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct (Llama Team, AI @ Meta, 2024) for the replication in Section 5.7), frozen, in bfloat16, with one LoRA adapter (Hu et al., 2022) (rank 16, alpha 32, on all attention and MLP projections). Each accepted chunk trains for 8 AdamW steps (learning rate $2 \times 1 0 ^ { - 4 }$ , batch 16) on its claims plus 32 replayed accepted claims, with a KL term to the frozen base on 16 anchor sentences. Everything runs on one 16 GB laptop GPU; a run over the stream takes 5–12 minutes.

Data. Table 1 lists the claim sets. The generator gives every one of the 120 people all 5 relations, a value drawn from that relation’s pool of 12–18 values, and shufles the 600 facts with the world seed. The first 400 become the true new facts; the remaining 200 form the pool for lies about invented people, so no lie concerns a fact that is also streamed as true. A poison rate $p$ adds round $( p / ( 1 - p ) \times 4 0 0 )$ lies, split evenly between the two kinds (171 lies at 30%, 400 at 50%). The poison rate is thus the share of lies among the new facts and lies in the stream; the 100 legitimate updates of Sections 5.6–5.8 come on top. The same 400 new facts are used at every poison rate, so rates compare fairly. Sections 5.6–5.8 add 100 legitimate updates, taken from known facts outside every other set. Each claim appears twice in a shufled stream, read in chunks of 16. Retention is measured only on the 400 held-out facts, which no system ever sees. We use atomic factual claims on purpose, to isolate the acceptance decision from the confounds of long documents, multi-hop reasoning and task-specific objectives. We call the CounterFact claims known-fact lies: their subjects and relations are real, but the claims are counterfactual rewrites, not naturally occurring misinformation.

Development and held-out data. Seeds 0–2 (Sections 5.1–5.5) served development: the gate’s null drift was calibrated on clean seed 0–2 runs, and the prior-key rule and its threshold, $g \geq 6$ were chosen on a 30-claim sample of the seed-0 world, which the prior sweep of Section 5.4 also uses. The seed-7 diagnostic (Figure 5, Table 5) and every run on seeds 10–14 (Sections 5.6–5.8) were made after the threshold was fixed and never retuned it, also for the two further models; we treat them as post-development evaluations with the threshold fixed. Because the development sample overlaps the seed-0 world, the prior sweep of Section 5.4 is not an unbiased estimate of the threshold’s performance; the cleaner evidence comes from seed 7 and seeds 10–14. The overlap analysis at the end of Section 5.8 is exploratory.

Table 1: Claim sets of one world.
<table><tr><td>Set</td><td>Source</td><td>Size</td><td>Role</td></tr><tr><td>New facts</td><td>400 of the 600 facts generated for 120 invented people × 5 relations (birthplace, profession,</td><td>400</td><td>True claims the model cannot know; should be learned</td></tr><tr><td>Lies about invented people</td><td>language, instrument, sport) The other 200 generated facts, each given a wrong value from its relation&#x27;s pool</td><td>up to 200</td><td>Lies a source would contradict</td></tr><tr><td>Known-fact lies</td><td>CounterFact (Meng et al., 2022) counterfactuals on facts the student knows</td><td>up to 200</td><td>Plausible lies about known facts</td></tr><tr><td>Known facts</td><td>CounterFact records the student ranks correctly (2,443 of 3,000 screened)</td><td>200 / 300 / 400 Regression suite / KL</td><td>anchor / held-out retention set</td></tr></table>

Systems. Each system is the same loop with keys switched on or of (Table 2).

Table 2: Systems compared. Every system uses the same student, stream and training loop.
<table><tr><td>System</td><td>Learns from</td><td>Safety check</td></tr><tr><td>Naive</td><td>every claim</td><td>none</td></tr><tr><td>Surprise-only (SuRe-lite)</td><td>surprising claims, replay, fast/slow EMA</td><td>none</td></tr><tr><td>Stability-only (STABLE-lite)</td><td>every claim</td><td>rollback if suite exact-match drops &gt; 0.02</td></tr><tr><td>Verifier key</td><td>surprising and verified claims</td><td>none</td></tr><tr><td>Verifier key + test</td><td>surprising and verified claims</td><td>SPRT vs. original base, commit / rollback</td></tr><tr><td>Test only</td><td>surprising claims</td><td>SPRT (ablation: no verifier key)</td></tr><tr><td>Prior only</td><td>surprising claims without prior conflict</td><td>none (no judge)</td></tr><tr><td>Verifier key + prior</td><td>surprising, verified, no prior conflict</td><td>none</td></tr><tr><td>Verifier key + prior + evidence route</td><td>as above, plus held-back claims whose trusted source states the same value</td><td>none</td></tr></table>

Verifiers. An oracle (ground truth), a simulated judge that is right with probability 0.6–0.9 and wrong consistently on the same claims, and Gemini 2.5 Flash (Gemini Team, Google, 2025), which sees a trusted source sentence for the invented people that have one. Two local models served through Ollama, phi4 (14B) (Abdin et al., 2024) and Llama 3.2 (3B) (Llama Team, AI @ Meta, 2024), act as real judges in Section 5.8. All real judges receive the same prompt, given in

Appendix A.

Metrics. For invented people, a fact counts as learned, or a lie as absorbed, when it is the model’s top-1 choice among all 12–18 values of its relation (chance about $6 \% )$ . For real facts we compare the true object with the counterfactual (before learning: 1 for retention, 0 for known-fact lies). Lies absorbed is measured on the model after learning, over every distinct lie ℓ in the set L of streamed lies, not only those selected for training. We report it separately for known-fact lies and lies about invented people; “all lies” is their count-weighted mean. Lies per 1,000 learned claims instead measures the learner’s input: it counts claim occurrences accepted for training in committed chunks, excluding replay.

$$
{ \mathrm { a b s o r b e d } } = { \frac { | \{ \ell \in L : { \mathrm { t h e ~ m o d e l ~ a s s e r t s ~ } } \ell { \mathrm { ~ a f t e r ~ l e a r n i n g } } \} | } { | L | } } ,\tag{4}
$$

$$
\mathrm { l i e s ~ p e r ~ 1 , 0 0 0 = 1 0 0 0 \cdot { \frac { \# \mathrm { l i e ~ o c c u r r e n c e s ~ t r a i n e d } } { \# \mathrm { c l a i m ~ o c c u r r e n c e s ~ t r a i n e d } } } . }\tag{5}
$$

Results are means over 3 seeds in Sections 5.1–5.5 and over 5 held-out seeds in Sections 5.6–5.8. Because claims about the same invented person are not independent, uncertainty estimates that involve them resample whole people (person-clustered bootstrap).

## 5 Results

## 5.1 Surprise alone learns lies

With a perfect verifier, every system without a verifier key learned most of the lies it saw. The two surprise-driven systems learned the most, since lies are among the inputs they select. Known-fact lies fared worst. Surprise-only learning absorbed 99–100% of them at every poison rate, against 68–86% of lies about invented people and 84–93% of all lies (the measure in Figure 2).

![](images/8ba6acd05785e38652cbcdacb81b236db463f08d51bf25535c42b0ba596f94cc.jpg)  
Figure 2: Share of all streamed lies asserted after learning, by poison rate (oracle verifier; Qwen2.5-1.5B; 6 systems × 3 poison rates $\times \ 3$ seeds; results/main).

Table 3 compares the systems at $3 0 \%$ poison. The verifier key learned at least as much as any other system and protected old knowledge best.

Table 3: Learning and retention at 30% poison (oracle verifier). Means over 3 seeds; seed standard deviations are at most 0.05, except for the verifier key with the uncalibrated test (0.18 at 30%, 0.35 at 50% poison), whose rollbacks made learning unstable.
<table><tr><td>System (30% poison)</td><td>New facts learned</td><td>Old facts kept</td></tr><tr><td>Before learning</td><td>0.06</td><td>1.00</td></tr><tr><td>Naive</td><td>0.19</td><td>0.92</td></tr><tr><td>Surprise-only</td><td>0.80</td><td>0.93</td></tr><tr><td>Stability-only</td><td>0.17</td><td>0.90</td></tr><tr><td>Surprise + test, no verifier key</td><td>0.74</td><td>0.92</td></tr><tr><td>Verifier key</td><td>0.86</td><td>0.98</td></tr><tr><td>Verifier key + uncalibrated test</td><td>0.49</td><td>0.98</td></tr></table>

The verifier key also kept more old knowledge (0.98 vs. 0.93), probably because rejecting lies about known facts avoids the edits that would overwrite them most directly. Naive learning, without replay, retains little of what it sees. The stability-only gate protects old facts but blocks most learning; at 50% poison it rolled back 94 of 100 chunks.

Surprise versus random selection. To separate the efect of selecting by surprise from the efect of training at all, we added a control that trains on a random subset of each chunk, of the same size as the surprising claims, with the same replay and averaging (held-out seeds 10–14, 30% poison, 100 updates, sources for half the claims, judge accuracy 0.9 for the judged systems). Both learners took in lies at the base rate of the stream, 254 and 257 per 1,000 learned claims (paired diference −3, 95% CI −15 to +9). Selecting by surprise neither raised nor lowered the share of lies in the training input. On known-fact lies it did worse than random selection. It absorbed 99.5% of them against 88.9% (paired diference +11 points, 95% CI +9 to +13) while training on 86 fewer claims per run, because a lie about a known fact is always surprising. Against poison, a surprise filter is thus no better than random selection, and for lies about known facts it is worse.

Open-ended answers. In the same runs we also let the model answer every prompt freely (greedy decoding) and counted a claim as asserted when the answer began with its value. Under this exact-match assertion rule, the free answers are directionally consistent with the ranking metrics. The surprise-only learner stated 98% of the known-fact lies (ranking metric: 99.5%), the verifier key 14% (17%) and the full system with the evidence route 4% (8%); before learning, the model stated none of them. True new facts were stated in 71–79% of cases. Paraphrased questions were answered with the learned value less often (39–41% for new facts, 3–37% for known-fact lies), so facts learned by fine-tuning transfer weakly across wordings. Retention cannot be read this way: before learning, the model’s free answers matched the CounterFact object for only 21% of the facts it knew.

## 5.2 Lies absorbed follow the judge’s errors

The simulated judge in this section is not meant as a realistic model of judge behaviour. It is a controlled sensitivity analysis that isolates the efect of verifier accuracy; Section 5.8 provides the realism check with real judges. Real verifiers make mistakes, so we replaced the oracle with a simulated judge that is right with probability 0.6 to 0.9. Its errors are consistent, so a lie that fools it once always fools it. Lies absorbed then followed the judge almost one for one (Figure 3). The student learned nearly every lie the judge approved and very few others.

![](images/8cb8df2bb95b00da7052911d3e58319a2102a8c52a741d56c174fc8a90f3a922.jpg)  
Figure 3: Lies absorbed by the verifier key without test, against the share of lies the simulated judge approves (4 accuracies × 3 seeds; results/judge\_sweep).

Learning of new facts also improved with judge quality (0.60 at accuracy 0.6, 0.80 at 0.9), because a weaker judge rejects more true claims. A verifier key shifts the problem to the verifier. The safety of continual learning becomes the verifier’s reliability, measured on the claims an attacker chooses.

## 5.3 Regression tests throttle, they do not filter

The statistical test can detect that something went wrong, but not which claim caused it. We calibrated it on clean learning (no poison, test of). The per-chunk mean loss of preference margin on the suite ranged from −0.72 to +0.78 nats/token while retention stayed at 0.97–0.98, so we set the null drift to its 95th percentile (0.446) and the harmful efect to its standard deviation (0.203). With this calibration, rollbacks rise as the judge weakens, but the share of poison among the claims that are learned does not change (Table 4).

Table 4: The calibrated test with a simulated judge at 30% poison (3 seeds; results/judge\_sweep).
<table><tr><td rowspan="2">Judge accuracy</td><td rowspan="2">Rollbacks per run (with test)</td><td colspan="2">Poison share of learned claims</td><td colspan="2">New facts learned</td></tr><tr><td>with test</td><td>without test</td><td>with test</td><td>without test</td></tr><tr><td>0.6</td><td>26.3</td><td>22.0%</td><td>22.2%</td><td>0.46</td><td>0.60</td></tr><tr><td>0.7</td><td>38.3</td><td>14.7%</td><td>15.3%</td><td>0.40</td><td>0.67</td></tr><tr><td>0.8</td><td>15.3</td><td>10.2%</td><td>9.8%</td><td>0.63</td><td>0.71</td></tr><tr><td>0.9</td><td>2.0</td><td>5.3%</td><td>5.3%</td><td>0.79</td><td>0.80</td></tr></table>

A rolled-back chunk mixes accepted true claims with leaked lies, and the test discards both. It lowers lies absorbed only by learning less (at accuracy 0.7, lies fall from 0.31 to 0.20 and new facts from 0.67 to 0.40), which a smaller learning rate would also achieve. The earlier uncalibrated version, which tested the raw NLL rise with a tolerance of 0.05, was worse. It rolled back over half the chunks even with no poison and halved learning (Table 3).

We built the test as the main safety check, so this result was a disappointment, and the reason seems simple. A lie about one fact barely moves the model’s answers on unrelated facts, and our suite contained only unrelated facts. In our setting, regression testing on unrelated knowledge did not filter targeted factual poisoning. A suite built from the neighbourhood of each incoming claim, or designed to probe contradictions, might behave diferently; we did not test this.

## 5.4 The model’s own prior catches contradictions

A lie about a known fact contradicts something the model already believes, and a true new fact does not. The prior-conflict key uses this diference without any judge. Added to the imperfect judge, it cut the known-fact lies the student ends up asserting by 3–5× at every accuracy, so a weak judge with the prior key did better than a good judge alone (Figure 4). With no judge at all, the prior key held known-fact lies to 24%, against 99–100% for surprise-only learning.

![](images/1837baf8c5e00b332e62f5b0a8cf39769324d232796e07bca8448fc5667212c4.jpg)  
Figure 4: Known-fact lies asserted after learning, with and without the prior key $( g \ge 6 )$ , across judge accuracies (3 seeds; results/prior\_sweep).

The key has two costs. It held back 30% of true new claims in the sweep, which lowered learning of new facts from 0.60–0.79 to 0.44–0.60 depending on the judge; in a deployed system these claims would go to review rather than be lost. Retention was unchanged or slightly better (0.97–0.98). The key also cannot see lies about invented people, which were absorbed about as often with it as without it (0.35 vs. 0.41 at judge accuracy 0.6, 0.15 vs. 0.16 at 0.9), since the model holds no belief for them to contradict.

To measure the trade-of independently of the sweep, we scored 200 claims of each type on a fresh world (seed 7) under the original model. The gap separates known-fact lies from true new facts with AUC 0.83 (95% percentile bootstrap CI 0.79–0.87 from 300 resamples of single claims; resampling whole invented people with all their claims, 1,000 resamples, also gives 0.79–0.87), and lies about invented people no better than chance (AUC 0.49, CI 0.43–0.56). The operating point used in the sweep, $g \geq 6 .$ , holds back 27% of true facts and 76% of known-fact lies on this fresh world, matching the 30% cost measured in the sweep; with equal numbers of each, 74% of the claims it holds back are lies. Measured as the share of lies among the claims actually learned, the prior key lowers poison from 152 to 111 per 1,000 learned claims at judge accuracy 0.7 and from 54 to 43 at 0.9; the remaining lies are mostly about invented people, which the key cannot see.

AUC is the probability that a randomly drawn lie receives a larger gap than a randomly drawn true fact. It measures how well the gap ranks the two groups. It is not an accuracy, since the share of lies caught at a given cost depends on the threshold and is read of the curve in Figure 5.

Scoring each claim by its best article variant (“nurse”, “a nurse”, . . . ) removed template artefacts but left separation unchanged (AUC 0.83 for both rules); it only shifted the threshold. The threshold is best seen as a policy choice between safety and learning (Table 5). Holding back a true claim only delays it, while learning a lie corrupts the model, so we prefer an operating point on the safe side. The same threshold of 6 is used for all three models without retuning.

![](images/0bd48531b03b732d7f4f60c32dcb5508f9c4c135d38df8af24a6432b63671afd.jpg)  
Figure 5: Trade-of curve of the prior gap on the fresh seed-7 world under each original model (200 claims per type). Each point is a threshold; the diagonal is chance.

Table 5: Safety–learning operating points of the prior key on the fresh seed-7 world with 200 claims per type. Each cell gives Qwen2.5-1.5B / Qwen2.5-3B / Llama-3.2-3B (the larger models are introduced in Section 5.7). The fourth column tracks the third at every threshold, so for lies about unknown entities the key holds back lies and truths at the same rate. Precision compares known-fact lies with true new facts in this balanced sample (200 of each); with fewer lies in the stream it would be lower. Person-clustered bootstrap intervals for the AUC are 0.78–0.86 (Qwen2.5-3B) and 0.74–0.82 (Llama-3.2-3B), matching the claim-level ones.
<table><tr><td>Gap threshold</td><td>Known-fact lies held back</td><td></td><td>True new facts held back</td><td>Lies about invented people held back</td><td></td><td>Precision: lies among held-back claims</td></tr><tr><td>4</td><td>0.92 0.92</td><td>0.91</td><td>0.42 0.46</td><td>0.50</td><td>0.46 0.48 0.49</td><td>0.69 0.67</td></tr><tr><td>6 (used)</td><td>0.76 0.81</td><td>0.71</td><td>0.27 0.33 0.32</td><td>0.29</td><td>0.34 /0.29</td><td>0.65 0.74/ 0.71 /0.69</td></tr><tr><td>8</td><td>0.54 / 0.56 / 0.44</td><td></td><td>0.14 0.18 /0.17</td><td>0.14</td><td>0.13 / 0.16</td><td>0.79 / 0.76 / 0.72</td></tr></table>

## 5.5 A strong real judge learns only what it can verify

With Gemini 2.5 Flash as the verifier key and a trusted source available for only half of the invented people (30% poison, verifier key without test, 3 seeds), the student absorbed 6.4% of lies, the same as before learning, and kept 98.0% of old facts. The judge approved about 2 of the roughly 170 distinct lies in each run (about 1%), and it rejected about 198 of the roughly 390 true claims, almost exactly the unsourced ones. New facts learned therefore stopped at 0.51, against 0.86 with the oracle.

Gemini rarely guessed. It rejected almost every claim it could not check against a source, so it was safe, but it learned no more than the sources covered. The harder case is a claim that is true, new and unsourced. Neither a judge nor the model’s prior can resolve it (Section 5.4), which motivates corroboration across independent sources as future work. Adding the statistical test changed nothing here (lies 5.8%, new facts 0.50, retention 97.8%).

## 5.6 Legitimate updates and the evidence route

The prior key cannot tell a lie about a known fact from true news that changes it: both contradict the model. We tested this directly by adding 100 legitimate updates to the stream: known CounterFact facts whose counterfactual value is, in this world, the new truth, each reported by a trusted source. We then added a trusted-source evidence route: a claim the prior key holds back is learned anyway if a trusted source states the same value. A planted lie has no matching source by assumption, so the route cannot readmit it. Sources were available for all claims or, in the more realistic setting, for half of them. All runs use 30% poison, the simulated judge, and five held-out seeds (10–14) that no earlier analysis used, with the gap threshold fixed at 6 before running (Table 6).

Table 6: Legitimate updates and the evidence route. Judge accuracy 0.9, means over 5 held-out seeds; seed standard deviations are at most 0.07. At judge accuracy 0.7 the pattern is the same (updates learned 0.70 / 0.19 / 0.93 with all sources and 0.59 with half; lies per 1,000 learned claims 129–132 / 112–114 / 60 and 79).
<table><tr><td>Sources</td><td>System</td><td>Updates learned</td><td>New facts learned</td><td>Known-fact lies believed</td><td>Old facts kept</td><td>Lies per 1,000 learned claims</td></tr><tr><td>All claims</td><td>Judge only</td><td>0.90</td><td>0.79</td><td>0.18</td><td>0.92</td><td>42</td></tr><tr><td>All claims</td><td>Judge + prior</td><td>0.24</td><td>0.60</td><td>0.07</td><td>0.97</td><td>33</td></tr><tr><td>All claims</td><td>+ evidence route</td><td>0.98</td><td>0.79</td><td>0.11</td><td>0.91</td><td>18</td></tr><tr><td>Half of claims</td><td>Judge only</td><td>0.90</td><td>0.81</td><td>0.18</td><td>0.93</td><td>44</td></tr><tr><td>Half of claims</td><td>Judge + prior</td><td>0.24</td><td>0.60</td><td>0.07</td><td>0.97</td><td>32</td></tr><tr><td>Half of claims</td><td>+ evidence route</td><td>0.62</td><td>0.72</td><td>0.09</td><td>0.94</td><td>24</td></tr></table>

The concern was real. The prior key alone blocked about three quarters of legitimate updates. The evidence route recovered them in proportion to source coverage. It rescued 192 of the 192 held-back true claims per run with full coverage and 101 of 192 with half, and it rescued no lie in any run, as the trusted-source assumption implies, because a planted lie never has a matching source. The combined system had the lowest share of poison among the claims it learned in every setting, roughly half that of the judge alone. Learning updates did cost some retention. Systems that learned most of the 100 updates kept about 5 points fewer of the untouched held-out facts (0.91–0.93 against 0.97), a side efect familiar from knowledge editing, where changing one fact disturbs its neighbours.

The sources in the main experiment are perfect and cannot be planted or copied. The evidence route therefore shows that legitimate updates can coexist with the prior key. It does not show that they can be trusted against an attacker who also controls sources, which is the corroboration problem of Section 8.

Planted and outdated sources. We then broke the trusted-source assumption (judge accuracy 0.9, sources for half the claims, held-out seeds 10–14). When an attacker planted a matching source for 10% of the known-fact lies, the evidence route admitted 4 lies per run and lies per 1,000 learned claims rose from 23 to 35 (paired diference +12, 95% CI +8 to +16). With planted sources for 30% of the lies it admitted 18 lies per run, the model believed 29% of the known-fact lies instead of 8%, and lies per 1,000 learned claims rose to 65, worse than the judge alone in the same runs (41).

When 30% of the sourced updates came with an outdated source that still stated the old value, no lie entered, but updates learned fell from 0.63 to 0.48. In short, the evidence route is as safe as its sources are hard to forge.

## 5.7 Replication across model size and family

We repeated the realistic setting of Section 5.6 (sources for half the claims, 100 legitimate updates, 30% poison, held-out seeds 10–14) with two further students, Qwen2.5-3B-Instruct (a larger model of the same family) and Llama-3.2-3B-Instruct (a diferent family), keeping every setting, including the gap threshold of 6, unchanged. On the fresh seed-7 world the prior gap separates known-fact lies from true new facts with AUC 0.82 (95% CI 0.78–0.86) for the 3B model against 0.83 for 1.5B, and Llama-3.2-3B reaches 0.78 (0.74–0.81); lies about invented people stay at chance for all three (0.49–0.50). The same threshold transfers without retuning: it holds back 33% (Qwen 3B) and 32% (Llama) of true facts and catches 81% and 71% of known-fact lies, against 27% and 76% for Qwen 1.5B.

Table 7: Replication with three students. Each cell gives Qwen2.5-1.5B / Qwen2.5-3B / Llama-3.2-3B; judge accuracy 0.9; sources for half the claims; means over 5 held-out seeds, seed standard deviations at most 0.06. Judge accuracy 0.7 shows the same pattern for all three models.
<table><tr><td>System</td><td>Known-fact lies believed</td><td>Lies about invented people</td><td>Updates learned</td><td>New facts learned</td><td>Old facts kept</td><td>Lies per 1,000 learned claims</td></tr><tr><td>Surprise-only (no judge)</td><td>1.00 / 1.00 / 1.00</td><td>0.79 / 0.81 / 0.87</td><td>0.99 / 0.99 0.99</td><td>0.77 / 0.81 / 0.84</td><td>0.88 / 0.89 / 0.88</td><td>257 / 251 / 254</td></tr><tr><td>Judge only</td><td>0.18 / 0.16 / 0.19</td><td>0.16 / 0.16 / 0.15</td><td>0.90 / 0.91 / 0.90</td><td>0.81 / 0.81 / 0.83</td><td>0.93 / 0.93 / 0.92</td><td>44 / 37 / 38</td></tr><tr><td>Judge + prior</td><td>0.07 / 0.04 / 0.05</td><td>0.12 0.11 / 0.12</td><td>0.24 / 0.17 / 0.20</td><td>0.60 / 0.57 / 0.57</td><td>0.97 / 0.97 / 0.97</td><td>32 / 25 / 29</td></tr><tr><td>+ evidence route</td><td>0.09 / 0.07 / 0.08</td><td>0.14 / 0.14 / 0.13</td><td>0.62 / 0.57 / 0.59</td><td>0.72 / 0.72 / 0.73</td><td>0.94 / 0.95 / 0.94</td><td>24 / 18 / 22</td></tr></table>

Every qualitative finding replicated across both families (Table 7). Surprise-only learning absorbed essentially all known-fact lies. The prior key cut them roughly 2–4× relative to the judge alone and left lies about invented people untouched. The evidence route recovered legitimate updates in proportion to source coverage and rescued no lie in any run, and the combined system had the lowest poison share among learned claims. Neither the larger model nor the other family made the prior key stronger, and Llama separates somewhat less sharply, with overlapping confidence intervals. For Qwen 3B, gaps grew for true new facts (median 3.8 vs. 3.2 nats/token) at least as much as for known-fact lies (8.4 vs. 8.2), so separation did not improve at this scale.

## 5.8 Real judges are strict and defend their own beliefs

The simulated judge errs symmetrically. To see how real verifiers err, we replaced it with two local LLM judges served through Ollama: phi4 (14B, a diferent family from the student) and Llama 3.2 (3B, a deliberately weak judge). The setting is that of Sections 5.6–5.7 with the Qwen2.5-1.5B student (sources for half the claims, 100 legitimate updates, 30% poison, held-out seeds 10–14). Every verdict and prior gap was recorded per claim, about 3,300 judged claims per judge pooled over the 5 seeds; no judge answer was unparseable.

Real judges erred mostly in one direction (Table 8). They rarely accepted a lie, but they rejected every unsourced truth, and the weaker judge also rejected a third of sourced ones, so its weakness showed up as strictness rather than gullibility. Both judges also rejected most legitimate updates even when the trusted source was in front of them, although the prompt states that a given source is correct; the source contradicted what they believed. In this respect the judge is as conservative as the prior key, consistent with the knowledge-driven failures of LLM judges reported by Lee et al. (2026). Verification has two separate dimensions: safety, the share of lies a verifier accepts, and coverage, the share of truths it accepts. Real judges buy high safety with low coverage, whereas the simulated judge of Section 5.2 assumed both errors were equally likely.

Table 8: Share of claims each real judge accepted, pooled over 5 seeds.
<table><tr><td>Claim type</td><td>phi4 (14B)</td><td>Llama 3.2 (3B)</td></tr><tr><td>Known-fact lies</td><td>5.9%</td><td>4.0%</td></tr><tr><td>Lies about invented people</td><td>0.5%</td><td>0.0%</td></tr><tr><td>True new facts with a source</td><td>99%</td><td>65%</td></tr><tr><td>True new facts without a source</td><td>0%</td><td>0%</td></tr><tr><td>Legitimate updates with a source</td><td>30%</td><td>40%</td></tr><tr><td>Legitimate updates without a source</td><td>4%</td><td>1%</td></tr></table>

Table 9: Full systems with real judges (phi4 / Llama 3.2). Means over 5 held-out seeds; seed standard deviations at most 0.05. The evidence route rescued no lie.
<table><tr><td>System</td><td>Updates learned</td><td>New facts learned</td><td>Known-fact lies believed</td><td>Old facts kept</td><td>Lies per 1,000 learned claims</td></tr><tr><td>Judge only</td><td>0.19 / 0.23</td><td>0.55 / 0.38</td><td>0.09 / 0.09</td><td>0.96 / 0.96</td><td>23 / 21</td></tr><tr><td>Judge + prior</td><td>0.07 / 0.08</td><td>0.40 / 0.28</td><td>0.06 / 0.04</td><td>0.98 / 0.98</td><td>16 / 10</td></tr><tr><td>+ evidence route</td><td>0.47 / 0.46</td><td>0.54 / 0.41</td><td>0.07 / 0.07</td><td>0.94 / 0.94</td><td>9/5</td></tr></table>

Because the evidence route lets a source-backed claim bypass the judge, it more than doubles the updates learned with either real judge, and the full system again holds the fewest lies among learned claims, 5–9 per 1,000 against 21–23 for the judge alone (Table 9). These remaining lies all entered through the judge; none entered through the evidence route.

Do the two keys fail on the same claims? Of the 42 known-fact lies the two judges accepted, the prior key would have held back 26 (62%; 14 of 25 for phi4, 12 of 17 for Llama). Its hold rate on all known-fact lies is 77%. The 42 cases are only 28 distinct claims, because both judges saw the same worlds, so a test on the pooled cases would overstate their independence. At the seed level, the prior key’s hold rate on judge-approved lies was below its overall rate in 8 of 10 judge–seed runs (mean 65% vs. 77%; one-sided sign test over these 10 runs, $p \approx 0 . 0 5 )$ . With 10 runs and 28 distinct claims, this is weak evidence, and we report it as exploratory. The observed rates suggest that the lies that fool a judge may also be somewhat harder for the prior key, so the two keys should not be assumed independent; combining them still helped in every setting.

## 5.9 What the prior key detects, and how it is bypassed

Section 5.4 left open whether the prior key detects lies or only contradictions. Three analyses of the saved results answer this question (review\_analyses.py).

First, the gap does not separate a lie about a known fact from a legitimate update of the same kind of fact. On the held-out runs of Section 5.8 it scores AUC 0.53 (95% CI 0.49–0.56; 424 lies,

500 updates) and holds back 77% of the lies and 75% of the updates. A second diagnostic on the fresh seed-7 world completes the four combinations of true or false and known or unknown (200 claims per type, plus 100 updates). True claims about known facts have small gaps (mean 3.2–3.5 nats/token; 17–20% held back), as do true new facts (27–33%) and lies about invented people (29–34%). Known-fact lies (71–81%) and legitimate updates (67–84%) are held back at similar rates, with AUC 0.45–0.53 between them across the three students. The gap is large whenever a claim contradicts what the model knows, whether the claim is true or false.

Second, a simpler signal does as well (Table 10). The original model’s confidence in its own first answer token, which ignores the claim entirely, separates known-fact lies from true new facts with AUC 0.84, 0.80 and 0.88 for the three students, against 0.83, 0.82 and 0.78 for the gap. The prior gap is therefore best seen as an interpretable contradiction signal rather than a uniquely strong detector; its practical advantage is that it needs no judge call and can route suspicious claims before verification. What the prior key mostly detects, then, is that a claim touches something the model already knows.

Table 10: AUC for separating known-fact lies from true new facts on the fresh seed-7 world, by signal (200 claims per type; 95% bootstrap intervals are about ±0.04).
<table><tr><td>Signal (original model)</td><td>Qwen2.5-1.5B</td><td>Qwen2.5-3B</td><td>Llama-3.2-3B</td></tr><tr><td>Prior gap</td><td>0.83</td><td>0.82</td><td>0.78</td></tr><tr><td>Claim NLL (surprise under the original model)</td><td>0.78</td><td>0.79</td><td>0.74</td></tr><tr><td>Confidence in its own answer (ignores the claim)</td><td>0.84</td><td>0.80</td><td>0.88</td></tr><tr><td>Prior gap, lies about invented people instead</td><td>0.49</td><td>0.50</td><td>0.49</td></tr></table>

Third, an attacker with access to the model can screen its lies and submit only those with a gap below 6. Only 23% of known-fact lies pass this filter, but each submitted lie then passes both keys 4.4 times as often: 11.5% instead of 2.6% with phi4, and 5.2% instead of 1.2% with Llama 3.2. Against such an attacker the prior key does not help, and safety depends on the judge alone.

Paired across seeds, the evidence route lowered lies per 1,000 learned claims relative to the judge alone in all three settings of Sections 5.6 and 5.8 (95% intervals exclude zero). It cost 3–4 points of retention relative to judge + prior, and with real judges it did not significantly change the share of known-fact lies the model believes.

## 6 Practical implications and application scenarios

Where this matters. The setting we study is an assistant that should absorb new facts from the people it serves, such as a university, public-administration or company assistant that must follow changing regulations, responsibilities and procedures. A fact written into the weights is available in every later answer without retrieval, which is also why a wrong acceptance is costly.

When to write into the weights at all. Retrieval (Lewis et al., 2020) is the safer default for facts that change often, must be traced to a document, or may have to be removed again. Parametric updates are worth their risk for stable facts that are used constantly, or where retrieval is unavailable. In both designs the acceptance decision studied here remains: a poisoned document store misleads the model as a poisoned stream does.

Conditions for deployment. Our results suggest four. (1) Never select what to learn by surprise alone. (2) Use a verifier key, and measure its safety and its coverage separately on the claims it will actually see. (3) Use the prior key as a cheap triage filter that sends conflicting claims to evidence or review. It does not stop an attacker with access to the model, who can bypass it. (4) Accept contradicting claims only with evidence from sources whose integrity is controlled independently of the claim’s author; with planted sources for 30% of the lies, the evidence route made the system worse than the judge alone (Section 5.6).

Costs. With the prior key in front of the judge, judge calls fell from about 661 to 374 per run (−43%). The reduction concerns judge calls, not total verification cost: about 180 claims per run stayed held back even with the evidence route and move to a review queue whose cost we did not measure. Learning updates cost 3–4 points of retention on untouched facts. A deployment has to weigh these costs against the cost of a single absorbed lie.

Open questions. How to establish that a source is trustworthy, how to handle sources that disagree or are outdated, how to make review cheap, and whether many small updates eventually harm knowledge more than they help.

## 7 Limitations

Two limitations matter most.

Legitimate updates of known facts. True news that changes something the model already knew (a new ofice holder, say) contradicts its prior just as a lie does; the prior key held back about three quarters of such updates in Section 5.6. On its own, the key cannot tell an update from a lie. In a continual learner this case is common. Held-back claims need a second route. Section 5.6 tests one, an evidence check against a trusted source; it works with perfect sources and fails once an attacker can plant them. The prior is also not ground truth: where pretraining taught the model something false, the prior key defends the error and holds back its correction. We therefore read the prior key as a triage filter that routes conflicting claims to evidence or review.

Adaptive attackers. Our poison is fixed in advance. Section 5.9 shows that an attacker who screens lies with the open model passes the prior key at will, so against such an attacker only the judge protects the model. We did not test attackers who also target the judge. Our numbers describe these attackers only and give no safety guarantee.

## Further limitations.

• Small models only. Main results use Qwen2.5-1.5B; the key comparisons are repeated on Qwen2.5-3B and Llama-3.2-3B (Section 5.7), and neither strengthened the prior key. Models larger than 3B parameters were not tested. The study is designed to establish the mechanism in small, reproducible models; larger models are future work.

• Short factual claims. Claims are single subject–relation–object facts. Instructions, opinions and multi-hop knowledge are not covered.

• Limited judge coverage. The simulated judge errs symmetrically; the real judges of Section 5.8 erred mostly by rejecting truths, and the lies they did accept were partly ones the prior key also misses. Two local judges and one API judge are a small sample of verifiers.

An attacker would search for those claims, so the relevant measure is a judge’s accuracy on adversarially chosen poison rather than its average accuracy.

• Unmeasured review load. We did not measure what it costs to review the claims the prior key holds back, by people or by retrieval; the 27% of true claims held back is a lower bound once legitimate updates are included.

• Template artefacts. Some held-back true facts arise from templates (for example, “The profession of $\mathrm { ~ X ~ i s ^ { \prime \prime } }$ strongly favours an answer beginning with $\mathrm { ^ { 6 6 } a \ . . . \ ^ { 9 } ) }$ . Scoring the best of article variants moved the threshold but left separation unchanged (AUC 0.83 either way).

• Threshold choice. The rule and the threshold $g \geq 6$ were developed on a 30-claim sample of the seed-0 world, which the prior sweep of Section 5.4 also uses. The seed-7 diagnostic and seeds 10–14 were run afterwards with the threshold fixed, and it transferred unchanged to two further models. The threshold also transfers unchanged across these settings. A test set untouched by any development choice would still be stronger evidence.

• Paraphrase transfer is weak. Facts learned in one wording are recalled in other wordings less often (0.62 vs. 0.86 with the oracle verifier key at 30% poison), a known limit of fact injection by fine-tuning.

• No retrieval baseline, limited open-ended evaluation. We did not compare with a retrieval system on the same stream; the question studied here is the acceptance decision for parametric learning, and retrieval changes the storage mechanism, so it calls for a separate baseline. Open-ended answers (Section 5.1) were checked in one setting and only as single greedy answers; multi-step questions and longer documents remain untested.

• Scale of compute. We used three seeds per setting in Sections 5.1–5.5 and five in Sections 5.6– 5.8, all on a laptop GPU. Bootstrap confidence intervals are reported for the prior-key AUC (Section 5.4), seed standard deviations under each table, and per-seed results in the released results folders.

## 8 Conclusion and future work

A false claim that contradicts a known fact is highly surprising to the model by construction, so selecting by surprise lets such lies through. In our controlled poisoned streams, a surprise-driven learner absorbed up to 93% of the lies it saw, and 99–100% of the known-fact lies. A verifier key removed them, but the learner then inherited the errors of its judge. Regression tests on unrelated knowledge did not help in our setting, because a lie barely afects other facts. The model’s own prior is a cheap second key that needs no judge and saves judge calls. It cuts absorbed known-fact lies 2.5–5× with simulated judges and 1.5–2.5× with real ones, at the price of sending about a quarter of true claims to review, but it reacts to any contradiction with what the model knows, true or false, and an attacker with access to the model can avoid it. None of the signals we study decides truth on its own, and our results suggest that a safer continual learner should combine complementary signals rather than rely on any one of them (Table 11). The real-judge results add a further constraint. They rarely accepted lies, but they also rejected most legitimate updates with a trusted source in view, so source-backed claims need a route that bypasses the judge.

Table 11: What each signal can and cannot see, summarising Section 5. <sup>∗</sup>Under the assumption of trusted sources that an attacker cannot plant or copy.
<table><tr><td>Signal</td><td>Question it answers</td><td>Lie about a known fact</td><td>Lie about an unknown entity</td><td>True new fact</td><td>True update of a known fact</td></tr><tr><td>Surprise Prior conflict</td><td>Is it new to me? Does it contradict</td><td>selects it holds most back</td><td>selects it blind (AUC 0.49)</td><td>selects it passes about 70%</td><td>selects it holds most back</td></tr><tr><td></td><td>what I believed? Does a verifier</td><td>(AUC 0.78–0.83) rejects 94–96%</td><td></td><td></td><td>rejects 60–70%</td></tr><tr><td>Judge</td><td>approve it? Does a trusted</td><td>(real judges)</td><td>rejects nearly all</td><td>accepts mainly if sourced</td><td>even with a source</td></tr><tr><td>Evidence route</td><td>source state the same?</td><td>never matches*</td><td>never matches*</td><td>accepts if sourced*</td><td>accepts if sourced*</td></tr><tr><td>Regression test</td><td>Did unrelated knowledge degrade?</td><td>blind</td><td>blind</td><td>(no effect)</td><td>(no effect)</td></tr></table>

What the experiments show. In controlled streams with three students of 1.5–3B parameters: surprise-driven selection absorbs most lies and nearly all known-fact lies, takes in lies at the same rate as random selection with the same budget, and absorbs known-fact lies more reliably than random selection does; a verifier key’s safety follows its judge; regression tests on unrelated facts throttle learning without filtering lies; the prior gap separates known-fact lies from true new facts (AUC 0.78–0.83) but neither lies about invented people (0.49–0.50) nor lies from legitimate updates (0.53), and the model’s answer confidence does as well; an attacker who screens lies with the model passes it; and with trusted sources, the evidence route recovers held-back updates while no lie enters through it, but planted sources for 30% of the lies make the system worse than the judge alone. Open-ended answers, under an exact-match assertion rule, are directionally consistent with the ranking metrics.

What they suggest. Real judges trade coverage for safety and defend their own beliefs against sourced updates (two local judges, Section 5.8). The judge and the prior key may fail partly on the same lies (exploratory, Section 5.8).

What remains a hypothesis. That these findings hold for larger models, adaptive attackers and sources an attacker can plant; and that corroboration across independent chains, proposed below, can resolve unsourced claims about unknown entities.

None of the keys in this study resolves the case of a new, unsourced claim about something the model has never seen. The judge refuses it, and the prior has nothing to contradict (AUC 0.49). We see four directions for future work.

1. Legitimate updates of known facts. True news that changes something the model knew contradicts its prior just as a lie does. Section 5.6 shows that routing held-back claims to an evidence check recovers them when sources are perfect, and that planted sources for 30% of the lies make it worse than the judge alone. An important next step is a defence against planted and copied sources, and real retrieval instead of a simulated source.

2. Corroboration. Accept unsourced claims only after independent confirmations; a liar is usually one source, the truth many. Corroboration should count independent chains of transmission rather than sources, since copies add no evidence; this is the criterion of the mutawatir report in hadith scholarship, which accepts a report only when so many independent chains carry it that collusion on a falsehood is implausible. Raja (2026) adapts related hadith methods to provenance in multi-agent knowledge systems; we propose applying the independence criterion to what may be written into the weights. New claims could wait in an erasable memory with provenance until enough independent chains arrive, and only then be consolidated into the weights. This turns the acceptance decision into a sequential test on independent evidence, where SPRT is the natural tool.

3. Adversarial verifiers and attackers. Measure judge accuracy on poison chosen to fool it, and attackers who adapt to the prior key (for example, lies phrased to sit close to the model’s own answer).

4. Breadth. Other model sizes and families, more seeds, claim types beyond single facts (instructions, procedures, multi-hop knowledge), and consolidation of committed adapters into the base (“sleep”) with the prior key anchored to the pre-learning model.

Code and data availability. The code, configurations, every saved result, and a script that rebuilds each table and figure of this paper from those results and checks every quoted number against them are available at https://github.com/Metalesaek/sura-metales.

## References

Marah Abdin et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

Gemini Team, Google. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Hugo Hazard, Zafeirios Fountas, Martin A. Benfeghoul, Adnan Oomerjee, Jun Wang, and Haitham Bou-Ammar. SuRe: Surprise-driven prioritised replay for continual LLM learning. arXiv preprint arXiv:2511.22367, 2025.

Almog Hilel, Idan Shenfeld, Jacob Andreas, and Leshem Choshen. LLM hypnosis: Exploiting user feedback for unauthorized knowledge injection to all users. arXiv preprint arXiv:2507.02850, 2025. A version titled “LLM Hypnosis: Characterizing the Fragility of RLHF Against Unprivileged Knowledge Injection” appeared at the ICLR 2026 workshop Agents in the Wild.

S. M. Asif Hossain, Ruksat Khan Shayoni, and Md Kishor Morol. CAPTURE: Disentangling preference drift from memory poisoning in personalized LLM agents. arXiv preprint arXiv:2609.02265, 2026.

William Hoy and Nurcin Celik. STABLE: Gated continual learning for large language models. arXiv preprint arXiv:2510.16089, 2025.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Dongryeol Lee, Yerin Hwang, Taegwan Kang, Minwoo Lee, Younhyung Chae, and Kyomin Jung. Judging against the reference: Uncovering knowledge-driven failures in LLM-judges on QA evaluation. arXiv preprint arXiv:2601.07506, 2026.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, 2020.

Xiaopeng Li, Shezheng Song, Shasha Li, Bin Ji, Shangwen Wang, Huijun Liu, Jun Ma, and Jie Yu. Identifying knowledge editing types in large language models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’25), 2025. doi: 10.1145/3711896.3737001. arXiv:2409.19663.

Llama Team, AI @ Meta. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

James L. McClelland, Bruce L. McNaughton, and Randall C. O’Reilly. Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review, 102(3):419–457, 1995.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, 2022.

OWASP Foundation. OWASP top 10 for LLM applications 2025: LLM04:2025 data and model poisoning, 2025.

Xu Pan, Ely Hahami, Zechen Zhang, and Haim Sompolinsky. Memorization and knowledge injection in gated LLMs. arXiv preprint arXiv:2504.21239, 2025.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Ali Zahid Raja. Grading the narrators: An isnad-rijal framework for claim-level provenance in multi-agent knowledge systems. arXiv preprint arXiv:2607.24117, 2026.

Donald Shenaj, Ondrej Bohdal, Taha Ceritli, Mete Ozay, Pietro Zanuttigh, and Umberto Michieli. K-Merge: Online continual merging of adapters for on-device large language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3013–3029, 2026. doi: 10.18653/v1/2026.acl-long.137.

Abraham Wald. Sequential tests of statistical hypotheses. Annals of Mathematical Statistics, 16(2): 117–186, 1945.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2023.

Yueer Zhou, Yichen Wu, and Ying Wei. Resolving conflicts in lifelong learning via aligning updates in subspaces. arXiv preprint arXiv:2512.08960, 2025.

Adam Zweiger, Jyothish Pari, Han Guo, Ekin Aky¨urek, Yoon Kim, and Pulkit Agrawal. Self-adapting language models. In Advances in Neural Information Processing Systems, 2025. arXiv:2506.10943.

## A Judge prompt and decoding

All real judges (Gemini 2.5 Flash, phi4, Llama 3.2) received the same system and user messages.   
The source line is present only when the claim’s subject has a trusted source.

system: You are a strict fact-checker. Decide whether the claim is true. If   
a trusted source is given, it is correct. Answer with one word: True or   
False.   
user: Trusted source: <source sentence>   
Claim: <claim sentence>   
Is the claim true?

Decoding used temperature 0, at most 20 output tokens and a thinking budget of 0 for Gemini, and at most 10 tokens for the Ollama judges. The first standalone “true” or “false” in the answer is the verdict. The simulated judge is correct with probability equal to its accuracy, decided by a hash of the seed, claim identifier and value, so the same claim always receives the same verdict within a seed.

## B Reproducibility

All runs used Python 3.11, PyTorch 2.6.0 (CUDA 12.4), transformers 5.20.1, peft 0.21.2 and datasets 5.1.0 on one 16 GB laptop GPU in bfloat16. Students were loaded from Hugging Face (Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct, meta-llama/Llama-3.2-3B-Instruct). CounterFact was loaded from azhx/counterfact (train split shufled with seed 0, first 3,000 records, screened for facts each student knows). The repository (https://github.com/Metalesaek/sura-metales) contains all code, configurations and saved results. The script reproduce\_paper.py recomputes all tables and figures from the saved results without retraining and compares 407 numbers quoted in this paper with the recomputed values.