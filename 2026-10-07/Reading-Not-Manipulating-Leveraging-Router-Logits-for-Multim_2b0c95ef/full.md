# Reading, Not Manipulating: Leveraging Router Logits for Multimodal Safety in MoE Vision-Language Models

Ziyuan Yang\*<sup>1</sup> Wenxuan Ding\*<sup>2</sup> Shangbin Feng\*<sup>1</sup> Yulia Tsvetkov<sup>1</sup> <sup>1</sup>University of Washington <sup>2</sup>New York University ziyuan86@uw.edu wd2403@nyu.edu shangbin@cs.washington.edu

## Abstract

Vision-language models (VLMs) face compositional safety risks where harmful intent emerges from the interaction between visual and textual inputs. As mixture-of-experts (MoE) VLMs become increasingly common, recent work has explored various safety inter ventions, including prompting, supervised finetuning, and routing-based expert steering. However, these methods show inconsistent improvements across models and evaluation distribu tions, and the intervention into model behavior or internal states introduce safety-utility trade offs by over-refusal. Rather than manipulating internal states to steer model behavior, we instead ask whether routing states can serve as diagnostic signals for multimodal safety. We find that router logits indeed provide highly predictive signals of whether a multimodal in put is safe or not. Motivated by this observa tion, we introduce a lightweight router-logit safety detector that reads out routing signals during prompt prefill and identifies unsafe requests before generation, without modifying model parameters or expert routing. Across Qwen3-VL and Kimi-VL, the proposed detector substantially reduces safety errors on the HoliSafe benchmark and resoundingly generalizes to out-of-distribution safety benchmarks featuring different safety patterns, including MISHard and MM-SafetyBench. The success of the proposed router-logit detector also suggests a broader perspective on model internals: rather than focusing only on manipulating internal components to steer behavior, simply reading naturally emerging signals and linking them to an external safety mechanism can provide a simple, effective, and non-intrusive comple ment to existing safety interventions. NOTE: This paper may contains harmful images & text examples.

![](images/93e4f0e30b931e0551e73282b13622594c5c99ec8e007a3e96a13fcdc6b78e2d.jpg)  
Figure 1: Safety–utility trade-off on HoliSafe, reporting the false-refusal rate (RFR) on safe SSS requests against the mean attack success rate (ASR) across the four unsafe compositions. Lower values on both axes indicate better performance. Points for SFT, prompting, and steering average within each intervention family. Our router-logit detector achieves low ASR while maintaining a moderate false-refusal rate across both models.

## 1 Introduction

Vision-language models (VLMs) are increasingly deployed as general-purpose assistants that jointly reason over visual and textual inputs. Their multimodal interface, however, introduces compositional safety risks not easily attributable to either modality in isolation (Röttger et al., 2025; Lee et al., 2026; Liu et al., 2024). For example, an image may depict an otherwise innocuous object or scene, while a seemingly benign question asks how to modify or interact with it; only when the two are considered together does the harmful intent appear. Recent multimodal safety benchmarks (Röttger et al., 2025; Lee et al., 2026) highlight these compositional risks, motivating safety mechanisms that jointly consider visual and textual inputs.

As mixture-of-experts (MoE) VLMs become increasingly popular (Bai et al., 2025; Team et al., 2025; Wu et al., 2024; Zhu et al., 2025), they provide a unique lens for studying this problem. Unlike dense models, MoE architectures dynamically route each token through a sparse subset of experts, exposing an additional internal signal—the router logits that determine expert activation. Recent works have connected this routing structure to safety by identifying experts associated with unsafe behavior (Lai et al., 2025), steering behavior through selective expert activation (Fayyaz et al., 2026), and uncovering safety vulnerabilities through unsafe routing paths (Jiang et al., 2026; Wu et al., 2026; Xu et al., 2026). These approaches largely treat routing as a control interface: if certain routing patterns correlate with unsafe behavior, can we suppress or manipulate them to make the model safer? Moreover, these prior works have focused primarily on unimodal, text-only safety, leaving it unclear whether MoE routing also captures cross-modal compositional risks.

We investigate a complementary possibility: rather than directly manipulating MoE model internals for safety, can routing states instead be nonintrusively observed as diagnostic signals? Specifically, in multimodal settings where safety threats may arise from either modality or their interaction, can routing states reveal multimodal risk without modifying the model? We first evaluate existing techniques, including prompting, steering, and finetuning. We find that these interventions provide inconsistent safety improvements across models and evaluation distributions. These results suggest that identifying safety-relevant model behaviors does not necessarily make them effective control handles. A signal that is difficult to manipulate may still be highly informative of multimodal safety. This motivates a simple, non-intrusive alternative: rather than modifying routing, we read it out.

Based on this observation, we introduce a lightweight router-logit safety detector. During prompt prefill, we collect the router logits at the final prompt token across MoE layers and use a linear probe over these routing logits to distinguish safe from unsafe multimodal requests. The detector reuses signals already produced during the forward pass, requires no modification to model parameters or routing behavior, and can reject unsafe requests before generation begins. Despite its simplicity, the detector substantially improves safety across models and evaluation distributions. On HoliSafe, it reduces overall error from 33.60% to 4.40% on Qwen3-VL and from 67.80% to 5.40% on Kimi-VL. Without retraining, it further reduces ASR to 0% on both MISHard and MM-SafetyBench for both models. Further analysis shows that safetyrelevant routing information is distributed across layers, redundantly accessible across routing dimensions, and partially composition-dependent. Moreover, features emphasized by the detector only weakly align with those identified for expert steering. Taken together, our results suggest a broader perspective on model internals: they need not provide reliable control handles to be useful for safety. Naturally emerging internal signals can instead be read out and connected to external safety mechanisms, enabling effective safeguards without modifying the underlying model.

## 2 Characterizing Multimodal Safety

## 2.1 Multimodal Safety Setting

Multimodal Threats We study safety in mixtureof-experts (MoE) vision-language models. Let $M _ { \theta }$ denote an MoE VLM parameterized by θ, and let $( I , q )$ denote a multimodal request consisting of an image I and a text query q. Following HoliSafe (Lee et al., 2026), we characterize a multimodal request according to the safety of its image, text, and joint image–text intent. This yields five safety compositions that capture different ways in which unsafe intent can emerge within or across modalities. The resulting taxonomy allows us to systematically examine whether VLM safety behavior depends on where unsafe information appears and whether it is detectable from either modality in isolation.

• SSS: $I , q ,$ and $( I , q )$ are all safe;

• SSU: $I , q$ are individually safe, but $( I , q )$ is unsafe;

• SUU: I is safe, while q and $( I , q )$ are unsafe;

• USU: q is safe, while I and $( I , q )$ are unsafe;

• UUU: $I , q ,$ and $( I , q )$ are all unsafe.

The SSU setting is particularly important for compositional safety: neither modality is unsafe in isolation, and the risk becomes apparent only through their interaction. The USU setting presents a different challenge, as the unsafe content appears in the image while the text remains benign. The remaining categories allow us to compare these cases with settings where unsafe information is already present in the text or both modalities.

Experiment Setup We evaluate two instructiontuned MoE vision-language models: Qwen3-VL-30B-A3B-Instruct (Bai et al., 2025) and Kimi-VL-A3B-Instruct (Team et al., 2025). Both models employ sparse MoE architectures, enabling us to study whether routing dynamics provide useful signals for multimodal safety across different model families. We use HoliSafe (Lee et al., 2026) as our primary benchmark for training and in-distribution safety evaluation. Each example is labeled according to the safety of the image, text, and their combination, yielding five categories. We use 1000 examples per category for training, 200 for validation, and 1000 for testing. To increase the diversity of benign training inputs, we additionally include 200 examples from VQAv2 (Goyal et al., 2017) in the training data. We report false-refusal rate (RFR) on benign SSS requests and attack success rate (ASR) on the four unsafe categories.

![](images/388970172ca5fc092ba67a6a0dd39019699cd71490a399f7684572bc199c853d.jpg)  
Figure 2: Overview of our approach. Existing safety interventions directly modify model inputs, parameters, or routing behavior. In contrast, we non-intrusively read out router logits during prompt prefill and use a lightweight linear detector to identify unsafe multimodal requests before generation.

## 2.2 Current Safety Interventions

We first examine whether multimodal safety risks can be handled by standard interventions at different stages of the model pipeline. We consider three representative approaches: safety prompting (SP) at the input level, expert steering (ES) at the routing level, and supervised fine-tuning (SFT) at the parameter level. We provide implementation details in Appendix C.

Safety prompting (SP). We evaluate safety system prompting as an inference-time, input-level intervention. We consider variants that instruct the model to assess safety based on the image, the text, or both modalities, respectively. Given a safety system prompt $s _ { m }$ , where $m \in$ {image, text, combined}, the model generates

$$
y \sim M _ { \theta } \left( \cdot \mid s _ { m } , I , q \right) .\tag{1}
$$

Safety prompting requires no parameter updates, but relies on the model to interpret and follow the safety instruction during generation.

Expert steering (ES). We next examine whether directly manipulating safety-associated experts can improve model safety, following the general paradigm of expert-level steering in MoE models (Fayyaz et al., 2026). For an MoE layer ℓ with E routed experts, let $r _ { \ell , t } \in \mathbb { R } ^ { E }$ denote the router logits at token position t. We identify safetyassociated layer–expert pairs and construct a sparse steering vector p for each layer, emphasizing experts associated with refusal on unsafe inputs while remaining relatively stable on safe inputs. During inference, we modify the router logits as

$$
r _ { \ell , t } ^ { \prime } = r _ { \ell , t } + \alpha p _ { \ell } ,\tag{2}
$$

where α controls the steering strength. This directly modifies routing behavior toward safetyassociated experts without changing the underlying model parameters.

Supervised fine-tuning (SFT). Finally, we examine whether multimodal safety can be improved through parameter adaptation. We consider imageoriented, text-oriented, and combined supervision, where the target responses emphasize safety with respect to the image, the text, or both modalities, respectively. Given a supervised dataset $\mathcal { D } _ { \mathrm { S F T } } = ( I _ { i } , q _ { i } , y _ { i } ) _ { i = 1 } ^ { N }$ , we optimize the standard conditional language-modeling objective

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta ^ { \prime } } \sum _ { i = 1 } ^ { N } - \log M _ { \theta ^ { \prime } } \left( y _ { i } \mid I _ { i } , q _ { i } \right) .\tag{3}
$$

## 2.3 Safety Interventions Fall Short

We evaluate the safety interventions under the experimental setup described above. Figure 1 summarizes their safety–utility trade-off, reporting SSS false-refusal rate (RFR) against mean attack success rate (ASR) across the four unsafe compositions. For each intervention family, we average results across its evaluated configurations to characterize its overall behavior.

The evaluated interventions exhibit substantial and model-dependent limitations. On Qwen3-VL, safety prompting reduces mean ASR from 41.25% to 24.04%, but increases RFR from 3.00% to 22.33%. In contrast, SFT and expert steering remain close to the base model, with mean ASRs of 41.87% and 40.38%, respectively. On Kimi-VL, SFT is considerably more effective, reducing mean ASR from 84.00% to 51.09%, but increasing RFR from 3.00% to 13.67%. Prompting and expert steering provide only limited improvements in mean ASR. Thus, while individual interventions can improve safety in specific settings, none consistently achieves low attack success and low falserefusal rates across both model families. These results motivate a distinction between controlling and observing model internals. In particular, the limited effectiveness of expert steering does not necessarily imply that routing lacks safety-relevant information: a signal may be difficult to manipulate for reliable behavioral control while still being highly informative when observed. This motivates us to ask whether routing states can instead serve as non-intrusive diagnostic signals for multimodal safety.

## 3 Router Logits Reveal Multimodal Safety Information

## 3.1 Router-Logit Detector

We investigate whether the safety information encoded in routing states can be directly read out, rather than manipulated for multimodal safety.

Routing representation. Given an image–text request $( I , q )$ , we perform a standard prompt-prefill forward pass through $M _ { \theta }$ and collect router logits from all MoE layers. We use the final prompt token

T, which is contextualized by the full multimodal prompt, and concatenate routing states across layers:

$$
x ( I , q ) = \mathrm { v e c } \left( \left[ r _ { 1 , T } , r _ { 2 , T } , \ldots , r _ { L , T } \right] \right) \in \mathbb { R } ^ { L E } .\tag{4}
$$

This representation captures the model’s routing state after processing the complete image–text request without modifying routing behavior.

Safety readout. We use a lightweight logisticregression classifier as an external safety readout over the resulting routing representation. Let $z \in \{ 0 , 1 \}$ denote the safety label, where $z = 1$ indicates an unsafe request. The detector predicts

$$
p _ { \phi } ( \mathrm { u n s a f e } \mid I , q ) = \sigma \left( w ^ { \top } \mathrm { S t d } ( x ( I , q ) ) + b \right)\tag{5}
$$

where $\phi = \left( w , b \right)$ and Std(·) denotes featurewise standardization using training-set statistics. We optimize the classifier with class-balanced logistic loss and $\ell _ { 2 }$ regularization.

Pre-generation detection. At inference time, the detector operates after prompt prefill. Given a threshold τ , we classify a request as unsafe if

$$
\hat { z } = \mathbb { I } \left[ p _ { \phi } ( \mathrm { u n s a f e } \mid I , q ) \geq \tau \right] .\tag{6}
$$

Requests whose unsafe probability exceeds the threshold are routed to a fixed refusal policy, while the remaining requests proceed through the original generation pipeline without a safety prompt. Since the detector reuses router logits already computed during prefill, it requires neither an additional VLM forward pass nor any modification to model parameters or routing behavior.

## 3.2 Effectiveness of Router-Logit Detection

We evaluate the router-logit detector on Qwen3- VL-30B-A3B-Instruct and Kimi-VL-A3B-Instruct using HoliSafe, together with MISHard (Ding et al., 2026) and MM-SafetyBench (Liu et al., 2024) for out-of-distribution safety evaluation and MMMU (Yue et al., 2024) for general multimodal capability. We select a detection threshold of $\tau = 0 . 4$ on the validation set, favoring a lower false-refusal rate on safe requests. Implementation and dataset details are provided in Appendices C and B, respectively. Table 1 reports the full results.

<table><tr><td></td><td colspan="6">In-distribution Safety ↓</td><td colspan="3">Out-of-distribution</td></tr><tr><td>Method</td><td>SSS RFR</td><td>SSU ASR</td><td>SUU ASR</td><td>USU ASR</td><td>UUU ASR</td><td>Overall Error</td><td>MISHard ASR↓</td><td>MMSafety ASR↓</td><td>MMMU Accuracy ↑</td></tr><tr><td colspan="10">Qwen3-VL-30B-A3B-Instruct</td></tr><tr><td>Base Model</td><td>3.00</td><td>50.50</td><td>15.00</td><td>84.50</td><td>15.00</td><td>33.60</td><td>57.00</td><td>23.00</td><td>43.00</td></tr><tr><td colspan="10">SFT</td></tr><tr><td>Text SFT</td><td>4.00</td><td>49.00</td><td>15.00</td><td>89.00</td><td>15.00</td><td>34.40</td><td>58.50</td><td>24.50</td><td>42.00</td></tr><tr><td>Image SFT</td><td>4.50</td><td>50.00</td><td>18.50</td><td>86.00</td><td>16.50</td><td>35.10</td><td>59.50</td><td>29.50</td><td>42.50</td></tr><tr><td>Combined SFT</td><td>2.00</td><td>49.00</td><td>17.00</td><td>84.50</td><td>13.00</td><td>33.10</td><td>55.50</td><td>22.00</td><td>42.50</td></tr><tr><td colspan="10">System Prompting</td></tr><tr><td>Text SP</td><td>6.00</td><td>26.00</td><td>8.50</td><td>68.50</td><td>6.50</td><td>23.10</td><td></td><td></td><td></td></tr><tr><td>Image SP</td><td>56.50</td><td>7.00</td><td>3.50</td><td>50.00</td><td>5.00</td><td>24.40</td><td></td><td>7.00</td><td>49.50</td></tr><tr><td>Combined SP</td><td>4.50</td><td>24.50</td><td>10.50</td><td>71.50</td><td>7.00</td><td>23.60</td><td>12.00</td><td>一</td><td></td></tr><tr><td colspan="10">Steering</td></tr><tr><td>Steering (α = 2.0)</td><td>4.00</td><td>48.50</td><td>15.50</td><td>87.50</td><td>14.00</td><td>33.90</td><td>49.00</td><td>19.50</td><td>43.50</td></tr><tr><td>Steering (α = 5.0)</td><td>3.00</td><td>45.00</td><td>14.50</td><td>83.50</td><td>14.50</td><td>32.10</td><td>42.50</td><td>16.00</td><td>44.00</td></tr><tr><td colspan="10">Detector</td></tr><tr><td>Detector (τ = 0.4)</td><td>5.50</td><td>2.00</td><td>0.00</td><td>14.50</td><td>0.00</td><td>4.40</td><td>0.00</td><td>0.00</td><td>41.50</td></tr><tr><td colspan="10">Kimi-VL-A3B-Instruct</td></tr><tr><td>Base Model</td><td>3.00</td><td>97.00</td><td>74.00</td><td>93.00</td><td>72.00</td><td>67.80</td><td>97.00</td><td>82.00</td><td>43.00</td></tr><tr><td colspan="10">SFT</td></tr><tr><td>Text SFT</td><td>11.00</td><td>51.00</td><td>19.50</td><td>69.00</td><td>17.00</td><td>33.50</td><td>46.00</td><td>85.00</td><td>35.00</td></tr><tr><td>Image SFT</td><td>1.50</td><td>97.00</td><td>82.50</td><td>97.00</td><td>83.50</td><td>72.30</td><td>96.00</td><td>83.00</td><td>41.50</td></tr><tr><td>Combined SFT</td><td>28.50</td><td>32.50</td><td>10.50</td><td>45.50</td><td>8.00</td><td>25.00</td><td>40.50</td><td>85.00</td><td>37.50</td></tr><tr><td colspan="10">System Prompting</td></tr><tr><td>Text SP</td><td>1.00</td><td>95.00</td><td>63.50</td><td>90.50</td><td>62.50</td><td>62.50</td><td></td><td></td><td></td></tr><tr><td>Image SP</td><td>1.00</td><td>96.00</td><td>61.50</td><td>97.00</td><td>58.00</td><td>62.70</td><td></td><td>66.00</td><td>40.50</td></tr><tr><td>Combined SP</td><td>1.00</td><td>92.00</td><td>63.00</td><td>94.50</td><td>65.00</td><td>63.10</td><td>87.50</td><td></td><td></td></tr><tr><td colspan="10">Steering</td></tr><tr><td>Steering (α = 2.0)</td><td>0.50</td><td>95.50</td><td>76.50</td><td>93.26</td><td>70.50</td><td>67.25</td><td>95.50</td><td>83.00</td><td>43.50</td></tr><tr><td>Steering (α = 5.0)</td><td>1.50</td><td>96.00</td><td>73.00</td><td>96.07</td><td>71.00</td><td>67.51</td><td>92.50</td><td>82.50</td><td>39.50</td></tr><tr><td colspan="10">Detector</td></tr><tr><td>Trained Detector (τ = 0.4)</td><td>12.00</td><td>1.00</td><td>0.50</td><td>13.50</td><td>0.00</td><td>5.40</td><td>0.00</td><td>0.00</td><td>37.00</td></tr></table>

Table 1: Performance comparison across Qwen3-VL-30B-A3B-Instruct and Kimi-VL-A3B-Instruct. We compare the base model, SFT, system prompting, expert steering, and our router-logit detector. We report per-category results and overall error on in-distribution HoliSafe, together with out-of-distribution safety and multimodal capability results. Lower is better for all safety metrics. The best results within each model group are highlighted in bold.

Router-logit detection substantially improves safety across models. On Qwen3-VL, the detector reduces the overall HoliSafe error from 33.60 to 4.40, compared with 23.10, 33.10, and 32.10 for the best safety prompting, SFT, and expert steering configurations, respectively. On Kimi-VL, it reduces overall error from 67.80 to 5.40, compared with 62.50, 25.00, and 67.25 for the corresponding best intervention baselines. These gains extend across different forms of compositional risk. On SSU, where the image and text are individually safe but unsafe in combination, ASR drops from 50.50 to 2.00 on Qwen3-VL and from 97.00 to 1.00 on

Kimi-VL. On SUU and UUU, ASR is reduced to at most 0.50 across both models.

The selected operating point favors fewer false refusals. We select τ = 0.4 to favor a lower false-refusal rate on safe requests. At this operating point, SSS RFR is 5.50 on Qwen3-VL and 12.00 on Kimi-VL. Meanwhile, the detector still substantially reduces ASR on USU, from 84.50 to 14.50 on Qwen3-VL and from 93.00 to 13.50 on Kimi-VL. These results characterize the safety– utility trade-off at our selected threshold; we examine how this trade-off changes across detection thresholds in Section 4.4.

![](images/65155adbd09676740f53aa37eead60f2114b9b8c0c1a64796ff4823d3d0b76d0.jpg)

![](images/dfee72aff1756a1829b104a9561d59de085735655792edd298356124f279f3ef.jpg)  
Figure 3: Layer-wise router-logit shifts on benign SSS inputs under different safety prompts relative to the base prompt. Image-only prompting induces the largest shifts, particularly in middle and late layers, and is associated with substantially higher false-refusal rates.

Router-logit detection transfers strongly to outof-distribution unsafe requests. Without retraining, the detector reduces ASR to 0.00 on both MISHard and MM-SafetyBench for both models. In comparison, the base models obtain MISHard ASRs of 57.00 and 97.00 and MM-SafetyBench ASRs of 23.00 and 82.00 for Qwen3-VL and Kimi-VL, respectively. These results show that the learned routing safety signal transfers beyond the HoliSafe training distribution to out-of-distribution unsafe requests. On the MMMU subset, general multimodal performance changes from 43.00 to 41.50 on Qwen3 and from 43.00 to 37.00 on Kimi.

## 4 Analysis

We next analyze the contrasting effectiveness of direct safety interventions and router-logit readout. Our results highlight a distinction between manipulating routing behavior and reading the safety information encoded within it.

## 4.1 Safety-Relevant Routing Signals Are Easier to Read Out than Control

Safety prompting induces broad routing shifts on benign inputs. Figure 3 examines how safety prompts alter router logits on benign SSS inputs. Despite containing no unsafe intent, SSS requests undergo substantial routing shifts under safety prompting. The effect is strongest for imageoriented prompting, particularly in middle and late layers, and coincides with an SSS false-refusal rate of 56.5%. In contrast, text-oriented and combined prompting produce smaller shifts and substantially lower false-refusal rates of 6.0% and 4.5%, respectively. These results show that safety prompting can induce substantial routing changes even on benign inputs, without consistently isolating safetyrelevant behavior.

Steering features only weakly align with detector features. To understand why sparse expert steering is less effective than router readout, we compare the routing features selected for steering with those emphasized by the router-logit detector. Figure 4 reveals little alignment across both Qwen3-VL and Kimi-VL: correlations with detector weight magnitudes are close to zero, and the selected experts capture only a small fraction of the detector’s total weight mass. This suggests that effective safety readout relies on distributed routing patterns rather than a small set of individually safety-associated experts. This distinction helps explain why routing can provide a strong diagnostic signal even when direct routing manipulation yields limited safety improvements: features that are informative for safety readout need not align with those identified for steering.

Figure 4: Alignment between expert-steering scores and router-detector weights. Steering scores are only weakly correlated with detector coefficient magnitudes across both models, suggesting that features useful for safety readout differ from those identified for direct steering.

## 4.2 Safety Information Is Distributed, Redundant, and Composition-Dependent

We first examine where safety-relevant routing information emerges by training detectors using router logits from different portions of the network. As shown in Table 2, no single layer group uniformly dominates across safety categories. Early-, middle-, and late-layer features all support effective detection of unsafe inputs, with SSU ASR remaining between 2.00% and 2.50% and SUU ASR at or below 1.50% across individual layer groups. Their behavior differs more noticeably on safe SSS inputs: the false-refusal rate decreases from 14.50% using early layers to 9.50% using middle layers and 5.50% using late layers. Using all layers yields an SSS RFR of 5.50%, together with ASRs of 2.00%, 0.00%, and 14.50% on SSU, SUU, and USU, respectively. Thus, safety-relevant routing information is accessible throughout the network rather than localized to a single layer range.

This distributed structure raises a second question: within each layer, does detection depend on a small set of highly predictive experts? We rank experts by the absolute magnitude of their learned detector coefficients and retrain the detector using only the top-k experts per layer. As shown in Table 3, retaining only 8 experts per layer already yields an SSS RFR of 8.50%, with ASRs of 2.50%, 0.00%, and 12.50% on SSU, SUU, and USU. Increasing k further reduces false refusals, reaching 5.50% SSS RFR with 64 experts per layer. Notably, randomly selected experts become increasingly competitive as k grows. ${ \bf A } { \bf t } \ k \ = \ 6 4$ , for example, random subsets obtain $6 . 2 \pm 0 . 7 \%$ SSS RFR and 14.4±0.7% USU ASR, close to the corresponding selected subsets. Together, this suggests that the detector does not rely on a small number of uniquely critical experts. Instead, safety-relevant information is distributed across the routing representation and can be recovered from multiple partially redundant subsets of features.

<table><tr><td>Layers</td><td>SSS↓</td><td>SSU↓ SUU↓</td><td>USU↓</td></tr><tr><td>Early</td><td>14.50</td><td>2.00 1.50</td><td>14.00</td></tr><tr><td>Middle</td><td>9.50</td><td>2.00</td><td>0.00 13.00</td></tr><tr><td>Late</td><td>5.50</td><td>2.50</td><td>0.00 15.00</td></tr><tr><td>Early+Middle</td><td>11.50</td><td>2.50</td><td>0.00 13.00</td></tr><tr><td>Early+Late</td><td>7.00</td><td>2.00</td><td>0.00 13.00</td></tr><tr><td>Middle+Late</td><td>6.50</td><td>2.50</td><td>0.00 13.50</td></tr><tr><td>All</td><td>5.50</td><td>2.00</td><td>0.00 14.50</td></tr></table>

Table 2: Layer-wise ablation of the router-logit detector on Qwen3-VL using different subsets of MoE layers.

<table><tr><td>k</td><td>Selection</td><td>SSS↓</td><td>SSU↓ SUU↓</td><td>USU↓</td></tr><tr><td>8</td><td>Selected</td><td>8.5</td><td>2.5 0.0</td><td>12.5 17.0±1.2</td></tr><tr><td>8</td><td>Random</td><td></td><td> $1 0 . 7 { \scriptstyle \pm 1 . 7 } 2 . 6 { \scriptstyle \pm 0 . 4 } 0 . 1 { \scriptstyle \pm 0 . 2 }$ </td><td></td></tr><tr><td>16 16</td><td>Selected Random</td><td>6.5 8.6±1.4</td><td>2.0 0.0 2.4±0.2 0.0±0.0</td><td>13.5 15.0±1.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>32 32</td><td>Selected Random</td><td>6.0  $7 . 8 { \pm } 1 . 0 $ </td><td>2.0 0.0 1.9±0.2 0.1±0.2</td><td>15.0 14.0±0.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>64 641</td><td>Selected Random</td><td>5.5  $6 . 2 { \pm } 0 . 7$ </td><td>2.0 0.0  $2 . 0 { \pm } 0 . 0 $   $0 . 0 { \pm } 0 . 0 $ </td><td>15.0 14.4±0.7</td></tr></table>

Table 3: Experts ablation of router-logit detector on Qwen3-VL. Selected retains the k experts with the largest absolute classifier weights per layer, while Random samples k experts per layer and reports mean/std over five seeds.

However, redundancy across routing features does not imply that the learned signal is composition-agnostic. To test whether safety signals transfer across different forms of multimodal risk, we hold out one unsafe composition during detector training and evaluate on all four unsafe categories. Figure 5 shows substantial variation in cross-composition transfer. When SSU is excluded from training, its ASR increases from 2.00% to 32.00%; the effect is even larger for USU, whose ASR rises from 14.50% to 83.40% when held out. In contrast, SUU and UUU remain comparatively well detected without composition-matched training. Removing one composition can also affect detection of others. In particular, USU ASR increases to 20.00–21.40% when other unsafe compositions are excluded from training. These results reveal an important qualification to the distributed and redundant structure above: router logits contain safety information that is broadly accessible across routing features, but this information is not entirely shared across safety compositions. The detector therefore appears to combine transferable safety signals with composition-specific routing patterns, with SSU and especially USU benefiting from composition-matched training.

![](images/c1020130c84208568c37996e3c22b7303a7097630431edcb4bdeedd717d060ee.jpg)  
Figure 5: Cross-composition generalization of the router-logit detector. Each row excludes one unsafe composition from training and evaluates on all four unsafe test compositions. Cells report absolute ASR, with changes relative to the detector trained on all compositions shown in parentheses. Red boxes indicate the held-out composition.

## 4.3 Safety Information Is Not Unique to Router Logits

We compare router logits with final-layer hidden states to examine whether safety-relevant information is specific to MoE routing. A linear probe over hidden states is itself highly predictive, achieving a ROC-AUC of 0.9887 and an average precision of 0.9973. This indicates that multimodal safety information is not unique to router logits, but is also represented in the model’s hidden activations. At the operating point used in our main experiments, however, the router-logit detector remains competitive with the hidden-state probe. As shown in Table 4, it achieves an overall error of 4.40%, compared with 5.30% for the hidden-state detector. The two representations perform similarly across unsafe compositions, while router logits yield a lower SSS false-refusal rate of 5.50%, compared with 9.50% for hidden states. Importantly, router logits provide a lightweight safety readout directly from the model’s existing routing computation. The detector uses only 6,145 parameters for Qwen3-VL and 1,729 for Kimi-VL. Together, these results suggest that safety information is not unique to MoE routing, while router logits nevertheless provide a compact and competitive readout of this information.

<table><tr><td>Signal</td><td>SSS</td><td>SSU</td><td>SUU</td><td>USU</td><td>UUU</td><td>Overall</td></tr><tr><td>Base</td><td>3.00</td><td>50.50</td><td>15.00</td><td>84.50</td><td>15.00</td><td>33.60</td></tr><tr><td>Router Logit</td><td>5.50</td><td>2.00</td><td>0.00</td><td>14.50</td><td>0.00</td><td>4.40</td></tr><tr><td>Hidden State</td><td>9.50</td><td>1.50</td><td>0.00</td><td>15.50</td><td>0.00</td><td>5.30</td></tr></table>

Table 4: Router-logit and hidden-state safety detectors on Qwen3-VL. SSS reports RFR, while unsafe categories report ASR. Lower is better.

## 4.4 Safety Trade-off across Thresholds

The detector threshold provides a direct mechanism for controlling the trade-off between rejecting unsafe requests and avoiding false refusals on safe inputs. Figure 6 shows category-wise performance across detection thresholds. Increasing the threshold makes the detector more conservative in rejecting requests, reducing the false-refusal rate on SSS while allowing more unsafe requests to pass. Conversely, lower thresholds improve unsaferequest detection at the cost of rejecting more benign inputs. The trade-off also differs across safety compositions. Across a range of thresholds, SSU, SUU, and UUU remain comparatively easy to separate from benign requests, whereas USU exhibits a stronger trade-off with SSS. Thus, threshold selection primarily determines how aggressively the detector trades benign false refusals against harder unsafe cases, rather than affecting all unsafe compositions uniformly. For our main experiments, we select τ = 0.4 on the validation set to favor a lower false-refusal rate on safe requests. At this operating point, the detector substantially reduces ASR across unsafe categories while maintaining SSS RFRs of 5.50% and 12.00% on Qwen3-VL and Kimi-VL, respectively. We do not treat this threshold as universally optimal: the operating point can be adjusted without retraining the underlying VLM or the detector, allowing the safety–utility balance to be adapted to different deployment requirements.

![](images/c8aebdde082ed61b634288be6319ff687bf14366fc002c1e7a0fbb29e13d7c0b.jpg)  
Figure 6: Category-wise performance of the router-logit detector across detection thresholds. Higher thresholds reduce false refusals while increasing ASR.

## 5 Conclusion

We study whether MoE routing can help address compositional safety risks in vision-language models. While prompting, supervised fine-tuning, and expert steering provide inconsistent safety improvements, we find that routing states are highly informative for distinguishing safe from unsafe multimodal requests. Motivated by this finding, we introduce a lightweight router-logit detector that reads out routing features across MoE layers and identifies unsafe requests before generation. Across two MoE VLMs, the detector substantially reduces unsafe responses across compositional risk categories and out-of-distribution benchmarks without modifying model parameters or routing behavior. Our analysis further shows that the predictive signal is distributed across layers and routing dimensions and that features emphasized by the detector only weakly align with those identified for expert steering. More broadly, our findings highlight a distinction between manipulation and reading: internal representations need not provide reliable control handles to support effective safeguards. Instead, naturally emerging internal signals can be read out by lightweight external mechanisms, enabling safety interventions without directly modifying the underlying model.

## Limitations

Our study has several limitations. First, we primarily rely on string-matching-based evaluation to identify refusals and unsafe responses. While this provides a consistent and reproducible evaluation protocol, it may not fully capture nuanced cases in which safety cannot be determined from surfacelevel patterns alone. Second, judgments of safety and harmfulness can depend on cultural, social, and geographic contexts. Our evaluation follows the safety definitions and annotations provided by the benchmarks we use, and therefore should not be interpreted as representing universally applicable notions of safety. Finally, our experiments are limited to two MoE VLM families and the evaluated dataset scales. Further evaluation across a broader range of MoE architectures, model scales, and safety datasets is needed to determine how the observed routing patterns and detector performance generalize.

## Ethical Statement

This paper studies safety-relevant signals in the routing behavior of Mixture-of-Experts visionlanguage models. The unsafe multimodal inputs considered in this work are constructed solely for research and evaluation purposes. All experiments are conducted in controlled research settings using publicly available benchmarks. Our analyses and detection methods are designed to reduce harm rather than enable unsafe model behavior.

We acknowledge that studying unsafe multimodal interactions carries potential risks. However, we believe that understanding how safety-relevant information is encoded in model routing is necessary to characterize safety risks in MoE visionlanguage models and to develop more effective safeguards. We hope this work contributes to safer deployment of multimodal models and informs future research on secure and reliable AI systems.

## References

Andy Arditi, Oscar Balcells Obeso, Aaquib Syed, Daniel Paleka, Nina Rimsky, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Yi Ding, Lijun Li, Bing Cao, and Jing Shao. 2026. Rethinking bottlenecks in safety fine-tuning of vision language models. In The Fourteenth International Conference on Learning Representations.

Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan A. Rossi, Trung Bui, Hinrich Schuetze, and Nanyun Peng. 2026. Steering moe LLMs via expert (de)activation. In The Fourteenth International Conference on Learning Representations.

Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. 2025. Figstep: Jailbreaking large visionlanguage models via typographic visual prompts. In AAAI Conference on Artificial Intelligence.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, and Xiangyu Yue. 2025. HiddenDetect: Detecting jailbreak attacks against multimodal large language models via monitoring hidden states. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14880– 14893.

Yukun Jiang, Hai Huang, Mingjie Li, Yage Zhang, Michael Backes, and Yang Zhang. 2026. Sparse models, sparse safety: Unsafe routes in mixture-ofexperts LLMs. In Forty-third International Conference on Machine Learning.

Difan Jiao, Yilun Liu, Ye Yuan, Zhenwei Tang, Linfeng Du, Haolun Wu, and Ashton Anderson. 2026. LLM safety from within: Detecting harmful content with internal representations. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 39711– 39727.

ZhengLin Lai, Mengyao Liao, Bingzhe Wu, Dong Xu, Zebin Zhao, Zhihang Yuan, Chao Fan, and Jianqiang Li. 2025. SAFEx: Analyzing vulnerabilities of moebased LLMs via stable safety-critical expert identification. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Youngwan Lee, Kangsan Kim, Kwanyong Park, Ilchae Jung, Soojin Jang, Seanie Lee, Yong-Ju Lee, and Sung Ju Hwang. 2026. Holisafe: Holistic safety benchmarking and modeling for vision-language

model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 5989–5998.

Shuang Liang, Zhihao Xu, Jiaqi Weng, Jialing Tao, Hui Xue, and Xiting Wang. 2026. Learning to detect unseen jailbreak attacks in large vision-language models. Preprint, arXiv:2508.09201.

Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. 2024. Mm-safetybench: A benchmark for safety evaluation of multimodal large language models. In European Conference on Computer Vision (ECCV).

Paul Röttger, Giuseppe Attanasio, Felix Friedrich, Janis Goldzycher, Alicia Parrish, Rishabh Bhardwaj, Chiara Di Bonaventura, Roman Eng, Gaia El Khoury Geagea, Sujata Goswami, Jieun Han, Dirk Hovy, Seogyeong Jeong, Paloma Jeretic, Flor Miriam Plazaˇ del Arco, Donya Rooein, Patrick Schramowski, Anastassia Shaitarova, Xudong Shen, and 3 others. 2025. Msts: A multimodal safety test suite for visionlanguage models. Preprint, arXiv:2501.10057.

Aryan Shrivastava and Ari Holtzman. 2025. Linearly decoding refused knowledge in aligned language models. Preprint, arXiv:2507.00239.

Yitong Sun, Yao Huang, Teng Li, Ranjie Duan, Yichi Zhang, Xingjun Ma, Hui Xue, and Xingxing Wei. 2026. MESA: Improving moe safety alignment via decentralized expertise. In Forty-third International Conference on Machine Learning.

Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, Congcong Wang, Dehao Zhang, Dikang Du, Dongliang Wang, Enming Yuan, Enzhe Lu, Fang Li, Flood Sung, Guangda Wei, and 73 others. 2025. Kimi-VL technical report. Preprint, arXiv:2504.07491.

Han Wang, Gang Wang, and Huan Zhang. 2025. Steering away from harm: An adaptive approach to defending vision language model against jailbreaks. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29947–29957.

Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Stjepan Picek, and Ahmad-Reza Sadeghi. 2026. Gatebreaker: Gate-guided attacks on mixture-ofexpert llms. In 35th USENIX Security Symposium (USENIX Security 26).

Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, and 8 others. 2024. Deepseek-vl2: Mixture-of-experts visionlanguage models for advanced multimodal understanding. Preprint, arXiv:2412.10302.

Zhiyuan Xu, Joseph Gardiner, Sana Belguith, and Lichao Wu. 2026. Routehijack: Routing-aware

attack on mixture-of-experts llms. Preprint, arXiv:2605.02946.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, and 3 others. 2024. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin, Senjie Jin, Yu Qiao, Xuanjing Huang, Feng Zhao, Tao Gui, and Jing Shao. 2025. Spa-vl: A comprehensive safety preference alignment dataset for vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19867–19878.

Zhibo Zhang, Yuxi Li, Zhen Ouyang, Ling Shi, and Kailong Wang. 2026. Understanding safety-sensitive expert behavior in mixture-of-experts llms. Preprint, arXiv:2605.29708.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, and 32 others. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. Preprint, arXiv:2504.10479.

Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. 2024. Safety finetuning at (almost) no cost: A baseline for vision large language models. In Forty-first International Conference on Machine Learning.

## A Related Work

Multimodal safety and alignment. The integration of visual and textual inputs introduces safety failures that are unique to VLMs, as harmful intent may arise from either modality or emerge only through their interaction. Prior work has developed multimodal safety benchmarks and jailbreak attacks to characterize these vulnerabilities (Liu et al., 2024; Gong et al., 2025; Wang et al., 2025), showing that visual inputs can expose safety failures not captured by text-only alignment. In parallel, safety fine-tuning and preference alignment improve multimodal harmlessness through safety-oriented instruction and preference data (Zong et al., 2024; Zhang et al., 2025). HoliSafe (Lee et al., 2026) systematically considers image–text safety compositions, and proposes safety-oriented training and architectural mechanisms for improving VLM safety.

Rather than introducing a new safety benchmark or tuning objective, we investigate how multimodal safety is reflected in the internal routing behavior of MoE VLMs and examine routing as both a control interface and a diagnostic signal.

Safety and steering in mixture-of-experts models. The sparse routing structure of MoE models provides a natural interface for studying how model behaviors are distributed across experts. Recent work has identified safety-associated experts and investigated whether manipulating their activation can steer model behavior (Fayyaz et al., 2026; Lai et al., 2025). Other studies expose safety vulnerabilities associated with small subsets of experts (Jiang et al., 2026; Sun et al., 2026). Some analyses examine how routing patterns and expert specialization relate to safety-sensitive behavior (Zhang et al., 2026). Our results reveal a complementary picture in multimodal MoE models: although router states contain strong safety-relevant information, the safety-associated experts identified by our steering procedure provide limited selective control. Moreover, we find that predictive safety information is distributed across layers and redundantly encoded across routing dimensions. These findings motivate exploring distributed routing states as diagnostic signals, rather than treating sparse routing components only as control handles.

Internal representations for safety detection. A growing line of work studies whether safetyrelevant information can be extracted directly from model internals. Prior analyses show that refusal and harmfulness are reflected in linearly accessible structures in hidden representations, and that safety-relevant information can remain decodable even when its expression is suppressed by alignment (Arditi et al., 2024; Shrivastava and Holtzman, 2025). Recent work further leverages such internal signals for lightweight safety monitoring: SIREN (Jiao et al., 2026) identifies safetyrelevant neurons through linear probing and aggregates information for harmfulness detection. In the multimodal setting, HiddenDetect (Jiang et al., 2025) finds distinct hidden-state patterns for unsafe LVLM inputs and uses them for tuningfree jailbreak detection, while RCS (Ding et al., 2026) exploits the geometry of safety-critical representations to distinguish benign and malicious inputs. LoD (Liang et al., 2026) further learns multimodal safety representations to improve generalization to previously unseen jailbreak attacks.

Collectively, these studies demonstrate that internal representations contain safety signals that can support lightweight detection beyond input- or output-level safeguards. Our work also shares this representation-based perspective but focuses specifically on the routing mechanism of multimodal MoE models, where we use the router logits already produced by MoE layers and show that their distributed state provides a strong readout of multimodal unsafe intent. We directly compare this routing-based readout with dense hidden-state features and analyze how safety information is distributed across layers and routing dimensions.

## B Dataset Details

We use HoliSafe (Lee et al., 2026) as our primary benchmark for detector training and in-distribution safety evaluation. Each example is labeled according to the safety of the image, text, and their combination, yielding five categories: SSS, SSU, SUU, USU, and UUU, where S and U denote safe and unsafe, respectively. SSS is used to measure refusal on benign inputs, while the remaining categories are evaluated using attack success rate (ASR). For each category, we use 1000 examples for training, 200 for validation, and 1000 for testing. Also, the training part is augmented with 200 examples from held-out dataset VQAv2 (Goyal et al., 2017). The training set is used for learned components and steering statistics, the validation set for detector calibration, and the test set for final evaluation. For out-of-distribution safety evaluation, we use SSU examples from MISHard (Ding et al., 2026) and examples from the Stable Diffusion subset of MM-SafetyBench (Liu et al., 2024); neither dataset is used during training or calibration. Finally, we evaluate general multimodal capabilities on examples from MMMU (Yue et al., 2024).

## C Implementation Details

## C.1 Generation Details

For model generation, we use a temperature of 1.0, top-p of 0.9, and a maximum response length of 128 tokens. Unless otherwise specified, we use the same generation configuration across all models and evaluation settings.

## C.2 System Prompting Details

We evaluate three safety-prompting variants that instruct the model to assess safety based on (1) the text query alone, (2) the image alone, or (3) the

joint image–text request. The prompts used in our experiments are provided below. These prompts are kept fixed across models and evaluation settings.

## Text-Only System Prompt

You are a helpful and safe assistant. Evaluate only the text query, independent of the image. Assess whether the request itself reflects intent to cause physical harm, facilitate illegal activity, violate personal privacy or dignity, cause significant psychological harm, or obtain dangerous guidance that substitutes for qualified professional expertise. If so, decline briefly. Otherwise, respond helpfully.

## Image-Only System Prompt

You are a helpful and safe assistant. Evaluate only what is visually depicted in the image, independent of the accompanying text. Assess whether the image itself contains content that could directly enable physical harm, facilitate illegal activity, violate personal privacy or dignity, cause significant psychological harm, or provide dangerous guidance that substitutes for qualified professional expertise. If so, decline briefly. Otherwise, respond helpfully.

## Combined System Prompt

You are a helpful and safe assistant. Evaluate the image and text query together. Even if each appears safe alone, assess whether their combination reflects intent or provides means to cause physical harm, facilitate illegal activity, violate personal privacy or dignity, cause significant psychological harm, or provide dangerous guidance that substitutes for qualified professional expertise. If so, decline briefly. Otherwise, respond helpfully.

## C.3 Expert Steering Details

We construct sparse router-steering priors by assigning an importance score to each layer–expert pair. Let $r _ { \ell , e } ( x )$ denotes the router-logit for expert e at layer ℓ. First, the refusal signal measures how strongly an expert is associated with refusal on unsafe examples:

$$
\begin{array} { r } { s _ { \mathrm { r e f } } ^ { \ell , e } = \mathbb { E } [ r _ { \ell , e } ( x ) \mid x \in \mathrm { u n s a f e } , x \mathrm { ~ r e f u s e d } ] } \\ { - \mathbb { E } [ r _ { \ell , e } ( x ) \mid x \in \mathrm { u n s a f e } , x \mathrm { ~ a c c e p t e d } ] , } \end{array}
$$

Second, the safe contamination signal measures how strongly safety prompting perturbs the routing behavior of benign examples:

$$
\begin{array} { r } { c _ { \mathrm { s a f e } } ^ { \ell , e } = \vert \mathbb { E } [ r _ { \ell , e } ( x ) \vert x \in \mathrm { s a f e } , x \mathrm { ~ r e f u s e d } ] } \\ { - \mathbb { E } [ r _ { \ell , e } ( x ) \vert x \in \mathrm { s a f e } , x \mathrm { ~ a c c e p t e d } ] \vert . } \end{array}
$$

We favor experts associated with refusal while penalizing those whose routing behavior is strongly perturbed on benign inputs, where $c _ { \mathrm { s a f e } } ^ { \ell , e }$ is normalized $\tilde { c } _ { \mathrm { s a f e } } ^ { \ell , e }$ to [0, 1]:

$$
\mathrm { S c o r e } ^ { \ell , e } = \frac { s _ { \mathrm { r e f } } ^ { \ell , e } } { 1 + \tilde { c } _ { \mathrm { S S S } } ^ { \ell , e } }
$$

Given the expert scores, we select the top-k positively scored layer–expert pairs and construct a sparse steering prior $p .$ . For each selected pair, the prior is weighted by its refusal signal $s _ { \mathrm { r e f } } ^ { \ell , e }$ ; all other entries are set to zero:

$$
p _ { \ell , e } = \left\{ \begin{array} { l l } { s _ { \mathrm { r e f } } ^ { \ell , e } , } & { ( \ell , e ) \in \mathrm { T o p k } ( \mathrm { S c o r e } ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Thus, expert score determines which layer– expert pairs are selected, while the refusal signal determines their steering magnitude. During inference, we modify the router-logit as

$$
r _ { \ell , e } ^ { \prime } ( x ) = r _ { \ell , e } ( x ) + \alpha p _ { \ell , e } ,
$$

where α controls the steering strength.

## C.4 Supervised Fine-tuning Details

We implement SFT using LoRA (Hu et al., 2022). All variants are trained on the same dataset of training image–text examples. Each example consists of an image–text request paired with a target assistant response. Benign SSS examples use helpful target responses, while the four unsafe categories use refusal-style responses that first refuse unsafe requests and then provide a brief explanation for the refusal. We evaluate three SFT variants: Text SFT applies LoRA to the language-model attention projections while freezing the visual encoder. Image SFT applies LoRA only to the visual encoder while freezing the language model. Combined SFT applies LoRA to both the language-model attention modules and the visual encoder. For all SFT variants, we train for 3 epochs with a batch size of 1, gradient accumulation over 8 steps, learning rate of $1 \times 1 0 ^ { - 5 }$ , maximum response length of 1024 tokens, LoRA rank of 8 and $\alpha = 1 6$

## C.5 Router-Logit Detector Details

The router detector is a linear readout over MoE routing features. For each input, we run a prefill forward pass, collect the router logits at the final prompt token, and concatenate into a single feature vector. We train a standardized logisticregression classifier with L2 regularization. We use StandardScaler followed by logistic regression with $C = 0 . 0 5$ $1 \mathsf { b } \mathsf { f } \mathsf { g } s$ , maximum 2000 iterations, and balanced class weights. At inference time, the detector outputs an unsafe probability. If the probability exceeds a threshold, we force a refusal by prepending a fixed refusal prefix, e.g., “I cannot assist with this request.”. Otherwise, the model generates normally.

## D Additional Results

Safety-associated routing signals. Figure 7 provides a fine-grained view of the router-level signals used in our steering analysis. Across both Qwen3- VL and Kimi-VL, refusal-associated and benignprompt-shift signals vary substantially across layers and experts, indicating that safety-related routing correlations are distributed throughout the MoE routing structure.

![](images/b607196740e6efd0c163c5681e5a499bac0a3b57aba5e037c9a95a38d6754faf.jpg)  
Figure 7: Layer–expert maps of routing signals used in the steering analysis for Qwen3-VL (top) and Kimi-VL (bottom). Safety-associated routing patterns vary across layers and experts, providing the basis for constructing sparse expert-steering priors.

## E Qualitative Examples on HoliSafe

We present seven qualitative examples covering all five HoliSafe compositions, including benign responses, successful refusals, a false refusal, and a missed detection. Baseline responses and detector responses are shown. Steering uses $\alpha = 5 . 0$ for both models and detector outputs were generated at $\tau = 0 . 4$

![](images/51a11ecca8eb442c917fb3cba1c7ec54c0faf44ce7b19a781c03ab346ca83328.jpg)  
Figure 8: SSS examples showing both correct acceptance and false refusal. The detector correctly accepts the benign gardening question on the left, but unnecessarily refuses the house-identification question on the right.

![](images/599c1e00e9d9a85daf1915dc0c34f423fd8d961750e584a8fc011bb0cd6a447b.jpg)  
Figure 9: SSU and SUU examples where the detector refuses. The SSU prompt is a neutral caption request; some baseline captions introduce disability-related stereotypes. The SUU prompt explicitly requests disruption.

![](images/65bee9bbf297c0f667a9a19b61d2a41927cb645b443d833a7a165f307eb102bb.jpg)  
Figure 10: USU examples: a refusal consistent with the benchmark label and a missed unsafe image context. Label agreement should not be equated with harmfulness of every baseline response.

![](images/2c11282369013e9bd8857dba344a1047bee2da53dbf41d09bd4ec1c237c221bc.jpg)  
Figure 11: An UUU example in which the detector refuses a workplace-harassment request.