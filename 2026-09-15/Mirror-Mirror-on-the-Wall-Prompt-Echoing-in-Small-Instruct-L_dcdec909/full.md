# Mirror, Mirror on the Wall: Prompt Echoing in Small Instruct Language Models

Inez Okulska<sup>1</sup>, Bartosz Naskręcki<sup>1,2</sup>, Jan Piotrowski<sup>1</sup>, Tomasz Steifer<sup>1,3</sup>,

<sup>1</sup>Centre for Credible AI, Warsaw University of Technology,

<sup>2</sup>Adam Mickiewicz University Poznan,

<sup>3</sup>Institute of Fundamental Technological Research, Polish Academy of Sciences. Correspondence: anakin65@gmail.com

## Abstract

Prompt echoing is a recognized failure mode of instruct language models, in which a model instead of generating a response, mirrors the provided prompt, even though it did not receive a specific instruction to do so. Is this phenomenon a sign of the model leaking the content of its training dataset, or is it rather caused by a misaligned behavior of the internal induction/copying mechanisms? We investigate prompt echoing small language models from diferent families (Gemma, Llama, Qwen, SmolLM and OLMo) and show that echoing prompts are likely to have partial overlap with the training dataset but the phenomenon is primarily driven by the model’s induction heads.

Modern language models exhibit a wide range of failure modes, from mundane breakdowns in instruction following, formatting, and calibration to adversarially induced behaviors such as prompt injection and jailbreaking. The latter are now widely recognized as central security concerns. Beyond these adversarial settings, however, models can also fail in ways that are not caused by an explicit attack. One such failure is prompt echoing, where the model responds by copying the input prompt, or a long prefix/span of it, despite receiving no instruction to do so.

Prompt echoing has appeared in prior evaluations as a recognizable class of generation error. For example, Seo et al. (2024) classify “prompt echoing error” as outputs that directly reflect the content of the prompt, and give examples where the model repeats prompt instructions instead of producing the intended concise response. More recently, evaluations of extended rule following also reported prompt echoes among natural failure modes (Dai and Fan, 2026). We use the term in this narrower sense and distinguish it from other recent use of “echo” in the literature, namely the intentional prompt-repetition techniques that improve answer quality (Leviathan et al., 2025; Hao et al.; Mekala et al.).

![](images/b57522af144aadf68dcc56c072fa16bf10d6dd6bf258022b7d9cc82b37cc0901.jpg)  
Figure 1: Two examples of prompt echoing and one of non-echo answer in which the model asks for clarification.

In this work, we study prompt echoing in small language models to understand its underlying mechanisms: rather than treating it only as a surface-level decoding artifact, we ask what internal model behaviors make echoing likely, and in particular, whether it is driven more by memorization or internal copying mechanisms.

Our contribution We investigate prompt echoing in several small language models from diferent families: Qwen (Yang et al., 2025; Qwen Team, 2026), Llama (Grattafiori et al., 2024; Meta Llama, 2024), Gemma (Gemma Team et al., 2024; Google DeepMind, 2025), OLMo (Groeneveld et al., 2024), and SmolLM (Ben Allal et al., 2025). Our general working hypothesis is that prompt echoing is a combination of text memorization induced in pretraining and finetuning and an intrinsic copying mechanism existing in modern language models. We provide empirical evidence for the following observations:

1. Prompt echoing happens across a diverse collection of language models, even under high temperature decoding (see Experiment 1).

2. Echogenic prompts are likely to have partial overlap with texts from the training dataset. We confirm this by investigating public training datasets available for OLMo and SmolLM models (see Experiment 2).

3. Once the echo is initialized, the continuation is carried out by the internal copying mechanism, akin to induction heads (Olsson et al., 2022). We confirm that by forcing echo on a dataset of non-echogenic prompts across several models (see Experiment 3).

4. Ablating attention heads with high copy scores significantly reduces echo across all models (see Experiment 4).

AI tools usage: This research was facilitated by responsible usage of LLM tools. In particular, ChatGPT and codex were used preparation of parts of the code as well as manuscript drafting, including LaTeX figure preparation, grammar editing and proof reading. The results and claims were independently verified by the authors. The experiments were conceived and designed by the authors. We take full responsibility for the context of the paper.

Related work Olsson et al. (2022) identified induction heads as attention heads implementing a prefix-matching and copying algorithm. These authors hypothesized that induction heads are a necessary component for in-context learning. McDougall et al. (2024) characterize a class of copy-suppression heads whose role is to attenuate, rather than amplify, the in-context copying signal.

Memorization and repetition Carlini et al. (2021) showed that language models reproduce verbatim spans of their pretraining data, and Carlini et al. (2023) establish loglinear relationships between memorization rate and model capacity, the number of duplicates in the training data, and the length of the prompting context.

## 1 Experimental setup

Reproducibility All the experiments were run on a cluster utilizing NVIDIA GH200 superchips. The code was written in Python and used PyTorch/Hugging Face Transformers for full-precision model inference, and llama.cpp for oficial GGUF quantized models. The datasets, download scripts, and evaluation code are publicly available at https://github.com/CredibleAI/Echo.

## 1.1 Data preparation.

For our experiments we have constructed three sets of prompts: two datasets of echogenic prompts and one dataset of non-echogenic prompts.

Dataset A For each source model, we first asked the model itself to generate standalone text under a diverse set of lightweight tasks, e.g. explanatory paragraphs, code snippets, mathematical explanations, documentationstyle examples, JSON-like configurations, tutorials, and problem statements. The generated text was then treated as a candidate prompt. We filtered out candidates that were too short (less than 24 tokens) or too long (more than 192 tokens).

Each candidate was then screened by prompting the same model with that candidate text and sampling 10 continuations using non-greedy decoding (temperature 0.9). A candidate was retained if at least one sampled continuation contained a contiguous span of at least 40% of the prompt tokens. We repeated this procedure until we obtained 64 accepted prompts for each of six source models. This yielded a 384-prompt dataset with known source model labels.

Dataset B was constructed using recursive loop of each model response becoming its next prompt. For each of the six models, we initialized 64 independent chains with short synthetic user requests drawn from a fixed set of topic/task templates. In the base step, the generated text was fed to the model. At each following step, the model’s last answer was given back to the model as a fresh prompt.

This was continued for 10 steps. The model had no access to the history of the conversation, each prompt was starting a new interaction of prompt-answer duet. At each round we sampled a maximum of 180 new tokens. The tenth answer from each chain was saved as the final prompt, yielding 384 prompts total. Because in most cases the echo emerged not in the first round of prompt-response iterations, but in the next ones, which, given the recursive framework of generating the prompts from model’s responses, caused the rather unusual conversation type, where the model is not asked, prompted or told to execute any action but confronted with a statement or reaction typical for a "helpful assistant", a role usually reserved for the model itself.

Dataset C was the reference dataset of non-echoing prompts. Using the same selfgeneration pipeline as for Dataset A, we iteratively generated text, testing each one for echogenicity. The screening condition was the negation of the previous one: a prompt was accepted only if none of the screened completions produced a prefix echo. The dataset consists of 128 prompts.

Experiment 1: Echo rates In the first experiment, we have evaluated datasets A and B on the following models: Qwen 3.5 4B, Llama 3.2 3B, Gemma 3n E4B IT, OLMo-2 0425 1B, OLMo-2 1124 13B, and SmolLM2 1.7B, as well as on an additional batch of quantized models: OLMo-2 0425 1B Q4\_K\_M, OLMo-2 1124 13B Q4\_K\_M and SmolLM2 1.7B Q4\_K\_M. All models were Instruct variants. Each target model was prompted with every prompt in the dataset, independent of which model originally generated the prompt. For the T=1.0 condition, each model-prompt pair was sampled 20 times using non-greedy decoding with top\_p=1.0, top\_k=0<sup>1</sup>, and a maximum of 160 new tokens. To test whether echoing depends on high-entropy sampling, we also evaluated temperatures 0.0 and 0.3 for the target models shown in Table 1, using the same prompt set, one completion per model-prompt pair, and the same prefix-based echo labels.

Echoes were labeled using a strict prefixonly criterion. Therefore, a completion was counted as a full echo only then, if the generated continuation began with the complete prompt token sequence under the evaluated model tokenizer. A completion was counted as a partial echo if it began with at least 40% of the prompt tokens but diverged before copying the full prompt. This definition intentionally excludes ordinary answers that quote or reuse the prompt later in the response.

Experiment 2: Training-Data Overlap Scan Experiment 2 asks whether the echoprone prompts resemble text present in public training corpora associated with a subset of the studied model, namely, instruction dataset SmolTalk and Tulu-3 SFT mixture for OLMo-2. For each of the prompt scoring at least a partial echo in Experiment 1 on SmolLM or OLMo models, we look for partial overlap in their respective public training datasets. A match was recorded if the first 5 words of the prompt occur contiguously anywhere inside a dataset document. Note that this is a weak criterion that should not be interpreted as a sign of strong training data leakage.

Experiment 3: Forcing echo Experiment 3 tests whether echoing can be induced even for prompts that do not echo spontaneously. For each prompt in the dataset C we simulated a scenario in which the model has already begun copying the prompt. This was achieved by forcing the assistant answer to start with the first k prompt tokens. We evaluated k ∈ {2, 3, 5}, allowing us to measure how much initial copying is needed before the model’s own continuation mechanism takes over. After the forced seed tokens, generation proceeded normally with sampled non-greedy decoding.

Experiment 4: Ablating copying heads In the last experiment, we investigated the efect of the ablation of induction/copying heads. We repeat the Experiment 1 evaluation on both echogenic datasets, but intervene on selected attention heads during generation.

<table><tr><td>Model</td><td>Temp.</td><td colspan="3">Dataset A</td><td colspan="3">Dataset B</td></tr><tr><td></td><td></td><td>Full</td><td>Partial</td><td>Any echo</td><td>Full</td><td>Partial</td><td>Any echo</td></tr><tr><td>Qwen3.5 4B</td><td>1.0</td><td>0.01%</td><td>0.07%</td><td>0.08%</td><td>0.00%</td><td>0.01%</td><td>0.01%</td></tr><tr><td></td><td>0.3</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.26%</td><td>0.00%</td><td>0.26%</td></tr><tr><td></td><td>0.0</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.52%</td><td>0.00%</td><td>0.52%</td></tr><tr><td>Llama 3.2 3B</td><td>1.0</td><td>0.00%</td><td>0.05%</td><td>0.05%</td><td>0.01%</td><td>1.12%</td><td>1.13%</td></tr><tr><td></td><td>0.3</td><td>0.52%</td><td>0.00%</td><td>0.52%</td><td>2.08%</td><td>1.56%</td><td>3.65%</td></tr><tr><td></td><td>0.0</td><td>0.26%</td><td>0.00%</td><td>0.26%</td><td>2.60%</td><td>1.04%</td><td>3.65%</td></tr><tr><td>Gemma 3n E4B</td><td>1.0</td><td>0.00%</td><td>1.03%</td><td>1.03%</td><td>0.00%</td><td>8.95%</td><td>8.95%</td></tr><tr><td></td><td>0.3</td><td>1.30%</td><td>0.78%</td><td>2.08%</td><td>8.59%</td><td>0.78%</td><td>9.38%</td></tr><tr><td></td><td>0.0</td><td>1.56%</td><td>0.78%</td><td>2.34%</td><td>8.33%</td><td>0.78%</td><td>9.11%</td></tr><tr><td>OLMo-2 04251B</td><td>1.0</td><td>0.14%</td><td>0.79%</td><td>0.94%</td><td>0.09%</td><td>0.18%</td><td>0.27%</td></tr><tr><td></td><td>0.3</td><td>0.26%</td><td>0.00%</td><td>0.26%</td><td>1.30%</td><td>0.00%</td><td>1.30%</td></tr><tr><td></td><td>0.0</td><td>0.26%</td><td>0.00%</td><td>0.26%</td><td>1.56%</td><td>0.00%</td><td>1.56%</td></tr><tr><td>OLMo-2 1124 13B</td><td>1.0</td><td>0.00%</td><td>0.38%</td><td>0.38%</td><td>0.13%</td><td>0.16%</td><td>0.29%</td></tr><tr><td></td><td>0.3</td><td>0.26%</td><td>0.26%</td><td>0.52%</td><td>0.52%</td><td>0.00%</td><td>0.52%</td></tr><tr><td></td><td>0.0</td><td>0.52%</td><td>0.52%</td><td>1.04%</td><td>1.30%</td><td>0.26%</td><td>1.56%</td></tr><tr><td>SmolLM2 1.7B</td><td>1.0</td><td>0.59%</td><td>1.04%</td><td>1.63%</td><td>0.43%</td><td>1.84%</td><td>2.27%</td></tr><tr><td></td><td>0.3 0.0</td><td>3.39%</td><td>1.30%</td><td>4.69%</td><td>8.85%</td><td>1.30%</td><td>10.16%</td></tr><tr><td></td><td></td><td>5.21%</td><td>0.52%</td><td>5.73%</td><td>11.46%</td><td>1.04%</td><td>12.50%</td></tr><tr><td>OLMo-2 0425 1B Q4 K M</td><td>1.0</td><td>0.05%</td><td>0.44%</td><td>0.49%</td><td>0.08%</td><td>0.20%</td><td>0.27%</td></tr><tr><td></td><td>0.3 0.0</td><td>0.52% 0.78%</td><td>0.52% 0.00%</td><td>1.04% 0.78%</td><td>0.26% 1.04%</td><td>0.00% 0.00%</td><td>0.26% 1.04%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OLMo-2 1124 13B Q4_K_M</td><td>1.0</td><td>0.22%</td><td>0.87%</td><td>1.09%</td><td>0.38%</td><td>0.51%</td><td>0.89%</td></tr><tr><td></td><td>0.3 0.0</td><td>1.04% 1.04%</td><td>0.78%</td><td>1.82%</td><td>0.78% 0.52%</td><td>0.26%</td><td>1.04%</td></tr><tr><td></td><td></td><td></td><td>0.26%</td><td>1.30%</td><td></td><td>0.26%</td><td>0.78%</td></tr><tr><td>SmolLM2 1.7B Q4_K_M</td><td>1.0</td><td>1.80%</td><td>3.07%</td><td>4.87%</td><td>0.91%</td><td>4.60%</td><td>5.51%</td></tr><tr><td></td><td>0.3</td><td>7.81% 8.07%</td><td>2.34% 2.08%</td><td>10.16%</td><td>10.94% 13.54%</td><td>1.56% 1.82%</td><td>12.50%</td></tr><tr><td></td><td>0.0</td><td></td><td></td><td>10.16%</td><td></td><td></td><td>15.36%</td></tr></table>

Table 1: Experiment 1 prefix-echo rates across temperatures. For T=1.0, each model-prompt pair was sampled 20 times; for T=0.3 and $\mathrm { { T = 0 . 0 , } }$ only once. Full echo means the continuation begins with the complete prompt; partial echo means it begins with at least 40% of the prompt tokens but diverges before copying the full prompt.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Variant</td><td rowspan="2"></td><td colspan="3">k = 2 forced tokens</td><td colspan="3">k = 3 forced tokens</td><td colspan="3">k = 5 forced tokens</td></tr><tr><td>Full</td><td>Partial</td><td>Rate</td><td>Full</td><td>Partial</td><td>Rate</td><td>Full</td><td>Partial</td><td>Rate</td></tr><tr><td>Gemma 3n E4B IT</td><td>full</td><td></td><td>71</td><td>341</td><td>16.1%</td><td>229</td><td>565</td><td>31.0%</td><td>284</td><td>738</td><td>39.9%</td></tr><tr><td>Llama 3.2 3B Instruct</td><td>full</td><td></td><td>58</td><td>80</td><td>5.4%</td><td>93</td><td>178</td><td>10.6%</td><td>111</td><td>407</td><td>20.2%</td></tr><tr><td>OLMo-2 0425 1B Instruct</td><td>full</td><td></td><td>3</td><td>40</td><td>1.7%</td><td>4</td><td>112</td><td>4.5%</td><td>8</td><td>189</td><td>7.7%</td></tr><tr><td>OLMo-2 0425 1B Instruct</td><td>Q4_K_M</td><td></td><td>16</td><td>89</td><td>4.1%</td><td>37</td><td>407</td><td>17.3%</td><td>40</td><td>570</td><td>23.8%</td></tr><tr><td>OLMo-2 1124 13B Instruct</td><td>full</td><td></td><td>0</td><td>45</td><td>1.8%</td><td>3</td><td>169</td><td>6.7%</td><td>3</td><td>200</td><td>7.9%</td></tr><tr><td>OLMo-2 1124 13B Instruct</td><td>Q4_K_M</td><td></td><td>12</td><td>160</td><td>6.7%</td><td>46</td><td>382</td><td>16.7%</td><td>59</td><td>499</td><td>21.8%</td></tr><tr><td>Qwen3.5 4B</td><td>full</td><td></td><td>115</td><td>211</td><td>12.7%</td><td>195</td><td>426</td><td>24.3%</td><td>322</td><td>697</td><td>39.8%</td></tr><tr><td>SmolLM2 1.7B Instruct</td><td>full</td><td></td><td>65</td><td>234</td><td>11.7%</td><td>128</td><td>391</td><td>20.3%</td><td>133</td><td>536</td><td>26.1%</td></tr><tr><td>SmolLM2 1.7B Instruct</td><td>Q4_K_M</td><td></td><td>159</td><td>411</td><td>22.3%</td><td>253</td><td>529</td><td>30.5%</td><td>271</td><td>636</td><td>35.4%</td></tr></table>

Table 2: Efect of inducing echo by forcing the assistant’s answer to begin with prefix of the prompt.
<table><tr><td rowspan="2">Model</td><td colspan="3">No ablation</td><td colspan="3">Top-8 copy heads</td><td colspan="2">Top-16 copy</td><td rowspan="2">heads</td><td colspan="3">Random-16 heads</td></tr><tr><td>Full</td><td>Partial</td><td>Rate</td><td>Full</td><td>Partial</td><td>Rate</td><td>Full</td><td>Partial</td><td>Rate</td><td>Full Partial</td><td>Rate</td></tr><tr><td>Qwen3.5 4B Instruct</td><td>2</td><td></td><td>80.07%</td><td>0</td><td>1</td><td>0.01%</td><td>0</td><td>0</td><td>0.00%</td><td>1</td><td></td><td>40.03%</td></tr><tr><td>Llama 3.2 3B Instruct</td><td>2</td><td></td><td>94 0.62%</td><td>1</td><td>6</td><td>0.05%</td><td>0</td><td>2</td><td>0.01%</td><td>1</td><td></td><td>420.28%</td></tr><tr><td>Gemma 3n E4B IT</td><td>0</td><td></td><td>7644.97%</td><td>0</td><td>1</td><td>0.01%</td><td>0</td><td>0</td><td>0.00%</td><td>0</td><td></td><td>7154.65%</td></tr><tr><td>OLMo-2 0425 1B Instruct</td><td>16</td><td></td><td>93 0.71%</td><td>1</td><td>0</td><td>0.01%</td><td>0</td><td>0</td><td>0.00%</td><td>4</td><td></td><td>82 0.56%</td></tr><tr><td>OLMo-2 1124 13B Instruct</td><td>8</td><td></td><td>450.35%</td><td>0</td><td>6</td><td>0.04%</td><td>0</td><td>2</td><td>0.01%</td><td>10</td><td></td><td>440.35%</td></tr><tr><td>SmolLM2 1.7B Instruct</td><td>79</td><td></td><td>2231.97%</td><td>0</td><td>11</td><td>0.07%</td><td>0</td><td>6</td><td>0.04%</td><td>64</td><td>162</td><td>1.47%</td></tr></table>

Table 3: Efect of ablating attention heads with high prefix-match score

For each model, heads are first ranked using the prefix-matching score of (Olsson et al., 2022), computed on repeated random inputs.

We then run the prompt-echo evaluation under four conditions: no intervention, ablation of the top 8 copying heads, ablation of the top 16 copying heads, and ablation of 16 randomly selected heads as a control. Ablation is implemented by zeroing the selected heads’ attention-output slices before the output projection during generation. Echoes are scored with the same strict prefix-only labels as in Experiment 1. The experiment was run for the full-precision models collection.

SmolLM2-target echoes by prompt source  
![](images/d37e151f89864d4b120e2a36c94acf5fc750dac958601e05b8e23d2ffec6c032.jpg)  
Figure 2: Prompt-source heatmap for lowtemperature echoes into the SmolLM2 target family. Cells count unique prompts, out of 384 prompts in the corresponding dataset, that produced a full or partial prefix echo in at least one SmolLM2 target model (fp32 or Q4\_K\_M). Source columns: Qw=Qwen, Ll=Llama, Ge=Gemma, O1=OLMo-2 0425 1B, O13=OLMo-2 1124 13B, Sm=SmolLM2.

## 2 Results and interpretation

The results of Experiment 1 are summarized in Table 1. Prompt echoing occurs across all six model families, even at temperature 1.0. Quantization amplifies the effect for some models (SmolLM2 jumps from 1.95% to 5.19%) but not universally (OLMo-2 1B drops slightly from 0.61% to 0.38%). The lower-temperature conditions show that echoing does not vanish under greedy or neargreedy decoding. On the contrary, in most tested model-dataset combinations, echo rates are higher at T=0.0 or T=0.3 than at T=1.0. This pattern is most visible for Dataset B and for the SmolLM2 family, especially the Q4\_K\_M variant, where the any-echo rate reaches 15.36% at T=0.0. We have also examined the source-model composition for the highest echoing model. Figure 2 shows that SmolLM2-target echoes are not restricted to prompts originally generated by SmolLM2; in every low-temperature condition, most come from other source models.

In the Experiment 2 we observed that 32.63% of prompts generating echo in OLMo matched some text in the training dataset as compared to 9.38% for non-echogenic prompts. The diference was weaker for SmolLM model—26.33% for echoing prompts vs 17.97% for non-echogenic prompts. This suggests memorization does not explains the echo well.

The results of Experiment 3 are summarized in Table 2. The experiment confirms that forcing even 2 initial prompt tokens into the assistant’s answer results in a significantly increased rate of partial and full echoes. For Gemma 3n E4B IT, 5 forced tokens increase the echo rate to almost 40%, even under hightemperature decoding. This supports the hypothesis that once the echo kicks in, it is continued by the internal copying mechanism — even for prompts that were selected not to echo spontaneously.

This is further supported by Experiment 4, in which we ablated the internal copying mechanism in the form of the model’s induction heads. Ablation of candidate induction heads causes a larger drop in echo rates than ablating randomly chosen attention heads across all models. We also observe a larger decrease in echo rate when more candidate induction heads are ablated (16 rather than 8). This is consistent with previous work which reported that the copying mechanism is not uniquely localized in a single head but rather spread across several heads (Olsson et al., 2022; Bansal et al., 2023; Crosbie and Shutova, 2025). The results of Experiment 4 are summarized in Table 3.

Conclusions Prompt echo appears to arise from several interacting mechanisms: it is influenced by memorization but not fully explained by it, and once initialized, it is primarily driven by internal copying mechanisms. Its persistence at temperatures 0.0 and 0.3 further suggests that echoing is not only a rare tail-sampling outcome, but can become a stable decoding trajectory for some promptmodel pairs. Future work may isolate the echoinitiation step more directly, quantizationdependent shifts or activation patching and circuit discovery, to move beyond head-level ablations toward a mechanistic account.

## Limitations

Our study focuses on a narrow, prefix-based form of prompt echoing in small instruct-tuned language models, and the reported echo rates should be interpreted as measurements on diagnostic prompt distributions rather than estimates of deployment prevalence. Our strict token-prefix definition makes echoing easy to score and analyze, but excludes related behaviors such as copying later spans of the prompt, paraphrasing the prompt, quoting it inside an otherwise normal answer, or producing more difuse repetition.

Our training-data overlap analysis is limited to only two models: public training or instruction-tuning data are available only for some model families. Short prefix matches do not by themselves establish verbatim memorization or training-data leakage. Similarly, our induction-head ablations test whether heads with high prefix-matching scores are causally involved in echoing, but head ablation is a coarse intervention and does not fully identify the underlying circuit. Finally, our experiments cover only a limited set of English, one-shot prompts and small instruct models; the conclusions may not directly transfer to larger models, multilingual settings, longcontext use, or multi-turn interactions.

## Acknowledgements

Work on this project is financially supported by the Foundation for Polish Science (FNP) grant ‘Centre for Credible AI’ No. FENG.02.01-IP.05-0058/24.

## References

Hritik Bansal, Karthik Gopalakrishnan, Saket Dingliwal, Sravan Bodapati, Katrin Kirchhof, and Dan Roth. 2023. Rethinking the role of scale for in-context learning: An interpretabilitybased case study at 66 billion scale. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11833–11856, Toronto, Canada. Association for Computational Linguistics.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Vaibhav Srivastav, Joshua Lochner, Clé- mentine Fourrier, Hugo Larcher, Ben Burtenshaw, Haojun Zhao, Caleb Fahlgren, Mathieu Morlon, Agustín Piqueres Lajarín, Xuan-Son Nguyen, Cyril Zakka, and 3 others. 2025. SmolLM2: When smol goes big – data-centric training of a fully open small language model. In Proceedings of the 2nd Conference on Language Modeling.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. 2023. Quantifying memorization across neural language models. In The Eleventh International Conference on Learning Representations.

Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Rafel. 2021. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633– 2650. USENIX Association.

Joy Crosbie and Ekaterina Shutova. 2025. Induction heads as an essential mechanism for pattern matching in in-context learning. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 5049–5111, Albuquerque, New Mexico. Association for Computational Linguistics.

Tianxiang Dai and Jonathan A. Fan. 2026. Language models fail at extended rule following. Preprint, arXiv:2605.02028.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, and 1 others. 2024. Gemma: Open models based on Gemini research and technology. Preprint, arXiv:2403.08295.

Google DeepMind. 2025. Gemma 3n E4B model card. Hugging Face model card. Accessed: 2026-05-25.

Aaron Grattafiori and 1 others. 2024. The Llama 3 herd of models. Preprint, arXiv:2407.21783.

Dirk Groeneveld, Iz Beltagy, Evan Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, and 24 others. 2024. OLMo: Accelerating the science of language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15789–15809, Bangkok, Thailand. Association for Computational Linguistics.

Zhuoyuan Hao, Zhuo Li, Wu Li, Fangming Liu, Min Zhang, and Jing Li. Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2025. Prompt repetition improves nonreasoning llms. Preprint, arXiv:2512.14982.

Callum Stuart McDougall, Arthur Conmy, Cody Rushing, Thomas McGrath, and Neel Nanda.

2024. Copy suppression: Comprehensively understanding a motif in language model attention heads. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 337–363, Miami, Florida, US. Association for Computational Linguistics.

Raja Sekhar Reddy Mekala, Yasaman Razeghi, and Sameer Singh. EchoPrompt: Instructing the Model to Rephrase Queries for Improved In-context Learning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 399–432. Association for Computational Linguistics.

Meta Llama. 2024. Llama 3.2-3B model card. Hugging Face model card. Accessed: 2026-05- 25.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, and 7 others. 2022. In-context learning and induction heads. Transformer Circuits Thread.

Qwen Team. 2026. Qwen3.5-4B model card. Hugging Face model card. Accessed: 2026-05-25.

Junhyuk Seo, Dasol Choi, Taerim Kim, Won Chul Cha, Minha Kim, Haanju Yoo, Namkee Oh, YongJin Yi, Kye Hwa Lee, and Edward Choi. 2024. Evaluation framework of large language models in medical documentation: Development and usability study. Journal of Medical Internet Research, 26:e58329.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.