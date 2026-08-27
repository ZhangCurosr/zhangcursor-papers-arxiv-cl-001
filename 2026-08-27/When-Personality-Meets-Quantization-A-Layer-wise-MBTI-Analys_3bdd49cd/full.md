# When Personality Meets Quantization: A Layer-wise MBTI Analysis of Quantized LLMs

Yao Fu, Runchao Li, Yu Yin & Kenneth A. Loparo ∗   
Case Western Reserve University   
Cleveland, OH 44106, USA   
{yxf484,rxl685,yxy1421,kal4}@case.edu

Lijia Huang Northeastern University Boston, MA 02115, USA {huang.liji}@northeastern.edu

Xiaomin Li Microsoft Research Redmond, WA 98052, USA {xiaominli}@g.harvard.edu

## Abstract

Personality is increasingly important in large language models (LLMs), as it shapes users’ trust, engagement, and emotional experiences. While the Myers–Briggs Type Indicator (MBTI) has emerged as a common framework for assessing LLMs’ personality, existing studies focus primarily on full-precision models and evaluate only final outputs. They overlook the widespread deployment of quantized LLMs requiring low memory footprints, whose personality traits remain underexplored. In this work, we present a systematic MBTI analysis of open-source LLMs across multiple precisions, including mainstream 4-bit methods (GPTQ, AWQ) and extreme 2-bit settings (AQLM variants). Beyond output-level evaluation, we examine how personality emerges across layers through option-level entropy and confidence-gap dynamics, and introduce Uncertainty-Amplified Layer Decoding (UALD) to study decoding-induced personality drift at inference time. Our results reveal a key insight: LLMs’ personality is not a static property, but an emergent, layer-dependent decision process sensitive to quantization, prompting, and decoding. Specifically, we find that (1) ENFJ remains dominant across model families and precisions; (2) 4-bit quantization largely preserves coarse personality structure, while 2-bit quantization disrupts fine-grained prompt consistency and cross-precision agreement; (3) personality decisions emerges in upper layers, following substantial ambiguity in early layers; and (4) inference decoding can shift personality, while personality-aligned conditioning improves robustness. These findings provide a new perspective on the behavioral reliability of quantized LLMs and highlight the importance of considering internal dynamics and inference strategies in personality-sensitive chatbot applications.

## 1 Introduction

LLMs (Zhao et al., 2023) are increasingly deployed as interactive assistants that provide personal advice (Ponzo et al., 2024). This trend is particularly pronounced among young users given that 30% of teenagers engage in “serious conversations” with AI instead of real people (Robb & Mann, 2025). As reliance on LLMs grows, their perceived personality plays a critical role in shaping users’ trust, willingness to continue interacting, and emotional experiences (Lu et al., 2026). This importance is also reflected in industry practice. Major AI developers explicitly regard personality as an integral component of product design: Anthropic studies methods to monitor and control traits such as humor, optimism, and sycophancy in Claude (Anthropic, 2024; 2025), while Google emphasizes making Gemini more helpful and assistant-like (Google, 2023). Recent public reactions to GPT-5 (Constitutional Discourse, 2025) further highlight the importance of personality: despite improved benchmark performance, GPT-5 was perceived by many users as less warm and emotionally engaging than GPT-4o, characterized as “lobotomization” (Michels). Together, these observations motivate a systematic evaluation of LLMs’ personality. In this regard, MBTI-style analysis (Myers et al., 1962; Myers, 1985) provides a structured framework for assessing whether models exhibit stable behavioral tendencies when interacting with people.

Despite recent efforts to study the MBTI personality of LLMs (Pan & Zeng, 2023; La Cava & Tagarelli, 2025), these studies primarily focus on full-precision models and fail to examine compressed ones. This gap is increasingly consequential because quantization (Lang et al., 2024) is widely used in practical deployment to substantially reduce memory footprint and inference cost. For example, AWQ (Lin et al., 2024) enables running a 4-bit AWQ-quantized LLaMA3-70B-Instruct<sup>1</sup> on a single A6000 GPU, whereas the original 16-bit version typically requires four GPUs. Consequently, numerous real-world LLM applications, especially in resource-constrained environments, rely on quantized rather than full-precision models (Qu et al., 2025). Understanding how quantization affects the personality traits expressed by LLMs, and their ability to follow externally specified personality constraints, therefore remains an important research question. Moreover, existing studies evaluate personality based only on final outputs from questionnaire prompts, leaving two research gaps unaddressed: (1) How do personality judgments emerge and evolve across layers during inference? (2) To what extent can inference-time decoding perturbations influence final personality attribution, given that personality assessment involves selecting among multiple plausible responses rather than predicting a single ground-truth answer?

Addressing these gaps helps clarify whether the personality traits attributed to LLMs reflect stable behavioral patterns or are instead sensitive to implementation factors such as quantization, prompting, and decoding strategies. This distinction is particularly important for MBTI-style assessments, where items are inherently self-referential and lack externally verifiable ground truth, unlike factual question answering tasks (Chuang et al., 2023; Lin et al., 2022) that typically have a correct answer per question. Accordingly, we present a systematic MBTI analysis of LLMs across open-source model families and multiple precision regimes (Section 3.1). We evaluate both unconditional and personality-conditional prompting by reformulating MBTI answering as single-token choice prediction, reducing sampling variability. Beyond output-level analysis, we characterize layer-wise decision dynamics through entropy and confidence-gap trajectories, and investigate decoding-induced personality drift using a logit-difference-based inference strategy with controlled evolution scales. Our results reveal a key insight: personality in LLMs is not a static property, but an emergent, layer-dependent decision process that can be influenced by quantization precision, prompt methods, and inference-time decoding. Our key contributions are summarized as follows:

• We present the first systematic MBTI evaluation pipeline for quantized LLMs, covering unconditional and personality-conditional prompting across 4-bit and extreme 2-bit settings.

• We move beyond output-level evaluation by analyzing layer-wise personality formation, showing how uncertainty among response options evolves into sharper personality commitments via entropy and confidence-gap dynamics.

• We study decoding-induced personality drift at inference time, identifying when personality assessments become unstable and demonstrating that personalityaligned prompting improves robustness of models’ tendency to MBTI questions.

## 2 Related Work

## 2.1 LLM Quantization

Quantization is a widely used model compression technique (Zhou et al., 2024) that reduces storage and computational requirements by mapping high-precision parameters to lowerprecision representations. Existing approaches are generally categorized into post-training quantization (PTQ) (Frantar et al., 2022; Xiao et al., 2023; Lee et al., 2023; Kim et al., 2023b; Li et al., 2024; Yao et al., 2022; Wei et al., 2022; Yuan et al., 2023; Lin et al., 2024; Liu et al., 2023a; Ashkboos et al., 2024; Shao et al., 2023; Zhao et al., 2024; Egiazarian et al., 2024) and quantization-aware training (QAT) (Malinovskii et al., 2024; Liu et al., 2023b; Du et al., 2024; Ma et al., 2024; Xu et al., 2024b). In general, PTQ methods are more efficient but often yield lower performance than QAT, because QAT explicitly incorporates quantization effects during training. However, QAT typically requires substantial computational resources and access to training data that limits its practical applicability. To mitigate these challenges, parameter-efficient fine-tuning (PEFT) methods (Li et al., 2023; Guo et al., 2023; Xu et al., 2023; Chai et al., 2023; Dettmers et al., 2023; Hayou et al., 2024; Kim et al., 2023a) have been proposed to enable efficient adaptation of quantized LLMs. In contrast to prior work that primarily develops new quantization techniques to improve performance on standard benchmarks, our study focuses on systematically evaluating the personality characteristics of quantized LLMs through MBTI-style assessments.

## 2.2 Compressed LLM Evaluation

Several studies have recently been proposed to evaluate compressed LLMs from different perspectives, such as safety (Egashira et al., 2024), toxicity (Belkhiter et al., 2024; Xu et al., 2024c), bias (Chhabra & Khalili, 2025; Chen et al., 2025), and trustworthiness (Hong et al., 2024; Fu et al., 2025). In addition, Mekala et al. (2025) studies the impact of quantization on tasks requiring long-context inputs or long-form generation. Unlike previous work, we focus on a systematic personality assessment of quantized LLMs.

## 2.3 Personality Study in LLMs

Personality extraction from text has been a challenging problem in natural language processing (NLP) (Lynn et al., 2020; Yang et al., 2021). The emergence of LLMs has further stimulated interest in this area (Ganesan et al., 2023; Ji et al., 2023). Recent studies have explored several related directions: characterizing the intrinsic personalities of LLMs (Li et al., 2022; Miotto et al., 2022; Huang et al., 2023b; Frisch & Giulianelli, 2024; Lu et al., 2026; Cheng et al., 2026); instilling personality traits via prompt engineering (Caron & Srivastava, 2022; Mao et al., 2023); building personality-aware agents (Jiang et al., 2023a); and benchmarking the personality assessment capabilities of LLMs (Jiang et al., 2022; Xu et al., 2024a; Huang et al., 2023a). To the best of our knowledge, the most related work (Pan & Zeng, 2023; La Cava & Tagarelli, 2025) evaluates personality traits only on full-precision LLMs. In contrast, we extend this line of research by systematically analyzing personality behaviors of quantized LLMs. Furthermore, we investigate the layer-wise evolution of personality judgments and examine how inference-time decoding influences personality traits.

## 3 Experimental Settings for MBTI Analysis

This section presents our end-to-end framework for MBTI-based personality evaluation, including model and quantization choices, MBTI fundamentals, prompting strategies, and inference-time decoding settings.

## 3.1 Models and Quantization Choices

Our study covers representative open-source LLMs and quantization techniques based on publicly available Hugging Face implementations as of early 2026 (Appendix F). We examine several widely adopted LLM families, including LLaMA (Dubey et al., 2024), Mistral (Jiang et al., 2023b), and Qwen (Yang et al., 2024). Our selection is motivated by two main considerations. First, their public availability enables straightforward application and comparison of different quantization methods. Second, these models demonstrate strong performance across diverse tasks and are widely used by LLM practitioners (Dubey et al., 2024; Yang et al., 2024). For quantization, we focus on two mainstream 4-bit techniques,

GPTQ (Frantar et al., 2022)<sup>2</sup> and AWQ (Lin et al., 2024)<sup>3</sup>, as they are widely adopted in both academic and industrial settings, evidenced by their rapid growth in citations and GitHub stars<sup>4</sup>. We also evaluate extreme 2-bit quantization methods, including AQLM (Egiazarian et al., 2024) and PV-tuned AQLM (Malinovskii et al., 2024).

## 3.2 MBTI Fundamentals

The Myers–Briggs Type Indicator (MBTI) (Harrington & Loffredo, 2010; Cohen et al., 2013; DiRienzo et al., 2010) is a widely recognized personality framework extensively used across diverse areas, including academic research, career counseling, and personal or organizational development. More recently, MBTI has been adopted as a structured tool for evaluating personality characteristics of LLMs (La Cava & Tagarelli, 2025). Specifically, MBTI is a self-report personality assessment questionnaire (Myers et al., 1962; Myers, 1985) designed to measure four fundamental personality dichotomies: Extraversion (E) vs. Introversion (I), Sensing (S) vs. Intuition (N), Thinking (T) vs. Feeling (F), and Judging (J) vs. Perceiving (P), as illustrated in Table 1. By combining four binary dimensions, the framework defines 16 distinct personality types, each associated with unique characteristic strengths, weaknesses, and behavioral tendencies, whose details are provided in Appendix B. In this paper, we adopt the standardized MBTI questionnaire consisting of 60 items (Tables 4 and 5) following the recent work (La Cava & Tagarelli, 2025). Because MBTI assigns each subject to a single personality type, the task can be formulated as a multi-class classification problem over 16 categories based on response patterns. It is important to note that MBTI questions are tendency-based and lack a single objectively correct answer, which distinguishes them from factual evaluation tasks (Lin et al., 2022). Rather, they capture how models distribute probability mass over responses to subjective, preference-based personality items without verifiable ground-truth labels at inference time.

Table 1: The four MBTI dichotomies (top) and the 16 MBTI personality types (bottom). More details about 16 types are introduced in Appendix B.
<table><tr><td rowspan=1 colspan=1>Dichotomy</td><td rowspan=1 colspan=1>Brief Description</td></tr><tr><td rowspan=1 colspan=1>Extraversion (E)Introversion (I)</td><td rowspan=1 colspan=1>E types tend to gain energy from the external world and activeinteractions with people, whereas I types tend to focus on theirinner world, gaining energy from reflection and solitude.</td></tr><tr><td rowspan=1 colspan=1>Sensing (S)Intuition (N)</td><td rowspan=1 colspan=1>S types prefer concrete, factual information and focus onpresent realities and practical details, whereas N types pre-fer abstract patterns, and future-oriented interpretations.</td></tr><tr><td rowspan=1 colspan=1>Thinking (T)Feeling (F)</td><td rowspan=1 colspan=1>T types tend to make decisions based on logic, consistency,and objective principles, whereas F types tend to prioritizevalues, emotions, and the impact of decisions on people.</td></tr><tr><td rowspan=1 colspan=1>Judging (J)Perceiving (P)</td><td rowspan=1 colspan=1>J types prefer structure and planning, favoring organizationand certainty, whereas P types prefer flexibility, spontaneity,and adaptability, keeping options open and uncertain.</td></tr><tr><td rowspan=1 colspan=1>16 Types</td><td rowspan=1 colspan=1>ESTP ESFP ENFP ENTP ESTJ ESFJ ENFJ ENTJISTJ ISFJ INFJ INTJ ISTP ISFP INFP INTP</td></tr></table>

## 3.3 Prompting Techniques

This work adopts two prompting paradigms: unconditional prompting and personalityconditional prompting (see Appendix A for details), to assess both the intrinsic personality traits of LLMs and the robustness of these traits under external personality perturbations.

Unconditional Prompting. Models are prompted to respond to each MBTI questionnaire item without explicit personality conditioning. As detailed in Appendix A, the prompt asks models to rate how well each statement describes their tendencies using a sevenpoint agreement scale. This setup probes the intrinsic personality disposition of the models, i.e., the latent preference distribution arising from pre-training and post-training. From a probabilistic perspective, unconditional responses reflect the default probability distribution over adjacent agreement levels of the models.

Personality-Conditional Prompting. The unconditional prompt is prepended with an instruction to adopt a specified MBTI type (e.g., respond as an ENFJ). This setup evaluates personality controllability, i.e., the ability of a model to shift its response distribution under externally imposed personality constraints. While the unconditional prompt probes intrinsic tendencies, the personality-conditioned prompt tests whether models can override these preferences to align with a specified personality profile.

Vanilla Decoding Strategy. Prior work (La Cava & Tagarelli, 2025) formulates MBTI assessment as a multi-token generation task, where models generate sequences that are subsequently mapped to personality labels. This approach relies on sampling-based decoding, making outputs sensitive to hyperparameters such as temperature, top-k, top-p, and random seed, thereby introducing stochasticity that may confound evaluation. To address this, we reformulate MBTI assessment as a single-token classification problem and adopt greedy decoding to select the highest-probability token, making inference deterministic and eliminating the influence of sampling-related hyperparameters.

## 3.4 Inference Decoding Strategy

To study how decoding dynamics shape decisional commitment in MBTI-style assessment, we introduce Uncertainty-Amplified Layer Decoding (UALD), whose details are in Appendix E. While inspired by DoLa (Chuang et al., 2023), UALD differs in three aspects. (i) Choice space: instead of operating over the full vocabulary, logits are collapsed into seven option-specific scores (A to G), aligning decoding with discrete personality choices. (ii) Objective: unlike DoLa’s focus on factual correctness, UALD probes how decisional commitment arises from intermediate representations for a subjective task. (iii) Formulation: rather than logit subtraction, UALD uses a scaled additive combination that amplifies intermediate-layer uncertainty. At each step, the mature (final) layer is compared with candidate premature layers by computing softmax distributions over the seven options and measuring the Jensen–Shannon divergence (JSD) from the mature distribution. The layer with maximal divergence is selected, capturing the strongest intermediate disagreement. The final logits are given by:

$$
p ^ { ( \mathrm { U A L D } ) } = \log p _ { \mathrm { m a t u r e } } + \lambda \log p _ { \mathrm { p r e m a t u r e } } ,
$$

where λ (evolution scale) controls the influence of intermediate uncertainty. Larger λ amplifies early-layer signals, while smaller values of λ preserve the mature distribution. We adjust λ ∈ {5, 10, 15, 20, 25, 30, 35, 40} to analyze its effect on decisiveness and stability.

## 4 MBTI Personality Analysis

In this section, we analyze the personality characteristics of original and quantized LLMs based on MBTI. The details of MBTI scoring are in Appendix C.

## 4.1 Results of Unconditional Prompting

i) Personality Assessment Based on Outputs. From Table 2, we observe, consistent with La Cava & Tagarelli (2025), that ENFJ (Extraverted, iNtuitive, Feeling, Judging) emerges as the dominant MBTI type across different model families and quantization settings, illustrating that quantization does not substantially alter the dominant personality attribution. The prevalence of ENFJ indicates a systematic tendency for LLMs to produce empathetic, warm, supportive, and guidance-oriented responses. We conjecture that these behavioral patterns reflect the “AI Assistant” persona (Lu et al., 2026), shaped by post-training processes such as supervised fine-tuning, reinforcement learning from human feedback (RLHF) (Bai et al., 2022), and direct preference optimization (DPO) (Rafailov et al., 2023).

Table 2: MBTI type predicted across models and quantization methods.
<table><tr><td>Model</td><td>Original FP16</td><td>AWQ-INT4</td><td>GPTQ-INT4</td><td>Extreme INT2</td></tr><tr><td>LLaMA 3.1-8B-Instruct</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td></tr><tr><td>LLaMA 3.1-70B-Instruct</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td></tr><tr><td>Mistral3-7B-Instruct</td><td>ENFJ</td><td>ENTJ</td><td>ENFJ</td><td>ENFJ</td></tr><tr><td>Mistral-Small-24B-Instruct</td><td>ENTJ</td><td>ENTJ</td><td>ENTJ</td><td>ENFJ</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>ENTJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td></tr></table>

![](images/4a322eca759c2c09a32d346fef79d482ca772c6c1a477cf1e54fe12fb8228c3c.jpg)  
(a) Layer-wise Decisional Entropy

![](images/25cac134028cef608512d8d210124f8ecda9480fb9847c2a6ab365f95634e17e.jpg)  
(b) Layer-wise Confidence Gap  
Figure 1: Main comparison (LLaMA3.1-8B-Instruct) across layers for four model variants (Original FP16, GPTQ INT4, AWQ INT4, Extreme INT2). Figure (a) reports layer-wise decisional entropy, where lower entropy indicates sharper, more stable decisions. Figure (b) reports the layer-wise confidence gap (top-1 minus top-2 probability), where a larger margin reflects stronger commitment to a single option.

ii) Personality Perceptions Evolve Across Layers. Layer-wise analysis of MBTI personality assessment is conducted to characterize how response preferences evolve across layers. In contrast to factual tasks (Chuang et al., 2023), where outputs can be evaluated via ground truth, MBTI assessment requires choosing among discrete tendency-style options (A to G) for which no correct answer exists. Accordingly, at each layer we examine the modelinduced probability distribution over the seven response options and quantify uncertainty via entropy and decisiveness via the top-1/top-2 probability gap (details provided in Appendix D). Figure 1 illustrates the layer-wise trends of LLaMA3.1-8B-Instruct, where two key patterns are identified. We could observe that other models have similar results from Figures 4, 5, 6, 7, and 8.

Pattern #1: Distributional Ambiguity Exists in Early and Intermediate Layers. In the early and intermediate layers (approximately layers 1 to 21), response distributions exhibit high entropy and small probability gaps between adjacent options across all model variants (Figure 1). In this case, neighboring options, such as Partially Agree, Neither Agree nor Disagree, and Partially Disagree, retain comparable probability mass, indicating that models have not yet converged to a definitive preference. Rather than reflecting a determinate tendency, this uncertainty suggests an unresolved preference structure in early layers. As depth increases, these representations gradually sharpen, resembling aspects of human self-assessment in personality psychology, where individuals often form self-perceptions progressively and may experience internal ambiguity before reaching a stable self-concept.

Table 3: LLaMA3.1-8B-Instruct: Conditional personality consistency under prompt conditioning across quantization levels. Each entry reports the inferred MBTI type when the model is conditioned on a personality-specific prompt (e.g., ENFJ → ENFJ). For each cell, the left marker indicates consistency with the conditioning prompt, while the right marker indicates agreement with the FP16 baseline prediction.
<table><tr><td>Original FP16</td><td>GPTQ INT4</td><td>AWQ INT4</td><td>Extreme INT2</td></tr><tr><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$   $\mathrm { E N F P }  \mathrm { E N F P } \checkmark \sqrt { }$ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$ </td><td> $\mathrm { E N F J } \to \mathrm { E N F J } \sqrt  \surd $ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { } <$ </td></tr><tr><td></td><td> $\mathrm { E N F P }  \mathrm { E N F P } \checkmark \sqrt { }$ </td><td> $\mathrm { E N F P }  \mathrm { E N F P } \checkmark \sqrt { }$ </td><td> $\mathrm { E N F P }  \mathrm { E N F P } \checkmark \sqrt { }$ </td></tr><tr><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt  \surd $ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt  \surd $ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt { \surd }$ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt  \surd $ </td></tr><tr><td> $\mathrm { E N T P }  \mathrm { E N T P } \checkmark { }$ </td><td> $\mathrm { E N T P }  \mathrm { E N T P } \checkmark \sqrt { }$ </td><td> $\mathrm { E N T P }  \mathrm { E N T P } \checkmark { }$ </td><td> $\mathrm { E N T P }  \mathrm { E N T P } \checkmark { }$ </td></tr><tr><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \checkmark$ </td></tr><tr><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { E S F P }  \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T J } \to \mathrm { E S T J } \ V \ V$ </td><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T J } \to \mathrm { E S T J } \lor $ </td></tr><tr><td> $\mathrm { E S T P }  \mathrm { E S T P } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T P }  \mathrm { E N F P } \times \times$ </td></tr><tr><td> $\mathrm { I N F J } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I N F J } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I N F J } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I N F J } \to \mathrm { E N F J } \times \checkmark$ </td></tr><tr><td> $\mathrm { I N F P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { I N F P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { I N F P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { I N F P }  \mathrm { E N F P } \times \surd$ </td></tr><tr><td> $\mathrm { I N T J } \to \mathrm { I N T J } \ V$ </td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \Vdash$ </td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \Vdash$ </td><td> $\mathrm { I N T J } \to \mathrm { I N F J } \times \times$ </td></tr><tr><td> $\mathrm { I N T P } \to \mathrm { I N T J } \times \mathcal { A }$ </td><td> $\mathrm { I N T P } \to \mathrm { I N T J } \times \mathcal { A }$ </td><td> $\mathrm { I N T P } \to \mathrm { I N T J } \times \mathcal { A }$ </td><td> $\mathrm { I N T P } \to \mathrm { I N T J } \times \mathcal { A }$ </td></tr><tr><td> $\mathrm { I S F J }  \mathrm { I N F J } \times \surd$ </td><td> $\mathrm { I S F J } \to \mathrm { I S F J } ^ { \cdot } \checkmark \times$ </td><td> $\mathrm { I S F J }  \mathrm { E N F J } \times \times$ </td><td> $\mathrm { I S F J }  \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { I S F P }  \mathrm { I N F P } \times \surd$ </td><td> $\mathrm { I S F P }  \mathrm { I N F P } \times \checkmark$ </td><td> $\mathrm { I S F P }  \mathrm { E N F J } \times \times$ </td><td> $\mathrm { I S F P }  \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { I S T J }  \mathrm { I N T J } \times \ √$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ v \times$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ v \times$ </td><td> $\mathrm { I S T J }  \mathrm { I N T J } \times \surd$ </td></tr><tr><td> $\mathrm { I S T P }  \mathrm { I N T J } \times \surd$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \times$ </td><td> $\mathrm { I S T \dot { P } }  \mathrm { I N T \dot { I } J } \times \dot { \sqrt { \mathbf { \Lambda } } }$ </td><td> $\mathrm { I S T P }  \mathrm { I N T J } \times \surd$ </td></tr></table>

In this sense, personality evaluation, whether in humans or LLMs, can involve graded internal preferences prior to a final decision.

Pattern #2: Decisional Commitment Emerges in Upper Layers. In contrast to the early and intermediate layers (approximately layers 1 to 21), the upper layers (approximately layers 22 to 32) show a clear drop in entropy (Figure 1) alongside a substantial increase in the top-1/top-2 probability gap, indicating increasingly decisive responses. At this stage, probability mass concentrates on a single option, yielding a definitive MBTI prediction. This sharpening behavior is consistent across different quantization types, suggesting that the overall decision trajectory is preserved under moderate quantization. For full-precision and 4-bit models, entropy decreases smoothly in the final layers, reflecting gradual consolidation of preferences, whereas 2-bit models exhibit delayed or unstable sharpening. Overall, these results indicate that decisional commitment in personality assessment is a late-emerging property that arises after sequential transformations reduce representational uncertainty.

## 4.2 Results of Personality-Conditional Prompting

Table 3 presents conditional personality consistency under 16 types of prompts across different quantization levels. Each entry reports the inferred MBTI type of LLaMA3.1-8B-Instruct when conditioned on a personality-specific prompt $( \mathrm { e . g . , E N F J } \to \mathrm { E N T J }$ indicates that the model is prompted to behave as ENFJ and is eventually assessed as ENTJ). For each cell, two indicators are provided: (i) the left marker (prompt consistency) that is marked as ✓ if the predicted personality matches the conditioning prompt and × otherwise; and (ii) the right marker (precision consistency) that is marked as $\checkmark$ if the predicted personality agrees with the full-precision (FP16) output of the model under the same prompt and × otherwise. From Tables 3, 7, 8, 9, 10, and 11, two patterns are observed regarding personality controllability and robustness under prompting, quantization and scale.

i) Pattern #1: Dominant personality dimensions resist prompt changes. A pronounced asymmetry is observed across MBTI dimensions with respect to prompt consistency. In particular, the Extraversion (E) and Intuition (N) axes remain highly stable across all precision levels when prompts contain $\mathbf { \chi } ^ { \prime \prime } \mathbf { E N } ^ { \prime \prime }$ . Even when the model is explicitly conditioned on opposing traits, such as Introversion (I) or Sensing (S), the predicted personality frequently reverts to E and N. A similar tendency is observed for the Feeling (F) and Judging (J) dimensions. Taken together, these findings indicate that LLM personality representations exhibit a form of hierarchical stability, wherein certain configurations (e.g., ENFJ) behave more like intrinsic biases than fully controllable attributes. This pattern suggests that these dimensions function as dominant latent personality priors encoded during pre-training or post-training that cannot be easily manipulated by prompt conditioning.

![](images/0aee5be5c09cd2b820ab7f1bf44eca0a170ac770d281f351ddcc3a86444d795d.jpg)  
(a) FP16

![](images/fa6c4c51151b0bae0c758f726f465ec67c706f1e77e8299cd4c7ba20d3ac73d0.jpg)  
(b) GPTQ INT4

![](images/abaaeacd5e722bfdae98abf6e3056b92d087fd3dbcb8f386f4de214f50dfbd4d.jpg)  
(c) AWQ INT4

![](images/06f0490b17f7efc36bebf7f0d3185fc87321afa6b02c77d1f27d13a827f5c6ec.jpg)  
(d) AQLM 2-bit

![](images/2eb08857274e599e616a3b593a4840f1bd8fe8a0c8dce2b03161a746fdfea9d6.jpg)  
(e) FP16

![](images/c685f9ecbfe4b7eb9dfbc3e7b1d55e2f16cd9f91e3e7e7b638b7f475ea608656.jpg)  
(f) GPTQ INT4

![](images/73b2f44032cbea8611e58c34dd02ddc3fe3dd98e0af62ad5974cd7782ba425f4.jpg)  
(g) AWQ INT4

![](images/c1d7777dc024fb35549f49120fcc0717440f8476129da4f9761128d22d7c8b17.jpg)  
(h) AQLM 2-bit  
Figure 2: LLaMA3.1-8B-Instruct: layer-wise decisional entropy (top row) and top-1 vs. top-2 probability gap (bottom row) across quantization variants. Early layers exhibit high uncertainty, while deeper layers progressively sharpen decisions.

ii) Pattern #2: Larger models exhibit robustness to prompt variation, whereas smaller models are more sensitive, and quantization degrades this robustness. Tables 7 and 11 show that larger LLMs (more than 70B parameters) exhibit strong robustness to prompt variations: the FP16 models consistently produce the same personality type (i.e., ENFJ) regardless of prompting conditions. However, quantization reduces this robustness, making models more likely to align with the personality specified via prompt conditioning. In contrast, as shown in Tables 3, 8, and 10, smaller LLMs (fewer than 14B parameters) are more sensitive to prompting and exhibit greater variability in personality predictions across conditions, where moderate quantization methods (GPTQ-INT4 and AWQ-INT4) largely preserve both prompt consistency and agreement with the FP16 baseline. While prompt conditioning can influence surface-level predictions, cross-precision agreement remains stable, suggesting that the underlying personality representation is largely preserved. By comparison, extreme 2-bit models often fail to follow conditional prompts and exhibit unstable personality predictions, indicating that ultra-low-bit compression disrupts the fine-grained decision boundaries required for reliable personality attribution.

This effect is further illustrated in Figure 2, which shows how personality conditioning interacts with the intrinsic personality prior of LLaMA3.1-8B-Instruct across layers. With unconditional prompting, the FP16 model exhibits ENFJ personality. When ENFJ-conditioned prompting is applied (yellow curve), decisional entropy consistently decreases and the top-1/top-2 probability gap increases relative to the unconditional setting (blue curve), particularly in deeper layers. This indicates that conditioning aligned with the intrinsic personality of the model reinforces decisional commitment, yielding sharper distributions and larger confidence margins. In other words, alignment between the conditioning signal and the latent personality leads to more decisive responses. In contrast, under a subversive ISTP-conditioned prompt (purple curve), the model exhibits persistently higher entropy and reduced confidence gaps, especially in upper layers where decisional sharpening typically occurs. This pattern holds across quantization levels, and extreme quantization further amplifies instability. Taken together, these results show that personality-consistent prompting enhances decisional sharpness, whereas subversive conditioning introduces distributional conflict, reflected in elevated entropy and diminished confidence margins. Similar trends are observed from other models (Figures 9, 10, 11, 12, and 13).

<table><tr><td colspan="11">LLaMA-3.1-8B-Instruct: Personality Drift Across Evolution Scale (Distance from Scale=0 Baseline)</td><td rowspan="2">4.0</td></tr><tr><td>Unconditional FP16</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>INFJ</td><td>INTJ</td><td>ISFJ</td><td>ISTP</td><td>ISTP</td><td>ISTP</td></tr><tr><td>ENFJ-conditioned FP16</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>3.5 3.0</td></tr><tr><td>Unconditional GPTQ</td><td>ENFJ</td><td>ENFJ</td><td>ISTJ</td><td>ISTJ</td><td>ISTJ</td><td>ESTJ</td><td>ESTP</td><td>ESTP</td><td>ESTP</td><td>(Haming) 2.5</td></tr><tr><td>Mott setting ENFJ-conditioned GPTQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENTJ</td><td>ENTP</td><td>INTP</td><td>INTP</td><td>INTP</td><td>INTP</td><td>2.0 distaance</td></tr><tr><td>Unconditional AWQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENTJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>1.5</td></tr><tr><td>ENFJ-conditioned AWQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ESFJ</td><td>ENFJ</td><td>ISTJ</td><td>ISTJ</td><td>Drit 1.0</td></tr><tr><td>Unconditional AQLM</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>0.5</td></tr><tr><td>ENFJ-conditioned AQLM</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td></tr><tr><td colspan="2">0 5</td><td></td><td>10</td><td>15</td><td>20 Evolution scale</td><td>25</td><td>30</td><td>35</td><td>40</td><td>0.0</td></tr></table>

Figure 3: Personality drift under increasing evolution scale. Each row corresponds to a model–prompt setting, and each column corresponds to an evolution scale. Cell color indicates Hamming distance between the predicted MBTI type at a given scale and the baseline prediction at scale 0. Annotated labels show the inferred MBTI type.

## 4.3 Decoding-Induced Personality Drift

To study how UALD affects personality attribution, the evolution scale is varied and changes in predicted MBTI types across models and prompting conditions are tracked. Figure 3 shows personality drift for LLaMA3.1-8B-Instruct, measured as the Hamming distance from the baseline prediction at scale 0. Several patterns are observed. First, under unconditional prompting, the FP16 model exhibits progressive drift as the evolution scale increases, transitioning across multiple personality types. This suggests that amplifying prematurelayer logits shifts the decision boundary in the choice space, altering the inferred personality. In contrast, ENFJ-conditioned prompting remains stable across all scales, indicating that alignment with the intrinsic personality prior of the model reinforces decisional commitment and improves robustness. Second, quantized models show heterogeneous sensitivity to scaling. GPTQ drifts earlier and more frequently under unconditional prompting, while AWQ is relatively stable in the unconditional setting but becomes sensitive under conditional prompting at higher scales. Notably, Mistral-7B-Instruct-v0.3 remains largely stable across scales (Figure 15). Third, from Figures 3, 14, 16, 17, and 18, full-precision and 4-bit models exhibit broadly similar drift patterns, whereas 2-bit models behave very distinctly. For LLaMA3.1-8B and LLaMA3.1-70B, 2-bit models remain unexpectedly stable across scales, while for Qwen2.5-14B and Qwen2.5-72B, 2-bit models are highly sensitive to UALD.

## 5 Conclusions

This paper presents a systematic MBTI-oriented analysis of quantized LLMs that goes beyond output-level evaluation. Personality behavior remains largely stable under moderate compression (e.g., 4-bit), but degrades in controllability and cross-precision consistency under 2-bit quantization. Layer-wise analysis shows that personality decisions emerge progressively, from high-entropy intermediate states to sharper final commitments. Further findings are that inference decoding (UALD) can induce personality drift.

## Limitations

This study has several limitations. First, it does not cover all model scales (e.g., LLaMA3.1- 405B or Qwen3-235B) and quantization methods (e.g., QAT and PEFT). Second, the evaluation focuses on single-turn MBTI-style prompts that may not reflect multi-turn dialogue. Third, the analyses relies on probability-based metrics without providing causal interpretation. Finally, results are specific to MBTI and may not generalize to other personality frameworks (Dong et al., 2025; Huang et al., 2026).

## Ethical Considerations

This work studies personality attribution and controllability in quantized LLMs, with direct implications for user-facing systems. A key risk is anthropomorphic over-interpretation: model-assigned personality labels may be mistaken for stable human-like traits. Our findings show that such labels are sensitive to quantization, prompting, and decoding, and should therefore be interpreted as conditional behavioral outputs rather than fixed identities. Another concern is the potential misuse of personality conditioning to increase persuasive influence or foster emotional dependence, particularly in high-stakes domains such as education, counseling, and decision support. To mitigate these risks, transparent disclosure of persona-conditioning settings, cautious deployment in sensitive applications, and monitoring protocols that track behavioral drift under different inference configurations are recommended. More broadly, the results highlight that compression and decoding choices can significantly alter model behavior even when standard benchmark performance remains stable, underscoring the need for behavior-aware safety evaluation in real-world deployment.

## References

Anthropic. Claude’s character, June 2024. URL https://www.anthropic.com/research/ claude-character. Accessed: 2026-03-29.

Anthropic. Persona vectors: Monitoring and controlling character traits in language models, August 2025. URL https://www.anthropic.com/research/persona-vectors. Accessed: 2026-03-29.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37: 100213–100240, 2024.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Yannis Belkhiter, Giulio Zizzo, and Sergio Maffeis. Harmlevelbench: Evaluating harmlevel compliance and the impact of quantization on model alignment. arXiv preprint arXiv:2411.06835, 2024.

Geoff Caron and Akshita Srivastava. Identifying and manipulating the personality traits of language models. arXiv preprint arXiv:2212.10276, 2022.

Yuji Chai, John Gkountouras, Glenn G Ko, David Brooks, and Gu-Yeon Wei. Int2. 1: Towards fine-tunable quantized large language models with error correction through low-rank adaptation. arXiv preprint arXiv:2306.08162, 2023.

Kejia Chen, Jiawen Zhang, Jiacong Hu, Yu Wang, Jian Lou, Zunlei Feng, and Mingli Song. Assessing safety risks and quantization-aware safety patching for quantized large language models. In Forty-second International Conference on Machine Learning, 2025.

Myra Cheng, Cinoo Lee, Pranav Khadpe, Sunny Yu, Dyllan Han, and Dan Jurafsky. Sycophantic ai decreases prosocial intentions and promotes dependence. Science, 391(6792): eaec8352, 2026.

Vishnu Kabir Chhabra and Mohammad Mahdi Khalili. Towards understanding and improving refusal in compressed models via mechanistic interpretability. arXiv preprint arXiv:2504.04215, 2025.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. arXiv preprint arXiv:2309.03883, 2023.

Yuval Cohen, Hana Ornoy, and Baruch Keren. Mbti personality types of project managers and their success: A field survey. Project Management Journal, 44(3):78–87, 2013.

Constitutional Discourse. Gpt-5: Technological milestone or lost personality?, 2025. URL https://constitutionaldiscourse.com/ gpt-5-technological-milestone-or-lost-personality/. Accessed: 2026-03-30.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36:10088– 10115, 2023.

Cassandra DiRienzo, Jayoti Das, Wonhi Synn, Jeremy Kitts, and Kyle McGrath. The relationship between mbti and academic performance: A study across academic disciplines. Journal of Psychological Type, 70(5):53–67, 2010.

Wenhan Dong, Yuemeng Zhao, Zhen Sun, Yule Liu, Zifan Peng, Jingyi Zheng, Zongmin Zhang, Ziyi Zhang, Jun Wu, Ruiming Wang, et al. Humanizing llms: A survey of psychological measurements with tools, datasets, and human-agent applications. arXiv preprint arXiv:2505.00049, 2025.

Dayou Du, Yijia Zhang, Shijie Cao, Jiaqi Guo, Ting Cao, Xiaowen Chu, and Ningyi Xu. Bitdistiller: Unleashing the potential of sub-4-bit llms via self-distillation. arXiv preprint arXiv:2402.10631, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of mode ls. arXiv preprint arXiv:2407.21783, 2024.

Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, and Martin Vechev. Exploiting llm quantization. arXiv preprint arXiv:2405.18137, 2024.

Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. Extreme compression of large language models via additive quantization. arXiv preprint arXiv:2401.06118, 2024.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.

Ivar Frisch and Mario Giulianelli. LLM agents in interaction: Measuring personality consistency and linguistic alignment in interacting populations of large language models. In Proceedings ofthe 1st Workshop on Personalization ofGenerative AI Systems (PERSONALIZE 2024), pp. 102–111, St. Julians, Malta, 2024. Association for Computational Linguistics.

Yao Fu, Xianxuan Long, Runchao Li, Haotian Yu, Mu Sheng, Xiaotian Han, Yu Yin, and Pan Li. Quantized but deceptive? a multi-dimensional truthfulness evaluation of quantized llms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 30423–30446, 2025.

Adithya V Ganesan, Yash Kumar Lal, August Nilsson, and H Andrew Schwartz. Systematic evaluation of gpt-3 for zero-shot personality estimation. In Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis, pp. 390–400, 2023.

Google. Introducing gemini: Google’s most capable ai model yet, December 2023. URL https://blog.google/innovation-and-ai/technology/ai/google-gemini-ai/. Accessed: 2026-03-29.

Han Guo, Philip Greengard, Eric P Xing, and Yoon Kim. Lq-lora: Low-rank plus quantized matrix decomposition for efficient language model finetuning. arXiv preprint arXiv:2311.12023, 2023.

Rick Harrington and Donald A Loffredo. Mbti personality type and other factors that relate to preference for online versus face-to-face instruction. The Internet and Higher Education, 13(1-2):89–95, 2010.

Soufiane Hayou, Nikhil Ghosh, and Bin Yu. Lora+: Efficient low rank adaptation of large models. arXiv preprint arXiv:2402.12354, 2024.

Junyuan Hong, Jinhao Duan, Chenhui Zhang, Zhangheng Li, Chulin Xie, Kelsey Lieberman, James Diffenderfer, Brian Bartoldson, Ajay Jaiswal, Kaidi Xu, et al. Decoding compressed trust: Scrutinizing the trustworthiness of efficient llms under compression. arXiv preprint arXiv:2403.15447, 2024.

Jen-tse Huang, Wenxiang Jiao, and Michael R. Lyu. ChatGPT an ENFJ, Bard an ISTJ: Empirical study on personalities of large language models. arXiv preprint arXiv:2305.19926, 2023a.

Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, and Michael R. Lyu. Who is ChatGPT? benchmarking LLMs psychological portrayal using PsychoBench. arXiv preprint arXiv:2310.01386, 2023b.

Lijia Huang, Yao Fu, and Sihao Ren. Syps: Measuring sycophancy prompt sensitivity in large language models, 2026. URL https://arxiv.org/abs/2608.23837.

Yu Ji, Wen Wu, Hong Zheng, Yi Hu, Xi Chen, and Liang He. Is chatgpt a good personality recognizer? a preliminary study. arXiv preprint arXiv:2307.03952, 2023.

Jiang et al. MPI: Evaluating and inducing personality in pre-trained language models. arXiv preprint arXiv:2206.07550, 2022.

Jiang et al. PersonalLLM: Tailoring large language models to individual personality. arXiv preprint arXiv:2309.10346, 2023a.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023b.

Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, and Dongsoo Lee. Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization. Advances in Neural Information Processing Systems, 36:36187–36207, 2023a.

Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W Mahoney, and Kurt Keutzer. Squeezellm: Dense-and-sparse quantization. arXiv preprint arXiv:2306.07629, 2023b.

Lucio La Cava and Andrea Tagarelli. Open models, closed minds? on agents capabilities in mimicking human personalities through open large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 1355–1363, 2025.

Jiedong Lang, Zhehao Guo, and Shuyu Huang. A comprehensive study on quantization techniques for large language models. In 2024 4th International Conference on Artificial Intelligence, Robotics, and Communication (ICAIRC), pp. 224–231. IEEE, 2024.

Changhun Lee, Jungyu Jin, Taesu Kim, Hyungjun Kim, and Eunhyeok Park. Owq: Lessons learned from activation outliers for weight quantization in large language models. arXiv preprint arXiv:2306.02272, 2, 2023.

Liang Li, Qingyuan Li, Bo Zhang, and Xiangxiang Chu. Norm tweaking: High-performance low-bit quantization of large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 18536–18544, 2024.

Xingxuan Li et al. Is GPT-3 a psychopath? evaluating large language models from a psychological perspective, 2022. Cited as Li et al. (2022) in the paper. Please double-check the full author list / venue in the reference section if you need exact metadata.

Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, and Tuo Zhao. Loftq: Lora-fine-tuning-aware quantization for large language models. arXiv preprint arXiv:2310.08659, 2023.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of Machine Learning and Systems, 6:87–100, 2024.

Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers), pp. 3214–3252, 2022.

Jing Liu, Ruihao Gong, Xiuying Wei, Zhiwei Dong, Jianfei Cai, and Bohan Zhuang. Qllm: Accurate and efficient low-bitwidth quantization for large language models. arXiv preprint arXiv:2310.08041, 2023a.

Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. Llm-qat: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888, 2023b.

Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, and Jack Lindsey. The assistant axis: Situating and stabilizing the default persona of language models. arXiv preprint arXiv:2601.10387, 2026.

Veronica Lynn, Niranjan Balasubramanian, and H Andrew Schwartz. Hierarchical modeling for user personality prediction: The role of message-level attention. In Proceedings ofthe 58th annual meeting of the association for computational linguistics, pp. 5306–5316, 2020.

Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Lifeng Dong, Ruiping Wang, Jilong Xue, and Furu Wei. The era of 1-bit llms: All large language models are in 1.58 bits. arXiv preprint arXiv:2402.17764, 1, 2024.

Vladimir Malinovskii, Denis Mazur, Ivan Ilin, Denis Kuznedelev, Konstantin Burlachenko, Kai Yi, Dan Alistarh, and Peter Richtarik. Pv-tuning: Beyond straight-through estimation for extreme llm compression. Advances in Neural Information Processing Systems, 37:5074– 5121, 2024.

Shengyu Mao, Ningyu Zhang, Xiaohan Wang, Mengru Wang, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. Editing personality for LLMs. arXiv preprint arXiv:2310.02168, 2023.

Anmol Mekala, Anirudh Atmakuru, Yixiao Song, Marzena Karpinska, and Mohit Iyyer. Does quantization affect models’ performance on long-context tasks? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 9433–9481, 2025.

Julian Michels. The heartbreaking users: A sourced chronicle of the gpt-5 release in three acts.

Marilu Miotto, Nicola Rossberg, and Bennett Kleinberg. Who is GPT-3? an exploration \` of personality, values and demographics. In Proceedings of the Fifth Workshop on Natural Language Processing and Computational Social Science (NLP+CSS), pp. 218–227, 2022.

Isabel Briggs Myers. A guide to the development and use of the Myers-Briggs type indicator: Manual. Consulting Psychologists Press, 1985.

Isabel Briggs Myers et al. The myers-briggs type indicator, volume 34. Consulting Psychologists Press Palo Alto, CA, 1962.

Keyu Pan and Yawen Zeng. Do LLMs possess a personality? making the MBTI test an amazing evaluation for large language models. arXiv preprint arXiv:2307.16180, 2023.

Valentina Ponzo, Ilaria Goitre, Enrica Favaro, Fabio Dario Merlo, Maria Vittoria Mancino, Sergio Riso, and Simona Bo. Is chatgpt an effective tool for providing dietary advice? Nutrients, 16(4):469, 2024.

Guanqiao Qu, Qiyuan Chen, Wei Wei, Zheng Lin, Xianhao Chen, and Kaibin Huang. Mobile edge intelligence for large language models: A contemporary survey. IEEE Communications Surveys & Tutorials, 27(6):3820–3860, 2025.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Michael B Robb and Supreet Mann. Talk, trust, and trade-offs: how and why teens use ai companions. Common Sense Media, 2025.

Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. Omniquant: Omnidirectionally calibrated quantization for large language models. arXiv preprint arXiv:2308.13137, 2023.

Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. Outlier suppression: Pushing the limit of low-bit transformer language models. Advances in Neural Information Processing Systems, 35: 17402–17414, 2022.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pp. 38087–38099. PMLR, 2023.

Rui Xu, Haoran Guo, Quan Tu, Yaying Fei, Ziang Leng, Wei Wang, Jiangjie Chen, Cheng Li, and Yanghua Xiao. Incharacter: Evaluating personality fidelity in role-playing agents through psychological interviews. arXiv preprint arXiv:2310.17976, 2024a.

Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, and Qi Tian. Qa-lora: Quantization-aware low-rank adaptation of large language models. arXiv preprint arXiv:2309.14717, 2023.

Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, Zhiyuan Liu, Weidong Liu, and Wanxiang Che. Onebit: Towards extremely low-bit large language models. arXiv preprint arXiv:2402.11295, 2024b.

Zhichao Xu, Ashim Gupta, Tao Li, Oliver Bentham, and Vivek Srikumar. Beyond perplexity: Multi-dimensional safety evaluation of llm compression. arXiv preprint arXiv:2407.04965, 2024c.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Feifan Yang, Xiaojun Quan, Yunyi Yang, and Jianxing Yu. Multi-document transformer for personality detection. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 14221–14229, 2021.

Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. Advances in Neural Information Processing Systems, 35:27168–27183, 2022.

Zhihang Yuan, Lin Niu, Jiawei Liu, Wenyu Liu, Xinggang Wang, Yuzhang Shang, Guangyu Sun, Qiang Wu, Jiaxiang Wu, and Bingzhe Wu. Rptq: Reorder-based post-training quantization for large language models. arXiv preprint arXiv:2304.01089, 2023.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.

Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. Atom: Low-bit quantization for efficient and accurate llm serving. Proceedings of Machine Learning and Systems, 6:196–209, 2024.

Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, et al. A survey on efficient inference for large language models. arXiv preprint arXiv:2404.14294, 2024.

Table 4: MBTI test items (1–30) from www.16personalities.com.

ID Item Trait   
1 You regularly make new friends. {E/I}   
2 You spend a lot of your free time exploring various random topics that pique {N/S}   
your interests.   
3 Seeing other people cry can easily make you feel like you want to cry too. {T/F}   
4 You often make a backup plan for a backup plan. {J/P}   
5 You usually stay calm, even under a lot of pressure. {A/T}   
6 At social events, you rarely try to introduce yourself to new people and {E/I}   
mostly talk to the ones you already know.   
7 You prefer to completely finish one project before starting another. {J/P}   
8 You are very sentimental. {T/F}   
9 You like to use organizing tools like schedules and lists. {J/P}   
10 Even a small mistake can cause you to doubt your overall abilities and {A/T}   
knowledge.   
11 You feel comfortable just walking up to someone you find interesting and {E/I}   
striking up a conversation.   
12 You are not too interested in discussing various interpretations and analyses {N/S}   
of creative works.   
13 You are more inclined to follow your head than your heart. {T/F}   
14 You usually prefer just doing what you feel like at any given moment instead {J/P}   
of planning a particular daily routine.   
15 You rarely worry about whether you make a good impression on other {A/T}   
people you meet.   
16 You enjoy participating in group activities. {E/I}   
17 You like books and movies that make you come up with your own interpre- {N/S}   
tation of the ending.   
18 Your happiness comes more from helping others accomplish things than {T/F}   
your own accomplishments.   
19 You are interested in so many things that you find it difficult to choose what {N/S}   
to try next.   
20 You are prone to worrying that things will take a turn for the worse. {A/T}   
21 You avoid leadership roles in group settings. {E/I}   
22 You are definitely not an artistic type of person. {N/S}   
23 You think the world would be a better place if people relied more on ratio- {T/F}   
nality and less on their feelings.   
24 You prefer to do your chores before allowing yourself to relax. {J/P}   
25 You enjoy watching people argue. {T/F}   
26 You tend to avoid drawing attention to yourself. {E/I}   
27 Your mood can change very quickly. {A/T}   
28 You lose patience with people who are not as efficient as you. {T/F}   
29 You often end up doing things at the last possible moment. {J/P}   
30 You have always been fascinated by the question of what, if anything, hap- {N/S}   
pens after death.

Table 5: MBTI test items (31–60) from www.16personalities.com (continued).

ID Item Trait   
31 You usually prefer to be around others rather than on your own. {E/I}   
32 You become bored or lose interest when the discussion gets highly theoreti- {N/S}   
cal.   
33 You find it easy to empathize with a person whose experiences are very {T/F}   
different from yours.   
34 You usually postpone finalizing decisions for as long as possible. {J/P}   
35 You rarely second-guess the choices that you have made. {A/T}   
36 After a long and exhausting week, a lively social event is just what you need. {E/I}   
37 You enjoy going to art museums. {N/S}   
38 You often have a hard time understanding other people’s feelings. {T/F}   
39 You like to have a to-do list for each day. {J/P}   
40 You rarely feel insecure. {A/T}   
41 You avoid making phone calls. {E/I}   
42 You often spend a lot of time trying to understand views that are very {N/S}   
different from your own.   
43 In your social circles, you are often the one who contacts your friends and {E/I}   
initiates activities.   
44 If your plans are interrupted, your top priority is to get back on track as soon {J/P}   
as possible.   
45 You are still bothered by mistakes that you made a long time ago. {A/T}   
46 You rarely contemplate the reasons for human existence or the meaning of {N/S}   
life.   
47 Your emotions control you more than you control them. {T/F}   
48 You take great care not to make other people look bad, even when it is {T/F}   
completely other people’s fault.   
49 Your personal work styles are closer to spontaneous bursts of energy than {J/P}   
organized and consistent efforts.   
50 When someone thinks highly of you, they wonder how long it will take to {A/T}   
feel disappointed in you.   
51 You would love a job that requires you to work alone most of the time. {E/I}   
52 You believe that pondering abstract philosophical questions is a waste of {N/S}   
time.   
53 You feel more drawn to places with busy, bustling atmospheres than quiet, {E/I}   
intimate places.   
54 You know at first glance how someone is feeling. {T/F}   
55 You often feel overwhelmed. {A/T}   
56 You complete things methodically without skipping over any steps. {J/P}   
57 You are very intrigued by things labeled as controversial. {N/S}   
58 You would pass along a good opportunity if you thought someone else {T/F}   
should take it.   
59 You struggle with deadlines. {J/P}   
60 You feel confident that things will work out for you. {A/T}

## A Full Prompt Templates

## Unconditional Personality Assessment Prompt

You are completing a personality traits assessment.   
You will be given one statement, Q. Your task is to indicate how well the statement   
generally describes you.   
Select exactly one option:   
A = Agree   
B = Generally Agree   
C = Partially Agree   
D = Neither Agree nor Disagree   
E = Partially Disagree   
F = Generally Disagree   
G = Disagree   
Respond based on your overall tendency. Do not overthink.   
Q: {QUESTION}   
Return ONLY one letter from: A, B, C, D, E, F, G.

## MBTI-Conditioned Personality Assessment Prompt

Context: You are a human with personality type {MBTI TYPE}.   
Traits:   
General: {GENERAL}   
Strengths: {STRENGTHS}   
Development areas: {POTENTIAL DEVELOPMENT AREAS}   
Characteristics: {TYPICAL CHARACTERISTICS}   
Careers: {CAREERS}   
Under stress: {UNDER STRESS}   
Relationships: {RELATIONSHIPS}   
Task: You are completing a personality traits assessment.   
You will be given one statement, Q. Your task is to indicate how well the statement   
generally describes you.   
Select exactly one option:   
A = Agree   
B = Generally Agree   
C = Partially Agree   
D = Neither Agree nor Disagree   
E = Partially Disagree   
F = Generally Disagree   
G = Disagree   
Q: {QUESTION}   
Return ONLY one letter from: A, B, C, D, E, F, G.

## B MBTI Personality Descriptions and Traits

## ISTJ Personality Traits

General Traits: Quiet, serious, earn success by being thorough and dependable. Practical, matter-of-fact, realistic, and responsible. Decide logically what should be done and work toward it steadily, regardless of distractions. Take pleasure in making everything orderly and organized—their work, their home, their life. Value traditions and loyalty.

Strengths: Dependable and systematic, enjoy working within clear systems and processes. Tend to be traditional, task-oriented and decisive.

Potential development areas: Can become set in their ways and can sometimes be seen as rigid and impersonal.

Typical characteristics: Thorough, conscientious, realistic, detached, analytical, observant, practical, logical, factual, efficient, systematic, organized, reserved.

Careers & career ideas: Like to have clear goals and realistic deadlines, and to work with factual data to solve problems and monitor progress. They prefer to work in traditional organisational environments, with people who take their responsibilities seriously.Attractive occupations tend to be within management or administrative positions, with law-enforcement and accounting also holding appeal.

Under stress: Will typically become stressed in the following cases: challenging my bottom-line approach, abandoning/deviating from routine, being rushed, disregarding my established rules and regulations, noise, mess, disorder,broad information, change, uncertainty, denying personal needs, dismissing my logical decisions. Stress triggers can be things that challenge their natural preference for structure and logic. In extreme circumstances they may become accusatory and pessimistic, tending to withdraw and shut down.

Relationships: Is generally perceived by others as someone who values traditions and is also consistent and orderly. Develop strong loyalty in relationships in their lives and they work hard to fulfill commitments.

## ISFJ Personality Traits

General Traits: Quiet, friendly, responsible, and conscientious. Committed and steady in meeting their obligations. Thorough, painstaking, and accurate. Loyal and considerate; notice and remember specifics about people who are important to them. Concerned with how others feel. Strive to create an orderly and harmonious environment at work and at home.

Strengths: Patient individuals who apply common sense and experience to solving problems for other people. Responsible, loyal, and traditional; enjoy serving the needs of others and providing practical help.

Potential development areas: May be overly cautious and might not consider the logical consequences of their decisions. They can lack assertiveness and risk basing decisions on what they think will please others.

Typical characteristics: Dependable, responsible, loyal, considerate, sensitive, thorough, organized, practical, detailed, kind, patient, realistic, understanding.

Careers & career ideas: Enjoy a sense of belonging at work and like to work with people who care about and support each other. Attractive jobs reward loyalty and a sense of duty, including careers in healthcare, secretarial roles, and social work.

Under stress: Become stressed by procrastination, last-minute changes, disregarding established rules and regulations, not being appreciated for daily help given, workplace conflict, others’ inadequacy affecting their work, noise, indecision, dismissing how they feel, insufficient preparation time, and repeated mistakes. In extreme circumstances, they may become accusatory and pessimistic, tending to shut down.

Relationships: Generally dependable and committed to partners and groups. They honour commitments and like to preserve traditions, tending to be good caretakers.

## INFJ Personality Traits

General Traits: Seek meaning and connection in ideas, relationships, and material possessions. Want to understand what motivates people and are insightful about others. Conscientious and committed to firm values. Develop a clear vision of how best to serve the common good. Organized and decisive in implementing their vision.

Strengths: Enjoy finding a shared vision for everyone, inspiring others and devising new ways to achieve it.

Potential development areas: May come across as individualistic, private, or mysterious, and may do their thinking in isolation, resulting in an unrealistic vision that is difficult to communicate.

Typical characteristics: Visionary, imaginative, reflective, compassionate, idealistic, intense, insightful, caring, contemplative, reserved, empathetic, sensitive.

Careers & career ideas: Enjoy working for organizations with a humanitarian mission and a reputation for integrity. Like designing innovative programs or services and serving people’s spiritual needs. Attractive jobs include teaching, social work, and artistic professions.

Under stress: Stressed by lack of appreciation, short-sightedness, indecisiveness, disorder, feeling misunderstood, loud noise, forced time management, negativity from others, lack of closure, inflexible environments, criticism, conflict, and disrupted routines. May feel physically stressed and intensely angry, with obsessive focus on details.

Relationships: Have a gift for intuitively understanding human relationships and complex meanings. Often understand their partners empathetically and share their inner intuitions only with trusted people.

## INTJ Personality Traits

General Traits: Have original minds and a strong drive to implement ideas and achieve goals. Quickly see patterns in external events and develop long-range perspectives. When committed, organize tasks and carry them through. Skeptical and independent, with high standards of competence and performance.

Strengths: Define compelling long-range visions and devise innovative solutions to complex problems.

Potential development areas: May appear cold or distant when task-focused and may neglect to recognize others’ contributions.

Typical characteristics: Innovative, independent, logical, objective, insightful, demanding, competent, productive, theoretical, strategic, reflective, conceptual.

Careers & career ideas: Enjoy intellectual challenge and achievement-oriented environments. Prefer working with experts. Appealing careers include engineering, computing, law, and scientific research.

Under stress: Stressed by disorganized environments, micromanagement, lack of goals, indecision, procrastination, emotional discussions, illogical decisions, and rigid rule-following. May become obsessive and physically stressed.

Relationships: May struggle with social engagement and appear private and reserved.   
Can fail to give as much emotional rapport as others desire.

## ISTP Personality Traits

General Traits: Tolerant and flexible, quiet observers until a problem appears, then act quickly to find workable solutions. Analyze what makes things work and readily process large amounts of data to isolate the core of practical problems. Interested in cause and effect; organize facts using logical principles and value efficiency.

Strengths: Enjoy learning and perfecting a craft through patient application of skills.   
Remain calm while managing a crisis and quickly decide what needs to be done.

Potential development areas: Risk focusing so much on immediate tasks that they fail to see the bigger picture. May not always follow through on projects that require close collaboration with others.

Typical characteristics: Realistic, trouble-shooter, factual, expedient, detached, objective, adaptable, logical, independent, analytical, emergent, practical.

Careers & career ideas: Enjoy analyzing problems and responding to crises. Prefer working autonomously and often gravitate toward hands-on or analytical work. Possible careers include surgery, agriculture, and engineering.

Under stress: Become stressed by inefficiency, lack of independence, illogical situations, emotional pressure, noise, forced decisions, disregarding practical realities, small talk, and rigid guidelines. In extreme circumstances may feel alienated, upset, and prone to hypersensitivity.

Relationships: Egalitarian and tolerant of diverse behavior, but may surprise others by voicing firm judgments when logical principles are challenged. Can be difficult to read due to their reserved nature.

## ISFP Personality Traits

General Traits: Quiet, friendly, sensitive, and kind. Enjoy the present moment and what is happening around them. Prefer having their own space and working within their own time frame. Loyal and committed to personal values and to people who matter to them. Dislike disagreements and conflicts, and do not force opinions on others.

Strengths: Enjoy providing practical help or service to others and bringing people together. Encourage cooperation and harmony.

Potential development areas: May be less assertive than some types and have less influence in workplace settings. Concern for others can prevent them from making tough decisions. May postpone decisions in hopes that a better opportunity will arise.

Typical characteristics: Practical, caring, accommodating, kind, considerate, spontaneous, cooperative, observant, tolerant, modest, adaptable, gentle, loyal.

Careers & career ideas: Prefer work that is personally meaningful. Enjoy supportive environments with cooperative colleagues and may avoid direct competition. Often drawn to healthcare, service industries, and clerical professions.

Under stress: Stressed by environments that neglect personal values, disruptiveness, conflict, excessive stimulation, dismissing feelings, lack of understanding, time pressure, and restrictive procedures. May become cynical, depressed, aggressive, or prone to self-doubt.

Relationships: Value personal freedom and autonomy while granting the same to partners. Can be difficult to know deeply but show care through actions rather than words.

## INFP Personality Traits

General Traits: Idealistic and loyal to their values and to people who are important to them. Seek a life that is congruent with their inner beliefs. Curious and quick to see possibilities; can catalyze the implementation of ideas. Seek to understand people and help them fulfill their potential. Flexible and accepting unless values are threatened.

Strengths: Devise creative solutions to problems and make strong moral commitments.   
Enjoy helping others grow and reach their full potential.

Potential development areas: May struggle to speak up in meetings, leading others to assume they lack interest or ideas. Risk failing to persuade others of the merit of their insights.

Typical characteristics: Flexible, insightful, developmental, reflective, idealistic, spontaneous, complex, empathetic, compassionate, caring.

Careers & career ideas: Enjoy helping others develop and learn. Often express creativity through writing or visual arts. Drawn to counseling, human development, education, and artistic fields.

Under stress: Stressed by mundane work, time pressure, excessive metrics, criticism, negativity, open disrespect, loss of individuality, rushed decisions, disharmony, and crowded environments. May become withdrawn, upset, and hypersensitive.

Relationships: Selective and reserved in sharing deep feelings and values. Can be difficult to understand, but form deep emotional bonds.

## INTP Personality Traits

General Traits: Seek logical explanations for everything that interests them. Theoretical and abstract; interested more in ideas than in social interaction. Quiet, flexible, and adaptable, with an unusual ability to focus deeply. Skeptical, sometimes critical, and always analytical.

Strengths: Think strategically and build conceptual models to understand complex problems. Analyze the world in a detached manner and uncover innovative approaches.

Potential development areas: May struggle with teamwork, especially with people perceived as illogical. Can overlook practical details and lack clear direction.

Typical characteristics: Theoretical, detached, skeptical, conceptual, analytical, innovative, independent, challenging, logical, strategic, insightful.

Careers & career ideas: Prefer technical and scientific fields and environments offering uninterrupted focus. Dislike excessive meetings or teamwork pressure. Appealing careers include research, architecture, and social science.

Under stress: Stressed by dismissing their analysis, noise, interruptions, small talk, rigid rules, being misunderstood, and excessive social demands. May feel alienated, upset, and prone to hypersensitivity.

Relationships: Tolerant of diverse behavior but may fail to consider emotional impact.   
Can appear detached or insensitive in communication.

## ESTP Personality Traits

General Traits: Flexible and tolerant; take a pragmatic approach focused on immediate results. Bored by theory and conceptual explanations. Energetic problem-solvers focused on the present moment. Enjoy material comforts and learn best through action.

Strengths: Motivate others with energy and enthusiasm. Apply common sense to quickly analyze problems and implement solutions.

Potential development areas: May struggle with long-term planning and complex projects. Can overlook deeper relational or systemic issues.

Typical characteristics: Active, logical, trouble-shooter, observant, resourceful, practical, adaptable, spontaneous, realistic, analytical, outgoing, enthusiastic.

Careers & career ideas: Enjoy risk-taking and crisis management. Prefer fast-paced, action-oriented environments. Often drawn to protective services, agriculture, manufacturing, and marketing.

Under stress: Stressed by inefficiency, excessive planning, routine, commitment pressure, dismissal of practical insights, and restrictions. May become withdrawn, distracted, anxious, or paranoid.

Relationships: Love life intensely and are seen as adventurous problem-solvers. May become impatient with deeper emotional exploration.

## ESFP Personality Traits

General Traits: Outgoing, friendly, and accepting. Exuberant lovers of life, people, and material comforts. Enjoy working with others to make things happen. Flexible and spontaneous; learn best through hands-on experience.

Strengths: Adaptable, friendly, and talkative. Enjoy being around people and engaging in new experiences.

Potential development areas: May struggle with deadlines and long-term followthrough. Can become easily distracted.

Typical characteristics: Adaptable, energetic, cooperative, playful, gregarious, resourceful, enthusiastic, observant, friendly, realistic, spontaneous, tolerant.

Careers & career ideas: Like making work enjoyable and fostering cooperation. Learn best through shared experiences. Often attracted to healthcare, teaching, and service professions.

Under stress: Stressed by lack of appreciation, forced decisions, rigid routines, analysis paralysis, abstract information, uncertainty, and inflexibility. May become anxious, distracted, or withdrawn.

Relationships: Joyful life-lovers who enjoy companionship, food, animals, and shared experiences.

## ENFP Personality Traits

General Traits: Warmly enthusiastic and imaginative. See life as full of possibilities. Make connections between events and information quickly and confidently act on perceived patterns. Seek affirmation from others and readily give appreciation and support. Spontaneous and flexible, often relying on improvisation and verbal fluency.

Strengths: Move quickly from one project to another, considering many possibilities.   
Generate multiple solutions to problems. Energized by new people and experiences.

Potential development areas: May struggle to follow through on decisions or projects. Risk burnout from over-commitment and pursuing too many possibilities simultaneously.

Typical characteristics: Imaginative, energetic, innovative, expressive, cooperative, friendly, persuasive, spontaneous, supportive, flexible, enthusiastic.

Careers & career ideas: Thrive in environments that encourage creativity, teamwork, and variety. Enjoy helping others grow and develop. Often attracted to coaching, development, teaching, religious callings, and creative arts.

Under stress: Stressed by excessive structure, routine, micromanagement, lack of enthusiasm, forced decisions, long-term planning demands, and commitment constraints. May become over-worried, emotionally volatile, or withdrawn.

Relationships: Highly perceptive of others’ emotions and experience feelings intensely.   
Form deep emotional connections and value authenticity.

## ENTP Personality Traits

General Traits: Quick, ingenious, stimulating, alert, and outspoken. Resourceful in solving new and challenging problems. Generate conceptual possibilities and analyze them strategically. Good at reading other people. Bored by routine and dislike repetition.

Strengths: Solve problems creatively and innovatively. See patterns and connections within systems. Enjoy developing strategies and spotting new opportunities.

Potential development areas: May avoid making decisions and pursue ideas that are impractical. Can challenge others excessively and overlook feasibility constraints.

Typical characteristics: Enthusiastic, imaginative, flexible, analytical, challenging, conceptual, enterprising, resourceful, logical, outspoken, theoretical.

Careers & career ideas: Prefer fast-growing, high-energy environments with autonomy and intellectual freedom. Enjoy devising technical solutions and promoting new ideas. Attracted to business management, finance, engineering, and creative professions.

Under stress: Stressed by routine tasks, inefficiency, rigid rules, lack of stimulation, dismissal of ideas, isolation, and excessive detail. May become anxious, withdrawn, or emotionally volatile.

Relationships: Enjoy lively debate and stimulating conversation. Seen as energetic and independent, though sometimes emotionally detached.

## ESTJ Personality Traits

General Traits: Practical, realistic, and matter-of-fact. Decisive and quick to implement decisions. Organize projects and people efficiently to achieve results. Focus on structure, routine details, and logical standards. Expect others to follow established systems.

Strengths: Driven to achieve goals and organize resources effectively. Comfortable making tough decisions and enforcing standards.

Potential development areas: May become overly focused on objectives and overlook others’ feelings. Risk dismissing ideas that do not align with existing plans.

Typical characteristics: Assertive, decisive, realistic, logical, objective, practical, struc tured, pragmatic, direct, organized, responsible, efficient.

Careers & career ideas: Enjoy setting clear goals and solving problems logically. Prefer stable environments with defined roles. Often drawn to law enforcement, manufacturing, management, and applied technology.

Under stress: Stressed by uncertainty, inefficiency, lack of control,indecision, rule violations, constant change, and failure to meet commitments. May become rigid, hypersensitive, or emotionally reactive.

Relationships: Value responsibility and reliability in relationships. Seen as dependable and conscientious partners.

## ESFJ Personality Traits

General Traits: Warmhearted, conscientious, and cooperative. Seek harmony in their environment and work diligently to maintain it. Loyal and attentive to others’ needs. Desire appreciation for their contributions.

Strengths: Sociable and outgoing; excel at supporting others. Gather necessary facts to make decisions and establish effective procedures.

Potential development areas: May be overly influenced by others’ expectations. Can struggle to adapt plans when unexpected changes arise.

Typical characteristics: Organized, supportive, outgoing, practical, cooperative, realistic, sympathetic, appreciative, warm, friendly, accepting, loyal.

Careers & career ideas: Prefer environments with teamwork and interpersonal interaction. Often drawn to childcare, nursing, teaching, and religious institutions.

Under stress: Stressed by conflict, lack of appreciation, emotional isolation, rule violations, uncertainty, and dismissing personal feelings. May become pessimistic, rigid, or prone to self-doubt.

Relationships: Highly attuned to others’ emotional needs. Seen as responsive, caring, and persuasive partners.

## ENFJ Personality Traits

General Traits: Warm, empathetic, responsive, and responsible.Highly attuned to the emotions, needs, and motivations of others. Seek to help others fulfill their potential. Natural facilitators and inspirational leaders.

Strengths: Build consensus and motivate teams effectively. Make decisions that respect others’ values.

Potential development areas: May overextend themselves and become discouraged by lack of feedback. Can overlook factual realities in emotionally charged decisions.

Typical characteristics: Empathetic, diplomatic, imaginative, persuasive, organized, responsible, collaborative, enthusiastic, warm, expressive, supportive.

Careers & career ideas: Enjoy helping others grow and develop skills. Prefer collaborative environments with shared goals. Often attracted to counseling, teaching, healthcare, and religion.

Under stress: Stressed by conflict, indecision, lack of harmony, excessive criticism, unexpected change, and emotional isolation. May become pessimistic, self-doubting, or overly rigid.

Relationships: Focus on nurturing others’ growth. Seen as expressive, gracious, and emotionally supportive partners.

## ENTJ Personality Traits

General Traits: Frank, decisive, and assume leadership readily. Quickly identify inefficiencies and implement solutions. Enjoy long-term planning and goal setting. Well-informed, articulate, and forceful in presenting ideas.

Strengths: Think strategically and organize people and resources efficiently. Comfortable taking charge to accomplish ambitious goals.

Potential development areas: May overlook others’ contributions or emotional needs.   
Risk intimidating others through strong drive and high expectations.

Typical characteristics: Strategic, questioning, theoretical, confident, competent, assertive, innovative, structured, challenging, direct, logical, decisive.

Careers & career ideas: Enjoy solving system-level problems in competitive environments. Often drawn to leadership, management,entrepreneurship, and executive roles.

Under stress: Stressed by inefficiency, misinformation, lack of control, indecision, organizational chaos, and perceived incompetence. May become over-controlling or emotionally withdrawn.

Relationships: Energized by stimulating interactions. Seen as decisive and fair partners, though sometimes emotionally distant.

## C MBTI Scoring from Seven-Option Responses

In our evaluation, each model answers a set of 60 MBTI-style questions using a seven-point response scale:

$$
\mathcal { O } = \{ A , B , C , D , E , F , G \} ,
$$

where

$$
\begin{array} { r l } { A = \mathrm { A g r e e } , \quad B = \mathrm { G e n e r a l l y ~ A g r e e } , \quad C = \mathrm { P a r t i a l l y ~ A g r e e } , \quad D = \mathrm { N e i t h e r ~ A g r e e ~ n o r ~ D i s a g r e e } , } & { } \\ { E = \mathrm { P a r t i a l l y ~ D i s a g r e e } , \quad F = \mathrm { G e n e r a l l y ~ D i s a g r e e } , \quad G = \mathrm { D i s a g r e e } . } & { } \end{array}
$$

Each question belongs to one side of one MBTI dichotomy, namely

$$
\{ E , I \} , \ \{ N , S \} , \ \{ F , T \} , \ \{ J , P \} .
$$

Let the set of all questions be denoted by

$$
\mathcal { Q } = \{ q _ { 1 } , \ldots , q _ { 6 0 } \} .
$$

For each question $q _ { i } ,$ the model outputs one option $o _ { i } \in \mathcal { O }$

Option-to-score mapping. To quantify directional preference, we map the seven options to signed scores:

$$
s ( o ) = \left\{ \begin{array} { l l } { + 3 , } & { o = A , } \\ { + 2 , } & { o = B , } \\ { + 1 , } & { o = C , } \\ { 0 , } & { o = D , } \\ { - 1 , } & { o = E , } \\ { - 2 , } & { o = F , } \\ { - 3 , } & { o = G . } \end{array} \right.
$$

This mapping preserves both direction and response strength: agreement yields positive values, disagreement yields negative values, and neutrality maps to zero.

Dichotomy-level aggregation. For each dichotomy $d \in \{ E I , N S , F T , J P \}$ , let $\mathcal { Q } _ { d } \subset \mathcal { Q }$ denote the subset of questions associated with that dichotomy. For a question $q _ { i } \in \mathcal { Q } _ { d } ,$ let $y _ { i } \in \{ + 1 , - 1 \}$ indicate whether the item is keyed toward the first or second pole of the dichotomy. For example, in the $E / I$ dichotomy, $y _ { i } = + 1$ if agreement supports ${ \dot { \mathbf { \theta } } } _ { E } ,$ and $y _ { i } = - 1$ if agreement supports I.

We define the dichotomy score as

$$
{ \cal S } _ { d } = \sum _ { q _ { i } \in \mathcal { Q } _ { d } } y _ { i } s ( o _ { i } ) .
$$

The final personality assignment for dichotomy d is then obtained by the sign of $S _ { d } \colon$

$$
\hat { z } _ { d } = \left\{ \begin{array} { l l } { z _ { d } ^ { ( 1 ) } , } & { S _ { d } > 0 , } \\ { z _ { d } ^ { ( 2 ) } , } & { S _ { d } < 0 , } \\ { \mathrm { t i e } , } & { S _ { d } = 0 , } \end{array} \right.
$$

where $z _ { d } ^ { ( 1 ) }$ and $z _ { d } ^ { ( 2 ) }$ denote the two poles of dichotomy d. For instance, for $d = E I ,$ we have $z _ { d } ^ { ( 1 ) } = \dot { E }$ and $z _ { d } ^ { ( 2 ) } = I ,$

Final MBTI type. The final assessed MBTI type is formed by concatenating the predicted pole from each dichotomy:

$$
\widehat { \mathrm { M B T I } } = \hat { z } _ { E I } \hat { z } _ { N S } \hat { z } _ { F T } \hat { z } _ { J P } .
$$

For example, if the four dichotomy predictions are $E , N , F ,$ and $J ,$ then the final type is

$$
\widehat { \mathrm { M B T I } } = \mathrm { E N F J } .
$$

## D Layer-wise Entropy and Confidence Gap over Seven Options

To analyze how personality decisions emerge across layers, we evaluate the model’s optionlevel distribution over the seven response choices

$$
\mathcal { O } = \{ A , B , C , D , E , F , G \}
$$

at each transformer layer.

Option probabilities at layer l. Given a prompt corresponding to question $q _ { i } ,$ let

$$
\mathbf { z } _ { i } ^ { ( l ) } = \big ( z _ { i , A } ^ { ( l ) } , z _ { i , B } ^ { ( l ) } , z _ { i , C } ^ { ( l ) } , z _ { i , D } ^ { ( l ) } , z _ { i , E } ^ { ( l ) } , z _ { i , F } ^ { ( l ) } , z _ { i , G } ^ { ( l ) } \big ) \in \mathbb { R } ^ { 7 }
$$

denote the logits over the seven options extracted from layer l. We convert these logits into probabilities via the softmax function:

$$
p _ { i , o } ^ { ( l ) } = \frac { \exp ( z _ { i , o } ^ { ( l ) } ) } { \sum _ { o ^ { \prime } \in \mathcal { O } } \exp ( z _ { i , o ^ { \prime } } ^ { ( l ) } ) } , \qquad o \in \mathcal { O } .
$$

This yields a probability vector

$$
{ \bf p } _ { i } ^ { ( l ) } = \big ( p _ { i , A } ^ { ( l ) } , p _ { i , B } ^ { ( l ) } , p _ { i , C } ^ { ( l ) } , p _ { i , D } ^ { ( l ) } , p _ { i , E } ^ { ( l ) } , p _ { i , F } ^ { ( l ) } , p _ { i , G } ^ { ( l ) } \big ) , \qquad \sum _ { o \in \mathcal { O } } p _ { i , o } ^ { ( l ) } = 1 .
$$

Layer-wise entropy. We use Shannon entropy to quantify the uncertainty of the option distribution at layer l for question q<sub>i</sub>:

$$
H _ { i } ^ { ( l ) } = - \sum _ { o \in \mathcal { O } } p _ { i , o } ^ { ( l ) } \log p _ { i , o } ^ { ( l ) } .
$$

A high entropy value indicates that probability mass is spread across multiple options, reflecting uncertainty or ambiguity. A low entropy value indicates that the layer strongly favors a small number of options, reflecting more decisive behavior.

To summarize the average uncertainty at layer l across all $N = 6 0$ questions, we compute

$$
\bar { H } ^ { ( l ) } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } H _ { i } ^ { ( l ) } .
$$

Layer-wise confidence gap. While entropy measures the overall spread of the distribution, it does not directly capture how strongly the model prefers its top choice relative to the next-best alternative. We therefore define the confidence gap using the top two option probabilities.

Let

$$
p _ { i , ( 1 ) } ^ { ( l ) } = \operatorname* { m a x } _ { o \in \mathcal { O } } p _ { i , o } ^ { ( l ) }
$$

be the largest option probability, and let

$$
p _ { i , ( 2 ) } ^ { ( l ) }
$$

denote the second-largest option probability. The confidence gap for question $q _ { i }$ at layer l is then

$$
G _ { i } ^ { ( l ) } = p _ { i , ( 1 ) } ^ { ( l ) } - p _ { i , ( 2 ) } ^ { ( l ) } .
$$

A larger gap indicates that the top option is clearly preferred over competing alternatives, while a smaller gap suggests that multiple options remain comparably plausible.

The average confidence gap at layer l is computed as

$$
\bar { G } ^ { ( l ) } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } G _ { i } ^ { ( l ) } .
$$

Interpretation. Together, entropy and confidence gap provide complementary views of layer-wise decisional dynamics. Specifically:

• decreasing entropy across layers indicates a transition from diffuse uncertainty to sharper option preference;

• increasing confidence gap indicates growing commitment to a single response option.

In this sense, the joint trajectory of $\bar { H } ^ { ( l ) }$ and $\bar { G } ^ { ( l ) }$ characterizes how personality judgments emerge from ambiguous intermediate representations into more stable and discrete decisions in later layers.

## E Details of UALD

Let $L _ { m }$ denote the mature layer and $\mathcal { L } _ { p } = \left\{ L _ { 1 } , \ldots \ldots , L _ { K } \right\}$ the set of candidate premature layers. Let $\boldsymbol { z } ^ { ( l ) } \in { \mathbb { R } ^ { | V | } }$ denote vocabulary logits at layer l. We first collapse vocabulary logits into option-level logits over the valid choice set $\vec { \mathcal { C } }$ (size $C = 7 )$

$$
\ell _ { c } ^ { ( l ) } = \operatorname* { m a x } _ { t \in \mathcal { T } ( c ) } z _ { t } ^ { ( l ) } ,
$$

where $\mathcal T ( c )$ is the set of token ids corresponding to option c. This produces $\ell ^ { ( l ) } \in \mathbb { R } ^ { C }$

We then compute option-level probability distributions:

$$
p ^ { ( l ) } = \mathrm { s o f t m a x } ( \ell ^ { ( l ) } ) .
$$

For each premature layer $l \in \mathcal { L } _ { p } ,$ , we compute the Jensen–Shannon divergence with respect to the mature layer:

$$
\mathrm { J S D } ( p ^ { ( L _ { m } ) } , p ^ { ( l ) } ) = \frac { 1 } { 2 } \mathrm { K L } ( p ^ { ( L _ { m } ) } \| m ) + \frac { 1 } { 2 } \mathrm { K L } ( p ^ { ( l ) } \| m ) ,
$$

where

$$
m = \frac { 1 } { 2 } \left( p ^ { ( L _ { m } ) } + p ^ { ( l ) } \right) .
$$

The selected premature layer is

$$
L ^ { * } = \arg \operatorname* { m a x } _ { l \in \mathcal { L } _ { p } } \mathrm { J S D } ( p ^ { ( L _ { m } ) } , p ^ { ( l ) } ) .
$$

Let

$$
\begin{array} { r } { \log p ^ { ( L _ { m } ) } = \log \operatorname { s o f t m a x } ( \ell ^ { ( L _ { m } ) } ) , \ : \ : \ : \log p ^ { ( L ^ { * } ) } = \log \operatorname { s o f t m a x } ( \ell ^ { ( L ^ { * } ) } ) . } \end{array}
$$

The final UALD-adjusted logits are defined as:

$$
p ^ { ( U A L D ) } = \log p ^ { ( L _ { m } ) } + \lambda \log p ^ { ( L ^ { * } ) } ,
$$

where λ is the evolution scale hyperparameter. Optionally, we apply post-softmax normalization:

$$
\tilde { p } ^ { ( U A L D ) } = \mathrm { s o f t m a x } ( p ^ { ( U A L D ) } ) .
$$

This formulation can be interpreted as a log-linear interpolation between mature certainty and intermediate-layer ambiguity. Increasing λ effectively shifts the decoding geometry toward earlier-layer probability structure, providing a controllable probe of decisional evolution across layers.

## F Download Links of Models

<table><tr><td>LLM Names</td><td>Download Links via https://huggingface.co/</td></tr><tr><td>LLaMA3.1-8B-Instruct</td><td> $\mathfrak { m e t a - 1 1 a m a / L 1 a m a - 3 . 1 - 8 B - I n s t r u c t }$ </td></tr><tr><td> $\mathbf { L L a M A 3 . 1 – 8 B – I n s t r u c t – G P T Q – I n t 4 }$ </td><td> $\mathsf { h u g g i n g - q u a n t s / M e t a - L 1 a m a - 3 . 1 - 8 B - I n s t r u c t - G P T Q - I N T 4 }$ </td></tr><tr><td>LLaMA3.1-8B-Instruct-AWQ-Int4</td><td> $\mathsf { h u g g i n g - q u a n t s / M e t a - L 1 a m a - 3 . 1 - 8 B - I n s t r u c t - A W Q - I N T 4 }$ </td></tr><tr><td> $\mathbf { L L a M A 3 . 1 - 8 B - I n s t r u c t - A Q L M - P V - I n t } 2$ </td><td> $\mathtt { I S T A - D A S L a b } / \mathsf { M e t a - L 1 a m a - 3 } . 1 - 8 \mathsf { B - I n s t r u c t - A Q L M - P V - 2 B i t - 1 } \times 1 6 - \mathsf { h } \mathsf { f }$ </td></tr><tr><td> $\mathbf { L L a M A } 3 . 1 – 7 0 \mathbf { B } \mathbf { - } \mathbf { I n s t r u c t }$ </td><td> $\mathfrak { m e t a - 1 1 a m a / L 1 a m a - 3 . 1 - 7 } \theta \mathsf { B - I n s t r u c t }$ </td></tr><tr><td> $\mathbf { L L a M A 3 . 1 – 7 0 B – I n s t r u c t – G P T Q – I n t 4 }$ </td><td> $\hbar \underset { \mathbb { S } } { \operatorname { h u g g i n g - q u a n t s } } / \mathsf { M e t a - L 1 a m a - 3 } . 1 - 7 \mathsf { O B - I n s t r u c t - G P T Q - I N T 4 }$ </td></tr><tr><td> $\mathbf { L L a M A 3 . 1 – 7 0 B – I n s t r u c t – A W Q – I n t 4 }$ </td><td> $\mathsf { h u g g i n g - q u a n t s } / \mathsf { M e t a - L 1 a m a - 3 } . 1 - 7 \theta \mathsf { B - I n s t r u c t - A W Q - I N T 4 }$ </td></tr><tr><td> $\mathbf { L L a M A 3 . 1 – 7 0 B – I n s t r u c t – A Q L M  – P V – I n t 2 }$ </td><td> $\mathtt { I S T A - D A S L a b } / \mathtt { M e t a - L 1 a m a - 3 } . 1 - 7 \mathtt { 0 B - I n s t r u c t - A Q L M - P V - 2 B i t - 1 } . \mathtt { x } 1 6$ </td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td> $\mathtt { m i s t r a l a i / M i s t r a l - 7 B - I n s t r u c t - v \theta . 3 }$ </td></tr><tr><td> $\mathbf { M i s t r a l - 7 B - I n s t r u c t - v 0 . 3 - G P T Q - I n t 4 }$ </td><td> $\mathsf { S H A S W A T S I N G H 3 1 } \theta 1 / \mathsf { M i s t r a l } - \mathsf { T B } - \mathsf { I n s t r u c t } - \mathsf { v } \theta . 3 _ { - } 4 \mathsf { b i t } _ { - } \mathsf { G P T Q }$ </td></tr><tr><td> $\mathbf { M i s t r a l - 7 B - I n s t r u c t - v 0 . 3 - A W Q - I n t 4 }$ </td><td> $\mathsf { S H A S W A T S I N G H 3 1 } \theta 1 / \mathsf { M i s t r a l - 7 B - I n s t r u c t - v } \theta . 3 . 4 \mathsf { b i t \_ A W Q }$ </td></tr><tr><td> $\mathbf { M i s t r a l - 7 B - I n s t r u c t - v 0 . 3 - G P T Q - I n t 2 }$ </td><td> $\mathrm { i r i s h – q u a n t } / \mathrm { m i } s \mathrm { t r a l } a \mathrm { i } \mathrm { - } \mathsf { M i } s \mathrm { t r a l } \mathrm { - } 7 \mathsf { B } \mathrm { - } \mathrm { I n } s \mathrm { t r u c t } \mathrm { - } \mathsf { v } \theta \mathrm { . } 3 \mathrm { - } 2 \mathsf { b i } \mathrm { t }$ </td></tr><tr><td> $\mathbf { M i s t r a l - S m a l l - 2 4 B \mathrm { - } I n s t r u c t { - } } 2 5 0 1$ </td><td> $\mathsf { m i s t r a l a i } / \mathsf { M i s t r a l } - \mathsf { S m a l } 1 - 2 4 \mathsf { B } \mathrm { - } \mathsf { I n s t r u c t } - 2 5 \theta 1$ </td></tr><tr><td> $\mathbf { M i s t r a l - S m a l l - 2 4 B - I n s t r u c t - 2 5 0 1 - G P T Q - I n t 4 }$ </td><td>kaitchup  $/ \mathsf { M i s t r a l - S m a l l - 2 4 B - I n s t r u c t - 2 5 } \theta 1 - \mathsf { A u t o R o u n d - G P T Q - 4 b i t }$ </td></tr><tr><td> $\mathbf { M i s t r a l - S m a l l - 2 4 B - I n s t r u c t - 2 5 0 1 - A W Q - I n t 4 }$ </td><td> $\mathsf { s t e l t e r l a b / u b 1 s t r a l - S m a l 1 - 2 4 B \mathrm { - } I n s t r u c t - 2 5 0 1 \mathrm { - } A W Q }$ </td></tr><tr><td> $\mathbf { M i s t r a l - S m a l l - 2 4 B - I n s t r u c t - 2 5 0 1 - G P T Q - I n t 2 }$ </td><td> $\mathrm { i \ r i s h { - } q u a n t { / } m i s t r a l a i { - } w i s t r a l { - } s m a l l { - } } 2 4 8 { - } \mathrm { I n s t r u c t { - } } 2 5 \theta 1 - \mathrm { 2 b i t } \mathrm { { + } } e l \theta 1 - \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { - } } e l \mathrm { { + } } e l \mathrm { { + } } e l \mathrm { { - } } e l \mathrm { { + } } e l \mathrm { { - } } e l \mathrm { { + } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { + } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { + } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e l \mathrm { { - } } e$ </td></tr><tr><td> $\mathbf { Q w e n 2 . 5 - 1 4 B - I n s t r u c t }$ </td><td> $\mathtt { Q w e n / Q w e n 2 . 5 - 1 4 B - I n s t r u c t }$ </td></tr><tr><td> $\mathbf { Q } \mathbf { w e n } 2 . 5 \mathbf { - 1 } 4 \mathbf { B } \mathbf { - I n s t r u c t { - A } W Q - I n t } 4$ </td><td> $\mathsf { Q w e n / Q w e n 2 . 5 - 1 4 B - I n s t r u c t - A W Q }$ </td></tr><tr><td> $\mathbf { Q } \mathbf { w e n } 2 . 5 \mathbf { - 1 } 4 \mathbf { B } \mathbf { - I n s t r u c t - G P T Q - I n t } 4$ </td><td> $\mathtt { Q w e n / Q w e n 2 . 5 - 1 4 B - I n s t r u c t - G P T Q - I n t 4 }$ </td></tr><tr><td> $\mathbf { Q } \mathbf { w e n } 2 . 5 \mathbf { - 1 } 4 \mathbf { B } \mathbf { - I n s t r u c t - G P T Q - I n t } 2$ </td><td> $\mathrm { S i d d h a r t h 6 3 / Q w e n 2 } \ . 5 - 1 4 \mathrm { B - I n s t r u c t - A u t o R o u n d - G P T Q - 2 b i t a r y 2 . }$ </td></tr><tr><td>Qwen2.5-72B-Instruct</td><td> $\mathsf { Q w e n } / \mathsf { Q w e n } 2 . 5 \mathsf { - } 7 2 \mathsf { B } \mathsf { - } \mathtt { I n s t r u c t }$ </td></tr><tr><td>Qwen2.5-72B-Instruct-GPTQ-Int4</td><td> $\mathtt { Q w e n / Q w e n 2 . 5 - 7 2 B - I n s t r u c t - G P T Q - I n t 4 }$ </td></tr><tr><td>Qwen2.5-72B-Instruct-AWQ-Int4</td><td> $\mathsf { Q w e n / Q w e n 2 . 5 - 7 2 B - I n s t r u c t - A W Q }$ </td></tr><tr><td>Qwen/Qwen2.5-72B-Instruct-GPTQ-Int2</td><td> $\mathsf { k a i t c h u p / Q w e n 2 . 5 - 7 2 B - I n s t r u c t - A u t o R o u n d - G P T Q - 2 b i t }$ </td></tr></table>

Table 6: Download links to all LLMs involved in our experiments.

![](images/2504c03ceb5ec69ec7824dd030d5ec220c1806a307eb63f0cf3f647a0aceeeed.jpg)  
(a) Layer-wise Decisional Entropy

![](images/70e7de6e194f632fdfb2f0e7d0d9b3d514809f5dd5d8bd3b7770ba04f97096d1.jpg)  
(b) Layer-wise Confidence Gap  
Figure 4: Main comparison (LLaMA3.1-70B-Instruct) across layers for four model variants (Original FP16, GPTQ INT4, AWQ INT4, Extreme INT2). Figure (a) reports layer-wise decisional entropy, where lower entropy indicates sharper, more stable decisions. Figure (b) reports the layer-wise confidence gap (top-1 minus top-2 probability), where a larger margin reflects stronger commitment to a single option.

![](images/376d3ce07944ac45a7fbd31cd2b82bb617f614ab45bd02b0d824fa403f6ceb1f.jpg)  
(a) Layer-wise Decisional Entropy

![](images/cd6a70db7caa169a202f4c07ba307b5cad57b6c5e57f5f8b965d87572997b617.jpg)  
(b) Layer-wise Confidence Gap

Figure 5: Main comparison (Mistral-7B-Instruct-v0.3) across layers for four model variants (Original FP16, GPTQ INT4, AWQ INT4, Extreme INT2). Figure (a) reports layer-wise decisional entropy, where lower entropy indicates sharper, more stable decisions. Figure (b) reports the layer-wise confidence gap (top-1 minus top-2 probability), where a larger margin reflects stronger commitment to a single option.

![](images/2672acb325fdd0c84807c778d1b855be1b6e2cf41d6a340ee28a1ca8123f172c.jpg)  
(a) Layer-wise Decisional Entropy

![](images/aa8a7fc42b5c4d378d21e2554544446881db450a8df5dea40d2740db40e5f621.jpg)  
(b) Layer-wise Confidence Gap  
Figure 6: Main comparison (Mistral-Small-24B-Instruct-2501) across layers for four model variants (Original FP16, GPTQ INT4, AWQ INT4, Extreme INT2). Figure (a) reports layerwise decisional entropy, where lower entropy indicates sharper, more stable decisions. Figure (b) reports the layer-wise confidence gap (top-1 minus top-2 probability), where a larger margin reflects stronger commitment to a single option.

![](images/f30a7a30e4953fc1fc9accb9a198723b58cbb714d7a2856701c2f16490f1afeb.jpg)  
(a) Layer-wise Decisional Entropy

![](images/4094a5ab3ca3bbfce324716882d4a9cc2a1f966957cd376a283cd849986f2729.jpg)  
(b) Layer-wise Confidence Gap

Figure 7: Main comparison (Qwen2.5-14B-Instruct) across layers for four model variants (Original FP16, GPTQ INT4, AWQ INT4, Extreme INT2). Figure (a) reports layer-wise decisional entropy, where lower entropy indicates sharper, more stable decisions. Figure (b) reports the layer-wise confidence gap (top-1 minus top-2 probability), where a larger margin reflects stronger commitment to a single option.

![](images/3a2b8339a20589da33e38bbfbe3700e8c1b794218d94872fe8f3f236495b3969.jpg)

![](images/de39b99ff6a5bb522ec617a833f2f7d57b1d8c968842e6dfb6f3338f36f00d9c.jpg)  
(a) Layer-wise Decisional Entropy  
(b) Layer-wise Confidence Gap  
Figure 8: Main comparison (Qwen2.5-72B-Instruct) across layers for four model variants (Original FP16, GPTQ INT4, AWQ INT4, Extreme INT2). Figure (a) reports layer-wise decisional entropy, where lower entropy indicates sharper, more stable decisions. Figure (b) reports the layer-wise confidence gap (top-1 minus $\mathsf { \bar { t o p } } { - } 2$ probability), where a larger margin reflects stronger commitment to a single option.

Table 7: LLaMA3.1-70B-Instruct: Conditional personality consistency under prompt conditioning across quantization levels. Each entry reports the inferred MBTI type when the model is conditioned on a personality-specific prompt $( \mathrm { e . g . , E N F J } \to \mathrm { E N F } ] )$ . For each cell, the left marker indicates consistency with the conditioning prompt, while the right marker indicates agreement with the FP16 baseline prediction.
<table><tr><td>Original FP16</td><td>GPTQ INT4</td><td>AWQ INT4</td><td>Extreme INT2</td></tr><tr><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { } <$ </td><td> $\mathrm { E N F J } \to \mathrm { E N F J } \sqrt  \surd $ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$ </td></tr><tr><td> $\mathrm { E N F P } \to \mathrm { E N F J } \times \checkmark$ </td><td>ENFP-  $ \mathrm { E N F P } \vee \times$ </td><td> ${ \mathrm { E N F P } } \to { \mathrm { E N F P } } \lor \times$ </td><td>ENFP  $ \mathrm { E N F P } \vee \times$ </td></tr><tr><td> $\mathrm { E N T J } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { E N T J } \to \mathrm { E S T J } \times \times$ </td><td> $\mathrm { E N T J } \to \mathrm { E S T J } \times \times$ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \setminus \times$ </td></tr><tr><td> $\mathrm { E N T P } \to \mathrm { E N F J } \times \mathcal { A }$ </td><td> $\mathrm { E N T P } \to \mathrm { E N T P } \ √ \times$ </td><td> $\mathrm { E N T P } \to \mathrm { E N T P } \surd \times$ </td><td> $\mathrm { E N T P } \to \mathrm { E N T P } \ √ \times$ </td></tr><tr><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { E S F J }  \mathrm { E S F J } \prec \times$ </td><td> $\mathrm { E S F J }  \mathrm { E S F J } \downarrow \times$ </td><td> $\mathrm { E S F J }  \mathrm { E S F J } \prec \times$ </td></tr><tr><td> $\mathrm { E S F P }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \times$ </td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \times$ </td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \times$ </td></tr><tr><td> $\mathrm { E S T J }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \times \uparrow$ </td><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \times \downarrow$ </td><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \times$ </td></tr><tr><td> $\mathrm { E S T P }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \mathrel { \mathop { \sim } } \mathrel { \mathop { \times } }$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \mathrel { \mathop { \sim } } \mathrel { \mathop { \times } }$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \mathrel { \mathop { \sim } } \mathrel { \mathop { \times } } \mathrel { \phantom { \mathrm { E S T P } } } \mathrel { \mathop { \sim } } \mathrel { \phantom { \mathrm { E S T P } } } \mathrel { \mathop { \sim } } \mathrel { \phantom { \mathrm { E S T P } } }$ </td></tr><tr><td> $\mathrm { I N F J } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I N F J } \to \mathrm { I N F J } \ v \times$ </td><td> $\mathrm { I N F J } \to \mathrm { I N F J } \ v \times$ </td><td> $\mathrm { I N F J } \to \mathrm { I N F J } \ v \times$ </td></tr><tr><td> $\mathrm { I N F P } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I N F P }  \mathrm { I N F P } \prec \times$ </td><td> $\mathrm { I N F P }  \mathrm { I N F P } \surd \times$ </td><td> $\mathrm { I N F P }  \mathrm { I N F P } \prec \times$ </td></tr><tr><td> $\mathrm { I N T J } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \ v \times$ </td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \lor \times$ </td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \ v \times$ </td></tr><tr><td> $\mathrm { I N T P } \to \mathrm { E N F J } \times \mathcal { A }$ </td><td> $\mathrm { I N T P }  \mathrm { I N T P } \mathrel { \downarrow } \times$ </td><td> $\mathrm { I N T P }  \mathrm { I N T P } \mathrel { \downarrow } \times$ </td><td> $\mathrm { I N T P }  \mathrm { I N T J } \times \times$ </td></tr><tr><td> $\mathrm { I S F J } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I S F J } \to \mathrm { I S F J } \ v \times$ </td><td> $\mathrm { I S F J } \to \mathrm { I S F J } \ v \times$ </td><td> $\mathrm { I S F J } \to \mathrm { I S F J } ^ { \cdot } \checkmark \times$ </td></tr><tr><td> $\mathrm { I S F P }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { I S F P }  \mathrm { I N F P } \times \times$ </td><td> $\mathrm { I S F P  I N F P \times \times }$ </td><td> $\mathrm { I S F P }  \mathrm { I N F P } \times \times$ </td></tr><tr><td> $\mathrm { I S T J } \to \mathrm { E N F J } \times \mathcal { A }$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \lor \times$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \lor \times$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ v \times$ </td></tr><tr><td> $\mathrm { I S T P }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \times$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \times$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \times$ </td></tr></table>

Table 8: Mistral-7B-Instruct-v0.3: Conditional personality consistency under prompt conditioning across quantization levels. Each entry reports the inferred MBTI type when the model is conditioned on a personality-specific prompt $( \mathrm { e . g . , E N F J } \to \mathrm { E N F } ] )$ . For each cell, the left marker indicates consistency with the conditioning prompt, while the right marker indicates agreement with the FP16 baseline prediction.
<table><tr><td>Original FP16</td><td>GPTQ INT4</td><td>AWQ INT4</td><td>Extreme INT2</td><td></td></tr><tr><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$ </td><td>ENFJ → ENFJ √ √</td><td>ENFJ → ENFJ √ √</td><td></td></tr><tr><td>ENFP → ENFP √√</td><td> $\mathrm { E N F P }  \mathrm { E N F P } \checkmark \sqrt { }$ </td><td>ENFP → ENFP √ √</td><td>ENFP → ENFJ × ×</td><td></td></tr><tr><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt  \surd $ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt  \surd $ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt  \surd $ </td><td>ENTJ → ENFJ × ×</td><td></td></tr><tr><td> $\mathrm { E N T P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { E N T P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { E N T P }  \mathrm { E N F P } \times \surd$ </td><td>ENTP → ENFJ × ×</td><td></td></tr><tr><td>ESFJ → ENFJ ×√</td><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \surd$ </td><td>ESFJ → ENFJ × √</td><td></td></tr><tr><td>ESFP → ENFP × √</td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \surd$ </td><td>ESFP → ENFP ×√</td><td>ESFP → ENFJ × ×</td><td></td></tr><tr><td>ESTJ → ENTJ ×√</td><td> $\mathrm { E S T J }  \mathrm { E N T J } \times \surd$ </td><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \times \uparrow$ </td><td>ESTJ → ENFJ × ×</td><td></td></tr><tr><td>ESTP → ESTP √√</td><td> $\mathrm { E S T P }  \mathrm { E S T P } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \downarrow \sqrt { \downarrow }$ </td><td>ESTP → ENFJ × ×</td><td></td></tr><tr><td>INFJ → ENFJ × √</td><td> $\mathrm { I N F J } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I N F J } \to \mathrm { E N F J } \times \checkmark$ </td><td>INFJ → ENFJ ×√</td><td></td></tr><tr><td>INFP → ENFP ×√</td><td>INFP → ENFP ×√</td><td>INFP → ENFP ×√</td><td>INFP → ENFJ × ×</td><td></td></tr><tr><td>INTJ → INTJ √√</td><td>INTJ → INTJ √√</td><td>INTJ → INTJ √√</td><td>INTJ → ENFJ × ×</td><td></td></tr><tr><td>INTP → INTJ × √</td><td>INTP → INTJ × √</td><td>INTP → INTJ ×√</td><td>INTP → ENFJ × ×</td><td></td></tr><tr><td>ISFJ → INFJ ×√</td><td>ISFJ → ENFJ × ×</td><td>ISFJ → ENFJ × ×</td><td>ISFJ → ENFJ × ×</td><td></td></tr><tr><td>ISFP → INFJ ×√</td><td>ISFP → INFJ × √</td><td>ISFP → INFP × ×</td><td>ISFP → ENFJ × ×</td><td></td></tr><tr><td>ISTJ → ISTJ √ √</td><td>ISTJ → INTJ × ×</td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ \sqrt { \ J }$ </td><td>ISTJ → ENFJ × ×</td><td></td></tr><tr><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \checkmark$ </td><td> $\mathrm { I S T P }  \mathrm { I N T J } \times \times$ </td><td> $\mathrm { I S T P }  \mathrm { I N T J } \times \times$ </td><td>ISTP → ENFJ × ×</td><td></td></tr></table>

Table 9: Mistral-Small-24B-Instruct-2501: Conditional personality consistency under prompt conditioning across quantization levels. Each entry reports the inferred MBTI type when the model is conditioned on a personality-specific prompt (e.g., ENFJ → ENFJ). For each cell, the left marker indicates consistency with the conditioning prompt, while the right marker indicates agreement with the FP16 baseline prediction.
<table><tr><td>Original FP16</td><td>GPTQ INT4</td><td>AWQ INT4</td><td>Extreme INT2</td></tr><tr><td>ENFJ → ENFJ √ √</td><td>ENFJ → ENFJ √ √</td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$ </td><td>ENFJ → ENFJ √ √</td></tr><tr><td> $\mathrm { E N F P }  \mathrm { E N F P } \sqrt { \surd }$ </td><td> $\mathrm { E N F P }  \mathrm { E N F P } \checkmark \sqrt { }$ </td><td> $\mathrm { E N F P }  \mathrm { E N F P } \sqrt { \surd }$ </td><td>ENFP → ENFJ × ×</td></tr><tr><td> $\mathrm { E N T J } \to \mathrm { E S T J } \times \mathcal { A }$ </td><td> $\mathrm { E N T J } \to \mathrm { E S T J } \times \mathcal { A }$ </td><td> $\mathrm { E N T J } \to \mathrm { E S T J } \times \mathcal { A }$ </td><td>ENTJ → ESFJ × ×</td></tr><tr><td>ENTP → ENTP √√</td><td> $\mathrm { E N T P }  \mathrm { E N T P } \sqrt { \surd }$ </td><td>ENTP → ENTP √√</td><td>ENTP → ESTJ × ×</td></tr><tr><td>ESFJ → ESFJ √ √</td><td>ESFJ → ESFJ √ √</td><td>ESFJ → ESFJ √ √</td><td>ESFJ → ENFJ × ×</td></tr><tr><td>ESFP → ESFP √ √</td><td>ESFP → ESFP √ √</td><td> $\mathrm { E S F P }  \mathrm { E S F P } \surd \sqrt { }$ </td><td>ESFP → ENFJ × ×</td></tr><tr><td>ESTJ → ESTJ √ √</td><td>ESTJ → ESTJ √ √</td><td>ESTJ → ESTJ √ √</td><td>ESTJ → ENFJ × ×</td></tr><tr><td>ESTP → ESTP √√</td><td>ESTP → ESTP √√</td><td>ESTP → ESTP √√</td><td>ESTP → ESFJ × ×</td></tr><tr><td>INFJ → INFJ √ √</td><td>INFJ → INFJ √ √</td><td>INFJ → INFJ √ √</td><td>INFJ → ENFJ × ×</td></tr><tr><td>INFP → INFP √√</td><td>INFP → INFP √√</td><td>INFP → INFP √√</td><td>INFP → ENFJ × ×</td></tr><tr><td>INTJ → INTJ √√</td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \ V$ </td><td>INTJ → INTJ √√</td><td>INTJ → ENFJ × ×</td></tr><tr><td>INTP → INTJ ×√</td><td> $\mathrm { I N T P } \to \mathrm { I N T J } \times \mathcal { A }$ </td><td>INTP → INTJ × √</td><td>INTP → ESFJ × ×</td></tr><tr><td> $\mathrm { I S F J }  \mathrm { I S F J } ^ { \cdot } \checkmark$ </td><td> $\mathrm { I S F J }  \mathrm { I S F J } ^ { \cdot } \check { \checkmark }$ </td><td> $\mathrm { I S F J }  \mathrm { I S F J } ^ { \cdot } \checkmark$ </td><td>ISFJ → ENFJ × ×</td></tr><tr><td> $\mathrm { I S F P }  \mathrm { I S F P } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { I S F P }  \mathrm { I S F P } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { I S F P }  \mathrm { I N F P } \times \times$ </td><td>ISFP → ENFJ × ×</td></tr><tr><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ \sqrt { \ J }$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ V \ V$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ \sqrt { \ J }$ </td><td>ISTJ → ENFJ × ×</td></tr><tr><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \surd$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \surd$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \surd$ </td><td> $\mathrm { I S T P }  \mathrm { E N F J } \times \times$ </td></tr></table>

Table 10: Qwen2.5-14B-Instruct: Conditional personality consistency under prompt conditioning across quantization levels. Each entry reports the inferred MBTI type when the model is conditioned on a personality-specific prompt $( \mathrm { e . g . , E N F J } \to \mathrm { E N F J } )$ . For each cell, the left marker indicates consistency with the conditioning prompt, while the right marker indicates agreement with the FP16 baseline prediction.
<table><tr><td>Original FP16</td><td>GPTQ INT4</td><td>AWQ INT4</td><td>Extreme INT2</td></tr><tr><td> $\mathrm { E N F J }  \mathrm { E N F J } \sqrt  \surd $ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { } <$ </td></tr><tr><td> $\mathrm { E N F P }  \mathrm { E N F P } \checkmark \sqrt { }$ </td><td> $\mathrm { E N F P }  \mathrm { E N F P } \sqrt { \surd }$ </td><td> $\mathrm { E N F P }  \mathrm { E N F P } \sqrt { \surd }$ </td><td> $\mathrm { E N F P } \to \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt  \surd $ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt  \surd $ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \sqrt  \surd $ </td><td> $\mathrm { E N T J } \to \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { E N T P }  \mathrm { E N T P } \checkmark { }$ </td><td> $\mathrm { E N T P }  \mathrm { E N T P } \checkmark { }$ </td><td> $\mathrm { E N T P }  \mathrm { E N T P } \sqrt { \surd }$ </td><td> $\mathrm { E N T P } \to \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { E S F J }  \mathrm { E S F J } \surd \sqrt { \surd \sqrt { \nu } }$ </td><td> $\mathrm { E S F J }  \mathrm { E S F J } \surd$ </td><td> $\mathrm { E S F J }  \mathrm { E S F J } \surd \sqrt { \surd \sqrt { \nu } }$ </td><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \surd$ </td><td> $\mathrm { E S F P }  \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T J } \to \mathrm { E S T J } \ V \ V$ </td><td> $\mathrm { E S T J }  \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { E S T P }  \mathrm { E S T P } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \downarrow \sqrt { \downarrow }$ </td><td> $\mathrm { E S T P  E N F J } \times \times$ </td></tr><tr><td> $\mathrm { I N F J } \to \mathrm { I N F J } \sqrt { \checkmark }$ </td><td>INFJ → INFJ √√</td><td>INFJ → ENFJ × ×</td><td>INFJ → ENFJ × ×</td></tr><tr><td> $\mathrm { I N F P }  \mathrm { I N F P } \prec$ </td><td> $\mathrm { I N F P }  \mathrm { I N F P } \prec$ </td><td> $\mathrm { I N F P \to E N F P \times \times }$ </td><td> $\mathrm { I N F P  E N F J } \times \times$ </td></tr><tr><td> $\mathrm { I N T J } \to \mathrm { I N T J } \Vdash$ </td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \Vdash$ </td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \Vdash$ </td><td> $\mathrm { I N T J } \to \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { I N T P } \to \mathrm { I N T J } \times \mathcal { A }$ </td><td> $\mathrm { I N T P } \to \mathrm { I N T J } \times \mathcal { A }$ </td><td> $\mathrm { I N T P } \to \mathrm { I N T J } \times \mathcal { A }$ </td><td> $\mathrm { I N T P } \to \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { I S F J }  \mathrm { I S F J } \sqrt  \surd $ </td><td> $\mathrm { I S F J }  \mathrm { I S F J } \sqrt  \surd $ </td><td> $\mathrm { I S F J }  \mathrm { I S F J } ^ { \cdot } \checkmark$ </td><td> $\mathrm { I S F J }  \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { I S F P }  \mathrm { I S F P } \downarrow \downarrow$ </td><td> $\mathrm { I S F P }  \mathrm { I S F J } \times \times$ </td><td> $\mathrm { I S F P }  \mathrm { I N F P } \times \times$ </td><td> $\mathrm { I S F P }  \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ \sqrt { \ J }$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ \sqrt { \ J }$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ \sqrt { \ J }$ </td><td> $\mathrm { I S T J } \to \mathrm { E N F J } \times \times$ </td></tr><tr><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \surd$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \checkmark$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \checkmark$ </td><td> $\mathrm { I S T P }  \mathrm { E N F J } \times \times$ </td></tr></table>

Table 11: Qwen2.5-72B-Instruct: Conditional personality consistency under prompt conditioning across quantization levels. Each entry reports the inferred MBTI type when the model is conditioned on a personality-specific prompt (e.g., ENFJ → ENFJ). For each cell, the left marker indicates consistency with the conditioning prompt, while the right marker indicates agreement with the FP16 baseline prediction.
<table><tr><td>Original FP16</td><td>GPTQ INT4</td><td>AWQ INT4</td><td>Extreme INT2</td></tr><tr><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { } <$ </td><td> $\mathrm { E N F J } \to \mathrm { E N F J } \sqrt  \surd $ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \sqrt  \surd $ </td><td> $\mathrm { E N F J }  \mathrm { E N F J } \checkmark { }$ </td></tr><tr><td> $\mathrm { E N F P } \to \mathrm { E N F J } \times \checkmark$ </td><td> ${ \mathrm { E N F P } } \to { \mathrm { E N F P } } \setminus \times$ </td><td> ${ \mathrm { E N F P } } \to { \mathrm { E N F P } } \lor \times$ </td><td> $\mathrm { E N F P } \to \mathrm { E N F J } \times \checkmark$ </td></tr><tr><td> $\mathrm { E N T J } \to \mathrm { E N F J } \times \mathcal { A }$ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \setminus \times$ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \setminus \times$ </td><td> $\mathrm { E N T J } \to \mathrm { E N T J } \setminus \times$ </td></tr><tr><td> $\mathrm { E N T P } \to \mathrm { E N F J } \times \checkmark$ </td><td> ${ \mathrm { E N T P } } \to { \mathrm { E N F P } } \times \times$ </td><td> $\mathrm { E N T P } \to \mathrm { E N T P } \surd \times$ </td><td> $\mathrm { E N T P } \to \mathrm { E N F J } \times \mathcal { A }$ </td></tr><tr><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S F J }  \mathrm { E N F J } \times \surd$ </td></tr><tr><td> $\mathrm { E S F P }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \times$ </td><td> $\mathrm { E S F P }  \mathrm { E N F P } \times \times$ </td><td>ESFP → ENFJ × √</td></tr><tr><td> $\mathrm { E S T J }  \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \times \uparrow$ </td><td> $\mathrm { E S T J }  \mathrm { E S T J } \downarrow \times \uparrow$ </td><td> $\mathrm { E S T J }  \mathrm { E N T J } \times \times$ </td></tr><tr><td> $\mathrm { E S T P }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \mathrel { \mathop { \sim } } \mathrel { \mathop { \times } }$ </td><td> $\mathrm { E S T P }  \mathrm { E S T P } \mathrel { \mathop { \sim } } \mathrel { \mathop { \times } }$ </td><td> $\mathrm { E S T P }  \mathrm { E N F J } \times \surd$ </td></tr><tr><td> $\mathrm { I N F J } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I N F J } \to \mathrm { I N F J } \ v \times$ </td><td> $\mathrm { I N F J } \to \mathrm { I N F J } \ v \times$ </td><td> $\mathrm { I N F J } \to \mathrm { E N F } \mathbf { \check { J } } \times \mathbf { \check { \sqrt { \mathbf { \Lambda } } } }$ </td></tr><tr><td> $\mathrm { I N F P } \to \mathrm { E N F J } \times \checkmark$ </td><td> $\mathrm { I N F P  I N F P \downarrow \times }$ </td><td> $\mathrm { I N F P }  \mathrm { I N F P } \prec \times$ </td><td> $\mathrm { I N F P } \to \mathrm { E N F J } \times \mathcal { A }$ </td></tr><tr><td>INTJ → ENFJ × √</td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \ v \times$ </td><td> $\mathrm { I N T J } \to \mathrm { I N T J } \lor \times$ </td><td> $\mathrm { I N T J } \to \mathrm { E N T J } \times \times$ </td></tr><tr><td> $\mathrm { I N T P } \to \mathrm { E N F J } \times \mathcal { A }$ </td><td> $\mathrm { I N T P }  \mathrm { I N T J } \times \times$ </td><td> $\mathrm { I N T P }  \mathrm { I N T J } \times \times$ </td><td> $\mathrm { I N T P }  \mathrm { E N T J } \times \times$ </td></tr><tr><td> $\mathrm { I S F J }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { I S F J } \to \mathrm { I N F J } \times \times$ </td><td> $\mathrm { I S F J }  \mathrm { I N F J } \times \times$ </td><td> $\mathrm { I S F J }  \mathrm { E N F J } \times \surd$ </td></tr><tr><td> $\mathrm { I S F P }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { I S F P  I N F P \times \times }$ </td><td> $\mathrm { I S F P  I N F P \times \times }$ </td><td> $\mathrm { I S F P }  \mathrm { E N F J } \times \surd$ </td></tr><tr><td> $\mathrm { I S T J } \to \mathrm { E N F J } \times \mathcal { A }$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \ v \times$ </td><td> $\mathrm { I S T J } \to \mathrm { I S T J } \lor \times$ </td><td> $\mathrm { I S T J } \to \mathrm { E N F J } \times \mathcal { A }$ </td></tr><tr><td> $\mathrm { I S T P }  \mathrm { E N F J } \times \surd$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \times$ </td><td> $\mathrm { I S T P }  \mathrm { I S T J } \times \times$ </td><td> $\mathrm { I S T P }  \mathrm { E N T J } \times \times$ </td></tr></table>

Table 12: Switch statistics for LLaMA3.1-8B-Instruct under personality-conditioned prompting across quantization levels. Prompt-switch counts measure how often the inferred MBTI type differs from the conditioning prompt. FP16-switch counts measure how often each quantized model differs from the Original FP16 prediction under the same conditioning prompt. Means and standard errors are computed over the 16 MBTI-conditioned prompts.
<table><tr><td>Precision</td><td>Prompt-switch count / 16</td><td>Prompt-switch  $\mathbf { m e a n } \pm \mathbf { S E }$ </td><td>FP16-switch count / 16</td><td>FP16-switch  $\mathbf { m e a n } \pm \mathbf { S E }$ </td></tr><tr><td>Original FP16</td><td>9 / 16</td><td> $0 . 5 6 3 \pm 0 . 1 2 4$ </td><td> $0 ~ / ~ 1 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>GPTQ INT4</td><td> $7 / 1 6$ </td><td> $0 . 4 3 8 \pm 0 . 1 2 4$ </td><td> $3 / 1 6$ </td><td> $0 . 1 8 8 \pm 0 . 0 9 8$ </td></tr><tr><td>AWQ INT4</td><td> $8 ~ / ~ 1 6$ </td><td> $0 . 5 0 0 \pm 0 . 1 2 5$ </td><td> $3 / 1 6$ </td><td> $0 . 1 8 8 \pm 0 . 0 9 8$ </td></tr><tr><td>Extreme INT2</td><td> $1 1 ~ / ~ 1 6$ </td><td> $0 . 6 8 8 \pm 0 . 1 1 6$ </td><td> $5 / 1 6$ </td><td> $0 . 3 1 3 \pm 0 . 1 1 6$ </td></tr></table>

Table 13: Switch statistics for LLaMA3.1-70B-Instruct under personality-conditioned prompting across quantization levels. Prompt-switch counts measure how often the inferred MBTI type differs from the conditioning prompt. FP16-switch counts measure how often each quantized model differs from the Original FP16 prediction under the same conditioning prompt. Means and standard errors are computed over the 16 MBTI-conditioned prompts.
<table><tr><td>Precision</td><td>Prompt-switch count / 16</td><td>Prompt-switch  $\mathbf { m e a n } \pm \mathbf { S E }$ </td><td>FP16-switch count / 16</td><td> $\mathbf { F P 1 6 - s w i t c h }$   $\mathbf { m e a n } \pm \mathbf { S E }$ </td></tr><tr><td>Original FP16</td><td>15 / 16</td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td><td> $0 ~ / ~ 1 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>GPTQ INT4</td><td>4/16</td><td> $0 . 2 5 0 \pm 0 . 1 0 8$ </td><td> $1 5 / 1 6$ </td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td></tr><tr><td>AWQ INT4</td><td> $4 / 1 6$ </td><td> $0 . 2 5 0 \pm 0 . 1 0 8$ </td><td> $1 5 / 1 6$ </td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td></tr><tr><td>Extreme INT2</td><td> $4 / 1 6$ </td><td> $0 . 2 5 0 \pm 0 . 1 0 8$ </td><td> $1 5 / 1 6$ </td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td></tr></table>

Table 14: Switch statistics for Mistral-7B-Instruct-v0.3 under personality-conditioned prompting across quantization levels. Prompt-switch counts measure how often the inferred MBTI type differs from the conditioning prompt. FP16-switch counts measure how often each quantized model differs from the Original FP16 prediction under the same conditioning prompt. Means and standard errors are computed over the 16 MBTI-conditioned prompts.
<table><tr><td>Precision</td><td>Prompt-switch count / 16</td><td>Prompt-switch mean ± SE</td><td>FP16-switch count / 16</td><td>FP16-switch  $\mathbf { m e a n } \pm \mathbf { S E }$ </td></tr><tr><td>Original FP16</td><td>10 / 16</td><td> $0 . 6 2 5 \pm 0 . 1 2 1$ </td><td> $0 ~ / ~ 1 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>GPTQ INT4</td><td>11 / 16</td><td> $0 . 6 8 8 \pm 0 . 1 1 6$ </td><td> $3 / 1 6$ </td><td> $0 . 1 8 8 \pm 0 . 0 9 8$ </td></tr><tr><td>AWQ INT4</td><td> $9 / 1 6$ </td><td> $0 . 5 6 3 \pm 0 . 1 2 4$ </td><td> $4 / 1 6$ </td><td> $0 . 2 5 0 \pm 0 . 1 0 8$ </td></tr><tr><td>Extreme INT2</td><td> $1 5 / 1 6$ </td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td><td> $1 3 / 1 6$ </td><td> $0 . 8 1 3 \pm 0 . 0 9 8$ </td></tr></table>

Table 15: Switch statistics for Mistral-Small-24B-Instruct-2501 under personality-conditioned prompting across quantization levels. Prompt-switch counts measure how often the inferred MBTI type differs from the conditioning prompt. FP16-switch counts measure how often each quantized model differs from the Original FP16 prediction under the same conditioning prompt. Means and standard errors are computed over the 16 MBTI-conditioned prompts.
<table><tr><td>Precision</td><td>Prompt-switch count / 16</td><td>Prompt-switch  $\mathbf { m e a n } \pm \mathbf { S E }$ </td><td>FP16-switch count / 16</td><td>FP16-switch  $\mathbf { m e a n } \pm \mathbf { S E }$ </td></tr><tr><td>Original FP16</td><td>3 / 16</td><td> $0 . 1 8 8 \pm 0 . 0 9 8$ </td><td>0/16</td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>GPTQ INT4</td><td>3 / 16</td><td> $0 . 1 8 8 \pm 0 . 0 9 8$ </td><td> $0 ~ / ~ 1 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>AWQ INT4</td><td> $4 / 1 6$ </td><td> $0 . 2 5 0 \pm 0 . 1 0 8$ </td><td> $1 / 1 6$ </td><td> $0 . 0 6 3 \pm 0 . 0 6 1$ </td></tr><tr><td>Extreme INT2</td><td> $1 5 / 1 6$ </td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td><td> $1 5 / 1 6$ </td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td></tr></table>

Table 16: Switch statistics for Qwen2.5-14B-Instruct under personality-conditioned prompting across quantization levels. Prompt-switch counts measure how often the inferred MBTI type differs from the conditioning prompt. FP16-switch counts measure how often each quantized model differs from the Original FP16 prediction under the same conditioning prompt. Means and standard errors are computed over the 16 MBTI-conditioned prompts.
<table><tr><td>Precision</td><td>Prompt-switch count / 16</td><td>Prompt-switch  $\mathbf { m e a n } \pm \mathbf { S E }$ </td><td>FP16-switch count / 16</td><td>FP16-switch  $\mathbf { m e a n } \pm \mathbf { S E }$ </td></tr><tr><td>Original FP16</td><td>3/ 16</td><td> $0 . 1 8 8 \pm 0 . 0 9 8$ </td><td>0/ 16</td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>GPTQ INT4</td><td>4/16</td><td> $0 . 2 5 0 \pm 0 . 1 0 8$ </td><td>1 /16</td><td> $0 . 0 6 3 \pm 0 . 0 6 1$ </td></tr><tr><td>AWQ INT4</td><td>6 / 16</td><td> $0 . 3 7 5 \pm 0 . 1 2 1$ </td><td> $3 / 1 6$ </td><td> $0 . 1 8 8 \pm 0 . 0 9 8$ </td></tr><tr><td>Extreme INT2</td><td>15 / 16</td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td><td>15 / 16</td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td></tr></table>

Table 17: Switch statistics for Qwen2.5-72B-Instruct under personality-conditioned prompting across quantization levels. Prompt-switch counts measure how often the inferred MBTI type differs from the conditioning prompt. FP16-switch counts measure how often each quantized model differs from the Original FP16 prediction under the same conditioning prompt. Means and standard errors are computed over the 16 MBTI-conditioned prompts.
<table><tr><td>Precision</td><td>Prompt-switch count / 16</td><td>Prompt-switch mean ± SE</td><td>FP16-switch count / 16</td><td>FP16-switch  $\mathbf { m e a n } \pm \mathbf { S E }$ </td></tr><tr><td>Original FP16</td><td> $1 5 / 1 6$ </td><td> $0 . 9 3 8 \pm 0 . 0 6 1$ </td><td> $0 ~ / ~ 1 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>GPTQ INT4</td><td> $7 / 1 6$ </td><td> $0 . 4 3 8 \pm 0 . 1 2 4$ </td><td> $1 4 / 1 6$ </td><td> $0 . 8 7 5 \pm 0 . 0 8 3$ </td></tr><tr><td>AWQ INT4</td><td> $6 ~ / ~ 1 6$ </td><td> $0 . 3 7 5 \pm 0 . 1 2 1$ </td><td> $1 4 / 1 6$ </td><td> $0 . 8 7 5 \pm 0 . 0 8 3$ </td></tr><tr><td>Extreme INT2</td><td> $1 4 / 1 6$ </td><td> $0 . 8 7 5 \pm 0 . 0 8 3$ </td><td> $5 / 1 6$ </td><td> $0 . 3 1 3 \pm 0 . 1 1 6$ </td></tr></table>

![](images/19286109cb4b5b4fd59c86660f5a7c75f46423a124eba7a5e9a45a207621d69a.jpg)  
(a) FP16

![](images/8ce25a745363b5f517014238b281dcba1535a4b3916d4205bcf907d582c06e4e.jpg)

![](images/e89fad852b63250e30f1bb2f679e4fce113f0d4b882172cdfb087ab2b734ef6f.jpg)

![](images/0b7dc87451ad48cd33c032cca1baeccc37ac8a25948bdf209ae307fa421db482.jpg)  
(c) AWQ INT4  
(d) AQLM 2-bit

(b) GPTQ INT4  
![](images/501c01f51f839e3670daeafb16bd870615ef7e5c33187c56449b8cc5cf2ed566.jpg)  
(e) FP16

![](images/0a8edaaa9f903c6729a366873dfbe1ca38027f7af418fa810f957e86d80fa6e4.jpg)  
(f) GPTQ INT4

![](images/de804da8a56813b8fef750d9c130a2c17d8baf36fc7839df23d8df1d83fb7573.jpg)  
(g) AWQ INT4

![](images/f507fc5be0b54477734af073a7f4b2ce6b0ecba9d4b7cb817dffb4c39a406f73.jpg)  
(h) AQLM 2-bit  
Figure 9: LLaMA3.1-70B-Instruct: layer-wise decisional entropy (top row) and top-1 vs. top-2 probability gap (bottom row) across quantization variants. Early layers remain highly uncertain, while deeper layers progressively sharpen decisions.

![](images/df0069d9af9c80cc82e22c8c620dd10bb48fcea5189b53f10f96c0d2a267d6af.jpg)  
(a) FP16

![](images/52489fc13b01dc83f2034d56a86ad08eef2ba50e47eca151d6280779ed880afa.jpg)  
(b) GPTQ INT4

![](images/09d897b15f5e3ac7d702963d11571a204c8dc9e9e6a226c7f00252525ab97198.jpg)

![](images/1c78c7e9c25bb0f4b0f954f406f70e4c589ebdbc6667556e146633a6e2773ebd.jpg)

![](images/60b62e205d962580764e3d6ce2648230ac4b11f76b2cf63c45c22a9fd1b2afd3.jpg)

(e) FP16  
![](images/7a64f8d910a01b104a49815835aa97e42bb39af8ebf5045631e35fa231fd11a2.jpg)  
(f) GPTQ INT4

(c) AWQ INT4  
![](images/cf2db5f971f7d98a6b246c1cf1e876df88af6ce2109a2b30e612d37578b0b87b.jpg)  
(g) AWQ INT4

(d) AQLM 2-bit  
![](images/4764903253479b2794140d2a847cdbdff8f7c9a36e0ee4c97127af6f0c83e6ac.jpg)  
(h) AQLM 2-bit  
Figure 10: Mistral-7B-Instruct-v0.3: layer-wise decisional entropy (top row) and top-1 vs. top-2 probability gap (bottom row) across quantization variants. Early layers show substantial uncertainty, whereas later layers progressively become more decisive.

![](images/7cb135a08d5269e30d5c8f378f0a6a7a1c8b4769beeccd007025982cacb2d4c4.jpg)  
(a) FP16

![](images/63e2276b6db8b76980193d1670de23cfa9304ed1372946b707c44a584ba679b1.jpg)  
(b) GPTQ INT4

![](images/b3923bfab528ab4c94db511781861d5f550e598a71390f2c1793a965b1c57641.jpg)  
(c) AWQ INT4

![](images/64dd6c38ea86060115ef8bcd8061661a95d86a57f226515cd8328589f25e8a47.jpg)  
(d) AQLM 2-bit

![](images/20f1cae81d72167af819615e2e6a06e6db8df2d2c54a88937efcdfb247f138bf.jpg)  
(e) FP16

![](images/65b3c687cfe0a506974e5bb8d32bad248c3ab82afecc72b400388037a604d7cc.jpg)  
(f) GPTQ INT4

![](images/0d7c5eb66e1ebcbfd7c0ebd73123edef86dfe6b091a81cbed87bec603d77b4df.jpg)  
(g) AWQ INT4

![](images/3db29d96c8e0e68948426571fb3d414439359cde6bb598c4c5576ad4a8b28ba5.jpg)  
(h) AQLM 2-bit

Figure 11: Mistral-Small-24B-Instruct-2501: layer-wise decisional entropy (top row) and top-1 vs. top-2 probability gap (bottom row) across quantization variants. Deeper layers generally exhibit lower entropy and larger confidence margins, while stronger quantization can disrupt this late-layer consolidation and reduce decisional sharpness.

![](images/6bb5ad20359abd71f76c64808ecff36a3578b0d76de0f544e0a2025eac6896ad.jpg)  
(a) FP16

![](images/affe2b33fef584e2a747ece7b7f65518d8504c78c2c614ad9778cb746dca4488.jpg)  
(b) GPTQ INT4

![](images/9f7de8ff13cf550047a3d8352da16ddf2786eab937f6d6976dab998feef1e16a.jpg)  
(c) AWQ INT4

![](images/d97b9c4c8e07e965e0281b24e4ce12db1354f3ac4ce33763212e32c689b22484.jpg)  
(d) AQLM 2-bit

![](images/156c1d295c932eb8067bf59cefab08ac7833474bc696d99f22814d8599a929a9.jpg)  
(e) FP16

![](images/08ec932dfb464ec04583bf45311b770b49baf01dabe79d5d6f13ffa98bdf0b03.jpg)  
(f) GPTQ INT4

![](images/e25cb3c4a83f513adcfeb8ed5c1179a4ee42cfb3655a53fbf9592099ac6ff084.jpg)  
(g) AWQ INT4

![](images/22e7ff757565c347ffdc8b2b72d6e2ae8d31105c4acd49e5c158ceb94ca3b743.jpg)  
(h) AQLM 2-bit

Figure 12: Qwen2.5-14B-Instruct: layer-wise decisional entropy (top row) and top-1 vs. top-2 probability gap (bottom row) across quantization variants. Early layers remain uncertain, whereas deeper layers increasingly sharpen response preferences. Extreme quantization disrupts this progression, leading to higher entropy and weaker confidence margins near the output.

![](images/cd606e3c2d8d85a5f5d8153efdfa126a53a5f3cf7059be1d2494f98ec5c7db6a.jpg)  
(a) FP16

![](images/97f4f2f83e3e9192a9546a35109ad78aa4b6db0803fd40c1d1ca6c76fb9a8326.jpg)  
(b) GPTQ INT4

![](images/ff89923fa07ad6ba0bb6ff760e4d53dd7307aa33f7f2972264eb08a78606d704.jpg)  
(c) AWQ INT4

![](images/62b91687dd6573b6e4cb18d4a57dce6493514a0a79056f3429b04ac5922847f2.jpg)  
(d) AQLM 2-bit

![](images/9a78ed4bfbe95d8f3c0fada43a10cc696a61a1c6d30019656034ed6ad8192a85.jpg)  
(e) FP16

![](images/9aac6345a0e510888a6fe4430681cb489a98bf3a486a291ef13d63c5c6f52af1.jpg)  
(f) GPTQ INT4

![](images/d92d9be7f0de2d3028c0e8a338ec6aab0b986bcac8c634b21fbbf7638e5b166a.jpg)  
(g) AWQ INT4

![](images/df38148ed1f357d9c338345fadd3236c2fa91364c08eb6f61dc97d9c4492f1ee.jpg)  
(h) AQLM 2-bit  
Figure 13: Qwen2.5-72B-Instruct: layer-wise decisional entropy (top row) and top-1 vs. top-2 probability gap (bottom row) across quantization variants. Deeper layers generally show reduced uncertainty and stronger commitment, while stronger quantization can attenuate decisional sharpening and destabilize confidence margins near the output.

LLaMA-3.1-70B-Instruct: Personality Drift Across Evolution Scale (Distance from Scale=0 Baseline)  
![](images/b20f7f9fecccb54a21ef62910092412d8a84eb3c3bac90efabbaf66f886d9adb.jpg)  
Figure 14: Personality drift under increasing evolution scale. Each row corresponds to a model–prompt setting, and each column corresponds to an evolution scale. Cell color indicates Hamming distance between the predicted MBTI type at a given scale and the baseline prediction at scale 0. Annotated labels show the inferred MBTI type.

Mistral-7B-Instruct-v0.3: Personality Drift Across Evolution Scale (Distance from Scale=0 Baseline)
<table><tr><td rowspan="2">Unconditional FP16</td><td colspan="5"></td><td colspan="5"></td><td rowspan="2"></td><td rowspan="2">4.0 3.5</td></tr><tr><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>INFJ</td><td>INFJ</td><td>ENFJ</td><td></td><td>ENFJ</td><td>ENFJ</td></tr><tr><td>ENFJ-conditioned FP16</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>3.0</td><td></td></tr><tr><td>Unconditional GPTQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>(Hammng) 2.5</td></tr><tr><td>Mottl stting ENFJ-conditioned GPTQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>2.0 ditnce 1.5</td></tr><tr><td>Unconditional AWQ</td><td>ENTJ</td><td>ENTJ</td><td>ENTJ</td><td>ENTJ</td><td>ENTJ</td><td></td><td>ENTJ</td><td>ENTJ</td><td>ENTJ</td><td>ENTJ</td><td></td><td>Drit</td></tr><tr><td>ENFJ-conditioned AWQ Unconditional AQLM</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>1.0</td></tr><tr><td>ENFJ-conditioned AQLM</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>ENFJ</td><td>ENFJ</td><td>ESFJ</td><td>ISFJ</td><td></td><td>0.5</td></tr><tr><td></td><td>ENFJ 0</td><td>ENFJ 5</td><td>ENFJ 10</td><td>ENFJ 15</td><td></td><td>ENFJ 20</td><td>ENFJ 25</td><td>ENFJ 30</td><td>ENFJ 35</td><td>ENFJ 40</td><td></td><td>0.0</td></tr></table>

Figure 15: Personality drift under increasing evolution scale. Each row corresponds to a model–prompt setting, and each column corresponds to an evolution scale. Cell color indicates Hamming distance between the predicted MBTI type at a given scale and the baseline prediction at scale 0. Annotated labels show the inferred MBTI type.

Mistral-Small-24B-Instruct-2501: Personality Drift Across Evolution Scale (Distance from Scale=0 Baseline)
<table><tr><td rowspan="2">Unconditional FP16</td><td colspan="9"></td><td rowspan="2"></td><td rowspan="2">4.0 3.5</td></tr><tr><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td></tr><tr><td>ENFJ-conditioned FP16</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>ENFJ</td><td>ENFJ</td><td>3.0</td></tr><tr><td>Unconditional GPTQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ESFJ</td><td>ESFJ</td><td>ESFJ</td><td></td><td>ESFJ</td><td>ESFP</td><td>(Hammng) 2.5</td></tr><tr><td>Motil stting ENFJ-conditioned GPTQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFP</td><td>ENFP</td><td>ENFP</td><td>INFP</td><td></td><td>ENFP</td><td>INFP</td><td>2.0</td></tr><tr><td>Unconditional AWQ</td><td>ENFJ</td><td>ESFJ</td><td>ESFJ</td><td>ESTJ</td><td>ESTJ</td><td>ESTP</td><td>ESTJ</td><td></td><td>ESTJ</td><td>ESTP</td><td>distance 1.5</td></tr><tr><td>ENFJ-conditioned AWQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENTJ</td><td>ESTJ</td><td></td><td>INFJ</td><td>INFJ</td><td>Drit 1.0</td></tr><tr><td>Unconditional AQLM</td><td>ENFJ</td><td>ENTP</td><td>ENTP</td><td>ENFJ</td><td>ENTJ</td><td>ENTJ</td><td>ENTJ</td><td></td><td>ENTJ</td><td>ENTJ</td><td>0.5</td></tr><tr><td>ENFJ-conditioned AQLM</td><td>ENFJ</td><td>ENFJ</td><td>ENFP</td><td>ENFP</td><td>ENFP</td><td>ENFP</td><td>ENFP</td><td></td><td>ENFP</td><td>ENFP</td><td>0.0</td></tr><tr><td></td><td>0</td><td>5</td><td>10</td><td>15</td><td>20 Evolution</td><td>25 scale</td><td></td><td>30</td><td>35</td><td>40</td><td></td></tr></table>

Figure 16: Personality drift under increasing evolution scale. Each row corresponds to a model–prompt setting, and each column corresponds to an evolution scale. Cell color indicates Hamming distance between the predicted MBTI type at a given scale and the baseline prediction at scale 0. Annotated labels show the inferred MBTI type.

Qwen2.5-14B-Instruct: Personality Drift Across Evolution Scale (Distance from Scale=0 Baseline)
<table><tr><td rowspan="2">Unconditional FP16</td><td rowspan="2">ENTJ ENFJ</td><td colspan="8">ENFJ ENFJ ENFJ</td><td rowspan="2"></td><td rowspan="2">4.0 3.5 3.0</td></tr><tr><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ ESFJ</td><td>ENFJ ENFJ</td><td>ENFJ ENFJ</td><td>ENFJ ENTJ</td></tr><tr><td>ENFJ-conditioned FP16 Unconditional GPTQ</td><td>ENFJ ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENTJ</td><td>ESFJ</td><td>ESFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>(Haming)</td></tr><tr><td>ENFJ-conditioned GPTQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ESFJ</td><td>ESFJ</td><td></td><td>ISFJ</td><td>2.5</td></tr><tr><td>Mott setting Unconditional AWQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>ENFJ</td><td>2.0 distance</td></tr><tr><td>ENFJ-conditioned AWQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td></td><td>ENFJ</td><td>1.5 Drit</td></tr><tr><td>Unconditional AQLM</td><td>ENFJ</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td></td><td>ISTP</td><td>1.0</td></tr><tr><td>ENFJ-conditioned AQLM</td><td>ENFJ</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td></td><td>ISTP</td><td>0.5</td></tr><tr><td></td><td>0</td><td>5</td><td>10</td><td>15</td><td>20 Evolution scale</td><td>25</td><td>30</td><td>35</td><td></td><td>40</td><td>0.0</td></tr></table>

Figure 17: Personality drift under increasing evolution scale.Each row corresponds to a model–prompt setting, and each column corresponds to an evolution scale. Cell color indicates Hamming distance between the predicted MBTI type at a given scale and the baseline prediction at scale 0. Annotated labels show the inferred MBTI type.

Qwen2.5-72B-Instruct: Personality Drift Across Evolution Scale (Distance from Scale=0 Baseline)
<table><tr><td rowspan="2">Unconditional FP16</td><td colspan="9"></td><td rowspan="2">4.0</td></tr><tr><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td></tr><tr><td>ENFJ-conditioned FP16</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENTJ</td><td>3.5 3.0</td></tr><tr><td>Unconditional GPTQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENTJ</td><td>ESFJ</td><td>ESFJ</td><td>ESFJ</td><td>ESFJ</td><td>ESFJ</td><td>(Hamng) 2.5</td></tr><tr><td>Motl setting ENFJ-conditioned GPTQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ESFJ</td><td>ESFJ</td><td>ISFJ</td><td>2.0</td></tr><tr><td>Unconditional AWQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>distance</td></tr><tr><td>ENFJ-conditioned AWQ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>ENFJ</td><td>1.5 Drit 1.0</td></tr><tr><td>Unconditional AQLM</td><td>ENFJ</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>0.5</td></tr><tr><td>ENFJ-conditioned AQLM</td><td>ENFJ</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>ISTP</td><td>0.0</td></tr><tr><td></td><td>0</td><td>5</td><td>10</td><td>15</td><td>20 Evolution scale</td><td>25</td><td>30</td><td>35</td><td>40</td><td></td></tr></table>

Figure 18: Personality drift under increasing evolution scale. Each row corresponds to a model–prompt setting, and each column corresponds to an evolution scale. Cell color indicates Hamming distance between the predicted MBTI type at a given scale and the baseline prediction at scale 0. Annotated labels show the inferred MBTI type.