(b) Reference-Guided Task Example

# REFCRAFT: BENCHMARKING VISUAL-REFERENCE-CONDITIONED AGENTIC GENERATION THROUGH WEB DEVELOPMENT

Yuchi Wang<sup>12∗</sup>, Rongyao Fang<sup>2</sup>, Weihao Yu<sup>23</sup>, Xilin Wei<sup>24</sup>, Tiancheng Gu<sup>25</sup>, Dingkang Yang<sup>14</sup>, Hang Zhang<sup>2</sup>, Dayiheng Liu<sup>2</sup>, Hongsheng Li<sup>1†</sup>, Shuai Bai<sup>2†</sup>

<sup>1</sup>MMLab, CUHK <sup>2</sup>Alibaba Token Foundry <sup>3</sup>The Chinese Univerity of Hong Kong

<sup>4</sup>Fudan University <sup>5</sup>The University of Sydney

wangyuchi369@gmail.com hsli@ee.cuhk.edu.hk baishuai.bs@alibaba-inc.com

https://wangyuchi369.github.io/refcraft/

(a) Benchmark Composition  
![](images/e93189a013a315e6d20099493dbfa8e5fe603a384fc288a04e6036ef69c7dfb3.jpg)

![](images/41e120681541063cbee6112d911f8dc176f5bbd933bd5995dab2afb4cc28b08e.jpg)  
(c) Model Score vs. Token Use

![](images/bbcd68301beabd6e8707235629f29be683dedfd568ee621a66bf134d54f1fb0c.jpg)  
Figure 1: Overview of RefCraft. (a) RefCraft spans a broad spectrum of practical web development domains and features visual references that vary in intensity and source. (b) An example task in which a new webpage borrows the color palette from source webpage A and the layout from source webpage B. The agent may iteratively render, inspect, and revise the webpage. (c) Overall score versus reported token usage for nine frontier models. The dashed line denotes the Pareto frontier. The results show that the task remains far from solved.

## ABSTRACT

Beyond basic tool use and coding skills, effectively incorporating visual references into agentic generation is equally essential for applying AI agents to real-world productivity tasks. We study this crucial capability through webpage development, a practical and readily evaluable setting in which existing benchmarks remain largely idealized: they typically focus on reconstructing visual targets or generating webpages from textual instructions alone, overlooking the more common workflow of developing from visual examples. Specifically, we introduce RefCraft, a benchmark comprising 300 realistic multimodal webpage-development tasks that combine natural-language instructions with screenshots from one to five reference webpages. Agents must selectively interpret and utilize these references while rendering, visually inspecting, and iteratively refining their outputs. We evaluate the resulting open-ended artifacts using independently verifiable keypoints and complementary programmatic and VLM judges. Across nine frontier (M)LLMs, the strongest system achieves an overall score of only 52.12 out of 100, revealing that visual-reference-conditioned agentic generation remains a substantial challenge for current models in practical webpage-development settings.

## 1 INTRODUCTION

“Creativity is just connecting things.”

— Steve Jobs

Recently, as coding and co-working assistants, (M)LLMs have transformed numerous fields and substantially improved productivity (OpenAI, 2026b; Anthropic, 2026a; Qwen Team, 2026b; Kimi et al., 2026). This success has further fueled growing enthusiasm among researchers and practitioners for extending large models to an even broader range of practical tasks, as exemplified by GPT-6 Astra (OpenAI, 2026d) and Claude Fable 5.1 (Anthropic, 2026b). When we scrutinize some common productivity tasks, we find that visual references play an important role: designers may first study successful examples before developing new proposals and then continuously refine their designs through visual inspection, while CAD engineers examine existing parts to design new ones that fit and function properly. This observation highlights that, for both human experts and AI agents, basic tool use and coding skills alone do not encompass thefull spectrum ofproductive capabilities—effectively incorporating visual reference and inspection into agentic generation is equally essential.

To evaluate this important yet subtle capability, we ground it in a focused and readily verifiable testbed: webpage development, which is ubiquitous in everyday digital life and produces highly visual, easily assessable outputs. In fact, a growing body of benchmarks has investigated general webpage-related tasks, but most either use natural-language instructions without visual references (Lu et al., 2026; Zhu et al., 2025) or focus solely on reconstruction tasks, such as reconstructing webpages from screenshots (Yun et al., 2024; Wang et al., 2026) or videos (Chen et al., 2026). More recent studies have broadened the evaluation setting by incorporating interactive operations (Xiao et al., 2025a) or introducing webpage editing and repair tasks (Lei et al., 2026). Nevertheless, we argue that these prior efforts remain academically idealized and differ considerably from common real-world development workflows. In practice, frontend developers and users often begin with a rough design idea and browse existing websites for inspiration. For instance, a user may adopt the appealing layout of a reference website while redesigning its fonts, content, and other elements for a new purpose. Fully reconstructing an existing website is a far less common or representative task. To bridge this gap and, more importantly, evaluate whether AI agents can effectively conduct such a visual-reference-conditioned agentic generation process, we introduce RefCraft.

RefCraft comprises 300 diverse and realistic multimodal queries spanning a broad range of web development scenarios. Each query combines a natural-language user instruction with screenshots from one to five reference webpages, and models solve the task in an inspect-and-iterate setting where they can render and refine intermediate outputs. As shown in Fig. 1, RefCraft varies substantially in domains, reference composition, and the intended degree of reference utilization—from closely following a reference to selectively adapting its overall style or specific local elements. Evaluating such tasks requires moving beyond conventional target-based evaluation: a query may admit multiple valid solutions, and visual similarity alone cannot determine whether the references have been used appropriately. We therefore decompose each complex and subtle query into a verified set of key points, turning open-ended requirements into independently assessable criteria. Moreover, webpages are both visual and interactive artifacts, encompassing subjective perceptual qualities as well as objective executable behaviors. We thus adopt a dual-track evaluation protocol that uses programmatic judges for deterministic requirements and a VLM judge for perceptual and semantic ones. Together, these designs enable reliable evaluation of whether a generated webpage fulfills the user’s intended use of visual references, rather than merely resembling them.

We benchmark 9 frontier (M)LLMs on RefCraft under a unified harness. Despite access to visual references and an inspect-and-iterate workflow, the best-performing system, GPT-6 Astra (OpenAI, 2026d), achieves an overall score of only 52.12, leaving substantial room for improvement. Component-level results reveal that models often satisfy many individual requirements but fail to realize an essential reference relationship or core behavior. To assess the reliability of RefCraft, we compare its model ranking with both our expert human study and the human-preference leaderboard of Arena.AI<sup>1</sup>, finding strong agreement with both. These findings indicate that current agents can reproduce salient visual attributes but still struggle to translate instruction-dependent reference relationships into coherent designs—precisely the capability that matters to human users.

## 2 RELATED WORK

## 2.1 VISUAL-REFERENCE-CONDITIONED AGENTIC GENERATION

In recent years, we have witnessed rapid advances in AI agents. From early frameworks such as Re-Act (Yao et al., 2022) to coding agents such as SWE-agent (Yang et al., 2024) and OpenHands (Wang et al., 2025a), substantial effort has been devoted to reasoning, code generation, long-horizon planning, and tool use (Schick et al., 2023; Wang et al., 2024; Yang et al., 2026). In comparison, vision has received less attention in agentic generation and is typically used only as a perceptual interface for interpreting image or video observations (Xie et al., 2024; Qin et al., 2025; Chen et al., 2024). Meanwhile, visual inputs serve as important conditioning signals in generative models, particularly diffusion-based models for image and video generation (Seedance et al., 2026; Bai et al., 2025; Wang et al., 2025b). Yet methods ranging from DreamBooth (Ruiz et al., 2023) and ControlNet (Zhang et al., 2023) to IP-Adapter (Ye et al., 2023) and the OmniGen series (Xiao et al., 2025c; Wu et al., 2025) often assign predefined, coarse-grained roles to visual references—for example, preserving subject identity, transferring overall style, or imposing structural constraints. Consequently, both prominent research directions leave a challenging yet practical setting underexplored: visualreference-conditioned agentic generation, where an agent must interpret subtle user instructions and selectively incorporate heterogeneous visual references throughout the agentic generation process.

## 2.2 WEB DEVELOPMENT BENCHMARKS

Existing web development benchmarks broadly follow two task formulations. Text-driven benchmarks, such as WebGen-Bench (Lu et al., 2026) and FrontendBench (Zhu et al., 2025), ask models to construct webpages or applications from natural-language specifications (Tran et al., 2026; Xu et al., 2025). Visual-target benchmarks instead formulate generation as reconstruction, including screenshot-to-code tasks such as Design2Code (Si et al., 2025) and Web2Code (Yun et al., 2024), video- or interaction-conditioned generation (Chen et al., 2026; Xiao et al., 2025a), and sketch-to-code generation (Jain et al., 2019). More recent benchmarks, including DesignBench (Xiao et al., 2025b), ArtifactsBench (Zhang et al., 2025), and WebCompass (Lei et al., 2026), extend beyond one-shot generation to interactive behavior, editing, repair, and multi-turn refinement (Yang et al., 2025; Sun et al., 2025; Saxena et al., 2026). Evaluation for these tasks has evolved accordingly, moving from visual-similarity measures based on CLIP features (Radford et al., 2021), layout blocks, colors, and positions (Wang et al., 2026; Si et al., 2025) toward MLLM- or agent-as-a-judge evaluation over rendered webpages (Zhang et al., 2025; Lei et al., 2026; Zhuge et al., 2024). Nevertheless, existing benchmarks typically treat visual input as a target to reconstruct or an artifact to edit. They rarely capture the common workflow in which users consult one or more webpages as partial design references, which is precisely the setting studied in our work.

## 3 BENCHMARK DESIGN

Fig. 2 presents the overall design of RefCraft. Each benchmark instance, corresponding to a single task, consists of a textual user query, one to five visual references, and a verified set of evaluation keypoints. We construct these instances in three stages: curating a diverse reference pool (§ 3.2), synthesizing controllable user queries (§ 3.3), and decomposing each query into independently verifiable keypoints (§ 3.4). Further implementation details are provided in Appendix B.

![](images/a819ecd0f7e41545dbc33e3cb63b50f1cc0fb8cfa61d10ec94b6073d010c799b.jpg)  
Figure 2: Construction and evaluation of RefCraft. (a) Reference webpages are captured across interaction states. (b) A specification spanning six visual dimensions and five requirement types guides natural-language query synthesis. (c) Task requirements are decomposed into program-verifiable and VLM-assessed keypoints. (d) Given the query and reference images, agents iteratively build, render, inspect, and revise a webpage in a shared harness. (e) Keypoints guide evidence collection from generated artifacts. (f) Complementary programmatic and VLM judges produce the base keypoint score $B ,$ , core-function score $C ,$ and reference-following score R.

## 3.1 TASK FORMULATION

Each task t in RefCraft consists of a natural-language instruction $q _ { t }$ and a set of visual references $\mathcal { R } _ { t } = \{ R _ { i } \} _ { i = 1 } ^ { N _ { t } } ,$ , where $1 \le N _ { t } \le 5$ and each $R _ { i }$ contains screenshots sampled from a reference webpage. Together with the evaluation keypoints $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ , they define the task $\tau _ { t } = \bar { ( } q _ { t } , \mathcal { R } _ { t } , \mathcal { K } _ { t } )$ . Model m executes the task and produces $h _ { m , t } = \mathrm { A g e n t } _ { m } ( q _ { t } , \mathcal { R } _ { t } )$ , a self-contained HTML webpage, without access to $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ . During generation, the agent may render its intermediate implementation, visually inspect the result, and iteratively revise the webpage. Unlike screenshot reconstruction, the objective is not to reproduce any reference exactly. Depending on the instruction, the agent may need to closely follow one reference, combine properties from multiple references, adapt selected design elements, or deliberately avoid particular characteristics. The task therefore evaluates whether an agent can interpret and appropriately utilize visual references within an inspect-and-iterate workflow.

## 3.2 REFERENCE POOL CURATION

Reference webpages are important because they determine what information an agent can draw upon and therefore form the foundation of RefCraft. We curate the reference webpage pool according to three principles: diversity, covering a broad range of topics and design styles; visual richness, providing sufficiently salient properties for users to reference and evaluators to verify; and realism, reflecting designs that may plausibly appear in practical development workflows. Following these principles, we combine three sources: (1) openly licensed websites and webpage templates; (2) conventionally styled webpages synthesized by multiple models to increase diversity; and (3) deliberately synthesized webpages that deviate from conventional modern styles and common model-generated patterns. The rationale is that synthesized pages may inherit common model priors, allowing an agent to produce superficially similar designs based on its prior knowledge without consulting the references. To mitigate this shortcut, the third group incorporates less conventional visual characteristics, such as unusual color palettes, uncommon font combinations, and hard-edged shadows.

We represent each reference using images rather than source code or a live URL, preventing implementation details from leaking to the evaluated agent and better reflecting practical scenarios. Because a single static screenshot cannot expose dynamic properties such as hover effects or interactive state changes, we use Playwright<sup>2</sup> to capture a sequence of visual frames along scripted interaction traces involving scrolling, hovering, clicking, typing, and key presses. To support subsequent task construction, we also profile the observable information in each reference along six dimensions: layout, palette, typography, components, interaction, and overall visual character. Each dimension is annotated as rich, weak, or none, together with a concise description of the observed evidence. This profile prevents later queries from referring to properties that are absent or visually ambiguous.

## 3.3 CONTROLLABLE QUERY SYNTHESIS

Directly prompting an LLM to generate user queries leads to substantial mode collapse: the resulting instructions are often similarly phrased and repeatedly emphasize only a small subset of the available reference information. We instead adopt a specification-first approach: we first determine the intended reference usage and task composition and then verbalize them as a natural user request.

Reference-use specification. As shown in Fig. 2(b), let D denote the six visual dimensions above. We represent the reference-use specification for task t as $\mathcal { Z } _ { t } = \{ ( d , o _ { t , d } , \rho _ { t , d } ) : d \in \mathcal { D } \}$ . For each dimension, $o _ { t , d } \in \{ c o p y$ , adapt, invert, specify, ignore} specifies the requirement type, while $\rho _ { t , d }$ identifies the assigned reference, with $\rho _ { t , d } = \emptyset$ for specify and ignore. The five types are copy, which closely follows the reference; adapt, which preserves only selected characteristics; invert, which deliberately avoids the referenced characteristic; specify, which follows an explicit textual requirement rather than a reference; and ignore, which leaves the dimension unconstrained. We retain a sampled reference-dependent requirement only when the assigned reference contains sufficiently clear and discriminable evidence for the corresponding dimension.

Compositional query design. We construct a design matrix whose axes describe the user persona, target content, form and tone of the request, selected references, the reference-use specification $\mathcal { Z } _ { t }$ and other relevant factors. Values are sampled across these axes and subsequently filtered to remove contradictory, implausible, or otherwise pathological combinations. An LLM then converts each valid combination into a fluent natural-language instruction. This procedure preserves linguistic naturalness while maintaining systematic control over task diversity. Additional generation details and examples are provided in Appendix B.2.

## 3.4 VERIFIABLE KEYPOINT CONSTRUCTION

A query in RefCraft may admit many valid implementations, and its requirements can be fine-grained, subtle, and distributed across both language and visual references. Consequently, evaluating against a single canonical webpage or a holistic similarity score would be inappropriate. Inspired by rubricbased evaluation (Yu et al., 2026; Hashemi et al., 2024; Pathak et al., 2025), we instead decompose each query into a set of non-overlapping keypoints, each capturing one independently assessable requirement. We partition $\mathscr { K } _ { t } = \mathscr { K } _ { t } ^ { \mathrm { p r o g } } \dot { \cup } \mathscr { K } _ { t } ^ { \mathrm { v i m } }$ into disjoint tracks according to how each keypoint can be most reliably verified. Program-verifiable keypoints ${ \cal K } _ { t } ^ { \mathrm { p r o g } }$ capture deterministic properties such as color distance, contrast, responsive behavior, DOM state changes, and robustness to abnormal content. The remaining keypoints require semantic or perceptual judgment and are assigned to the VLM-assessed track ${ \mathcal { K } } _ { t } ^ { \mathrm { { v l m } } }$ . We follow the principle that whenever reliable deterministic verification is possible, we favor the programmatic track to reduce reliance on VLM judgments, which can be overly lenient. We additionally designate essential user requirements and page-integrity constraints as critical items, ensuring that they receive greater emphasis in the final evaluation.

We apply a multi-stage validation procedure to ensure keypoint quality. A separate model first reviews the full set of keypoints for completeness and may add missing requirements. It then examines each keypoint independently to identify hallucinated, redundant, or unverifiable criteria. Finally, ten web development experts review and revise the resulting keypoints. This process produces a verified evaluation specification for every benchmark instance.

## 4 EVALUATION PROTOCOL

Given the benchmark instances constructed above, our evaluation protocol standardizes how models interact with the task, converts their unconstrained HTML outputs into structured evidence, and scores them against the corresponding keypoints.

## 4.1 UNIFIED AGENTIC ENVIRONMENT

We evaluate each model as a coding agent, reflecting the increasingly common use of agentic harnesses in practical development workflows . To ensure comparability, all models operate through the same open-source coding harness, opencode <sup>3</sup>, with identical tools, workspace structure, and output requirements. For each task, the agent receives the user instruction, visual references, and an environment brief. It is given a dedicated working directory and may render, screenshot, and inspect intermediate implementations before revising them. The final result must be stored as a single selfcontained HTML file, ensuring a uniform artifact format across models. We restrict access that could reveal benchmark internals or ground-truth evaluation criteria. Each working directory is isolated from the benchmark repository, and attempts to access the repository or evaluation keypoints are denied. Network access remains available to preserve a realistic coding environment, but searching for or downloading the original source of a reference webpage invalidates the run.

## 4.2 EVIDENCE COLLECTION

Generated HTML files vary substantially in structure and may exhibit visual and interactive behaviors that cannot be reliably evaluated from source code alone. Therefore, for each artifact $h _ { m , t } ,$ we collect structured evidence $\mathcal { E } _ { m , t } = ( \mathcal { V } _ { m , t } , \mathcal { M } _ { m , t } )$ , where $\nu _ { m , t }$ contains captured views and $\mathcal { M } _ { m , t }$ contains browser measurements. Specifically, for each benchmark instance, we derive a task-specific capture script specifying a sequence of actions, such as scrolling, hovering, clicking, and typing. The script is designed to expose the page states required to evaluate as many relevant keypoints as possible. We load the generated HTML in Playwright, execute the actions sequentially, and capture screenshots of the resulting states. In parallel, we apply a fixed suite of browser-based measurements to assess properties such as rendering integrity, responsive behavior, robustness to abnormal content, and whether interactive controls produce genuine state changes. Together, the screenshots and measurements provide downstream judges with targeted evidence rather than requiring them to infer page behavior from raw HTML, which would be poorly aligned with practical usage.

## 4.3 DUAL-TRACK JUDGING AND SCORE AGGREGATION

Our evaluation first produces keypoint-level scores from $\mathcal { E } _ { m , t }$ using complementary programmatic and VLM judges. The programmatic judge compares browser measurements against deterministic keypoints and returns a binary verdict. For keypoints requiring semantic or perceptual judgment, the VLM judge receives the keypoint, relevant screenshots, and supporting measurements, and assigns a rating on a four-level scale from 0 to 3. Each keypoint is evaluated independently to reduce interference between unrelated requirements. Denote these judgments by $b _ { t , k } \in \{ 0 , 1 \}$ and $r _ { t , k } \in \{ 0 , 1 , 2 , 3 \}$ , respectively, omitting the model index for clarity. Both tracks are normalized to the same percentage scale: $s _ { t , k } = 1 0 0 b _ { t , k }$ for $k \in \mathbb { K } _ { t } ^ { \mathrm { p r o g } }$ and $s _ { t , k } = 1 0 0 r _ { t , k } / 3$ for $k \in  { \mathcal { K } } _ { t } ^ { \mathrm { v l m } }$

We aggregate these keypoint-level judgments into three complementary task-level quantities. The base keypoint score $B _ { t }$ is the mean of all normalized keypoint scores: $\begin{array} { r } { B _ { t } = | \mathcal { K } _ { t } | ^ { - 1 } \sum _ { k \in \mathcal { K } _ { t } } s _ { t , k } } \end{array}$ The core-function score $C _ { t }$ captures the weakest essential functionality of the generated webpage. The core goals include renderability and task-specific essential interactions, and $C _ { t }$ is defined as the minimum completion rate across these goals. This metric reflects the intuition that even strong performance on peripheral requirements cannot compensate for a broken primary function, as such failures fundamentally undermine the user experience. The reference-following score $R _ { t }$ measures whether the generated webpage realizes the requested relationship to its visual references, which is the central capability evaluated by RefCraft. We compute $R _ { t }$ by first averaging the keypoint scores within each reference group and then averaging across all applicable groups. Because core functionality and reference adherence are both essential to user-perceived quality, $C _ { t }$ and $R _ { t }$ act as gates on the aggregate keypoint score. The final task score therefore takes the minimum over the applicable quantities rather than naively averaging them:

Table 1: Results on the 300-task RefCraft benchmark. We report Overall and its three component scores: keypoint completion (Keypts., B), core functionality (Core, C), and reference following (Ref., R). Rollout statistics report mean agent turns, total tokens (Total Tok.), and tool calls per task. Bold and underlined values indicate the best and second-best entries in each column, respectively (higher scores and lower resource counts).
<table><tr><td></td><td colspan="4">Evaluation</td><td colspan="3">Rollout Statistics</td></tr><tr><td>Model</td><td>Overall↑</td><td>Keypts. B ↑</td><td>Core C ↑</td><td>Ref. R ↑</td><td>Turns</td><td>Total Tok.†</td><td>Calls</td></tr><tr><td>GPT-6 Astra</td><td>52.12</td><td>75.68</td><td>94.89</td><td>57.93</td><td>21.7</td><td>1.40M</td><td>30.1</td></tr><tr><td>Qwen3.8-Max</td><td>48.95</td><td>73.86</td><td>85.56</td><td>62.61</td><td>9.8</td><td>0.68 M</td><td>11.3</td></tr><tr><td>ø Grok 4.6</td><td>48.83</td><td>72.96</td><td>81.17</td><td>66.25</td><td>21.2</td><td>1.28 M</td><td>41.2</td></tr><tr><td>Kimi K3</td><td>47.55</td><td>72.74</td><td>85.00</td><td>61.33</td><td>25.4</td><td>2.09M</td><td>31.3</td></tr><tr><td>Claude Fable 5</td><td>44.61</td><td>72.05</td><td>77.83</td><td>61.44</td><td>15.2</td><td>1.10M</td><td>16.9</td></tr><tr><td>Z GLM-5.3</td><td>42.60</td><td>71.23</td><td>83.56</td><td>55.77</td><td>38.3</td><td>2.99M</td><td>42.2</td></tr><tr><td>DeepSeek-V4-Pro</td><td>37.09</td><td>71.16</td><td>71.72</td><td>53.46</td><td>25.3</td><td>1.12M</td><td>25.6</td></tr><tr><td>0 Seed2.1 Pro</td><td>32.06</td><td>69.00</td><td>56.56</td><td>58.67</td><td>22.7</td><td>1.01 M</td><td>21.8</td></tr><tr><td>Gemini 3.1 Pro </td><td>19.01</td><td>61.45</td><td>33.44</td><td>65.00</td><td>26.7</td><td>2.34M</td><td>25.9</td></tr></table>

<sup>†</sup> Token counts are taken from API usage reports and sum input, cache-read, cache-write, output, and reasoning tokens. M denotes millions.

$$
S _ { t } = \left\{ \begin{array} { l l } { \operatorname* { m i n } \{ B _ { t } , C _ { t } , R _ { t } \} , } & { \mathrm { i f ~ } R _ { t } \mathrm { ~ i s ~ d e f i n e d } , } \\ { \operatorname* { m i n } \{ B _ { t } , C _ { t } \} , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \qquad \bar { S } _ { m } = \frac { 1 } { | \mathcal { T } _ { m } | } \sum _ { t \in \mathcal { T } _ { m } } S _ { t } ,
$$

where $\mathcal { T } _ { m }$ denotes the set of tasks evaluated for model m. Additional details on judging and score aggregation are provided in Appendix B.5.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

We evaluate nine frontier models—GPT-6 Astra (OpenAI, 2026d), Claude Fable 5 (Anthropic, 2026a), Qwen3.8-Max (Qwen Team, 2026b), Grok 4.6 (xAI, 2026), Kimi K3 (Kimi et al., 2026), GLM-5.3 (Z.ai, 2026), DeepSeek-V4-Pro (Xu et al., 2026), Seed2.1 Pro (ByteDance Seed Team, 2026b), and Gemini 3.1 Pro (Google DeepMind, 2026)—as coding agents on RefCraft. All models are evaluated using the same OpenCode v1.18.25 harness, with high reasoning effort and a maximum of 100 turns. Keypoints are generated by Claude Opus 5 (Anthropic, 2026f) and verified by GPT-5.6 Sol (OpenAI, 2026c). We use Gemini 3.1 Pro as the VLM judge. We report the overall score S together with its three constituent metrics. Further details are provided in Appendix B.

## 5.2 MAIN RESULTS

Tab. 1 presents the overall and component scores of the evaluated frontier models, together with efficiency statistics such as token usage and the number of interaction turns. From these results, we could make the following observations:

Visual-reference-conditioned generation remains far from solved. As shown in Tab. 1, GPT-6 Astra achieves the highest overall score of 52.12 out of 100, most models cluster around 40, and Gemini 3.1 Pro scores below 20. These results leave substantial room for improvement. Moreover, the relatively low overall scores contrast with the higher and more concentrated keypoint scores of 61.45–75.68, indicating that models can satisfy many individual requirements yet struggle to produce artifacts that simultaneously preserve essential functionality and realize the user-specified relationship to the references. This gap captures the central challenge posed by RefCraft: a visually plausible webpage cannot be considered successful if it uses references inconsistently with the user’s instructions, reflecting how users assess artifacts in real-world settings.

Table 2: Benefits of references and revision.
<table><tr><td>Setting</td><td>GPT-6 Astra</td><td>DeepSeek- V4-Pro</td></tr><tr><td>(a) Reference ablation</td></tr><tr><td>Without references 17.3</td><td>11.1</td></tr><tr><td>With references</td><td>52.1 37.1</td></tr><tr><td>Gain +34.8</td><td>+26.0</td></tr><tr><td>(b) Inspection and revision</td><td></td></tr><tr><td>Initial 47.4</td><td>30.7</td></tr><tr><td>52.1</td><td>37.1</td></tr><tr><td>Final Gain +4.7</td><td>+6.4</td></tr></table>

Table 3: Human evaluation agreement.
<table><tr><td>Category</td><td>Human- human Krippendorff α</td><td>Evaluator vs. expert median Weighted Cohen κ</td></tr><tr><td colspan="3">(a) Keypoint-level agreement</td></tr><tr><td>Visual attributes</td><td>0.79</td><td>0.72</td></tr><tr><td>Visual style</td><td>0.65</td><td>0.47</td></tr><tr><td>Interaction</td><td>0.70</td><td>0.55</td></tr><tr><td>All keypoints</td><td>0.72</td><td>0.64</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">(b) Task-level Pearson r Evaluator vs. mean expert score</td></tr></table>

Reference use and functional execution constitute distinct bottlenecks. For seven of the nine models, the reference-following score R is the lowest component, identifying reference utilization as the dominant bottleneck. However, reference use alone does not determine end-to-end performance: Gemini 3.1 Pro achieves an R of 65.00 but ranks ninth overall because its core-function score is only 33.44, whereas GPT-6 Astra leads with the strongest core-function score of 94.89 despite a lower R. These contrasting failure regimes demonstrate that reliable implementation and appropriate reference utilization capture complementary capabilities, motivating their joint consideration in RefCraft.

More budget does not necessarily yield more effective performance. Performance is not monotonic in the number of turns, total reported tokens, or tool calls. Qwen3.8-Max outperforms Grok 4.6 while using substantially fewer total tokens (0.68M versus 1.28M), turns, and tool calls. GLM-5.3 uses the most turns, total tokens (2.99M), and tool calls, yet obtains a substantially lower overall score than the leading models. We further present a cost–efficiency Pareto curve in Fig. 1, which highlights the distinct efficiency profiles of different models.

## 5.3 FURTHER ANALYSIS

## 5.3.1 FINE-GRAINED CAPABILITY ANALYSIS

![](images/d65477df714b57ec620fb5a97748aadb84ed5a2433a6226b1f0d7dbb34d17e50.jpg)

![](images/1bd00ed52406edfc1c5364685ebd0f047f070c15eeb537e6035ce70898c88a3d.jpg)  
Figure 3: Fine-grained performance by (a) visual dimension and (b) requirement type, averaged equally across models.

We further analyze model performance across fine-grained categories, including visual dimensions and requirement types. Additional results are provided in Appendix A.5.

Fig. 3(a) presents performance across the six visual dimensions defined in § 3. Using an equal-weight average across models, palette (92.4), layout (83.1), and components (82.2) score higher than typography (76.1) and overall visual character (58.8). Interaction achieves the lowest score (43.6)

and ranks last for nearly all evaluated systems. These results indicate that agents transfer concrete, locally observable attributes more reliably than properties requiring coordination across multiple design choices or interactive states. Disaggregating keypoints by requirement type reveals differences in difficulty (Fig. 3 (b)). Models perform best on invert (98.87) and specify (87.68), followed by copy (82.42), whereas adapt receives the lowest score (75.49). This suggests that selectively transforming a referenced attribute to fit a new context is more challenging than directly reproducing, explicitly specifying, or avoiding it, despite being central to practical reference-guided development.

## 5.3.2 REFERENCE UTILIZATION AND VISUAL INSPECTION CAPABILITIES

As discussed in § 1, effectively incorporating visual references and refining intermediate outputs through visual inspection are important capabilities in agentic generation. We assess these capabilities via our RefCraft. (1) Reference ablation. We remove all visual references while leaving the original user query unchanged. The agent is explicitly informed that the references mentioned in the query are unavailable and is instructed to satisfy all requirements using its prior knowledge to make reasonable choices. (2) Pre- and post-inspection comparison. We evaluate the artifact produced in the first generation round, before the agent receives a rendered view of its output, and compare it with the final artifact. The results are reported in Tab. 2. Removing the visual references substantially reduces performance for both models. GPT-6 Astra drops from 52.1 to 17.3 points in overall score, while DeepSeek-V4-Pro drops from 37.1 to 11.1. These results indicate that effectively incorporating the supplied visual references is a crucial capability and that models cannot recover the reference-dependent requirements from the textual query and their priors alone. We also observe that performance improves from the initial to the final artifacts when models have additional opportunities to inspect and revise their outputs. Although this comparison does not rigorously isolate the causal contribution of visual inspection, since the final artifacts also benefit from additional tool calls, the consistent improvements suggest that the complete inspection-and-revision process is important.

![](images/4aec008f1a54b363b681a6557722a066548fb0b0673866283c8ed1994a767a1c.jpg)  
Figure 4: Two failure cases identified by RefCraft: one in reference following and the other in core functionality.

## 5.3.3 EVALUATION RELIABILITY AND QUALITATIVE ANALYSIS

To assess the reliability of our evaluation protocol, we sample 100 model–task artifacts and recruit ten web-development experts to evaluate each artifact independently. We aggregate the expert judgments using the median label for each keypoint and the mean score for each task. As shown in Tab. 3, the evaluator achieves substantial agreement with the expert consensus, both overall and across evaluation aspects, with a weighted κ of 0.64 across all keypoints. More subjective criteria, such as overall visual character, exhibit greater human–human and evaluator–human disagreement, as expected. The task-level scores also correlate strongly with the mean expert scores (Pearson r = 0.747), indicating that this alignment extends to aggregate assessments of task performance. We further compare our overall model ranking with the Image-to-Webdev leaderboard from Arena.AI, a popular platform for human evaluation of frontier models<sup>4</sup>. The two rankings achieve a Spearman correlation of $\rho = 0 . 8 9$ suggesting broad consistency between RefCraft and real-world user needs and preferences.

Fig. 4 presents two qualitative failure cases detected by RefCraft. In Task 96, Kimi K3 produces a polished news-timeline page that satisfies many keypoints and all core goals. Yet it adopts a centered, single-column feed rather than the requested full-width, two-column image–text layout, thereby failing the reference requirement. In Task 123, Grok 4.6 produces a step sequencer that follows the visual references, but its save function fails to retain edits, although edit persistence is explicitly requested in the query and can be implemented in JavaScript. Together, these cases illustrate how RefCraft identifies both unmet reference requirements and broken core functionality.

## 6 CONCLUSION

In this paper, we introduced RefCraft, a benchmark for studying reference-based webpage development and, more broadly, visual-reference-conditioned agentic generation in frontier models. RefCraft comprises 300 realistic web-development tasks in an inspect-and-iterate setting. We decompose each query into independently verifiable keypoints and evaluate the generated artifacts using comple mentary programmatic and VLM judges. Our evaluation of nine frontier (M)LLMs shows that this capability remains far from solved: the strongest system achieves an overall score of only 52.12. We hope that RefCraft will provide a valuable testbed for this crucial yet underexplored capability and stimulate further progress in visual-reference-conditioned agentic generation.

## AI USE STATEMENT

We used generative AI models in both benchmark construction and evaluation, including synthesizing reference webpages, profiling visual evidence, constructing and reviewing tasks and keypoints, and performing VLM-based evaluation (§ B). AI assistants also supported manuscript preparation, including routine tool development and language polishing. The authors reviewed all AI-assisted content and take full responsibility for the final text, experimental claims, and research artifacts.

## REPRODUCIBILITY STATEMENT

Task construction, scoring, and implementation details are documented in § 3, § 4, and § B, including prompts, browser settings, and execution budgets. Upon publication, we plan to publicly release RefCraft, including task inputs, evaluation scripts, previously generated HTML artifacts, run records, and item-level judgments.

## REFERENCES

Anthropic. System Card: Claude Opus 4.5. Technical report, Anthropic, November 2025. URL https://www.anthropic.com/claude-opus-4-5-system-card.

Anthropic. Claude Fable 5 & Claude Mythos 5 System Card. Technical report, June 2026a. URL https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf.

Anthropic. Claude Fable 5.1 and Claude Mythos 5.1, September 2026b. URL https://www.anthropic.com/claude-fable-and-mythos-5-1. Accessed: 2026-09-12.

Anthropic. System Card: Claude Opus 4.6. Technical report, Anthropic, February 2026c. URL https://www.anthropic.com/claude-opus-4-6-system-card.

Anthropic. System Card: Claude Opus 4.8. Technical report, Anthropic, May 2026d. URL https://www.anthropic.com/claude-opus-4-8-system-card.

Anthropic. System Card: Claude Sonnet 4.6. Technical report, Anthropic, February 2026e. URL https://www.anthropic.com/claude-sonnet-4-6-system-card.

Anthropic. Introducing Claude Opus 5. Technical report, Anthropic, 2026f. URL https://www.anthropic.com/news/claude-opus-5.

Jianhong Bai, Tianyu He, Yuchi Wang, Junliang Guo, Haoji Hu, Zuozhu Liu, and Jiang Bian. Uniedit: A unified tuning-free framework for video motion and appearance editing. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pp. 10171–10180, 2025.

ByteDance Seed Team. Seed2.0 Model Card: Towards Intelligence Frontier for Real-World Complexity. Technical report, ByteDance, February 2026a. URL https://lf3-static.bytednsdoc.c om/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2/0214/Seed2.0%20Model%20Card.pdf.

ByteDance Seed Team. Seed2.1 Model Card. Technical report, June 2026b. URL https://lf3-static.by tednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2.1/Seed2\_1\_Model\_Card.pdf. Covers Seed2.1 Pro and Seed2.1 Turbo.

Liang Chen, Yichi Zhang, Shuhuai Ren, Haozhe Zhao, Zefan Cai, Yuchi Wang, Peiyi Wang, Xiangdi Meng, Tianyu Liu, and Baobao Chang. Pca-bench: Evaluating multimodal large language models in perception-cognition-action chain. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 1086–1104, 2024.

Yang Chen, Minghao Liu, Yufan Shen, Yunwen Li, Tianyuan Huang, Xinyu Fang, Tianyu Zheng, Wenxuan Huang, Cheng Yang, Licheng Wen, et al. Iwr-bench: Can lvlms reconstruct interactive webpage from a user interaction video? In International Conference on Learning Representations, volume 2026, pp. 52158–52189, 2026.

GLM-5 Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, et al. GLM-5: from Vibe Coding to Agentic Engineering. arXiv preprint arXiv:2602.15763, 2026. URL https://arxiv.org/abs/2602.15763.

Google DeepMind. Gemini 3.1 Pro Model Card. Technical report, February 2026. URL https://deepmind.google/models/model-cards/gemini-3-1-pro/. Covers the gemini-3.1-pro-preview API model.

Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, and Chris Kedzie. Llm-rubric: A multidimensional, calibrated approach to automated evaluation of natural language texts. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 13806–13834, 2024.

Vanita Jain, Piyush Agrawal, Subham Banga, Rishabh Kapoor, and Shashwat Gulyani. Sketch2code: transformation of sketches to ui in real-time using deep neural network. arXiv preprint arXiv:1910.08930, 2019.

Team Kimi, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Xinping Lei, Xinyu Che, Junqi Xiong, Chenchen Zhang, Yukai Huang, Chenyu Zhou, Haoyang Huang, Minghao Liu, Letian Zhu, Hongyi Ye, et al. Webcompass: Towards multimodal web coding evaluation for code language models. arXiv preprint arXiv:2604.18224, 2026.

Zimu Lu, Yunqiao Yang, Houxing Ren, Haotian Hou, Han Xiao, Ke Wang, Weikang Shi, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Webgen-bench: Evaluating llms on generating interactive and functional websites from scratch. Advances in Neural Information Processing Systems, 38, 2026.

MiniMax. MiniMax M2.7: Early Echoes of Self-Evolution, March 2026. URL https://www.minimax.io/news/minimax-m27-en.

OpenAI. Introducing GPT-5.2, December 2025. URL https://openai.com/index/introducing-gpt-5-2/.

OpenAI. Introducing GPT-5.4, March 2026a. URL https://openai.com/index/introducing-gpt-5-4/.

OpenAI. GPT-5.5 Model, 2026b. URL https://developers.openai.com/api/docs/models/gpt-5.5. Accessed: 2026-09-12.

OpenAI. GPT-5.6 Sol Model, 2026c. URL https://developers.openai.com/api/docs/models/gpt-5.6-sol.

OpenAI. GPT-6 Astra Model, 2026d. URL https://developers.openai.com/api/docs/models/gpt-6-astra. Accessed: 2026-09-12.

Aditya Pathak, Rachit Gandhi, Vaibhav Uttam, Arnav Ramamoorthy, Pratyush Ghosh, Aaryan Raj Jindal, Shreyash Verma, Aditya Mittal, Aashna Ased, Chirag Khatri, et al. Rubric is all you need: Improving llm-based code evaluation with question-specific rubrics. In Proceedings ofthe 2025 ACM Conference on International Computing Education Research V. 1, pp. 181–195, 2025.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

Qwen Team. Qwen3.6-Plus: Towards Real World Agents, April 2026a. URL https://qwen.ai/blog?id=qwen3.6.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026b. URL https://qwen.ai/blog?id=qwen3.8.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22500–22510. IEEE, 2023.

Siddhant Saxena, Nilesh Trivedi, and Vinayaka Jyothi. Swe-webdevbench: Evaluating coding agent application platforms as virtual software agencies. arXiv preprint arXiv:2605.04637, 2026.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551, 2023.

Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

Chenglei Si, Yanzhe Zhang, Ryan Li, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: Benchmarking multimodal code generation for automated front-end engineering. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3956–3974, 2025.

Haoyu Sun, Huichen Will Wang, Jiawei Gu, Linjie Li, and Yu Cheng. Fullfront: Benchmarking mllms across the full front-end engineering workflow. arXiv preprint arXiv:2505.17399, 2025.

Hung Tran, Langston Nashold, Rayan Krishnan, Antoine Bigeard, and Alex Gu. Vibe code bench: Evaluating ai models on end-to-end web application development. In Proceedings ofthe ACM Conference on AI and Agentic Systems, pp. 514–536, 2026.

Kuang-Da Wang, Zhao Wang, Wei-Yao Wang, Yotaro Shimose, and Shingo Takamatsu. Webgen-v bench: Structured representation for enhancing visual design in llm-based web generation and evaluation. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 9927–9938, 2026.

Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better llm agents. arXiv preprint arXiv:2402.01030, 2024.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai software developers as generalist agents. In International Conference on Learning Representations, volume 2025, pp. 65882–65919, 2025a.

Yuchi Wang, Junliang Guo, Jianhong Bai, Runyi Yu, Tianyu He, Xu Tan, Xu Sun, and Jiang Bian. Instructavatar: Text-guided emotion and motion control for avatar generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 8132–8140, 2025b.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv e-prints, pp. arXiv–2506, 2025.

xAI. Introducing Grok 4.6, August 2026. URL https://x.ai/news/grok-4-6.

Jingyu Xiao, Yuxuan Wan, Yintong Huo, Zixin Wang, Xinyi Xu, Wenxuan Wang, Zhiyao Xu, Yuhang Wang, and Michael R Lyu. Interaction2code: Benchmarking mllm-based interactive webpage code generation from interactive prototyping. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), pp. 241–253. IEEE, 2025a.

Jingyu Xiao, Ming Wang, Man Ho Lam, Yuxuan Wan, Junliang Liu, Yintong Huo, and Michael R Lyu. Designbench: A comprehensive benchmark for mllm-based front-end code generation. arXiv preprint arXiv:2506.06251, 2025b.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13294–13304. IEEE, 2025c.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Kai Xu, YiWei Mao, XinYi Guan, and ZiLong Feng. Web-bench: A llm code benchmark based on web standards and frameworks. arXiv preprint arXiv:2505.07473, 2025.

John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

Kaiyu Yang, Yuyan Bu, Jing Yi, Yuchi Wang, Biyu Zhou, Juntao Dai, Songlin Hu, and Yaodong Yang. When lower privileges suffice: Investigating over-privileged tool selection in llm agents. ArXiv, abs/2606.20023, 2026. URL https://api.semanticscholar.org/CorpusID:289626701.

Zhen Yang, Wenyi Hong, Mingde Xu, Xinyue Fan, Weihan Wang, Jiale Cheng, Xiaotao Gu, and Jie Tang. Ui2codeˆ n: Ui-to-code generation as interactive visual optimization. arXiv preprint arXiv:2511.08195, 2025.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Weihao Yu, Rongyao Fang, Yuxuan Cai, Linjiang Huang, Yuhuan Yang, Xianwei Zhuang, Junyang Lin, Yixuan Yuan, and Shuai Bai. Specv: Specification verification for robust unified multimodal evaluation. In European Conference on Computer Vision, pp. 59–75. Springer, 2026.

Sukmin Yun, Haokun Lin, Rusiru Thushara, Mohammad Q Bhat, Yongxin Wang, Zutao Jiang, Mingkai Deng, Jinhong Wang, Tianhua Tao, Junbo Li, et al. Web2code: A large-scale webpage-to-code dataset and evaluation framework for multimodal llms. Advances in neural information processing systems, 37:112134–112157, 2024.

Z.ai. GLM-5.3: Frontier Coding with Emergent Cyber Capabilities, 2026. URL https://z.ai/blog/glm-5.3.

Chenchen Zhang, Yuhang Li, Can Xu, Jiaheng Liu, Ao Liu, Changzhi Zhou, Ken Deng, Dengpeng Wu, Guanhua Huang, Kejiao Li, et al. Artifactsbench: Bridging the visual-interactive gap in llm code generation evaluation. arXiv preprint arXiv:2507.04952, 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3813–3824. IEEE, 2023.

Hongda Zhu, Yiwen Zhang, Bing Zhao, Jingzhe Ding, Siyao Liu, Tong Liu, Dandan Wang, Yanan Liu, and Zhaojian Li. Frontendbench: A benchmark for evaluating llms on front-end development via automatic evaluation. arXiv preprint arXiv:2506.13832, 2025.

Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, et al. Agent-as-a-judge: Evaluate agents with agents. arXiv preprint arXiv:2410.10934, 2024.

## APPENDIX OVERVIEW

We organize the appendix as follows:

• Appendix A presents additional experiments and analyses, including:

– benchmark statistics (Appendix A.1)

– pairwise model comparisons (Appendix A.2)

– judge sensitivity and stability (Appendix A.3)

– evaluation protocol ablations (Appendix A.4)

– complete capability breakdowns (Appendix A.5)

– cross-harness sensitivity (Appendix A.6)

– a worked example of RefCraft (Appendix A.7)

• Appendix B provides additional methodological details, including:

– reference collection and visual evidence (Appendix B.1)

– controlled task construction (Appendix B.2)

– keypoint construction and quality assurance (Appendix B.3)

– standardized agentic execution (Appendix B.4)

– evidence collection and dual-track judging (Appendix B.5)

## A ADDITIONAL EXPERIMENTS

The main experiments establish aggregate model performance and identify challenges in reference utilization and functional execution. Here, we examine how these results vary across individual tasks, and how their interpretation depends on the evaluation and execution setup. These analyses comple ment the main comparison by distinguishing performance differences from variation introduced by the judge, scoring protocol, or coding harness

## A.1 BENCHMARK STATISTICS

We summarize the composition of RefCraft in Fig. 5. The final 300 tasks span eight application domains, 40 subdomains, and 186 distinct webpage types. Each task uses one to five reference webpages, with three- and four-reference tasks accounting for 78 and 72 instances, respectively. The benchmark also varies the intended degree of reference use, from loose guidance to near-copying, as determined by requirement types such as adapt and copy, and includes 151 English and 149 Chinese requests. Together, these distributions capture variation in both what agents are asked to build and how they should use the supplied references. The 744-page reference pool is counted separately because each source page may support multiple tasks.

During evaluation, each keypoint is instantiated as a single evaluation record. This process yields 12,682 records, comprising 10,542 VLM-assessed criteria, 1,540 programmatic probes, and 600 renderability records. The first two categories form the 12,082-item score-bearing set, whereas the renderability records support auditing and core evaluation without contributing additional scoring items. Within the score-bearing set, 3,106 criteria assess directive-related requirements; Fig. 5(d) presents their breakdown.

## A.2 PAIRWISE MODEL COMPARISONS

We provide detailed task-paired comparisons among the nine evaluated models in Tab. 4. For each model pair, we compute the mean difference in overall score across the same tasks and report the corresponding proportions of wins, ties, and losses. The head-to-head results are broadly consistent with the aggregate ranking, while showing that a higher-ranked model does not uniformly outperform its counterpart. In particular, comparisons between closely ranked systems contain substantial proportions of both ties and losses. These paired results clarify how a leaderboard advantage is distributed across tasks, rather than treating a difference in average score as evidence of uniformly superior performance.

![](images/731674526eebfd81ae496328499e1ddd76c3ad85d0b78b0fabf8fde359313e04.jpg)  
Figure 5: Composition and diversity of RefCraft.

## A.3 JUDGE SENSITIVITY AND STABILITY

In our evaluation, we use Gemini 3.1 Pro as the VLM judge. A natural question is whether the broad model comparison depends on the choice of automatic evaluator. We examine this sensitivity by re-scoring artifacts from the nine evaluated systems on a fixed 32-task sample using Gemini 3.1 Pro, GPT-6 Astra, and Claude Opus 5. Each judge receives identical tasks, keypoints, reference mappings, rendered views, browser measurements, and interaction evidence. As shown in Fig. 6, the three judges produce similar model-level score patterns, suggesting that the broad comparison on this sample is not specific to a single evaluator.

We further examine the stability of the same judge across repeated evaluations. Specifically, we run Gemini 3.1 Pro three times on the same sample with identical inputs. Pairwise correlations between repeated scores reach 0.92, indicating that the overall score pattern is reproducible across runs.

![](images/a50154f0a75d2474bf6dd254132871c081c87e5e260868e9faae6dc07d921cb2.jpg)  
Figure 6: Judge sensitivity on the fixed 32-task sample. Colored points and connecting lines show scores assigned by Gemini 3.1 Pro, GPT-6 Astra, and Claude Opus 5. Whiskers indicate illustrative uncertainty intervals. Relative scoring patterns are broadly consistent across the three models.

Table 4: Paired comparisons among the nine main-table models. ∆ is Model A minus Model B.
<table><tr><td>Model A</td><td>Model B</td><td>Δ</td><td>A win (%)</td><td>Tie (%)</td><td>B win (%)</td></tr><tr><td>GPT-6 Astra</td><td>Qwen3.8-Max</td><td>3.17</td><td>38.3</td><td>24.7</td><td>37.0</td></tr><tr><td>GPT-6 Astra</td><td>Grok 4.6</td><td>3.29</td><td>42.7</td><td>21.7</td><td>35.7</td></tr><tr><td>GPT-6 Astra</td><td>Kimi K3</td><td>4.57</td><td>44.0</td><td>20.7</td><td>35.3</td></tr><tr><td>GPT-6 Astra</td><td>Claude Fable 5</td><td>7.51</td><td>46.7</td><td>20.7</td><td>32.7</td></tr><tr><td>GPT-6 Astra</td><td>GLM-5.3</td><td>9.52</td><td>47.3</td><td>23.7</td><td>29.0</td></tr><tr><td>GPT-6 Astra</td><td>DeepSeek-V4-Pro</td><td>15.03</td><td>51.0</td><td>25.3</td><td>23.7</td></tr><tr><td>GPT-6 Astra</td><td>Seed2.1 Pro</td><td>20.06</td><td>61.3</td><td>22.0</td><td>16.7</td></tr><tr><td>GPT-6 Astra</td><td>Gemini 3.1 Pro</td><td>33.11</td><td>71.0</td><td>20.3</td><td>8.7</td></tr><tr><td>Qwen3.8-Max</td><td>Grok 4.6</td><td>0.12</td><td>37.0</td><td>26.0</td><td>37.0</td></tr><tr><td>Qwen3.8-Max</td><td>Kimi K3</td><td>1.40</td><td>41.3</td><td>24.0</td><td>34.7</td></tr><tr><td>Qwen3.8-Max</td><td>Claude Fable 5</td><td>4.34</td><td>41.0</td><td>25.0</td><td>34.0</td></tr><tr><td>Qwen3.8-Max</td><td>GLM-5.3</td><td>6.35</td><td>44.7</td><td>28.0</td><td>27.3</td></tr><tr><td>Qwen3.8-Max</td><td>DeepSeek-V4-Pro</td><td>11.86</td><td>48.0</td><td>30.3</td><td>21.7</td></tr><tr><td>Qwen3.8-Max</td><td>Seed2.1 Pro</td><td>16.89</td><td>55.7</td><td>26.0</td><td>18.3</td></tr><tr><td>Qwen3.8-Max</td><td>Gemini 3.1 Pro</td><td>29.94</td><td>68.0</td><td>23.7</td><td>8.3</td></tr><tr><td>Grok 4.6</td><td>Kimi K3</td><td>1.28</td><td>42.3</td><td>24.0</td><td>33.7</td></tr><tr><td>Grok 4.6</td><td>Claude Fable 5</td><td>4.21</td><td>39.7</td><td>27.3</td><td>33.0</td></tr><tr><td>Grok 4.6</td><td>GLM-5.3</td><td>6.23</td><td>43.3</td><td>23.3</td><td>33.3</td></tr><tr><td>Grok 4.6</td><td>DeepSeek-V4-Pro</td><td>11.74</td><td>43.7</td><td>32.7</td><td>23.7</td></tr><tr><td>Grok 4.6</td><td>Seed2.1 Pro</td><td>16.77</td><td>54.0</td><td>28.0</td><td>18.0</td></tr><tr><td>Grok 4.6</td><td>Gemini 3.1 Pro</td><td>29.82</td><td>67.0</td><td>26.7</td><td>6.3</td></tr><tr><td>Kimi K3</td><td>Claude Fable 5</td><td>2.94</td><td>37.3</td><td>26.0</td><td>36.7</td></tr><tr><td>Kimi K3</td><td>GLM-5.3</td><td>4.95</td><td>41.3</td><td>26.3</td><td>32.3</td></tr><tr><td>Kimi K3</td><td>DeepSeek-V4-Pro</td><td>10.46</td><td>44.3</td><td>32.7</td><td>23.0</td></tr><tr><td>Kimi K3</td><td>Seed2.1 Pro</td><td>15.49</td><td>53.0</td><td>26.0</td><td>21.0</td></tr><tr><td>Kimi K3</td><td>Gemini 3.1 Pro</td><td>28.54</td><td>67.7</td><td>22.7</td><td>9.7</td></tr><tr><td>Claude Fable 5</td><td>GLM-5.3</td><td>2.02</td><td>40.3</td><td>26.7</td><td>33.0</td></tr><tr><td>Claude Fable 5</td><td>DeepSeek-V4-Pro</td><td>7.52</td><td>43.3</td><td>33.0</td><td>23.7</td></tr><tr><td>Claude Fable 5</td><td>Seed2.1 Pro</td><td>12.56</td><td>49.3</td><td>31.0</td><td>19.7</td></tr><tr><td>Claude Fable 5</td><td>Gemini 3.1 Pro</td><td>25.60</td><td>61.3</td><td>29.0</td><td>9.7</td></tr><tr><td>GLM-5.3</td><td>DeepSeek-V4-Pro</td><td>5.51</td><td>37.3</td><td>35.7</td><td>27.0</td></tr><tr><td>GLM-5.3</td><td>Seed2.1 Pro</td><td>10.54</td><td>47.7</td><td>29.0</td><td>23.3</td></tr><tr><td>GLM-5.3</td><td>Gemini 3.1 Pro</td><td>23.59</td><td>60.0</td><td>27.0</td><td>13.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>Seed2.1 Pro</td><td>5.03</td><td>37.7</td><td>39.0</td><td>23.3</td></tr><tr><td>DeepSeek-V4-Pro</td><td>Gemini 3.1 Pro</td><td>18.08</td><td>51.7</td><td>35.0</td><td>13.3</td></tr><tr><td>Seed2.1 Pro</td><td>Gemini 3.1 Pro</td><td>13.05</td><td>40.7</td><td>46.0</td><td>13.3</td></tr></table>

## A.4 EVALUATION PROTOCOL ABLATIONS

The protocol in § 4 combines requirement-level assessment with visual and executable evidence. To examine the role of these design choices, we compare the complete protocol with two plausible alternatives on the same generated artifacts. Holistic evaluation replaces keypoint decomposition and B/C/R aggregation with a direct overall rating, testing how the assessment changes when requirements are no longer scored and aggregated separately. Screenshot-only evaluation retains the rendered images but removes browser measurements and action traces, testing what can be established from visual evidence alone.

As shown in Fig. 7, holistic scores exceed complete-protocol scores by 42.3 points. Because holistic evaluation changes both the rating granularity and the aggregation rule, this gap alone cannot establish which protocol better captures artifact quality. Nevertheless, the large increase suggests that holistic evaluation may overestimate model capabilities by overlooking failures exposed by the complete protocol. Removing executable evidence has a clearer effect on evaluation coverage. Unresolved core cases increase from 26 to 437 out of 918, while unresolved keypoints increase from 15 to 1,987 out of 7,608. These results indicate that visual evidence alone is often insufficient to determine whether requirements are satisfied, making screenshot-only evaluation incomplete and unreliable for assessing overall artifact behavior.

![](images/94fe9678998ec919393c355e7ee848c2d46367aeb1e4d30ef64d6f9db8c81f26.jpg)

![](images/7f792905978a82648f1869e6485b96f3d32e7c8635de026b5bbf056a0c53539d.jpg)  
Figure 7: Protocol sensitivity and evidence coverage. (a) Changes in Overall score relative to the complete protocol: +42.3 under holistic rating and −4.6 under screenshot-only evaluation. (b) Observed rates of unresolved core cases across matched artifacts.

## A.5 COMPLETE CAPABILITY BREAKDOWNS

The fine-grained analysis in § 5.3.1 focuses on visual dimensions and requirement types. Tab. 6 and 5 complement this analysis by examining application domains and evidence sources across the nine evaluated models. The domain breakdown reports task-level overall scores, capturing end-to-end performance within each task group. The source breakdown instead reports reference-use keypoint scores, obtained by first aggregating the relevant keypoints within each task and then averaging across tasks. Across domains, equal-model averages are highest for personal and portfolio (44.7), organizations (44.4), and commerce (44.1), and lowest for data and monitoring (33.7) and events and community (37.1). Tab. 5 shows that models score higher on text-specified requirements than on requirements associated with atypical synthetic, licensed/template, or conventional synthetic references. This gap suggests that explicit textual constraints are easier to follow than properties that must be inferred from visual references.

Table 5: Reference-related keypoint performance by evidence source.
<table><tr><td>Model</td><td>Licensed/ template</td><td>Atypical synthetic</td><td>Conventional synthetic</td><td>Text specification</td></tr><tr><td>GPT-6 Astra</td><td>82.7</td><td>78.8</td><td>80.7</td><td>91.6</td></tr><tr><td>Qwen3.8-Max</td><td>81.9</td><td>80.0</td><td>88.5</td><td>90.9</td></tr><tr><td>Grok 4.6</td><td>83.3</td><td>82.5</td><td>87.4</td><td>91.1</td></tr><tr><td>Kimi K3</td><td>81.6</td><td>79.3</td><td>81.7</td><td>85.5</td></tr><tr><td>Claude Fable 5</td><td>81.7</td><td>78.5</td><td>88.0</td><td>87.4</td></tr><tr><td>GLM-5.3</td><td>80.5</td><td>75.7</td><td>80.8</td><td>88.1</td></tr><tr><td>DeepSeek-V4-Pro</td><td>79.3</td><td>76.0</td><td>80.3</td><td>86.1</td></tr><tr><td>Seed2.1 Pro</td><td>80.0</td><td>76.8</td><td>81.9</td><td>81.8</td></tr><tr><td>Gemini 3.1 Pro</td><td>80.9</td><td>81.7</td><td>83.7</td><td>86.6</td></tr></table>

Table 6: Task-level Overall score across the eight high-level application domains.
<table><tr><td>Model</td><td>Content/ media</td><td></td><td>Commerce Organizations Branding</td><td></td><td>Events/ community</td><td>Data/ monitoring</td><td>Tools/ apps</td><td>Personal/ portfolio</td></tr><tr><td>GPT-6 Astra</td><td>50.2</td><td>57.6</td><td>53.9</td><td>54.4</td><td>46.8</td><td>46.0</td><td>54.4</td><td>53.6</td></tr><tr><td>Qwen3.8-Max</td><td>49.2</td><td>50.5</td><td>50.2</td><td>49.4</td><td>42.2</td><td>50.0</td><td>49.4</td><td>51.2</td></tr><tr><td>Grok 4.6</td><td>46.0</td><td>52.0</td><td>55.6</td><td>49.3</td><td>49.6</td><td>34.7</td><td>49.6</td><td>52.6</td></tr><tr><td>Kimi K3</td><td>43.9</td><td>49.4</td><td>53.5</td><td>46.8</td><td>39.5</td><td>44.5</td><td>51.2</td><td>54.7</td></tr><tr><td>Claude Fable 5</td><td>42.6</td><td>50.6</td><td>50.7</td><td>43.8</td><td>43.5</td><td>35.8</td><td>46.9</td><td>40.9</td></tr><tr><td>GLM-5.3</td><td>45.9</td><td>40.4</td><td>45.8</td><td>39.9</td><td>38.2</td><td>35.7</td><td>48.2</td><td>46.7</td></tr><tr><td>DeepSeek-V4-Pro</td><td>40.3</td><td>38.7</td><td>37.7</td><td>38.2</td><td>32.4</td><td>25.0</td><td>37.4</td><td>45.6</td></tr><tr><td>Seed2.1 Pro</td><td>33.9</td><td>37.9</td><td>32.2</td><td>38.8</td><td>22.0</td><td>20.2</td><td>33.0</td><td>36.4</td></tr><tr><td>Gemini 3.1 Pro</td><td>16.1</td><td>19.8</td><td>20.4</td><td>20.3</td><td>19.9</td><td>11.3</td><td>24.7</td><td>21.0</td></tr></table>

## A.6 CROSS-HARNESS SENSITIVITY

The shared OpenCode configuration ensures a consistent main comparison across models, but real-world agent systems use diverse harnesses. To test sensitivity to the choice of harness, we evaluate GPT-6 Astra and Kimi K3 on a fixed 32-task sample using two popular alternatives: Claude Code<sup>5</sup> and Codex<sup>6</sup>. We also include the released OpenCode artifacts for the same 32 tasks, re-scored using the same evaluation protocol, as the main-harness baseline.

For the Claude Code and Codex reruns, we match the task text, ordered references, initial workspaces, and nominal resource limits, while retaining each harness’s native system prompts, tools, image preprocessing, and context-management mechanisms. Tab. 7 summarizes the overall scores for each model–harness combination. Scores vary modestly across harnesses. GPT-6 Astra ranges from 49.2 to 53.0, a spread of 2.8 points, while Kimi K3 ranges from 43.1 to 49.4, a spread of 1.1 points. GPT-6 Astra outperforms Kimi K3 under all three harnesses, preserving their relative ordering. These results suggest that the harness affects absolute scores, while overall performance and model ordering remain relatively stable on this sample.

Table 7: Cross-harness sensitivity on the fixed 32-task sample. We report Overall scores across three harnesses.
<table><tr><td>Model</td><td>Harness</td><td>Overall score</td></tr><tr><td>GPT-6 Astra</td><td>OpenCode (released)</td><td>49.2</td></tr><tr><td>GPT-6 Astra</td><td>Claude Code</td><td>51.3</td></tr><tr><td>GPT-6 Astra</td><td>Codex</td><td>53.0</td></tr><tr><td>Kimi K3</td><td>OpenCode (released)</td><td>48.2</td></tr><tr><td>Kimi K3</td><td>Claude Code</td><td>49.4</td></tr><tr><td>Kimi K3</td><td>Codex</td><td>43.1</td></tr></table>

## A.7 A WORKED EXAMPLE OF REFCRAFT

Fig. 9 traces how GPT-6 Astra, running in OpenCode, completes Task 163: a marine repair quote builder. The model first identifies which properties to borrow from each reference and which to avoid, combining bold typography and light service cards while rejecting the dark reference palette. It then builds a self-contained index.html, inspects rendered views, and revises the tablet heading and booking controls. Functional checks verify that repair selections update the quote and that editing a confirmed plan reopens the booking form; the final page also shows no horizontal overflow across 13 tested widths. This example illustrates how reference interpretation, visual refinement, and interaction testing work together to turn a user request into a functional webpage.

## B ADDITIONAL IMPLEMENTATION DETAILS

§ 3 and § 4 describe the design of RefCraft and its evaluation protocol. Here, we explain how these designs are implemented. We follow the same progression as the main text: reference webpages are first converted into visual evidence, which supports the construction of user requests and their evaluation keypoints. Once these task components are fixed, an agent uses the request and reference images to develop a webpage, while a separate evaluator assesses the submitted artifact against the hidden keypoints. The details below apply to the RefCraft benchmark and clarify how information passes between these stages.

## B.1 REFERENCE COLLECTION AND VISUAL EVIDENCE

We begin by preparing a reusable collection of webpages from which task-specific references can be assembled. Following the three-source design in § 3.2, this collection contains 744 pages: 444 openly licensed template-derived pages, 200 conventionally styled synthetic pages, and 100 atypical synthetic pages. These counts describe the available source pool rather than the benchmark instances.

A task draws visual references from one to five source pages, and the same source page may support different requests that ask agents to use its properties in different ways.

Template sources and licensing. For the template-derived portion, we collect openly licensed pages from collections including HTML5 UP, Colorlib, Start Bootstrap, Creative Tim, AdminLTE, Themefisher, and CodeWithSadee. The retained copies comprise 367 MIT-licensed pages, 76 under CC BY 3.0, and one under Apache-2.0. For each page, we record its source URL, author, license, and design-family identifier, preserving the connection between the benchmark reference and its original source. We also package the page’s styles, scripts, and image assets into a self-contained HTML snapshot. This snapshot provides a stable input to the capture process: the browser renders the stored page rather than relying on the continued availability or current appearance of the live website.

Synthetic reference generation. The synthetic portion complements these existing templates by extending the range of designs available for task construction. Across the 300 synthetic pages, we use GPT-5.2 and GPT-5.4 (OpenAI, 2025; 2026a), Claude Opus 4.5, Claude Opus 4.6, Claude Opus 4.8, and Claude Sonnet 4.6 (Anthropic, 2025; 2026c;d;e), Gemini 3.1 Pro (Google DeepMind, 2026), Qwen3.6-Plus (Qwen Team, 2026a), Seed2.0 Pro (ByteDance Seed Team, 2026a), GLM-5 (GLM-5 Team et al., 2026), and MiniMax M2.7 (MiniMax, 2026) as reference generators. As discussed in the main text, however, model-generated pages can themselves converge on familiar design patterns. For the 100 atypical pages, we therefore pair a controlled design sampler with Claude Opus 4.8 (Anthropic, 2026d) to introduce less conventional visual combinations. The sampler combines 16 predefined design families (e.g., Swiss typography <sup>7</sup>, brutalism, and terminal interfaces) with varied palettes and family-specific font stacks, using flat fills, square corners, and optional hard shadows. The model then refines the copy and page structure, while automated checks enforce the palette, font, and styling constraints. These models participate in constructing the reference pool, a role distinct from that of the nine systems subsequently evaluated on the benchmark.

From webpages to task evidence. The stored webpages are construction assets, but the references shown to agents consist only of images. To produce these visual inputs, Playwright renders each page at a 1920 × 1080 viewport and captures scripted scrolling and interaction states, yielding 5,522 frames across the pool. For conventionally styled synthetic pages, we retain their original page-specific demonstration scripts. For template-derived and atypical pages, we generate scripts from the measured page height and available hover targets (identified through known selectors or DOM probing), interleaving stepwise scrolling, hovers, and short pauses. We retain the capture-action script and a time-stamped frame manifest for each page so that the resulting images remain associated with the actions and source that produced them. When references are assembled into a task, frame order and source-page boundaries are preserved. The agent can thus distinguish views of the same page from views of different references without receiving their implementations. Before a request can be written, we also need to establish which reference properties these images actually expose. Gemini 3.1 Pro (Google DeepMind, 2026) profiles the frames along the six visual dimensions in § 3.2, providing a description of the visible evidence and a rich, weak, or none label for each dimension. These profiles connect reference collection to query synthesis: they restrict the assignment of reference-dependent requirements to properties supported by the supplied images, rather than by information available only in the underlying webpage.

## B.2 CONTROLLED TASK CONSTRUCTION

We construct each user request from a structured recipe that combines the reference-use specification in § 3.3 with choices about the target webpage and how the user expresses their needs. The target domain is drawn from a taxonomy of 8 broad categories, 40 subcategories, and 206 site types, ranging from productivity tools such as kanban boards to salon booking pages and API documentation. The recipe also selects one of 14 user profiles, each defining an identity, priorities, and speaking style. For example, a shop owner may emphasize contact details in everyday language, whereas a front-end engineer focuses on semantic structure and responsive behavior.

The recipe also controls the intended page’s functional complexity, content density, and explicit constraints. Functional complexity ranges from static pages to stateful applications and data visualization. Up to three tags specify requirements concerning audience, presentation, content volume, delivery, tone, or functionality. These can call for mobile-first display, long-form text, image-led content, offline operation, or bilingual switching. Concrete visual requirements, such as body text of at least 18px, a specified brand color, or square-cornered buttons, are recorded independently of the references.

Separately, we sample the language, form, and level of detail of the request. Requests are written in Chinese or English and take the form of flowing prose, casual messages, a flat list of requirements, or a structured product requirements document (PRD) with section headings and acceptance criteria. The level of detail is sampled independently of the page’s functional complexity. It ranges from a single overall need, through a brief description of three to five sections, to detailed content, data, and interactions for each section. Target length depends on both form and level of detail and is measured in Chinese characters or English words. Two exemplars matching the request’s language and form guide organization and tone without supplying topics or wording to reuse.

Gemini 3.1 Pro then converts the recipe and ordered screenshot groups into a natural-language request using the prompt in Fig. 10. The prompt asks the model to identify each source by its appearance or position and explain how its selected properties should inform the design. Each reference is used only for the specified aspects of the page, and its content is not reused. Explicit values, such as the 18px minimum for body text, must be preserved so that constraints remain checkable after rewording. Visual descriptions must be grounded in the screenshots; motion that is not visible in them is phrased as a new user requirement. We retain the resulting request with its recipe for keypoint construction. Fig. 8 shows three examples with different forms and user profiles.

![](images/adb539b98b221dba7fe55f5b0bc4ae87da8c8ebe8c57f4e722d39be937697278.jpg)  
Figure 8: Generated user query examples. English requests from the final task set illustrate casual, bulleted, and structured PRD forms. Task 132 is shown in full; Tasks 279 and 161 are excerpted, with [...] marking omissions. Original wording is retained.

## B.3 KEYPOINT CONSTRUCTION AND QUALITY ASSURANCE

Keypoint construction must preserve the reference relationships established during query synthesis while also covering the content and functionality expressed in the completed request. We handle these two parts separately. Reference assignments and explicit textual constraints are converted programmatically into fixed items from the structured specification, retaining their intended sources and requirement types. Claude Opus 5 (Anthropic, 2026f) then reads the user request and adds keypoints for content, structure, navigation, and functionality without rewriting the fixed items. This division keeps the specified reference relationships intact while allowing the keypoints to cover requirements that emerge in the natural-language request.

To make the added keypoints assessable, the construction prompt (Fig. 12) links each criterion to both the user request and the evidence needed to evaluate it. Each item must first cite the supporting request text and clarify what it checks: the existence of an element, the correctness of its implementation, or the precision of its details. Distinguishing these levels prevents the mere presence of an element from being treated as sufficient evidence that it has been implemented as instructed. The resulting criterion must then be answerable from the evidence available during evaluation, ruling out questions requiring source-code inspection or unavailable timing measurements. For interaction-dependent requirements, this also means identifying both the control and the action needed to reach the relevant state, so that the subsequent browser-capture process can expose the behavior being assessed.

The generated keypoints are then reviewed by GPT-5.6 Sol (OpenAI, 2026c) with high reasoning effort and the prompt shown in Fig. 13. The review separates completeness from item validity: a global pass first proposes missing requirements, after which a local pass examines individual items and returns keep, modify, or remove, together with a reason. Both passes use the reference-use specification to check whether the criteria preserve the intended task. In particular, an ignored dimension must not introduce a requirement, an adaptation must not become an exact-match demand, and an inversion must not reward copying. These checks address errors that could otherwise make a correct implementation appear unsuccessful. Generation and model review use a configured maximum output of 128,000 tokens. Finally, the expert review described in § 3 revises the keypoints before they are used for evaluation. Fig. 11 presents the instruction template for expert keypoint review.

Final task set. The final benchmark contains 300 tasks, each pairing a user request and ordered reference screenshots with a verified set of keypoints. Together, these tasks use 534 distinct webpages in 972 reference assignments and expose 3,491 reference frames. Stable task identifiers and input hashes preserve the correspondence between each request, its references, and its evaluation specification when these components are passed to the execution and judging stages.

## B.4 STANDARDIZED AGENTIC EXECUTION

With the task components fixed, execution exposes only the information intended for the evaluated agent: the user request, ordered reference images, and an environment brief. The verified keypoints remain hidden and are used later by the evaluator. All released runs use OpenCode 1.18.25 with the same 100-turn cap, implementing the shared coding-agent setting described in § 4.1.

The environment brief is matched to the language of the request and explains both the required deliverable and the available development workflow. It asks the agent to “write the finished page to index.html,” requires a single HTML file, and states that “grading renders the page offline.” It also describes the inspection loop explicitly as “draft → screenshot → compare → fix → screenshot again.” To make this loop executable, the workspace provides shell and file-editing tools together with a Playwright-based shot.py helper. The helper renders the current HTML at widths of 1920, 768, and 390 pixels and returns image paths along with page height, console errors, and attempted external requests. These outputs allow the agent to inspect its rendered implementation and identify problems before revising the page.

Network access remains available during development, but it does not relax the offline submission requirement. Dependencies must be inlined in the final page so that its evaluation does not rely on external resources. Likewise, the availability of development tools does not grant access to benchmark data, hidden criteria, or original reference implementations. Such access is denied and audited. At completion, we retain one official artifact per model–task pair, together with a run record containing the enforced limits, stop reason, transport attempts, token usage, and artifact hashes. This record links the webpage evaluated downstream to the trajectory and resource use reported for that run.

## B.5 EVIDENCE COLLECTION AND DUAL-TRACK JUDGING

The agent’s inspection tools support development, but the evidence used for scoring is collected separately from the submitted index.html. This distinction ensures that the evaluator examines the final webpage through its own capture and measurement procedure rather than relying on the views the agent happened to inspect. The hidden keypoints guide this procedure by identifying the states and properties that need to be exposed. The resulting evidence is then passed to the programmatic and VLM judging tracks described in § 4.3.

Capture and measurement settings. Playwright/Chromium first renders the artifact at 1920×1080, with a target of five to ten distinct primary frames. Since requirements can concern states that are not visible on initial load, a deterministic action planner uses the keypoints to plan at most eight states to expose. For each item, it tries up to four candidate controls and allows 900 ms for a state transition. These settings govern how the browser attempts to obtain the interaction evidence specified during keypoint construction.

The evaluator also measures the page at widths of 1280, 768, and 390 pixels to assess behavior beyond the primary desktop view. Additional stress fixtures test robustness when content differs from the initial presentation: eligible collections are replaced with zero or 300 rows, an 80-character unbroken string is injected, and mixed CJK/emoji content is supplied. The resulting measurements complement the ordinary rendered views by exposing behavior under narrow layouts and abnormal content. If a measurement is unavailable or inapplicable, that status is recorded explicitly. The absence of a usable measurement is not treated as evidence that the corresponding test succeeded.

From evidence to judgments. For program-verifiable keypoints, the collected browser measurements determine pass/fail verdicts. Requirements that need semantic or perceptual interpretation are instead assigned to Gemini 3.1 pro, configured with high thinking level and a maximum output of 65,535 tokens. This judge receives visual and measured evidence rather than the submitted HTML, reference source code, or solver identity. Its task is to assess what the webpage demonstrates against the stated requirement, rather than infer success from implementation claims.

The perceptual judge processes batches of 5 keypoints, with a separate assessment for each item. Each batch contains the user request, the supporting basis for each item, the reference assignments, and normalized browser facts. To keep these facts and images interpretable, the images follow a fixed order: reference frames, up to ten main artifact views, up to two views per narrow viewport, and up to two stress-fixture views. An accompanying manifest identifies the groups, making clear which images describe the reference and which show the submitted page under each evaluation condition.

Before assigning a rating, the judge must record the expected requirement, the observed evidence, and the deviation between them. This sequence asks it to establish a concrete comparison before choosing a score. Fig. 14 presents these audit instructions and the rating rubric. The full template also specifies evidence handling and exceptions for unavailable external inputs. The resulting item scores are combined with core-function evidence using the aggregation rule in § 4.3. We retain render manifests, measurements, action outcomes, and item verdicts separately, allowing the recorded judgments to be inspected and the scores to be re-aggregated without another model call.

![](images/6eb5d43f12ee3be2ec90c8a0a26c10200071bd2de1e0552646a3a9de23ef2d05.jpg)  
Figure 9: An end-to-end development trajectory on Task 163. Boxes and enlarged screenshot crops highlight reference cues, responsive-layout revisions, and changes in the quote and booking state.

![](images/ea1161e39da06ea24f6baf685ed123fc6afb3ff945fd767a45330bcbaf7559c8.jpg)  
Figure 10: Prompt for user-query generation. Across all prompt figures, [...] denotes omitted passages. Formatting and line breaks are adjusted for presentation.

![](images/10471b6d5ee7100d5dd452dd0033f8a9bda7e77e7119607d52197699d60038d3.jpg)  
Figure 11: Instruction template for expert keypoint review. Reviewers check coverage, evidence grounding, reference-use consistency, and verifiability, then return revised keypoints with reasons for changes and unresolved issues.

![](images/cd3c90f3e0ee978f175603e3c44fbd3022d8ac5980e5415f740129599332f6dd.jpg)  
Figure 12: Keypoint-construction prompt excerpts.

![](images/11b1c2142fffa6aeb19255922d33c0194fef95f9d11cefa4d5148a845656bd02.jpg)  
Figure 13: Keypoint-review prompt excerpts. The global pass checks coverage, and the local pass reviews individual items against the reference-use specification and supporting evidence.

<table><tr><td colspan="2">Perceptual-judging prompt</td></tr><tr><td colspan="2">[...]</td></tr><tr><td colspan="2">For each item, determine the following three things in order and include them in the output: 1. expected (required value): Extract a specific, verifiable requirement from bas i s. Write a value,</td></tr><tr><td colspan="2">not a paraphrase. • √ &quot;5 cards&quot; / &quot;dominant color #1a2340 (from the reference screenshot header)&quot; / &quot;exact text: &#x27;Live</td></tr><tr><td colspan="2">leaderboard&#x27;&quot; • × &quot;Colors must match the reference&quot; (this restates the requirement rather than giving a value).</td></tr><tr><td colspan="2">2. observed (measured value): The actually measured value of the same dimension in the fact block/rendered frames. This must also be a value.</td></tr><tr><td colspan="2">• √ &quot;3 cards (see frame 04)&quot; / &quot;#4a90d9 (first entry in fact block colors.bg_by_area)&quot; / &quot;text reads &#x27;Live rankings&#x27;&quot;</td></tr><tr><td colspan="2">• × &quot;Implemented correctly&quot; / &quot;Meets the requirement&quot; (these are conclusions, not measurements). • If the value cannot be measured from any supplied material → write &quot;cannot confirm&quot; in observed.</td></tr><tr><td colspan="2">3. deviation (difference): The difference between expected and observed. This field must not be omitted. • Write &quot;none&quot; only when there is no difference at all. Writing &quot;none&quot; commits you to proving</td></tr><tr><td colspan="2">character-by-character/value-by-value equality. • Otherwise, specify the difference: &quot;count differs by 2 (3/5)&quot; / &quot;similar hue but brightness differs by about 40%&quot; 7 &quot;text has been reworded&quot;.</td></tr><tr><td colspan="2">[...] Derive the score from deviation (do not form an impression and then justify it)</td></tr><tr><td colspan="2">Do not reverse this order. Use a four-level 0-1-2-3 scale (poor / fair / good / excellent). A score of 3 is extremely difficult to earn: it requires perfect, indisputable evidence.</td></tr><tr><td colspan="2">Level / Score Meaning and corresponding deviation</td></tr><tr><td>Poor 0</td><td>Entirely missing, entirely inconsistent, clearly deviating (wrong count, reworded/summa- rized text, wrong position/hierarchy, clearly mismatched attributes)</td></tr><tr><td>Fair 1</td><td>Corresponding deviation: One or more clear deviations, or complete absence Correct direction with acceptable deviations (roughly correct but imprecise details, right color family but different values, slightly too few items, minor layout differences)</td></tr><tr><td>Good</td><td>Corresponding deviation: Quantifiable deviations, but recognizably the same overall design Essentially correct, with only tiny imperfections (color difference &lt;5%, spacing difference 1-2 px, almost indistinguishable)</td></tr><tr><td>Excellent</td><td>Corresponding deviation: Tiny or no deviation, but not verifiable value-by-value perfec- tion</td></tr><tr><td>3</td><td>Perfect match: expected and observed are equal character by character/value by value, with zero deviation and sufficient evidence Corresponding deviation: deviation is genuinely &quot;none&quot;, and both values can be checked</td></tr><tr><td>[...]</td><td>character by character</td></tr></table>

Figure 14: Perceptual-judging prompt excerpts. The judge must record expected and observed values and their deviation before applying the four-level rubric.