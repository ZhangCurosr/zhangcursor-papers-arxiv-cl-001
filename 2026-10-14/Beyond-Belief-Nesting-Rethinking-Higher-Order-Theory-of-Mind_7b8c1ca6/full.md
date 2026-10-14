# Beyond Belief Nesting: Rethinking Higher-Order Theory of Mind in Large Language Models

Shanshan You<sup>1,2</sup>, Zhengqi Wang<sup>1,2</sup>, Hao Liu<sup>3</sup>, Yuqin Liu<sup>1,2</sup>, Ya He<sup>1,2</sup>, Xin Hao<sup>1,2</sup>, Liyuan Wang<sup>1,2</sup>\*

<sup>1</sup> Department of Psychological and Cognitive Sciences, Tsinghua University

<sup>2</sup> IDG/McGovern Institute for Brain Research, Tsinghua University

<sup>3</sup> Department of Cognitive Sciences, University of California, Irvine yss26@mails.tsinghua.edu.cn, liyuanwang@tsinghua.edu.cn

## Abstract

Theory of mind (ToM), the ability to reason about others’ mental states, underlies effective communication and coordination. Its higherorder form enables reasoning over nested perspectives in strategic interactions. However, existing evaluations often equate belief-nesting depth with reasoning order, allowing nominally higher-order questions to admit lower-order shortcuts. We formalize effective order through the distinct events supporting successive belief prefixes, revealing widespread order inflation across existing benchmarks. We introduce HOPTOM, with certified full-order questions and controls for lower-order alternatives. Under direct prompting, mean accuracy across four frontier LLMs falls from 73.5% at first order to 20.0% at fourth order, and human participants also struggle. Causal interventions link a recurring class of errors to evidence for shallower belief prefixes. We introduce HOP-TRACE, a plug-and-play inference-time method that encodes each interaction once into perspective records and resolves nested queries deterministically. On certified narrative questions, it improves fourth- and fifth-order accuracy by 22.0 and 38.5 percentage points over strong prompting baselines, while using 20–39 times fewer tokens than multi-call methods. These results sharpen how higher-order ToM is evaluated and offer a practical way to improve perspective reasoning in interactive agents.

## 1 Introduction

Theory of mind (ToM) allows intelligence to reason about others’ beliefs, knowledge, and intentions, supporting effective communication and interaction (Premack and Woodruff, 1978; Wimmer and Perner, 1983). Higher-order ToM extends this capacity to recursively nested mental states: what one agent believes another knows or believes (Perner and Wimmer, 1985). Such reasoning becomes crucial when agents coordinate with multiple partners, information is shared selectively, and their views of one another diverge. As large language models (LLMs) increasingly participate in multi-party interactions, evaluating their ability to track these nested perspectives has become an important test of how well they can reason about others. However, higher-order ToM is commonly evaluated by counting the belief operators in a question (Kinderman et al., 1998; Wu et al., 2023; Street et al., 2026). Consider asking, Where does Amy think Ben thinks Carol thinks Dan thinks the cake is? (Fig. 1). If all four witness the cake’s last move and know that the others witnessed it, the answer follows from a single shared event. The question names four belief layers, but a respondent can answer it without distinguishing evidence for those layers. This creates a critical problem: nested wording alone does not establish the perspective reasoning a question tests. Analyses of human recursive mindreading likewise show that partial-chain strategies can inflate performance estimates (Wilson et al., 2023).

![](images/0a168861977be6b0df63ca84d08ebf1f087231e8342e1727cde503412dfd190d.jpg)  
Figure 1: Nested beliefs behind a fourth-order question.

We make this distinction precise by representing a story as belief-changing events indexed by the perspectives to which they are available. For each successive prefix of a nested query, we identify the event that settles its belief and define the query’s effective order by the number of distinct settling events. This shifts the assessment of order from the wording of a question to the information structure needed to resolve its nested perspectives. Auditing representative ToM benchmarks reveals widespread order inflation: many questions labeled as higher-order have substantially lower effective orders. Consequently, their scores alone cannot establish the depth of perspective reasoning achieved by either humans or language models.

![](images/6f9e1dacede32047e7b8231158f2a7e461f688b838426d820a7c6c15bd1c2427.jpg)  
Figure 2: Belief nesting can overstate effective order. Left: The same fourth-order query has r = 1 in A and r = 4 in B. Colors mark the event supporting each prefix. Right: HOPTRACE encodes perspective records and resolves each prefix independently. Sequential backward lookup instead selects t = 2 and incorrectly answers “crate”.

To provide a more discriminating evaluation, we introduce HOPTOM (Higher-order perspective Theory of Mind). We design the perspective structure and queries before writing each story, allowing effective order to be verified and lower-order alternatives to be examined through explicit controls. On the resulting certified questions, mean accuracy across four LLMs under direct prompting falls from 73.5% at first order to 20.0% at fourth. Human participants also find the certified questions difficult. Moreover, causal analyses reveal a recurring failure mode: when answering a nested question incorrectly, models often retrieve evidence supporting a shallower belief chain. These findings expose a gap between recognizing the agents mentioned in a question and maintaining the perspective relations needed to answer it.

We address this gap with HOPTRACE, a plugand-play inference-time method that encodes an interaction once into ordered perspective records and resolves nested queries with a deterministic solver. The records preserve directed awareness, and the solver searches independently for the event supporting each belief prefix. One encoding serves multiple queries and yields an inspectable trace for each answer. On certified narrative questions, HOPTRACE improves fourth- and fifth-order accuracy by 22.0 and 38.5 percentage points over strong prompting baselines. It uses one LLM call per interaction and 20–39 times fewer tokens than multi-call methods. Ablation studies validate the efficacy of ordered perspectives, independent prefix search, and program execution.

Our contributions include: (1) an eventperspective formalization of effective order and an audit of order inflation in existing benchmarks; (2) a certified benchmark that verifies the intended perspective structure and includes controlled lowerorder alternatives; (3) a plug-and-play inferencetime method that combines reusable perspective encoding with deterministic query resolution; and (4) behavioral and causal analyses that characterize higher-order ToM performance in LLMs and explain a recurring source of model errors.

## 2 Related Work

Theory of Mind (ToM). Psychological studies commonly assess higher-order ToM with nestedbelief questions and find that performance declines as nesting deepens (Kinderman et al., 1998; O’Grady et al., 2015). However, Wilson et al. (2023) showed that some deeply nested items admit partial-chain solutions, with performance dropping sharply when those shortcuts are blocked. Epistemic logic formally represents nested beliefs and changes in agents’ information (Fagin et al., 1995; Baltag et al., 1998). These lines of work highlight a distinction between the depth of a question and the belief layers its story makes distinguishable. Our effective-order formulation measures this distinction at the item level through the events supporting successive belief prefixes.

ToM in Language Models. LLM evaluations have expanded from classic false-belief tasks to dialogues and deeper belief questions (Kosinski, 2024; Kim et al., 2023; Wu et al., 2023; Street et al., 2026). Existing efforts have likewise used perspective filtering, temporal belief tracking, and recursive prompting to improve ToM performance (Wilf et al., 2024; Hou et al., 2024; Lei et al., 2026), while mechanistic work has identified retrieval operations involved in belief tracking (Prakash et al., 2026). Yet nesting-depth labels do not reveal how many distinct events support a query’s belief layers, leaving the source of success on higher-order questions unclear. We use effective order to audit this gap, design controlled evaluations, and guide inference over explicit perspective records.

## 3 True Higher-Order ToM

Belief-nesting depth can misclassify shortcutsolvable questions as higher-order. We define $e f -$ fective order from the distinct events supporting successive belief prefixes, giving a structural criterion for genuinely higher-order questions, and then perform an in-depth empirical analysis.

## 3.1 Problem Formulation

Interactions and queries. Consider human or artificial agents interacting over time. Their observations and communications may change both their beliefs about a target variable $x ,$ such as an object’s location, and their beliefs about one another’s beliefs. A higher-order ToM query is a belief chain $q = [ q _ { 1 } , \ldots , q _ { k } ]$ ], read from the outside in. For example, $[ a , c , b ]$ asks what a believes c believes b believes about x. Its syntactic order is $k = | q |$

Perspective records. We encode the interaction as time-ordered records $E _ { t } ~ = ~ V _ { t } [ \pi _ { t } ; \Delta _ { t } ] ( m _ { t } )$ Each record carries a value $m _ { t }$ of $x .$ , which may be initially held, observed, or communicated and need not be correct. The structure $\pi _ { t }$ specifies agents’ access to this value and their beliefs about others access. The set $\Delta _ { t }$ lists agents whose own belief becomes $m _ { t }$ and may be empty when only beliefs about others change. We write $\pi _ { t }$ as a chain of groups $G _ { 1 } \to \cdots \to G _ { \ell } .$ . Access is common knowledge within a group, while an arrow denotes onesided observation of the next group’s access. A pseudo-observer ${ \sim } c$ is an agent whom others incorrectly believe to have received $m _ { t }$ . These records cover observation, communication, deception, and misperception. Initial states are encoded to support every queried prefix, ensuring that a latest supporting record exists. Beliefs persist until updated.

<table><tr><td>Benchmark</td><td>N</td><td> $k _ { 0 }$ </td><td>k Mean r Full order</td></tr><tr><td>ToMi</td><td>1,998 2</td><td>2 1.36</td><td>36.2%</td></tr><tr><td>ExploreToM</td><td>3,298 2</td><td>2 1.07</td><td>6.6%</td></tr><tr><td>Hi-ToM</td><td>120</td><td>4 4 2.07</td><td>2.5%</td></tr><tr><td>HoPToM (Ours)</td><td>200 0 5 5</td><td>5.00</td><td>100%</td></tr></table>

Table 1: Effective order at each benchmark’s highest published level. N is the number of questions, k is the belief-chain length. Full order is the share with $r = k$

Effective order. Let $p _ { j } = [ q _ { 1 } , \dotsc , q _ { j } ]$ be the j-th prefix of $q .$ A record supports $p _ { 1 }$ if $q _ { 1 } \in \Delta _ { t }$ . For $j \geq 2$ , it supports $p _ { j }$ if $\pi _ { t }$ covers the chain. This requires $q _ { 1 }$ to be an actual observer in the outermost group. Each successive agent must be represented, actually or presumptively, in the same group or the next group along a directed observation link. For example, $\{ a , b \} \to c$ covers $[ a , c ]$ and $[ a , b , c ]$ , but not [c, a]. We resolve each prefix independently against the full interaction history. Let $s _ { j }$ be the index of the latest record supporting $p _ { j }$ . That record determines the value of x attributed under $p _ { j }$ . The record indexed by $s _ { k }$ determines the answer to the full query. The effective order is

$$
r ( q , x ) = \left| \left\{ s _ { 1 } , \ldots , s _ { k } \right\} \right| \leq k .\tag{1}
$$

A query has full effective order if $r ( q , x ) = k _ { \mathrm { : } }$ meaning that its successive prefixes have distinct latest supporting events. This rules out sharedevent collapse. Coinciding answers to lower-order queries are assessed separately.

## 3.2 Rethinking Previous ToM Benchmarks

We audit questions at the highest published level in three existing benchmarks: ToMi (Le et al., 2019), ExploreToM (Sclar et al., 2025), and Hi-ToM (Wu et al., 2023). For these benchmarks, we derive perspective records from their generators’ action logs. The audit is independent of released answer labels. Appendix A reports encoding checks, label inconsistencies, and results at other orders.

The published higher-order labels substantially overstate the order tested (Table 1). At the highest level of each benchmark, only 36.2% of ToMi questions, 6.6% of ExploreToM questions, and 2.5% of Hi-ToM questions have full effective order. Hi-ToM shows how this inflation grows with nesting depth (Table 2). From second to fourth order, mean effective order rises only from 1.40 to 2.07, while the full-order share falls from 40.0% to 2.5%. At fourth order, 89 of 120 questions have $r \leq 2 .$ and only 6.7% of those with claim events reach full order. An additional shortcut arises from answer coincidence: the correct answer matches the answer to a shorter prefix or the target’s actual final location in 66.7% of second-order and 85.0% of fourth-order questions.

<table><tr><td>Subset</td><td></td><td></td><td></td><td>k Mean r Full order State match</td><td>r counts</td></tr><tr><td rowspan="4">All</td><td>1</td><td>1.00</td><td>100%</td><td>50.0%</td><td>120</td></tr><tr><td>2</td><td>1.40</td><td>40.0%</td><td>66.7%</td><td>72 /48</td></tr><tr><td>3</td><td>1.68</td><td>9.2%</td><td>79.2%</td><td>50 / 59 / 11</td></tr><tr><td>4</td><td>2.07</td><td>2.5%</td><td>85.0%</td><td>26 / 63 / 28 / 3</td></tr><tr><td rowspan="3">w/o Claims 3</td><td>2</td><td>1.29</td><td>29.3%</td><td>73.3%</td><td>53 / 22</td></tr><tr><td></td><td>1.47</td><td>2.7%</td><td>86.7%</td><td>42 / 31 / 2</td></tr><tr><td>4</td><td>1.84</td><td>0.0%</td><td>89.3%</td><td>24 / 39 / 12 / 0</td></tr><tr><td rowspan="3">w/ Claims</td><td>2</td><td>1.58</td><td>57.8%</td><td>55.6%</td><td>19 /26</td></tr><tr><td>3</td><td>2.02</td><td>20.0%</td><td>66.7%</td><td>8/28 / 9</td></tr><tr><td>4</td><td>2.44</td><td>6.7%</td><td>77.8%</td><td> $2 / 2 4 / 1 6 / 3$ </td></tr></table>

Table 2: Hi-ToM audit by belief-chain length. Each order contains 120 unique questions, including 75 without and 45 with claim events. Full order is the share with $r = k$ . State match is the share whose answer equals the answer to a shorter prefix or the target’s actual final location. The last column gives counts for $r = 1 , \ldots , k .$

Three patterns produce structural collapse. In subsumed perspectives, an added agent shares the event supporting an outer layer (Fig. 2, left). In idle gaps, agents have different observation histories, but the intervening events do not change x. In reported mental states, a single claim or explicit statement settles multiple prefixes at once. Subsumed perspectives, idle gaps, and reported mental states account for 54%, 33%, and 13% of collapsed layers in Hi-ToM, respectively. The audited benchmarks are therefore dominated by pseudo-higher-order items: their questions contain nested beliefs, but their interactions do not require a distinct supporting event for each layer. Establishing higher-order ToM requires certifying these perspective distinctions and controlling shortcuts in the answers.

## 3.3 HOPTOM: A Certified Benchmark for Higher-Order ToM

Perspective-first construction. The audit motivates specifying the perspective structure and queries before writing their natural-language realizations. For each query, we identify the record supporting every belief prefix and certify full effective order when $r = k$ . In our certified items, these supporting records also occur in decreasing time order from the outermost to the innermost prefix. This temporal constraint is part of the benchmark design, not the definition of effective order. We vary the positions of supporting events and interleave other updates to reduce the regularity of simple backward-stepping heuristics.

<table><tr><td>Model</td><td>1st</td><td>2nd</td><td>3rd</td><td>4th</td><td>5th</td><td>All</td></tr><tr><td>GPT-6 Sol</td><td>92.0</td><td>62.0</td><td>58.5</td><td>32.5</td><td>36.0</td><td>56.2</td></tr><tr><td>Claude Opus 5.5</td><td>69.5</td><td>33.0</td><td>42.5</td><td>16.0</td><td>21.5</td><td>36.5</td></tr><tr><td>Gemini 3Flash</td><td>72.5</td><td>34.5</td><td>41.0</td><td>17.0</td><td>16.0</td><td>36.2</td></tr><tr><td>DeepSeek-V4-Pro</td><td>60.0</td><td>49.0</td><td>32.5</td><td>14.5</td><td>14.0</td><td>34.0</td></tr><tr><td>Average</td><td>73.5</td><td>44.6</td><td>43.6</td><td>20.0</td><td>21.9</td><td>40.7</td></tr></table>

Table 3: Direct-prompting accuracy (%) on the 1,000 certified full-order questions of HOPTOM. Each order contains 200 questions.

Scenarios and questions. We design 10 prototypes spanning 10 scenario types in six families, including asymmetric observation, deception, misperception, and delayed information (Table 6). Each prototype yields 10 variants that preserve its perspective structure while changing names, settings, objects, and wording. Each variant is realized as both a narrative and a multi-party dialogue. The resulting benchmark contains 200 story realizations of 100 underlying interactions.

Each realization has five certified questions, one at each syntactic order from one to five, and four controls targeting collapsed belief layers or coinciding lower-order answers. This yields 1,000 certified questions and 800 controls. Questions use open-ended answers to avoid reliance on candidate options (Balepur et al., 2024; Zheng et al., 2024). We also provide a multiple-choice version (Appendix E.3). All realizations were checked for fluency, and each certified question was manually verified against its supporting-record trace.

## 3.4 Empirical Analysis

We evaluate four frontier LLMs under direct prompting and two open-weight models, Gemma 4 31B and Qwen3-32B. A complementary human pilot uses three stories that participants could not reread while answering (Appendix C).

Certified questions become difficult at higher orders. Across the four frontier LLMs, mean accuracy falls from 73.5% at first order to 20.0% at fourth order (Table 3). GPT-6 Sol drops from 92.0% to 32.5%, and the other three score 14.5– 17.0% at fourth order. Fifth-order accuracy stays low, averaging 21.9%. Strong performance on simpler belief questions therefore does not carry over to certified fourth- and fifth-order questions under direct prompting. Human participants also strug-<sup></sup>gle, with 33.3% accuracy on certified questions overall and about 9% at orders four and five combined (Fig. 3a). The slight rise from fourth to fifth order for GPT-6 Sol and Claude Opus 5.5 may reflect how certified items are built. Here, supporting events run backward in time across prefixes, so the deepest layer falls on one of the story’s earliest belief changes. A model that names an early location may then answer correctly.

<sup></sup>b.  
![](images/d31c75994598d6217761a9943d6a06359ee0260667cfbe6b410389452e9f4bf7.jpg)

![](images/4a697d02c3371ff03602a8271e2e24a1eb5ca383c12d6a92131a28010e6df348.jpg)  
<sup>0")</sup>Figure 3: Results on certified questions and controls. (a) <sup>			</sup>Fourth- and fifth-order results for human participants, <sup>	</sup>	<sup>	</sup> who could not reread the stories, and four frontier LLMs. <sup></sup> <sup></sup> 	 <sup></sup> <sup></sup>  <sup>  </sup>(b) Results by syntactic order for open-weight models.

Controls reveal when lower-order alternatives help. At fourth order, human accuracy is 41.7% on controls but only 6.7% on certified questions (Fig. 3a). GPT-6 Sol shows the same direction of contrast, though less strongly. The other frontier LLMs perform poorly in both conditions. Gemma 4 31B and Qwen3-32B show clear control advantages at orders two through four (Fig. 3b). At fifth order, human accuracy is low in both conditions, and the control advantage narrows or reverses for the weaker models. These patterns show that lowerorder alternatives can substantially inflate performance when respondents can exploit them. When both conditions are difficult, their accuracy gap becomes less informative about the underlying difference in perspective demands.

## 4 Method

Full-order questions require tracking distinct perspective updates and selecting the evidence that supports each belief prefix. We introduce HOP-TRACE, a plug-and-play inference-time method that makes these lookups explicit while reusing one LLM encoding across multiple queries.

Overview. Let S denote an account of an interaction, such as a narrative, dialogue, or event log. Our formulation associates each prefix of a belief-chain query with its latest supporting record (Sec. 3.1). Interpreting the same interaction afresh for every query repeats the work of tracking agents’ perspectives, while chaining partial answers can select the wrong records. HOPTRACE separates perspective <sup></sup> <sup>	</sup>encoding from query resolution (Fig. 2, right). An <sup>	</sup>LLM encodes $S$ once, and a deterministic solver answers queries from the resulting records:

$$
\mathcal { V } = \operatorname { E n c } ( S ) = \langle E _ { 1 } , . . . , E _ { n } \rangle ,\tag{2}
$$

$$
( \hat { y } , \tau ) = \mathrm { S o l v e } ( \mathcal { V } , q , x ) , \quad \hat { y } = m _ { s _ { k } } .\tag{3}
$$

Here $E _ { t } = V _ { t } [ \pi _ { t } ; \Delta _ { t } ] ( m _ { t } )$ is a perspective record tagged with its target variable, which is omitted from the notation when x is fixed. The trace $\tau =$ $\langle ( p _ { j } , E _ { s _ { j } } ) \rangle _ { j = 1 } ^ { k }$ identifies the supporting record for every prefix.

D1: Ordered perspectives. Who observes an event and who knows about that observation can lead to different nested beliefs. The encoder therefore preserves joint access, one-sided awareness, and presumed observation in the ordered perspective structure $\pi _ { t }$ . It records changes to agents’ own beliefs in $\Delta _ { t }$ . A flat set of agents cannot represent these distinctions.

D2: Independent per-prefix search. The record supporting a prefix depends on that prefix’s perspective, not selected for the preceding prefix. Stepping backward through individual agents’ first-order beliefs can therefore select the wrong evidence, even when the correct supporting records are ordered in time. The solver instead searches the full record sequence independently for each $p _ { j }$

D3: Program execution. Applying perspective coverage and temporal selection across several prefixes can itself be a source of model error. We implement these rules as the deterministic procedure in Algorithm 1. The solver uses only the encoded records and returns an inspectable trace with each answer. It is exact relative to those records under the specified support rule. Encoding requires one LLM call per S, after which each query takes at most k backward scans of the relevant records.

## 5 Experiments

In this section, we first describe the experimental setup and then analyze main results.

Algorithm 1 Deterministic query resolution   
Require: Time-ordered records $\mathcal { V } = \langle E _ { 1 } , \ldots , E _ { n } \rangle$ , query   
$q = [ q _ { 1 } , \ldots , q _ { k } ] ,$ , target x   
1: $\bar { \nu } _ { x } $ records tagged with $x ,$ retaining their indices in $\nu$   
2: $\tau  [ ]$   
3: for $j = 1$ to k do   
4: $p _ { j } \gets [ q _ { 1 } , \dots , q _ { j } ]$   
5: for $E _ { t }$ in $\nu _ { x }$ from latest to earliest do   
6: if $E _ { t }$ supports $p _ { j }$ then   
7: $s _ { j } \gets t$   
8: Append $( p _ { j } , E _ { t } )$ to τ   
9: break   
10: end if   
11: end for   
12: end for   
13: $\hat { y } \gets m _ { s _ { k } }$ ▷ value carried by $E _ { s _ { k } }$   
14: return $( \ddot { \hat { y } } , \tau )$

## 5.1 Experimental Setup

Evaluation benchmarks. HOPTOM contains 100 underlying interactions, each realized as a narrative and a dialogue, for 200 texts and 1,800 question instances. The primary method comparison uses the 100 narratives and their 500 certified fullorder questions, with 100 questions at each order. Their 400 controls are used in the ablation analysis. The evaluation in Sec. 3.4 uses both language formats. We also evaluate on Hi-ToM (Wu et al., 2023), which has 120 stories and 600 unique questions spanning orders zero to four. Each question is released in two prompt formats. Dataset handling is detailed in Appendix A.

Models and baselines. We evaluate GPT-6 Sol (OpenAI, 2026), Claude Opus 5.5 (Anthropic, 2026), Gemini 3 Flash (Google DeepMind, 2026), and DeepSeek-V4-Pro (DeepSeek-AI et al., 2026). Gemma 4 31B (Team et al., 2026) and Qwen3- 32B (Yang et al., 2025) serve as open-weight encoders. We compare HOPTRACE with direct answering, chain-of-thought prompting (CoT) (Wei et al., 2022), TimeToM (Hou et al., 2024), and RecToM (Lei et al., 2026). Comparisons within each model use the same underlying LLM. Prompt adaptations and execution settings are described in Appendix E.2.

## 5.2 Experimental Results

Overall performance. On the 500 certified narrative questions, HOPTRACE reaches 76.6% mean accuracy across four LLMs (Table 4). CoT, Time-ToM, and RecToM reach 55.2%, 55.3%, and 54.9%, respectively, giving HOPTRACE an advantage of at least 21.3 percentage points. The difference is largest at higher orders. At fourth and fifth order, HOPTRACE reaches 70.3% and 81.0%, compared with 48.3% and 42.5% for the strongest baseline at each order. Our method also improves open-weight models on all 1,800 questions. Gemma 4 31B rises from 33.2% under direct prompting to 67.7% with HOPTRACE, while Qwen3-32B rises from 25.3% to 38.7%. Qwen’s smaller gain coincides with encodings that often omit directed or presumed observations. Since the solver is unchanged across encoders, the quality of the perspective records remains consequential.

<table><tr><td></td><td></td><td colspan="6">HoPToM (True Higher-Order)</td><td>Hi-ToM</td></tr><tr><td></td><td>Method</td><td>1st</td><td>2nd</td><td>3rd</td><td>4th</td><td>5th</td><td>All</td><td>All</td></tr><tr><td rowspan="4">GPT-OI[</td><td>CoT</td><td>94</td><td>58</td><td>52</td><td>42</td><td>44</td><td>58.0</td><td>80.3</td></tr><tr><td>TimeToM</td><td>78</td><td>83</td><td>56</td><td>58</td><td>50</td><td>65.0</td><td>78.5</td></tr><tr><td>RecToM</td><td>85</td><td>74</td><td>52</td><td>58</td><td>36</td><td>61.0</td><td>79.2</td></tr><tr><td>HOPTRACE</td><td>89</td><td>70</td><td>67</td><td>76</td><td>90</td><td>78.4</td><td>82.4</td></tr><tr><td rowspan="4">Depek</td><td>CoT</td><td>76</td><td>68</td><td>35</td><td>23</td><td>14</td><td>43.2</td><td>62.5</td></tr><tr><td>TimeToM</td><td>55</td><td>48</td><td>44</td><td>53</td><td>41</td><td>48.2</td><td>65.8</td></tr><tr><td>RecToM</td><td>62</td><td>49</td><td>43</td><td>47</td><td>38</td><td>47.8</td><td>61.3</td></tr><tr><td>HOPTRACE</td><td>94</td><td>69</td><td>71</td><td>70</td><td>81</td><td>77.0</td><td>87.6</td></tr><tr><td rowspan="4">Gem-ash</td><td>CoT</td><td>98</td><td>63</td><td>56</td><td>43</td><td>40</td><td>60.0</td><td>70.6</td></tr><tr><td>TimeToM</td><td>79</td><td>64</td><td>38</td><td>51</td><td>39</td><td>54.2</td><td>45.6</td></tr><tr><td>RecToM</td><td>74</td><td>61</td><td>35</td><td>16</td><td>42</td><td>45.6</td><td>67.5</td></tr><tr><td>HOPTRACE</td><td>91</td><td>64</td><td>68</td><td>73</td><td>74</td><td>74.0</td><td>87.6</td></tr><tr><td rowspan="4">CIuu-ppus</td><td>CoT</td><td>93</td><td>59</td><td>56</td><td>49</td><td>41</td><td>59.6</td><td>79.8</td></tr><tr><td>TimeToM</td><td>78</td><td>75</td><td>45</td><td>31</td><td>40</td><td>53.8</td><td>80.0</td></tr><tr><td>RecToM</td><td>77</td><td>75</td><td>66</td><td>59</td><td>49</td><td>65.2</td><td>79.8</td></tr><tr><td>HOPTRACE</td><td>93</td><td>65</td><td>85</td><td>62</td><td>79</td><td>76.8</td><td>81.9</td></tr><tr><td rowspan="4">Average</td><td>CoT</td><td>90.3</td><td></td><td></td><td></td><td></td><td></td><td>73.3</td></tr><tr><td>TimeToM</td><td>72.5</td><td>62.0 67.5</td><td>49.8 45.8</td><td>39.3 48.3</td><td>34.8 42.5</td><td>55.2 55.3</td><td>67.5</td></tr><tr><td>RecToM</td><td>74.5</td><td>64.8</td><td>49.0</td><td>45.0</td><td>41.3</td><td>54.9</td><td>72.0</td></tr><tr><td>HOPTRACE</td><td>91.8 67.0</td><td></td><td>72.8</td><td>70.3</td><td>81.0</td><td>76.6</td><td>84.9</td></tr></table>

Table 4: Accuracy (%) on HOPTOM and Hi-ToM. Models: GPT-6 Sol, DeepSeek-V4-Pro, Gemini 3 Flash, and Claude Opus 5.5. Best per model and column in bold.

Efficiency. HOPTRACE uses one LLM call and 4,235 tokens per story on average while answering all nine questions (Fig. 4a). TimeToM uses 34.1 calls and 86,259 tokens per story, and RecToM uses 32.4 calls and 164,641 tokens, each for five certified questions. Even if the full encoding cost of HOPTRACE is assigned to those five questions, its token use is 20.4× and 38.9× lower, respectively. Wall-clock measurements show the same pattern, with 4–217 s per story for HOPTRACE and 268–1,258 s for the two recursive baselines across models (Appendix D).

Reasoning models. Extended thinking strengthens direct answering, providing a more demanding test of HOPTRACE. With thinking enabled for both direct answering and perspective encoding, HOP-TRACE raises mean accuracy from 60.5% to 83.2% at fourth order and from 57.9% to 85.8% at fifth order on HOPTOM (Fig. 5). All four models improve at both orders. The gains are most pronounced on the higher-order questions that HOPTRACE targets, showing that explicit perspective records and deterministic prefix resolution remain valuable even for reasoning-enabled models. As one encoding serves multiple queries, these gains do not require a separate thinking-enabled LLM call for each question.

![](images/3d57e96953a691c126df3e9a149fa43353c0631dab08740e2fb30bafb33ead1e.jpg)

![](images/7410fb9307742aed212926850a493ca35e3964b28b716c8052ba22a6dce16f0d.jpg)  
Figure 4: Accuracy, inference cost, and ablations on the narrative part of HOPTOM. (a) Accuracy on 500 certified full-order questions against tokens per story. HOPTRACE answers all nine questions per story with one encoding call, <sub>0,.3</sub> <sup>+\$/0\$#/\$</sup> <sup>."& \$2\$"10',+ \*\$0&,#</sup>while TimeToM and RecToM answer the five certified questions. (b) Accuracy of the full method and its ablations on all 900 narrative questions. Hatching denotes individual models and dashed lines denote model averages.

![](images/19d733b3f4e3009a9a1b096d8d1b3cb18027b08f0dbd59f9b2e9ac0238341ce0.jpg)  
Figure 5: Fourth- and fifth-order accuracy on the certified questions of HOPTRACE with reasoning enabled.

Transfer to Hi-ToM. HOPTRACE also achieves the highest accuracy for every model on Hi-ToM, ranging from 81.9% to 87.6% (Table 4). This shows that the encoder–solver pipeline applies to a separately constructed benchmark with different story conventions and a multiple-choice answer format. The result has a narrower interpretation for higher-order ability: as the audit in Sec. 3.2 shows, only 2.5% of Hi-ToM’s nominally fourthorder questions have full effective order. Aggregate Hi-ToM accuracy therefore provides limited evidence about performance on full-order questions.

Ablation study. The three ablations correspond to D1–D3 in Sec. 4 (Fig. 4b). Flattening the ordered perspectives in D1 reduces average accuracy by 23.5 points. The loss is consistent across models, showing the value of distinguishing joint access from directed and presumed observation. Replacing independent prefix search with time-nested first-order lookup in D2 causes the largest drop, 39.9 points. Chronology alone does not identify which event supports a nested perspective. In D3, LLM execution instead of program execution costs 20.4 points on average. GPT-6 Sol and Claude Opus 5.5 remain close to the full method, while DeepSeek-V4-Pro and Gemini 3 Flash fall to 29.2% and 34.4%. Deterministic execution thus makes the encoded records substantially more useful across models. On the 90 prototype questions, replacing model-generated records with gold records raises accuracy from 73.1% to 100%. The solver can resolve these questions from correctly specified records, leaving perspective encoding as the source of the observed gap on this subset.

Mechanism. We use attention interventions and activation patching to examine direct answers from Gemma 4 31B and Qwen3-32B (Appendix F). Many wrong answers repeat the answer to a shallower prefix of the query, including 75% for Gemma and 50% for Qwen3 at fourth order (Fig. 6a). At second order, most errors give the innermost agent’s own belief. For correct answers, blocking the paragraph containing the full query’s supporting event reduces answer log-probability by 12–17 nats, while blocking intermediate-prefix paragraphs has much smaller effects (Fig. 6b). Shallower-prefix errors reverse this pattern: that prefix’s paragraph has the largest effect in 81% and 90% of cases, and the full query’s paragraph has little effect. These errors come from evidence selected for the wrong perspective.

![](images/eb14c935e08bc55c4885062673b68827f7646c2c747e05d47149047beaf56776.jpg)  
b.

![](images/55a7932a83047eaba56cb0451f28412de72a32a2ae338a59b66749510d5e6e71.jpg)  
Figure 6: Direct answers. (a) Wrong answers by order k (%). Shallower prefix: answer to a shorter prefix. Innermost belief : the last-named agent’s belief. (b) Logprobability drop when supporting paragraph is hidden.

With HOPTRACE, the model only encodes the story into perspective records, and the solver resolves the query prefix by prefix. On shortcutresistant certified questions, accuracy rises from about 10% under direct answering to 58% with Gemma and 31% with Qwen3 (Fig. 7a). Encoding is a local task: about half of the records draw on a single story paragraph (54% for Gemma and 46% for Qwen3; Fig. 7b). Patching shows that each agent’s information moves to the final token directly from its name, not through the other agents (Fig. 9). For correct answers, the record selected by the solver draws on the full query’s supporting paragraph in 89% of cases for both models. Direct answering thus tends to stop short when tracing the chain back, whereas HOPTRACE leaves this backtracking to the solver and the model to the local encoding it handles well.

## 6 Conclusion and Discussion

Higher-order ToM cannot be assessed reliably from the nesting depth of a question alone. We formalize the perspective distinctions created by an interaction through effective order, reveal their frequent collapse in existing benchmarks, and construct a controlled evaluation that exposes substantial difficulty for both LLMs and humans. We also show that explicit perspective records, combined with deterministic query resolution, markedly improve model performance. Together, these findings provide a clearer basis for measuring higher-order ToM and a practical direction for strengthening perspective reasoning as interactions unfold.

a.  
![](images/5521007be8ed9c816397b7e1144bc2aead780d39598e17ee2336633fa40f6a21.jpg)

b.  
![](images/1d67b2bba6a1df83d2dcd4b050f09327fe496d8ec8f9fc7481af124babe64409.jpg)  
Figure 7: HOPTRACE vs. direct answering. (a) Accuracy on shortcut-resistant certified questions. (b) Paragraphs each record depends on.

Perspective tracking as a bottleneck. Models often retrieve evidence appropriate for a shallower prefix, whereas the solver answers all prototype questions given gold records. This points to perspective encoding and evidence selection as major constraints in the evaluated systems. Human recursive mindreading is also sensitive to task demands (Stiller and Dunbar, 2007; Wilson et al., 2023). Whether explicit records help human participants remains an open question.

Writing higher-order questions. Effective order can be checked before an interaction is written in natural language. Full effective order excludes shared-event collapse, but neither rules out coinciding answers nor establishes the minimum computation a respondent uses. Future benchmarks should report effective alongside syntactic order, include lower-order answer controls, and vary event timing to limit positional shortcuts.

Dividing the labor. One perspective encoding serves multiple queries, while program execution produces an inspectable supporting trace. This division suggests a way to reason over longer interactions, where agents must continually update what they believe others have seen, learned, or communicated. Improving the fidelity of those records remains essential.

## Limitations

Our formalism focuses on discrete beliefs about specified target variables, leaving uncertainty, graded trust, and beliefs inferred from indirect cues for future extensions. HOPTOM uses controlled written interactions, which could be extended to longer exchanges and multimodal observations. Our mechanistic interventions examine two openweight models, and similar analyses across other model families would broaden their scope.

## Acknowledgments

This work was supported by the NSFC Project (No. T2622023), the Beijing Nova Program (No. 202604841279), and the Beijing Major Science and Technology Project (No. Z251100008425003).

## References

Anthropic. 2026. System card: Claude Opus 5.5. System card. Accessed October 2026.

Akshatha Arodi and Jackie Chi Kit Cheung. 2021. Textual time travel: A temporally informed approach to theory of mind. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4162–4172, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Nishant Balepur, Abhilasha Ravichander, and Rachel Rudinger. 2024. Artifacts or abduction: How do LLMs answer multiple-choice questions without the question? In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10308–10330, Bangkok, Thailand. Association for Computational Linguistics.

Alexandru Baltag, Lawrence S. Moss, and Slawomir Solecki. 1998. The logic of public announcements, common knowledge, and private suspicions. In Proceedings of the 7th Conference on Theoretical Aspects of Rationality and Knowledge, TARK ’98, page 43–56, San Francisco, CA, USA. Morgan Kaufmann Publishers Inc.

Zhuang Chen, Jincenzi Wu, Jinfeng Zhou, Bosi Wen, Guanqun Bi, Gongyao Jiang, Yaru Cao, Mengting Hu, Yunghwei Lai, Zexuan Xiong, and Minlie Huang. 2024. ToMBench: Benchmarking theory of mind in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15959–15983, Bangkok, Thailand. Association for Computational Linguistics.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang,

Chaofan Lin, Chen Dong, Chenchen Ling, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyu Hou, Chenhao Xu, Chenze Shao, Chong Ruan, Conner Sun, and 300 others. 2026. Deepseek-v4: Towards highly efficient million-token context intelligence. Preprint, arXiv:2606.19348.

Ronald Fagin, Joseph Y. Halpern, Yoram Moses, and Moshe Y. Vardi. 1995. Reasoning About Knowledge. MIT Press.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. 2023. Dissecting recall of factual associations in auto-regressive language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235, Singapore. Association for Computational Linguistics.

Google DeepMind. 2026. Gemini 3 flash model card. Model card. Gemini 3 Flash; accessed October 2026.

Guiyang Hou, Wenqi Zhang, Yongliang Shen, Linjuan Wu, and Weiming Lu. 2024. TimeToM: Temporal space is the key to unlocking the door of large language models’ theory-of-mind. In Findings of the Association for Computational Linguistics: ACL 2024, pages 11532–11547, Bangkok, Thailand. Association for Computational Linguistics.

Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. 2023. FANToM: A benchmark for stress-testing machine theory of mind in interactions. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 14397–14413, Singapore. Association for Computational Linguistics.

Peter Kinderman, Robin Dunbar, and Richard P. Bentall. 1998. Theory-of-mind deficits and causal attributions. British Journal ofPsychology, 89(2):191–204.

Michal Kosinski. 2024. Evaluating large language models in theory of mind tasks. Proceedings of the National Academy ofSciences, 121(45):e2405460121.

Matthew Le, Y-Lan Boureau, and Maximilian Nickel. 2019. Revisiting the evaluation of theory of mind through question answering. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5872–5877, Hong Kong, China. Association for Computational Linguistics.

Chao Lei, Guang Hu, Meng Yang, Yanbei Jiang, and Nir Lipovetzky. 2026. Mind the perspective: Let’s reason recursively for theory of mind. Preprint, arXiv:2606.11724.

Kevin Meng, David Bau, Alex J Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems.

OpenAI. 2026. GPT-6 system card. Model card. Accessed October 2026.

Cathleen O’Grady, Christian Kliesch, Kenny Smith, and Thomas C. Scott-Phillips. 2015. The ease and extent of recursive mindreading, across implicit and explicit tasks. Evolution and Human Behavior, 36(4):313– 322.

Josef Perner and Heinz Wimmer. 1985. “john thinks that mary thinks that. . . ” attribution of second-order beliefs by 5- to 10-year-old children. Journal of Experimental Child Psychology, 39(3):437–471.

Nikhil Prakash, Natalie Shapira, Arnab Sen Sharma, Christoph Riedl, Yonatan Belinkov, Tamar Rott Shaham, David Bau, and Atticus Geiger. 2026. Language models use lookbacks to track beliefs. In The Fourteenth International Conference on Learning Representations.

David Premack and Guy Woodruff. 1978. Does the chimpanzee have a theory of mind? Behavioral and Brain Sciences, 1(4):515–526.

Melanie Sclar, Jane Dwivedi-Yu, Maryam Fazel Zarandi, Yulia Tsvetkov, Yonatan Bisk, Yejin Choi, and Asli Celikyilmaz. 2025. Explore theory of mind: program-guided adversarial data generation for theory of mind reasoning. In The Thirteenth International Conference on Learning Representations.

James Stiller and R.I.M. Dunbar. 2007. Perspectivetaking and memory capacity predict social network size. Social Networks, 29(1):93–104.

Winnie Street, John Oliver Siy, Geoff Keeling, Adrien Baranes, Benjamin Barnett, Michael McKibben, Tatenda Kanyere, Alison Lentz, Blaise Agüera y Arcas, and Robin I. M. Dunbar. 2026. Llms achieve adult human performance on higher-order theory of mind tasks. Frontiers in Human Neuroscience, Volume 19 - 2025.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle˘ Casbon, Mayank Chaturvedi, Aditya Chawla, Victor Cotruta, Alice Coucke, Phil Culliton, Robert Dadashi, Lucas Dixon, Mohamed Elhawaty, Utku Evci, and 304 others. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems.

Alex Wilf, Sihyun Lee, Paul Pu Liang, and Louis-Philippe Morency. 2024. Think twice: Perspectivetaking improves large language models’ theory-ofmind capabilities. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8292–8308, Bangkok, Thailand. Association for Computational Linguistics.

Ross Wilson, Ales Hruby, Daniel Perez-Zapata, Sanne W van der Kleij, and Ian A Apperly. 2023. Is recursive “mindreading” really an exception to limitations on recursive thinking? Journal ofExperimental Psychology: General, 152(5):1454.

Heinz Wimmer and Josef Perner. 1983. Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children’s understanding of deception. Cognition, 13(1):103–128.

Yufan Wu, Yinghui He, Yilin Jia, Rada Mihalcea, Yulong Chen, and Naihao Deng. 2023. Hi-ToM: A benchmark for evaluating higher-order theory of mind reasoning in large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10691–10706, Singapore. Association for Computational Linguistics.

Hainiu Xu, Runcong Zhao, Lixing Zhu, Jinhua Du, and Yulan He. 2024. OpenToM: A comprehensive benchmark for evaluating theory-of-mind reasoning capabilities of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8593–8623, Bangkok, Thailand. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2024. Large language models are not robust multiple choice selectors. In The Twelfth International Conference on Learning Representations.

## A Benchmark Audit

Table 5 lists the benchmarks with nested-belief questions. For each audited subset, we encode the stories into perspective records and apply the support rule of Sec. 3.1 to every belief prefix. As an encoding check, we compare the solver’s answers with the released labels.

ToMi. We encode the test split of the balanced release programmatically and audit its 1,998 secondorder questions. Mean effective order is 1.36, and 36.2% of the questions have full effective order. The other 1,275 questions have collapsed layers: 766 through idle gaps and 509 through subsumed perspectives. Our solver matches the released gold answer on all 550 “ToM” questions and on 68.0% of the 1,448 “no-ToM” questions. This gap is consistent with the label inconsistencies reported by Arodi and Cheung (2021).

ExploreToM. The released sample has 619 story structures and 13,309 questions: 3,278 first-order, 9,184 second-order, and 847 memory questions. We compile its action scripts into perspective records and audit the 3,905 location questions (607 first-order and 3,298 second-order). On this subset, the solver matches the released answers on 98.5% of questions. For the second-order location questions, mean effective order is 1.07 and 6.6% have full effective order. Of the remaining 3,080 questions, 2,881 contain idle gaps and 199 contain subsumed perspectives.

Hi-ToM. Hi-ToM has 120 unique questions at each published level from zero to four. Every question appears twice, once in a chain-of-thought prompt (CoTP) and once in a vanilla prompt (VP). For 138 questions the two copies carry different gold answers (0, 0, 35, 48, and 55 at levels zero to four), which leaves 120, 120, 85, 72, and 65 questions with consistent labels. Answer-based scoring uses the released labels. The effective-order audit does not depend on gold answers, so it covers all 600 unique questions. Table 2 reports it for orders one to four.

MoToMQA. MoToMQA consists of handwritten stories without action logs, and its statements involve desires, intentions, and emotions besides beliefs. We therefore report its audit here rather than in Table 1. MoToMQA’s published levels two to six count the respondent and every mental-state verb, including those in coordinated clauses, so its 70 ToM statements have syntactic orders 1, 2, 3, 3–4, and 3–4. We annotate the event that supports each belief prefix with an LLM-based annotator. Mean effective order at the five published levels is 1.00, 1.07, 1.50, 1.50, and 2.00. No statement above published level three has full effective order, and the highest effective order observed is three. In 10 of the 21 true statements with k ≥ 3, the story states the nested mental state explicitly, so a single sentence settles several prefixes at once.

## B HOPTOM Construction

Each prototype combines one or more of the 10 scenario types in Table 6.

Each narrative realization has one control at each syntactic order from two to five, giving 400 controls across the 100 narrative realizations. By collapse type, 120 contain subsumed perspectives, 130 idle gaps, and 100 both; the remaining 50 have full effective order but an answer that coincides with a lower-order alternative. Their mean effective orders at orders two to five are 1.20, 1.80, 2.70, and 3.10.

Each of the 100 underlying interactions is realized in two story formats: a prose narrative and a script-style dialogue with character lines and stage directions. The two realizations of an interaction share the same questions, gold answers, belief chains, and labels, and differ only in how the story is presented. Every question is provided with open-ended answers and as a seven-option multiple-choice item with the same stem and gold answer (Appendix E.3). HOPTOM is bilingual: every story and question is provided in Chinese and English with identical belief chains, orders, labels, and option positions, giving 400 stories and 3,600 questions across the two languages.

<table><tr><td>Benchmark</td><td>Source</td><td>Max level</td><td>Size</td><td>Format</td><td>Order convention</td></tr><tr><td>ToMi†</td><td>Le et al. (2019)</td><td>2</td><td>5,994 test questions</td><td>open-ended</td><td>nesting depth</td></tr><tr><td>Hi-ToM†</td><td>Wu et al. (2023)</td><td>4</td><td>120 stories, 600 unique questions</td><td>15-way choice</td><td>nesting depth</td></tr><tr><td>FANToM</td><td>Kim et al. (2023)</td><td></td><td>256 conversations, 870 short sets</td><td>choice, free-form, list</td><td>nesting depth</td></tr><tr><td>OpenToM</td><td>Xu et al. (2024)</td><td>222</td><td>696 narratives, 16,008 questions</td><td>2/3-way choice</td><td>nesting depth</td></tr><tr><td>ToMBench</td><td>Chen et al. (2024)</td><td></td><td>2,860 items</td><td>4-way choice</td><td>task taxonomy</td></tr><tr><td>ExploreToM†</td><td>Sclar et al. (2025)</td><td>2</td><td>619 stories, 13,309 questions (sample)</td><td>open-ended, yes/no</td><td>nesting depth</td></tr><tr><td>MoToMQA†</td><td>Street et al. (2026)</td><td>6</td><td>7 stories, 70 ToM statements</td><td>true/false</td><td>respondent included in level</td></tr><tr><td>HoPToM</td><td>this work</td><td>5</td><td>200 realizations, 1,800 questions</td><td>open-ended, optional choice</td><td>certified effective order</td></tr></table>

Table 5: Benchmarks with nested-belief questions. Maximum levels follow each source’s convention. <sup>†</sup>Audited in this work; Appendix A describes the audited subsets.

<table><tr><td>Family</td><td>Type</td><td>Definition</td></tr><tr><td>Asymmetric Epistemic Access</td><td>present but inattentive</td><td>An agent is physically present during an event but does not attend to it and therefore does not acquire the relevant information.</td></tr><tr><td></td><td>secret observation</td><td>An agent observes an event without one or more other agents knowing that the observation occurred.</td></tr><tr><td>Temporal Presence Dynamics</td><td>leave and return</td><td>An agent leaves before a relevant event and later returns without direct access to what occurred during the absence.</td></tr><tr><td></td><td>late joining</td><td>An agent enters after earlier events have occurred and lacks the information available before arrival.</td></tr><tr><td>sentation</td><td>Strategic Information Pre- deceptive communication</td><td>An agent intentionally provides false or misleading information to influence another agent&#x27;s belief.</td></tr><tr><td>Unintentional Epistemic Error</td><td>observation misperception</td><td>An agent observes an event but incorrectly interprets what happened.</td></tr><tr><td></td><td>unintentional misreport</td><td>An agent communicates incorrect information because of an error rather than an intention to deceive.</td></tr><tr><td>Asynchronous Epistemic Updating</td><td>delayed meta-information</td><td>An agent learns only later that another agent previously observed, knew, or communicated a relevant fact.</td></tr><tr><td></td><td>nonmonotonic trace</td><td>Later evidence causes an agent to revise, retract, or reinstate an earlier belief.</td></tr><tr><td>Cyclic Query</td><td>cyclic query</td><td>A query follows a nested belief chain that returns to an earlier agent, such as &quot;A believes that B believes that A believes something.&quot;</td></tr></table>

Table 6: The ten scenario types of HOPTOM, grouped into six families. A prototype can combine several types.

## C Human Study

Procedure. Participants were recruited through an Chinese online survey panel in September 2026 and completed a web questionnaire with three stories, drawn from prototypes P1 (“the painting in the garden”), P3 (“the museum tickets”), and P9 (“the passport and the power bank”). Each story was followed by its five certified questions (orders one to five) and then its four controls (orders two to five). For each question, participants typed an answer and rated their confidence on a five-point scale.

Participants. Twenty participants answered all 27 questions: 11 female and 9 male; 12 aged 21– 30, 6 aged 31–40, and 2 aged 41–50; 17 with a bachelor’s degree, 2 with a master’s degree, and 1 with a college diploma. Mean completion time was 24.0 minutes.

Accuracy. Table 7 reports accuracy by syntactic order. On certified questions, accuracy is 58.3%, 31.7%, and 58.3% at orders one to three, and falls to 6.7% and 11.7% at orders four and five. Controls are answered more accurately than certified questions at orders two and four, with the largest gap at order two (76.7% versus 31.7%).

<table><tr><td></td><td>1st</td><td>2nd</td><td>3rd</td><td>4th</td><td>5th</td><td>All</td></tr><tr><td>Certified</td><td>58.3</td><td>31.7</td><td>58.3</td><td>6.7</td><td>11.7</td><td>33.3</td></tr><tr><td>Control</td><td>1</td><td>76.7</td><td>55.0</td><td>41.7</td><td>11.7</td><td>46.3</td></tr><tr><td>Overall</td><td>58.3</td><td>54.2</td><td>56.7</td><td>24.2</td><td>11.7</td><td>39.1</td></tr></table>

Table 7: Human accuracy (%) by syntactic order.

Confidence. Mean confidence was 3.82 out of five. It was higher for correct than for incorrect answers (4.10 versus 3.61; paired permutation test, p < 0.001). Accuracy was 24.7%, 30.0%, and 45.4% at low, medium, and high confidence. Across participants, mean confidence did not correlate significantly with accuracy (Spearman ρ = -0.14, p = 0.55).

Ethics. The study was approved by the institutional review board of the authors’ institution. Before starting, participants read a description of the study and gave informed consent; they could withdraw at any time. Each participant received monetary compensation for completing the questionnaire. We collected no identifying information beyond the demographic items above.

## D Experimental Details

We queried the closed models in September 2026, using the identifiers gpt-6-sol, claude-opus-5-5, Gemini 3 Flash, and deepseek-v4-pro. The main runs do not request extended thinking. In the thinking runs, Claude Opus 5.5 uses adaptive thinking at high effort and Gemini 3 Flash uses its default alias. We ran Gemma 4 31B and Qwen3-32B with Hugging Face Transformers in bf16 on four NVIDIA A800 (80 GB) GPUs, with greedy decoding and thinking disabled.

## E Additional Results

## E.1 Full-Benchmark Results

Across all 1,800 questions (certified questions and controls, narrative and dialogue realizations), CoT reaches 59.4%, 59.1%, 57.7%, and 45.6% with GPT-6 Sol, Claude Opus 5.5, Gemini 3 Flash, and DeepSeek-V4-Pro. SimToM (Wilf et al., 2024) reaches 57.6%, 59.2%, 40.8%, and 40.7%, and HOPTRACE reaches 71.2%, 70.4%, 72.9%, and 72.3%. On the controls of the 100 narrative realizations, HOPTRACE reaches 67.8%, 68.2%, 68.2%, and 67.0% in the same model order. Certified results are provided in Table 4.

## E.2 Inference Time

Per narrative realization, HOPTRACE takes 217, 98, 4, and 29 seconds with GPT-6 Sol, Claude Opus 5.5, Gemini 3 Flash, and DeepSeek-V4-Pro. Time-ToM takes 808, 278, 442, and 1,067 seconds, and RecToM takes 827, 268, 455, and 1,258 seconds.

## E.3 Open-Ended and Multiple-Choice Answers

Format. Each multiple-choice item reuses the story and question of an open-ended item and lists seven options: the gold answer, five distractors, and “I don’t know”. The distractors come from a per-prototype pool of states that the story mentions for the same target variable, so they can include the target’s actual state or the answer to a shorter belief prefix. For each of the nine question positions, both the gold answer and “I don’t know” are spread evenly across the seven labels over the 100 interactions, and “I don’t know” never occupies the gold position. Models answer the 900 questions of the 100 narrative realizations. Appendix F.5 analyzes how open-weight models answer this format.

<table><tr><td>Model</td><td>Open MC</td><td></td><td>p</td><td>Certified</td><td>Controls</td></tr><tr><td>Claude Opus 5.5</td><td>34.0</td><td></td><td> $9 2 . 4 \ < \ 1 0 ^ { - 1 0 0 }$ </td><td> $3 5 . 2  8 9 . 2 $ </td><td> $3 2 . 5  9 6 . 5$ </td></tr><tr><td>GPT-6 Sol</td><td>55.4</td><td></td><td> $5 8 . 3 \qquad 0 . 1 2$ </td><td> $5 5 . 4  5 5 . 6$ </td><td> $5 5 . 5  6 1 . 8$ </td></tr><tr><td>Gemini 3 Flash</td><td>34.2</td><td></td><td> $4 4 . 0 ~ < { 1 0 } ^ { - 1 0 }$ </td><td> $3 4 . 0  4 5 . 0$ </td><td> $3 4 . 5  4 2 . 8$ </td></tr><tr><td>DeepSeek-V4-Pro</td><td>36.2</td><td></td><td> $3 0 . 0 \ \mathrm { ~ 2 ~ } \times 1 0 ^ { - 4 }$ </td><td> $3 4 . 2  3 3 . 2$ </td><td> $3 8 . 8  2 6 . 0$ </td></tr></table>

Table 8: Open-ended and seven-option multiple-choice accuracy (%) on the 900 questions of the 100 narrative realizations. Open and MC: accuracy in each format. p: paired McNemar test between the two formats. Certified and Controls: accuracy on each question subset, open-ended → multiple choice.

Results. The effect of answer options depends on the model (Table 8). Claude Opus 5.5 gains 58.4 points and reaches 96.5% on controls. Gemini 3 Flash gains 9.8 points. DeepSeek-V4-Pro loses 6.2 points, mostly on controls. GPT-6 Sol gains 2.9 points, which is not significant under the paired test. Options can supply cues and allow elimination, and they can also introduce response biases. We therefore report open-ended accuracy as the primary measure of answering nested-belief queries without candidate answers.

Mechanistic analysis. Appendix F.5 examines how options help the two open-weight models. Options raise accuracy from 33% to 45% for Gemma 4 31B and from 25% to 37% for Qwen3-32B, and accuracy on shortcut-resistant certified questions rises from 10% to 24–29%. Among correct multiplechoice answers, the paragraph that decides the answer is still the most influential one in 74–75% of cases. The dependence on this paragraph weakens with order, however: hiding it costs Qwen3 only about 1 nat at fifth order. Correct answers to deep questions therefore draw more on the options, consistent with answering by elimination.

## F Mechanistic Analysis

## F.1 Experimental Setup

We study Gemma 4 31B and Qwen3-32B in bf16 with greedy decoding on single questions. The analyses rely on two interventions. Segment hiding blocks attention from the question and answer tokens to one story paragraph or prompt block, and measures the drop in the log-probability of the model’s original answer in nats (Geva et al., 2023). We call a drop above 1 nat critical. Activation patching replaces the residual vector at a given layer and token position with the vector from another run and records how the answer changes (Meng et al., 2022). For a query q, the deciding record is $E _ { s _ { k } }$ , the record supporting the full prefix $p _ { k }$ . A question is shortcut-solvable if its gold answer equals the answer to a shorter prefix, the outermost or innermost agent’s own belief, or the target’s actual state, and shortcut-resistant otherwise.

![](images/3ef664a3a49d38b4fd0fdd80442b42b8da449d457d5f27c8bfc7a6786fe8fbbc.jpg)  
Figure 8: Direct-answering accuracy of Gemma 4 31B (solid) and Qwen3-32B (hatched) on certified questions of order $k \geq 2 .$ , split by whether the question is shortcutsolvable (Appendix F.1). Accuracy by order is in Fig. 3b, and wrong answers are broken down in Fig. 6a.

## F.2 Direct-Answering Behavior

On all 1,800 questions, Gemma and Qwen3 reach 33.2% and 25.3%. On certified questions of order $k \geq 2$ that are shortcut-resistant, both reach 10%, compared with 64% and 56% on shortcutsolvable ones (Fig. 8). Among their errors on certified fourth-order questions, 75% (Gemma) and 50% (Qwen3) equal the answer to a shallower prefix.

## F.3 Evidence Used by Direct Answers

For correct answers, hiding the paragraph of the deciding record lowers the answer’s log-probability by 16.7 nats for Gemma and 12.1 for Qwen3 (Fig. 6b). Paragraphs that support shorter prefixes matter far less: 0.5 and 0.2 nats when hidden one at a time, and 1.2 and 0.25 nats when hidden together. Hiding the same number of random paragraphs gives 0.06 and 0.02 nats. At every effective order, about one paragraph per answer passes the 1-nat threshold.

We separately examine wrong answers that equal a shallower prefix’s answer (214 for Gemma, 187 for Qwen3). In 81% and 90% of these cases, the paragraph supporting that prefix has the largest effect, with mean drops of 12.6 and 11.2 nats. Hiding the deciding paragraph of the full query changes the log-probability by only 0.01 and 0.13 nats. For this class of errors, the answer depends mainly on evidence for a shallower query.

## F.4 Activation Patching

On third-order pairs that differ in one agent, the changed agent’s information moves from its name to the final token at similar depths for all three query positions (Fig. 9). Patching the other agents’ tokens has little effect, apart from a weak signal at later agents’ tokens when $q _ { 1 }$ is replaced. These interventions do not rule out intermediate nested representations.

## F.5 Multiple Choice

With seven options, including “I don’t know”, accuracy rises from 33% to 45% for Gemma and from 25% to 37% for Qwen3 (Fig. 10a–b). Both models also favor option A, although gold positions are balanced across labels (Appendix E.3).

Among correct multiple-choice answers, the deciding paragraph is still the most influential paragraph in 74–75% of cases, including answers that are correct only with options (Fig. 10c). The options add a second source of answer information. Hiding the deciding paragraph costs less than in the open-ended format: 9.0 instead of 16.9 nats for Gemma and 4.0 instead of 12.5 for Qwen3. Hiding only the correct option line costs 1.6–1.7 nats. At fifth order, the deciding paragraph’s effect for Qwen3 falls to about 1 nat, so these answers rely more on the options, consistent with answering by elimination (Fig. 10d).

## F.6 HOPTRACE Encoding

With one encoding call and programmatic query resolution, Gemma reaches 67.7% and Qwen3 reaches 38.7% on all 1,800 questions. To examine the encoding step, we append the generated records to the story and hide story paragraphs while measuring how much each record depends on them. Record order follows paragraph order in 96% (Gemma) and 99% (Qwen3) of cases. The record selected by the solver depends on the deciding paragraph in 89% of correct answers but in only 13–26% of wrong answers. Many of the remaining errors therefore originate in the encoded records.

![](images/92a243e6cdc23ffdc4f129295bbec6ae592f78088c94addf08823b86aa745c29.jpg)  
Figure 9: Token-by-layer patching on third-order pairs that differ in one agent. Each cell gives the share of pairs whose answer flips when the residual vector at that question token (rows) and layer (columns) is patched. Columns: the replaced agent $q _ { 1 } , q _ { 2 } , \mathrm { o r } q _ { 3 }$

![](images/4b6d78f830da1efff86bbee2ae422d9e9b09b2b152d20ead7f745293a891579c.jpg)

![](images/c5900eac9f528002f3697b5b4bf7703428c96e2d1f68cd1ec5b28fc386092632.jpg)

c.  
![](images/e2e33be4c5f6157ab48e699c93f9eca7109997fb829dcee8557180d78cf1f5a1.jpg)  
d.

![](images/61c92045b555f13b37ee38ea02ecbe325b6972fc6f06c6fa54d084bb4558dcaa.jpg)  
Figure 10: Open-ended and seven-option multiple-choice answers by Gemma 4 31B (solid lines and bars) and Qwen3-32B (dashed lines, hatched bars). (a) Accuracy on certified questions by order. (b) Accuracy on shortcutresistant certified questions of order $k \geq 2 .$ (c) Correct multiple-choice answers: log-probability drop when the deciding paragraph is hidden (open-ended and multiple-choice formats), another paragraph is hidden, the option block is hidden, or only the correct option line is hidden. (d) Drop when the deciding paragraph is hidden, by order.