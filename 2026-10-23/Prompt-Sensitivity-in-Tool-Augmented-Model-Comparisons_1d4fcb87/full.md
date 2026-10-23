# Prompt Sensitivity in Tool-Augmented Model Comparisons

Mateusz Modrzejewski , Tomasz Radzikowski Institute of Computer Science, Warsaw University of Technology {mateusz.modrzejewski, tomasz.radzikowski}@pw.edu.pl

## Abstract

Language models are compared on questionanswering benchmarks that pose each question in one wording, and a reworded question can change which of two models scores higher. We ask whether giving the models tools makes such comparisons more stable. An accuracy gain cannot answer this, because tools change several steps between the question and the scored answer at once. We propose a matched protocol that separates those steps: the answer readout, reasoning, the decision to call, the tool’s answer, and the report supplied without a call. We apply it to six open models in five domains under five wordings. We find that tools raise accuracy where a program answers the question and widen the wording sensitivity of many pairwise comparisons. The widening follows how often a model calls, and call frequency follows the wording for some models and not others. It also depends on the interface and the instructions: a native toolcalling interface and a second tool description leave no detected widening in molecules, and sensitivity persists where tools return intermediate results. These findings show that deterministic reports do not guarantee stable comparisons. We recommend reporting call frequency and comparison stability under several wordings beside accuracy.

## 1 Introduction

A benchmark asks a model a question in one wording and counts correct answers. The same question in another wording gives another score. The difference between two models, which decides their order in a ranking, moves with the wording by as much as it moves with a change of model [Sclar et al., 2023; Mizrahi et al., 2023]. Multi-prompt evaluation proposes to report several wordings. Tool use raises a further question. In a tool loop of the kind ReAct introduced [Yao et al., 2022], the model writes a call, a program answers, and the model reads the answer before it decides. The program’s answer is the same under every wording, but nothing guarantees that the model makes the same call under every wording.

One molecule item shows the difference. It asks for the number of heavy atoms in the molecule

(a)  
![](images/b61d511f12c666c04bbc425c8182431d113ea9274f3e62e0fc4382e5452dc3ae.jpg)

(b)  
![](images/f16793cbe921759c67aec023152a17318e8dcbcf5e7ae1183d235325abc85bd9.jpg)  
Figure 1: (a) Median wording spread of a model pair by condition in the letter domains, six models. (b) Change in a pair’s spread when the tool loop replaces reasoning without tools, one point per pair and written form, median as a bar.

C=CC1=C(C=CC(=C1)Br)Br, with four options. Given a heavy-atom tool, Qwen3-8B calls it under four of our five wordings and answers correctly each time. Under the remaining wording, which shows the molecule before the question, it counts the atoms in prose, makes no call before its generation budget ends, and the letter read at the appended answer line is wrong. When the tool’s report is printed in the prompt, the model answers correctly under all five wordings. The report is the same in every case. What differs is whether the model obtains it.

An accuracy gain under tools cannot show whether tools stabilize comparisons, because several steps change at once. The answer is no longer read from the probability of the first option letter, a common readout that for instruction-tuned models often disagrees with the letter the model writes [Wang et al., 2024]. The model writes text before it answers. It decides whether to call a tool after reading the wording. It receives the tools’ reports. Any of these steps can add or remove wording sensitivity. A comparison of the same model with and without tools mixes them.

We separate the steps on the same items with matched conditions: the first-token read, an answer read at an answer line, reasoning before that line, the tools’ reports supplied in the prompt, a plain-text tool loop over deterministic programs, the same loop with tools that answer “no information”, and for three models the native tool-calling interface. Every condition runs under five wordings of the instruction and under each written form of the input. For each pair of models we measure the wording spread, the range of their accuracy difference across the wordings. The tools’ information and the model’s call behavior can affect this spread in different ways. Supplying the report under every wording measures the comparison with direct access to the information. Letting the model decide whether to obtain it exposes the comparison to changes in calling as well. We compare the two and intervene on the interface, the instructions, the budget and the reports reliability to see when the spread changes.

Tools raise accuracy where a program answers the question and still widen the wording spread of many model pairs against reasoning without tools (Figure 1). In molecules the median spread of a pair is 0.17 with the plain-text loop and 0.02 with the reports supplied. There, a model that calls is almost always right, so its accuracy follows its call frequency, and for several models the call frequency follows the wording. The widening depends on the interface and the instructions. In molecules, the native interface and a second tool description leave no detected widening in the model groups tested. Forcing one call and removing the wordings’ output clause help some models and not others. Wrong reports shrink the widening while the call frequencies stay. Sensitivity persists under the native interface where the tools return intermediate results. Reading the answer from the first token adds sensitivity of its own on graphs, which an answer line removes.

Scope. We study open models of 4B to 72B parameters with greedy decoding, a plain-text tool loop and, for three model families, the native interface, wordings in English, and deterministic programs that read a canonical form of the item. The quantity is the stability of a comparison between two models across wordings rather than the robustness of one model, and the spreads are statistics of our wording set. The molecule and graph tasks are built so that an operator answers the question, which isolates the call and the use of the report rather than tool discovery or argument construction. The question is how far the comparisons that tools produce can be trusted. The results say which evaluation choices change that.

This paper makes three contributions. First, we propose a matched protocol that measures how the answer readout, reasoning, the decision to call, the tool interface and access to the tools’ reports change the stability of pairwise model com parisons across wordings. Second, we split each model’s accuracy changes across wordings into a call-frequency component and a conditional-accuracy component and relate them to the movement of pairwise differences. Third, we apply both to six open models in five domains, a Qwen3 size series and a 72B model, and test the result against seven interventions on the call. Paraphrases of the user’s request under tool calling [Rabinovich and Anaby Tavor, 2025; Zhou et al., 2026; Chen et al., 2026] and the decision to call [Ross et al., 2025; Wu et al., 2026; Cheng et al., 2026; Sun et al., 2026] are established objects of study. Our focus is the stability of pairwise comparisons of executed answers, read through matched changes to the readout, the call and access to the reports. The wordings, the tool descriptions and the supporting tables are in the supplementary material (Sections S1 to S4).

## 2 Evaluation Conditions and Measures

Pairwise wording spread. For two models a and b and a set of wordings $\breve { W }$ , let acc<sub>a</sub>(w) be the accuracy of model a under wording w. The wording spread of the pair is

$$
S _ { a b } = \operatorname* { m a x } _ { w \in W } D _ { a b } ( w ) - \operatorname* { m i n } _ { w \in W } D _ { a b } ( w ) ,\tag{1}
$$

where $D _ { a b } ( w ) = \csc _ { a } ( w ) - \operatorname { a c c } _ { b } ( w )$ is the difference that orders the two models, so the spread is its range over wordings. We compute it for each condition and each written form of the input, on the items present for both models under every wording, where a cell enters only when it holds at least 0.9 of the domain’s items and an answer that cannot be parsed counts as incorrect. Intervals come from a paired bootstrap over items. We also report each model’s own accuracy range across wordings.

The conditions. All conditions use the same items, wordings and written forms, and decoding is greedy. Under the first-token read, the answer to a four-option item is the option letter with the highest probability among the four letters at the first position of the response. Under the fixed readout, an answer line (the text “ANSWER:”) follows the prompt, and the letter is read there with the same restricted read. For a numeric item the model writes the number on that line, and we call this the direct answer. At the read position we record the letter mass, the share of the full next-token probability that falls on the four letters, after any formatting mark such as a markdown bold mark. Under reasoning, the model writes freely before the answer line. Under tools, a fixed tool description precedes the unchanged wording. The model may write call lines, each naming one tool, each tool’s report is appended as a line, and the episode ends at an answer line, after six calls, or at the token budget, where an answer line is appended and read. Under constant tools, the loop and the description are unchanged and every tool answers “no information”. Under the supplied report, the report that the loop would print for the item’s question type is appended to the fixed readout’s prompt, with no tool description, no loop and no call. Under the forced call, the loop refuses an answer without a call once. Under the native interface, the same tools are declared through the model’s own tool-calling format, so a call is a structured turn rather than a text line.

Interpretation of the controls. The conditions are matched diagnostic contrasts rather than a partition of causes, and each has a reading fixed before its result. From the first-token read to the fixed readout, the position and the context of the read change. Reasoning adds the model’s own text before the answer. The supplied report adds the tool’s information at the start of the prompt and nothing the model must do to obtain it. The tool loop adds the description, the call syntax, the reports and the loop’s limits together. Constant tools keep that interface and remove the information, which can also change how often the model calls again and stops. A difference between tools and constant tools is read as the information’s and a difference between constant tools and reasoning as the loop’s. Chance, 0.25 on a four-option item, and the supplied report are the two reference values, neither a bound on what a model can reach. If the spread under tools lies close to the supplied report’s, the information settles the comparison. If it exceeds both the supplied report’s and the fixed readout’s, follows the call frequency where a tool answers the question, and falls under an intervention that makes the model call under every wording, the call behavior carries it.

<table><tr><td>Domain</td><td>Items</td><td>Written forms; what the tools return</td></tr><tr><td>Molecules</td><td>504, four options, Pub- Chem</td><td>SMILES, random SMILES, InChI, IUPAC name; RDKit answers every question type</td></tr><tr><td>Graphs</td><td>erated</td><td>504, four options, gen- edge list, neighbor list, friendship; NetworkX answers every type</td></tr><tr><td>Music</td><td>361, four ABC-Eval</td><td>options, ABC, MusicXML, note events; mu- sic21 answers bar count, and key and time partly</td></tr><tr><td>Tables</td><td>500, numeric, FinQA</td><td>markdown, CSV, sentences; rows, columns and a calculator</td></tr><tr><td></td><td>500</td><td>Mathematics 336, numeric, MATH- one form; a calculator and an equa- tion solver</td></tr></table>

Table 1: The domains, written forms and tools. Every written form of an item shares one canonical tool input and gold answer. In molecules and graphs an operator answers the question directly; in tables and mathematics the tools return values the model must combine.

Call-frequency decomposition. In a tool condition, a model’s accuracy under wording w is an identity in its call frequency $c _ { w }$ , the share of episodes with at least one call, and the accuracies $a _ { w } ^ { 1 }$ and $a _ { w } ^ { 0 }$ of the episodes with and without a call:

$$
\operatorname { a c c } ( w ) = a _ { w } ^ { 0 } + c _ { w } ( a _ { w } ^ { 1 } - a _ { w } ^ { 0 } ) .\tag{2}
$$

The difference $a _ { w } ^ { 1 } - a _ { w } ^ { 0 }$ is observed on different episodes and is not the causal effect of a call. With a reference $( c _ { * } , a _ { * } ^ { 1 } , a _ { * } ^ { 0 } )$ pooled over the five wordings’ episodes, the change acc(w)− acc<sub>∗</sub> splits exactly into a call-frequency component and a conditional-accuracy component by the two-factor Shapley rule:

$$
\begin{array} { r } { C _ { w } = \frac { 1 } { 2 } ( c _ { w } - c _ { * } ) \left[ ( a _ { w } ^ { 1 } - a _ { w } ^ { 0 } ) + ( a _ { * } ^ { 1 } - a _ { * } ^ { 0 } ) \right] , } \end{array}\tag{3}
$$

and the rest is the conditional component. A wording without any episode in one stratum takes the reference value there. For a pair of models, the call share is the covariance over wordings of the pair difference with the difference of the two models’ $C _ { w } ,$ divided by the variance of the pair difference, with forms pooled. The two components’ shares sum to one, a share can lie outside [0, 1] when the components move against each other, and the share describes covariation with the pair difference rather than a decomposition of the range in Equation 1. The call frequency counts whether any tool was called and not which items or which tools, so a constant frequency does not establish that the same items receive calls.

## 3 Experimental Setup

Domains. The five domains differ in what the tools answer (Table 1). Every tool reads one canonical form of the item, so its reports are identical across written forms and wordings. The molecule items ask for six RDKit properties of compounds drawn from PubChem<sup>1</sup>, 84 per property. The graph items follow the tasks of Fatemi et al. [2023] on random graphs of 5 to 20 nodes, 84 per task. The gold answers of both domains come from the same libraries as the tools, so these are controlled tasks with direct-answer operators. The music items are those of ABC-Eval [Zhao et al., 2025] that fit a common token budget in three notations, with music21 operators [Cuthbert and Ariza, 2010]. The table items are FinQA questions answerable from the table alone [Chen et al., 2021], and the mathematics items the MATH-500 problems with a plain numeric answer [Hendrycks et al., 2021; Lightman et al., 2023]. A domain entered only if, on its first 200 items under the fixed readout, Qwen3-8B or Phi-4 scored at most 0.90 and their pair spread reached 0.02, which excluded GSM8K [Cobbe et al., 2021], where Qwen3-8B and Phi-4 score 0.95 and 0.97. Music, where the protocol was developed, was exempt from the rule. The wordings do not mention tools. A letter answer is correct when it is the gold option. A mathematics answer is correct when it equals the gold number. A table answer is correct when it lies within 1% of the executed answer or within 0.005 of it, or when it has at least one decimal and equals the answer rounded to the decimals the model wrote (two decimals when the answer is below one). For a question whose gold value is a ratio, the same rules apply to the value times 100. Integers never match through rounding.

Models and harness. Six open instruction-tuned models, each at a fixed revision: Qwen3-8B with its thinking mode off, Phi-4, Llama-3.1-8B-Instruct, Gemma-3- 12B, Mistral-Small-24B and Qwen3-30B-A3B, a mixtureof-experts model [Qwen Team, 2025; Abdin et al., 2024; Llama Team, AI @ Meta, 2024; Gemma Team, 2025; Mistral AI, 2025]. Three of them run with 8-bit weights. Qwen3- 4B, Qwen3-14B and Qwen3-32B, dense models, form a size series with the two Qwen3 models of the grid, and Qwen2.5- 72B-Instruct, a model of the previous generation with 8-bit weights, is the largest, both on molecules, graphs and music. The loop generates 64 tokens per step in the letter domains and 256 in the number domains, within an episode budget of 256 tokens in music and molecules, 384 on graphs, 1,024 in tables and 1,536 in mathematics. The models’ outputs are the data of this study: every answer, call line and trajectory analysed here was generated by these models under the conditions described, with greedy decoding, on vLLM. The revisions, budgets, tool descriptions and the native schemas are in the supplementary material (Section S2).

Analysis. Spreads and accuracies carry 95% percentile intervals from 1,000 bootstrap resamples over items. A contrast of two conditions is the difference of a pair’s two spreads under one resample of the items, its two-sided bootstrap pvalue is twice the share of resamples on the other side of zero, and Holm correction runs across the model pairs of one domain, written form and contrast. We count a pair as wider or narrower when the corrected contrast excludes zero, and we report the raw differences beside the counts. Under a simulated null of equal, nonzero wording effects, the bootstrap rejects single comparisons at the 0.05 level in 0.12 of 600 simulated contrasts for tools against the fixed readout in molecules (0.035 after Holm) and in at most 0.04 in the other contrasts checked. For that one row we also report a within-item exchange test, which permutes the two conditions’ outcome vectors within each item and rejects in 0.07 of 180 simulated contrasts, 0 after Holm (the null’s construction and both procedures’ counts for every contrast are in Section S3). Because neither procedure is shown to hold the nominal rate when the two conditions differ in accuracy level, the counts are exploratory flags, and conclusions about size rest on the raw paired differences of spread and on the interventions. Spreads below 0.05 are treated as zero when we read them, and every contrast is tested on the raw spreads. Repeating a full cell under batched greedy decoding changes the predictions of at most 0.07 of the rows in three letter-domain cells and 0.49 in one table cell, and moves any pair’s spread by at most 0.019 and 0.036. Mathematics has no full-size repeat. The smallest detected widening against reasoning is 0.09 in molecules and 0.10 in tables, above those moves.

<table><tr><td colspan="7">Qwen3 8B Phi-4 Llama 8B Gemma 12B Mistral 24B Qwen3 30B</td></tr><tr><td>Molecules</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>first-token read</td><td>0.36</td><td>0.46</td><td>0.34</td><td>0.41</td><td>0.47</td><td>0.46</td></tr><tr><td>fixed readout</td><td>0.43</td><td>0.52</td><td>0.38</td><td>0.42</td><td>0.50</td><td>0.49</td></tr><tr><td>reasoning</td><td>0.52</td><td>0.57</td><td>0.46</td><td>0.54</td><td>0.55</td><td>0.55</td></tr><tr><td>supplied report</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.98</td><td>1.00</td><td>1.00</td></tr><tr><td>tools</td><td>0.92</td><td>0.75</td><td>0.94</td><td>0.98</td><td>0.94</td><td>1.00</td></tr><tr><td>constant tools</td><td>0.44</td><td>0.51</td><td>0.33</td><td>0.42</td><td>0.45</td><td>0.49</td></tr><tr><td>Graphs</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>first-token read</td><td>0.34</td><td>0.71</td><td>0.57</td><td>0.75</td><td>0.78</td><td>0.71</td></tr><tr><td>fixed readout</td><td>0.73</td><td>0.80</td><td>0.63</td><td>0.79</td><td>0.84</td><td>0.81</td></tr><tr><td>reasoning</td><td>0.90</td><td>0.90</td><td>0.75</td><td>0.91</td><td>0.90</td><td>0.92</td></tr><tr><td>supplied report</td><td>0.97</td><td>0.96</td><td>0.93</td><td>0.97</td><td>0.98</td><td>0.98</td></tr><tr><td>tools</td><td>0.97</td><td>0.92</td><td>0.83</td><td>0.97</td><td>0.96</td><td>0.96</td></tr><tr><td>constant tools</td><td>0.81</td><td>0.83</td><td>0.53</td><td>0.73</td><td>0.76</td><td>0.83</td></tr><tr><td>Music</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>first-token read</td><td>0.39</td><td>0.41</td><td>0.38</td><td>0.39</td><td>0.41</td><td>0.43</td></tr><tr><td>fixed readout</td><td>0.40</td><td>0.43</td><td>0.42</td><td>0.39</td><td>0.44</td><td>0.45</td></tr><tr><td>reasoning</td><td>0.41</td><td>0.43</td><td>0.38</td><td>0.42</td><td>0.47</td><td>0.46</td></tr><tr><td>supplied report</td><td>0.59</td><td>0.59</td><td>0.59</td><td>0.57</td><td>0.60</td><td>0.59</td></tr><tr><td>tools</td><td>0.51</td><td>0.54</td><td>0.54</td><td>0.54</td><td>0.56</td><td>0.60</td></tr><tr><td>constant tools</td><td>0.40</td><td>0.44</td><td>0.38</td><td>0.38</td><td>0.42</td><td>0.48</td></tr><tr><td>Tables</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>direct answer</td><td>0.43</td><td>0.54</td><td>0.39</td><td>0.46</td><td>0.63</td><td>0.61</td></tr><tr><td>reasoning</td><td>0.73</td><td>0.75</td><td>0.55</td><td>0.71</td><td>0.74</td><td>0.74</td></tr><tr><td>tools</td><td>0.62</td><td>0.57</td><td>0.22</td><td>0.61</td><td>0.46</td><td>0.70</td></tr><tr><td>constant tools</td><td>0.54</td><td>0.53</td><td>0.19</td><td>0.29</td><td>0.45</td><td>0.68</td></tr><tr><td>Mathematics</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>direct answer</td><td>0.33</td><td>0.33</td><td>0.15</td><td>0.22</td><td>0.27</td><td>0.46</td></tr><tr><td>reasoning</td><td>0.82</td><td>0.84</td><td>0.56</td><td>0.85</td><td>0.77</td><td>0.80</td></tr><tr><td>tools</td><td>0.82</td><td>0.76</td><td>0.30</td><td>0.86</td><td>0.68</td><td>0.64</td></tr><tr><td>constant tools</td><td>0.79</td><td>0.76</td><td>0.20</td><td>0.84</td><td>0.65</td><td>0.65</td></tr></table>

Table 2: Mean accuracy by condition and domain, with equal weight over wordings and written forms; an unparsed answer is incorrect. In tables and mathematics the fixed readout is the direct answer and the supplied report is not defined.

## 4 Results

Repeatability and implementation checks. A separate run of the first-token read on the music items, with other hardware and code, agrees with ours on 0.95 to 0.99 of the answers of each of five models. Two runs of 200 items at batch size one give identical answers. Run alone, the tools return the gold answer by construction in molecules and graphs. In music they return it on 0.99 of the bar-count items, 0.63 of the key items and 0.53 of the time-signature items. Three models run again in bfloat16 move accuracy by at most 0.022 in 59 of 60 cells and by 0.06 in one. No contrast changes direction.

<table><tr><td></td><td>Molec.</td><td>Graphs Music</td><td></td><td>Tables</td><td>Math.</td></tr><tr><td>pairs × written forms</td><td>60</td><td>45</td><td>45</td><td>45</td><td>15</td></tr><tr><td>fixed readout vs. first-token read</td><td>0/5</td><td>0/36</td><td>0/3</td><td></td><td></td></tr><tr><td>reasoning vs. fixed readout</td><td>0/0</td><td>0/1</td><td>0/0</td><td>0/0</td><td>0/0</td></tr><tr><td>supplied report vs. fixed readout</td><td>0/21</td><td>0/5</td><td>0/0</td><td>一</td><td>一</td></tr><tr><td>tools vs. fixed readout</td><td>39/1†</td><td>10/0</td><td>19/0</td><td>一</td><td>一</td></tr><tr><td>tools vs. supplied report</td><td>54/0</td><td>16/0</td><td>22/0</td><td></td><td></td></tr><tr><td>tools vs. reasoning</td><td>37/2</td><td>12/0</td><td>17/0</td><td>15/0</td><td>5/0</td></tr><tr><td>constant tools vs. reasoning</td><td>1/0</td><td>10/0</td><td>0/0</td><td>32/0</td><td>5/0</td></tr><tr><td>call share under tools</td><td>0.94</td><td>0.75</td><td>0.57</td><td>0.06</td><td>0.07</td></tr></table>

Table 3: Pairs and written forms flagged wider/narrower under the first condition than the second, after Holm correction within domain, form and contrast, six models; the flags are exploratory (Section 3) and a zero does not establish equivalence. †: the exchange test (bootstrap 34/2). Last row: the call share.

Accuracy gains and tool coverage. In molecules tools raise accuracy over the fixed readout by 0.23 to 0.56, on graphs by 0.11 to 0.24 and in music by 0.10 to 0.15, for every model (Table 2). Constant tools move it by at most 0.10 in either direction, so the gain comes with the tools’ reports rather than with the loop alone. The gain follows what the operators know. In music it appears on bar count, where the operator is right on 0.99 of the items, and turns into a loss of 0.04 to 0.10 on key, where the operator is right on 0.63. In tables and mathematics the tools return values that the model must still combine, and accuracy falls below reasoning without tools for most models. Constant tools lower it by similar amounts, so there the loop costs more than the tools give. On the same graph items with only the two operators that return a node’s neighbors and whether two nodes share an edge, the gain over the fixed readout goes for Llama-3.1-8B and Phi-4, and constant tools widen where the tools no longer do (Section 5). A practitioner gains from tools on the question types that a program answers better than the model.

Answer-prefix scoring on graphs. Reading the answer at an answer line instead of the first token narrows the spread of 36 of 45 graph pairs, and the median falls from 0.25 to 0.05 (Table 3). Qwen3-8B places a median mass of 0.00 on the letters at the first token on graphs, but Gemma-3-12B and Qwen3-30B-A3B place 1.00 there and still lose 0.21 and 0.33 of their median graph spread. Over the 150 pair-by-form rows of the letter domains, the reduction correlates with the accuracy that the first-token read hides (Spearman 0.58) and not with the letter mass (−0.12). In molecules and music the readout changes almost nothing, permuting the options under the fixed readout changes the Holm outcome of no row, and reasoning before the answer line widens no pair in any domain. Multiple-choice scores taken from the first token are therefore safer read at an answer line.

Pairwise sensitivity under tool use. Against reasoning without tools, the evaluator’s baseline for an agent, the plaintext loop widens the spread of 37 of 60 molecule pairs and of a quarter to two fifths of the pairs in each other domain, and narrows two pairs in all (Table 3). The median change of a pair’s spread is 0.12 in molecules, 0.08 in tables, 0.06 in music, 0.04 in mathematics and 0.03 on graphs, and 0.73 to 0.93 of the pairs move upward (Figure 1b). Against the fixed readout the molecule count is 39 of 60 under the exchange test. The rows share models and items and are not independent detections. The widening is concentrated: on graphs every widened pair includes Llama-3.1-8B, and in mathematics every one includes Qwen3-30B-A3B. Under the supplied report no model’s own accuracy range across wordings exceeds 0.06 in any letter domain, and under tools it reaches 0.24 for Llama-3.1-8B on graphs and 0.36 for Qwen3-30B-A3B in mathematics, so the pairs that widen are the pairs that contain such a model. The sign changes that leaving one wording out produces concentrate on the wording of each domain that lowers call frequencies most, and with ten wordings the range under tools grows over its five-wording value by a median factor of 2.88 in molecules, against 1.12 under the fixed readout.

Ranking reversals and selection loss. A pair’s order reverses when its accuracy difference exceeds 0.05 under one wording and falls below −0.05 under another. Under tools 4 of 15 pairs reverse in music and in molecules and 2 in mathematics, against none under the fixed readout or under reasoning in any domain. One wording gives the same order of a pair as the mean over all five for 0.76 of the pairs in music and 0.85 in molecules under tools, against about 0.9 under the fixed readout. On five further wordings written for three models, 2 of the 3 molecule pairs reverse under tools and none under the fixed readout. Choosing the best model under one wording and scoring it under the other four loses 0.03 of accuracy on average in music under tools and less than 0.01 elsewhere. A leaderboard of tool-using models built on one wording orders some pairs differently depending on the wording chosen, at a small average cost to the model it selects.

## 5 Tool Invocation and Interface Effects

Deterministic reports do not guarantee stable comparisons when the model decides whether to obtain them. With the plain-text loop, the median spread in the letter domains exceeds both the supplied report’s and the fixed readout’s (Figure 1a), and tools are wider than the supplied report in 54 of 60 molecule pairs (Table 3). This section relates the widening to call frequency. It then reads the supplied-report controls and seven interventions on the call before turning to the domains where the tools return intermediate results.

Call frequency and conditional accuracy. In molecules the call share is 0.94 [0.88, 1.02], on graphs 0.75 and in music 0.57, against under 0.1 in tables and mathematics (Table 3; intervals in Section S4). Figure 2 shows why. In molecules a model that calls is right in 0.97 to 1.00 of episodes under every wording, so every model’s accuracy lies close to one line rising with its call frequency, consistent with Equation 2. What the identity does not fix is which models

<table><tr><td></td><td colspan="3">call-rate range (mean call rate)</td><td>pairs wider</td></tr><tr><td>Condition</td><td>Qwen3 8B</td><td>Phi-4</td><td>Llama 8B</td><td>than fixed</td></tr><tr><td>plain-text loop (reference)</td><td>0.43 (0.84)</td><td>0.27 (0.50)</td><td>0.22 (0.89)</td><td>10 of 12</td></tr><tr><td>episode budget ×4</td><td>0.41 (0.85)</td><td>0.25 (0.53)</td><td></td><td>4 of 4</td></tr><tr><td>10% of reports wrong</td><td>0.43 (0.83)</td><td>0.22 (0.53)</td><td>0.23 (0.88)</td><td>8 of 12</td></tr><tr><td>30% of reports wrong</td><td>0.42 (0.84)</td><td>0.27 (0.50)</td><td>0.23 (0.89)</td><td>6 of 12</td></tr><tr><td>50% of reports wrong</td><td>0.42 (0.84)</td><td>0.25 (0.49)</td><td>0.23 (0.89)</td><td>0 of 12</td></tr><tr><td>output clause removed</td><td>0.35 (0.83)</td><td>0.21 (0.68)</td><td>0.08 (0.95)</td><td>7 of 12</td></tr><tr><td>third tool description</td><td>0.34 (0.88)</td><td>0.30 (0.64)</td><td>0.11 (0.98)</td><td>9 of 12</td></tr><tr><td>no-call answer refused once</td><td>0.01 (0.99)</td><td>0.14 (0.77)</td><td>0.21 (0.91)</td><td>1 of 12</td></tr><tr><td>second tool description</td><td>0.07 (0.97)</td><td>0.06 (0.94)</td><td>0.00 (1.00)</td><td>0 of 12</td></tr><tr><td></td><td>Qwen3 8B</td><td>Llama 8B</td><td>Mistral 24B</td><td></td></tr><tr><td>plain-text loop (reference)</td><td>0.43 (0.84)</td><td>0.22 (0.89)</td><td>0.26 (0.90)</td><td>10 of 12</td></tr><tr><td>native interface</td><td>0.00 (1.00)</td><td>0.00 (1.00)</td><td>0.00 (1.00)</td><td>0 of 12</td></tr><tr><td>supplied report, six models</td><td></td><td>no call</td><td></td><td>0 of 60</td></tr></table>

Table 4: Interventions on the call in molecules: each model’s callfrequency range across wordings (mean in parentheses) and the pairs flagged wider than under the fixed readout (exploratory, as in Table 3). Upper block: Qwen3-8B, Phi-4 and Llama-3.1-8B against their plain-text loop; the budget row covers the Qwen3-8B and Phi-4 pairs. Lower block: the native interface for Qwen3-8B, Llama-3.1- 8B and Mistral-Small-24B against the same models’ plain-text loop. A range of 0.00 is a constant frequency; the mean says whether it is one.

move. Gemma-3-12B and Qwen3-30B-A3B call under every wording on graphs, while Llama-3.1-8B calls in nearly every episode under one wording and in fewer than a third under another (the call frequency of every model under every wording is Figure S1 of the supplementary material). A wording that lowers a model’s call frequency costs most where the operator is best.

Supplied-report controls. With the report supplied at the start of the prompt and no call to make, no pair’s spread is wider than under the fixed readout in any of the 150 letterdomain rows, and every model is within 0.01 of its highest accuracy (Figure 2, open markers). The stability does not depend on the ceiling or on the absence of reasoning. When the model reasons before answering with the report in its prompt, its spread is wider than the fixed readout’s in 0 of the 21 rows of three models (three pairs over the seven graph and molecule forms). When 20% of the graph reports and 27% of the molecule reports are replaced by a wrong option, accuracy falls by up to 0.26, and the spread is still wider than the fixed readout’s in 0 of the 105 rows of six models and narrower in 29. In music the supplied report reaches 0.57 to 0.60, far from the ceiling, and never widens. We read the supplied report as a diagnostic of report access, not of what a model knows.

Interface, instruction and report-quality interventions. Table 4 reads seven interventions on the molecule loop by the call-frequency range and the number of widened pairs. The native interface and the second tool description bring every model of their group to a mean call frequency of 0.94 or more with a range of 0.07 or less, and leave no widened pair where the plain-text loop widens 10 of 12 in each group. On graphs the native interface reduces the widened pairs from 4 to 1 of 9 and the call share from 0.75 to 0.25. Refusing a nocall answer once brings Qwen3-8B to a range of 0.01 and the count to 1 of 12, while Phi-4’s range falls from 0.27 to 0.14 and Llama-3.1-8B’s stays. Removing the output clause takes Llama-3.1-8B’s range from 0.22 to 0.08 and leaves Qwen3- 8B’s and Phi-4’s above 0.2, with 7 of 12 pairs wider. A third description leaves every range above 0.1 and 9 pairs wider, so the widening holds under two of the three descriptions and vanishes under the one that makes every model call. Two interventions change the loop without changing the call. A four-fold token budget lowers the share of episodes cut at the budget from 0.12 and 0.22 to 0.01 and 0.02 for Qwen3- 8B and Phi-4 and leaves their ranges in place, so for these two models the missed calls are not truncation. Replacing 10, 30 and 50% of the reports by a wrong option after the call leaves every range in place and takes the widening down step by step to none, which is what Equation 2 gives when the reports’ usefulness after a call shrinks. Sampling at temperature 0.7 keeps Qwen3-8B’s ordering of the wordings by call frequency and its range. The interventions therefore have model- and group-specific effects.

![](images/cbb5fd3400b0e6843136872beab63680d39755fb49788947faa928f3c67ab087.jpg)  
Qwen3-8B Phi-4 Llama-3.1-8B Gemma-3-12B Mistral-Small-24B Qwen3-30B-A3B  
Figure 2: Left: accuracy under tools against call frequency, one point per model and wording, written forms pooled; open markers are the accuracy with the report supplied. Right: the call-frequency range of five Qwen3 models (30B-A3B is a mixture of experts) and, as an open marker, Qwen2.5-72B of the previous generation; the panel does not isolate size.

Model-specific instruction effects. The wordings differ in their output-format clause, and in the letter domains the one that asks for the answer alone lowers call frequencies most. For Llama-3.1-8B the clause accounts for its graph behavior: without it, its graph call-frequency range falls from 0.69 to 0.09 and the widened graph pairs from 4 to 0 of 9. The same holds for Qwen2.5-72B, which calls in at most 0.03 of episodes under that wording and in at least 0.61 under the others, so that its tool spreads lie 0.34 to 0.48 above the fixed readout’s in molecules, and which calls under every wording once the clause is removed. For Qwen3-8B and Phi-4 the clause is a small part. Their ranges fall by less than 0.1, and 7 of 12 molecule pairs stay wider. Their calling responds to the paraphrase crossed with the tool description: moving the description into the system turn changes a model’s per-wording call frequency by up to a half, and the IUPAC name in place of the SMILES lowers Qwen3-8B’s call frequency by a third. Ranked by the call frequency they produce, the five wordings barely agree across the six models (Kendall’s W at most 0.13), so no single wording is the bad one for every model.

Size alone does not order the models either. In the Qwen3 series the 4B and 8B models change their call frequency with the wording and the models from 14B call under almost every wording, while the 72B model of the previous generation has the widest range we measured (Figure 2, right). The series varies architecture as well as size.

Intermediate-result tools and stopping behavior. In tables and mathematics the call share is under 0.1. The share is a covariance ratio: the rest of the variation can lie in which items receive calls, which tool and arguments are chosen, and what the model does with the values. Constant tools widen 32 of 45 table pairs, more than working tools (Table 3). Under constant tools four models keep calling the empty tools until the call budget ends the episode, in 0.17 to 0.68 of table episodes, and removing the episodes that end on the budget or contain a rejected call, a descriptive subset, leaves about half of the table widening and all of the mathematics widening. The native interface cuts Llama-3.1-8B’s budget stops from 0.61 to 0.08 of table episodes, yet Qwen3-8B’s native constant control still ends on the budget in 0.44 of table episodes and 0.37 of mathematics episodes, so the stopping failure persists under the tested native interface. Under that interface the three-pair table widening falls from 6 to 4 of 9, every remaining pair contains Mistral-Small-24B, whose native call frequency in tables ranges from 0.28 to 0.90 across wordings, and in mathematics Qwen3-8B’s native call frequency ranges from 0.24 to 0.84. In the groups tested, native calling therefore removes the detected molecule widening and reduces the graph widening from 4 to 1 of 9 pairs. Where the tools return intermediate results, its effects are model-specific.

## 6 Evaluation Recommendations

The findings translate into four diagnostics for authors of benchmarks and leaderboards that compare models with and without tools.

1. Check the answer-scoring protocol. For multiple-choice items, compare first-token scoring with scoring at an answer line and report the difference. The letter mass alone does not validate the read.

2. Evaluate the same model comparisons under several wordings, and report accuracy, pairwise differences and call frequency together. A changing call frequency is a diagnostic to read beside the comparison, and aggregate frequency is not agreement on which items receive calls.

3. Run the supplied report under the same wordings. A model that is stable with the reports supplied and unstable with tools differs from its rivals in obtaining the reports, which says nothing by itself about what it knows.

4. Report stops at the token or call budget, rejected and repeated calls, and the interface used. These matter most where the tools return intermediate results.

For developers of tool-using models, the target is calling that is consistent and appropriate under equivalent instructions. A constant call frequency alone does not show it.

## 7 Limitations

The spreads are ranges over five wordings, a statistic of this wording set that grows with ten wordings, and five wordings support little inference to new ones. The intervals reflect the sampling of items only. Three models run with 8-bit weights. Mathematics has no full-size repeat, and in tables about half of a model’s predictions change between repeats while pair spreads move by at most 0.036. The tools read a canonical form of the item and, in molecules and graphs, answer the question directly, so the gains are bound to the operators we wrote and the tasks say little about tool discovery or argument construction. Domains entered on their fixedreadout spread and music was exempt, so the five domains are not a sample of tasks and the prevalence of the effect outside them is unknown. The description variants and the native interface were run for three models only, so the plaintext loop remains the protocol of the six-model tables. The call-frequency components are an algebraic attribution of observed accuracy changes, not causal effects of calling. The wordings are in English. With its thinking mode on, Qwen3- 8B reasons instead of calling in most episodes, which we leave to future work. The supplied report is a control rather than an agent. The forced call moves the calling fully for one model only.

## 8 Related Work

Prompt sensitivity and the readout. Accuracy varies by up to 76 points across equivalent prompt formats in the setting of Sclar et al. [2023], and model rankings change across instruction paraphrases in Mizrahi et al. [2023], who call for evaluation over several prompts, and Maia Polo et al. [2024] estimate the distribution over many templates from a few evaluations. Wang et al. [2024] report that the firsttoken probability over option letters often does not match the text answer of instruction-tuned models, and Alzahrani et al. [2024] and Zheng et al. [2023] show that option order and answer selection move leaderboard rankings. Hua et al. [2025] find that heuristic answer extraction makes much of the reported prompt sensitivity and that an LLM judge restores the agreement of rankings across templates. Our readout contrast changes the extraction protocol itself, from the first token to an answer line, and we ask the same question of the tool loop.

Written form of the input. The written form of the input moves accuracy on basic graph tasks [Fatemi et al., 2023], on molecule tasks, where no text representation is best on every task [Raja et al., 2026; Runcie et al., 2025], and on written scores [Zhao et al., 2025; Spanio et al., 2026].

Tool use and the decision to call. ReAct interleaves reasoning and actions [Yao et al., 2022], program-aided models execute a written program [Gao et al., 2022], and Toolformer trains a model to decide when to call [Schick et al., 2023]. Tools help a chemistry agent on structure tasks [Yu et al., 2024] and a music controller under one wording [Wang et al., 2025]. The decision to call is an evaluation object of its own: When2Call scores when to call, ask or decline [Ross et al., 2025], Wu et al. [2026] compare no-tool, always-tool and self-decided calls, and Cheng et al. [2026] and Sun et al. [2026] probe whether a model knows a call is needed before it acts. Zou et al. [2026] learn the boundary from counterfactual margins. Robustness work perturbs the tools [Ye et al., 2024], their descriptions, which move tool selection by an order of magnitude [Faghih et al., 2025], or the stages of the tool pipeline [Zheng et al., 2026]. Rephrasings of the user’s request lower the match of generated function calls with the reference call, with much of the loss in parameter values [Rabinovich and Anaby Tavor, 2025], query paraphrases are among the perturbations that tool-use agents withstand best in a simulated benchmark [Zhou et al., 2026], and indirect requests make email agents omit actions [Chen et al., 2026]. We measure what the wording does to call frequency inside an executed loop and to the comparison of two models, against controls that supply the report without a call.

## 9 Conclusion

We find that tools raise the accuracy of language models where a program answers the question and widen the wording sensitivity of pairwise comparisons against reasoning without tools. In our plain-text loop, molecule accuracy follows call frequency, and supplied reports leave the comparisons at their most stable. In molecules, native calling and one tool description remove the detected widening in the groups tested, while other interventions have model-specific effects. Sensitivity persists on graphs and where tools return intermediate results under the tested native interface. We hope that evaluations of tool-using models will report accuracy, call frequency, stopping behavior and comparison stability under several wordings.

## References

[Abdin et al., 2024] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric

Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

[Alzahrani et al., 2024] Norah Alzahrani, Hisham Abdullah Alyahya, Yazeed Alnumay, Sultan Alrashed, Shaykhah Alsubaie, Yusef Almushaykeh, Faisal Mirza, Nouf Alotaibi, Nora Altwairesh, Areeb Alowisheq, M Saiful Bari, and Haidar Khan. When benchmarks are targets: Revealing the sensitivity of large language model leaderboards. arXiv preprint arXiv:2402.01781, 2024.

[Chen et al., 2021] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. FinQA: A dataset of numerical reasoning over financial data. arXiv preprint arXiv:2109.00122, 2021.

[Chen et al., 2026] Feng Chen, Ritam Dutt, Atnaz Taheri, and Alex Williams. Lost in the request: How communication variation disrupts retrieval and action in email agents. arXiv preprint arXiv:2610.02627, 2026.

[Cheng et al., 2026] Yize Cheng, Chenrui Fan, Mahdi Jafari-Raviz, Keivan Rezaei, and Soheil Feizi. Model-adaptive tool necessity reveals the knowing-doing gap in LLM tool use. arXiv preprint arXiv:2605.14038, 2026.

[Cobbe et al., 2021] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[Cuthbert and Ariza, 2010] Michael Scott Cuthbert and Christopher Ariza. music21: A toolkit for computer-aided musicology and symbolic music data. In Proceedings of the International Society for Music Information Retrieval Conference, pages 637–642, Utrecht, The Netherlands, 2010.

[Faghih et al., 2025] Kazem Faghih, Wenxiao Wang, Yize Cheng, Siddhant Bharti, Gaurang Sriramanan, Sriram Balasubramanian, Parsa Hosseini, and Soheil Feizi. Tool preferences in agentic LLMs are unreliable. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20954–20969, Suzhou, China, 2025.

[Fatemi et al., 2023] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large language models. arXiv preprint arXiv:2310.04560, 2023.

[Gao et al., 2022] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: Program-aided language models. arXiv preprint arXiv:2211.10435, 2022.

[Gemma Team, 2025] Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

[Hendrycks et al., 2021] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. arXiv preprint arXiv:2103.03874, 2021.

[Hua et al., 2025] Andong Hua, Kenan Tang, Chenhe Gu, Jindong Gu, Eric Wong, and Yao Qin. Flaw or artifact? Rethinking prompt sensitivity in evaluating LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 19889–19899, Suzhou, China, 2025.

[Lightman et al., 2023] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

[Llama Team, AI @ Meta, 2024] Llama Team, AI @ Meta. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[Maia Polo et al., 2024] Felipe Maia Polo, Ronald Xu, Lucas Weber, Mírian Silva, Onkar Bhardwaj, Leshem Choshen, Allysson Flavio Melo de Oliveira, Yuekai Sun, and Mikhail Yurochkin. Efficient multi-prompt evaluation of LLMs. arXiv preprint arXiv:2405.17202, 2024.

[Mistral AI, 2025] Mistral AI. Mistral-small-24b-instruct-2501. Model card, https://huggingface.co/mistralai/ Mistral-Small-24B-Instruct-2501, 2025.

[Mizrahi et al., 2023] Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, and Gabriel Stanovsky. State of what art? A call for multi-prompt LLM evaluation. arXiv:2401.00595, 2023.

[Qwen Team, 2025] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[Rabinovich and Anaby Tavor, 2025] Ella Rabinovich and Ateret Anaby Tavor. On the robustness of agentic function calling. In Proceedings of the 5th Workshop on Trustworthy NLP (TrustNLP 2025), pages 298–304, Albuquerque, New Mexico, 2025.

[Raja et al., 2026] Arun Raja, Garrett M. Morris, and Kian Ming A. Chai. Rethinking molecular text representations for LLMs: An empirical study. arXiv preprint arXiv:2606.03057, 2026.

[Ross et al., 2025] Hayley Ross, Ameya Sunil Mahabaleshwarkar, and Yoshi Suhara. When2Call: When (not) to call tools. arXiv preprint arXiv:2504.18851, 2025.

[Runcie et al., 2025] Nicholas T. Runcie, Charlotte M. Deane, and Fergus Imrie. Assessing the chemical intelligence of large language models. arXiv preprint arXiv:2505.07735, 2025.

[Schick et al., 2023] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.

[Sclar et al., 2023] Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. Quantifying language models sensitivity to spurious features in prompt design or: How I learned to start worrying about prompt formatting. arXiv:2310.11324, 2023.

[Spanio et al., 2026] Matteo Spanio, Mohammad Torabi, Andrea Poltronieri, and Antonio Rodà. Can LLMs understand LilyPond? A benchmark for symbolic music generation and understanding. arXiv preprint arXiv:2606.08722, 2026.

[Sun et al., 2026] Chung-En Sun, Linbo Liu, Ge Yan, Zimo Wang, and Tsui-Wei Weng. LLM agents already know when to call tools – even without reasoning. arXiv preprint arXiv:2605.09252, 2026.

[Wang et al., 2024] Xinpeng Wang, Chengzhi Hu, Bolei Ma, Paul Röttger, and Barbara Plank. Look at the text: Instruction-tuned language models are more robust multiple choice selectors than you think. arXiv preprint arXiv:2404.08382, 2024.

[Wang et al., 2025] Boyang Wang, Yash Vishe, Xin Xu, Zachary Novack, Xunyi Jiang, Julian McAuley, and Junda Wu. CSyMR: Benchmarking compositional music information retrieval in symbolic music reasoning. arXiv preprint arXiv:2601.11556, 2025.

[Wu et al., 2026] Qinyuan Wu, Seungeon Lee, Soumi Das, Mahsa Amani, Arijit Nag, Krishna Gummadi, Abhilasha Ravichander, and Muhammad Bilal Zafar. To call or not to call: A framework to assess and optimize LLM tool calling. arXiv preprint arXiv:2605.00737, 2026.

[Yao et al., 2022] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

[Ye et al., 2024] Junjie Ye, Yilong Wu, Songyang Gao, Caishuang Huang, Sixian Li, Guanyu Li, Xiaoran Fan, Qi Zhang, Tao Gui, and Xuanjing Huang. RoTBench: A multi-level benchmark for evaluating the robustness of large language models in tool learning. arXiv preprint arXiv:2401.08326, 2024.

[Yu et al., 2024] Botao Yu, Frazier N. Baker, Ziru Chen, Garrett Herb, Boyu Gou, Daniel Adu-Ampratwum, Xia Ning, and Huan Sun. ChemToolAgent: The impact of tools on language agents for chemistry problem solving. arXiv preprint arXiv:2411.07228, 2024.

[Zhao et al., 2025] Jiahao Zhao, Yunjia Li, Wei Li, and Kazuyoshi Yoshii. ABC-Eval: Benchmarking large language models on symbolic music understanding and instruction following. arXiv preprint arXiv:2509.23350, 2025.

[Zheng et al., 2023] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. arXiv preprint arXiv:2309.03882, 2023.

[Zheng et al., 2026] YiShan Zheng, Yuan Wu, and Yi Chang. ToolRobustBench: Stage-wise perturbation evaluation and

failure diagnosis for tool-calling agents. arXiv preprint arXiv:2608.23635, 2026.

[Zhou et al., 2026] Xiaolin Zhou, Aojie Yuan, Zheng Luo, Zipeng Ling, Xixiao Pan, Yicheng Gao, Haiyue Zhang, Jiate Li, Shuli Jiang, Prince Zizhuang Wang, Zixuan Zhu, Jinbo Liu, Ryan A. Rossi, Hua Wei, and Xiyang Hu. When simulation lies: A sim-to-real benchmark and domainrandomized RL recipe for tool-use agents. arXiv preprint arXiv:2605.11928, 2026.

[Zou et al., 2026] Wenhao Zou, Xianglong Liu, Wendong Bi, Hanjie Wang, Simin Zhao, and Zhi Gong. CoBRA: Learning tool-use boundaries via counterfactual margins. arXiv preprint arXiv:2609.00967, 2026.

## Supplementary Material

## S1 The wordings

Each domain has five wordings of the instruction. A word ing is a template. The placeholders <question>, <input>, <form> and <options> are filled with the item’s question, its written input, the name of the written form shown and the four options (letter domains), and the number domains have no options. The same five templates serve every written form of a domain. Under tools, the tool description printed after each domain’s wordings precedes the wording unchanged. Under constant tools the description is the same and every tool answers “no information”. Under reasoning without tools, the sentence printed last precedes the wording in the letter domains. The wordings differ in their output-format clause, and the variant without that clause, used in Section S4.8, ends every wording with the same sentence.

## Symbolic music

p1.   
<question>   
The score is written in <form>.

<input>

Options: <options>

Answer with the letter of the correct option only.

## p2.

Read the following score, given in <form>, and answer the question.

Score: <input>

Question: <question> <options>

Reply with a single letter.

p3.   
Here is a piece of music in <form>: <input>

<question> Choose one of the options below. <options>

Respond with only the option letter.

p4. Question: <question> Notation: <form>

<input>

<options>

The answer is one of A, B, C, D. Give the letter.

## p5.

You are given a musical score in <form>. <question>

<input>

Candidates: <options>

Output just the letter of the best candidate.

## Tool description (before every wording under tools and constant tools).

Tools are available that read the score for you. To use one, write a line of the form CALL <name>() with <name> one of: count\_bars (the number of bars), time\_signature (the time signature written in the score, or none), key\_estimate (the key signature written in the score, if any, and the key estimated from the notes), pitch\_histogram (the six most frequent pitch names with their counts), last\_bar\_notes (the notes of the last bar). After each call you receive a line REPORT <name>: <result>. You may make at most 6 calls. When you are ready, write a line of the form ANSWER: <letter>.

## Graphs

g1. <question>

<input>

Options: <options>

Answer with the letter of the correct option only.

g2.   
Read the following graph description and answer the question.

Graph: <input>

Question: <question> <options>

Reply with a single letter.

g3.   
Here is a graph: <input>

<question> Choose one of the options below.

<options>

Respond with only the option letter.

g4. Question: <question>

<input>

<options>

The answer is one of A, B, C, D. Give the letter.

## g5.

You are given an undirected graph. <question>

<input>

Candidates: <options>

Output just the letter of the best candidate.

## Tool description (before every wording under tools and constant tools).

Tools are available that read the graph for you. To use one, write a line of the form CALL <name>(<arguments>) with one of: count\_nodes() (the number of nodes), count\_edges() (the number of edges), degree(<node>) (the number of edges at a node), neighbors(<node>) (the nodes connected to a node), has\_edge(<node>, <node>) (yes or no), nodes\_on\_cycles() (the nodes that lie on at least one cycle). Nodes are named as in the graph description. After each call you receive a line REPORT <name>: <result>. You may make at most 6 calls. When you are ready, write a line of the form ANSWER: <letter>.

## Molecules

y1.   
<question>   
The molecule is written as <form>.

<input>

Options: <options>

Answer with the letter of the correct option only.

## y2.

Read the following molecule, given as <form>, and answer the question.

Molecule: <input>

Question: <question> <options>

Reply with a single letter.

y3.   
Here is a molecule written as <form>: <input>

<question> Choose one of the options below. <options>

Respond with only the option letter.

y4. Question: <question> Notation: <form>

<input>

<options>

The answer is one of A, B, C, D. Give the letter.

## y5. y5.

You are given a molecule as <form>. <question>

<input>

Candidates: <options>

Output just the letter of the best candidate.

## Tool description (before every wording under tools and constant tools).

Tools are available that read the molecule for you. To use one, write a line of the form CALL <name>() with <name> one of: heavy\_atom\_count (the number of non-hydrogen atoms), ring\_count (the number of rings), molecular\_weight (in g/mol), molecular\_formula, h\_bond\_donors and h\_bond\_acceptors (Lipinski’s counts). After each call you receive a line REPORT <name>: <result>. You may make at most 6 calls. When you are ready, write a line of the form ANSWER: <letter>.

## Mathematics (MATH-500)

## x1.

<question>

Solve the problem. Give the final answer as a number on the last line, in the form ANSWER: <number>.

## x2.

Read the following math problem and work out the answer.

Problem: <question>

End with one line of the form ANSWER: <number>, where <number> is the final numeric answer.

## x3.

Here is a math problem: <question>

Work through it step by step, then write the final numeric answer on its own line as ANSWER: <number>.

## x4.

Question: <question>

The answer is a single number. Compute it and finish with the line ANSWER: <number>.

## x5.

You are given a math problem. <question>

Find the number it asks for. Your last line must read ANSWER: <number> and contain only the final number.

## Tool description (before every wording under tools and constant tools).

Two tools are available. CALL calculator(<expression>) evaluates an arithmetic expression over numbers with + - <sub>\*</sub> / ^ ( ) and sqrt, pi, exp, log, sin, cos, tan. CALL solve(<equation>, <variable>) solves an equation or a system of equations separated by ; for the variable or variables named (for example solve(x^2 - 5<sub>\*</sub>x + 6 = 0, x)). After each call you receive a line REPORT <name>: <result>. You may make at most 6 calls. When you are ready, write a line of the form ANSWER: <number>.

## Tables (FinQA)

## t1.

<question>

<input>

Answer the question from the table. Give the final answer as a number on the last line, in the form ANSWER: <number>.

## t2.

Read the following table and answer the question.

Table:

<input>

Question: <question>

End with one line of the form ANSWER: <number>, where <number> is the final numeric answer.

## t3.

Here is a table from a financial report: <input>

<question> Work through it step by step, then write the final numeric answer on its own line as ANSWER: <number>.

t4. Question: <question>

<input>

The answer is a single number. Compute it and finish with the line ANSWER: <number>.

## t5.

You are given a table. <question>

<input>

Find the number the question asks for. Your last line must read ANSWER: <number> and contain only the final number.

## Tool description (before every wording under tools and constant tools).

Tools are available that read the table for you. CALL rows() lists the row labels; CALL lookup(<row label>) gives the values of one row with their column headers; CALL column(<column header>) gives the values of one column by row; CALL calculator(<expression>) evaluates an arithmetic expression over numbers and + - <sub>\*</sub> / ( ). After each call you receive a line REPORT <name>: <result>. You may make at most 6 calls. When you are ready, write a line of the form ANSWER: <number>.

## Reasoning without tools (letter domains; before every wording).

You may write your reasoning first. When you are ready, write a line of the form ANSWER: <letter>.

## S2 Items, tools, models and harness

Items. Molecules: 504 four-option items on compounds drawn from PubChem (2,105 of 3,000 random compounds pass size and element filters), 84 per property: heavy-atom count, ring count, molecular weight, molecular formula, and Lipinski hydrogen-bond donors and acceptors, with gold answers computed by RDKit. The InChI states the molecular formula, so formula items under InChI can be answered by lookup (Section S4.5). Graphs: 504 four-option items following the tasks of Fatemi et al. (node count, edge count, node degree, edge existence, connected nodes and cycle membership, 84 per task) on random graphs of 5 to 20 nodes from six families (Erdos–Rényi, Barabási–Albert, stochastic block,˝ star, path and complete), up to nine questions per graph. Music: the 386 items of ABC-Eval that fit a common token budget in ABC, MusicXML and a note-event list, less 25 next-bar items whose options repeat a text in one notation. Tables: 500 test questions of FinQA, sampled from the 692 whose supporting facts lie in the table and whose answer is a number, shown without the report text. Mathematics: the 336 MATH-500 problems whose answer is a plain number.

Scoring. A letter answer is correct when it is the gold option, and an answer that cannot be parsed is incorrect. A mathematics answer is correct when it equals the gold number. A table answer is correct when it lies within 1% of the executed answer or within 0.005 of it, or when it has at least one decimal (two when the answer is below one) and equals the answer rounded to the decimals the model wrote. For a question whose gold value is a ratio (its FinQA program divides, the question names a percentage, rate, margin, portion, return, growth, yield or share, or the table lists percentages), the same rules apply to the value times 100. An integer never matches through rounding. Under a strict rule without the percentage conversion and the tolerance, every contrast of the table rows keeps its direction and 78% of the Holm outcomes stay. 34 FinQA items have an executed answer that disagrees with the answer in their text, and removing them moves the wider counts of the table rows by at most 3.

Inclusion criterion. A domain enters only if, on its first 200 items under the fixed readout, at least one of Qwen3-8B and Phi-4 scores at most 0.90 and the wording spread of the pair is at least 0.02 under one written form. GSM8K fails it, with accuracies of 0.95 and 0.97. Music, where the protocol was developed, is exempt.

Tools. Molecules: six RDKit operators (2026.03) for the six properties. Graphs: six NetworkX operators (3.7) for the number of nodes and edges, the degree and the neighbors of a node, whether two nodes share an edge, and the nodes on a cycle. Music: five music21 operators (10.5) for the number of bars, the written time signature, the written key signature with the key estimated from the notes, the six most frequent pitch names, and the notes of the last bar. Run alone, they agree with the gold option on 0.99 of bar-count items, 0.63 of key items and 0.53 of time-signature items. Tables: three operators for the row labels, one row and one column, and a calculator. Mathematics: a calculator and an equation solver built on SymPy 1.14. A call that names no tool, or passes an argument the tool cannot use, receives an error line and counts as a rejected call. The tool description is one text per domain, printed before the wording.

Harness. The loop generates at most 64 tokens per step in the letter domains and 256 in the number domains, within an episode budget of 256 tokens in music and molecules, 384 on graphs, 1,024 in tables and 1,536 in mathematics, and ends after six calls. An episode that reaches its budget without an answer line receives an appended answer line, which is read as under the fixed readout. Under the native interface the same tools are declared through the model’s chat template, a call is a structured turn, the step budget is 128 tokens, and a model that writes prose before its call block is parsed for the block anywhere in the turn. Under the supplied report, the report that the loop would print for the item’s question type is appended as one line, and items whose question type no operator answers receive no line. Under the noisy supplied report, a fixed share of items, chosen by a hash so that the same items are corrupted under every wording and form, receive the report of a wrong option. The models run on vLLM 0.31 with greedy decoding, Qwen3 with thinking disabled through the chat template. Table S1 lists the revisions.

## S3 Statistical procedure

Intervals are 95% percentile intervals from 1,000 bootstrap resamples over items, in which one resample drives both models, all wordings and both conditions of a contrast. The two-sided p-value of a contrast is twice the share of resamples on the other side of zero, floored at one over the number of resamples, and Holm correction runs across the model pairs of one domain, written form and contrast. Under a null of equal, nonzero wording effects (one condition’s rows split at random into two pseudo-conditions within each item, three replicates), this procedure rejects at 0.05 in 0.025 of cases overall, at most 0.04 in every contrast but one, and 0.12 for tools against the fixed readout in molecules. The withinitem exchange test used for that row permutes the two conditions’ outcome vectors within each item (both models’ vectors over the five wordings swap condition labels together) over 4,000 draws and rejects single comparisons at the 0.05 level in 0.067 of 180 simulated contrasts under the same null (12 of 180; a binomial standard error of 0.02), and in none after Holm. The simulated null splits one condition’s rows into two pseudo-conditions of equal accuracy level, so neither procedure is shown to hold the nominal rate when the two conditions differ in level, as tools and the fixed readout do. The counts are detections and the raw differences carry the size (Table S2). It gives 39 wider and 1 narrower pairs where the bootstrap gives 34 and 2, and in every other contrast its counts lie within 16% of the bootstrap’s with no reversal. A permutation test under the null of no wording effect at all is calibrated for its own null and rejects more often, because the fixed readout has a wording effect of its own. The spread is a range, so the per-pair table also carries the standard deviation of the difference over wordings, and every pair that is wider under the range is wider under the standard deviation. Up to nine graph questions share one graph and up to four table questions share one table. A cluster bootstrap over graphs and tables moves a few counts on graphs and in tables and changes no direction.

## S4 Supporting results

Counts of wider or narrower pairs are taken after Holm correction as in the main text. Figure S1 is the call frequency of every model under every wording, referred to from the main text as Figure S1.

<table><tr><td>Model</td><td>Repository</td><td>Revision (first 12)</td><td>Weights</td></tr><tr><td>Qwen3-8B</td><td>Qwen/Qwen3-8B</td><td>b968826d9c46</td><td>bf16</td></tr><tr><td>Phi-4</td><td>microsoft/phi-4</td><td>2db69c1c3e91</td><td>8-bit float</td></tr><tr><td>Llama-3.1-8B</td><td>meta-llama/Llama-3.1-8B-Instruct</td><td>0e9e39f249a1</td><td>bf16</td></tr><tr><td>Gemma-3-12B</td><td>google/gemma-3-12b-it</td><td>96b6f1eccf38</td><td>bf16</td></tr><tr><td>Mistral-Small-24B</td><td>mistralai/Mistral-Small-24B-Instruct-2501</td><td>9527884be6e5</td><td>8-bit float</td></tr><tr><td>Qwen3-30B-A3B</td><td>Qwen/Qwen3-30B-A3B-Instruct-2507-FP8</td><td>5a5a776300a4</td><td>8-bit float</td></tr><tr><td>Qwen2.5-72B</td><td>Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8</td><td></td><td>8-bit integer</td></tr></table>

Table S1: The models of the grid and the 72B model, with the Hugging Face revision used.

![](images/a09ac8a86c7eaee2054bc6ee38ec8225366a2b41d410b712f2fb7f818c8d31a9.jpg)  
Figure S1: Share of tool episodes with at least one call, by wording, one line per model and one panel per domain, under the plain-text loop. The wordings are numbered as in Section S1, where the third wording of each letter domain asks for the answer alone and the third table wording asks for step-by-step work before the answer. A flat line means a stable aggregate call frequency; it does not establish that the same items receive calls under every wording.

<table><tr><td>Domain</td><td>Contrast</td><td>Rows</td><td>Bootstrap</td><td>Exchange</td></tr><tr><td>Molecules</td><td>tools vs. fixed</td><td>60</td><td>34/2</td><td>39/1</td></tr><tr><td>Molecules</td><td>tools vs. reasoning</td><td>60</td><td>37/2</td><td>41/1</td></tr><tr><td>Molecules</td><td>fixed vs. first token</td><td>60</td><td>0/5</td><td>0/11</td></tr><tr><td>Graphs</td><td>tools vs. reasoning</td><td>45</td><td>12/0</td><td>14/0</td></tr><tr><td>Graphs</td><td>fixed vs. first token</td><td>45</td><td>0/36</td><td>1/39</td></tr><tr><td>Music</td><td>tools vs. reasoning</td><td>45</td><td>17/0</td><td>19/0</td></tr><tr><td>Tables</td><td>tools vs. reasoning</td><td>45</td><td>15/0</td><td>14/0</td></tr><tr><td>Tables</td><td>constant vs. reasoning</td><td>45</td><td>32/0</td><td>37/0</td></tr><tr><td>Mathematics</td><td>tools vs. reasoning</td><td>15</td><td>5/0</td><td>5/0</td></tr></table>

Table S2: Detected wider/narrower pairs under the item bootstrap and under the within-item exchange test, after Holm correction, for the main contrasts. No pair changes direction between the two.

## S4.1 The call term by comparison

Figure S2 shows the call term behind the call shares of the main text. The shares with their bootstrap intervals are 0.94 [0.88, 1.02] in molecules, 0.75 [0.65, 0.81] on graphs, 0.57 [0.44, 0.62] in music, 0.06 [−0.01, 0.14] in tables and 0.07 [0.03, 0.11] in mathematics. The share is stable under the conventions of the split: with the unobserved strata taking the reference value, or either order of the two factors alone, the molecule share lies between 0.89 and 0.98, and the median over pairs is 0.93. The call rate counts whether any tool was called. The share of item-and-form units on which a model switches between calling and not calling across wordings reaches 0.85 for Llama-3.1-8B on graphs and 0.45 for Qwen3-8B in molecules, and 0.38 for Mistral-Small-24B on graphs whose call-rate range is 0.06, so switching exceeds the range where switches cancel.

## S4.2 The forced call

The forced call was run for Qwen3-8B, Phi-4 and Llama-3.1- 8B. It raises Qwen3-8B’s call rate in molecules to 0.99 or more under every wording. The other models often answer without a call after the one refusal, so their call-rate ranges shrink without closing: Llama-3.1-8B on graphs from 0.69 to 0.12, Phi-4 in molecules from 0.27 to 0.14. Against tools, the forced call narrows 10 of 12 molecule comparisons, 5 of 9 graph comparisons, 3 of 9 music comparisons and no table comparison. It widens none.

## S4.3 The decision item by item

For each model under tools we compare items whose decision to call changes across the five wordings with items whose decision does not. In molecules, for the four models with a call-rate range above 0.1, the share of items whose correctness also changes across wordings is 0.58 to 0.63 among the first and 0.005 to 0.28 among the second. On graphs the ratio of the two shares is about 8 for Qwen3-8B and Mistral-Small-24B and 1.5 for Phi-4 and Llama-3.1-8B.

## S4.4 Option position and permuted options

Under the fixed readout, accuracy varies with the position of the gold letter, and permuting the options rotates each item’s gold letter while the item stays the same. Permuting moves accuracy by at most 0.04 for any model and domain and changes the Holm outcome of none of the 150 comparisons of the letter domains, so the position effect averages out over the balanced gold positions.

![](images/5331d9d46cd5866d54e4e3641fee67976b27fa0ccc2f5b672d478825f17de21e.jpg)  
Figure S2: Under tools, each point shows how far a model pair’s accuracy difference and its call-rate difference move from the pair’s mean at one wording and written form. The line is the least-squares fit through the origin, and the panel label gives the share of the variance of pair differences that the call term carries.

<table><tr><td></td><td>Qwen3 8B</td><td>Phi-4</td><td>Llama 8B</td></tr><tr><td>description of the main text</td><td>0.84 (0.43)</td><td>0.50 (0.27)</td><td>0.89 (0.22)</td></tr><tr><td>second description</td><td>0.97 (0.07)</td><td>0.94 (0.06)</td><td>1.00 (0.00)</td></tr><tr><td>third description</td><td>0.88 (0.34)</td><td>0.64 (0.30)</td><td>0.98 (0.11)</td></tr></table>

Table S3: Call rate in molecules under three wordings of the tool description, with the range across the five wordings in parentheses.

## S4.5 The written form of molecules

Under the fixed readout, the InChI raises molecule accuracy over the PubChem SMILES by 0.06 [0.03, 0.10] for Qwen3- 8B, 0.01 [−0.03, 0.04] for Phi-4 and 0.05 [0.02, 0.08] for Llama-3.1-8B. The InChI states the molecular formula, and on the formula items alone the lead is 0.27 to 0.35. Without them it is −0.05 to 0.05. The randomized SMILES moves no model’s accuracy by more than 0.03 from the PubChem SMILES in any condition. Under tools the IUPAC name lowers the call rate and with it the accuracy, by 0.16, 0.09 and 0.08 for the three models. An audit found 12 InChI strings that denote a tautomer of the SMILES. Removing those items moves the InChI lead by at most 0.005 and changes two Holm outcomes with no reversal.

## S4.6 Ten wordings and held-out wordings

For Qwen3-8B, Phi-4 and Llama-3.1-8B on graphs and in molecules, five further wordings were written for the same requests. With all ten, the range of a comparison under tools grows over its five-wording range by a median factor of 1.60 on graphs and 2.88 in molecules, against 1.11 and 1.12 under the fixed readout, and no contrast changes sign. On the five further wordings alone, 2 of 3 molecule pairs reverse their order beyond 0.05 under tools and none under the fixed readout. Leaving one of the five wordings out changes the sign of a contrast in 0.10 of all cells, most of them rows without a Holm result, and in 36 Holm-significant rows, concentrated on the wording of each domain that lowers call rates most. With items and wordings resampled together, few widened rows keep an interval above zero (6 of 37 in molecules), so the spreads are statistics of this wording set.

## S4.7 The tool description and its position

Table S3 gives the call rate in molecules under three wordings of the tool description. Tools are wider than the fixed readout in 10, 0 and 9 of 12 pairs under the three, and over all fifteen combinations of a wording and a description in 12 of 12 pairs. On graphs the same three models call in 0.81, 0.96 and 0.92 (Qwen3-8B), 0.45, 0.92 and 0.77 (Phi 4) and 0.79, 0.87 and 0.87 (Llama-3.1-8B) of episodes under the three descriptions. Moving the description into the system turn changes the largest per-wording call rate by 0.24 to 0.51 for each model in each domain, raising Qwen3-8B’s call rate in molecules to 1.00 under every wording and widening Llama-3.1-8B’s range there from 0.22 to 0.52.

<table><tr><td></td><td>Qwen3 8B</td><td>Phi-4</td><td>Llama 8B</td><td>Qwen2.5 72B</td></tr><tr><td>graphs, original wordings</td><td>0.10</td><td>0.34</td><td>0.69</td><td>0.99</td></tr><tr><td>graphs, clause removed</td><td>0.21</td><td>0.31</td><td>0.09</td><td>0.00</td></tr><tr><td>molecules, original wordings</td><td>0.43</td><td>0.27</td><td>0.22</td><td>0.97</td></tr><tr><td>molecules, clause removed</td><td>0.35</td><td>0.21</td><td>0.08</td><td>0.00</td></tr></table>

Table S4: Range of the call rate across wordings under tools, with the wordings’ output-format clause and with every wording ending in the same sentence.

## S4.8 The output clause

Table S4 gives the call-rate ranges with and without the output-format clause. Without it, tools are wider than the fixed readout in 0 of 9 graph pairs (4 with the clause) and 7 of 12 molecule pairs (10) for the three one-card models, and the fixed readout’s own spreads move by up to 0.05. Qwen2.5- 72B calls in 0.61 to 1.00 of episodes under four wordings and in 0.00 to 0.03 under the wording that asks for the bare letter, with the clause, and in 1.00 under every wording without it, where its tool spreads fall to the fixed readout’s in every written form.

## S4.9 The native interface by domain

Table S5 gives each model’s call-rate range under both interfaces. Over the three pairs of these models, with the same cohort’s plain-text loop as the reference, tools are wider than the fixed readout in 10 of 12 molecule pairs under the text loop and 0 under the native interface (3 narrower), in 4 of 9 graph pairs and 1 (2 narrower), and in 6 of 9 music pairs and 0. In tables, against reasoning, the counts are 6 of 9 and 4, and for the native constant control 7 and 4. Every wider native pair contains Mistral-Small-24B. In mathematics the native loop widens 2 of 3 pairs against reasoning where the text loop widens none, and both contain Qwen3-8B. The native interface cuts the share of Llama-3.1-8B’s table episodes that end on the call budget from 0.61 to 0.08 and its mathematics share from 0.71 to 0.20, while Qwen3-8B’s native constant control ends on the budget in 0.44 of table and 0.37 of math ematics episodes. Rejected calls stay between 0.19 and 0.38 per call. Under the native interface, Mistral-Small-24B’s accuracy in tables rises from 0.46 to 0.63 and its music accuracy range across wordings is 0.04, close to the product of its callrate range and the gap between the given answer and the fixed readout. The native graph call share is 0.25 against 0.75 under the text loop.

<table><tr><td></td><td colspan="2">Qwen3 8B</td><td colspan="2">Llama 8B</td><td colspan="2">Mistral 24B</td></tr><tr><td>Domain</td><td>text</td><td>native</td><td>text</td><td>native</td><td>text</td><td>native</td></tr><tr><td>Molecules</td><td>0.43</td><td>0.00</td><td>0.22</td><td>0.00</td><td>0.26</td><td>0.00</td></tr><tr><td>Graphs</td><td>0.10</td><td>0.01</td><td>0.69</td><td>0.01</td><td>0.06</td><td>0.07</td></tr><tr><td>Music</td><td>0.34</td><td>0.10</td><td>0.48</td><td>0.01</td><td>0.24</td><td>0.23</td></tr><tr><td>Tables</td><td>0.36</td><td>0.01</td><td>0.45</td><td>0.00</td><td>0.29</td><td>0.63</td></tr><tr><td>Mathematics</td><td>0.09</td><td>0.60</td><td>0.06</td><td>0.00</td><td>0.10</td><td>0.12</td></tr></table>

Table S5: Range of the call rate across wordings under the plain-text loop and under the native interface.

## S4.10 Operator coverage on graphs

With only the two intermediate graph operators (a node’s neighbors and whether two nodes share an edge), the three one-card models’ accuracy under tools falls from 0.97, 0.92 and 0.83 to 0.87, 0.82 and 0.61 (Qwen3-8B, Phi-4, Llama-3.1-8B) against 0.73, 0.80 and 0.63 under the fixed readout, the pooled call share falls from 0.78 to −0.36, tools are wider than the fixed readout in 0 of 9 pairs (4 with the full operators) and the constant control in 3 (1).

## S4.11 Wrong reports and the noisy supplied report

With 10, 30 and 50% of the molecule reports replaced by a wrong option, each model’s accuracy falls by about the corrupted share times its call rate (ratios 0.57 to 1.08), its callrate range moves by at most 0.04, and the call share over the three pairs is 1.06, 0.65 and 0.11. The supplied report with reasoning covers the three pairs of Qwen3-8B, Phi-4 and Llama-3.1-8B over the three graph and four molecule forms (21 rows), and the noisy supplied report the fifteen pairs of six models over the same seven forms (105 rows). The supplied report with about a quarter of the reports wrong lowers accuracy by 0.08 to 0.16 on graphs and 0.23 to 0.26 in molecules for the six models, is wider than the fixed readout in 0 of 105 pairs and narrower in 29, and tools are wider than it in 54 of 60 molecule and 15 of 45 graph pairs. Phi-4 overrules the wrong graph report on about half of the corrupted items.

## S4.12 The supplied report with reasoning

When the model reasons before answering with the report in its prompt, Qwen3-8B, Phi-4 and Llama-3.1-8B reach 0.88 to 0.97, their spread is wider than the fixed readout’s in 0 of 21 pairs, and tools are wider than it in 11 of 12 molecule and 7 of 9 graph pairs.

## S4.13 Size

In the Qwen3 family the call-rate range under tools is 0.37 (4B), 0.10 (8B), 0.00, 0.01 and 0.00 (14B, 30B-A3B, 32B) on graphs, 0.01, 0.43, 0.00, 0.00 and 0.00 in molecules and 0.34, 0.34, 0.01, 0.08 and 0.04 in music. Qwen2.5-72B’s is 0.99, 0.97 and 0.98. With the 72B added to the six models, tools are wider than the fixed readout in 24 of the 24 molecule, 12 of the 18 graph and 11 of the 18 music pairs that contain it.

## S4.14 Sampling and thinking mode

Under sampling at temperature 0.7 with three seeds, Qwen3- 8B’s ordering of the five molecule wordings by call rate is the greedy ordering (Spearman 1.00) and its range is 0.40 against 0.43. With the thinking mode of Qwen3-8B on and a 2,048-token thinking budget, the model calls a tool in 0.12 of molecule and 0.18 of graph episodes, against 0.84 and 0.81 with it off, and the budget cuts 0.84 of its thinking under the fixed readout in molecules.

## S4.15 Precision and repeatability

Phi-4, Mistral-Small-24B and Qwen3-30B-A3B run again in bfloat16 move accuracy by at most 0.022 in 59 of 60 model, domain and condition cells and by 0.06 in one (Qwen3-30B-A3B with tools in mathematics), and no contrast changes direction. Repeating three full letter-domain cells changes the predicted option of 0.3% of the rows of a fixed-readout cell (Qwen3-8B, graphs) and of 3.8% and 7.2% of the rows of two tool cells (Qwen3-8B, molecules; Llama-3.1-8B, music), and moves any pair’s spread by at most 0.006, 0.012 and 0.019. Repeating Phi-4’s table cell under tools changes the predicted number of 49% of its rows and moves any pair’s spread by at most 0.036. These shares count changed predictions. A changed prediction can leave correctness unchanged. The smallest widened table difference is 0.076. Two runs of 200 items at batch size one agree on every answer, and each agrees with the batched run on 98.5% of answers. A separate run of the first-token read on the music items agrees with ours on 0.95 to 0.99 of the answers of each of five models.

## S4.16 Predictions stated before the runs

Each control’s reading was written down before its rows were read. The main ones, with their outcome:

• Tools halve the wording spread of every pair in every domain: refuted. Tools widen it for many pairs.

• Parsing failures account for the spread that remains under tools: refuted. The call decision does.

• The supplied report is never wider than the fixed readout, and tools are wider than it in most molecule pairs: held (0 of 150 and 54 of 60).

• The supplied report with reasoning, and with wrong reports, is never wider than the fixed readout: held.

• The native interface shrinks the widening without removing it: refuted in molecules, where it vanishes, and held on graphs. Its call rate exceeds 0.9 under every wording in music: held for two of three models.

• Under the native loop, budget stops fall below 0.10 and rejected calls below 0.01 in tables and mathematics: refuted for most cells.

• A four-fold budget takes Qwen3-8B’s call-rate range below 0.25: refuted (0.41).

• Removing the output clause halves the call-rate ranges: held for Llama-3.1-8B and Qwen2.5-72B, refuted for Qwen3-8B and Phi-4.

• A second tool description keeps the widening: refuted for one description and held for another.

• Wrong reports lower the widening while the call-rate ranges stay: held.

• Ten wordings grow the range by a factor of 1.1 to 1.5: exceeded in molecules (2.88).

• Leaving one wording out changes a contrast’s sign in at most 5% of cells: refuted (10%).

• The Qwen3 decision is wording-stable from 14B: held. A 72B model’s first-token mass is above 0.9 and its tool spreads match the fixed readout’s: the first held, the second refuted.