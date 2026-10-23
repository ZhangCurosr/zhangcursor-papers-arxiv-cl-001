# TT-SAX: A Two-Track Time Series Symbolization for Natural Language Shape Querying

Zhao Tan, Xianglong Bao, Zhipeng Xi, and Ming Jin<sup>(B)</sup>

Grifith University, Australia zhao.tan@griffith.edu.au, xianglong.bao@griffithuni.edu.au, z.xi@griffith.edu.au, ming.jin@griffith.edu.au

Abstract. Natural language querying for time series databases aims to let users find what they need in long histories of data by asking in plain language. Shape queries ask how a series moves. They are the main challenge of this task. Recent work makes them feasible by using symbolization as a bridge between the query and the long history. However, existing symbolizations were designed for compression, similarity indexing, or classification. What a symbolization should provide for natural language querying has not been systematically studied. We therefore examine the definitions of six classical symbolizations against this task and identify three limitations. (1) Direction is missing from the string, (2) direction is measured at too small a scale, and (3) the letters are arbitrary. Three design principles follow. We follow these principles and make a minimal redesign of SAX called TT-SAX (two-track SAX). The level letters stay. Each segment gains a trend letter. It is measured across neighboring segments rather than inside one, and graded to match the adjectives that queries use. We conduct extensive experiments on the public NLQTSBench benchmark, comparing TT-SAX with the six classical symbolizations across six LLMs. TT-SAX outperforms all six on shape queries, and ablations trace its gains back to the three limitations one by one. This paper is the first systematic study of time series symbolization with natural language querying as the design goal. It provides a foundation for future research on this task. Our code is available at https://github.com/Atlamtiz/tt-sax.

Keywords: natural language querying · time series databases · symbolic representation · shape querying

## 1 Introduction

Time series are among the fastest growing kinds of data. Sensors, servers, and markets produce them continuously, and time series databases (TSDBs) now store histories of billions of points. Querying has not kept pace with storage. “On which day of the past year was the closing price highest?” asks about a value, and one ordinary SQL query answers it. On the same curve, users often want to ask about shapes such as “on which days did the price plunge and then recover quickly?” Answering such shape queries at database scale remains an open challenge. This paper is concerned with letting non-expert users pose shape queries to a TSDB in a single sentence. This is the main challenge within natural language querying for time series databases (NLQ4TSDB) [16].

![](images/a3a52100a3a40459786438727ab83142cf3ca528a3dbed863752ecfb112a6f61.jpg)  
Fig. 1. How symbolization serves a natural language shape query, illustrated with SAX. (1) Each window of the series is symbolized into a short string of letters. (2) An LLM compiles the query into a regular expression over the letters. (3) The regex runs over every window’s string.

The recently proposed Sonar-TS [16] answers shape queries at database scale by using symbolization as a bridge between language and data. On one side, the time series is compressed ofline into short strings of discrete symbols. On the other, a large language model (LLM) compiles the query into a regular expression over these symbols. Figure 1 shows an example. The query above, “plunge and then recover quickly”, compiles to [de]+.\*a+.\*[de]+. This means a run of high letters, then bottom letters, then high letters again. In this way, a shape query over a long history of raw values becomes one regular expression matched against short strings. This search is cheap enough to cover an entire database.

Sonar-TS adopts the most basic symbolization, SAX, and leaves open which symbolization suits NLQ4TSDB. Symbolization itself has a long history. SAX [8] writes one level letter per segment of a window. Its descendants add more numeric detail to the string, such as extrema, slopes, angles, and adaptive pieces [9,10,19,12,2]. These additions serve distance and classification, never an LLM. We argue that these symbolizations run into three limitations here. (1) Direction is missing. In level-only symbolizations such as SAX and ESAX [9], a rise never appears as a symbol; it is only implied by the order of level letters. A regex must then guess the level spellings of a rise, and either covers them all and matches too much, or covers a few and misses. (2) Direction is measured at too small a scale. 1d-SAX [10], TFSAX [19], and TSAX [12] do write direction, but they measure it inside a single segment. A segment is short, so the measurement follows the noise inside a long trend, and one sustained rise is written as a broken run of trend letters. Longer segments would smooth the measurement, but they would also coarsen the level letters and erase short shapes. (3) The letters are arbitrary. A shape query ultimately lands on the trend letters, yet most existing symbolizations spell them with abstract codes. For example, 1d-SAX writes a steep rise as the letter “e”. For distance functions and classifiers this is harmless, since renaming every letter changes nothing. For an LLM, whether the alphabet itself helps or hurts retrieval is a question that has received little study.

The three limitations point to three principles. Direction should be written into the string. It should be measured at the scale that queries describe. And the letters should be chosen for the LLM, not left arbitrary. We follow these principles with a minimal redesign of SAX, called TT-SAX (two-track SAX). Each window is written as two words. The level word is the SAX word, untouched. The trend word gives each segment one trend letter, measured across neighboring segments. The letter has five grades spelled F, f, s, r, R for steep fall, slow fall, stable, slow rise, and steep rise. Trend letters are not new. We choose and grade them for the LLM.

Our contributions are as follows.

– We analyze six classical symbolizations against NLQ4TSDB and identify three limitations, which yield three design principles.

We propose TT-SAX, a minimal redesign of SAX that follows these principles. It pairs the level word with an aligned trend word, measured across neighboring segments and spelled with a mnemonic alphabet.

We evaluate TT-SAX against the six classical symbolizations on the public NLQTSBench benchmark across six LLMs, with ablations attributing its gains to the three design choices.

## 2 Background and Related Work

Natural language querying for time series. Querying a TSDB in plain language splits into value queries and shape queries. Value queries inherit mature Text-to-SQL technology [4]; shape queries do not, and systems for them are recent. Sonar-TS [16] searches with a regular expression over stored SAX words and verifies the surviving candidates on their raw values. Its planner is guided by skills, guideline lists learned automatically in the spirit of Reflexion [14]. The same work introduces NLQTSBench, whose composite trend subset is our workload. Sonar-TS fixes the symbolization to SAX and does not ask whether SAX is the right choice.

Symbolizations. SAX [8] z-normalizes a window, averages it over w segments, and maps each mean to a letter with Gaussian breakpoints. Its descendants add information per segment, such as minimum and maximum letters (ESAX [9]), a regression slope letter (1d-SAX [10]), a trend angle letter (TFSAX [19]), or the sign of the slope (TSAX [12]). TVA [3] pairs each SAX letter with a mnemonic U/D/S letter from the slope inside the segment. Zalewski et al. [21] measure across segments. They quantize the diferences between successive segment means with data-driven clusters. Their measurement is the closest precedent of our trend word. ABBA [2] abandons the fixed grid for an adaptive polygonal chain whose pieces are clustered into letters. Frequency and projection based symbolizations such as SFA [13] and SPARTAN [18] are excluded by construction. Their symbols index Fourier or principal coeficients, so a time-ordered shape cannot be written as a pattern over them. Comparisons of symbolizations [18] judge them by reconstruction, distance, and classification. We found no prior work that designs or evaluates a symbolization for use by an LLM.

Shape queries. Shape querying is older than natural language interfaces. Every line of work meets the same two obstacles, scale and authoring. Formal languages such as SDL [1] require the user to write the shape as an expression. We borrow the term shape query from SDL. SAXRegEx [20] runs regular expressions over SAX strings and comes closest to our substrate. However, its regex is derived from an example pattern that the analyst supplies and tunes in a visual tool. Sketch interfaces such as Qetch [11] ease authoring but not scale. ShapeSearch [15] accepts natural language through a fixed parser into a closed algebra over in-memory trendlines. Qute [5] maps trend vocabulary onto numeric shapes. TSseek [7] reaches database scale with a spatial index, but its queries are formal trend expressions, not language. Neural aligners such as time series language models [17] and contrastive retrievers [6] take language directly. But they run a model on every candidate window, a cost that does not scale to a database. Each line addresses one obstacle at most. The symbolic bridge addresses both. An LLM authors an open-ended pattern from the query. The pattern runs over a symbol index at database scale.

## 3 TT-SAX

TT-SAX (two-track SAX) writes each window as two words separated by a bar, as Figure 2 shows. The level word is the SAX word itself, w letters from Gaussian breakpoints on the window-normalized segment means. The trend word adds one trend letter per segment.

TT-SAX adds the trend word to the symbol string to meet the needs of natural language shape queries. TVA, 1d-SAX, TFSAX, and TSAX also write a trend letter, but they measure the trend inside a single segment. A segment is short, so these measurements are sensitive to the noise inside a long trend. TT-SAX instead measures the trend of segment i across its neighbors, as the centered diference $d _ { i } = ( p _ { i + 1 } - p _ { i - 1 } ) / 2$ of the real-valued segment means $p _ { 1 } , \ldots , p _ { w }$ . At the two ends we use the one-sided diferences $d _ { 1 } = p _ { 2 } - p _ { 1 }$ and $d _ { w } = p _ { w } - p _ { w - 1 }$ The trend letter thus summarizes a two-sided context and stays aligned with the level letter at the same position. In essence this is smoothing, and it matches the macroscopic way natural language describes a shape. We analyze the number of neighboring segments in Section 5.2.

![](images/476c273771f8bfc5b9dc12341b4c57ae2ec9b8cdd7339f19f19ae757746f3e85.jpg)  
Fig. 2. An illustration of TT-SAX. Each window is written as two words, a trend word with one trend letter per segment, and the level word.

Each $d _ { i }$ then becomes one of five letters. Two fixed thresholds split the size of the change into stable, slow, and steep. They are $\tau _ { 1 } = 0 . 1 0$ and $\tau _ { 2 } = 0 . 3 0$ on the window-normalized scale. The window is normalized first, so steep means steep relative to that window’s own variation, not an absolute physical rate. The sign of the change picks the direction. Both values were fixed before any experiment and were never selected on development or test labels. Section 5.2 reports a post-hoc sensitivity check. The base letters are mnemonic, f for fall, s for stable, and r for rise. Case separates slow from steep, giving F, f, s, r, R. The alphabet is symmetric between rises and falls and is written for the LLM. The alphabet afects how well an LLM understands and uses the trend letters, and Section 5.2 analyzes alternative alphabets.

Cross-segment slope information is not new by itself. Zalewski et al. [21] discretize the first diferences between successive segment means, $p _ { i + 1 } - p _ { i }$ . TT-SAX uses a related signal for a diferent purpose. Its diference is centered, so the direction belongs to the central segment and aligns with its level letter. Its grades come from fixed, symmetric thresholds, not from learned clusters. Its letters are chosen for natural language. The contribution is not a new numerical operation but this combination. A contextual trend measurement whose width is decoupled from the level grid, two-track alignment, and a language-oriented alphabet together serve natural language shape querying.

## 4 Experimental Setup

Dataset. We use NLQTSBench [16], the only public benchmark of NLQ4TSDB. Its composite trend subset provides 220 queries over multi-domain real-world channels. Each query names a trend pattern of two to four parts, drawn from {rise, fall, stable} with adjectives {rapid, slow, steady, fluctuating}. It asks for the days in a long history that show this pattern, for example “among days with a rapid rise then a stable stretch, the five where the rise is fastest”. We also synthesize 1,000 fresh queries with the benchmark’s public generator and reserve them for the stress test in Section 5.4.

![](images/060e98ce54ca7d6718937d626ab059ba8e570574fc08ab0a20d15248f480ce2f.jpg)  
Fig. 3. The evaluation pipeline. The only variable is the orange slot, the symbolization. It determines both the symbol strings and the encoding card the model reads.

Pipeline. Figure 3 shows the pipeline that every experiment shares. In the symbolize step, every time series is turned into symbol strings. The symbolization that does this is the variable of the study. In the compile step, an LLM turns the query into one regular expression over the letters. Its prompt consists of the query, the encoding card, and skills. The encoding card is the short manual of the chosen symbolization. For example, it covers how many levels there are and what each letter means. Skills are usage guidelines learned beforehand on development queries, separately for each symbolization. Skills are optional, and with them a symbolization comes closer to its ceiling. In the match step, the regular expression runs over every string, and the matching windows are the candidates. In short, everything in this pipeline is shared except the symbolization. The strings and the card follow from that choice. Any diference in what the search returns therefore traces back to the symbolization.

Metrics. Symbolic search is the first stage of search-then-verify. It returns candidate windows, and a verifier then checks them on the raw values. A missed answer is lost, while an extra candidate only costs verification. A good search therefore keeps all the answers and few other candidates. Recall measures the first, candidates the second, and lift reads the two as one number.

– Recall: the fraction of a query’s answer windows that the search keeps.

– Candidates: the fraction of all windows that the search keeps.

– Lift: recall divided by candidates, with a floor on candidates. This is the lift metric of data mining, the precision of the search divided by the precision of a random filter. For a query with recall $R _ { q }$ and candidates $C _ { q }$ ，

$$
L _ { q } = \frac { R _ { q } } { \operatorname* { m a x } ( C _ { q } , 1 0 \% ) } , \qquad \mathrm { l i f t } = \frac { 1 } { n } \sum _ { q = 1 } ^ { n } L _ { q } ,\tag{1}
$$

where n is the number of queries, and the floor of 10% was fixed in advance to stop a pattern that matches almost nothing from scoring high by luck.

Baselines. We compare against six classical symbolizations. SAX [8] is the base. It cuts a day into w segments and writes the mean of each segment as one of α level letters. ESAX [9] writes three level letters per segment, its minimum, mean, and maximum, in the order they occur. 1d-SAX [10] writes two letters per segment, one for the level and one for the slope of a fitted line. TFSAX [19] also writes a second letter, taken from the angle of the segment’s net change. TSAX [12] appends ‘u’ or ‘d’ to each level letter, for whether the segment goes up or down. ABBA [2] drops the fixed grid. It covers the day with a few adaptive linear pieces, clusters them, and writes one letter per piece, so its strings are short and vary in length. All symbolizations including TT-SAX run at one shared setting with no per-method tuning, namely w = 24 segments per day where a fixed grid applies and α = 5 letters per attribute.

Model. The main experiments use DeepSeek-V4-Flash. For robustness across models, Section 5.4 repeats the comparison with DeepSeek-V4-Pro, Gemini-3.7- Flash, GPT-5.6-Luna, Qwen3.6-35B, and Claude Haiku 4.5. All models are called through their APIs in August and September 2026 with temperature zero and reasoning disabled.

Skills learning. The skills in the prompt are learned beforehand, separately for each symbolization, following Sonar-TS’s recipe. We generate 80 development queries with the benchmark’s generator. The skills are distilled from the model’s own failures on them over four rounds. The learning is repeated with three seeds, and we report test means.

## 5 Results

## 5.1 Main Results

Table 1 reports the three metrics for TT-SAX and the six baselines, with and without skills. Without skills, SAX, TSAX, and ABBA are no better than a random filter, whose lift is 1 by definition. ESAX and TFSAX do only slightly better. 1d-SAX reaches the highest recall, but it buys that recall with high candidates. TT-SAX matches its recall with far fewer candidates. Its lift of 2.42 is 1.7× that of 1d-SAX, and every baseline stays below 1.5. With skills every symbolization improves, and the weakest improves most. ABBA rises from 0.37 to 0.69 recall. ESAX keeps the fewest candidates, but its recall is now the lowest. 1d-SAX again pays for its recall with candidates. TT-SAX reaches the best recall while its candidates barely grow. Its lift of 3.04 is 1.4× that of the next best.

Two conclusions follow. First, doing well on one metric alone is easy. A looser pattern matches more windows, so recall rises but candidates rise with it. This is how 1d-SAX and TSAX score. A tighter pattern shrinks candidates but loses answers too. This is how ABBA and ESAX score. The hard part is doing well on both at once. TT-SAX is the only symbolization that does, so its lift stands apart. Second, skills change the numbers and reorder several baselines. ABBA

Table 1. Main results of TT-SAX. Best in bold, second best underlined. n = 220 queries; every metric is averaged over queries; the results with skills average three skill-learning seeds, the results without skills are single runs.
<table><tr><td></td><td colspan="3">w/o skills</td><td colspan="3">w/ skills</td></tr><tr><td>Symbolization</td><td>Recall ↑</td><td>Cand. ↓</td><td>Lift ↑</td><td>Recall ↑</td><td>Cand. ↓</td><td>Lift ↑</td></tr><tr><td>SAX</td><td>0.43</td><td>35%</td><td>1.00</td><td>0.59</td><td>36%</td><td>1.56</td></tr><tr><td>ESAX</td><td>0.47</td><td>34%</td><td>1.19</td><td>0.49</td><td>28%</td><td>1.53</td></tr><tr><td>1d-SAX</td><td>0.73</td><td>51%</td><td>1.42</td><td>0.83</td><td>49%</td><td>2.11</td></tr><tr><td>TFSAX</td><td>0.55</td><td>38%</td><td>1.39</td><td>0.59</td><td>33%</td><td>2.16</td></tr><tr><td>TSAX</td><td>0.62</td><td>50%</td><td>1.01</td><td>0.65</td><td>43%</td><td>1.78</td></tr><tr><td>ABBA</td><td>0.37</td><td>25%</td><td>0.94</td><td>0.69</td><td>41%</td><td>1.94</td></tr><tr><td>TT-SAX (ours)</td><td>0.71</td><td>32%</td><td>2.42</td><td>0.87</td><td>36%</td><td>3.04</td></tr></table>

rises from last to fourth on lift and ESAX falls to last, yet TT-SAX keeps the highest lift under both conditions. Skills and symbolization interact, but skills do not close the gap. On lift, TT-SAX outperforms every other symbolization on natural language shape queries.

## 5.2 Ablation Studies

We ablate TT-SAX in three ways. We remove one of the two words, widen the trend measurement, and change the trend letters. All ablations run without skills, which removes the learning loop as a source of variation.

Removing a track. Table 2 reports the ablation of the two words. Removing the trend word hurts most. Without it the model has to spell the trend through sequences of level letters. These patterns keep many windows yet catch few answers. Lift falls to 1.00, the level of a random filter. Removing the level word also lowers recall, but the candidates shrink with it. The lift holds at 2.03, above every baseline without skills in Table 1. This means the trend word alone still finds answers eficiently, which supports the premise of TT-SAX. What natural language shape querying needs most is a direct expression of the trend.

Table 2. Ablation of the two words.
<table><tr><td>Variant</td><td>Recall ↑</td><td>Cand. ↓</td><td>Lift ↑</td></tr><tr><td>TT-SAX</td><td>0.71</td><td>32%</td><td>2.42</td></tr><tr><td>w/o trend word</td><td>0.43</td><td>35%</td><td>1.00</td></tr><tr><td>w/o level word</td><td>0.55</td><td>26%</td><td>2.03</td></tr></table>

![](images/c55a24b9bf6d46287fbdb1dbbd3ab3291dc8869b84e9d91bb56e5631061c921e.jpg)  
Fig. 4. Recall (left) and candidates (right) as the trend measurement widens, for TT-SAX and the three baselines that also write a trend letter.

Widening the trend measurement. TT-SAX measures the trend of a segment across its neighbors rather than inside the segment alone. The choice follows from how queries are worded. Natural language describes macroscopic movement, and a segment is short, so the change inside it is closer to noise than to trend. We test the choice by widening the trend measurement of TT-SAX and of the three baselines that also write a trend letter. Each uses its own formula, refitting its slope, angle, or sign over the wider span. Figure 4 reports the results.

TT-SAX’s recall rises and then falls as neighboring segments are added, and is best at two neighbors. This supports the choice behind the trend word. Inside a single segment, noise dominates the change. We also note that the best neighbor count is not universal. It depends on the query distribution, the data, and the other parameters of the symbolization. We therefore keep the shared settings of the NLQTSBench benchmark throughout and do not search this space. The baselines also write trend letters, but they define them diferently, so widening does not give them a comparable gain. TSAX is the clearest case. Its trend alphabet has two letters, up and down. A wider span is usually a slower one, but the alphabet has no letter for slow. Every widened span is still written as up or down, and nothing is gained. 1d-SAX grades its slope on a scale computed for the segment’s length, so the widened slopes no longer fit the scale. TFSAX divides its angle by the turning points the wider span brings in. These results motivate decoupling the trend measurement from the level grid.

The trend word has one more parameter, the two thresholds that cut the size of the change. Five other pairs, from 0.05/0.20 to 0.15/0.40, give recall between 0.65 and 0.68, candidates between 31% and 33%, and lift between 2.11 and 2.30. Each of these lifts is above every baseline without skills. The default pair was fixed before the experiments and was left unchanged.

Changing the letters. In natural language shape querying, every symbolization comes with an encoding card. The card tells the model what each letter means, and the model reads it to compile the query into a regex. Whether the choice of letters changes this compilation is unknown. Classical symbolization never faced this question. Its letters go into distance computations, where a letter is just a name. We test thirty alphabets in three groups, one per design philosophy. Group I gives the letters meaning. Each is the first letter of a direction word, as in TT-SAX’s FfsrR, with f for fall and r for rise. Group II drops the meaning but keeps the pairing. The letters still form a fall family and a rise family by case, as in KkxmM, yet they start no word. Group III drops the pairing too. The letters are a plain alphabet run, as in vwxyz, and only their order separates fall from rise. Eight case-paired alphabets are additionally rewritten with their cases swapped.

![](images/ab01c26cc40dcac7e63084dfbd21057814d068942a3627fe99ff2d4b7f53597e.jpg)

![](images/738dd9977b8d30bf3205c3c32ef049300fc62c24a392dc63dbd9787eaeb7be80.jpg)  
Fig. 5. Ablation of the trend letters. Group I letters are the initials of direction words, e.g., FfsrR. Group II letters pair by case but carry no meaning, e.g., KkxmM. Group III letters are plain alphabet runs, e.g., vwxyz. In the case-swapped alphabets the uppercase and lowercase of every case-paired letter are exchanged, so FfsrR becomes fFsRr.

We run the thirty alphabets, ten per group, on one fixed symbolization and two LLMs, DeepSeek-V4-Flash and Qwen3.6-35B. Figure 5 reports the results. First, Group I is the best group with both LLMs, and its lead grows on the weaker one. A plausible reading is that a weaker model follows the card less reliably, so letters that already carry the meaning help it most. Second, the assignment of uppercase and lowercase matters. Rewriting TT-SAX’s FfsrR as fFsRr drops recall from 0.71 to 0.61. The average loss across the eight swapped pairs is five points. The two alphabets carry the same information, and each encoding card states its mapping precisely. The loss therefore shows that the LLM does not take the mapping from the card alone. It is consistent with a preferred reading of the two cases, which the swapped alphabets go against. In summary, renaming the letters has no efect on distance and compression, the classical uses of symbolization. For natural language shape querying it does. The

![](images/a2be979069912d30f0367fe5711ff1a627e45feb434b6b77679644afc6d49eec.jpg)  
Fig. 6. Skills transfer results of TT-SAX.

TT-SAX alphabet is built for this. Its direction words give the LLM the meaning it needs, its uppercase letters keep the convention the LLM expects, and the whole alphabet stays readable to the human.

## 5.3 Skills Transfer

An LLM compiles the query into a regex by reading the encoding card and the skills. The skills are learned by self-evolution on development queries with Sonar-TS’s recipe [16]. The model answers, analyzes its failures, and summarizes skills round after round. Models difer in ability and style, so the skills they evolve difer too. Can the skills learned by one model serve another? Three models learn their own TT-SAX skills on the same development queries. DeepSeek-V4-Flash, DeepSeek-V4-Pro, and Qwen3.6-35B give Skills-F, Skills-P, and Skills-Q. Each model then uses all three sets. Figure 6 reports the results.

Three findings follow. First, the skills are transferable. Every combination of model and skills beats that model without skills. Second, a set of skills keeps its character on every model. With Skills-Q every model reaches its highest recall, and with Skills-F every model keeps its fewest candidates. Third, a model’s own skills are not necessarily its best. On lift, Skills-F is the best set for all three models. For DeepSeek-V4-Pro and Qwen3.6-35B it beats their own skills.

## 5.4 Robustness Analysis

We assess TT-SAX under five further LLMs. We replay both conditions with prompts identical to the main experiment. Figure 7 reports the results. TT-SAX is robust. Its average lift over the five models is 1.81 without skills and 3.28 with, against 1.24 and 1.93 for the best classical symbolization.

Most of our experiments run on the public benchmark, and its 220 queries are few. A symbolization could look strong by fitting the particulars of a small query set, and a larger set could give a diferent result. We therefore use the reserved set of 1,000 queries from the benchmark’s public generator [16]. We deduplicate them against the benchmark and development queries, so the set is larger but not diferently distributed. We rerun TT-SAX and all six baselines on it. Figure 8 shows the results. Every symbolization moves by at most three points in recall and in candidates. The skills help as before. TT-SAX keeps its place nearest the upper-left corner. It is the only symbolization with lift above two on both sets in both conditions. The advantage holds on a query set nearly five times larger.

![](images/8873b3453f6864c36aed599e885027c3540cf49c282f998969a8901f21a075bd.jpg)  
Fig. 7. Lift of every symbolization under six LLMs, without skills (arrow tail) and with skills (arrow head).

![](images/845645f50805e9a6e9ad3ca7789f6a7e39dcf6fa501c811a964f155c50d12e64.jpg)  
Fig. 8. Stress test on a fresh set of 1,000 queries.

## 6 Conclusion

This paper studied how time series symbolization should be designed for natural language shape querying. We systematically analyzed the limitations of classical symbolizations in this setting. We then proposed TT-SAX, which represents level and contextual trend information in two aligned tracks. Further analyses indicate that its gains are associated with an explicit trend word, an appropriate measurement span, and meaningful letters. The gains remain consistent across six LLMs. The results demonstrate the importance of designing symbolizations that are both numerically informative and accessible to LLMs.

Acknowledgments. This research was supported by the NVIDIA Academic Grant in Higher Education and Developer Program and the CSIRO–National Science Foundation (US) AI Research Collaboration Program.

## References

1. Agrawal, R., Psaila, G., Wimmers, E.L., Zaït, M.: Querying shapes of histories. In: Proceedings of the 21st International Conference on Very Large Data Bases (VLDB). pp. 502–514. Morgan Kaufmann (1995), http://www.vldb.org/conf/1995/P502. PDF

2. Elsworth, S., Güttel, S.: ABBA: adaptive Brownian bridge-based symbolic aggregation of time series. Data Mining and Knowledge Discovery 34(4), 1175–1200 (2020). https://doi.org/10.1007/s10618-020-00689-6

3. Esmael, B., Arnaout, A., Fruhwirth, R.K., Thonhauser, G.: Multivariate time series classification by combining trend-based and value-based approximations. In: Computational Science and Its Applications – ICCSA 2012, Part IV. Lecture Notes in Computer Science, vol. 7336, pp. 392–403. Springer (2012). https://doi.org/ 10.1007/978-3-642-31128-4\_29

4. Hong, Z., Yuan, Z., Zhang, Q., Chen, H., Dong, J., Huang, F., Huang, X.: Next-generation database interfaces: A survey of LLM-based text-to-SQL. IEEE Transactions on Knowledge and Data Engineering 37(12), 7328–7345 (2025). https://doi.org/10.1109/TKDE.2025.3609486

5. Imani, S., Alaee, S., Keogh, E.: Qute: Query by text search for time series data. In: Proceedings of the Future Technologies Conference (FTC) 2020, Volume 2. Advances in Intelligent Systems and Computing, vol. 1289, pp. 412–427. Springer (2021). https://doi.org/10.1007/978-3-030-63089-8\_27

6. Ito, A., Dohi, K., Kawaguchi, Y.: CLaSP: Learning concepts for time-series signals from natural language supervision. In: 2025 33rd European Signal Processing Conference (EUSIPCO). pp. 1817–1821. IEEE (2025). https://doi.org/10.23919/ EUSIPCO63237.2025.11226094

7. Li, X., Alnuaim, K., Eltabakh, M.Y., Rundensteiner, E.A.: TSseek: Regular expression-based similarity search for distributed time series datasets. arXiv preprint arXiv:2606.09824 (2026). https://doi.org/10.48550/arXiv.2606.09824

8. Lin, J., Keogh, E., Wei, L., Lonardi, S.: Experiencing SAX: a novel symbolic representation of time series. Data Mining and Knowledge Discovery 15(2), 107–144 (2007). https://doi.org/10.1007/s10618-007-0064-z

9. Lkhagva, B., Suzuki, Y., Kawagoe, K.: Extended sax: Extension of symbolic aggregate approximation for financial time series data representation. DEWS2006 4A-i8 7 (2006)

10. Malinowski, S., Guyet, T., Quiniou, R., Tavenard, R.: 1d-SAX: A novel symbolic representation for time series. In: Advances in Intelligent Data Analysis XII (IDA 2013). Lecture Notes in Computer Science, vol. 8207, pp. 273–284. Springer (2013). https://doi.org/10.1007/978-3-642-41398-8\_24

11. Mannino, M., Abouzied, A.: Expressive time series querying with hand-drawn scalefree sketches. In: Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems. pp. 1–13. ACM (2018). https://doi.org/10.1145/3173574. 3173962

12. Muhammad Fuad, M.M.: TSAX is trending. In: Computational Science – ICCS 2021, Part II. Lecture Notes in Computer Science, vol. 12743, pp. 284–296. Springer (2021). https://doi.org/10.1007/978-3-030-77964-1\_23

13. Schäfer, P., Högqvist, M.: SFA: a symbolic Fourier approximation and index for similarity search in high dimensional datasets. In: Proceedings of the 15th International Conference on Extending Database Technology (EDBT). pp. 516–527. ACM (2012). https://doi.org/10.1145/2247596.2247656

14. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S.: Reflexion: language agents with verbal reinforcement learning. In: Advances in Neural Information Processing Systems 36 (NeurIPS 2023). pp. 8634– 8652 (2023), https://papers.nips.cc/paper\_files/paper/2023/hash/ 1b44b878bb782e6954cd888628510e90-Abstract-Conference.html

15. Siddiqui, T., Luh, P., Wang, Z., Karahalios, K., Parameswaran, A.: ShapeSearch: A flexible and eficient system for shape-based exploration of trendlines. In: Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data. pp. 51–65. ACM (2020). https://doi.org/10.1145/3318464.3389722

16. Tan, Z., Zhao, Y., Wang, S., Xu, C., Liang, Y., Liu, X., Pan, S., Jin, M.: Sonar-TS: Search-then-verify natural language querying for time series databases. arXiv preprint arXiv:2602.17001 (2026). https://doi.org/10.48550/arXiv.2602.17001, accepted at ICML 2026

17. Xie, Z., Li, Z., He, X., Xu, L., Wen, X., Zhang, T., Chen, J., Shi, R., Pei, D.: ChatTS: Aligning time series with LLMs via synthetic data for enhanced understanding and reasoning. Proceedings of the VLDB Endowment 18(8), 2385–2398 (2025). https://doi.org/10.14778/3742728.3742735

18. Yang, F., Paparrizos, J.: SPARTAN: Data-adaptive symbolic time-series approximation. Proceedings of the ACM on Management of Data 3(3), 1–30 (2025). https://doi.org/10.1145/3725357

19. Yu, Y., Zhu, Y., Wan, D., Liu, H., Zhao, Q.: A novel symbolic aggregate approximation for time series. In: Proceedings of the 13th International Conference on Ubiquitous Information Management and Communication (IMCOM) 2019. Advances in Intelligent Systems and Computing, vol. 935, pp. 805–822. Springer (2019). https://doi.org/10.1007/978-3-030-19063-7\_65

20. Yu, Y., Becker, T., Trinh, L.M., Behrisch, M.: SAXRegEx: Multivariate time series pattern search with symbolic representation, regular expression, and query expansion. Computers & Graphics 112, 13–21 (2023). https://doi.org/10.1016/ j.cag.2023.03.002

21. Zalewski, W., Silva, F., Lee, H.D., Maletzke, A.G., Wu, F.C.: Time series discretization based on the approximation of the local slope information. In: Advances in Artificial Intelligence – IBERAMIA 2012. Lecture Notes in Computer Science, vol. 7637, pp. 91–100. Springer (2012). https://doi.org/10.1007/978-3-642-34654-5\_10