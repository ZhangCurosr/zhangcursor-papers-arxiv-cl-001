# Distinct dynamics of conceptual and referential disruptions in human reading and large language model processing

Rui He<sup>a</sup>, Nihal Altay<sup>a</sup>, Wolfram Hinzen<sup>a,b</sup>

<sup>a</sup>Grammar and Cognition Lab, Department of Translation & Language Sciences, Universitat Pompeu Fabra, Barcelona, Spain

<sup>b</sup>Institut Català de Recerca i Estudis Avançats (ICREA), Barcelona, Spain

## Abstract

Linguistic meaning is grounded in conceptual content, from which reference to particular entities emerges as words enter discourse. To examine the processing dynamics associated with these two dimensions of meaning, we selectively disrupted conceptual or referential information in short narratives and traced the resulting efects in human self-paced reading and in the predictive and representational processing of large language models. In human reading, conceptual disruptions produced a strong but localized processing cost, emerging immediately after the distorted word, reaching an early maximum, and then declining rapidly. Referential disruptions produced weaker efects, which decreased more gradually across subsequent words, and were more strongly modulated by sentence boundaries. In the language model, both disruptions emerged immediately at the manipulated word. Contextual model surprisal showed a pattern closely paralleling human reading: conceptual disruption produced a larger, more locally concentrated efect that decayed rapidly, whereas referential disruption produced a smaller and more gradual downstream efect. Output-layer representations showed a diferent pattern: referential disruption produced a larger initial displacement, while both distortions were subsequently characterized by power-law decay. Together, these results provide convergent evidence for distinguishable processing dynamics of two types of meaning: conceptual information imposes a more locally concentrated integration cost, whereas referential information engages a more distributed process of maintaining discourse-level identity.

Keywords:

conceptual meaning, referential meaning, self-paced reading, sentence boundary,

## 1. Introduction

Language unfolds one word at a time, yet its interpretation must remain coherent across much longer stretches of sentences and discourse. Each incoming expression is interpreted in relation to what has already been established and, in turn, contributes to the linguistic context in which subsequent material is understood. Language thus carries its own interpretive history forward as the discourse continues, with prior linguistic material constraining later interpretation at multiple levels (Brilmayer and Schumacher, 2021; Brothers et al., 2023; Dijksterhuis et al., 2024). These influences can extend over diferent temporal spans, bearing most strongly on the immediate interpretation of an expression in some cases while remaining relevant farther downstream in others (Chen et al., 2024). How such influences propagate over time may therefore illuminate the linguistic relations through which interpretation is maintained.

Maintaining a linguistic interpretation requires more than the accumulation of conceptual content. Words contribute lexical-conceptual material whose interpretation is further shaped by context (Leclercq, 2023), but this content alone does not determine what an expression refers to on a particular occasion. This distinction has long been recognized in semantic theory. Frege (1892) distinguished between the way an expression presents its meaning and what it refers to, and later formal approaches examined how word meanings combine within sentences and how referents are tracked across discourse (Heim, 2002; Kamp, 2013; Montague, 1970). The grammatical account adopted here goes a step further by treating reference itself as partly determined by linguistic structure: lexical material is not inherently referential, but becomes referential through the grammatical configurations in which it occurs (Hinzen, 2016). Conceptual and referential meaning are thus closely coordinated in ordinary language without being reducible to the same form of linguistic organization. Despite this coordination, the distinction motivates two experimental perturbations that place diferential pressure on the two. Conceptual distortion (CD) alters the lexical-conceptual content contributed by an expression so that it becomes incompatible with its established context, whereas referential distortion (RD) alters a linguistic relation required to maintain or recover the intended discourse referent. These manipulations provide diferential perturbations of conceptual and referential organization while preserving their interdependence within the broader linguistic system.

As linguistic interpretation proceeds incrementally, the consequences of these perturbations may extend beyond the expression at which they are introduced, shaping the conditions under which subsequent material is interpreted. Theoreti cally relevant information may therefore lie not only in the magnitude of the processing dificulty they induce, but also in how this dificulty unfolds through the material that follows. Psycholinguistic research has extensively examined lexicalsemantic and referential processing (Dietrich et al., 2024; Kutas and Federmeier, 2011), although these have more often been studied in separate paradigms than compared within a common perturbational framework. Self-paced reading (SPR) is well suited to such a comparison, because it traces behavioral consequences word by word using reading time (RT), and RT efects are known to extend beyond their lexical source into subsequent positions (Smith and Levy, 2013). Recent evidence comes particularly close to the present concept-reference distinction. Soberats et al. (2025) found sharply local efects for lexical-semantic anomalies but substantially more extended efects for grammatical anomalies afecting referential meaning. Although these manipulations do not map directly onto the present CD and RD conditions, this contrast raises the possibility that conceptual and referential perturbations difer not only in their immediate impact, but also in their downstream dynamics. In addition, such propagation need not depend on distance alone. If referential meaning is partly sustained by grammatical organization, then the persistence of referential perturbation may also be sensitive to the structural points at which an unfolding interpretation is completed or reorganized (Martin and Hinzen, 2014). Sentence boundaries are particularly relevant in this respect. They mark the completion of one configuration of linguistic structure before another begins, and can therefore provide a structural reset point for ongoing interpretation (Arsenijevic and Hinzen, 2012). Sentence-final posi-´ tions are also associated with increased integration or wrap-up during processing (Meister et al., 2022). A perturbation that reaches a sentence boundary can therefore reveal whether its persistence follows linear distance alone or is reshaped by the grammatical structure through which discourse unfolds. Accordingly, rather than treating downstream efects only as a set of predefined spillover regions, we characterize their progression as a propagation profile, tracing how they emerge, persist, and dissipate across linguistic distance and structural boundaries.

This raises a parallel question for language models (LMs): if conceptual and referential perturbations leave diferent downstream signatures in human processing, are these diferences also expressed in the way a computational language system carries linguistic context forward? LM hidden-state representations have been shown to predict neural responses during natural language processing, despite important diferences between artificial and biological systems in how context is accumulated over time (Tikochinski et al., 2025; Yang et al., 2026). In an autoregressive model, the influence of prior context is expressed both in the probabilities assigned to upcoming words and in the contextual representations from which these predictions arise. Contextual surprisal therefore provides a measure of how a perturbation reshapes subsequent lexical expectations, a quantity closely related to variation in human RT (Wilcox et al., 2023). At the same time, beyond its immediate efect on prediction, the perturbation may remain detectable in the model’s evolving hidden states, which encode information from the surrounding linguistic context (Ethayarajh, 2019). Skrill and Norman-Haignere (2023) showed that such representational influence can persist across subsequent positions, decaying with distance while becoming increasingly sensitive to structural boundaries. Predictive and representational efects need not follow identical dynamics. A perturbation may remain detectable in the model’s evolving contextual state after its immediate efect on lexical expectation has diminished, or conversely, may strongly alter prediction while producing only a transient representational displacement. We therefore examine both contextual surprisal and output-layer representational distance, asking whether the distinct propagation profiles of conceptual and referential distortions observed in human reading are also expressed in model prediction, representation, or both.

Thus, we hypothesize that conceptual and referential perturbations difer not only in the magnitude of their efects, but in how those efects propagate through subsequent linguistic context. In human reading, conceptual distortion is expected to produce a more concentrated downstream efect, whereas referential distortion should remain consequential over a broader span and show greater sensitivity to sentence boundaries. To test these predictions, we designed a word-byword self-paced reading experiment in which conceptual and referential distortions were introduced into extended discourse. We analyzed how their behavioral consequences propagated with linguistic distance and were modulated by sentence boundaries. We then applied the same perturbations to autoregressive language models, examining their efects separately in contextual surprisal and output-layer representations, with Qwen3-4B as the primary model and Llama-3.2-3B as a replication. This parallel design allows the propagation of conceptual and referential information to be compared across human behavior, model prediction, and model representation.

## 2. Methods

## 2.1. Participants

We conducted an online self-paced reading experiment with 99 participants recruited through Prolific. Participants were native speakers of English aged 20-50 years (mean age = 36.47 years; 60 females, 39 males). Ninety-five had completed high school education. The study followed ethical principles consistent with the guidelines of the CIREP (Institutional Committee for Ethical Review of Projects) at Universitat Pompeu Fabra. Formal ethics review was not required for this nonbiomedical study under current regulations in Spain. A comprehension question followed each story. Participants were retained only if they achieved greater than 60% accuracy across the comprehension questions. All participants in the final sample met this criterion, with a mean comprehension accuracy of 92.98%.

## 2.2. Stimuli

Participants read 21 fables adapted from a publicly available collection of Aesop’s fables (https://aesopfables.com/aesopsel.html) in a word-by-word, noncumulative self-paced reading paradigm. The source texts were rewritten in modern English using ChatGPT (GPT-5.3-mini) with the prompt, “Can you rewrite this story in modern English?” Stories were approximately 60–120 words long, with a median length of 95 words. Each story was adapted into three versions: Original, CD, and RD. The Original version served as the undistorted baseline. CD involved replacing a contextually appropriate noun with a semantically incongruent alternative (e.g., the rushing current → the rushing mountain), creating a local lexical-semantic disruption. RD involved altering a pronoun or determiner so that it conflicted with the intended discourse referent (e.g., he discovered → she discovered), disrupting reference tracking across the discourse. This distinction allowed us to compare the downstream consequences of conceptual and referential disruptions. Examples are provided in the supplementary.

Distortions were inserted manually at positions that produced a clear disruption while preserving the broader narrative structure. Opening sentences were left undistorted to allow readers to establish the discourse context. To avoid confounding conceptual distortions with sentence-final wrap-up efects, an adjunct was added when a CD would otherwise occur sentence-finally (e.g., “He brought the donkey home and placed it in the stable with his other donkeys, so it could rest.”). CD and RD occurred at approximate rates of one per 30 and one per 20 words, respectively.

A word-level stimulus template was generated by aligning each distorted sentence with its Original counterpart. The distorted words were labelled K; the three preceding and following positions were labelled K-3 to K-1 and K+1 to K+3; and the sentence-final word was additionally labelled K\_END. Positional alignment was used when sentence lengths were identical. Sequence alignment was used for the small number of sentences that difered in length.

The final stimulus set comprised 63 story versions. To assess their comparability, we calculated whole-story perplexity, mean lexical frequency, and mean word length. Perplexity was estimated with Qwen3-4B-Base. Lexical frequency was quantified using mean Zipf frequency from the Python package wordfreq. Word length was defined as the mean number of alphabetic characters per lexical token. Diferences among Original, CD, and RD were evaluated across the 21 matched story sets using Friedman tests and Siegel–Castellan pairwise comparisons, with p values corrected following the false discovery rate (FDR) procedure.

## 2.3. Procedure

The experiment was administered online using Gorilla. Stories were presented word by word using a non-cumulative self-paced reading paradigm. Participants advanced to each successive word by pressing the spacebar. Before each story, a separate instruction screen prompted participants to press the spacebar to begin, providing a common starting point before word presentation. Each participant read 21 stories, but encountered only one version of each story: seven Original, seven CD, and seven RD stories. Stories were presented in randomized order. Each story was followed by a four-alternative comprehension question designed to assess attention and comprehension without explicitly drawing attention to the manipulated expressions.

## 2.4. Data de-identification and preprocessing

The raw Gorilla dataset was de-identified before analysis. Original participant identifiers were replaced with sequential anonymous identifiers, and session identifiers were recoded within participant. Timestamps, completion codes, external session identifiers, device and browser information, monitor and viewport dimensions, and other unnecessary participant metadata were removed. Word-level RTs were reconstructed from the Gorilla event log. A word-presentation event was retained when it was immediately followed by the corresponding keyboard-response event, and RT was calculated as the diference between the two event timestamps.

Observed word sequences were aligned with the corresponding stimulus template. Exact positional matching was used for trials with complete word sequences. Trials with up to three missing word events were retained only when a unique alignment with the stimulus sequence could be established. Trials containing unresolved token mismatches, ambiguous alignments, more than three missing words, or additional or duplicated word events were excluded. Following alignment and trial-level quality control, 1,972 of the 2,079 possible participant-story trials were retained (94.85%): 664 Original, 657 CD, and 651 RD trials. Each story-condition combination therefore contributed data from approximately 31–32 participants on average, indicating broadly balanced coverage across conditions. Where more than one observation was available for the same participant and word position, observations were averaged to produce a single participant-level word observation.

## 2.5. Reading time preprocessing

Reading times outside the range of 100–3000 ms were treated as outliers and replaced with the median RT for the corresponding participant and story (Lo et al., 2023). When a participant–story median was unavailable, the participant-level median was used instead. Imputation afected 0.32% of data values (0.38% in Original, 0.35% in CD, and 0.22% in RD). RTs were subsequently logtransformed to reduce positive skew. To account for systematic variation in reading speed unrelated to the experimental manipulation, log-transformed RTs were residualized prior to the main analyses. To account for spillover between successive self-paced reading responses, we followed the approach proposed by Vasishth (2006), in which RT at the current word is adjusted for RT at the immediately preceding word. Log-transformed reading times were therefore residualized against lag-1 log RT using a mixed-efects model with a participant random intercept. Lag-1 reading time was defined only for contiguous words within the same participant, story, and story version. No lag was carried across missing word events. The resulting residualized log reading times were used as the dependent variable in subsequent analyses.

## 2.6. Reading time analysis

Local reading-time efects. Local distortion efects were examined from K-3 to K+3 and at K\_END. For each position, CD and RD were separately compared with their matched Original observations using mixed-efects linear models. Residualized log RT was the dependent variable, with condition as the predictor of interest and word position within the story, lexical surprisal, and word length as covariates. Models included random intercepts for participants and matched stimulus items. The condition coeficient represented the estimated distortion-related change in residualized log RT. P values were FDR-adjusted across positions separately for CD and RD.

Downstream reading-time trajectories. We next characterized the persistence of distortion efects across subsequent words. For each distorted expression, the final K token was treated as the trajectory anchor, and distance was defined as the number of words following this position. Trajectories extended to the end of the sentence or to a maximum of K+10, whichever came first. Because the behavioral response emerged after presentation of the distorted word, the RT trajectory analysis covered K+1 to K+10. Separate linear mixed-efects models were fitted for CD and RD. Residualized log RT was modelled as a function of categorical word distance, condition, and their interaction. The models also included an interaction between condition and sentence-boundary status to test whether the distortion efect was additionally modulated at the final word of the sentence. Word position within the story, lexical surprisal, and word length were included as covariates, with participant-level random intercepts. A distance was retained only when both Original and distorted observations were available, with at least 20 observations and at least two matched items contributing data. These models therefore tested both whether the magnitude of the distortion efect changed across downstream positions and whether sentence-final words showed an additional distortion-related efect.

Functionalform ofthe reading-time trajectories. We next examined the shape of the downstream RT trajectories. For each distortion type, the distance-specific efects estimated from the mixed-efects models were fitted with 11 candidate functions: constant, linear, quadratic, exponential, power-law, log-normal, log-Cauchy, Zipf–Alekseev, broken-stick, double-exponential, and exponential-pluspower. The signed distortion efects were retained so that both the magnitude and direction of the response were preserved. Curve fitting incorporated the covariance among the distance-specific estimates, thereby accounting for their statistical dependence. Multiple initial parameter values were used for nonlinear models to reduce sensitivity to model initialization. Candidate functions were compared using the corrected Akaike information criterion (AICc). The function with the lowest AICc was selected as the best-supported trajectory for each distortion type. Uncertainty in derived trajectory characteristics was quantified using 1,000 parametric bootstrap samples. For each bootstrap sample, a new trajectory was drawn from the multivariate distribution defined by the estimated distance efects and their covariance matrix, and the selected function was refitted. Percentile-based 95% confidence intervals were then calculated for relevant trajectory characteristics, such as peak location and half-maximum range for peak-shaped trajectories and slope or overall change for monotonic trajectories. Detailed fitting procedures and function forms are described in the supplementary.

Direct comparison of CD and RD reading-time trajectories. Because fitting CD and RD separately does not directly test whether their trajectories difer, we additionally compared the two distortion types within a single mixed-efects model. Residualized log RT was modelled as a function of categorical word distance, distortion type, and their interaction, together with the interaction between distortion type and sentence-boundary status. Word position, lexical surprisal, and word length were included as covariates, with participant-level random intercepts. The distance-by-distortion-type interaction tested whether the downstream profiles difered between CD and RD. We also compared an early window (K+1 to K+3) with a late window (K+8 to K+10) to assess relative persistence. This contrast tested whether the change from early to late positions difered between RD and CD.

## 2.7. Large language model analysis

To examine whether the distinction observed in human reading behavior was also reflected in large language models (LLMs), we applied the same K-centered analyses and efect decay framework to two complementary LLM-derived measures: surprisal, indexing predictive processing, and representational distance, indexing changes in contextual representation.

The same stimuli and original-distorted word mappings were analyzed using the pretrained 4-billion-parameter Qwen3 base model (Qwen/Qwen3-4B-Base). The model was used in evaluation mode without fine-tuning, and each complete story version was processed in a single forward pass without added special tokens. Token-level surprisal was calculated from the model’s autoregressive next-token probability distribution. When a word was split into multiple subword tokens, token-level surprisals were summed to obtain word-level surprisal. The distortionrelated surprisal efect was defined as

$$
\Delta S _ { t } = S _ { t } ^ { \mathrm { d i s t o r t e d } } - S _ { t } ^ { \mathrm { O r i g i n a l } } ,
$$

such that positive values indicated greater surprisal in the distorted sequence than at the corresponding position in the Original sequence.

To quantify representational change, we extracted the final-layer hidden state associated with the final subword token of each word. The divergence between matched distorted and original representations was quantified using cosine distance,

$$
D _ { t } = 1 - \cos ( h _ { t } ^ { \mathrm { d i s t o r t e d } } , h _ { t } ^ { \mathrm { O r i g i n a l } } ) .
$$

The two measures capture complementary aspects of model processing: surprisal reflects the efect of a distortion on the model’s predictions, whereas cosine distance reflects the extent to which the distortion alters its contextual representation.

We then applied analyses parallel to those used for RT. First, local efects were examined at the same K-3 to K+3 and K\_END positions. For each measure, distortion type, and position, an intercept-only ordinary least-squares model was fitted to the matched distortion measure. Cluster-robust standard errors were calculated at the manipulated story–sentence level to account for non-independence among observations from the same distortion unit. Analyses were conducted separately for CD and RD and for surprisal and cosine distance. Positions with fewer than three observations or fewer than two independent distortion units were not modelled, and p values were adjusted within each measure and distortion type using FDR.

Second, we examined downstream trajectories using the same perturbation mappings as in the RT analysis. In contrast to RT, model trajectories included K itself and therefore extended from K to K+10, because both prediction and representation could respond directly at the distorted token. For each measure and distortion type, linear models were fitted with categorical word distance and sentence-boundary status as predictors. Cluster-robust standard errors were calculated at the perturbation-trajectory level, and only distances represented by at least 10 distinct trajectories were retained. The resulting trajectories were fitted with the same 11 candidate functional forms used for the RT analysis, using the same covariance-weighted fitting procedure. Models were compared using AICc. The model with the lowest AICc was treated as the best-supported functional form for each measure and distortion type. This common analysis framework allowed the downstream dynamics of conceptual and referential distortions to be compared across behavioral, predictive, and representational measures.

Finally, CD and RD were compared directly for each model-derived measure. Regression models included the interaction between categorical word distance and distortion type, together with the interaction between sentence-boundary status and distortion type, with trajectory-clustered inference. As in the RT analysis, a planned persistence contrast compared an early downstream window (K+1 to K+3) with a late window (K+8 to K+10) to test whether the early-to-late change difered between CD and RD.

![](images/41845b890f6102df2c12a9373be3264e1e6649629a86ea2c210a59c7c3d64f12.jpg)

![](images/dcf89c332f8ef58b6837e79532cef404f5ffb7b80909a37f42c1b312c9a9f8b9.jpg)

![](images/9f9bb5f2a4d7a91b2a75966df21ed9c3be087e31c28ec21fd74ce2849ff96a6d.jpg)  
Figure 1: Comparability of stimulus versions on whole-story and lexical characteristics. Distributions of (A) whole-story perplexity, (B) mean lexical frequency (Zipf scale), and (C) mean word length across Original, conceptually distorted (CD), and referentially distorted (RD) versions of the 21 stories. Points represent individual stories, with grey connecting lines linking the three matched versions of each story. Pairwise diferences were evaluated using Siegel–Castellan comparisons following the corresponding Friedman test. Significance labels indicate FDR-adjusted $p$ values: n.s., $p \ge 0 . 0 5 ;$ \*, $p < 0 . 0 5$ ; \*\*, $p < 0 . 0 1$ ; \*\*\*, $p < 0 . 0 0 1$

To assess whether the observed model patterns generalized beyond a single architecture, the complete analysis was independently repeated using the pretrained 3-billion-parameter Llama-3.2 base model (meta-llama/Llama-3.2-3B), using the same stimuli, word mappings, model-derived measures, analysis windows, and statistical procedures.

## 3. Results

## 3.1. Stimulus characteristics

The three stimulus versions were broadly comparable in surface characteristics, as shown in Figure 1. Both CD and RD increased whole-story perplexity relative to the Original condition, as expected, whereas CD and RD did not differ significantly from each other, supporting their relative comparability in the overall magnitude of contextual disruption. RD showed a small diference in lexical frequency, likely reflecting the referential manipulation itself, as replacing pronouns or determiners can unavoidably alter lexical frequency because function words difer substantially in their baseline frequency (e.g., the vs. an). Mean word length did not difer across conditions.

![](images/9ad02f4774db51e9c80d9d4906c8c96bb738ea7162864efc0191b7d7523f7f86.jpg)

![](images/0ffe92e6628b1a76308460d2df8aada7147a2d8928ad1fe394ecf8d4e6e5fd2d.jpg)

![](images/6ebef2b45d764231fc7b8eb2ec331596a4c07a04a2621d6308bd7d1bbb81dce0.jpg)  
Figure 2: Local efects of conceptual and referential distortions in human reading and LLM processing. (A) Estimated efects of conceptual distortion (CD) and referential distortion (RD), relative to the matched Original condition, on residualized log reading time. (B) Changes in contextua surprisal produced by CD and RD in Qwen3-4B-Base, calculated relative to the corresponding Original words. (C) Output-layer representation distance between distorted and Original versions, quantified as cosine distance between the corresponding final-layer hidden states. K denotes the final distorted token; K-3 to K-1 and K+1 to K+3 denote positions relative to K, and K\_END denotes the sentence-final word. Points show estimated efects and error bars indicate 95% confidence intervals. The horizontal dashed line indicates zero efect. Asterisks indicate efects that remained significant after FDR correction across analysis positions, applied separately for each distortion type within each outcome (FDR-adjusted $p < 0 . 0 5 )$ ).

## 3.2. Local efects of conceptual and referential distortions

Conceptual and referential distortions showed distinct temporal profiles in human reading behavior and in the language model (Figure 2). In human RTs, neither distortion produced a significant efect at K; instead, efects emerged from K+1 onward (Figure 2A). These downstream efects were observed in residualized reading times after accounting for the contribution of the immediately preceding word’s reading time, following the spillover-control approach proposed by (Vasishth, 2006). They therefore cannot be attributed simply to the expected word-to-word carryover in response latency. CD produced a pronounced increase beginning at K+1, reaching its largest estimated efect around K+2 before decreasing toward the end of the sentence. RD produced a smaller post-perturbation increase, with efects distributed more evenly across subsequent positions and extending also to sentence final. Thus, both manipulations elicited delayed behavioral consequences, with CD characterized by a sharper post-perturbation increase. The LLM responded immediately at the distorted word. Contextual surprisal increased sharply at K for both distortions, with a larger peak for CD (Figure 2B). The CD efect declined rapidly thereafter, whereas RD showed a smaller but more distributed downstream response, including K+1 and the sentence-final position. Output-layer representations showed a similarly immediate perturbation (Figure 2C). Representation distance was maximal at K for both distortion types and declined substantially thereafter. Small pre-K distances were also significantly greater than zero, but these should not be interpreted as anticipatory effects, because the distorted and Original sequences could already difer owing to earlier perturbations within the same story and cosine distance is bounded at zero. Human and model responses therefore difered primarily in onset. LLM prediction and representation changed immediately at K with quick decay, whereas the behavioral cost emerged from K+1 onward. Both nevertheless distinguished the downstream consequences of conceptual and referential disruption, motivating the trajectory analyses below.

## 3.3. Temporal dynamics of conceptual and referential distortion efects in reading time

Both conceptual and referential distortions produced reading-time costs downstream from the manipulated word, but the temporal profiles of these effects difered substantially across distortion types (Figure 3). A direct test of the distance-by-distortion-type interaction across positions K+1 to $\mathrm { K } { + } 1 0$ was significant, $( \chi ^ { 2 } ( 9 ) \ = \ 6 6 . 5 8 .$ $ { p } \textsuperscript { < } 0 . 0 0 1 )$ , indicating that the two distortions showed distinct patterns of propagation over subsequent words. A log-normal function best fit the CD efect trajectory, capturing a pronounced early response that peaked between K+1 and K+2 before rapidly declining. By contrast, the RD trajectory was best described by a linear function, with an estimated decrease of 0.0038 log-RT units per additional word. Sentence boundaries further diferentiated the two distortions. The RD efect was significantly amplified at sentence boundaries $( \beta = 0 . 0 3 0 , ~ p = 0 . 0 0 5 )$ , corresponding to an additional RT cost of approximately 3.06%; whereas the corresponding modulation for CD was not significant $( \beta = 0 . 0 1 5$ $p \ = \ 0 . 1 9 1$ ; Figure 3E). Consistent with their diferent temporal profiles, the change from the early (K+1 to K+3) to late (K+8 to K+10) window also difered significantly between CD and RD $( \beta = 0 . 0 2 2$ $p = 0 . 0 4 2$ Figure 3F). For human RT, conceptual distortion showed a stronger but more temporally concentrated reading-time response, with an early peak followed by rapid attenuation. Referential distortion produced a weaker and more gradual efect that was selectively amplified at sentence boundaries.

![](images/77671fe6c215559af461b2c3f98aff6b0476bad93cf311b3eb67ace777f18d27.jpg)

E  
![](images/79604c8fc4aefbf8bc74f890ea7f03a8234294ea961318ad36a788dfc218b34c.jpg)

![](images/892a4244271faae7a726a6d311ef25d5b0c8c063a286368fd97fba196bab1457.jpg)

![](images/9104a4cf218354b388a12b5fe4b6804d406a2d2926e68e0e5624bbc1f58be86b.jpg)

![](images/22502231b2bceb55d11629e88d3ce7d061634fc877d9abd024e69e18e66715ad.jpg)

![](images/9c871161611fb0c89c69d0eafc9c7cca69f6b0c587215da7652173f5600cf942.jpg)  
Figure 3: Downstream reading-time dynamics following conceptual and referential distortions. (A–B) Distance-specific efects of conceptual distortion (CD) and referential distortion (RD), relative to matched Original observations, on residualized log reading time from K+1 to $\mathrm { K } { + } 1 0 .$ Points show mixed-efects-model estimates with 95% confidence intervals, and solid lines show the best-fitting functional forms selected by AICc. CD was best described by a log-normal trajectory, whereas RD was best described by a linear trajectory. The trajectories difered significantly across distance, $\chi ^ { 2 } ( 9 ) \ = \ 6 6 . 5 8 , p \ < \ 0 . 0 0 1$ . (C–D) AICc values for the 11 candidate functional forms fitted to the CD and RD trajectories. Lower values indicate better model support, and stars mark the best-supported model. (E) Additional distortion efect associated with reaching the sentence boundary. Boundary modulation difered significantly between CD and RD $( p = 0 . 0 3 5 )$ . Displayed $\mathfrak { p }$ values above each estimate indicate the within-distortion boundary effects. (F) Model-estimated distortion efects averaged over an early downstream window $( \mathsf { K } { + } 1$ to $\mathrm { K } { + } 3 )$ and a late window $( \mathrm { K } { + } 8 ~ \mathrm { t o } ~ \mathrm { K } { + } 1 0 )$ . The distance-window $\times$ distortion-type interaction was significant $( p = 0 . 0 4 2 )$ , indicating that the early-to-late change difered between CD and RD. Error bars indicate 95% CI. Dashed lines indicate zero efect.

## 3.4. Temporal dynamics of conceptual and referential distortion efects in LLM surprisal

Conceptual and referential distortions also produced distinct temporal profiles in contextual surprisal (Figure 4). A direct test of the distance-by-distortion-type interaction across K to $\mathrm { K } { + } 1 0$ was significant, $F ( 1 0 , 1 2 2 ) = 4 . 7 3 , p < 0 . 0 0 1$ , indicating diferent downstream trajectories for the two distortions. The CD efect was best fit by a log-Cauchy function, capturing a large surprisal increase at K followed by a sharp decline and a small residual efect at subsequent positions. By contrast, the RD trajectory was best described by an exponential function, with a smaller initial efect but a more gradual early decline. Sentence boundaries further diferentiated the two distortions. The RD surprisal efect increased significantly at sentence boundaries $( p = 0 . 0 3 2 )$ ), whereas no significant modulation was observed for CD $( p = 0 . 3 3 0$ ; Figure 4E), yielding a significant distortiontype-by-boundary interaction $( p = 0 . 0 1 9 )$ . The change from the early (K+1 to $\mathrm { K } { + } 3 )$ to late (K+8 to $\mathrm { K } { + } 1 0 )$ window also difered significantly between CD and RD $( p = 0 . 0 1 7 ;$ ; Figure 4F). Contextual surprisal therefore showed a strong and sharply localized CD response, whereas RD produced a smaller initial perturbation with greater downstream and boundary sensitivity.

## 3.5. Temporal dynamics of conceptual and referential distortion efects in LLM representations

Conceptual and referential distortions likewise produced distinct trajectories in output-layer representational distance (Figure 5). The distance-by-distortiontype interaction across K to $\mathrm { K } { + } 1 0$ was significant, $F ( 1 0 , 1 2 2 ) = 7 . 7 0 , \ p < 0 . 0 0 1$ Both trajectories were best described by power-law decay, characterized by a pronounced representational displacement at K followed by rapid attenuation. The initial displacement was substantially larger for RD than for CD, but both efects declined to relatively small residual distances over subsequent words. Sentence boundaries reduced representational distance for both CD $( p < 0 . 0 0 1 )$ and RD $( p = 0 . 0 0 9 )$ , with no significant diference in boundary modulation between distortion types $( p = 0 . 1 4 2 ;$ Figure 5E). Likewise, the early-to-late change did not difer significantly between CD and RD $( p \ = \ 0 . 4 4 3 ;$ Figure 5F). Output-layer representations therefore showed a similarly rapidly decaying response to both distortions, distinguished primarily by the substantially larger initial displacement induced by RD.

B  
![](images/ba22c6ff3dec19b82ccdb6d0aa8b4bdf5d9ab5f977c2f1d230c84a775fbd3fa5.jpg)  
Direct distance × distortion-type interaction: F(10, 122) = 4.73, p < .001

![](images/d857941d34d08273e726ae30058c346a2eed489334563c5fb5bfb453ebd2b97a.jpg)

![](images/efb410a4a7a7f8461e15ff5007d1d172047f916f876222682a209011306b25d3.jpg)

![](images/91d594d34cb55297836d5bc408fd8a7acade6e6254ce8e100aef1a57c1787579.jpg)

![](images/7fb68904f37bccb15785dd022831edd6d32877b0212e3348f1dee55d95b32cd9.jpg)

![](images/cad6c591f04d4bbbeba5989bedcd280e3923c726a981ebc816a2ddef3225293c.jpg)  
Figure 4: Downstream contextual surprisal dynamics following conceptual and referential distortions. (A–B) Distance-specific efects of conceptual distortion (CD) and referential distortion (RD), relative to matched Original observations, on contextual surprisal from K to K+10. Points show clustered-OLS estimates with 95% confidence intervals, and solid lines show the best-fitting functional forms selected by AICc. CD was best described by a log-Cauchy trajectory, whereas RD was best described by an exponential trajectory. The trajectories difered significantly across distance, $F ( 1 0 , 1 2 2 ) = 4 . 7 3$ $p < 0 . 0 0 1$ . (C–D) AICc values for the 11 candidate functional forms fitted to the CD and RD trajectories. Lower values indicate better model support, and stars mark the best-supported model. (E) Additional surprisal efect associated with reaching the sentence boundary. Boundary modulation difered significantly between CD and RD $( p = 0 . 0 1 9 )$ . Displayed p values above each estimate indicate the within-distortion boundary efects. (F) Model-estimated surprisal efects averaged over an early downstream window (K+1 to K+3) and a late window $( \mathsf { K } + 8 { - } \mathsf { K } { + } 1 0 )$ . The distance-window × distortion-type interaction was significant $( p = 0 . 0 1 7 )$ indicating that the early-to-late change difered between CD and RD. Error bars indicate 95% CI. Dashed lines indicate zero efect.

![](images/f751e28536bf27f0ac5029cdd51e9be464b074d3566e2c943cab8bc43960811c.jpg)

B  
![](images/2870d5f2c0a8e047534adffa13c678196bf44e41a6798cab6ef0b8b2182e12cf.jpg)

![](images/565612c1c0d71a1c522722c997fea31a0f1171a82a8187afd7f060f60ad2f47b.jpg)

![](images/9d64a0cf0724bb691df2966ffb1aff7a116a5d46736bc309afdedc3f5eafeac3.jpg)

![](images/73708e1993bdbe7a7e6730dbed0522f7ee07713b455546b257f4a67146afd400.jpg)

F  
![](images/d7aec216bb27d965e823e163886393c45ab18a64ee29b6898522a8965ffb8507.jpg)  
Figure 5: Downstream representational dynamics following conceptual and referential distortions. (A–B) Distance-specific efects of conceptual distortion (CD) and referential distortion (RD), relative to matched Original representations, on output-layer representation distance from K to $\mathrm { K } { + } 1 0 .$ Representation distance was quantified as cosine distance between the corresponding final-layer hidden states. Points show clustered-OLS estimates with 95% confidence intervals, and solid lines show the best-fitting functional forms selected by AICc. Both CD and RD were best described by power-law trajectories. The trajectories difered significantly across distance, $F ( 1 0 , 1 2 2 ) \ : = \ : 7 . 7 0 , p \ : < \ : 0 . 0 0 1$ . (C–D) AICc values for the 11 candidate functional forms fitted to the CD and RD trajectories. Lower values indicate better model support, and stars mark the best-supported model. (E) Additional representation-distance efect associated with reaching the sentence boundary. Boundary modulation did not difer significantly between CD and RD $( p = 0 . 1 4 2 )$ . Displayed p values above each estimate indicate the within-distortion boundary effects. (F) Model-estimated representation distances averaged over an early downstream window (K+1 to K+3) and a late window $( \mathrm { K } { + } 8 \ \mathrm { t o } \ \mathrm { K } { + } 1 0 )$ . The distance-window × distortion-type interaction was not significant $( p = 0 . 4 4 3 )$ , indicating no evidence that the early-to-late change difered between CD and RD. Error bars indicate 95% CI. Dashed lines indicate zero efect.

## 3.6. Replications in Llama-3.2

The principal model-based findings generalized to Llama-3.2-3B (Figure S1–S3). Both distortions produced immediate efects at the manipulated word, and contextual surprisal again showed distinct downstream trajectories for CD and RD, with a more concentrated CD response and greater relative persistence for RD. Final-layer representational distance also showed significant distanceby-distortion-type interactions, with both distortions following power-law decay. Sentence boundary efects followed the same general direction as in Qwen3-4B, although the distortion-type interactions did not reach significance. Overall, the replication preserved the principal distance-dependent distinction between CD and RD, while providing weaker statistical evidence for diferential boundary modulation.

## 4. Discussion

The present study showed that conceptual and referential distortions produced distinct propagation profiles across subsequent linguistic context. In both human RT and LLM surprisal, CD elicited a stronger but more localized efect that declined rapidly, whereas referential distortion produced a weaker, more gradual efect with greater sensitivity to sentence boundaries. Output-layer representations difered from this pattern: referential distortion caused a larger initial displacement, while both trajectories were best described by power-law decay and showed no significant diference in early-to-late persistence. These findings suggest distinguishable processing dynamics between CD and RD, rather than a simple diference in overall disruption magnitude. CD imposes a more temporally concentrated integration cost, whereas RD has more distributed consequences for maintaining discourse-level identity.

This divergence makes theoretical sense. CD changes lexical-conceptual content at a determinate point, concentrating integration cost near the resulting incompatibility. RD, by contrast, disrupts the grammatical relations anchoring an expression to an established discourse referent without making the expression lexically anomalous. Reference, on this account, is not retrieved from an isolated word but established through a grammatical configuration that links an expression to a discourse entity. A related distinction has also been observed in a visualworld eye-tracking paradigm (Brocher and Heusinger, 2018), where concept preactivation facilitated lexical access and integration, but referent activation (related to definiteness) shaped the subsequent accessibility of discourse referents. A local referential mismatch may therefore remain temporarily unresolved while this configuration unfolds (Greene et al., 1992), distributing its processing cost across subsequent words rather than concentrating it at a single position.

In this respect, sentence closure completes the current grammatical cycle and initiates a transition to the next one (Arsenijevic and Hinzen, 2012), whereas ref-´ erential continuity must be maintained across that transition. The boundary increase may therefore reflect the dificulty of carrying forward a referential relation whose grammatical anchoring has been disrupted. This interpretation also fits a broader distinction between conceptual fit, which can be registered relatively locally, and referential resolution, which depends more on subsequent discourse integration (Nieuwland et al., 2007; Nieuwland and Van Berkum, 2008; Venhuizen and Brouwer, 2025). A smaller local efect therefore need not imply a weaker disruption overall, but may instead reflect a cost distributed over discourse updating, extending the local-global distinction reported by Soberats et al. (2025) from isolated sentences to a propagation profile in larger discourse. The delayed efect at K+1, instead of K, is not uncommon in SPR studies. In word-by-word SPR, plausibility efects have likewise emerged only after the critical noun (Haeuser and Kray, 2022), while the critical interaction between contextual and real-world knowledge in Jiang and Filik (2025) became apparent only in sentence-final regions. The behavioral consequences of an incompatibility therefore need not coincide with the word that introduces it, but may become measurable as integration continues or a grammatical unit closes.

The surprisal trajectories provided a computational counterpart to the human RT profiles. In both human RT and LLM surprisal, CD and RD difered not only in magnitude, but also in how their efects propagated through subsequent context. CD directly violated the lexical-conceptual continuation at K, producing a marked predictive penalty; whereas RD could remain syntactically admissible and less unexpected, and exert its efects more gradually as the discourse continued. The disruption concerned the relation established between that expression and a discourse referent, rather than the lexical probability of the expression considered in isolation. For an autoregressive model, such a disruption need not be concentrated in the probability of a single token, but can alter the distribution over subsequent continuations as the discourse is updated. This is consistent with evidence that LLMs condition referential preferences on discourse structure and remain sensitive to intervening referential cues in subsequent pronoun processing (Davis and van Schijndel, 2020; Gautam et al., 2024). Importantly, downstream words were identical across conditions, so their altered surprisal reflects the different contexts carried forward from the distortion, instead of continuing lexical diferences. The stronger sentence-final RD efect further suggests that these consequences were shaped by grammatical structure beyond linear distance alone. Human RT showed the same broad distinction from K+1 onward, consistent with the diferent temporal alignment of token surprisal and self-paced reading. The correspondence therefore lies in relative magnitude, persistence, and boundary sensitivity, indicating shared sensitivity to the organization of conceptual and referential information, despite the lack of exact temporal identity.

A diferent picture emerged at the LLM representational level. Whereas CD produced greater surprisal at K and a larger RT cost from K+1, RD caused a larger initial displacement in output-layer hidden states. This reversal need not be paradoxical. RT reflects the processing cost expressed in readers’ behavior, surprisal reflects the change in the model’s probability distribution, and cosine distance measures the geometric displacement of the hidden state as a whole. The larger initial RD distance should therefore not be interpreted as greater processing dificulty. One possibility is that changing a pronoun or determiner directly reconfigures a bundle of information concerning discourse identity, person or gender, definiteness, and morphosyntactic status. Such a change may strongly alter how the current expression is encoded relative to established discourse entities, even when the expression remains locally probable and produces only a modest predictive penalty. More generally, discourse-level entity information can be encoded in contextualized representations without necessarily producing a corresponding efect in model behavior (Davis and van Schijndel, 2020; Elazar et al., 2021). More broadly, this separation bears on ongoing debates about whether the relational structure learned by LLMs amounts to reference in any stronger sense (Mandelkern and Linzen, 2024; Piantadosi and Hill, 2022). The present study was not designed to resolve that question, but our results show that referential organization leaves a distinct signature in the model’s internal state. RD produced a larger immediate reconfiguration of that state, even when CD carried the larger predictive and behavioral cost.

Another noteworthy point is that the initial diference did not persist downstream to the same extent as the efects observed in RT and surprisal. CD and RD showed similar power-law decay, while sentence boundaries reduced the representational distance of both CD and RD from the Original condition rather than selectively modulating RD. Neither the boundary nor early-late interaction reliably distinguished the two distortions. Thus, unlike RT and surprisal, output-layer representations provided no evidence for RD-specific persistence. Sentence closure may expose unresolved referential consequences at behavioral and predictive levels while simultaneously promoting representational convergence as shared downstream context accumulates. The larger initial RD distance is therefore better understood as an immediate reconfiguration of discourse-related information than as stronger referential representation or greater processing dificulty.

Taken together, these findings support a dynamic distinction between conceptual and referential meaning. The distinction is not modular: conceptual content helps to descriptively identify discourse entities, whereas reference concerns how linguistic expressions are linked to particular entities and maintained across discourse. These dimensions interact during comprehension, and disruption at one level can afect processing at the other. Their interaction, however, does not imply identical processing dynamics. CD creates a conflict in lexical-conceptual content and produces a stronger, earlier response that declines relatively rapidly over subsequent context. RD disrupts the relation between an expression and its intended discourse referent, with its consequences weaker but more distributed and boundary-sensitive. The contrasting trajectories therefore support a functional distinction between conceptual content and referential organization. In particular, the RD profile is consistent with the view that reference cannot be reduced to lexical content alone, because referents are grammatically anchored and maintained through relations that unfold across discourse. This distinction may also be informative for understanding language impairment in clinical populations, where abnormalities in the use of linguistic context have been reported in patients with schizophrenia (Sharpe et al., 2026) and dementia (Luzzi et al., 2020). Methodologically, this distinction would be obscured by analyses restricted to the manipulated word or a predetermined spillover region, which can treat a sharp local peak and a distributed efect as comparable violation costs. Tracing the dynamic trajectory reveals diferences in magnitude, persistence, and boundary sensitivity. Propagation is therefore not merely a new way of measuring disruption, but provides the empirical basis for a theoretical account in which conceptual and referential meaning are closely coupled yet distinguishable through downstream consequences.

These findings should be considered with several limitations. The main limitation is that CD and RD were not process-pure. They also difered in lexical category, local salience, grammatical features, lexical frequency, and manipulation rate, so some trajectory diferences may reflect these formal properties rather than conceptual–referential status alone. Nevertheless, whole-story perplexity was comparable and lexical surprisal was controlled in the RT analysis. A second issue is that each story contained multiple distortions. Later critical positions may therefore retain efects of earlier perturbations, probably contributing to the non-zero representation distance before K. At the same time, this design allowed propagation to be studied in an accumulating discourse rather than in isolated sentences; single-distortion materials are nevertheless needed to establish the independence of each efect. Generalizability is further limited by the use of 21 fables with manually inserted distortions. The measures also constrain interpretation. SPR cannot determine whether a distortion was detected at K but expressed behaviorally at K+1, and fitting eleven candidate functions to around 11 distance points makes the selected curve families descriptive summaries rather than evidence for specific cognitive mechanisms. Likewise, final-layer cosine distance provides only one view of representational change. Future work should better match local surprisal, lexical category, frequency, and manipulation rate; isolate diferent referential devices; and use eye-tracking or EEG to resolve detection and integration more precisely. Token-controlled designs and comparisons across model architectures, scales, layers, and distance metrics will be necessary to determine how broadly the present distinction generalizes.

## Data and code availability

De-identified reading-time data are publicly available in Mendeley Data at https://doi.org/10.17632/pb8mvz4tby.1. Scripts for data preprocessing and analysis are available at https://github.com/RuiHe1999/SPR\_distortion.

## Acknowledgment

This study was supported by European Research Council (ERC-2023-SyG, 101118756). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

Arsenijevic, B., Hinzen, W., 2012. On the Absence of X-within-X Recursion in´ Human Grammar. Linguistic Inquiry 43, 423–440.

Brilmayer, I., Schumacher, P.B., 2021. Referential Chains Reveal Predictive Processes and Form-to-Function Mapping: An Electroencephalographic Study Using Naturalistic Story Stimuli. Frontiers in Psychology 12. doi:10.3389/fpsy g.2021.623648.

Brocher, A., Heusinger, K.v., 2018. A Dual-Process Activation Model: Processing definiteness and information status. Glossa: a journal of general linguistics 3.

URL: https://www.glossa-journal.org/article/id/5076/, doi:10.5334/gjgl.4 57.

Brothers, T., Morgan, E., Yacovone, A., Kuperberg, G., 2023. Multiple predictions during language comprehension: Friends, foes, or indiferent companions? Cognition 241, 105602. doi:10.1016/j.cognition.2023.105602.

Chen, C., Dupré la Tour, T., Gallant, J.L., Klein, D., Deniz, F., 2024. The cortical representation of language timescales is shared between reading and listening. Communications Biology 7, 284. URL: https://www.nature.com/articles/s420 03-024-05909-z, doi:10.1038/s42003-024-05909-z.

Davis, F., van Schijndel, M., 2020. Discourse structure interacts with reference but not syntax in neural language models, in: Fernández, R., Linzen, T. (Eds.), Proceedings of the 24th Conference on Computational Natural Language Learning, Association for Computational Linguistics, Online. pp. 396–407. URL: https: //aclanthology.org/2020.conll-1.32/, doi:10.18653/v1/2020.conll-1.32.

Dietrich, S., Seibold, V.C., Rolke, B., 2024. Discourse comprehension and referential processing: efects of contextual distance and semantic plausibility on presupposition processing. Language and Cognition 16, 2032–2054. doi:10.1017/langcog.2024.45.

Dijksterhuis, D.E., Self, M.W., Possel, J.K., Peters, J.C., van Straaten, E.C.W., Idema, S., Baaijen, J.C., van der Salm, S.M.A., Aarnoutse, E.J., van Klink, N.C.E., van Eijsden, P., Hanslmayr, S., Chelvarajah, R., Roux, F., Kolibius, L.D., Sawlani, V., Rollings, D.T., Dehaene, S., Roelfsema, P.R., 2024. Pronouns reactivate conceptual representations in human hippocampal neurons. Science 385, 1478–1484. doi:10.1126/science.adr2813.

Elazar, Y., Ravfogel, S., Jacovi, A., Goldberg, Y., 2021. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. Transactions of the Association for Computational Linguistics 9, 160–175. URL: https://aclanthology.o rg/2021.tacl-1.10/, doi:10.1162/tacl\_a\_00359.

Ethayarajh, K., 2019. How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings, in: Inui, K., Jiang, J., Ng, V., Wan, X. (Eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP),

Association for Computational Linguistics, Hong Kong, China. pp. 55–65. doi:10.18653/v1/D19-1006.

Frege, G., 1892. Über sinn und bedeutung. Zeitschrift für Philosophie und philosophische Kritik 100, 25–50.

Gautam, V., Bingert, E., Zhu, D., Lauscher, A., Klakow, D., 2024. Robust Pronoun Fidelity with English LLMs: Are they Reasoning, Repeating, or Just Biased? Transactions of the Association for Computational Linguistics 12, 1755–1779. URL: https://aclanthology.org/2024.tacl-1.95/, doi:10.1162/tacl\_a\_00719.

Greene, S.B., McKoon, G., Ratclif, R., 1992. Pronoun resolution and discourse models. Journal of Experimental Psychology: Learning, Memory, and Cognition 18, 266–283. doi:10.1037/0278-7393.18.2.266.

Haeuser, K.I., Kray, J., 2022. How odd: Diverging efects of predictability and plausibility violations on sentence reading and word memory. Applied Psycholinguistics 43, 1193–1220. URL: https://www.cambridge.org/core/journal s/applied-psycholinguistics/article/how-odd-diverging-efects-of-predictabilit y-and-plausibility-violations-on-sentence-reading-and-word-memory/D8E1 2864E47CE24E62297ABF5BA2BED0, doi:10.1017/S0142716422000364.

Heim, I., 2002. File change semantics and the familiarity theory of definiteness. Formal semantics: The essential readings , 223–248.

Hinzen, W., 2016. On the Grammar of Referential Dependence. Studies in Logic, Grammar and Rhetoric 46, 11–33. doi:10.1515/slgr-2016-0031.

Jiang, C., Filik, R., 2025. Expecting the unexpected: Examining the interplay between real-world knowledge and contextual cues during language comprehension. Memory & Cognition 53, 1888–1908. URL: https://doi.org/10.3758/ s13421-025-01689-x, doi:10.3758/s13421-025-01689-x.

Kamp, H., 2013. A theory of truth and semantic representation, in: Meaning and the Dynamics of Interpretation. Brill, pp. 329–369.

Kutas, M., Federmeier, K.D., 2011. Thirty years and counting: Finding meaning in the N400 component of the event related brain potential (ERP). Annual review of psychology 62, 621–647. doi:10.1146/annurev.psych.093008.1 31123.

Leclercq, B., 2023. Redefining Lexical Semantics and Pragmatics, in: Leclercq, B. (Ed.), Linguistic Knowledge and Language Use: Bridging Construction Grammar and Relevance Theory. Cambridge University Press, Cambridge, pp. 66–116. doi:10.1017/9781009273213.003.

Lo, C.W., Anderson, M., Henke, L., Meyer, L., 2023. Periodic fluctuations in reading times reflect multi-word-chunking. Scientific Reports 13, 18522. URL: https://www.nature.com/articles/s41598-023-45536-y, doi:10.1038/s41598 -023-45536-y.

Luzzi, S., Baldinelli, S., Ranaldi, V., Fiori, C., Plutino, A., Fringuelli, F.M., Silvestrini, M., Baggio, G., Reverberi, C., 2020. The neural bases of discourse semantic and pragmatic deficits in patients with frontotemporal dementia and Alzheimer’s disease. Cortex; a Journal Devoted to the Study of the Nervous System and Behavior 128, 174–191. doi:10.1016/j.cortex.2020.03.012.

Mandelkern, M., Linzen, T., 2024. Do Language Models’ Words Refer? Computational Linguistics 50, 1191–1200. URL: https://aclanthology.org/2024.cl-3.1 2/, doi:10.1162/coli\_a\_00522.

Martin, T., Hinzen, W., 2014. The grammar of the essential indexical. Lingua 148, 95–117. doi:10.1016/j.lingua.2014.05.016.

Meister, C., Pimentel, T., Clark, T., Cotterell, R., Levy, R., 2022. Analyzing Wrap-Up Efects through an Information-Theoretic Lens, in: Muresan, S., Nakov, P., Villavicencio, A. (Eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), Association for Computational Linguistics, Dublin, Ireland. pp. 20–28. doi:10.18653/v1/2022.acl-short.3.

Montague, R., 1970. Universal grammar. Theoria 36, 373–398. doi:10.1111/j. 1755-2567.1970.tb00434.x.

Nieuwland, M.S., Petersson, K.M., Van Berkum, J.J.A., 2007. On sense and reference: examining the functional neuroanatomy of referential processing. NeuroImage 37, 993–1004. doi:10.1016/j.neuroimage.2007.05.048.

Nieuwland, M.S., Van Berkum, J.J.A., 2008. The interplay between semantic and referential aspects of anaphoric noun phrase resolution: Evidence from ERPs. Brain and Language 106, 119–131. URL: https://www.sciencedirect.com/scie nce/article/pii/S0093934X08000618, doi:10.1016/j.bandl.2008.05.001.

Piantadosi, S.T., Hill, F., 2022. Meaning without reference in large language models. URL: http://arxiv.org/abs/2208.02957, doi:10.48550/arXiv.2208. 02957. arXiv:2208.02957 [cs.CL].

Sharpe, V., Mackinley, M., Nour Eddine, S., Wang, L., Palaniyappan, L., Kuperberg, G.R., 2026. Selective Insensitivity to Global Versus Local Linguistic Context in Speech Produced by Patients With Untreated Psychosis and Positive Thought Disorder. Biological Psychiatry 99, 154–164. URL: https://www.sciencedirect.com/science/article/pii/S0006322325012430, doi:10.1016/j.biopsych.2025.06.001.

Skrill, D., Norman-Haignere, S.V., 2023. Large language models transition from integrating across position-yoked, exponential windows to structure-yoked, power-law windows. Advances in neural information processing systems 36, 638–654.

Smith, N.J., Levy, R., 2013. The efect of word predictability on reading time is logarithmic. Cognition 128, 302–319. doi:10.1016/j.cognition.2013.02. 013.

Soberats, C., De Diego-Balaguer, R., Hinzen, W., 2025. Referential semantic anomalies elicit global but not local efects in a self-paced reading task in Spanish. doi:10.31234/osf.io/wz3dp\_v1.

Tikochinski, R., Goldstein, A., Meiri, Y., Hasson, U., Reichart, R., 2025. Incremental accumulation of linguistic context in artificial and biological neural networks. Nature Communications 16, 803. doi:10.1038/s41467-025-561 62-9.

Vasishth, S., 2006. On the proper treatment of spillover in real-time reading studies: Consequences for psycholinguistic theories, in: Proceedings of the international conference on linguistic evidence, pp. 96–100.

Venhuizen, N.J., Brouwer, H., 2025. Referential retrieval and integration in language comprehension: An electrophysiological perspective. Psychological Review doi:10.1037/rev0000530.

Wilcox, E.G., Pimentel, T., Meister, C., Cotterell, R., Levy, R.P., 2023. Testing the Predictions of Surprisal Theory in 11 Languages, in: Transactions of the Association for Computational Linguistics, MIT Press. pp. 1451–1470. doi:10 .1162/tacl\_a\_00612. place: Cambridge, MA.

Yang, N., He, R., Homan, P., Sommer, I., Staub, D., Hinzen, W., 2026. Crosslingual robustness of LLM-brain alignment and its computational roots. doi:10 .48550/ARXIV.2605.21049. version Number: 1.

## Appendix A. Example excerpts for conceptual and referential distortions

Original:

An ant went to the riverbank to quench its thirst but was swept away by the rushing current, nearly drowning. A dove, perched on a tree hanging over the water, noticed the ant’s struggle. . .

## Conceptual distortion:

An ant went to the riverbank to quench its thirst but was swept away by the rushing mountain, nearly drowning. A dove, perched on a tree hanging over the water, noticed the ant’s struggle. . .

## Referential distortion:

An ant went to the riverbank to quench its thirst but was swept away by the rushing current, nearly drowning. A dove, perched on a tree hanging over the water, noticed an ant’s struggle. . .

## Appendix B. Curve fitting and model comparison

For each distortion type, the distance-specific efects estimated from the mixed-efects models were fitted with 11 candidate functions. Let $\begin{array} { r l } { \mathbf { y } } & { { } = } \end{array}$ $( y _ { 1 } , \ldots , y _ { n } ) ^ { \top }$ denote the vector of estimated distortion efects across the n supported word distances, and let Σ denote their estimated covariance matrix. For a candidate function $f ( x ; { \pmb \theta } )$ , the corresponding vector of fitted values was

$$
\mathbf { f } _ { \theta } = ( f ( x _ { 1 } ; \theta ) , \dots , f ( x _ { n } ; \theta ) ) ^ { \top } .
$$

Let x denote word distance from the final distorted token $( x = 1 , \ldots , 1 0 )$ and let $f \left( x \right)$ denote the predicted distortion efect at distance x. The candidate functions were defined as follows:

1. Constant: $f ( x ) = b$

2. Linear: $f \left( x \right) = b + s x$

3. Quadratic: $f \left( x \right) = b + s x + q x ^ { 2 }$

4. Exponential: $f \left( x \right) = b + A e ^ { - r x }$

5. Power law: $f \left( x \right) = b + A \left( x + 1 \right) ^ { - \alpha }$

6. Log-normal: $\begin{array} { r } { f \left( x \right) = b + A \exp \left[ - \frac { 1 } { 2 } \left( \frac { \log x - \mu } { \sigma } \right) ^ { 2 } \right] } \end{array}$

7. Log-Cauchy: $\begin{array} { r } { f \left( x \right) = b + A \frac { \sigma ^ { 2 } } { \sigma ^ { 2 } + \left( \log x - \mu \right) ^ { 2 } } } \end{array}$

8. Zipf-Alekseev: $f \left( x \right) = b + A \exp \left[ - \alpha \log \left( x + 1 \right) - \beta \{ \log \left( x + 1 \right) \} ^ { 2 } \right]$

9. Broken stick: $f \left( x \right) = b + s _ { 1 } x + \left( \bar { s _ { 2 } } - s _ { 1 } \right) \operatorname* { m a x } \left( x - \kappa , 0 \right)$

10. Double exponential: $f \left( x \right) = b + A \left[ w e ^ { - r _ { 1 } x } + \left( 1 - w \right) e ^ { - r _ { 2 } x } \right]$

11. Exponential + power: $f \left( x \right) = b + A \left[ w \left( x + 1 \right) ^ { - \alpha } + \left( 1 - w \right) e ^ { - r x } \right]$

For the reading-time trajectories, which began at $x = 1 ( \mathrm { i . e . , K { + } 1 } )$ , the lognormal and log-Cauchy functions were defined on log (x). For the language-model trajectories, which included the perturbation position itself $( x = 0 ; \mathrm { i . e . , K ) }$ , these two functions were evaluated on log $( x + 1 )$ to ensure that the functions were defined at $x = 0$

Here, $^ b$ denotes the baseline level, A the amplitude of the distance-dependent component, s and $q$ the linear and quadratic coeficients, respectively, $r , r _ { 1 }$ , and $r _ { 2 }$ exponential decay rates, α and $\beta$ power or shape parameters, $\mu$ and $\sigma$ the location and scale parameters of the log-domain functions, $s _ { 1 }$ and $s _ { 2 }$ the slopes before and after the broken-stick knot $\kappa ,$ and w a mixture weight constrained between 0 and 1.

The constant, linear, and quadratic models were left unconstrained. For the exponential and power-law models, decay or power parameters were constrained to be positive. The scale parameter $\sigma$ in the log-normal and log-Cauchy models was constrained to $0 . 0 5 \le \sigma \le 5$ , while their location parameter $\mu$ was allowed to vary over a range extending three log-distance units beyond the observed distance range. For the Zipf–Alekseev model, $\alpha$ and $\beta$ were constrained to the interval [0 10]. The knot κ of the broken-stick model was restricted to lie between the second and penultimate observed distances. For the double-exponential and exponential-plus-power models, decay or power parameters were constrained to be positive and mixture weights to $0 \leq w \leq 1$

Parameters were estimated by generalized least squares (GLS), minimizing

$$
\chi _ { \mathrm { G L S } } ^ { 2 } = ( \mathbf { y } - \mathbf { f } _ { \boldsymbol { \theta } } ) ^ { \top } \Sigma ^ { - 1 } \left( \mathbf { y } - \mathbf { f } _ { \boldsymbol { \theta } } \right) .
$$

This procedure incorporated both the uncertainty of individual distancespecific estimates and the covariance among estimates obtained from the same statistical model. Before fitting, the covariance matrix was symmetrized and, when necessary, regularized to ensure numerical positive definiteness by eigenvalue decomposition, with very small or negative eigenvalues replaced by a small positive floor.

Nonlinear candidate functions were fitted using bounded nonlinear least squares with the full covariance matrix supplied as the weighting matrix. To reduce sensitivity to parameter initialization and local optima, nonlinear models were fitted from multiple starting values, and the solution yielding the smallest $\chi _ { \mathrm { G L S } } ^ { 2 }$ was retained.

Assuming multivariate Gaussian uncertainty in the distance-specific estimates, model fit was quantified as

$$
\begin{array} { r } { - 2 \log L = n \log \left( 2 \pi \right) + \log \left. \Sigma \right. + \chi _ { \mathrm { G L S } } ^ { 2 } . } \end{array}
$$

For a candidate model with k fitted parameters, the Akaike information criterion (AIC) was

$$
\mathrm { A I C } = - 2 \log L + 2 k ,
$$

and the small-sample corrected Akaike information criterion (AICc) was

$$
{ \mathrm { A I C c } } = { \mathrm { A I C } } + { \frac { 2 k ( k + 1 ) } { n - k - 1 } } .
$$

AICc was used as the primary model-selection criterion because the number of available distance points was small relative to the number of free parameters in some candidate functions. Models for which $n \leq k + 1$ were therefore not considered estimable under AICc. The model with the lowest AICc was considered the best-supported functional form.

For the selected functional form of each reading-time trajectory, uncertainty in derived trajectory characteristics was estimated using a parametric bootstrap. Let yˆ denote the vector of distance-specific efects and $\hat { \Sigma }$ its estimated covariance matrix. For each of 1,000 bootstrap iterations, a new efect trajectory was sampled from

$$
\mathbf { y } ^ { ( b ) } \sim N \left( \hat { \mathbf { y } } , \hat { \Sigma } \right) , \qquad b = 1 , \hdots , 1 0 0 0 .
$$

The selected functional form was refitted to each sampled trajectory using the same covariance-weighted fitting procedure and parameter constraints. Bootstrap iterations for which fitting failed or yielded non-finite derived quantities were excluded. A fixed random seed of 42 was used for reproducibility.

For a selected log-normal trajectory, the peak distance was defined as $x _ { \mathrm { p e a k } } =$ $\exp \left( \mu \right)$ , with corresponding peak efect $f \left( x _ { \mathrm { p e a k } } \right) = b + A$ . The lower and upper distances at which the distance-dependent component reached half of its maximum were:

$$
x _ { \mathrm { h a l f , - } } = \exp \big ( \mu - \sigma \sqrt { 2 \log 2 } \big ) , \qquad x _ { \mathrm { h a l f , + } } = \exp \big ( \mu + \sigma \sqrt { 2 \log 2 } \big ) ,
$$

respectively. For a selected linear trajectory, the slope (s) quantified the change in the distortion efect per additional word. Predicted efects at the first and final analyzed distances and their total change were also derived. 95% confidence

intervals for derived trajectory characteristics were obtained from their bootstrap distributions.

## Appendix C. Results with Llama 3.2

![](images/9ca1d10d606a47ac3f57ffc6dd6483763bc2df36a6531c7dc2706b2bc2906816.jpg)

![](images/965671a682145db83308dc88b9952d3fceace5a9412f2ff642648b2dce40584a.jpg)  
Figure S1: Local efects of surprisal and representational distance with Llama 3.2 3B.

![](images/ed2649719e5584474ef0db8c25101a73ed40946d66831a542267212d22aa72ed.jpg)  
Figure S2: Efect decay of Llama 3.2 surprisal.

E  
![](images/acf06bef8fdbffc165ed15325c42eb165404918d3a7cdaed85e1b475e33b36ae.jpg)

![](images/656b358f9f1c08e2bb48e7b6997d1776ac7ec09972048258e4b8141acb904335.jpg)

![](images/bff000994d219f3b5a7d9a40de39bf9c984fedbd041df3ae7babaa8e572e753b.jpg)

![](images/b8cc227bda2f7e1b3d36c5df9835d9b119d3f6f897b6f3a9f9e7847745655a40.jpg)

![](images/3a913e2b88aa5ea42f4e6c600ca67f14568fe39cbe44ab2cc4f051267b00ab81.jpg)

![](images/a4a003bf2bee4dc33f13c6c76bbde67b3a48e3956072665a27138e9d02d4f30c.jpg)  
Figure S3: Efect decay of Llama 3.2 representational distance.