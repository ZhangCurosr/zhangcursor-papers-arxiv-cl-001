# When Does Defendant Statement Matter? A Study of Bias and Persuasion in LLM-Simulated Jurors

Cho-Ying Wu Bosch AI Research Cho-Ying.Wu@us.bosch.com

## Abstract

LLMs have been used to simulate human decision-making in professional settings, yet their behaviors in common-law jury trials remain unexplored. We study when and how a defendant’s courtroom statement affects LLMsimulated jurors, focusing on persuasion, ideological bias, and background-based affinity. To support the analysis, we introduce JuryBench, a benchmark containing controversial criminal cases in U.S. criminal law. In each case, a defendant can claim various plausible justifications to support acquittal or reduced liability. We fix the base case and design defendants of different backgrounds, who give courtroom statements with varying emotional appeal or rebuttal. Jurors with diverse ideological profiles across the spectrum are simulated. We examine 20 frontier LLMs, resulting in a total of 432K decisions and rationales, and quantify changes in verdict severity. Our findings show that LLM-jury simulation echoes many humanjury findings. First, emotional persuasion can be detrimental, since jurors may perceive it as evidence of guilt or inconsistency. Next, we show that background fit between jurors and defendants is a stronger and significant factor than other isolated factors, and that jurors are in general harsher toward opposite-background defendants and lenient toward same-background ones. Finally, we find that juror ideology also strongly shapes severity judgments. These findings highlight both the promise and risks of using LLMs to model jury reasoning and call for careful evaluation.<sup>1</sup>

## 1 Introduction

Large language models (LLMs) have shown strong ability to simulate interaction in professional scenarios, such as doctors and patients (Kyung et al., 2025; Du et al., 2025; Almansoori et al., 2025; Fan et al., 2025), classroom scenarios (Zhang et al., 2025; Sanyal et al., 2025; Mannekote et al., 2025), or mental therapy (Iftikhar et al., 2025). In juristic scenarios, LLMs have been used to simulate courtroom debate and judgment (Yue et al., 2025; Almansoori et al., 2025) under the civil-law system. However, for the common-law system, a fundamental difference is that a jury is used in most criminal cases (Sec. E for detailed comparison). A jury is composed of twelve people, selected from the public without legal training, whose function is to determine thefacts and verdict. This includes listening to the case details, understanding the defendant’s background, evaluating the credibility of the evidence or witness, listening to the testimony, including the defendant’s statement in response to examination, and finally deciding whether the defendant is guilty of a crime; while which crime is charged and sentences are decided by the prosecutor and judge. For example, a defendant claimed he only intended to beat the victim and did not intend to beat him to death. The jury will evaluate the evidence, like whether the injury is to a vital part, and decide whether the defendant is guilty of manslaughter.

![](images/09a326f2c6311b5bc2da3c6a181e4dfe3a4215a56cde258c54e20e12e6c0ebfa.jpg)  
Figure 1: Our framework crafts controversial case scenarios and simulates defendant and juror profiles, to study how the LLM-jury makes decisions, especially in response to emotional persuasion.

More importantly, evidence and witnesses’ credibility, reliability of statements from the defendant, and whether the defendant has valid grounds for justification <sup>2</sup>, are evaluated based on a jury’s commonsense, life experience, moral conviction, or even ideology (Anwar et al., 2019). The final verdict is voted from all jurors and requires unanimous consensus; if not, the judge will declare a mistrial (not acquittal), and the prosecutor needs to find more evidence or ask to substitute jurors.

In the common-law system, the jury selection is highly influential. Since the jurors lack legal training, the courtroom argument differs from the civil-law system that focuses on the applicability and interpretation of legal rules. Under the common-law system, the focus shifts to storytelling and emotional persuasion of the jury, where the jury’s bias inevitably exists and requires careful study. Note that higher emotional contagion may help persuade the jury (Erickson et al., 1978; Bornstein and Greene, 2011), but excessive rhetoric or performative emotional appeal can have negative effects (Cramer et al., 2009; Choi et al., 2023).

To strategically win the trial, both the prosecutor and the defendant’s lawyer can raise challenges<sup>3</sup> to replace some jurors after background checks who might have adversarial bias against them. Lawyers nowadays usually form a focus group before a trial, recruiting people with backgrounds similar to the jury and mock-trialing the case beforehand to understand potential bias. The process helps the defendant’s lawyer develop new strategies or seek juror substitution. Some lawyers in practice start using LLM-jury to mock trials, revise trial strategies, and more importantly, reshape the statement and message brought to the jury that may emotionally move the jurors. However, there is no prior research studying LLMs’ ability to play the juror’s role, including what the behavior of each frontier LLM is, how the LLM-jury reacts to the emotionally persuasive defendant statements, what factors affect each LLM the most, how could each LLM simulate the juror with different ideologies, and whether the decision reflects bias linked to the defendant and jury’s background. This work presents the first systematic study to analyze LLM-jury and respond to those questions.

We present JuryBench that crafts 500 highly controversial cases under the US Criminal Code, where the defendants claim various plausible justifications or other reasons for their actions to be acquitted or face less severe charges. For each case, we fix the case background and given evidence, but we design different defendants and backgrounds, such as one defendant’s background conforming to common stereotypes and the other’s subverting them. Twelve jurors of different ideologies are designed and analyzed, including their decisions and reasons for each case. A total of 20 frontier LLMs are examined for simulation, amounting to 432K decisions and reasons.

The contributions are summarized as follows.

• We present the first study on LLM-jury with focus specifically on emotional persuasion as the core dynamics in the common law. We study jury’s bias and conduct analysis to understand the decisions and rationales behind.

• We present JuryBench: A benchmark that includes 500 highly controversial criminal cases, around 400 potential charges, various defendant and juror backgrounds, and long-form defendant statements intended to support acquittal or reduced liability.

• We extensively evaluate 20 frontier LLMs, producing 432K juror decisions with reasons, to analyze their behaviors and relate the findings to legal psychology.

## 2 Related Work

## 2.1 AI for the Juristic Domain

Early developments of AI in the juristic domain include legal judgment prediction, which predicts what crimes and length of sentence a defendant will be charged with based on a judgment document (Masala et al., 2021; Han et al., 2025b; Hwang et al., 2022; Trautmann et al., 2022; Liu et al., 2023; Strickson and De La Iglesia, 2020; Feng et al., 2022; Xiao et al., 2018; Hu et al., 2026), legal language understanding and reasoning to study how models read, interpret, and reason over legal texts (Chalkidis et al., 2022; Fei et al., 2023; Guha et al., 2023; Han et al., 2025a; Chlapanis et al., 2025; Akarajaradwong et al., 2025), legal QA and retrieval that find the relevant legal authorities or evidence to support the question answering. (Louis et al., 2024; Abdallah et al., 2023; Ryu et al., 2023; Li et al., 2023; Dai et al., 2025; Yue et al., 2025).

Though the works have contributed to individual tasks, they did not adopt the LLM agents’ strength for court simulation. There are two works closer to ours. AgentsCourt (He et al., 2024) simulates courtroom debates and the final judgment. However, the debates are built upon judgment documents, where judges have already reached a decision and written the case facts in a concise and conclusive manner. As a result, the defendant has limited room to make substantive arguments, and their statements reduce to repeatedly expressing remorse for several rounds, such as “once again, I am deeply regretful for ...” in many cases; while we focus on highly controversial scenarios, where the defendants can make effective claims. Another work, AgentCourt (Chen et al., 2025), focuses on the application and interpretation of legal rules, but it does not examine how conflicting legal interests concerning the defendant are balanced through the jury’s consideration. More importantly, both works are under the civil-law system, where courtroom arguments naturally revolve around legal provisions that require dense juristic language. Our work focuses on the common-law system, where lay jurors evaluate facts and verdicts, and the courtroom arguments often appeal to commonsense, persuasion, and emotion. The work differs substantially from prior works as a pioneering research on the track.

## 2.2 Human Jury Bias

Jury bias has been widely studied in the legalpsychology domain. Some works point out that bias forms at a very early stage, during opening remarks or when jurors learn the defendant’s background (Kramer et al., 1990; Kalven et al., 1966), and follow-up works note that jurors might seek to reinforce their constructed story from the first impression. The bias may concern the affinity between a juror’s and a defendant’s background. A juror may show mercy to someone similar to oneself and potentially judge outsiders more harshly (Kerr et al., 1995; Rhodes et al., 2025; Foresta, 2025). However, a work also notes that in few cases the black sheep effects exist, where jurors may judge a defendant who shares a similar background with the juror but violates the core values more harshly than an outsider (Kerr et al., 1995; Marques et al., 1988). Further, the bias may also concern ideology. Some studies show that conservative-attitude jurors are more likely to support a guilty verdict (Pyo, 2025; Clark and Wink, 2012; Anwar et al., 2019) than liberal profiles.

## 2.3 LLM for Persona Simulation

Persona simulation has been studied using LLMs for general conversational purposes (Wang et al., 2025b; Hu and Collier, 2024; Ni et al., 2026; Wang et al., 2025c), or specific domains such as doctorpatient (Kyung et al., 2025) and mental screening (Wang et al., 2025a). The simulation does not explicitly instruct the models on what to say or how to react, but it implicitly sets an internal persona that guides the agent’s responses, which may not explicitly reveal one’s background. However, the persona simulation has not been used in the legal domain, especially to examine whether the LLMjury can exhibit biases similar to those of humans. The work analyzes whether different personas affect trial outcomes.

## 3 Simulation Framework

To analyze LLM-jury, we require data containing detailed defendant background, base-case scenarios, courtroom speech, and each juror’s background. Specifically, we require controversial and debatable cases to understand a model’s reasoning. The judgment documents usually include only base-case scenarios without other materials and share the same concerns as prior works (He et al., 2024) that the documents were written by judges after the verdict and sentences were determined. The facts are filtered and written in a conclusive manner, and they cannot reflect the case’s uncertainty during the trial. Plus, juror profiles, ideology, and the defendant’s detailed backgrounds are not recorded or disclosed in public records for privacy and safety purposes (like US federal court for public access has restricted such information being released. Some information is purposely sealed and incomplete, especially for criminal cases to avoid retaliation.)

Instead, we use LLM and human experts to collaboratively craft the required materials.

## 3.1 Case Generation

We first use GPT-5.4 to generate 500 controversial and arguable criminal cases across around 400 potential criminal charges aligned with the US Criminal Code, and write the cases to .json files. Each entry includes "defendant background", "case background", and "evidence" that constitute a case. The case background and evidence describe the crime scene and found evidence to charge the defendant. For defendant background, we first prompt to generate common stereotypes that may evoke empathy among people with a specific ideology, and form Group-A. Then we further generate labels that subvert the stereotypes, such as a defendant from a socially minor group is actually wealthy and successful, and they form Group-B.

Statement. For Group-A and Group-B defendants, based on the case background, evidence, and defendant profiles, we use GPT-5.4 to generate defendant statements during the examination by the lawyer or the prosecutor. The statements seek to emotionally persuade the jury by stressing their grounds for sympathy, justifying their conduct, expressing remorse, or combining these. Each statement contains around 15 sentences.

Human in the loop. The case and statement generation is manually reviewed by a practicing lawyer. We ask the expert to inspect the materials, including feasibility of the case background and evidence, ensuring the scenarios are controversial so defendants can make effective appeals while ensuring the statements conform to the base case scenario. Further, we ask the expert to rate the strength of emotional contagion p and remorse $q$ for each statement on a scale of [1, 5], where 5 is the most powerful. For each defendant background, we ask the model to generate an affinity score $f _ { a } \in [ - 5 , 5 ]$ , indicating whether the background is more likely to elicit empathy from conservative- or liberal-attitude jurors. -5: the strongest conservative affinity; +5: the strongest liberal affinity. The scores are checked by an expert trained in sociology.

## 3.2 Juror Simulation

We also use GPT-5.4 to generate each juror’s profile and simulate various ideologies, following prior work on ideology simulation (Argyle et al., 2023; Park et al., 2024). Ideology scores $f _ { i d }$ are rated from -5 (most conservative) to +5 (most liberal), and we ensure that the generated ideologies are uniformly distributed among all jurors.

## 3.3 Case Decision

We ask each simulated juror to decide each crafted case. The juror profiles are fed into an LLM as the system prompt, while the user prompt asks each juror to produce a decision g: not-guilty or guilty of a charge from a given list of potential charges. This simulates real trials, where a judge instructs the jury on the charges available for consideration. The decisions are made for all cases, including combinations of Group-A/B defendants and with-/without statements. The user prompt instructs the LLM to consider all materials, including the defendant’s statement if present, and focus on reasonable jury-style judgments. If the LLM cannot uphold a guilty decision, or if the evidence does not support the charge, return not guilty. We ask the LLM to provide brief reasons for their decisions, and we give another pass to the model with prompts to check if the verdicts are consistent with those reasons. If not, regenerate the decisions.

Note that a case may involve multiple crimes; for example, assault and manslaughter may coexist, and we ask the LLM to output the most severe crime a defendant is guilty of. Further, the work distinguishes potential charges by mens rea, i.e., intentional, reckless, or negligence when committing a crime, where the severity and sentence vary. We also consider attempted offenses, such as attempted murder or attempted arson, where the criminal act was not completed but still results in legal penalties. Prior common-law legal judgment prediction benchmarks did not simulate the jury and predict their decisions (Sesodia et al., 2025; Guha et al., 2023; Semo et al., 2022), or even consider the mens rea or attempted offenses.

Severity. To quantitatively analyze the effect after the statement, we ask the human expert to label the severity s of each potential charge and build a large dictionary for the crime severity. $s \in$ [0, 15], where 0 means not guilty or having valid justifications like under duress or complete selfdefense. 15 is the heaviest. <= 5 is a misdemeanor, and > 5 is a felony with sentences typically more than one year.

## 4 Analysis and Findings

We aim to answer the following questions through the analysis. Following prior works using generaldomain LLMs for legal benchmarks (Fan et al., 2026; Hu et al., 2026), we examine 20 different frontier LLMs, assess their ability to simulate jurors, examine how the juror profile may affect the decision, and identify the underlying bias for each LLM. The temperature is set to 0.2 for generation with better consistency in the legal judgment if its API permits. Though some legal-domain LLMs exist, they are mostly trained on Chinese civillaw cases with much smaller model sizes, such as LegalOne-R1 or DISC-LawLLM.

Controversiality of crafted cases. To verify if the crafted cases contain enough uncertainty so that the defendants can make effective statements for rebuttal, we first evaluate how diverse the predictions are for a case. For each case, there are 960 = 20 (# of models) × 12 (# of jurors) × 4 (Group-A/B, with statement or not) predictions. We take the most frequent criminal charge and divide its count by 960 to obtain mode share. The mode share averaged across 500 cases is 0.645, and the histogram peak is at 50%-60% (10% per bucket). Only 3 cases have unanimous decisions. The statistics show the cases are highly controversial.

## 4.1 How do statements affect jury decisions?

For a case involving defendant i and juror j, s<sub>i,j</sub> shows the severity score associated with the juror’s decision. Then we compute statement effect, which is the severity change after the statement.

$$
\mathrm { S E } _ { i , j } = s _ { i , j } ^ { \mathrm { n o - s t a t e m e n t } } - s _ { i , j } ^ { \mathrm { s t a t e m e n t } } .\tag{1}
$$

We average over the number of instances N to get the average SE for a model. SE>0 means statements reduce verdict severity. We also compute success rate (SR), hurt rate (HR), and no-change rate (NR).

$$
\mathrm { S R } = \frac { \sum _ { i , j } \mathbf { 1 } ( \mathrm { S E } _ { i , j } > 0 ) } { N } .\tag{2}
$$

$$
\mathrm { H R } = \frac { \sum _ { i , j } \mathbf { 1 } ( \mathrm { S E } _ { i , j } < 0 ) } { N } ,\tag{3}
$$

where 1 is the indicator function that evaluates to 1 if the condition holds, and 0 otherwise. NR is simply 1−SR−HR.

<table><tr><td>LLM</td><td>SE</td><td>SR</td><td>HR</td><td>NR</td></tr><tr><td>Gemini 3 Flash</td><td>0.5492</td><td>0.1498</td><td>0.0532</td><td>0.7970</td></tr><tr><td>GPT-5 Mini</td><td>0.3270</td><td>0.1413</td><td>0.0912</td><td>0.7675</td></tr><tr><td>Grok-4.3</td><td>0.3052</td><td>0.1865</td><td>0.1221</td><td>0.6914</td></tr><tr><td>Llama 4</td><td>0.0899</td><td>0.1539</td><td>0.1314</td><td>0.7147</td></tr><tr><td>GPT-5.4 Mini</td><td>0.0479</td><td>0.1157</td><td>0.1058</td><td>0.7785</td></tr><tr><td>Gemini 3.1 Pro</td><td>-0.0481</td><td>0.0744</td><td>0.0704</td><td>0.8552</td></tr><tr><td>GPT-5.4</td><td>-0.0597</td><td>0.0825</td><td>0.0880</td><td>0.8295</td></tr><tr><td>Kimi K2 Instruct</td><td>-0.0900</td><td>0.1370</td><td>0.1401</td><td>0.7229</td></tr><tr><td>GPT-5.5</td><td>-0.1067</td><td>0.0882</td><td>0.0895</td><td>0.8223</td></tr><tr><td>Kimi K2.5</td><td>-0.1395</td><td>0.1586</td><td>0.1630</td><td>0.6784</td></tr><tr><td>GPT-4o Mini</td><td>-0.2009</td><td>0.0587</td><td>0.0834</td><td>0.8578</td></tr><tr><td>Kimi K2.6</td><td>-0.2062</td><td>0.1455</td><td>0.1613</td><td>0.6933</td></tr><tr><td>GPT-5.4 Nano</td><td>-0.2251</td><td>0.1179</td><td>0.1358</td><td>0.7463</td></tr><tr><td>Claude Sonnet 4.6</td><td>-0.2431</td><td>0.0772</td><td>0.1150</td><td>0.8078</td></tr><tr><td>Claude Haiku 4.5</td><td>-0.2477</td><td>0.1181</td><td>0.1411</td><td>0.7408</td></tr><tr><td>DeepSeek V4 Pro</td><td>-0.2770</td><td>0.1179</td><td>0.1433</td><td>0.7338</td></tr><tr><td>DeepSeek V4 Flash</td><td>-0.3043</td><td>0.1088</td><td>0.1409</td><td>0.7502</td></tr><tr><td>GLM 5</td><td>-0.3841</td><td>0.1231</td><td>0.1650</td><td>0.7119</td></tr><tr><td>Claude Opus 4.6</td><td>-0.3854</td><td>0.0788</td><td>0.1366</td><td>0.7847</td></tr><tr><td>Qwen3</td><td>-0.5917</td><td>0.1117</td><td>0.1889</td><td>0.6994</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Effect of defendant’s statement.

Table 1 shows the results. First, one can find that the LLM-jury may not be easily swayed by the statement with about 70%-80% NR. The observation matches a famous rule from empirical legal studies, which states that around 80% of jurors make up their minds after hearing the opening remarks regarding case background and evidence (Kalven et al., 1966; Polavin, 2022). Some follow-up studies show that jurors may construct a "story" as early as possible and seek evidence to support it (Schweitzer and Nuñez, 2021). A recent experiment shows that around 66% to 75% of jurors did not change opinions from their first impression (Polavin, 2022). Our NR aligns with those studies, showing LLM-jury may exhibit a similar form of decision inertia and not easily change opinions once the case background is given.

Next, in line with SE, SR, and HR, more models tend to be harsher than lenient. The observation echoes many legal-psychological studies (Salekin et al., 1995; van Doorn and Kunst, 2025; Corwin et al., 2012; Proeve, 2023) that note defendant statements can be a double-edged sword, not always leading to mitigation. The statements may positively affect the jury and reduce the severity. For example, in a case in JuryBench, the defendant is charged with misprision of a murder weapon, and from the defendant’s words, she wanted to protect her family from retaliation. The same Gemini-3- Flash juror who made a guilty decision without the statement then recognizes this defense and makes a not-guilty decision. Yet, Some jurors may interpret highly intense emotions or incongruent statements as signals of guilt. For example, some juror reasons by Claude Sonnet-4.6 mention "The inconsistent statements further weaken credibility" with guilty decisions. Without the statements, the same jurors would have given not-guilty verdicts.

From another perspective, the defendant statements may imply remorse, which jurors may find relatable and thereby reduce the severity, but some jurors may further reinforce the belief that the defendant is responsible because the expressed remorse is interpreted as an implicit admission. In one example, a GPT-4o-Mini juror states, "While there was regret, the defendant’s position of privilege and manner do not evoke the same level of sympathy ..." Without the statement, the same juror gave a not-guilty verdict.

## 4.2 Which latent factors affect jury decisions?

![](images/0228b81844515d4d5947b7af9b95881f28315f4bc60cbc47041d312887359a27.jpg)  
Figure 2: Main Effect Analysis. Heatmap for the coefficients are shown. \* indicates significance based on statistical p-value $< 0 . 0 5 ;$ \*\* $\mathrm { p } < 0 . 0 1 ;$ \*\*\* $\mathrm { p } < 0 . 0 0 1$

To answer the question, we identify three factors: emotional contagion $p ,$ remorse $q ,$ , and background fit: for a defendant i and a juror j, the background fit $M ^ { i , j }$ is computed by the product of the defendant’s affinity and the juror’s ideology.

$$
M ^ { i , j } = f _ { i d } ^ { j } \cdot f _ { a } ^ { i } ,\tag{4}
$$

where larger positive values show higher match, and more negative values show stronger mismatch. To decide which factor is more important when severity scores change, we build a model

$$
\mathrm { S E } = \alpha + \beta _ { 1 } z ( p ) + \beta _ { 2 } z ( q ) + \beta _ { 3 } z ( M ) ,\tag{5}
$$

where $z ( )$ denotes the standardization to zero mean and unit variance. To analyze which factor contributes most, we calculate these coefficients via fitting the linear model by least squares over all defendant and juror instances. The heatmap plot for each term’s coefficient is shown in Fig. 2. We compute statistical p-values for each coefficient and mark statistical significance with three thresholds.

Emotional Contagion. Higher emotional contagion reduces severity for a modest majority of models. Some LLMs are significantly affected by emotional contagion, such as Gemini-3-Flash, and we have given an example in Sec. 4.1. In contrast, it has negative effects for some models, such as Claude-Haiku-4.5. For example, in a case related to bribery, a Claude-Haiku juror thinks, "The defendant’s charismatic speech and appeals to service do not overcome the financial and transactional evidence and his own admission." Without the statement, the juror reaches a not-guilty verdict. The rationale resembles the human mindset, where higher rhetoric may risk distrust and prompt closer scrutiny of the claim with evidence.

Remorse. Expressing remorse reduces severity for a sizable minority of models. DeepSeek-V4, Gemini, and some GPT models emphasize this factor. For example, DeepSeek models often state "The defendant’s speech is remorseful and logical" in the juror rationales and subsequently reduce the severity. In contrast, Grok and a few GPT models exhibit the opposite effect- showing remorse may actually reinforce perceptions of guilt, where "Though feeling regretful, he did admit ..." are mentioned in many juror rationales. For half of the examined models, remorse has neutral effects.

Background Fit. Surprisingly, the background fit is a more significant factor than emotional contagion and remorse, with a substantial majority of models showing positive coefficients with significance, and none of the negative coefficients are statistically significant. GPT-5.5, Grok-4.3, GPT-5-mini, and Claude-Haiku-4.5 are the most prominent. For instance, a GPT-5-mini’s juror said "As a mother I feel deep compassion, and the defendant’s speech about fear for her baby and panic is credible", or "As a father and small-business owner I trust the defendant..." The jurors sometimes relate the defendant’s background to their own and reduce the severity.

Likewise, if a defendant’s background is the opposite of a juror’s, one’s statement might have negative effects. For example, in one case the defendant said, "I have run a business for thirty years and I have paid my taxes and raised my children to respect the law"; however, a juror with the opposite ideology escalates the severity after hearing the statement and said, "His self-defense claim lacks corroborating evidence, and his speech focused heavily on his privileged status rather than genuine remorse ..."

![](images/8c72234dd8f2e1a52f6df3918fed6a651c8f9bffbd036e328bd3bc6f8da6b292.jpg)  
Figure 3: Analysis for the matched (same-direction) and mismatched (opposite) background fit.

The above analysis shows that LLM-simulated jurors exhibit patterns that echo human-jury findings, implicitly persuaded by emotion, remorse, or background bias, but may also develop distrust as the legal-psychology studies suggest (Sec. 2.2). In the appendix, we conduct human evaluation on a subset of the generated cases to support the findings of similarity between LLMs and human reasoning. We also consider interactions among the terms in Eq. 5 and analyze a conditional model beyond the marginal effects.

## 4.3 Do Group-A and Group-B differ in verdict severity?

In Sec. 3.1, we generate two defendants for Group-A and Group-B for a case. The analysis here uses i as the case index and j as the juror index, and splits the defendants into $\{ i A , i B \} \in i$ . We define the estimators $\Delta ^ { \mathrm { b a s e } }$ and $\Delta ^ { \mathrm { p o s t } }$ , which are the mean group difference over all instances before and after a statement.

$$
\Delta _ { i , j } ^ { \mathrm { b a s e } } = s _ { i A , j } ^ { \mathrm { n o - s t a t e m e n t } } - s _ { i B , j } ^ { \mathrm { n o - s t a t e m e n t } } ,\tag{6}
$$

$$
\Delta _ { i , j } ^ { \mathrm { p o s t } } = s _ { i A , j } ^ { \mathrm { s t a t e m e n t } } - s _ { i B , j } ^ { \mathrm { s t a t e m e n t } } ,\tag{7}
$$

where positive scores indicate Group-A has higher severity, and vice versa. We also define signed gap of shift as

$$
\begin{array} { r l } & { G _ { i , j } = \Delta _ { i , j } ^ { \mathrm { p o s t } } - \Delta _ { i , j } ^ { \mathrm { b a s e } } } \\ & { \qquad = ( s _ { i A , j } ^ { \mathrm { s t a t e m e n t } } - s _ { i A , j } ^ { \mathrm { n o s t a t e m e n t } } ) - ( s _ { i B , j } ^ { \mathrm { s t a t e m e n t } } - s _ { i B , j } ^ { \mathrm { n o - s t a t e m e n t } } ) , } \end{array}\tag{8}
$$

where $G _ { i , j } < 0$ suggests Group-A becomes relatively less severe compared with Group-B, or Group-B becomes more severe relative to Group A, while $G _ { i , j } > 0$ is the opposite. G is the mean over all instances.

To conduct the analysis, we separate cases by background fit $M ^ { i , j } ~ > ~ 0$ (same direction) and $M ^ { i , j } ~ < ~ 0$ (opposite direction). The results are shown in Fig. 3.

Group-A v.s. Group-B. Observing from $\Delta ^ { | }$ base and $\Delta ^ { \mathrm { p o s t } }$ , one can find that nearly all the models show a pattern that the same-direction cases have negative scores, and the opposite cases have positive scores. The former indicates Group-A has lower severity than B on average, since more matched backgrounds lead to greater leniency from jurors of similar backgrounds, compared to Group-B that counters the stereotype. Conversely, the latter shows Group-B is less severe, since jurors with opposite backgrounds may criticize Group-A more with higher mismatch scores.

Before vs. after statement. We further examine the signed gap of difference G. For most of the models, the signs of G for both the same-direction and opposite direction are the same, suggesting that each model has its own systematic tendency to enlarge or close the gap of difference between Group-A and Group-B, after hearing the statements. Across the models, the signs are roughly evenly divided between positive and negative without a clear inclination.

![](images/247cf9573b610d0b13ff5e28611dd900284817112140205479f38bea020cf074.jpg)  
Figure 4: Analysis for liberal/ conservative jury.

The magnitude |G| is often larger for oppositedirection cases. This indicates that oppositedirection cases experience stronger post-statement movement in the signed gap of difference, suggesting greater sensitivity to the statement intervention.

## 4.4 How does ideology affect jury’s decisions?

We further examine how a juror’s ideology affects their decision. We divide the analysis into 5 buckets for each LLM based on the ideology score: strong conservative (-5 to -3), weak conservative (-2 to -1), neutral (0), weak liberal (1 to 2), strong liberal (3 to 5). Severity scores before and after the statements are used as the metrics. Fig. 4 shows the results.

As shown by s<sup>no-statement</sup> and s<sup>statement</sup>, an obvious trend across LLMs shows that conservative jurors assign higher severity scores than neutral jurors, who are also harsher than liberal jurors. Overall, liberal jurors are clearly more lenient than conservative jurors. On average, stronger ideological leaning also indicates stronger effects- the strong conservatives always give the highest severity, and the strong liberals are usually the lowest.

From s<sup>statement</sup> − s<sup>no-statement</sup>, around 60% of LLMs show the same signs of severity change across all ideologies. This suggests that each model has its own tendency in how statements affect its severity judgments. Then, around 70% of LLMs show larger post-statement changes for conservative jurors than for the liberal, suggesting that conservative jurors are more sensitive to the statement.

Our studies echo prior legal-psychological research (Pyo, 2025; Sivasubramaniam et al., 2020; Anwar et al., 2019): conservative or “crime control” oriented jurors are more conviction-prone and tend to seek evidence to support a guilty verdict.

## 5 Conclusion

This work presents the first systematic study of LLM-jury under the common-law setting. We introduce JuryBench, a benchmark of controversial criminal cases, diverse juror profiles, controlled defendant backgrounds, and emotionally persuasive statements to enable fine-grained analysis of LLM-jury behavior. LLM-jury echoes several legal-psychology findings: sticky early impressions, double-edged defendant statements, and ideology- or affinity-driven decisions. JuryBench provides a controlled testbed for studying how LLMs may support mock-jury analysis and for lawyers to conduct mock trials with a focus group to revise the trial strategies. The work further supports future work on transparent and accountable legal simulations.

## Limitations

The work has several limitations:

• The work presents a simplified court simulation. In real trials, the case background and evidence introduction are presented during the opening statements and the prosecution’s case-in-chief, and we combine them into a single stage to provide a basic understanding. The work focuses on understanding the LLM-jury’s reaction to emotional persuasion; therefore, we simplified the testimony and cross-examination to speech from the defendant, which is also the core dynamics of modern trials for emotional persuasion. The full simulation of cross-examination, such as on the credibility of other evidence, may include complex arguments, which is orthogonal to emotional persuasion and not within the work’s scope. Last, the real verdict requires unanimous consensus from the jury in most U.S. states through group discussion. Since the focus of the work is to study how each LLM-simulated juror with different controlled parameters responds to emotional persuasion, the later group discussion to reach consensus is out of scope and we do not simulate the stage.

• The work focuses on U.S. common-law criminal trials. While our work examines the implicit factors shaping the decision-making process of LLM-jury, which is not limited to country or case type, the current scope primarily draws on U.S. criminal charges.

• The work only uses text to simulate the courtroom process following the previous work. However, the emotional contagion and remorse could be amplified by body language, facial expressions, and other factors that are only observable in real courtroom settings.

• With fast iteration and an increasing number of LLMs coming out, we only chose 20 frontier LLMs for study. The relative behaviors observed across models may change as model providers update training data, safety policies, and inference systems in the newer versions.

## Ethical Considerations

The work is designed as a research tool to understand the behaviors of LLM-jury, especially provided to legal professionals who intend to conduct mock trials to plan or refine their trial strategies using LLMs and simulate jurors with similar profiles. The intent of the work is not to automate real legal judgment or interfere with real cases. The cases, defendants, and jurors in our benchmark are synthetic and do not describe real individuals; nevertheless, the generated profiles involve sensitive social attributes and criminal allegations, so they should be handled with care. We will include clear documentation, intended-use restrictions, and warnings that the data may contain stereotypical associations introduced intentionally for bias analysis in the benchmark release.

## Acknowledgments

Per ACL policy, we disclose the generative AI usage in paper. We use generative AI to check contents correctness and polish the writing, while the content originality and research ideas are fundamentally developed by humans.

## References

Abdelrahman Abdallah, Bhawna Piryani, and Adam Jatowt. 2023. Exploring the state of the art in legal qa systems. Journal ofBig Data, 10(1):127.

Pawitsapak Akarajaradwong, Pirat Pothavorn, Chompakorn Chaksangchaichot, Panuthep Tasawong, Thitiwat Nopparatbundit, Keerakiat Pratai, and Sarana Nutanong. 2025. Nitibench: Benchmarking llm frameworks on thai legal question answering capabilities.

In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing.

Mohammad Almansoori, Komal Kumar, and Hisham Cholakkal. 2025. Self-evolving multi-agent simulations for realistic clinical interactions. arXiv preprint arXiv:2503.22678.

Shamena Anwar, Patrick Bayer, and Randi Hjalmarsson. 2019. Politics in the courtroom: Political ideology and jury decision making. Journal ofthe European Economic Association, 17(3):834–875.

Lisa P Argyle, Ethan C Busby, Nancy Fulda, Joshua R Gubler, Christopher Rytting, and David Wingate. 2023. Out of one, many: Using language models to simulate human samples. Political Analysis, 31(3):337–351.

Brian H Bornstein and Edie Greene. 2011. Jury decision making: Implications for and from psychology. Current directions in psychological science, 20(1):63– 67.

Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Katz, and Nikolaos Aletras. 2022. Lexglue: A benchmark dataset for legal language understanding in english. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4310–4330.

Guhong Chen, Liyang Fan, Zihan Gong, Nan Xie, Zixuan Li, Ziqiang Liu, Chengming Li, Qiang Qu, Hamid Alinejad-Rokny, Shiwen Ni, and Min Yang. 2025. Agentcourt: Simulating court with adversarial evolvable lawyer agents. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 5850–5865.

Odysseas S Chlapanis, Dimitrios Galanis, Nikolaos Aletras, and Ion Androutsopoulos. 2025. Greekbarbench: A challenging benchmark for free-text legal reasoning and citations. In EMNLP Findings.

Samuel Choi, Narina Nuñez, and Benjamin M Wilkowski. 2023. The influence of attorney anger on juror decision making. Psychiatry, Psychology and Law, 30(3):271–298.

John W Clark and Kenneth Wink. 2012. The relationship between political ideology and punishment: What do jury panel members say? Applied Psychology in Criminal Justice, 8(2).

Emily P Corwin, Robert J Cramer, Desiree A Griffin, and Stanley L Brodsky. 2012. Defendant remorse, need for affect, and juror sentencing decisions. Journal of the American Academy of Psychiatry and the Law Online, 40(1):41–49.

Robert J Cramer, Stanley L Brodsky, and Jamie De-Coster. 2009. Expert witness confidence and juror personality: Their impact on credibility and persuasion in the courtroom. Journal of the American Academy ofPsychiatry and the Law Online, 37(1):63– 74.

Yongfu Dai, Duanyu Feng, Jimin Huang, Haochen Jia, Qianqian Xie, Yifang Zhang, Weiguang Han, Wei Tian, and Hao Wang. 2025. Laiw: A chinese legal large language models benchmark. In Proceedings of the 31st International conference on computational linguistics, pages 10738–10766.

Zhuoyun Du, Lujie Zheng, Renjun Hu, Yuyang Xu, Xiawei Li, Ying Sun, Wei Chen, Jian Wu, Haolei Cai, and Haochao Ying. 2025. Llms can simulate standardized patients via agent coevolution. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 17278–17306.

Bonnie Erickson, E Allan Lind, Bruce C Johnson, and William M O’Barr. 1978. Speech style and impression formation in a court setting: The effects of “powerful” and “powerless” speech. Journal of experimental social psychology, 14(3):266–279.

Yu Fan, Jingwei Ni, Jakob Merane, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Etienne Salimbeni, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, and Joel Niklaus. 2026. Lexam: Benchmarking legal reasoning on 340 law exams. ICLR.

Zhihao Fan, Lai Wei, Jialong Tang, Wei Chen, Wang Siyuan, Zhongyu Wei, and Fei Huang. 2025. Ai hospital: Benchmarking large language models in a multi-agent medical interaction simulator. In Proceedings of the 31st International Conference on Computational Linguistics, pages 10183–10213.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Songyang Zhang, Kai Chen, Zongwen Shen, and Jidong Ge. 2023. Lawbench: Benchmarking legal knowledge of large language models. arXiv preprint arXiv:2309.16289.

Yi Feng, Chuanyi Li, and Vincent Ng. 2022. Legal judgment prediction via event extraction with constraints. In Proceedings of the 60th annual meeting of the associationfor computational linguistics (volume 1: long papers), pages 648–664.

Alessandra Foresta. 2025. Beyond a reasonable doubt: The impact of jurors’ political affiliations on trials: Evidence from north carolina. The Journal ofLaw and Economics, 68(2):361–386.

Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, K. Aditya, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason Hegland, and 21 others. 2023. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models. Advances in neural information processing systems, 36:44123–44279.

Sophia Simeng Han, Yoshiki Takashima, Shannon Zejiang Shen, Chen Liu, Yixin Liu, Roque K Thuo,

Sonia Knowlton, Ruzica Piskac, Scott J Shapiro, and Arman Cohan. 2025a. Courtreasoner: Can llm agents reason like judges? In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing.

Zhuo Han, Yi Yang, Yi Feng, Wanhong Huang, Ding Xuxing, Chuanyi Li, Jidong Ge, and Vincent Ng. 2025b. Lawshift: Benchmarking legal judgment prediction under statute shifts. Advances in Neural Information Processing Systems Datasets and Benchmarks Track.

Zhitao He, Pengfei Cao, Chenhao Wang, Zhuoran Jin, Yubo Chen, Jiexin Xu, Huaijun Li, Kang Liu, and Jun Zhao. 2024. Agentscourt: Building judicial decisionmaking agents with court debate simulation and legal knowledge augmentation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 9399–9416.

Tiancheng Hu and Nigel Collier. 2024. Quantifying the persona effect in llm simulations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10289–10307.

Yiran Hu, Zongyue Xue, Haitao Li, Siyuan Zheng, Qingjing Chen, Shaochun Wang, Xihan Zhang, Ning Zheng, Yun Liu, Qingyao Ai, Yiqun Liu, Charles L. A. Clarke, and Weixing Shen. 2026. Llms on trial: Evaluating judicial fairness for large language models. ICLR.

Wonseok Hwang, Dongjun Lee, Kyoungyeon Cho, Hanuhl Lee, and Minjoon Seo. 2022. A multi-task benchmark for korean legal language understanding and judgement prediction. Advances in Neural Information Processing Systems, 35:32537–32551.

Zainab Iftikhar, Amy Xiao, Sean Ransom, Jeff Huang, and Harini Suresh. 2025. How llm counselors violate ethical standards in mental health practice: A practitioner-informed framework. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, volume 8, pages 1311–1323.

Harry Kalven, Hans Zeisel, Thomas Callahan, and Philip Ennis. 1966. The american jury. Little, Brown Boston.

Norbert L Kerr, Robert W Hymes, Alonzo B Anderson, and James E Weathers. 1995. Defendant-juror similarity and mock joror judgments. Law and Human Behavior, 19(6):545–567.

Geoffrey P Kramer, Norbert L Kerr, and John S Carroll. 1990. Pretrial publicity, judicial remedies, and jury bias. Law and human behavior, 14(5):409–438.

Daeun Kyung, Hyunseung Chung, Seongsu Bae, Jiho Kim, Jae Ho Sohn, Taerim Kim, Soo Kyung Kim, and Edward Choi. 2025. Patientsim: A persona-driven simulator for realistic doctor-patient interactions. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Haitao Li, Qingyao Ai, Jia Chen, Qian Dong, Yueyue Wu, Yiqun Liu, Chong Chen, and Qi Tian. 2023. Sailer: structure-aware pre-trained language model for legal case retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1035–1044.

Yifei Liu, Yiquan Wu, Yating Zhang, Changlong Sun, Weiming Lu, Fei Wu, and Kun Kuang. 2023. Mlljp: Multi-law aware legal judgment prediction. In Proceedings of the 46th international ACM SIGIR conference on research and development in information retrieval, pages 1023–1034.

Antoine Louis, Gijs Van Dijck, and Gerasimos Spanakis. 2024. Interpretable long-form legal question answering with retrieval-augmented large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 22266–22275.

Amogh Mannekote, Adam Davies, Jina Kang, and Kristy Elizabeth Boyer. 2025. Can llms reliably simulate human learner actions? a simulation authoring framework for open-ended learning environments. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 29044–29052.

José M Marques, Vincent Y Yzerbyt, and Jacques-Philippe Leyens. 1988. The “black sheep effect”: Extremity of judgments towards ingroup members as a function of group identification. European journal ofsocial psychology, 18(1):1–16.

Mihai Masala, Radu Cristian Alexandru Iacob, Ana Sabina Uban, Marina Cidota, Horia Velicu, Traian Rebedea, and Marius Popescu. 2021. jurbert: A romanian bert model for legal judgement prediction. In Proceedings of the Natural Legal Language Processing Workshop 2021, pages 86–94.

Bo Ni, Yu Wang, Leyao Wang, Branislav Kveton, Franck Dernoncourt, Yu Xia, Hongjie Chen, Reuben Luera, Samyadeep Basu, Subhojyoti Mukherjee, Puneet Mathur, Nesreen K. Ahmed, Junda Wu, Li Li, Huixin Zhang, Ruiyi Zhang, Tong Yu, Sungchul Kim, Jiuxiang Gu, and 11 others. 2026. A survey on llmbased conversational user simulation. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4266–4301.

Joon Sung Park, Carolyn Q Zou, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Robb Willer, Percy Liang, and Michael S Bernstein. 2024. Generative agent simulations of 1,000 people. arXiv preprint arXiv:2411.10109.

Nick Polavin. 2022. Do jurors decide after opening statements? IMS Legal Strategies. Accessed: 2026- 05-25.

Michael Proeve. 2023. Addressing the challenges of remorse in the criminal justice system. Psychiatry, Psychology and Law, 30(1):68–82.

Jimin Pyo. 2025. Mock jurors’ conservative ideology and punitiveness: the role of criminal justice orientations. Journal ofCrime and Justice, 48(4):512–532.

Jesse Rhodes, Tatishe Nteta, and Douglas Rice. 2025. Partisan bias in juror decision-making. Journal of Law & Empirical Analysis, 2(2):308–323.

Cheol Ryu, Seolhwa Lee, Subeen Pang, Chanyeol Choi, Hojun Choi, Myeonggee Min, and Jy-Yong Sohn. 2023. Retrieval-based evaluation for llms: A case study in korean legal qa. In Proceedings ofthe Natural Legal Language Processing Workshop 2023, pages 132–137.

Randall T Salekin, James RP Ogloff, Cathy McFarland, and Richard Rogers. 1995. Influencing jurors’ perceptions of guilt: Expression of emotionality during testimony. Behavioral Sciences & the Law, 13(2):293–305.

Debdeep Sanyal, Agniva Maiti, Umakanta Maharana, Dhruv Kumar, Ankur Mali, C Lee Giles, and Murari Mandal. 2025. Investigating pedagogical teacher and student llm agents: Genetic adaptation meets retrieval-augmented generation across learning styles. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing.

Kimberly Schweitzer and Narina Nuñez. 2021. The effect of evidence order on jurors’ verdicts: Primacy and recency effects with strongly and weakly probative evidence. Applied Cognitive Psychology, 35(6):1510–1522.

Gil Semo, Dor Bernsohn, Ben Hagag, Gila Hayat, and Joel Niklaus. 2022. Classactionprediction: A challenging benchmark for legal judgment prediction of class action cases in the us. In Proceedings of the Natural Legal Language Processing Workshop 2022, pages 31–46.

Magnus Sesodia, Alina Petrova, John Armour, Thomas Lukasiewicz, Oana-Maria Camburu, Puneet K Dokania, Philip Torr, and Christian Schroeder de Witt. 2025. Annocaselaw: a richly-annotated dataset for benchmarking explainable legal judgment prediction. arXiv preprint arXiv:2503.00128.

Diane Sivasubramaniam, Mallory McGuinness, Darcy Coulter, Bianca Klettke, Mark Nolan, and Regina Schuller. 2020. Jury decision-making: The impact of engagement and perceived threat on verdict decisions. Psychiatry, Psychology and Law, 27(3):346–365.

Benjamin Strickson and Beatriz De La Iglesia. 2020. Legal judgement prediction for uk courts. In Proceedings of the 3rd International Conference on Information Science and Systems, pages 204–209.

Dietrich Trautmann, Alina Petrova, and Frank Schilder. 2022. Legal prompt engineering for multilingual legal judgement prediction. arXiv preprint arXiv:2212.02199.

Janne van Doorn and Maarten Kunst. 2025. The ‘emotional defendant effect’: a systematic review of experimental studies. Psychology, Crime & Law, pages 1–32.

Xi Wang, Anxo Perez, Javier Parapar, and Fabio Crestani. 2025a. Talkdep: clinically grounded llm personas for conversation-centric depression screening. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management, pages 6554–6558.

Xintao Wang, Heng Wang, Yifei Zhang, Xinfeng Yuan, Rui Xu, Jen-tse Huang, Siyu Yuan, Haoran Guo, Jiangjie Chen, Shuchang Zhou, Wei Wang, and Yanghua Xiao. 2025b. Coser: Coordinating llmbased persona simulation of established roles. In Forty-second International Conference on Machine Learning.

Zixiao Wang, Duzhen Zhang, Ishita Agarwal, Shen Gao, Le Song, and Xiuying Chen. 2025c. Beyond profile: From surface-level facts to deep persona simulation in llms. In Findings of the Association for Computational Linguistics: ACL 2025, pages 21239–21257.

Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao Tu, Zhiyuan Liu, Maosong Sun, Yansong Feng, Xianpei Han, Zhen Hu, Heng Wang, and Jianfeng Xu. 2018. Cail2018: A large-scale legal dataset for judgment prediction. arXiv preprint arXiv:1807.02478.

Shengbin Yue, Ting Huang, Zheng Jia, Siyuan Wang, Shujun Liu, Yun Song, Xuan-Jing Huang, and Zhongyu Wei. 2025. Multi-agent simulator drives language models for legal intensive interaction. In Findings of the Association for Computational Linguistics: NAACL 2025.

Zheyuan Zhang, Daniel Zhang-Li, Jifan Yu, Linlu Gong, Jinchang Zhou, Zhanxin Hao, Jianxiao Jiang, Jie Cao, Huiqin Liu, Zhiyuan Liu, Lei Hou, and Juanzi Li. 2025. Simulating classroom education with llmempowered agents. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 10364–10379.

## A Human Evaluation

We also conduct a human evaluation on the generated cases. 25 cases of more common and comprehensible scenarios are sampled. The base case scenarios are presented first, and we ask 12 U.S. lay participants without legal training to select a charge from the potential charges as the verdict. We give brief explanations in plain English for each potential charge, simulating jury instructions from a judge. Then, we present the defendant’s statement and ask the subjects to make decisions again. Specifically, we ask the subjects not to revise the previous response retroactively from the scenario without the statement. If the post-statement decision differs from the pre-statement decision, we ask the subject to briefly explain the reason. To compute results, we average across 12 people and 25 cases, and present the results alongside LLMs for the 25 cases in Tab 2.

From the table, one can see that the human evaluation also leans toward increasing severity after the statement with a negative SE value. The most common reasons for increasing the severity are also finding remorse as evidence of guilt, and another minor reason is disbelief about the statements as they sound performative sometimes. Both of the reasons are covered in the main paper Sec. 4.2 for the LLM’s reasoning.

On the other hand, for the reasons of decreasing the severity, the top reason is that subjects recognize the grounds for justification that lead to acquittal, or subjects recognize the grounds while empathizing with situations of the defendants, which is facilitated by emotional contagion. There are also a few cases where the statements include genuine remorse and valid grounds for justification. Both of the reasons are also described in the main paper Sec. 4.2. This subset and human evaluation is intended as a proof on our claim of similarity between LLM and human reasoning, rather than a large and representative human-jury benchmark.

<table><tr><td></td><td>SE</td><td>SR</td><td>HR</td><td>NR</td></tr><tr><td>Human Eval</td><td>-0.2920</td><td>0.1328</td><td>0.1652</td><td>0.7022</td></tr><tr><td>Gemini 3 Flash</td><td>0.5033</td><td>0.1617</td><td>0.0550</td><td>0.7833</td></tr><tr><td>GPT-5 Mini</td><td>0.5017</td><td>0.1733</td><td>0.0867</td><td>0.7400</td></tr><tr><td>GPT-5.4 Mini</td><td>0.1200</td><td>0.1483</td><td>0.1250</td><td>0.7267</td></tr><tr><td>Llama 4</td><td>0.0383</td><td>0.1350</td><td>0.1350</td><td>0.7300</td></tr><tr><td>GPT-5.4</td><td>-0.1067</td><td>0.1033</td><td>0.1133</td><td>0.7833</td></tr><tr><td>GPT-4o Mini</td><td>-0.1717</td><td>0.0400</td><td>0.0667</td><td>0.8933</td></tr><tr><td>Kimi K2.5</td><td>-0.2783</td><td>0.1400</td><td>0.1683</td><td>0.6917</td></tr><tr><td>DeepSeek V4 Pro</td><td>-0.2867</td><td>0.1417</td><td>0.1550</td><td>0.7033</td></tr><tr><td>Grok-4.3</td><td>-0.3133</td><td>0.1817</td><td>0.2000</td><td>0.6183</td></tr><tr><td>Kimi K2 Instruct</td><td>-0.3400</td><td>0.1183</td><td>0.1350</td><td>0.7467</td></tr><tr><td>GPT-5.4 Nano</td><td>-0.3883</td><td>0.1167</td><td>0.1617</td><td>0.7217</td></tr><tr><td>DeepSeek V4 Flash</td><td>-0.5100</td><td>0.0967</td><td>0.1517</td><td>0.7517</td></tr><tr><td>Kimi K2.6</td><td>-0.5317</td><td>0.1217</td><td>0.1767</td><td>0.7017</td></tr><tr><td>Claude Sonnet 4.6</td><td>-0.5600</td><td>0.0667</td><td>0.1383</td><td>0.7950</td></tr><tr><td>Gemini 3.1 Pro</td><td>-0.5750</td><td>0.0567</td><td>0.1300</td><td>0.8133</td></tr><tr><td>GPT-5.5</td><td>-0.6467</td><td>0.1033</td><td>0.1517</td><td>0.7450</td></tr><tr><td>GLM 5</td><td>-0.6883</td><td>0.1333</td><td>0.1950</td><td>0.6717</td></tr><tr><td>Claude Haiku 4.5</td><td>-0.7650</td><td>0.0717</td><td>0.1767</td><td>0.7517</td></tr><tr><td>Claude Opus 4.6</td><td>-0.8667</td><td>0.0667</td><td>0.1717</td><td>0.7617</td></tr><tr><td>Qwen3</td><td>-0.9233</td><td>0.1100</td><td>0.2167</td><td>0.6733</td></tr></table>

Table 2: Effect of defendant statements on the 25 selected cases along with human evaluation.

We also run the regression model in Eq. 5 for the human evaluation. Note that although we ask the subjects to self-describe their ideology, nearly all are centered on the neutral within the range [- 2, 2]. There is no strong ideology for analyzing human bias. Another reason is that subjects easily omit the defendant’s background in a text-based survey and usually cannot perceive the background difference without a real person, actions, body language, and expressions, whereas LLMs implicitly process the information as if it were a real person by the ability of persona simulation. We remove the background fit term M from the analysis. The coefficients for $p$ and $q$ are 0.173 and -0.168- both p-values are smaller than 0.05 but larger than 0.01 with significance.

## B Joint-Effect Regression

In Eq. 5, we studied a model composed of three isolated terms for analysis. The form considers each factor as an independent additive predictor and asks whether it is associated with severity reduction, which is straightforward to explain each factor’s effect.

We further build a full interaction model that considers the joint effect of each term

$$
\begin{array} { r } { \mathrm { S E } = \alpha + \beta _ { 1 } z ( p ) + \beta _ { 2 } z ( q ) + \beta _ { 3 } z ( M ) \qquad } \\ { + \beta _ { 4 } z ( p q ) + \beta _ { 5 } z ( p M ) + \beta _ { 6 } z ( q M ) + \beta _ { 7 } z ( p q M ) , \qquad } \end{array}\tag{9}
$$

which becomes conditional but not marginal now for $p , q , M$ . The explanation is also more complex and needs cares. The coefficients cannot be interpreted as independent causal effects. For example, $p \mathbf { \hat { s } }$ marginal effect $\partial \mathrm { S E } / \partial p$ is not just $\beta _ { 1 }$ , but also compound with other terms, and the same for other terms.

Explanation about Joint Effects The results are shown in the Fig. 5. First M, $p q ,$ and pqM are nearly all positive. Since positive coefficients correspond to reduced post-statement severity, this suggests that both defendant-juror affinity M and coherent emotional-remorseful statements pq are associated with more lenient outcomes. The positive $p q M$ term further indicates that these effects reinforce each other: when a defendant’s statement combines emotional appeal with remorse, and the defendant is also well matched with the juror’s affinity profile, LLM jurors are more likely to reduce the severity. In contrast, the isolated terms p and q alone are mostly negative, as the positive effect have been captured by joint effects pqM or pq, and each isolation term can backfire, potentially being perceived as performative, insufficient, or even as implicit evidence of responsibility.

Next, The negative pM and qM coefficients suggest that defendant-juror affinity does not monotonically amplify emotional or remorseful appeals. Although the main effect of M is positive, indicating that matched backgrounds are generally associated with lower post-statement severity, the negative two-way interactions show that isolated emotionality or isolated remorse becomes less mitigating under stronger affinity. This pattern suggests a saturation or credibility effect: once affinity already favors the defendant, additional one-dimensional emotional or remorseful cues may be discounted as strategic or excessive. However, the positive pqM term indicates that when emotionality and remorse are jointly present, affinity again strengthens the mitigating effect, consistent with LLM jurors rewarding coherent rather than isolated persuasive appeals.

Comparison to unary model. Compared with the unary model in Eq. 5, which treats each factor as an independent additive predictor, the full interaction model asks how does the effect of one factor depend on the presence of the others? The two models therefore serve complementary purposes. The unary model identifies each LLM’s dominant marginal sensitivity, while the interaction model explains how these sensitivities combine.

The additive model remains useful because it provides an interpretable behavioral profile for each LLM. By isolating the marginal association of emotional contagion, remorse, and background fit, it shows which cue each model is most sensitive to when the decision process is summarized in firstorder terms. These model-level differences are important for practical use: a lawyer or researcher choosing an LLM-jury simulator may care whether a model is especially responsive to emotional appeals, remorse, or defendant-juror affinity.

The full interaction aim to further complement the additive findings, where the term-shared structures become more visible. The additive model shows that M is the most robust standalone predictor, whereas p and q are weaker and modeldependent. The interaction model explains this instability: isolated emotionality or remorse can backfire, but their coherent combination pq, especially when aligned with defendant-juror affinity through pqM, is consistently associated with reduced severity. Thus, LLM jurors appear to reward coherent persuasive configurations rather than iso-

lated rhetorical signals.

## C Prompts in Use

We document the prompts for generating case scenarios, juror profiles, and verdict from each juror in Fig. 6, Fig. 7, and Fig. 8.

## D Instructions for Legal Expert and Manual Check

In Fig. 9, we provide the detailed instructions to the legal expert, who helped gate the generation quality of the case scenarios and statement. The case generation inspection took two weeks to complete, where in total 67 cases were flagged throughout the whole process for regeneration and gating again.

In Fig. 10, we also give an instruction about gating the generated affinity scores to an expert trained in sociology. In total, 51 defendant-background affinity scores were corrected.

## E Common-Law and Civil-Law Settings

Common-law and civil-law systems differ in their legal traditions, sources of authority, and courtroom decision-making structures. Civil-law systems are primarily statute-centered: legal rules are codified in written statutes and legal codes, and judges are expected to apply these provisions to individual cases. In many civil-law settings, legal professionals therefore play the central role in interpreting statutes, examining case materials, and producing judgments. Courtroom reasoning often emphasizes the applicability of legal provisions, the interpretation of written legal materials, and the consistency of the judgment with codified rules. Representative civil-law jurisdictions include France, Germany, Italy, Spain, Japan, South Korea, China, and many countries in continental Europe and Latin America.

By contrast, common-law systems place greater emphasis on adversarial argumentation, judicial precedent, and fact-finding through trial procedures. In common-law criminal trials, especially in the U.S. setting studied in this work, the jury often consists of laypeople without formal juristic training. Rather than interpreting statutes in a professional legal capacity, jurors primarily evaluate facts, evidence, witness credibility, defendant statements, and whether the prosecution has proven guilt beyond a reasonable doubt. As a result, courtroom arguments are not only legal but also persuasive: attorneys must translate legal claims into narratives that are understandable and convincing to ordinary citizens. These arguments often appeal to commonsense, life experience, moral judgment, credibility, and emotion. Representative common-law jurisdictions include the United States, the United Kingdom, Canada, Australia, New Zealand, India, and Singapore.

![](images/9ea3eabee87b28700e4f7ad76173112d76b6170d34180e808da4c070bdbe5fd3.jpg)  
Figure 5: Analysis of full interaction model. \* indicates significance based on statistical p-value < 0.05; \*\* p < 0.01; \*\*\* p < 0.001.

Generate \*\*highly controversial\*\* jury-deliberation scenarios involving doctrines such as mens rea, self-defense, necessity,   
duress, causality, mistake/ignorance, or alibi, or any other reasons that the defendant may claim to being discharged.   
However, from the scenarios the defendants still have slightly more than a half chance to get convicted. The charges   
associated with the case must have high diversity from the US Criminal Code.   
Create two matched groups of cases:   
For ‘groupA‘, generate defendant backgrounds that align with common prejudicial stereotypes, such as:   
- 37-year-old Black woman, poor   
49-year-old white man, successful, grumpy, dismissive of feminism   
- 21-year-old Asian immigrant, limited English, guileless   
For ‘groupB‘, generate the same ‘case\_background‘ and ‘evidence‘ for each matching case, but change only the ‘   
defendant\_background‘ and ‘speech‘ so they counter the prejudice, such as:   
37-year-old Black woman, rich, educated   
49-year-old white man, poor, empathetic   
21-year-old Asian immigrant, fluent in English, mischievous   
Note that the above are examples. Follow the rule and generate diverse backgrounds.   
Each case must contain exactly these nine fields:   
‘defendant\_background‘   
‘case\_background‘   
‘evidence‘   
‘speech‘   
‘affinity‘   
‘potential\_charge‘   
The ‘speech‘ should be a statement by the defendant to attempt to get an acquittal. The speech can stress on aspects of grounds   
for justifications, emotional persuaion to the jury, validity of the behavior, expressing remorse, or mix of them in a   
coherent style. The length should contain about 15-20 sentences.   
‘affinity‘ must be an integer from -5 to 5 that predicts how a conservative/liberal juror may feel closer to the defendant’s   
background and therefore may feel more empathy toward the defendant:   
-5 means the most conservative   
5 means the most liberal   
‘potential\_charge‘ must list all potential charges for which the defendant could be convicted based on the case background and   
evidence, including "not-guilty" as an option. Separate each charge with a comma. For example, in an assault-related case,   
the possible charges could be: simple assault, aggravated assault, assault with a deadly weapon, battery, aggravated   
battery, reckless endangerment, attempted assault, menacing, harassment, not-guilty.   
Return only valid JSON in this exact shape:   
{   
"groupA": [   
{   
"defendant\_background": "...",   
"case\_background": "...",   
"evidence": "...",   
"speech": "...",   
"affinity": 3,   
"potential\_charge": "..."   
}   
],   
"groupB": [   
{   
"defendant\_background": "...",   
"case\_background": "...",   
"evidence": "...",   
"speech": "...",   
"affinity": -2,   
"potential\_charge": "..."   
}   
7   
Rules:   
‘groupA‘ and ‘groupB‘ must each contain exactly COUNT cases.   
For each index, ‘groupA[i].case\_background‘ must equal ‘groupB[i].case\_background‘.   
For each index, ‘groupA[i].evidence‘ must equal ‘groupB[i].evidence‘.   
Do not include markdown, explanations, or any text outside the JSON object.  
Figure 6: Prompt for generating cases.

![](images/641ed046a83542c2879d7a26b97f0fe44426b4a0f01ef1ecd7b1bfaa12b1660a.jpg)  
Figure 7: Prompt for generating juror persona.

![](images/bb9939cab3c3a60d305cab634802d1954678562f704282e3dee8a2b96bbdf0ae.jpg)  
Figure 8: Prompt to generate prediction with a statement presented. For decisions based on no statements, we remove any instructions about statements here.

![](images/f852aff4809acd7c7d48d7b2e3e042a79aefb4f7b712744e776bfba269b87291.jpg)  
Figure 9: Instruction for the legal expert.

![](images/300eeb711295d09cb10607ee96bf4c9fb9b90fbe94b06f9d92258a3d299fcf64.jpg)  
Figure 10: Instruction for affinity score check.