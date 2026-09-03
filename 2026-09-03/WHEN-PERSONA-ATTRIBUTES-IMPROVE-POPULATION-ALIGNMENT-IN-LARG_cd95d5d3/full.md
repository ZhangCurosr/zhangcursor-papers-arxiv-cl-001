# WHEN PERSONA ATTRIBUTES IMPROVE POPULATION ALIGNMENT IN LARGE LANGUAGE MODELS

Leon Fröhling <sup>1,\*</sup>, Jens Rupprecht <sup>2,\*</sup>, Markus Strohmaier <sup>1,2,3</sup>, and Claudia Wagner <sup>1,3,4</sup>

<sup>1</sup>GESIS – Leibniz Institute for the Social Sciences

<sup>2</sup>University of Mannheim

<sup>3</sup>Complexity Science Hub

<sup>4</sup>RWTH Aachen University

## ABSTRACT

Large Language Models (LLMs) are increasingly used to predict the responses of human participants in survey panels. Towards that goal, persona prompting has recently emerged as a technique to inform and align large pretrained language models. Persona prompting refers to the practice of using short textual descriptions of ’personas’ in prompts to steer the LLM’s generations. Personas describe individuals through different attributes such as their socio-demographics, attitudes, or behaviors, with the aim of aligning LLMs to produce responses that correlate with the corresponding human responses. Yet, recent work has produced mixed and partly conflicting results of persona prompting without clear patterns of success and failure. Among the few consistent findings is that the selection of persona attributes matters, and that using more attributes does not necessarily lead to better performance. It remains unclear how different attribute selection methods perform and how to choose among them. In this paper, we propose that observed human response variation of a survey question is a potential explanation for the mixed performance observed so far. In addition, we compare the performance of persona prompting associated with different methods for selecting persona attributes. We evaluate these methods on four different (general) social surveys across two countries, six LLMs, and twenty prediction tasks per survey. Our work helps to identify when persona prompting can be expected to be useful in survey prediction tasks, and provides new insights on the effectiveness of different attribute selection methods for LLM-based survey prediction using persona prompting.

![](images/05fcdc0f7c2b045dccc5906467b7e169ae80e9c2f70cf7c5f9f61101e05bd68e.jpg)  
Figure 1: Overview and Experimental Setup. Using survey data, we evaluate how well different persona attribute selection strategies help to predict responses of survey participants to questions with differing levels of human response variation. We hypothesize that persona-prompted LLMs are better at predicting responses to survey question with higher human response variation, and that persona attribute selection approaches optimized for persona prompting outperform purely data-driven alternatives.

## 1 Introduction

The use of personas to introduce diversity and simulate different human perspectives is rapidly gaining traction in Natural Language Processing (NLP) and Computational Social Science (CSS). By conditioning large language models (LLMs) on demographic, attitudinal, or behavioral characteristics, persona prompting promises a scalable approach to modeling heterogeneous viewpoints and social behavior, often combining empirical observations with synthetic data generation.

Despite this promise, empirical findings remain mixed. While some studies report substantial improvements in the representation of diverse perspectives, others find limited or inconsistent gains, raising fundamental questions about when and why persona prompting succeeds. In particular, the conditions under which personas enable LLMs to faithfully reproduce human judgments and behaviors remain poorly understood.

The limited empirical evidence available on the sources of these mixed results comes primarily from subjective annotation tasks. In this context, Brown et al. (2025) and Sarumi et al. (2024) show that the degree of agreement among human annotators predicts an LLM’s ability to reproduce human annotations. Their findings suggest that persona prompting is more effective when human judgments are relatively consistent, but less successful when opinions diverge substantially. While these studies provide important initial insights, it remains unclear whether this relationship generalizes beyond the specific annotation tasks that they analyzed.

To study this question in the context of surveys, we introduce the concept of human response variation, which we define as the variation observed in participants’ responses to a given survey question. This conceptualization is closely related to the notion of human label variation in NLP (Plank, 2022), which recognizes that variation in annotations may arise not only from task ambiguity but also from legitimate differences in interpretation and multiple plausible responses. Survey responses, however, differ in an important respect: variation is typically not a measurement artifact but the substantive phenomenon of interest. Respondents differ in their socio-demographic characteristics, opinions, attitudes, experiences, and behaviors, all of which can give rise to systematically different yet equally valid responses to the same question. Consequently, our concept of human response variation captures meaningful heterogeneity in the population. We propose differences in human response variation as a potential explanation for performance differences of the persona prompting approach in reproducing human response distributions across different survey response prediction tasks.

In addition to being a promising application, surveys provide the ideal evaluation environment for methodological advances: They provide detailed individual-level information ranging from basic socio-demographic characteristics to more complex behaviors and beliefs from which relevant attributes for the construction of persona descriptions can be selected, and naturally feature a range of diverse questions with varying levels of human response variation that can be used for evaluation. We operationalize human response variation through two different measures of dispersion, selected based on the type and scale of the response variable (Wilcox, 2008): normalized entropy for nominal and dissention for ordinal response variables.

We build upon one of the few consistent findings in the literature on persona prompting to further understand the factor influencing the performance of the approach: Selecting the right set of attributes to construct persona descriptions matters. While Luz de Araujo et al. (2025) and Rupprecht et al. (2026) observe that LLMs are sensitive to irrelevant persona details, with performance dropping sharply if they are included, Hwang et al. (2023) find that augmenting a set of demographic attributes and ideology scores with past user opinions leads to consistent accuracy gains across a range of tasks. Notably, previous work either added or removed whole blocks of attributes irrespective of the predictive task at hand. Instead, we benchmark different approaches for selecting persona attributes given a target survey question to predict responses for. Throughout this work, we refer to the prediction of responses for specific survey questions as survey response prediction tasks, given that both the prediction target and the inputs vary.

In this work, we address the following two research questions (RQs):

• RQ1: To what extent can human response variation explain the differences in persona prompting performance across different survey response prediction tasks?

• RQ2: To what extent can different persona attribute selection methods improve the performance of persona prompting on survey response prediction tasks?

To address these research questions, we systematically investigate the relationship between human response variation, persona variable selection methods, and persona prompting performance, using the methodology outlined in Figure 1. We use national-level data for the US and Germany from an international survey program run in parallel in 70 countries across the world, the World Value Survey (WVS) (EVS/WVS, 2024), as well as the general social surveys conducted in these two countries —the US General Social Survey (GSS) (Davern et al., 2025), and the German General Social Survey (GGSS) (Ackermann et al., 2025).

## 2 Background

## 2.1 Persona Prompting Studies Report Mixed Results

Persona prompting has widely been used to simulate human responses to survey questions. However, the findings remained mixed: for example, some work found that they were not able to recover known regression coefficients by conditioning LLMs with persona prompting (Bisbee et al., 2024), while others state the ambiguity of results (Lee et al., 2024; Qu & Wang, 2024; Sanders et al., 2023). At the same time, there is a large amount of work achieving promising results with persona-conditioned LLMs on various tasks (Aher et al., 2023; Argyle et al., 2023; Horton et al., 2023; Li & Conrad, 2026; Park et al., 2026).

Even though there already exists a wide range of applications for personas and methods for constructing them, only few previous works have set out to understand the reasons for why or why not the approach works in a given context. In the context of NLP, some authors have started to explore the connection between the selection of persona variables, the variation in human annotations, and the performance of persona prompting. Hu and Collier (2024) fit linear regressions to predict annotations using persona variables, finding that those variables usually explain less than 10% of variance across NLP tasks. When using these persona variables for persona prompting to replicate the tasks annotations, they find that variables that are most strongly correlated with the outcome measure lead to the best predictive performance, and that most gains are to be expected for tasks with high entropy but low standard deviation in human annotations. Furthermore, Hu et al. (2026) establish a link between the stage of model development and the models’ ability to represent differing levels of variation human behavior. They distinguish between base models, optimized for "[representing] the full, multi-modal diversity of human language and opinion during pre-training", and instruction-tuned models, optimized for "[concentrating] probability mass on a single, high-reward mode of the target preference distribution". They find that instruction-tuned LLMs working best for tasks with low-entropy outcomes and base models for tasks with high-entropy outcomes, suggesting that instruction-tuned models are better at accurately representing majority outcomes while base models are better able to replicate the full distribution of human behavior.

Taking a different angle, Hwang et al. (2023) use the response variation of human participants with the same basic demographic attributes to determine the degree to which additional attributes are needed to explain the observed response variation. They show that the agreement scores (measured via Cohen’s κ) between individuals with the same demographics are gathered around 0.5, with perfect agreement being equal to scores of 1. This finding indicates that using only this set of demographics is insufficient for modeling the disagreement, and that therefore additional attributes are needed for meaningfully explaining the differences in outcomes.

In addition to this empirical evidence from NLP research, suggesting that human response variation is a plausible explanation for the mixed performance of persona prompted LLMs in the survey response prediction task, we derive two complementary explanations from general observations on the development and use of LLMs.

Similar to the observations of Hwang et al. (2023), we hypothesize that persona prompting works best for predicting survey questions with high levels of human response variation, because the LLM requires the additional attributes available through the persona description to model deviations from its default response, i.e., the response it would generate without being conditioned on a persona. Intuitively, if two individuals with different attributes respond differently to a survey question, an LLM used for predicting their individual survey responses would need access to these differing attributes to be able to meaningfully model the differences in their responses.

Similar to the observations of Hu et al. (2026), we hypothesize that persona prompting works worst for predicting survey questions with low levels of human response variation, because the optimal response generation strategy for the LLM would be to simply generate its default response. As has been shown empirically and derived theoretically, the default response of LLMs tends towards a representation of the majority view (Chakraborty et al., 2024; Santurkar et al., 2023; Sorensen et al., 2024). In this view, any additional persona attribute introduces noise and has the potential to steer the LLM away from its majority-aligned default response, thus decreasing performance in cases in which the response distribution is highly concentrated around a single majority response. Hu et al. (2026) observe that this effect is particularly strong for instruction-tuned models, which they describe as optimized for producing this type of single best response instead of a full distribution of (equally) valid responses.

With respect to our first research question, we thus hypothesize that the human response variation of the survey question to be predicted has an impact on the performance of the persona prompting approach. We expect the approach to work better for questions with high human response variation versus for questions with low human response variation, and that this effect is reversed when not using personas.

## 2.2 Persona Attribute Selection Matters

In one of the few methodological contributions on the mechanism of persona prompting, moving beyond external factors potentially explaining performance differences, Luz de Araujo et al. (2025) point to the gap in the literature that we are aiming to address: "Despite a diversity of personas and tasks, most prior work does not systematically differentiate between relevant and irrelevant persona attributes or measure their specific influence on model behavior". In their own work, Luz de Araujo et al. (2025) formulate four desiderata of persona prompting, of which task-irrelevant attributes should not affect model performance and relevant attributes [..] should shape model performance in ways consistent with those attributes stand out as particularly important for our context. For a number of objective annotation tasks, they find that personas differentiated by relevant expertise and education mostly impact task performance in expected ways, and that the inclusion of irrelevant attributes such as color preferences and names leads to drops in performance of up to 30 percentage points.

Giorgi et al. (2024) study how explicit (defined via their age, gender, political ideology, race, and substance use) and implicit (defined via names that are highly frequent for specific demographics) personas in LLM prompts replicate patterns of human subjectivity in annotation and generation tasks. They find that explicitly providing attributes rather than having the LLM infer them from names works better, and that political ideology seems to be the driving factor behind the persona-conditioned generations, with gender being the least important one. Similar to the distinction between explicit and implicit personas by Giorgi et al. (2024), there are a number of works studying the impact of different formats or locations for presenting the persona information in the prompt, with few consistent findings across these different studies (Lutz et al., 2025; Neumann et al., 2025; Weeber et al., 2026).

In the survey context, Xie et al. (2026) generate responses to held-out survey questions in order to evaluate whether LLMs conditioned on demographic variables are successful in generating digital twins of real populations. They observe that including more attributes in the conditioning prompts increases statistic realism, i.e., improves the ability of persona-prompted LLMs to reproduce the actual human responses. Similarly, Park et al. (2026) have shown that conditioning LLM-generations on rich qualitative data such as semi-structured interviews leads to the best performance in predicting participants’ survey responses. However, they also find that using structured survey data for persona prompting works similarly well. In contrast, the performance drops significantly when using only the set of four commonly-used socio-demographic attributes age, gender, race and political ideology for persona construction.

Hwang et al. (2023) use embedding similarity between input attributes (in their case individuals’ past opinions) and the target question to construct personas from sets of attributes with differing importance, assuming that higher similarity between attribute and target question implies higher attribute importance. They find that using the top-3 attributes performs on par with using the combination of socio-demographics, ideology, and 16 randomly selected attributes, suggesting that selecting the most important attributes for persona prompting matters.

In contrast to previous work on persona prompting that either relies on a small set of socio-demographic attributes, augmented with manually selected, task-dependent additions such as political ideology or past opinions (Giorgi et al., 2024; Hwang et al., 2023), or that selects the same set of attributes across different tasks (Rupprecht et al., 2026), we benchmark different approaches for selecting an optimal set of attributes for each survey response prediction task.

We are particularly interested in comparing the performance of LLM-based approaches with that of statistical baseline approaches. We design the statistical baseline approaches to select persona attributes exclusively based on correlations and associations directly available from the full survey data. In contrast, for the LLM-based approaches, we hope to leverage the broad world knowledge that these models develop during the different stages of pre-training and alignment and that enables them to generalize to novel contexts and tasks, including survey response prediction (Chen et al., 2024; Wang, Antoniades, et al., 2025). However, because LLMs’ internal knowledge representations are different from those of humans, the relationships and correlations leveraged by LLMs during response prediction are not necessarily evident from an isolated survey and cannot fully be captured by traditional statistical techniques (Lappin, 2024; Suresh et al., 2023).

By explicitly prompting for a selection of optimal sets of attributes for use in a persona prompting application, we hope that the LLM-based attribute selection approaches leverage their internal representations of knowledge to select LLM-optimized attributes. With respect to our second research question, we assume that these LLM-optimized sets of selected attributes should lead to better performance of the persona prompting approach than using attributes selected by statistical baselines that rely on survey data alone.

## 3 Methodology

In this section, we present the different parts of our methodological setup, represented by the panels in Figure 1. We start by introducing our measures of human response variation as well as the persona attribute selection approaches we are testing. We describe how personas are constructed and used to predict human survey responses and conclude with our measures to evaluate and compare the performance of the different approaches.

## 3.1 Measuring Human Response Variation

As discussed in the introduction, we propose to use the human response variation observed in the human responses to a survey question as a measure for the difficulty that a persona-prompted LLM is facing in correctly predicting the human responses to the given question. To be able to compare the human response variation across different survey variables <sup>1</sup>, we require a measure that summarizes the spread of human responses across varying number of response options.

For survey variables with nominal responses, such as a binary yes-or-no response or a preference between different political parties, we can simply use the normalized Shannon entropy (Shannon, 1948) as a measure for the average uncertainty of the variable’s potential outcomes.

The entropy of a (survey response) distribution is given as

$$
H ( X ) = - \sum _ { i = 1 } ^ { n } p ( X _ { i } ) \log _ { 2 } p ( X _ { i } ) ,
$$

with $X _ { i } \in [ X _ { 1 } , . . . , X _ { n } ]$ as the set of all possible response options and $p ( X _ { i } )$ as the relative frequency of each response. The normalized entropy is then calculated by dividing the entropy by the maximal possible entropy associated with the response variable, which is given as $H _ { m a x } = \log _ { 2 } n$

However, survey variables with ordinal responses, such as Likert scales with responses ranging from, e.g., strongly disagree to strongly agree, cannot be analyzed by the Shannon entropy as it is invariant to the location of responses on the scale. Thus, Tastle and Wierman (2006) develop their consensus measure, which, if inverted, becomes the dissention measure, measuring the human response variation in ordinal variables.

The only requirement of this measure is the ability to identify the extreme values associated with the endpoints of the range of possible responses. Once these are defined, a numerical scale is applied to the response options, assigning a value of 1 to one endpoint and a value equivalent to the number of response options to the other endpoint. All response options in between are assigned integer values increasing in unit steps. In the case of a five-point Likert scale ranging from strongly disagree to strongly agree, strongly disagree would be equal to 1, disagree to 2, neutral to 3, agree to 4, and strongly agree to 5.

This first transformation results in $X = \{ 1 , 2 , 3 , 4 , 5 \}$ , with $X _ { 1 } = 1 , X _ { 2 } = 2$ , and so on. The width of X is given as $d _ { X } = X _ { m a x } - X _ { m i n }$ , and the expected value as

$$
E _ { X } = \sum _ { i = 1 } ^ { n } p ( X _ { i } ) X _ { i } = \mu _ { X } ,
$$

with n as the number of response options $( \mathrm { i . e . }$ , the length of X) and $p ( X _ { i } )$ the relative frequency associated with each response option $X _ { i }$

Finally, the consensus measure is calculated as

$$
\operatorname { C n s } ( X ) = 1 + \sum _ { i = 1 } ^ { n } p ( X _ { i } ) \log _ { 2 } ( 1 - { \frac { | X _ { i } - \mu _ { X } | } { d _ { X } } } ) .
$$

The inverse gives the measure of dissention, $\operatorname { D i s } ( X ) = 1 - \mathbf { C } \mathrm { n s } ( X )$

Both the normalized Shannon entropy for nominal and the dissention for ordinal variables have the same crucial properties: They are bound between 0 and 1, with values towards 0 corresponding to minimum human response variation and values towards 1 corresponding to maximum human response variation. While both measures take values in the same range, they are not directly comparable. We therefore select target variables —the set of survey variables we hold out to predict human responses for —based on the normalized entropy and dissention scores for nominal and ordinal survey variables separately.

## 3.2 Selecting Persona Attributes

In this subsection, we describe the different approaches we are benchmarking for the selection of an optimal set of persona attributes for each target variable. In contrast to, e.g., Rupprecht et al. (2026), we do not aim to produce a single set of attributes that is supposed to be optimal across all selected target variables, but we design our approaches to identify an optimal set of attributes for each target variable independently.

For RQ2, our main focus is on the comparison of statistical baseline approaches and LLM-based approaches for the selection of persona attributes. While the baseline approaches rely on the survey data and are agnostic towards the use of LLMs as the system for predicting survey responses, the LLM-based approaches are explicitly prompted to select the optimal set of attributes for predicting individual-level survey responses using a zero-shot persona-prompted LLM.

## Baseline Approaches

<table><tr><td colspan="2">Baseline Approaches</td></tr><tr><td>Feature Impor- tance</td><td>We use the same approach as Rupprecht et al. (2026), extracting feature importance scores from random forest classifiers fitted for the target variable, using all remaining variables as input features. We select the five most important features as the set of selected attributes.</td></tr><tr><td>Correlation</td><td>For the target variable, we calculate correlations with every remaining variable, select- ing the five variables with the highest correlations to the target variable as the set of selected attributes.</td></tr><tr><td>ilarity</td><td>Semantic Sim- We identify those survey questions that are semantically most similar to the target question as potentially most relevant for predicting it. We calculate semantic similarity based on embeddings of the survey questions produced using the ALL-MıNILM-L6- V2 available via SENTENCETRANSFORMERs 2, taking the five survey variables with question texts closest to that of the target variable as the set of selected attributes.</td></tr><tr><td>Human Response Variation LLM-based Approaches</td><td>We select the five variables with the highest human response variation scores as the set of selected attributes, following the traditional use of entropy as a measure of information (Shannon, 1948). Note that the human response variation of survey variables is independent of a given target variable—therefore, and in contrast to all other approaches, this approach results in a single set of attributes that is used across all target variables.</td></tr><tr><td colspan="2">The LLM is given the target question text as well as a list of all other variables&#x27; question</td></tr><tr><td>Set Selection Scoring</td><td>texts from the same survey. The LLM is then prompted to select the five most important survey variables for predicting individual-level survey responses to the target variable. The LLM is given the text of the target question as well as the text of a single question</td></tr><tr><td></td><td>from the same survey. The LLM is then prompted to score the importance of the corresponding survey variable for predicting individual-level survey responses to the target variable on a range from 0 (&quot;not important at all&quot;) to 100 (&quot;most important&quot;). This is repeated for all variables in the survey, resulting in a target-variable-specific importance ranking of survey variables. Based on this ranking, the five top-ranked variables are selected for persona construction.</td></tr></table>

Table 1: Persona attribute selection methods. Baseline and LLM-based approaches for selecting persona attributes for a given target variable. Apart from the Semantic Similarity approach, all baseline approaches have the full survey data as input. In contrast, the LLM-based approaches only have access to the question texts of the target variable as well as those of the remaining survey variables.

Table 1 further describes the different selection approaches, with the prompts used for the LLM-based approaches being documented in Appendix 7.2.

Naturally, we prevent the selection of survey variables that are variants of the target variable as persona attributes by removing all duplicate survey variables during pre-processing of the survey data (see Section 4.1) and by excluding the target variables from the list of available survey variables during persona attribute selection.

## 3.2.1 Stability

To check for the stability of the (non-deterministic) LLM-based selection approaches, we run them 100 times for each constellation of target question and LLM. For the two largest models, Qwen2.5-VL-32B and Llama3.3-70B, we reduce the number of runs to 30 due to computational resource constraints. Between runs, we shuffle the order in which the survey variables are presented to the LLM. Particularly for the set selection approaches, which fit a large number of survey variables into a single prompt, the LLM might have difficulties to put equal attention on all positions of the context window. Therefore, we expect the selected sets to differ between runs.

For the construction of personas based on the multiple runs of the LLM-based approaches, we use the five survey variables featured most often in the selected sets across runs for the set selection approach, and the five survey variables with the highest average score across runs for the scoring approach. We report our approaches for evaluating the stability of the LLM-based attribute selection approaches in Appendix 9.3.

## 3.3 Predicting Survey Responses

To not interfere with our evaluation of the different persona attribute selection approaches, we do not vary the prompts and generation parameters used for the actual survey response prediction. We also focus on zero-shot persona prompting, leaving the evaluation of alternative approaches such as few-shot prompting and fine-tuning for future work.

To predict the survey response of a survey participant for a given target variable, we insert the persona description constructed from the set of selected persona attributes, the text of the target question, as well as the range of possible response options as input into a structured prompt, asking the persona-prompted LLM to generate the most likely response given the persona description as output.

As the no-persona baseline for answering RQ1, we repeatedly prompt the LLM to predict responses to the given target variable without specifying a particular persona. We repeat this as many times as we have participants in a given survey. We provide more details on the response generation process as well as the prompts we use in Appendix 7.3.

## 3.4 Evaluating Survey Response Prediction Performance

We evaluate the performance of the persona selection approaches on the aggregated population-level. We compare the response distribution resulting from aggregating the individual-level response predictions of all personas with the human response distribution resulting from aggregating the responses of all survey participants for a given target question.

We use the Jensen-Shannon Distance (JSD) to assess how well the predicted responses approximate the human responses across the entire set of participants in a survey. The Jensen-Shannon Distance is normalized to the range from 0 to 1. Lower values indicate higher similarity between the two distributions and thus better performance of the approach in predicting the response distribution.

## 4 Experimental Setup

As laid out above, surveys offer the ideal environment for developing persona collections and evaluating the underlying methodology. We describe the different surveys we use in this work as well as the steps necessary to prepare them for the use in our persona prompting setup. We report the target variables we select for evaluation based on the measures of human response variation and discuss our choice of models for predicting survey responses.

<table><tr><td>Name</td><td>Year</td><td>N  $Q$ </td><td> $Q _ { S D } ^ { T }$ </td><td> $Q _ { B C } ^ { T }$ </td><td> $Q _ { A T } ^ { T }$ </td><td> $Q _ { R E } ^ { T }$ </td><td> $Q _ { C O } ^ { T }$ </td><td> $Q _ { N P } ^ { T }$ </td><td> $Q _ { D } ^ { T }$ </td><td> $Q _ { N } ^ { S }$ </td><td> $Q _ { O } ^ { S }$ </td></tr><tr><td>GGSS</td><td>2023</td><td>5,246 582</td><td>52</td><td>95</td><td>224</td><td>157</td><td>3</td><td>51</td><td>0</td><td>66</td><td>247</td></tr><tr><td>GSS</td><td>2024</td><td>3,309 207</td><td>38</td><td>31</td><td>74</td><td>7</td><td>4</td><td>31</td><td>22</td><td>51</td><td>54</td></tr><tr><td>WVS-DE</td><td>2018</td><td>1,528 348</td><td>31</td><td>60</td><td>203</td><td>23</td><td>3</td><td>25</td><td>3</td><td>64</td><td>199</td></tr><tr><td>WVS-US</td><td>2017</td><td>2,596 337</td><td>28</td><td>57</td><td>203</td><td>16</td><td>3</td><td>27</td><td>3</td><td>64</td><td>196</td></tr></table>

Table 2: Survey datasets used in our study. We use the latest available data from four surveys collected in two different countries (GGSS and WVS-DE from Germany, GSS and WVS-US from the US). N denotes the number of survey participants and $Q$ the number of variables in each of the four surveys. $Q _ { S D } ^ { T } , Q _ { B C } ^ { \hat { T } } , Q _ { A T } ^ { T } , Q _ { R E } ^ { T } , Q _ { C O } ^ { T }$ , and $Q _ { N P } ^ { T }$ are the number of variables annotated as the different types: socio-demographic (SD), behavioral/circumstantial (BC), attitudinal (AT), relational (RE), contextual (CO), non-participant (NP), and duplicate (D). $Q _ { N } ^ { S }$ and $Q _ { O } ^ { S }$ are the number of variables of types BC and AT annotated as being nominal (N) or ordinal (O), i.e., the survey variable from which we select target variables for evaluation. The discrepancy between $Q _ { B C } ^ { T } + Q _ { A T } ^ { T }$ and $Q _ { N } ^ { S } + Q _ { O } ^ { S }$ for GGSS is due to the presence of six continuous variables, which we do not consider for the selection of target variables.

## 4.1 Survey Data

We run the same set of experiments on data from different surveys in order to ensure the robustness and generalizability of our results. Additionally, we study two different socio-cultural contexts by using surveys from the United States and Germany. For both countries, we work with the respective general social survey —the General Social Survey (GSS) for the United States, and the German General Social Survey (GGSS) for Germany. By also using the data from the World Value Survey (WVS) for these two countries, we include an additional survey that is directly comparable acros the different countries it is run in. We document the steps necessary for preprocessing the survey data as well as the annotations of variable type and scale in Appendix 8.1. We report the number of variables featured in the different surveys as well as the results of our annotation efforts in Table 2.

## 4.2 Target Questions

To study the effects of the human response variation levels of different survey variables on the ability of personaprompted LLMs to correctly predict survey responses, we systematically select a set of target variables for evaluation of our experiments.

First, we consider only survey variables which were annotated as representing behavioral/ circumstantial or attitudinal information —these types of variables represent the much more interesting use case in the context of social science and survey research, compared to the prediction of socio-demographic (e.g., the individual’s age), relational (e.g., the profession of the individual’s father), contextual (e.g., the year in which the survey was taken), or non-participant (e.g., the interviewer’s age) survey responses.

Next, we calculate the appropriate measure of human response variation given the scale of the survey variable. In doing so, we produce a separate ranking of human response variation scores for nominal and ordinal variables. For both nominal and ordinal variables, we select the five variables with lowest and the five variables with highest human response variation scores as target variables. We exclude survey variables which have been answered by less than 500 participants from this selection to ensure sufficient data for evaluation.

![](images/7f8cf8a29dc1050d7b250eafe349b2a8c5121a11fa690232b4136a8cbb4fde0b.jpg)

![](images/1a2b1b02ce194470ee9f6dcee070bd8043808968d4c3f868876e59085d7361d0.jpg)

![](images/44ed570757f0685035f8684f3340063690598647caec72baf135c67772853652.jpg)

![](images/4c4fd3564f697ab436dea3c32bf48ab2cb81058946ee7ba86b126af61d612693.jpg)  
(a) Distribution of Human Response Variation as measured via normalized entropy for nominal response variables.

![](images/c4a5d7df56e773350ffca7cf2f062533182da25b329dcdff4ca3ed3579a1a6bd.jpg)

![](images/e6193e84be3c8005963d83fbe61d5d73a64e658875ddf005b8858ee9b8395cec.jpg)

![](images/dcb9dfbdea749532680842747761afcbebd6e700454b6037616fd374aa0a3cbf.jpg)

![](images/ecb49db66dd8822a0c373a3e8a4dc0d06ddcc4e3d150cf17c99969b67ddce26c.jpg)  
(b) Distribution of Human Response Variation as measured via dissention scores for ordinal response variables.

Figure 2: Distribution of human response variation across surveys for nominal (a) and ordinal (b) variables. Only variables of type behavioral/circumstantial and attitudinal are included, as we consider only those types of variables as potential target variables. The x-axis shows the human response variation, with higher values corresponding to higher levels of variation in the human responses to a given survey question. The y-axis shows the relative frequency of the buckets of widths 0.1. For each survey, we select the five nominal and the five ordinal variables with lowest and highest human response variation scores as target questions.

Figure 2 shows the distributions of human response variation scores across the four different surveys. Appendix 8.2 Tables 5 to 8 document the target variables that were selected based on their low and high human response variation scores, respectively. Appendix 8.2 Figures 12 to 15 show the response distributions of these target variables.

## 4.3 Model Selection

In the base configuration of our experiments, we always use the same LLM for the two steps potentially involving LLMs: the selection of persona attributes and the prediction of survey responses. We choose models from different model families and providers (Qwen models from Alibaba Cloud and Llama models from Meta AI) and with different parameter sizes (from 3B to 70B) in order to investigate the impact of model choice on the performance and to ensure generalizability of the persona prompting approach for survey response prediction. All models we use are available via Huggingface <sup>3</sup> and we use the vLLM <sup>4</sup> framework for interactions with models. Appendix 8.3 Table 9 gives an overview of the different models used as well as their (known) training cutoff dates. Data leakage from the surveys into the models we are using is practically impossible for GSS and GGSS and highly unlikely for the two WVS survey datasets.

![](images/5de120b4a09b76303dc336261e3a92a50d04288fdab4dc7b948a2f9b1528a3d5.jpg)  
Figure 3: Overview of our results. Persona prompting works better for questions with high human response variation and statistical baselines outperform LLM-based approaches for attribute selection. Each data point in this figure represents the averaged performance of a persona-prompted LLM in predicting the responses to 20 target questions (differentiated by position of data points and mean line, with high HRV questions oriented to the left and with red mean line, and low HRV questions oriented to the right and with blue mean line) from one of four different surveys (differentiated by shape, see legend). Along the x-axis are the different approaches used for selecting the persona attributes for persona prompting as well as the LLM baseline without persona conditioning (No Persona). The y-axis indicates the performance measured by the Jensen-Shannon Distance (lower is better) between the predicted and the actual response distributions across all participants in a given survey. The main takeaways from our experiments are that the persona prompting setups are better for predicting responses for target questions with high human response variation across all attribute selection approaches [red mean lines consistently lower than blue mean lines; RQ1], and that using statistical baselines based on the survey data alone (Correlation and Feature Importance) for selecting persona attributes lead to better persona prompting performance than the LLM-based attribute selection approaches (Set Selection and Scoring) [lower mean lines of statistical baselines; RQ2].

## 5 Results

We present the experimental results of applying the complete methodological pipeline shown in Figure 1 to the four selected surveys. While we generally observe that larger LLMs tend to perform better across attribute selection approaches, we focus our discussion of the results on the two research questions, separating them into disaggregated views for the different model families and surveys.

## 5.1 RQ1: Impact of Human Response Variation on Persona Prompting Performance

Figure 3 shows that the persona prompting approach performs better for predicting responses to target questions with high human response variation than to questions with low human response variation across all benchmarked attribute selection approaches. For the LLM-based prediction without persona conditioning (No Persona), this effect is reversed —the tested LLMs are better able to represent the distribution of human responses for questions with low human response variation. These two observations suggest that the inclusion of relevant persona attributes in prompts indeed allows the LLM to meaningfully model the variation in human responses.

In Appendix 9.1, we discuss the results disaggregated based on the two models families and based on the four different surveys. We show that the effect of human response variation on the predictive performance of predicting responses with and without personas generally also holds on these two levels of disaggregation, suggesting generalizability of our finding.

## 5.2 RQ2: Performance Ranking of Attribute Selection Approaches

Figure 3 shows that selecting persona attributes using statistical methods (Correlation and Feature Importance) leads to better alignment between the LLM-predicted and the actual survey responses than using the LLM-based selection approaches (Set Selection and Scoring). This is interesting since one could assume that LLMs can tap into the large amount of (scientific) background knowledge that they saw during pre-training to select the (theoretically) relevant attributes, or that they leverage potential differences in how knowledge is represented to select LLM-specific attributes. Our results show that this is clearly not the case since LLM-based attribute selections do not only perform worse but are also inconsistent in their selection of attributes (see Section 5.3).

When comparing the two LLM-based approaches against each other, we see that the set selection approach performs better than the scoring approach. In addition, using persona prompting for response prediction consistently leads to better alignment with the human responses than the LLM-baseline that does not condition on personas. This holds true across all approaches we test for the selection of persona attributes. Finally, the performance differences between the different attribute selection approaches are more pronounced for high than for low human response variation questions. We find that the performance order of attribute selection methods is consistent across model families and surveys (see Appendix 9.2).

## 5.3 Stability of LLM-based Attribute Selection Approaches

For the LLM-based attribute selection approaches (set selection and scoring), we separately analyze the consistency with which the six LLMs choose the five most important attributes for predicting a given target question across independent runs.

## 5.3.1 Stability of the Set Selection Approach

Figure 4 shows that the stability of the selection approach increases with model size. Both for the Llama and the Qwen models, more complex LLMs are able to more consistently select the same attributes as most important. Especially the large Llama model consistently picks the same set of attributes for some target questions, with the same attribute being selected as most important in almost 60% of runs across target questions. See Appendix 9.3 for a more extensive discussion of the set selection stability results.

Mean selection frequency by model — all surveys pooled  
![](images/a5f1e43152ad885aceee214d04d1abd8b66bb710e8a19a52c64eea943d2bc6b0.jpg)  
Predictor Rank (sorted by how often they are selected, descending)

Figure 4: The stability of the LLM-based set selection approach increases with model size. Each curve represents the distribution of the relative frequencies (y-axis) with which the different variables in a survey (x-axis) were chosen among the top-five most important attributes for predicting a given target question in one of 100 (30 for Qwen2.5-32B and Llama3.3-70B) independent runs. The dotted blue curve represents a perfectly stable selection mechanism, selecting the same five attributes in every single run, and the dotted red line represents a selection mechanism that is random guessing, selecting a different set of variables every time. We cut the x-axis at the 25 most frequently selected variables, given that afterwards the selection frequencies approach zero. The stability of the set selection approach increases with model size, with Llama3.3-70B selecting the same attribute as most important in 58% of the runs across target questions, with the same attribute selected as most important for some target questions in more than 80% of the runs.

## 5.3.2 Stability of the Scoring Approach

Figure 5 shows that LLMs are generally unable to reproduce the importance scores assigned to predictors across multiple independent runs, with observed mean Kendall’s W values between 0.25 and 0.5 indicating a weak to moderate correlation between the resulting rankings. For Llama models, the stability of the importance rankings produced in independent runs generally increases with model size. This is not the case for Qwen models, for which the largest model has the lowest mean correlation coefficient across different target questions. We see that especially for larger models such as Llama-3.3-70B-Instruct, the variance in rank correlation coefficients across runs is high, indicating that the order of scores can almost perfectly be reproduced for some target variables, but fluctuates highly for others. The opposite is the case for smaller models—their variance of rank correlation coefficients is lower, indicating that regardless of the target question the importance scores generated across different runs fluctuate highly.

![](images/da240d8714da0cafe951fb64d24a50226214e099558f31ec8bc4ccff7663e762.jpg)  
Figure 5: LLMs struggle to reproduce the same variable importance ranking across different runs in the LLMbased scoring approach. Each boxplot represents the distribution of the Kendall’s W values, a concordance correlation measure calculated from the variable importance rankings that we construct from repeated independent runs for the 20 target questions. We disaggregate results based on human response variation and the used LLM. The correlation between the importance rankings produced in independent runs is generally low. For Llama models, however, we see that the correlation increases with model size, implying higher consistency with which attributes are scored as important.

## 6 Discussion and Implications

For practitioners interested in simulating survey responses via persona prompting, our results suggest a two-step framework for practical use. In a first step, our measures of human response variation help identify types of survey questions for which the simulation can be expected to work best. As we have shown above, this will generally be the case for questions with high high human response variation, i.e., questions (and potentially tasks and topics) that are highly contested. In a second step, our results suggest to use approaches informed by existing survey data for the same or a similar question to identify the attributes that are most important for persona creation and lead to the best alignment between predicted and actual responses. This is similar to the idea by Hu and Collier (2024), who suggest to use simplistic regression analysis based on small-scale human data to estimate the likely performance of simulation efforts, enabling informed decision making before the commitment of additional resources.

Such a simplistic framework could potentially pave the way towards an AI-based adaptive survey design framework, in which specific questions are shown only to selected groups of participants, increasing the effectiveness of the allocation of survey resources. If we know that some types of questions are particularly suitable for simulation through persona prompting and that some more majority-aligned subgroups of the population can be modeled particularly well with

LLMs, then we can aspire to improve on the survey response prediction task even with the imperfect performance of current social simulation methods.

Interestingly, we observe that the statistical baseline approach which relies on pure correlations between the target variable and the input variables leads to the best selection of persona attributes and outperforms more complex approaches that rely on LLMs or semantic similarity measures. This implies that the response prediction via persona prompting LLMs requires input attributes that are correlated with the outcome variable —something that is obvious for existing methods, but much less so for LLM-based approaches. For traditional Machine Learning applications to perform well, feature selection approaches are typically designed to select features that are highly correlated with the target variable, but not highly correlated among each other. For persona prompting LLMs we do not find any evidence that correlation among selected features matters since the best performing approach simply focuses on the correlation between individual input variables and the target variable.

One of the assumptions of using persona prompting is that LLMs are able to activate internal representations of knowledge based on the provided persona description, but so far there has been no effort to study whether the response generation is then driven by correlations and associations that can also be observed in traditionally collected human survey data, or whether it relies on LLM-specific correlations and associations between persona attributes and target variables. While our work is far from providing definite answers to this question, it serves as a first indication that the internal representations of survey responses in LLMs might be similarly correlated as in human survey responses. If that was not the case, we would expect the LLM-based selection approaches, having access to the model’s internal representations, to be at an advantage and thus to perform better in predicting survey responses than survey-datainformed alternatives—which we do not observe.

A similar and promising direction of future work would be on the inner workings of the LLM in generating its responses conditioned on a given persona (e.g., using inner probing (Maiya et al., 2025)). While we currently control the inputs via the selection of attributes to feed into persona descriptions, we can only observe the generated outputs. Studying the parameters and connections that are actually activated during response generation and comparing them to some representations of the persona attributes provided as inputs would make for very relevant work towards forming a better understanding of the persona prompting approach. This strain of research connects to the literature on explainable AI and the concept of faithfulness in particular, which is concerned with establishing whether the explanations provided by the model, in our case the selection of persona attributes deemed important for response prediction, match the inner workings of the model (Chuang et al., 2026; Madsen et al., 2024; Matton et al., 2025). In our case, this would mean checking whether the provided persona attributes in the prompt lead to activations in the model that actually influence the predicted response

## Limitations

As is often the case in this type of work trying to fathom the frontiers of novel methods and applications, there are a number of limitations naturally arising by virtue of being incapable to consider every potentially relevant factor at once. In our case, the list of such omitted influence factors is long, including but not limited to: the design choices of the persona attribute selection methods (always choosing five attributes instead of varying the number; trying different prompts or even entirely different LLM-based attribute selection strategies); the choice of LLMs (using SOTA closed models; using models with known training data; using base models; using different models for attribute selection and for response generation); the persona prompting response generation setup (using different persona formats; using different generation parameters; using different response parsing mechanisms). While it would be interesting to explicitly account for and systematically vary all these factors, this would further blow up an already highly compute-intensive experimental setup.

Another type of omission results less from a (computational) resource constraint but from a lack of limited capacities and attention as a researcher. For this work in particular, a number of alternative, equally relevant baselines to those we already used are perceivable, in our case especially for the attribute selection approaches. Selecting attributes fully at random or not selecting at all, instead using all attributes for persona constructions, would help to further shed light on the importance of the attribute selection step. Selecting attributes based on established social science theories would help to further improve our understanding of why persona prompting works, whether it works the way it is supposed to, and when it should be expected (not) to work. However, such a theory-informed attribute selection approach is inherently difficult since there is often no single theory for each construct and even if there was one, there is often no straightforward way of turning a theory into a agreed upon ranked lists of persona attributes. Finally, we currently lack a baseline that uses traditional survey imputation methods for the response prediction task to better contextualize the performance and to provide convincing answers to the question whether these efforts are worth our while. However, we already know from previous work that in most settings persona-prompted LLMs are competitive with a random fores classifier used as the imputation baseline alternative (Rupprecht et al., 2026).

In addition, there are some parts of this work for which additional qualitative analyses could lead to additional interesting insight. For instance, we are currently not conducting any qualitative check of and comparison across the sets of selected attributes —neither across attribute selection approach to uncover biases or preferences, nor across LLMs to identify model-specific selection patterns, nor across surveys and countries to evaluate the cultural adaptability and sensitivity to nuance of the selection methods.

Apart from these practical limitations specific to our work, there exists a growing number of position papers on the promises and limitations of LLM-based social simulations, in-silico samples, and persona prompting. Anthis et al. (2025) describe five key challenges that LLM social simulations must address in order to achieve their promise: lack of human diversity, systematic biases, sycophancy, alienness and non-human like mechanisms leading to superficially accurate results, and generalizability across contexts.

## Ethical Considerations

Kirk et al. (2024) provide an extensive discussion of the benefits and risks of aligning LLMs to individuals. While our main application concerns the aggregated population level, the process of generating individual-level responses via persona prompting an be viewed as analogous to aligning LLMs to individuals. One individual-level risks discussed by Kirk et al. (2024) that is relevant for our context is the danger of essentialism and profiling, i.e., categorizing individuals based on an insufficient set of demographic information. The risk of flattening representation and caricaturing individuals has also been raised by Cheng, Durmus, and Jurafsky (2023), Cheng, Piccardi, and Yang (2023), and Wang, Morgenstern, and Dickerson (2025). We would argue that our work is contributing to reduce risks of essentialism, since we are actively exploring ways of selecting attributes that go beyond the limited set of socio-demographic attributes usually tried first in persona prompting applications

Another ethical consideration for the work with human participant data for developing and evaluating social simulations concerns participants’ privacy, and especially the danger of collecting and potentially disclosing increased quantities of sensitive information. While we acknowledge the importance of this consideration, we argue that our work is not adding to this risk: we are only reusing already collected survey data, and we make sure to only use locally-hosted models for our experiments, avoiding the possibility of any data leakage into hosted-models to occur (Cheng et al., 2026).

On the societal level, relevant ethical considerations discussed by Kirk et al. (2024) include the displacement of labor, environmental harms through increased LLM usage, and the potential for malicious use. These considerations are and remain relevant for all LLM-based applications.

Finally, there is the question for the overarching goal of exploring the potential of social simulations and the future of survey research with human participants. We explicitly caution against the outright replacement of human survey participants through silicon samples, but at the same time advocate for keeping on exploring the potential of LLM-based applications in the survey context. Only this type of methodological evaluation and advancement allows us to make informed decisions about when and when not to use novel methods and paradigms.

## References

Ackermann, K., Auspurg, K., Bühler, C., Carol, S., Friehs, M.-T., Hillmert, S., & Tausendpfund, M. (2025). German General Social Survey - ALLBUScompact 2023. https://doi.org/10.4232/1.14572

Aher, G. V., Arriaga, R. I., & Kalai, A. T. (2023). Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies. Proceedings of the 40th International Conference on Machine Learning. https://proceedings.mlr.press/v202/aher23a.html

Anthis, J. R., Liu, R., Richardson, S. M., Kozlowski, A. C., Koch, B., Brynjolfsson, E., Evans, J., & Bernstein, M. S. (2025). Position: LLM Social Simulations Are a Promising Research Method. Proceedings of the 42nd International Conference on Machine Learning. https://proceedings.mlr.press/v267/anthis25a.html

Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of One, Many: Using Language Models to Simulate Human Samples. Political Analysis, 31(3), 337–351. https://doi.org/10.1017/ pan.2023.2

Bisbee, J., Clinton, J. D., Dorff, C., Kenkel, B., & Larson, J. M. (2024). Synthetic Replacements for Human Survey Data? The Perils of Large Language Models. Political Analysis, 32(4), 401–416. https://doi.org/10.1017/pan.2024.5

Brown, M. A., Atreja, S., Hemphill, L., & Wu, P. Y. (2025). Evaluating How LLM Annotations Represent Diverse Views on Contentious Topics. arXiv. https://arxiv.org/abs/2503.23243

Chakraborty, S., Qiu, J., Yuan, H., Koppel, A., Huang, F., Manocha, D., Bedi, A., & Wang, M. (2024). MaxMin-RLHF: Towards Equitable Alignment of Large Language Models with Diverse Human Preferences. ICML 2024 Workshop on Models of Human Feedback for AI Alignment. https://openreview.net/forum?id=NCQp4KpT8R

Chen, J., Pan, X., Yu, D., Song, K., Wang, X., Yu, D., & Chen, J. (2024). Skills-in-Context Prompting: Unlocking Compositionality in Large Language Models. arXiv. https://arxiv.org/abs/2308.00304

Cheng, M., Durmus, E., & Jurafsky, D. (2023). Marked Personas: Using Natural Language Prompts to Measure Stereotypes in Language Models. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). https://doi.org/10.18653/v1/2023.acl-long.84

Cheng, M., Piccardi, T., & Yang, D. (2023). CoMPosT: Characterizing and Evaluating Caricature in LLM Simulations. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. https://doi.org/ 10.18653/v1/2023.emnlp-main.669

Cheng, S., Xu, H., Meng, S., Hao, S., Yue, C., & Li, Z. (2026). The Privacy Paradox of LLMs: User Perceptions and the Reality of PII Leakage. Proceedings ofthe 2026 CHI Conference on Human Factors in Computing Systems. https://doi.org/10.1145/3772318.3791809

Chuang, Y.-N., Wang, G., Chang, C.-Y., Tang, R., Zhong, S., Yang, F., Wen, A., Du, M., Cai, X., Braverman, V., & Hu, X. (2026). FaithLM: Towards Faithful Explanations for Large Language Models. Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers). https://doi.org/10.18653/v1/2026.eacl-long.177

Davern, M., Bautista, R., Freese, J., Herd, P., & Morgan, S. L. (2025). General Social Survey 1972-2024. https : //gss.norc.org

EVS/WVS. (2024). Joint EVS/WVS 2017–2022 Dataset (Joint EVS/WVS). https://doi.org/10.4232/1.14320

Giorgi, S., Liu, T., Aich, A., Isman, K. J., Sherman, G., Fried, Z., Sedoc, J., Ungar, L., & Curtis, B. (2024). Modeling Human Subjectivity in LLMs Using Explicit and Implicit Human Factors in Personas. Findings of the Association for Computational Linguistics: EMNLP 2024. https://doi.org/10.18653/v1/2024.findingsemnlp.420

Horton, J. J., Filippas, A., & Manning, B. S. (2023). Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus? (Working Paper No. 31122). National Bureau of Economic Research. https://doi.org/10.3386/w31122

Hu, T., Baumann, J., Lupo, L., Collier, N., Hovy, D., & Röttger, P. (2026). SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors. International Conference on Learning Representations. https ://proceedings . iclr. cc/paper \_ files/paper/2026/file/2fc4d6a6400addda7493b9db258c34e0 - Paper-Conference.pdf

Hu, T., & Collier, N. (2024). Quantifying the Persona Effect in LLM Simulations. Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers). https://doi.org/10.18653 v1/2024.acl-long.554

Hwang, E., Majumder, B., & Tandon, N. (2023). Aligning Language Models to User Opinions. Findings of the Association for Computational Linguistics: EMNLP 2023. https://doi.org/10.18653/v1/2023.findingsemnlp.393

Kirk, H. R., Vidgen, B., Röttger, P., & Hale, S. A. (2024). The Benefits, Risks and Bounds of Personalizing the Alignment of Large Language Models to Individuals. Nature Machine Intelligence, 6, 383–392. https://doi. org/10.1038/s42256-024-00820-y

Lappin, S. (2024). Assessing the Strengths and Weaknesses of Large Language Models. Journal of Logic, Language and Information, 33, 9–20. https://doi.org/10.1007/s10849-023-09409-x

Lee, S., Peng, T.-Q., Goldberg, M. H., Rosenthal, S. A., Kotcher, J. E., Maibach, E. W., & Leiserowitz, A. (2024). Can Large Language Models Estimate Public Opinion about Global Warming? An Empirical Assessment of Algorithmic Fidelity and Bias. PLoS Climate, 3(8). https://doi.org/10.1371/journal.pclm.0000429

Li, M., & Conrad, F. G. (2026). Persona-Based Simulation of Human Opinion at Population Scale. arXiv. https: //arxiv.org/abs/2603.27056

Lutz, M., Sen, I., Ahnert, G., Rogers, E., & Strohmaier, M. (2025). The Prompt Makes the Person(a): A Systematic Evaluation of Sociodemographic Persona Prompting for Large Language Models. Findings ofthe Association for Computational Linguistics: EMNLP 2025. https://doi.org/10.18653/v1/2025.findings-emnlp.1261

Luz de Araujo, P. H., Röttger, P., Hovy, D., & Roth, B. (2025). Principled Personas: Defining and Measuring the Intended Effects of Persona Prompting on Task Performance. Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. https://doi.org/10.18653/v1/2025.emnlp-main.1364

Madsen, A., Chandar, S., & Reddy, S. (2024). Are Self-Explanations from Large Language Models Faithful? Findings of the Association for Computational Linguistics: ACL 2024. https://doi.org/10.18653/v1/2024.findings-acl.19

Maiya, S., Liu, Y., Debnath, R., & Korhonen, A. (2025). Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). https://doi.org/10.18653/v1/2025.acl-long.444

Matton, K., Ness, R., Guttag, J., & Kiciman, E. (2025). Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations. International Conference on Learning Representations. https://proceedings.iclr.cc/paper\_ files/paper/2025/file/b5ec50eb177908f21f78ed0d76ed525c-Paper-Conference.pdf

Neumann, A., Kirsten, E., Zafar, M. B., & Singh, J. (2025). Position is Power: System Prompts as a Mechanism of Bias in Large Language Models (LLMs). Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency. https://doi.org/10.1145/3715275.3732038

Park, J. S., Zou, C. Q., Kamphorst, J., Egan, N., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Liang, P., Willer, R., & Bernstein, M. S. (2026). LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals. arXiv. https://arxiv.org/abs/2411.10109

Plank, B. (2022). The “Problem” of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation. Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. https://doi.org/ 10.18653/v1/2022.emnlp-main.731

Qu, Y., & Wang, J. (2024). Performance and Biases of Large Language Models in Public Opinion Simulation. Humanities and Social Sciences Communications, 11. https://doi.org/10.1057/s41599-024-03609-x

Rupprecht, J., Fröhling, L., Wagner, C., & Strohmaier, M. (2026). German General Social Survey Personas: A Survey-Derived Persona Prompt Collection for Population-Aligned LLM Studies. Proceedings ofthe Fifteenth Language Resources and Evaluation Conference (LREC 2026). https://doi.org/10.63317/2sod6uekicbg

Sanders, N. E., Ulinich, A., & Schneier, B. (2023). Demonstrations of the Potential of AI-based Political Issue Polling. Harvard Data Science Review, 5(4). https://doi.org/10.1162/99608f92.1d3cf75d

Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). Whose Opinions Do Language Models Reflect? Proceedings of the 40th International Conference on Machine Learning. https://proceedings. mlr.press/v202/santurkar23a.html

Sarumi, O. O., Neuendorf, B., Plepi, J., Flek, L., Schlötterer, J., & Welch, C. (2024). Corpus Considerations for Annotator Modeling and Scaling. Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). https://doi.org/10.18653/v1/2024.naacl-long.59

Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal, 27(3), 379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

Sorensen, T., Moore, J., Fisher, J., Gordon, M. L., Mireshghallah, N., Rytting, C. M., Ye, A., Jiang, L., Lu, X., Dziri, N., Althoff, T., & Choi, Y. (2024). Position: A Roadmap to Pluralistic Alignment. Proceedings of the 41st International Conference on Machine Learning. https://proceedings.mlr.press/v235/sorensen24a.html

Suresh, S., Mukherjee, K., Yu, X., Huang, W.-C., Padua, L., & Rogers, T. (2023). Conceptual Structure Coheres in Human Cognition But Not in Large Language Models. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. https://doi.org/10.18653/v1/2023.emnlp-main.47

Tastle, W. J., & Wierman, M. J. (2006). An Information Theoretic Measure for the Evaluation of Ordinal Scale Data. Behavior Research Methods, 38, 487–494. https://doi.org/10.3758/BF03192803

Wang, A., Morgenstern, J., & Dickerson, J. P. (2025). Large Language Models that Replace Human Participants Can Harmfully Misportray and Flatten Identity Groups. Nature Machine Intelligence, 7, 400–411. https: //doi.org/10.1038/s42256-025-00986-z

Wang, X., Antoniades, A., Elazar, Y., Amayuelas, A., Albalak, A., Zhang, K., & Wang, W. (2025). Generalization v.s. Memorization: Tracing Language Models’ Capabilities Back to Pretraining Data. International Conference on Learning Representations. https : / / proceedings . iclr . cc / paper \_ files / paper / 2025 / file / 7cdf000d22c6cda21f3cbd7467aaf26f-Paper-Conference.pdf

Weeber, F., Neplenbroek, V., Batzner, J., & Padó, S. (2026). One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). https://doi.org/10.18653/v1/2026.acl-long.2079

Wilcox, R. R. (2008). Variance. In P. J. Lavrakas (Ed.), Encyclopedia of Survey Research Methods (pp. 940–942). Sage Publications, Inc. https://doi.org/10.4135/9781412963947.n620

Xie, Y., Liang, L., Li, S., Lu, Y., Xiao, Z., Shi, M., Huang, J., Wang, M., & Xie, Y. (2026). Evaluating the Statistical Realism of LLM-Generated Social Science Data. Proceedings of the National Academy of Sciences, 123(19). https://doi.org/10.1073/pnas.2538145123

## 7 Appendix: Prompts

## 7.1 Paraphrasing of Survey Questions

In this subsection, we document the prompts used for paraphrasing the survey questions. This step was done using GPT-5.1 accessed via the OpenAI API with default parameter choices. Generated paraphrases were assessed manually. The quality and suitability of the paraphrases were found to be sufficiently high after manual validation, which is not surprising given how trivial the task was.

![](images/9624cd32a7f672fafe33c2090f565997139a78b213d7133af2944c1d9561b594.jpg)

Figure 6: User prompts used for the paraphrasing of survey questions. Placeholders are marked as <placeholder> and are replaced with the respective information.  
![](images/17ed17840fe72c2cc65f47b9c1969a68722d879994a02e05e7833f9d35b33a3b.jpg)  
Figure 7: User prompts used for the batch paraphrasing of survey questions. This prompt is appended to the base paraphrasing prompt for all but the first batch.

![](images/eb66d5bfc05efcaac0a8ae428e61486be58b380def9a189ada3fe795f948f291.jpg)

Figure 8: System and user prompts used for the Set Selection persona attribute selection approach.  
![](images/ecc551702152dd44640f5cf628557953208fbed4118863bf2343a0b944f033c9.jpg)  
Figure 9: System and user prompts used for the Scoring persona attribute selection approach.

![](images/fab115812f6a85122e06a49bb7b6e231a61f2d7a1f650d3a6d972ae0fcc01192.jpg)

Figure 10: System and user prompts used for the persona prompting response prediction approaches.  
![](images/b2e08d3c38e178ecbdd7822ab528d37ab688415f82168ab12eb5109b43eb2d45.jpg)  
Figure 11: System and user prompts used for the no persona baseline for response prediction.

## 8 Appendix: Experimental Setup

## 8.1 Survey Data

In the following, we document the preprocessing steps for preparing the survey data for our persona prompting experiments, as well as the codebooks used for the annotation of the scale and type of the survey variables.

For all surveys, we use the latest publicly available data release. To prepare the survey data for use in our experiments, a number of pre-processing steps are necessary, mostly transforming and standardizing the format of the survey questions and responses. Most surveys still rely on pdf codebooks to make their data accessible and legible to researchers. While this might work for more traditional survey applications, this makes the handling and use of the data at the scale required for persona applications burdensome. Thus, the first step is to bring the codebook, and particular the texts of the survey questions and the available response options, into machine-readable formats. Where necessary, we create custom codebook-parsers to extract the relevant information in a structured way.

Once the codebooks have been parsed, we annotate the survey variables for their scale (nominal, ordinal) as well as for the type of information they represent (socio-demographic, behavioral/circumstantial, attitudinal, relational, contextual, non-participant). See Tables 3 and 4 for the respective annotation codebooks. We require the scale of a variable for deciding which measure of disagreement is appropriate, and the type of information to identify the subsets of survey variables considered relevant for persona construction and evaluation. In addition, we identify and remove variables that can be considered direct duplicates of each other.

## Variable Scale

Nominal All survey variables with nominal response options, e.g., binary "yes"/"no" or "marked"/"unmarked" choices, as well as categories without clear order or clearly defined extreme values, e.g., party affiliation.

Ordinal All survey variables with ordinal response options, i.e., with clear order and clearly defined extreme values, e.g., Likert scales, age (measured in years), or year of birth.

Continuous All survey variables with continuous response scales, i.e., an open-ended, numeric response field. This mainly applies to variables for survey weights or different variants of income.

Table 3: Annotation codebook for the scale of survey variables. We annotate the scale of the survey variables in order to calculate the appropriate measure of human response variation.

## Variable Type

Socio- We annotate all variables that concern one of the following aspects as socio-demographic:   
Demographic age, gender, sex, income, wealth, race/ethnicity, origin, migration status, and education. In addition, we include "core" job-related variables, both for the current and the previous job: employment status, job title, and time in job.   
Behavioral/ Cir- In this category, we include all variables that concern past (e.g., "Have you smoked in   
cumstantial the last 30 days?") or future (e.g., "Do you plan to participate in upcoming elections?") behavior, as well as customary behavior (e.g., "Do you regularly attend church?"). In addition, we include circumstances (e.g., "Rate your general health", "Have you experienced bullying when growing up?") and general sentiments and feelings (e.g., "Do you feel depressed?", "How satisfied are you with your life?"). Additional details on the participant’s employment are also collected in this category, e.g., weekly working hours, possible risk factors at work, or personnel responsibility.

Attitudinal In this category, we include all variables that measure the participant’s opinions, attitudes, sentiments, and beliefs. This includes many questions starting with "In your opinion ..." or "Do you agree/disagree ...". This category also includes questions on the participant’s religious, political, and ideological beliefs.

Relational In this category, we include all variables that are not on the survey participant in isolation, but on their relationship to other members of their family or household, or characteristics and attributes of these other members. These other members typically are the participant’s spouse or partner, their parents, their children, or other members of their household.

Contextual In this category, we include a small set of variables that are immediately relevant for the survey context. They include the participant’s ID, as well as the country and year in which the survey was conducted.

Non-Participant Variables that are not measuring attributes of the participant. These include technical variables such as weights or flags, questions about the interview or attributes of the interviewer, as well as the interviewer’s perception of the participant.

Table 4: Annotation codebook for the type of survey variables. We annotate the type of the survey variable in order to only select relevant sets of survey variables for the selection of persona attributes and target variables.

## 8.2 Target Questions

Figures 12 to 15 show the human response distributions of the 20 target variables selected for each survey. The figures are separated into variables with nominal and ordinal scale, as well as into low and high response variation. Tables 5 to 8 provide the texts of those target questions, as well as their human response variation and the number of valid responses. The variable ID (Var-ID) provided both in the figures and in the tables serve as a link between the two sources of information.

![](images/2f7304043c0973df5f8bed58c888f2aa17faa21f4e8b4da00adb9fc427c0a21a.jpg)

![](images/847b5628ac4906d9092f92e38da9e302242c0d0dd078baeb6338062443f22680.jpg)

![](images/20d58971ae23cf0c88de8b4dbdb2c96c838f5fb9fa73c028f4e89cff74032d34.jpg)

![](images/c162093ea737a2f2c094da770283bf916be4b2d0d3b065c973c555c05386d991.jpg)

![](images/601ed621ce35de60a710400e09aff840a74a9624c89156d7428232aab30f3360.jpg)

![](images/ccfc413a6ba8b90e4783a695e766c23c755df46687f114541f57decb6cd38232.jpg)

![](images/5bed926fd6746da6d18e1fcd80704c58e7a093dc9420ae0b905812964c490539.jpg)

![](images/716583adde15987d2496da9bdd34a71d477831d014bad939edc666370497e520.jpg)

![](images/bae0ba9f3858d123a46da3a3b6a098177c25de5e28c90bae02ad73e349728ddd.jpg)

![](images/98978fc934d503101d3ca583a8d2bd103d1b35efe64a55bf673304e66908bb7b.jpg)

![](images/a3a1e9d61b3f9ab545b2f0ba5b5d9667266e72b740baa4bf22fb76dd14786e50.jpg)

![](images/9424b7b35699edd03204599162340d6c6b23abd34b50b732b013b19dd3b76822.jpg)

![](images/3b59270976cf9665c9cca59f79eaf59b367f0f358d340e97283a31d0f071d8e3.jpg)

![](images/f7c6e2a6865179e70219a7aa4d469b42c4695e2b9af0ab189e83aa56f560916a.jpg)

![](images/d419a6bf4a3bfc102cd4cc9b247e3ce1be5fcd74c96f76b79e8d5bda13d7b1eb.jpg)

![](images/17c77102f035b3cb722aa35dfe464fcfcfee23d39a8772a3487dba009cfd3b97.jpg)

![](images/0aefb1c0566ed8e7d8f99f73040b1bde12367492682714eecf024ac51af539a6.jpg)

![](images/f8ba0aa10542a4d8a6d5739fbe66b651c70d0d0179fa731212d8d76aa53538d3.jpg)

![](images/3db82158d664f1d75db87410918e7981e59e83f6917ae67d45d0fe58594e4650.jpg)

![](images/080b0ed205cbbb123da8a2e277f00d7a007b2f106e18dc34b129a74a2faf19cb.jpg)  
Figure 12: Human response distribution of the five nominal target variables with lowest human response variation. Each column represents target questions selected for one of the four surveys, with human response variation increasing from top to bottom.

![](images/04b4f0f47505ffc15f5aee6b7edeb5fa341346eee64b79ba3d81210f4608a9e0.jpg)

![](images/31e88a67e8d613282aea0d6f47d76c0454c98d80d48e10558b0dfb10f38d679b.jpg)

![](images/f71c5d9a984d939b724b19e2e87a04c3ac7cb152031ee64205af1809afc84040.jpg)

![](images/a97fc4a1d00c9922be926fcf86c8051edb50078bcc506e1fff2a2acf4cad3f0f.jpg)

![](images/05b879a26dcdb24017ab0d278002cce055445b7aa2735ff8a84da41822b81691.jpg)

![](images/bc70e9da953adcdf6c48655753e0f438b90405c27ff01cd439ddd65325ca37f0.jpg)

![](images/62cf058094ca51c66c9e70cad680a76ceeb5b368bac5f20b241d24eedc0c2f07.jpg)

![](images/98a4e3920d1f73631df85714d45df509b51a18111d902824b1499e67b19ff702.jpg)

![](images/206399a03cc95fef4c82ed4ea99ff7b1f0beb953a1ce5ad9205b0f36180ed172.jpg)

![](images/4bafbe678fcb87e11cd113dda2e326bdea0a6a390ad2ca13370916cf77c5f47a.jpg)

![](images/2a75c04978781463eb02b0ce0521f8ec3e2561e69de0c92b0cb4eba53fe1f888.jpg)

![](images/ef4d22e95304dd49cf90da76725f0b93150e32b71507393ec9e56a37c3b29627.jpg)

![](images/50df83929c943488fd500828b93737090516c200d82aff14ab8694b21db9c8f8.jpg)

![](images/07490bae4a8f11b35cea9d66aea97b9c38b6ec49f05d34a57a1e200b7919a3ce.jpg)

![](images/e9e8ef17ce253761af9a381a37c6b2c572f931c82e2c3324bf024f2af2f788f2.jpg)

![](images/2d834fc39f488d7f46385a6385c6efe4392dedda9526a931f9092a598a5511d6.jpg)

![](images/7f0723124fb5b75f82f4113e8afd09a56953b9d5bb24d50ec49609ff94af27ec.jpg)

![](images/9dcc05d829f781209d006f24005d2bdf3f716b7e11381de3cdf56c207ca91985.jpg)

![](images/2c2fc84457e16a99b24145a6d17b1e6d28c0594d646cb8219322bed628bab0af.jpg)

![](images/dba4f11fa5190a5faa6511767f817c23ee7675a8a868a3bddab8eea09a959481.jpg)  
Figure 13: Human response distribution of the five nominal target variables with highest human response variation.

![](images/3a0cdd7ad8077e8f2efd121ba924c110f5a91e8c2cb24e14e659cf169aeff21c.jpg)

![](images/61eae7168d8714dc595dfd9048ed0849e1069b7683be0fde9f911a1c7002bf15.jpg)

![](images/d1c9dfbd3f6955759fd96a418eb0ec0183edd16482165f266fa8482996f72efe.jpg)

![](images/4a59cb383820a42ad3617bf01bfd5c83920dbd434c2e48868a0878e54220de3c.jpg)

![](images/3802f58ee75c2a380d413c9b4a53239d0ace32e53c51f1caa5bf230aafcf4931.jpg)

![](images/35cdcf2a4bd0a9f188b423bd78f9b5b64210d6549af9f11fa719381181c67a4c.jpg)

![](images/745defb7ee1f1b5305168448b400235608243aed1c60c35d8c9a869086b96a71.jpg)

![](images/bb1719b578283be262d7547c8530928d03f1974afa4071041b58381bb12b3f4a.jpg)

![](images/d5e3871c9d71f6f9c100535b5009f7df798248314ba51d8761944baefa79672f.jpg)

![](images/d44cdab60d9907ef4b4919c81c1089a0b3593f318105378e85af575775c3650d.jpg)

![](images/631dec525e218a324268d7f7dce76eb65f4465f9bfd5b973fe013534788d628b.jpg)

![](images/815f8f4611c18a4e91f79c03126719575472f46c1f726aedf78a70adbad292dc.jpg)

![](images/3f5824286bb75d32e9bd340aa4ea5bc0cd8a2c1bd078c2b7a2de4c8c20dc7cf5.jpg)

![](images/62a891f4373f3ceffd5aba1459f55b89f5997db0270f20e96857e1f9510f9530.jpg)

![](images/00a6a1f8f0e43b23e55acfaa6dfbd89b2d2f1b18f9959387c8a405b413f0071a.jpg)

![](images/c0368769175d314b03dea8bda2937fefc19944ad9ce6c4d2882380e78210da70.jpg)

![](images/7a4a33dd3c75fec221c4a90df95a4559865665b62a09de8cdacb0e70a00f2dfe.jpg)

![](images/0b9b995c8ab5d0af22af71d49ba1afc1a1a3c5138ba616d338509e43ecaf65d2.jpg)

![](images/b87b4a14ea7adfec08167c8a560ce66f8669c5f3c6e69d78bfcc28dc831ae2d3.jpg)

![](images/cf692c7afa52d7e95480f0100ec6d97f9be264ecd91e80ca5dfdde2dac260bb7.jpg)  
Figure 14: Human response distribution of the five ordinal target variables with lowest human response variation.

![](images/70b79c2c7bb45c58d282cc9d9fae9ad99a4a543c6b456eae58d83b1a82224484.jpg)

![](images/a01784084e946c954419cbc8115244a476798a02fb169096e3d1143b733e22a2.jpg)

![](images/9142f8632cc61f0ccefa99fda6e0bfaf2bfb4b6cc919fddb27badd7989df9f28.jpg)

![](images/354c08f91f6c86460bb4aadd4bee0f19c4d972b63c7ec2149a54c6581dcb7ae9.jpg)

![](images/7bb8d0eb4ad7bdbc8742019748821c9d9b9035f00ff81c2f8db6ce97e2ec007f.jpg)

![](images/d4b4014c5bb7b73c079e9a35d27757eed5510923f2ee8d2466fe51caa227a1a6.jpg)

![](images/e66d891cef4e212e94f2a52a1e03d517261bc478d7d5794d07e5a4b6024e3ebf.jpg)

![](images/90384968f304bd3391045620050da9ba5b7810746c18a9f0868ffce13ead2726.jpg)

![](images/5bf23a004187092783dec643ef1c6770a3190d8e977119901d852035ae4b5a5c.jpg)

![](images/11762ce2550318a5cdd6c539299dcd221a507a2efe9da8d88cdb9968012e4d7c.jpg)

![](images/29c02fdaf9d953154651a59d8cf5df11407a835ec7f18240bcc303d9b3e24b1d.jpg)

![](images/c13bfe3b858369a62078eaa72e9234af3136372d1e070923111e53a3303c86ea.jpg)

![](images/e82b0513eabf39fac4b0d0cc48645e70df741cd7f88376541ee55cf8b81bf60f.jpg)

![](images/d2b93fc071a8690aea39e17ac8e159e3a78ed24071e65fa9c3bf298f453ee814.jpg)

![](images/a81f8912815ca3096132fc57356d6ba1f64694c30f8292ec397d06dc423f153c.jpg)

![](images/0231b55564c6a809941f2b56763ddbfdb22f234a0c92d9f7a144c265c86dacd8.jpg)

![](images/61c85bee0d5b210a0ce073ae1faf0326497685294aa018d1bf7e46c5bfb0c858.jpg)

![](images/0cf71f8fa13fa939ef925283cfa77b2e17d12df591b7b5d20aa47ba98e90539e.jpg)

![](images/69dd8f2db8d1cc6f05b87aca7e29b10c8ca3ea5c2d92f8e8d96d58732060e1eb.jpg)

![](images/c2072c906fab6b637ee25e28944bb048e8ed335cdb92e3024888f1500b36a578.jpg)  
Figure 15: Human response distribution of the five ordinal target variables with highest human response variation.

<table><tr><td>Var-ID</td><td>Question Text</td><td>Type</td><td>HRV</td><td>n</td></tr><tr><td>Nominal</td><td>Low Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>gss_21</td><td>How much time, if any, have you spent on active-duty military service?</td><td>BC</td><td>0.2596</td><td>3292</td></tr><tr><td>gss_212</td><td>What is the first other language you speak besides English (or Spanish)?</td><td>BC</td><td>0.2706</td><td>847</td></tr><tr><td>gss_142</td><td>Do you favor or oppose sex education being taught in public schools?</td><td>AT</td><td>0.3492</td><td>2131</td></tr><tr><td>gss_178</td><td>Do you think Black-white differences in average jobs, income, and housing are because most Black people have less in-born ability to learn?</td><td>AT</td><td>0.3892</td><td>2122</td></tr><tr><td>gss_167</td><td>Do you or your spouse go hunting, and if so, who hunts?</td><td>BC</td><td>0.3988</td><td>2229</td></tr><tr><td>Nominal</td><td>High Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>gss_186</td><td>Have you had a “born again" experience, meaning a turning point when you committed yourself to Christ?</td><td>BC</td><td>0.9695</td><td>3185</td></tr><tr><td>gss_416</td><td>Do you think most people would try to take advantage of you if they could, or would they try to be fair?</td><td>AT</td><td>0.9742</td><td>1221</td></tr><tr><td>gss_427</td><td>Do most people try to be helpful, or are they mostly just looking out for themselves?</td><td>AT</td><td>0.9755</td><td>1232</td></tr><tr><td>gss_179</td><td>Do you think Black-white differences in average jobs, income, and housing are because most Black people lack the educational opportunities needed to rise out of poverty?</td><td>AT</td><td>0.9996</td><td>2099</td></tr><tr><td>gss_177</td><td>Do you think Black-white differences in average jobs, income, and housing are mainly due to discrimination against Black people?</td><td>AT</td><td>0.9998</td><td>2085</td></tr><tr><td>Ordinal</td><td>Low Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>gss_380</td><td>How many sexual partners have you had in the last 12 months?</td><td>BC</td><td>0.1106</td><td>2041</td></tr><tr><td>gss_169</td><td>On a typical day, about how many hours do you personally watch television?</td><td>BC</td><td>0.1523</td><td>2152</td></tr><tr><td>gss_189</td><td>On a scale where 1 means almost all rich and 7 means almost all poor, how would you rate Blacks in general?</td><td>AT</td><td>0.2178</td><td>1068</td></tr><tr><td>gss_190</td><td>On a scale where 1 means almost all rich and 7 means almost all poor, how would you rate Hispanic Americans in general?</td><td>AT</td><td>0.2234</td><td>1061</td></tr><tr><td>gss_188</td><td>On a scale where 1 means almost all rich and 7 means almost all poor, how would you rate whites in general?</td><td>AT</td><td>0.2380</td><td>1067</td></tr><tr><td>Ordinal</td><td>High Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>gss_168</td><td>How often do you read a newspaper?</td><td>BC</td><td>0.6737</td><td>2168</td></tr><tr><td>gss_205</td><td>Do you favor or oppose giving women preference in hiring and promotion because of past discrimination, and how strongly?</td><td>AT</td><td>0.6825</td><td>1058</td></tr><tr><td>gss_441</td><td>Would you describe your religious affiliation as strong, not very strong, somewhat strong,</td><td>AT</td><td>0.7124</td><td>1799</td></tr><tr><td>gss_143</td><td>or having no religion? Should it be easier, more difficult, or about as it is now to obtain a divorce in this country?</td><td>AT</td><td>0.7330</td><td>2080</td></tr><tr><td>gss_147</td><td>How wrong, if at all, do you think it is for two adults of the same sex to have sexual relations?</td><td>AT</td><td>0.8882</td><td>2125</td></tr><tr><td>ggss_24</td><td>In the last three months, have you used the Internet for private purposes with any other devices?</td><td>BC</td><td>0.1720</td><td>4799</td></tr><tr><td>ggss_303</td><td>Are you afraid that in the near future you might become unemployed or have to change your current job?</td><td>AT</td><td>0.3957</td><td>2634</td></tr><tr><td>ggss_23</td><td>In the last three months, have you used the Internet for private purposes with an e-book reader?</td><td>BC</td><td>0.4160</td><td>4799</td></tr><tr><td>ggss_20</td><td>In the last three months, have you used the Internet for private purposes with a smart- phone?</td><td>BC</td><td>0.4295</td><td>4799</td></tr><tr><td>ggss_400</td><td>When you were 15 years old, did you live in the same household with both of your parents?</td><td>BC</td><td>0.4401</td><td>5188</td></tr><tr><td>Nominal</td><td>High Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>ggss_246</td><td>In your view, does everyone in Germany today have the opportunity to get an education that fully matches their talents and abilities?</td><td>AT</td><td>0.9971</td><td>3556</td></tr><tr><td>ggss_203</td><td>If you wanted to influence politics on an issue important to you, would you consider boycotting or deliberately buying certain products for political, ethical, or environmental</td><td>BC</td><td>0.9981</td><td>3417</td></tr><tr><td>ggss_109</td><td>reasons? Do you personally have contact with foreigners living in Germany in your neighborhood?</td><td></td><td>0.9994</td><td></td></tr><tr><td>ggss_174</td><td>Have you ever previously been a member of a church or religious community?</td><td>BC BC</td><td>0.9996</td><td>3591 2448</td></tr><tr><td>ggss_302</td><td>Does your main job include supervising the work of other employees or telling them</td><td>BC</td><td>1.0000</td><td>2965</td></tr><tr><td>Ordinal</td><td>what they have to do? Low Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>ggss_430</td><td>How many adults who belong to the survey's target population currently live in your</td><td>BC</td><td>0.1294</td><td>4988</td></tr><tr><td>ggss_429</td><td>household, including yourself?</td><td></td><td></td><td></td></tr><tr><td>ggss_428</td><td>Including yourself, how many people in total currently live in your household? How many additional people, apart from yourself, currently live in your household?</td><td>BC BC</td><td>0.1306 0.1333</td><td>5144 3963</td></tr><tr><td>ggss_87</td><td>How bad or not bad do you personally consider it if a man forces his wife to have sexual</td><td>AT</td><td>0.1405</td><td>3585</td></tr><tr><td>ggss_472</td><td>intercourse? How many of your own biological, step-, adoptive, or foster children live in your</td><td>BC</td><td>0.1561</td><td>5074</td></tr><tr><td>Ordinal</td><td>household? High Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>ggss_188</td><td>How important was the reason “I left the church because I can no longer do anything</td><td>BC</td><td>0.6505</td><td>1130</td></tr><tr><td>ggss_196</td><td>with faith" for your decision to leave the church? To what extent do you agree with the statement: “I would have no objection to a Muslim</td><td>AT</td><td>0.6615</td><td>3409</td></tr><tr><td>ggss_179</td><td>mayor in my municipality"? How important was the reason “I left the church because I was angry with pastors or</td><td>BC</td><td>0.6866</td><td>1120</td></tr><tr><td>ggss_183</td><td>other church staff’ for your decision to leave the church? How important was the reason “I left the church because I can believe without the church"</td><td>BC</td><td>0.7315</td><td>1129</td></tr><tr><td>ggss_169</td><td>for your decision to leave the church? What is your opinion about religious education in public schools in Germany?</td><td>AT</td><td>0.7982</td><td>5107</td></tr><tr><td>wvs_de_53</td><td>Would you object to having unmarried couples living together as neighbors?</td><td>AT</td><td>0.0571</td><td>1523</td></tr><tr><td>wvs_de_134</td><td>Are you an active member, inactive member, or not a member of a consumer organiza- tion?</td><td>BC</td><td>0.1020</td><td>1524</td></tr><tr><td>wvs_de_47</td><td>Would you object to having people of a different race as neighbors?</td><td>AT</td><td>0.1821</td><td>1523</td></tr><tr><td>wvs_de_51</td><td>Would you object to having people of a different religion as neighbors?</td><td>AT</td><td>0.1855</td><td>1523</td></tr><tr><td>wvs_de_136</td><td>Are you an active member, inactive member, or not a member of a women's group?</td><td>BC</td><td>0.2004</td><td>1523</td></tr><tr><td>Nominal</td><td colspan="4">High Human Response Variation</td></tr><tr><td>wvs_de_182</td><td>If you had to choose, which is more important to you: freedom or security?</td><td>AT</td><td>0.9919</td><td>1476</td></tr><tr><td>wvs_de_171</td><td>Have you avoided carrying much money for reasons of personal security?</td><td>BC</td><td>0.9935</td><td>1514</td></tr><tr><td>wvs_de_88</td><td>Generally speaking, do you think most people can be trusted, or do you think you must be very careful when dealing with people?</td><td>AT</td><td>0.9953</td><td>1482</td></tr><tr><td>wvs_de_183</td><td>If there were a war, would you be willing to fight for your country?</td><td>AT</td><td>0.9975</td><td>1375</td></tr><tr><td>wvs_de_189</td><td>Which of the following is second most important to you: a stable economy, a more humane and less impersonal society, a society where ideas matter more than money, or</td><td>AT</td><td>0.9987</td><td>1492</td></tr><tr><td>Ordinal</td><td colspan="4">the fight against crime? Low Human Response Variation</td></tr><tr><td>wvs_de_86</td><td>In the last 12 months, how often have you or your family gone without a safe place to live?</td><td>BC</td><td>0.0262</td><td>1525</td></tr><tr><td>wvs_de_211</td><td>To what extent is stealing property justifiable, if at all?</td><td>AT</td><td>0.0518</td><td>1528</td></tr><tr><td>wvs_de_221</td><td>To what extent is it justifiable, if at all, for a man to beat his wife?</td><td>AT</td><td>0.0551</td><td>1527</td></tr><tr><td>wvs_de_224</td><td>To what extent is terrorism for political, ideological or religious ends justifiable, if at all?</td><td>AT</td><td>0.0590</td><td>1525</td></tr><tr><td>wvs_de_82</td><td>In the last 12 months, how often have you or your family gone without enough food to</td><td>BC</td><td>0.0961</td><td>1527</td></tr><tr><td>eat?</td><td colspan="4"></td></tr><tr><td>Ordinal wvs_de_239</td><td>High Human Response Variation How often do you use social media to get information about what is happening in the</td><td>BC</td><td>0.8158</td><td>1527</td></tr><tr><td>wvs_de_236</td><td>country and the world? How often do you use your mobile phone to get information about what is happening in</td><td>BC</td><td>0.8370</td><td>1526</td></tr><tr><td>wvs_de_154</td><td>the country and the world? Do you agree or disagree that immigrants help your country by filling useful jobs in the</td><td>AT</td><td>0.8644</td><td>1490</td></tr><tr><td>wvs_de_74</td><td>workforce? If less importance were placed on work in people's lives, would you see this as good,</td><td>AT</td><td>0.8676</td><td>1477</td></tr><tr><td>wvs_de_160</td><td>bad, or would you not mind? Do you agree or disagree that immigrants harm your country by increasing unemploy-</td><td>AT</td><td>0.8770</td><td>1488</td></tr><tr><td>wvs_us_53</td><td>Would you object to having people of a different religion as neighbors?</td><td>AT</td><td>0.1818</td><td>2435</td></tr><tr><td>wvs_us_49</td><td>Would you object to having people of a different race as neighbors?</td><td>AT</td><td>0.2045</td><td>2435</td></tr><tr><td>wvs_us_55</td><td>Would you object to having unmarried couples living together as neighbors?</td><td>AT</td><td>0.2727</td><td>2435</td></tr><tr><td>wvs_us_48</td><td>Would you object to having drug addicts as neighbors?</td><td>AT</td><td>0.2781</td><td>2435</td></tr><tr><td>wvs_us_51</td><td>Would you object to having immigrants or foreign workers as neighbors?</td><td>AT</td><td>0.4294</td><td>2435</td></tr><tr><td>Nominal</td><td>High Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>wvs_us_128</td><td>Are you an active member, inactive member, or not a member of a church or religious organization?</td><td>BC</td><td>0.9843</td><td>2581</td></tr><tr><td>wvs_us_189</td><td>Which of the following national aims is second most important to you: maintaining order, giving people more say in government, fighting rising prices, or protecting</td><td>AT</td><td>0.9902</td><td>2501</td></tr><tr><td>wvs_us_174</td><td>freedom of speech? Have you chosen not to go out at night for reasons of personal security?</td><td>BC</td><td>0.9908</td><td>2576</td></tr><tr><td>wvs_us_38</td><td>Do you consider independence an especially important quality for children to learn</td><td>AT</td><td>0.9918</td><td>2592</td></tr><tr><td>wvs_us_37</td><td>at home? Do you consider good manners an especially important quality for children to learn</td><td>AT</td><td>1.0000</td><td>2592</td></tr><tr><td>at home? Ordinal</td><td>Low Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>wvs_us_31</td><td>How important is family in your life?</td><td>AT</td><td>0.1397</td><td>2593</td></tr><tr><td>wvs_us_88</td><td>In the last 12 months, how often have you or your family gone without a safe place</td><td>BC</td><td>0.1576</td><td>2586</td></tr><tr><td>wvs_us_63</td><td>to live? How much do you agree that when jobs are scarce, men should have more right to a</td><td>AT</td><td>0.1614</td><td>2591</td></tr><tr><td>wvs_us_223</td><td>job than women? To what extent is it justifiable, if at all, for a man to beat his wife?</td><td>AT</td><td>0.1643</td><td>2570</td></tr><tr><td>wvs_us_226</td><td>To what extent is terrorism for political, ideological or religious ends justifiable, if at</td><td>AT</td><td>0.1940</td><td>2570</td></tr><tr><td>all? Ordinal</td><td>High Human Response Variation</td><td></td><td></td><td></td></tr><tr><td>wvs_us_247</td><td>Have you donated to a group or campaign, might you do it, or would you never do</td><td>BC</td><td>0.6601</td><td>2563</td></tr><tr><td>it? wvs_us_89</td><td>Compared with your parents when they were about your age, is your standard of</td><td>AT</td><td>0.6795</td><td>2581</td></tr><tr><td>wvs_us_235</td><td>living better, worse, or about the same? How often do you use daily newspapers to get information about what is happening</td><td>BC</td><td>0.6897</td><td>2566</td></tr><tr><td>wvs_us_241</td><td>in the country and the world? How often do you use social media to get information about what is happening in</td><td>BC</td><td>0.7271</td><td>2562</td></tr><tr><td></td><td>the country and the world?</td><td></td><td></td><td></td></tr><tr><td>wvs_us_239</td><td>How often do you use email to get information about what is happening in the country and the world?</td><td>BC</td><td>0.7345</td><td>2561</td></tr></table>

Table 5: Target variables selected for the GSS. We show the paraphrased survey question texts of the five nominal and ordinal target variables with lowest and highest human response variation scores, respectively. Within the nominal and ordinal variables, variables are ordered in increasing order of their human response variation scores.

Table 6: Target variables selected for the GGSS.

Table 7: Target variables selected for the WVS\_DE.

Table 8: Target variables selected for the WVS\_US.

## 8.3 Model Selection

Table 9 shows the dates of knowledge-cutoff and release for the LLMs used in our experiments. While we can practically rule out the possibility that GSS and GGSS has leaked into any of the used LLMs’ training datasets, this would theoretically and based on the timing alone be possible for the two WVS datasets, both of which have been released 20/07/2020, well before the development of our LLMs. However, the complex storage of the data, distributed across various .csv and .pdf files, the lack of indicators of "valuable text data" contained in these files, as well as the comparability of the predictive performance for the two WVS surveys and the impossibly leaked other two surveys suggest that no actual data leakage into the LLMs has occurred.

<table><tr><td>Model</td><td>K-Cutoff</td><td>R-Date</td><td>Huggingface Model ID</td></tr><tr><td>QWEN2.5-3B-INSTRUCT</td><td>unknown</td><td>27/04/2025</td><td>Qwen/Qwen2.5-3B-Instruct</td></tr><tr><td>QWEN2.5-7B-INSTRUCT</td><td>unknown</td><td>27/04/2025</td><td>Qwen/Qwen2.5-7B-Instruct</td></tr><tr><td>QWEN2.5-VL-32B-INSTRUCT</td><td>unknown</td><td>27/04/2025</td><td>Qwen/Qwen2.5-VL-32B-Instruct</td></tr><tr><td>LLAMA-3.1-8B INSTRUCT</td><td>12/2023</td><td>23/07/2024</td><td>meta-llama/Llama-3.1-8B-Instruct</td></tr><tr><td>LLAMA-3.2-3B-INSTRUCT</td><td>12/2023</td><td>25/09/2024</td><td>meta-llama/Llama-3.2-3B-Instruct</td></tr><tr><td>LLAMA-3.3-70B-INSTRUCT</td><td>12/2023</td><td>06/12/2024</td><td>meta-llama/Llama-3.3-70B-Instruct</td></tr></table>

Table 9: Release details of LLMs used in our experiments. GSS (22/05/2025) and GGSS (10/01/2025) were released after the known knowledge cutoff (K-Cutoff) and even the release of the Llama-models, and less than two months before the release of the Qwen-models, making data leakage practically impossible.

## 9 Appendix: Results

## 9.1 Disaggregation based on Model Family

As discussed in the main text, we find that persona-prompted LLMs are consistently better at predicting responses to target questions with high human response variation, and that, in contrast, the no persona baseline performs better for predicting target questions with low rather than high human response variation. Figure 16 shows that this effect also holds when disaggregating based on the two model families —both Llama and Qwen models perform better in predicting responses to target questions with high human response variation than to target questions with low human response variation, and both model families show the opposite order of performance when not being conditioned on personas.

In addition, Figure 17 shows that the same effect mostly holds when disaggregating the results for the different surveys, with some exceptions. For example, the Llama models are better in predicting high response variation target questions than low response variation target questions for the GGSS, reversing the general finding. However, this seems to be survey- rather than country-specific, as the general finding still holds strongly for the German WVS data. For Qwen models, some of the attribute selection approaches (HRV, Set Selection, and Scoring) perform better for low human response variation questions than high human response variation questions and thus running counter to the general finding, which again holds for the data from the other German survey, WVS\_DE.

However, the strength of the effect, i.e., the distance between the mean lines for low and high response variation questions, is generally smaller for the surveys from Germany (GGSS and WVS\_DE) than for those from the United

States (GSS and WVS\_US). Particularly when using the statistical baselines for selecting persona attributes, the ability of the LLMs to reproduce human responses for target questions with low human response variation is much worse than for target questions with high human response variation. Here, the attributes selected based on the survey data seem to move the LLMs away from the "correct" baseline response, compared to attributes selected through other methods.

## 9.2 Disaggregation based on Survey

We already discuss in the main text that the statistical baselines for selecting attributes perform better than the LLMbased selection approaches. Figure 16 shows that the same performance order of persona attribute selection methods persists when disaggregating the results based on model families, with performance differences being particularly large for high response variation target questions. Figure 17 shows that the performance order of attribute selection approaches is also consistent across the different surveys, with differences between attribute selection approaches generally being less pronounced for the US-based surveys (GSS and WVS\_US).

![](images/673bb8db0b3a89ebc5ea7f5789ba9556895cef1306cc6bc01aee2b7080e02630.jpg)  
(a) Qwen Models

![](images/2bed84e51c67dc9784e0101d1713870de3f7a17f5e56a372813400ba4d4df7d7.jpg)  
(b) Llama Models  
Figure 16: Results from Figure 3, disaggregated by model family. The effects of human response variation on the performance as well as the general ordering of performance across attribute selection approaches are the same as reported for the aggregate results, implying that they are consistent across models from different families.

![](images/945da8b27292d216729a57620986a686b4a30da29aa2d94024eb6ff2efb4d937.jpg)  
(a) Qwen Models

![](images/20b9a2d6373c317d95a14293be6a7681dfaff7df45e409ca54e58b52e1168f15.jpg)  
(b) Llama Models  
Figure 17: Results from Figure 3, disaggregated by model family and survey. With some exceptions, e.g., the reversal of the effect of human response variation in the no persona baseline for GGSS data in Llama models and for some attribute selection methods in Qwen models, the general findings reported for the aggregated results hold true also across surveys.

## 9.3 Stability of LLM-based Attribute Selection Approaches

## 9.3.1 Stability of the Set Selection Approach

Evaluation Approach For the LLM-based set selection approach, we count how often each survey variable is selected among the five most important attributes for predicting a given target question across the 100 (30) independent runs. We order these selection counts in decreasing order and plot them per LLM for all 20 target questions across the four surveys, with the x-axis corresponding to the variables in a survey and the y-axis to the selection counts. For a perfectly stable selection method, we would expect every run to select the same set of five attributes as most important for each target question, and for a selection method that is random guessing, we would expect every variable to be selected equally often. We can therefore visually assess and compare the stability of the set selection approach across different LLMs.

Results As discussed in the main text, the stability of the set selection method increases with model size. For the smallest models in each family, the same attribute is on average selected as most important in only 15.1% (Qwen2.5-3B) and 14.8% (Llama3.2-3B) of runs conducted for each target questions, indicating a large variety of attributes considered as important across the runs.

For the mid-sized models, the stability increases substantially—Qwen2.5-7B now selects the same attribute as most important in 32.4% and Llama3.1-8B in 21.6% of the runs per target questions. Additionally, we see that for these models some target questions achieve very high consistency across runs, with the most important attribute for some few target questions being the same in nearly every run.

For Llama3.3-70B we see high stability of the attribute selection across runs. Not only does it consistently select the same attribute as most important in on average 57.3% of runs across target questions, but it even selects all five attributes consistently for a number of target questions. For Qwen2.5-32B, the stability of the set selection is similar to that of the mid-sized version of the model, with the most important attribute being selecting in roughly a third of the runs across target questions.

![](images/9702138c8a9fb94225884d142cf075cd2312396b7ff3f45f1d42065b66b4eb98.jpg)  
(a) Llama-3.2-3B-Instruct

![](images/a242338d1966b3985e55f84ebf092a4a989ccab0030f2683f9606ec32a6b22bc.jpg)  
(b) Qwen2.5-3B-Instruct

![](images/1c0ebacd64a3e25f7cf13565ef484e42a70500956a567652d6149aeda7b19ce2.jpg)  
(c) Llama-3.1-8B-Instruct

![](images/e2863be18b96c2830495b519461dc9e551a155d1d373ab9f0a764a79692b2cbe.jpg)  
(d) Qwen2.5-7B-Instruct

![](images/44094ea6b83598e161ab1e7fac9772b54a0c314979fa0a0a722a8b60b0a3afc9.jpg)  
(e) Llama-3.3-70B-Instruct

![](images/363edc55e408887e97020e0c39471ad944c2c9ee4b735ce7ede9bca5da0d5a3b.jpg)  
(f) Qwen2.5-VL-32B-Instruct

Figure 18: Results from Figure 4, disaggregated by model. The increased stability of models with higher parameter counts clearly shows, and particularly the ability of larger models to select the same attribute as most important for some target questions in more than 80% of runs.

## 9.3.2 Stability of the Scoring Approach

Evaluation Approach For the LLM-based scoring approach, we create a ranking of the variable importance scores for each of the independent 100 (30) runs. From these rankings, we then construct a distribution of concordance coefficients for each LLM scoring the 20 target questions across the four surveys. We compare the means and variances of these distributions to get a sense of the stability of the scoring approach across different LLMs. As we have to account for the possibility of predictors with the same score and thus sharing the same rank, we calculate the tie-corrected Kendall’s W for each of the 100 (30) runs.

Results For models from the Llama family, Figure 19 shows that the concordance when scoring the importance of survey items from the GSS is much higher compared to all other surveys. This difference can potentially be explained through the fact that the pool of predictors and thus the number of possible rank variations is smallest for the GSS. This makes it theoretically easier to achieve higher Kendall’s W values. Surprisingly, the largest Llama model seems to be the most erratic model across different surveys, with scoring predictors very consistently across multiple runs for the GSS and showing very high variation but generally low consistency for GGSS and WVS\_DE.

For models from the Qwen family, Figure 20 shows that there are barely any survey-specific differences between the surveys GGSS and WVS (USA & GER). However, the concordance for all models in the Qwen family when scoring items from the General Social Survey is much higher compared to all other surveys. This could again be due to the fact that GSS has the lowest number of survey variables and thus the lowest number of possible rankings. Differences between models sizes are less pronounced for the Qwen models than for the Llama models.

![](images/3d2d709a810118ffee6edd7b985c9466422e13510b2b05a3b5a69657bc3edbc0.jpg)

![](images/814da726b41e9e6140a9195eb2e2d85c1c15130d333af9bffade061a23a92285.jpg)

![](images/55a501124241b1fe17361e3ec4301bd329384fd5eb32fad78777a1c56eef12a4.jpg)  
(a) Llama-3.2-3B-Instruct

![](images/8dba482c242f2daedf004dae40107aea44426c7927d8910753c808d7acf847c5.jpg)

![](images/5c2e3f8e377604dd706d48bb3eed812180be735f7cb9b42c7de7c8cdf9cf5db2.jpg)

![](images/df03dfdff6962ccebd30e964bae865071c034ec0455055553d5504ed8286ec5e.jpg)

![](images/932b8e36e428a8587d562f41fc2c5c6abadd6d3aef4d0bd2b8ea0214ecd03451.jpg)  
(b) Llama-3.1-8B-Instruct

![](images/6bdc48b2c2dc010b9ae100f12ed43fd7788c1cfd5072ea69f43600d1bcc3cde2.jpg)

![](images/03cd8a8827051f414c1587abdadce31a6e6f2098d073d7a8d8a24aed8e17f191.jpg)

![](images/14689206448ab6bc0317230227970c4c5b9f3733fb2aaf918f74e17438fe3e80.jpg)

![](images/7d5b2c7b228099c659e250e872b8a5716921fb4ebd9f544dc6cf5078aee33c90.jpg)  
(c) Llama-3.3-70B-Instruct

![](images/12e58b7f795d5a726dd652f0e83303188597327d5e1e2e8858c9fa8770380f1a.jpg)  
Figure 19: Results from Figure 5, disaggregated by model family (Llama).

![](images/0c3b4fef7c1aac345640d4544602cb35b5834bc3bec1bedb3f0659aefaa26ed1.jpg)

![](images/130a8a6b1d3ccc945225b936c02bef201d19c97b5a02a72b09c7373b160d9b53.jpg)

(a) Qwen2.5-3B-Instruct  
(b) Qwen2.5-7B-Instruct  
![](images/5e24891d27dcea73d0fbc73ffd6aebb5c8a3269178cc464e9463b5fe9031d45c.jpg)  
(c) Qwen2.5-VL-32B-Instruct  
Figure 20: Results from Figure 5, disaggregated by model family (Qwen).