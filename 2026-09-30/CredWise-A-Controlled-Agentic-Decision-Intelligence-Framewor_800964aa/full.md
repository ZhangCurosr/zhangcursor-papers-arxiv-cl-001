# CredWise: A Controlled Agentic Decision-Intelligence Framework for Explainable and Auditable Credit-Risk Assessment

Aakash Kumar Tiwari<sup>a,∗</sup>

<sup>a</sup>Department of Mathematics, Indian Institute of Technology Kharagpur, Kharagpur 721302, West Bengal, India

## Abstract

Credit-risk prediction is important in banking, but a prediction alone does not explain why an applicant is risky or how it should be combined with other evidence. This paper presents CredWise, a decision-support framework that integrates credit-risk prediction, probability calibration, explainable artificial intelligence, policy retrieval, SQL analytics, and controlled agent-based workflows. An XGBoost model is trained on Lending Club data (1,345,310 loans, 18 features) using a temporal split: 2007–2016 for training, 2017 for validation, and 2018 for testing. On the 2018 test set, the calibrated model achieved a ROC-AUC of 0.7109, PR-AUC of 0.2993, F1-score of 0.3714, and accuracy of 65.44%. Calibration reduced the Brier score from 0.2157 to 0.1273 and the expected calibration error from 0.2862 to 0.0585. SHAP explanations were temporally stable, with a Spearman correlation of 0.9959 between 2017 and 2018 feature rankings. On 28 labeled queries covering nine policy sections, FAISS achieved the best Hit@1 (0.929) and MRR (0.964), while all three retrieval methods reached Hit@5 = 1.0. Agent routing achieved 95.6% accuracy (43 of 45 cases), and the SQL benchmark scored 1.0 on exact-match, execution-success, and result-match across six cases. These results show that CredWise can combine predictions, explanations, policy evidence, and structured analytics in one

controlled workflow. It is an academic research prototype, and final decisions remain with a human reviewer.

Keywords: credit risk, explainable artificial intelligence, probability calibration, retrieval-augmented generation, agentic artificial intelligence, decision intelligence, human-in-the-loop

## 1. Introduction

Credit risk assessment is an important task in the financial sector. Banks and lending institutions need to estimate whether a borrower is likely to repay a loan. Traditional credit scoring has mainly used statistical classification methods, while machine learning has been adopted to capture more complex patterns in credit data (Hand & Henley, 1997; Lessmann et al., 2015). However, the use of machine learning in credit risk also raises practical issues, including how to interpret model predictions, how reliable the predicted probabilities are, and how the predictions can be used in an actual decision-support process (Bussmann et al., 2021; Lessmann et al., 2024). A credit-risk model usually produces a probability of default, but this probability alone may not be enough for a lending decision. A reviewer may also want to understand why an applicant was given a high-risk score, which features afected the prediction, and whether the result is consistent with the lending policy. This is particularly relevant for models such as gradient boosting, which can learn complex relationships that are not easy to interpret directly (Chen & Guestrin, 2016; Friedman, 2001). Explainable artificial intelligence (XAI) methods can provide additional information about these predictions (Arrieta et al., 2020), and explainability is now an important part of reviewing machine-learning-based credit decisions (Bussmann et al., 2021; B¨ucker et al., 2022). Another issue is whether the predicted probabilities are reliable. A model can distinguish between higher-risk and lower-risk applicants reasonably well, while the predicted probabilities may still difer from the actual outcome frequencies. Probability calibration is used to address this issue (Niculescu-Mizil & Caruana, 2005; Platt, 1999; Guo et al., 2017), and the

Brier score is commonly used to measure the quality of probabilistic predictions (Brier, 1950). Therefore, a credit-risk decision-support system should consider both discrimination and probability quality instead of relying only on classifi cation performance. The way a model is evaluated is also important because borrower characteristics, lending practices, and economic conditions can change over time. This is related to concept drift, where the relationship between the input data and the target variable changes over time (Gama et al., 2014). A random train-test split may not capture such changes. Using earlier data for training and later data for testing can provide a more realistic view of how the model behaves on a later period. A practical decision-support system may also require information outside the prediction model, such as lending policy documents. Retrieval-Augmented Generation (RAG) combines information retrieval with language-model based processing (Lewis et al., 2020). Dense retrieval and sentence embeddings can be used to represent text for retrieval (Karpukhin et al., 2020; Reimers & Gurevych, 2019), while libraries such as FAISS sup port eficient similarity search (Johnson et al., 2021). In a credit-risk setting, this can be used to retrieve relevant policy evidence and present it together with a model prediction. Recent agent-based systems also allow language models to use tools and perform multi-step tasks (Yao et al., 2023; Wang et al., 2023). However, a financial decision-support system needs controlled routing of tasks, restricted access to data and tools, and a clear distinction between model evidence, retrieved policy information, and generated text. Human oversight is also important because automated systems should assist users rather than remove human responsibility from important decisions (Amershi et al., 2019). Earlier work studied multi-agent and retrieval-based workflows for enterprise customer support (Tiwari & Kumar, 2026). The present work applies related ideas to credit-risk decision support, where predictions, policy evidence, explanations, and structured analytics need to be brought together. Based on these requirements, this paper presents CredWise, an academic research prototype for controlled credit-risk decision support. The system uses an XGBoost classifier and calibrates its output probabilities before they are used. It uses

SHAP for feature-level explanations of individual predictions (Lundberg & Lee, 2017), a policy retrieval component over a lending-policy document, a SQL analytics component for structured portfolio analysis, and a controlled agent workflow to connect these components. The system is evaluated on a large Lending Club dataset using a temporal split, with 2007–2016 used for training, 2017 for validation, and 2018 kept as an untouched test period. The final processed dataset contains 1,345,310 loan records and 18 prediction features. The evaluation covers classification performance, probability calibration, temporal stability of explanations, feature drift, error and subgroup analysis, policy retrieval, SQL analytics, and agent routing. Bootstrap confidence intervals are also reported for the final test-set metrics (Efron & Tibshirani, 1986). CredWise is not designed to replace a human lending decision. Instead, it brings diferent types of evidence into one decision-support workflow. The model prediction, explanation, retrieved policy evidence, and supporting analytics remain separate and traceable parts of the process, while the final decision is left to a human reviewer. The main contributions of this work are as follows:

• We develop a controlled credit-risk decision-support framework that combines prediction, probability calibration, explainability, policy retrieval, SQL analytics, and agent-based workflow control.

• We evaluate the credit-risk model using a temporal train-validation-test setup (2007–2016 / 2017 / 2018).

• We study the efect of probability calibration and report both discrimination and calibration metrics for the final test period.

• We evaluate the temporal stability of SHAP-based feature importance and examine feature drift over time.

• We evaluate the retrieval, SQL, and agent-routing components using separate controlled test cases instead of treating the complete system as a single black box.

• We keep the system human-in-the-loop, with model and retrieval outputs provided as decision-support evidence and the final decision left to the reviewer.

The remainder of this paper is organized as follows. Section 2 discusses related work. Section 3 presents the research questions. Section 4 describes the dataset and experimental setup. Section 5 presents the CredWise framework. Section 6 reports the experimental results. Section 7 discusses the findings. Section 8 describes the limitations. Section 9 concludes the paper.

## 2. Related Work

The work related to CredWise covers four main areas: credit-risk modelling, explainability and probability calibration, retrieval and agent-based systems, and human-in-the-loop decision support.

## 2.1. Credit-Risk Modelling

Credit scoring has traditionally used statistical classification methods, as discussed by Hand and Henley (Hand & Henley, 1997). With the use of machine learning, many classification algorithms have also been applied to credit scoring. Lessmann et al. (Lessmann et al., 2015) compared a large number of these algorithms and showed that machine learning can provide useful predictive performance for credit scoring. More recent research has also discussed practical issues related to the use of machine learning for credit risk in financial applications (Lessmann et al., 2024). Tree-based boosting methods are well suited to structured tabular data. XGBoost (Chen & Guestrin, 2016) provides an eficient implementation of gradient tree boosting (Friedman, 2001) and is widely used for classification problems. These studies mainly focus on prediction performance. A decision-support system, however, may also need to show how reliable a predicted probability is, why a particular prediction was made, and what additional evidence can support the decision. CredWise therefore uses the prediction model as one part of a larger decision-support workflow rather than treating the model as the complete system.

## 2.2. Explainability and Probability Calibration

Interpretability is an important issue when machine learning is used for credit-risk assessment. Bussmann et al. (Bussmann et al., 2021) studied explainable machine learning in credit risk management and discussed interpretability in financial lending using models, visualizations, and summary explanations. Arrieta et al. (Arrieta et al., 2020) provided a broader review of the opportunities and challenges of explainable AI. SHAP (Lundberg & Lee, 2017) is a widely used method for explaining individual predictions through feature contributions. In CredWise, SHAP is used to identify features that increase or decrease the model output for an applicant and to examine global feature importance. The reliability of predicted probabilities is another important issue. A classifier may rank applicants well even when its predicted probabilities are not well calibrated (Niculescu-Mizil & Caruana, 2005; Platt, 1999; Guo et al., 2017). The Brier score provides a direct measure of the quality of probabilistic predictions (Brier, 1950). CredWise combines probability calibration with SHAP-based explanations so that the system provides both a calibrated risk probability and information about the features associated with the prediction.

## 2.3. Retrieval-Augmented Systems for Decision Support

A model prediction may not provide all the information needed to make a decision. For example, a reviewer may also need to check a lending policy document. Retrieval-Augmented Generation (RAG) combines information retrieval with language-model based generation (Lewis et al., 2020). Dense retrieval and sentence embeddings can be used to represent text for retrieval (Karpukhin et al., 2020; Reimers & Gurevych, 2019), while FAISS can be used for eficient similarity search (Johnson et al., 2021). Diferent retrieval methods can also be combined using approaches such as Reciprocal Rank Fusion (Cormack et al., 2009). This can be useful when policy documents contain both exact terms and semantically related expressions. Earlier work used FAISS- and BM25-based retrieval together with multiple agents and local language models for enterprise customer support (Tiwari & Kumar, 2026). In the present study, retrieval is used for a more specific purpose: retrieving lending-policy evidence and presenting it together with the model prediction and other decision evidence.

## 2.4. Agent-Based Systems and Human Oversight

Language-model based agents have been studied for tasks involving reasoning, tool use, and multi-step interaction. Examples include ReAct (Yao et al., 2023), surveys of autonomous LLM agents (Wang et al., 2023), and AgentBench (Liu et al., 2023). In financial decision support, however, additional controls are needed. An agent should not act as an unrestricted decision maker. The system should control which tools the agent can use, what data it can access, and how its output is presented to the human reviewer. This is consistent with guidelines for human-AI interaction (Amershi et al., 2019). An earlier study on local LLM evaluation found that consistency between automated judgments does not necessarily mean agreement with human ratings (Tiwari, 2026). This motivates the use of separate and controlled evaluations for the agent outputs in CredWise rather than assuming that consistent agent behaviour is suficient evidence of reliability.

## 2.5. Research Gap

Existing research provides important foundations for the individual parts of a credit-risk decision system, including prediction (Hand & Henley, 1997; Lessmann et al., 2015), explainability (Bussmann et al., 2021; Lundberg & Lee, 2017), calibration (Niculescu-Mizil & Caruana, 2005; Guo et al., 2017), retrieval (Lewis et al., 2020), and agent coordination (Yao et al., 2023; Wang et al., 2023). However, these components are often studied separately. There is therefore a need to examine how they can be connected in a controlled credit-risk decisionsupport workflow while keeping the source of each output clear. CredWise addresses this gap by bringing these components together in one academic research prototype. The language model or agent layer is not used as the final decision maker. Instead, predictions, explanations, retrieved policy evidence, and structured analytics are kept as separate sources of evidence and combined for review by a human decision maker.

## 3. Research Questions

The main aim of this study is to evaluate whether a controlled combination of credit-risk prediction, probability calibration, explainability, policy retrieval, structured analytics, and agent-based workflows can provide useful support to a human reviewer during credit-risk analysis. The evaluation is organized around the following research questions.

## 3.1. RQ1: Credit-Risk Prediction Performance

RQ1: How well does the final XGBoost model perform on a temporally separated test period? This question evaluates the final model on the 2018 test period using accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and Brier score. These metrics cover both classification performance and the quality of the predicted probabilities.

## 3.2. RQ2: Feature Set Comparison

Does adding interest-rate and related loan features improve prediction performance? Feature Set A is compared with Feature Set B in a controlled ablation experiment using the same modelling pipeline. Feature Set B adds int rate, installment, and sub grade. The comparison focuses on changes in ROC-AUC, PR-AUC, precision, recall, F1-score, and Brier score. The final model is then evaluated separately using the temporal train-validation-test setup described in Section 4.

## 3.3. RQ3: Probability Calibration

RQ3: Does probability calibration improve the reliability of the model’s predicted default probabilities? The raw probabilities produced by XGBoost are compared with sigmoid-calibrated probabilities using the Brier score and expected calibration error (ECE). Ranking metrics are also retained to check whether calibration changes the model’s ranking behaviour.

## 3.4. RQ4: Temporal Stability of Model Explanations

RQ4: Are the feature-importance patterns from SHAP explanations stable across the 2017 and 2018 periods? This question examines whether the main features contributing to the model predictions remain similar across the two periods. The analysis uses the Spearman correlation between global feature-importance rankings and the overlap among the top-ranked features.

## 3.5. RQ5: Feature Distribution Drift

RQ5: Which input features show the largest distribution changes between the training period and later evaluation periods? The Population Stability Index (PSI) is used to compare feature distributions between the 2007–2016 training period and the 2017 and 2018 periods. This identifies features whose distributions changed over time and provides additional context for interpreting temporal model performance.

## 3.6. RQ6: Controlled System Component Evaluation

RQ6: Can the policy retrieval, SQL analytics, and agent-routing components produce the expected outputs under controlled evaluation cases? The policy retrieval component is evaluated on 28 labeled queries using Hit@1, Hit@2, Hit@3, Hit@5, and MRR, for BM25, FAISS, and a reciprocal-rank-fusion hybrid. The SQL analytics component is evaluated on six benchmark cases covering query generation, execution, and result matching. The agent-routing component is evaluated on 45 predefined cases covering SQL, policy, risk, and decision-intelligence paths.

## 3.7. RQ7: Integrated Decision-Support Workflow

RQ7: Can the diferent evidence sources be combined into a single human-reviewable decision-support output? This question examines whether the model prediction, calibrated probability, SHAP evidence, policy evidence, and structured analytics can be combined into a controlled decisionintelligence output. The evaluation uses one representative applicant from the 2018 test period and checks the presence and validity of these components rather than using generated text itself as a measure of prediction accuracy.

## 4. Dataset and Experimental Setup

## 4.1. Dataset

The experiments use the Lending Club loan dataset, covering loans issued between 2007 and 2018, with information on loan characteristics, borrower characteristics, and credit history. The target variable is loan status: loans marked Fully Paid are treated as non-default cases and Charged Of loans as default cases; other status categories are excluded from the final binary dataset. After cleaning and preprocessing, the final dataset contains 1,345,310 loan records and 20 columns, including the target variable, with 18 features used for prediction. Features that could reveal the outcome after loan origination were removed; in particular, last pymnt amnt was excluded since it reflects repayment activity that occurs after the loan is issued. The target distribution contains 1,076,751 non-default cases and 268,559 default cases, so the dataset is imbalanced, with about 20% of observations in the default class.

![](images/d1ad004d5e4d917829eb3f9265cc2c8a33ff0b9cb4cc7f07cb0f885ff9274f1b.jpg)  
Figure 1: Class distribution in the processed Lending Club dataset.

Figure 1 shows the class distribution; this imbalance was addressed during training using a positive-class weight.

## 4.2. Temporal Data Split

A temporal split, rather than a random train-test split, was used so the model could be evaluated on a later period than it was trained on: 2007–2016 for training, 2017 for validation, and 2018 as the final test period. The 2018 data was not used for model fitting, hyperparameter selection, or calibration, so the test set represents a genuinely later time period than the training data. Table 1 summarizes the split.

Table 1: Temporal split of the processed dataset.
<table><tr><td>Period</td><td>Purpose</td><td>Samples</td><td>Default Rate</td></tr><tr><td>2007-2016</td><td>Training</td><td>1,119,699</td><td>19.70%</td></tr><tr><td>2017</td><td>Validation</td><td>169,300</td><td>23.12%</td></tr><tr><td>2018</td><td>Test</td><td>56,311</td><td>15.75%</td></tr><tr><td>Total</td><td></td><td>1,345,310</td><td></td></tr></table>

The default rate varies across periods, which is one reason for using a timebased evaluation. Figure 2 shows the yearly charge-of rate.

![](images/edd74dc6511bbc8fc59feeb4063b16043d691241984059a119bfa4aff45b8a2a.jpg)  
Figure 2: Year-wise charge-of rate in the Lending Club dataset.

## 4.3. Prediction Features

Two feature sets were used in the study. Feature Set A contains 15 basic borrower, loan, and credit-history features. Feature Set B adds int rate, installment, and sub grade, giving 18 features in total. The two sets are compared in the feature ablation experiment. The final model uses Feature Set B.

Table 2: Features used by the final CredWise model.
<table><tr><td>Feature</td><td>Meaning</td></tr><tr><td>loan_amnt</td><td>Loan amount</td></tr><tr><td>term</td><td>Loan term</td></tr><tr><td>int_rate</td><td>Interest rate</td></tr><tr><td>installment</td><td>Monthly payment</td></tr><tr><td>sub_grade</td><td>Loan sub-grade</td></tr><tr><td>emp_length</td><td>Employment length</td></tr><tr><td>home_ownership</td><td>Home ownership</td></tr><tr><td>annual_inc</td><td>Annual income</td></tr><tr><td>verification_status</td><td>Income verification</td></tr><tr><td>purpose</td><td>Loan purpose</td></tr><tr><td>dti</td><td>Debt-to-income ratio</td></tr><tr><td>delinq_2yrs</td><td>Recent delinquencies</td></tr><tr><td>inq_last_6mths</td><td>Recent credit inquiries</td></tr><tr><td>open_acc</td><td>Open credit accounts</td></tr><tr><td>pub_rec</td><td>Public records</td></tr><tr><td>revol_bal</td><td>Revolving balance</td></tr><tr><td>revol_util</td><td>Credit utilization</td></tr><tr><td>total_acc</td><td>Total credit accounts</td></tr></table>

The target variable loan status is not used as an input feature. Similarly, issue d and issue month are used only to create the temporal train-validationtest split.

## 4.4. Data Preprocessing

Numerical features were converted to numeric representations and categorical features were encoded, with all preprocessing steps fitted on the training data and then applied to the validation and test periods. The same preprocessing pipeline is stored with the final model so the evaluation transformation can be reproduced at inference time.

## 4.5. XGBoost Model

XGBoost was chosen for its structured numerical and categorical inputs, using gradient-boosted decision trees to produce a default probability. The model was trained on 2007–2016 data, validated on 2017 with early stopping, and used scale pos weight to account for class imbalance. The best validation iteration was 187, with a validation log-loss of 0.633752.

## 4.6. Probability Calibration

The raw XGBoost probability was calibrated before use in the decisionsupport layer, using sigmoid calibration for its simplicity and stability as a postprocessing step. Calibration was fitted without using the 2018 test outcomes; the calibrated probability is used by the risk engine, while the raw output is retained for comparison. The final decision threshold is 0.23: an applicant is classified as predicted default when the calibrated probability meets or exceeds this value.

## 4.7. Evaluation Metrics

Classification performance is evaluated using accuracy, precision, recall, and F1-score, with ROC-AUC measuring ranking performance and PR-AUC providing an additional view for the imbalanced default class. Probability quality is evaluated using the Brier score and Expected Calibration Error (ECE). For the final 2018 results, bootstrap resampling is used to estimate 95% confidence intervals, reflecting sampling variability in the test-set metrics.

## 4.8. Experimental Reproducibility

The final model, preprocessing pipeline, probability calibrator, SHAP explainer, and classification threshold are stored as separate artifacts and reloaded during evaluation to verify reproducible outputs. The full evaluation is performed without modifying the 2018 test set, with prediction, calibration, explanation, retrieval, SQL, and agent evaluations treated as independent experiments so each component can be examined separately.

## 5. CredWise Framework

CredWise is an academic decision-support prototype for credit-risk analysis that combines a machine learning prediction model with probability calibration, explainability, policy retrieval, SQL-based analytics, and controlled agent workflows. The main design goal is to keep these components separate while letting them work together: the prediction model provides the risk probability, the explainability component provides model evidence, the policy component provides relevant policy information, and the SQL component provides structured analytical information, all combined in a final decision-support layer for human review.

## 5.1. System Architecture

The overall architecture, shown in Figure 3, has five main stages: input and preprocessing, risk prediction, evidence generation, agent-based coordination, and human review.

![](images/926112b37c45683e7070c95e91b707bc101da8970fa217ef76dead0302cfc618.jpg)  
Figure 3: Overall architecture of the CredWise decision-support framework.

Applicant information passes through the same preprocessing pipeline used during training and is fed to the XGBoost model, which produces a raw default probability. This is calibrated to obtain the probability used by the risk engine, which is then compared against the classification threshold to determine the predicted class and assign a risk level; the prediction itself is not treated as a final lending decision. In parallel, the SHAP component explains the model prediction, the policy retrieval component retrieves relevant lending-policy sections, and the SQL analytics component provides structured information from the available data. These outputs are passed to the decision-intelligence layer, whose final output contains the model prediction, model evidence, policy evidence, and analytical context, with the human reviewer remaining responsible for the final decision.

## 5.1.1. Research Dashboard

The CredWise prototype provides a Streamlit-based research dashboard that brings the main outputs of the framework into a single review interface, as shown in Figure 4. The dashboard presents the final temporal test results together with calibration, explainability, model robustness, feature drift, error and subgroup analysis, agentic intelligence, structured analytics, and audit information.

![](images/cf4cb847ddf2eeda698ad0f9809851249748b9a653360a9db145ad9307007642.jpg)  
Figure 4: Streamlit-based CredWise research dashboard integrating risk prediction, explainability, robustness, feature drift, error analysis, agentic intelligence, analytics, and audit information.

The dashboard is intended as a research and inspection interface rather than as a replacement for the underlying evaluation procedures. The model prediction, SHAP explanations, policy evidence, analytical results, and evaluation outputs remain separate, while the dashboard allows them to be inspected within a common workflow. It also displays the temporal experimental protocol, including the 2007–2016 training period, 2017 validation period, and 2018 untouched test period, along with the main final-test metrics. The interface is therefore used to make the experimental outputs easier to inspect and compare during research evaluation. It does not change the underlying model prediction or replace the human review process.

## 5.2. Risk Prediction and Calibration

The risk prediction layer uses the final XGBoost model described in Section 4. It receives the 18 selected features x and produces a raw probability:

$$
p _ { \mathrm { r a w } } = f _ { \mathrm { X G B } } ( \mathbf { x } ) ,\tag{1}
$$

where $f _ { \mathrm { X G B } }$ is the trained model. The raw probability is not used directly and is instead passed to the calibration layer. The calibration layer converts the raw probability into a calibrated probability using a fitted sigmoid function g(·):

$$
p _ { \mathrm { c a l } } = g ( p _ { \mathrm { r a w } } ) .\tag{2}
$$

Calibration is evaluated separately from the classification model so that changes in probability quality can be measured without afecting the underlying ranking. With a classification threshold of 0.23,

$$
\hat { y } = \left\{ \begin{array} { l l } { 1 , } & { p _ { \mathrm { c a l } } \geq 0 . 2 3 , } \\ { 0 , } & { p _ { \mathrm { c a l } } < 0 . 2 3 , } \end{array} \right.\tag{3}
$$

where $\hat { y } = 1$ denotes predicted default and $\hat { y } = 0$ predicted non-default. The system also assigns a risk level from the calibrated probability:

$$
\mathrm { R i s k ~ L e v e l } = \left\{ \begin{array} { l l } { \mathrm { L o w ~ R i s k , } } & { p _ { \mathrm { c a l } } < 0 . 1 5 , } \\ { \mathrm { M e d i u m ~ R i s k , } } & { 0 . 1 5 \leq p _ { \mathrm { c a l } } < 0 . 3 0 , } \\ { \mathrm { H i g h ~ R i s k , } } & { p _ { \mathrm { c a l } } \geq 0 . 3 0 . } \end{array} \right.\tag{4}
$$

These ranges organize the decision-support output only and are not presented as regulatory or industry-standard categories.

## 5.3. Explainability and Evidence Generation

The explainability layer uses SHAP to describe each feature’s contribution to a model output (Lundberg & Lee, 2017). For an applicant,

$$
f ( \mathbf { x } ) = \phi _ { 0 } + \sum _ { i = 1 } ^ { M } \phi _ { i } ,\tag{5}
$$

where $\phi _ { 0 }$ is the base value and $\phi _ { i }$ is feature i’s contribution. These values identify features that raise or lower the model output and, when aggregated across the test set, support global analysis. Explanations are treated as model evidence, not as independent causal claims about borrower behaviour. The explanation component therefore provides both individual-level evidence and global feature-importance information. The individual contributions are used to identify risk-raising and risk-reducing features for a given applicant, while aggregated SHAP values are used to study feature importance and temporal stability.

## 5.4. Policy Retrieval and SQL Analytics

The policy document is divided into sections, each represented as a vector using a sentence embedding model and indexed in FAISS for similarity search (Reimers & Gurevych, 2019; Johnson et al., 2021). For a user query, the retrieval component searches the index and returns relevant sections with metadata, such as section and source, preserving a link between retrieved evidence and its source. This layer is intentionally separate from the prediction model, so a retrieved policy statement does not change the XGBoost probability but instead provides additional evidence for the reviewer. The SQL analytics layer supports structured analytical questions, such as portfolio-level counts or average predicted risk, through controlled, read-only query execution. Only approved tables and operations are allowed, while destructive operations, unauthorized tables, and multi-statement queries are blocked, reducing the risk that a language model or user query can modify the underlying database. SQL output is treated as analytical evidence and is kept separate from the model prediction. Together, the policy retrieval and SQL layers provide two additional sources of evidence: policy information from the indexed lending-policy document and structured analytical information from the available data. Neither component changes the underlying model prediction.

## 5.5. Controlled Agent Workflow and Decision Intelligence

A supervisor/routing layer identifies the type of each request and directs it to the appropriate component—for example, a policy question to the policy component, a portfolio-level numerical question to the SQL component, or a “why” question to the risk and explainability components:

User Query → Supervisor/Router → Specialized Component. (6)

Using specialized components rather than one general-purpose agent makes the source of each part of the final response easier to identify. This workflow draws on prior work on reasoning and tool-use agents (Yao et al., 2023; Wang et al., 2023), but CredWise uses a controlled version suited to a financial decisionsupport setting. The decision-intelligence layer assembles a structured package for each applicant, containing:

• calibrated probability of default and predicted class;

• assigned risk level and model/prediction information;

• SHAP-based risk-raising and risk-reducing features;

• retrieved policy evidence and relevant SQL-based analytics; and

• a structured summary for human review.

This layer organizes evidence rather than producing an independent prediction, preserving the distinction between model, policy, and analytical evidence. The final output therefore connects the specialized components without treating the agent or language model as the source of the underlying model prediction.

## 5.6. Human-in-the-Loop and Auditability

CredWise supports review but does not make the final lending decision, following the principle that AI systems in important workflows should keep users informed and support human control (Amershi et al., 2019), consistent with the emphasis on interpretable, understandable outputs in financial lending (Bussmann et al., 2021; B¨ucker et al., 2022). The reviewer considers the model probability, explanation, policy evidence, and analytical context together; no single evidence source is presented as suficient on its own. CredWise keeps four types of information distinct:

1. Model evidence: prediction probability and SHAP feature contributions.

2. Policy evidence: retrieved sections from the lending policy document.

3. Analytical evidence: results from the controlled SQL analytics layer.

4. Decision-support output: a structured summary combining the available evidence for human review.

This separation prevents a generated explanation from being mistaken for a model output or policy statement and makes each component easier to evaluate independently. Each component is evaluated separately before combination: the prediction model on temporal test data, calibration using probability-based metrics, SHAP for feature importance and temporal stability, policy retrieval using labeled queries, SQL using predefined analytical cases, and the agent layer using predefined routing cases. The decision-intelligence layer is then checked to confirm that the expected evidence fields are present and correctly connected. This component-wise strategy is important because a successful final response alone does not confirm that every underlying component is working correctly; detailed results are presented in Section 6.

## 6. Experimental Results

This section reports the experimental results of the CredWise framework. The evaluation is divided into model performance, feature ablation, probability calibration, explanation stability, feature drift, error analysis, subgroup analysis, and evaluation of the retrieval, SQL, and agent components. The final model is evaluated on the 2018 test period, which was not used for model training or model selection. Unless otherwise stated, the reported test results therefore refer to this 2018 period.

## 6.1. Overall Prediction Performance

The final XGBoost model was trained using data from 2007–2016 and validated on the 2017 period. Early stopping selected iteration 187, where the validation log-loss was 0.633752. On the 2018 test set, the calibrated model obtained an accuracy of 65.44%, precision of 26.02%, recall of 64.84%, and F1- score of 0.3714. The ROC-AUC was 0.7109 and the PR-AUC was 0.2993. The

Brier score after calibration was 0.1273. The results for the 2017 validation period and the 2018 test period are shown in Table 3.

Table 3: Performance of the final model on the 2017 validation and 2018 test periods.
<table><tr><td>Metric</td><td>2017</td><td>2018</td></tr><tr><td>Accuracy</td><td>0.6403</td><td>0.6544</td></tr><tr><td>Precision</td><td>0.3539</td><td>0.2602</td></tr><tr><td>Recall</td><td>0.6724</td><td>0.6484</td></tr><tr><td>F1-score</td><td>0.4637</td><td>0.3714</td></tr><tr><td>ROC-AUC</td><td>0.7091</td><td>0.7109</td></tr><tr><td>PR-AUC</td><td>0.4075</td><td>0.2993</td></tr><tr><td>Brier score</td><td>0.1606</td><td>0.1273</td></tr></table>

Figure 5 provides a visual comparison of the main performance metrics across the two periods.  
![](images/777bd272b861a8cb1ed41022210bf387b61785b90706b0f45489696b092c4ba3.jpg)  
Figure 5: Comparison of model performance between the 2017 validation period and the 2018 test period.

The ROC-AUC values are similar across the two periods, while the PR-AUC and F1-score are lower in 2018. This diference is partly associated with the diferent class distribution in the two periods. Therefore, performance is reported using several metrics rather than a single measure.

## 6.2. Feature Set Ablation

Two feature sets were evaluated to measure the efect of adding interest-rate and related loan features. Feature Set A contains 15 features, while Feature Set B adds int rate, installment, and sub grade. This ablation uses a random train-test split (rather than the temporal split used elsewhere in this paper) to isolate the efect of the added features without confounding it with the training/test-period diference. Table 4 presents the results.

Table 4: Feature set ablation results on the earlier random train-validation-test split.
<table><tr><td>Metric</td><td>Feature Set A</td><td>Feature Set B</td><td>Change</td></tr><tr><td>Accuracy</td><td>0.6462</td><td>0.6497</td><td>+0.0035</td></tr><tr><td>Precision</td><td>0.3113</td><td>0.3215</td><td>+0.0102</td></tr><tr><td>Recall</td><td>0.6370</td><td>0.6799</td><td>+0.0429</td></tr><tr><td>F1-score</td><td>0.4182</td><td>0.4366</td><td>+0.0184</td></tr><tr><td>ROC-AUC</td><td>0.6985</td><td>0.7198</td><td>+0.0213</td></tr><tr><td>PR-AUC</td><td>0.3650</td><td>0.3842</td><td>+0.0192</td></tr><tr><td>Brier score</td><td>0.2195</td><td>0.2140</td><td>-0.0054</td></tr></table>

Feature Set B gives higher ROC-AUC, PR-AUC, recall, precision, and F1- score than Feature Set A. Its Brier score is also lower by 0.0054. The results indicate that the three added features provide useful predictive information in this ablation experiment. The ablation experiment and the final model evaluation use diferent evaluation setups. The ablation results are obtained from the earlier random split, whereas the final model is evaluated using the temporal split with 2007–2016 for training, 2017 for validation, and 2018 for testing. Therefore, the Feature Set B ROC-AUC of 0.7198 in the ablation experiment should not be directly compared with the final temporal-test ROC-AUC of 0.7109. The diference does not result from probability calibration, since calibration does not change the ranking of model scores. This experiment is an ablation comparison within the specified modelling setup. It does not establish that the added features have a causal efect on loan outcomes.

## 6.3. Probability Calibration

The raw XGBoost probability was compared with the calibrated probability on the 2018 test period. Before calibration, the Brier score was 0.2157 and the ECE was 0.2862. After sigmoid calibration, the Brier score decreased to 0.1273 and the ECE decreased to 0.0585. Table 5 summarizes the calibration results.

Table 5: Probability calibration results on the 2018 test period.
<table><tr><td>Metric</td><td>Before Calibration</td><td>After Calibration</td></tr><tr><td>Brier score</td><td>0.2157</td><td>0.1273</td></tr><tr><td>ECE</td><td>0.2862</td><td>0.0585</td></tr></table>

The Brier score decreased by 0.0883 and the ECE decreased by 0.2278. The calibration step does not change the ordering of model scores. Therefore, the calibration result is interpreted as an improvement in probability quality rather than as a change in the underlying ranking model. For additional context, the default rate in the 2018 test set was 0.1575. A constant prediction equal to this test-set prevalence gives a Brier score of approximately 0.1327. The calibrated model therefore has a Brier score of 0.1273, corresponding to a Brier skill score of approximately 0.040 relative to this prevalence baseline. This comparison shows that the calibration result should be interpreted together with a simple prevalence baseline rather than only as a reduction from the raw XGBoost score.

## 6.4. Bootstrap Confidence Intervals

Bootstrap resampling was used to estimate 95% confidence intervals for the final 2018 test metrics. The results are reported in Table 6.

Table 6: Bootstrap estimates and 95% confidence intervals for the 2018 test results.
<table><tr><td>Metric</td><td>Estimate</td><td>95% CI</td></tr><tr><td>Accuracy</td><td>0.6544</td><td>[0.6505, 0.6584]</td></tr><tr><td>Precision</td><td>0.2602</td><td>[0.2544, 0.2661]</td></tr><tr><td>Recall</td><td>0.6484</td><td>[0.6380, 0.6579]</td></tr><tr><td>F1-score</td><td>0.3714</td><td>[0.3644, 0.3783]</td></tr><tr><td>ROC-AUC</td><td>0.7109</td><td>[0.7050, 0.7164]</td></tr><tr><td>PR-AUC</td><td>0.2993</td><td>[0.2904, 0.3080]</td></tr><tr><td>Brier score</td><td>0.1273</td><td>[0.1258, 0.1289]</td></tr></table>

The intervals provide an estimate of the variation of the reported metrics under bootstrap resampling of the 2018 test observations. They are not intended to represent variation across diferent datasets or future economic conditions.

## 6.5. SHAP Feature Importance and Temporal Stability

SHAP was used to examine the contribution of individual features to the model predictions. The global feature-importance analysis for 2018 is shown in Figure 6.

![](images/94055744c236a0684c7ad8660f7e507ff1a395b3e24a4849dcfb31c3c0c74c59.jpg)  
Figure 6: Global SHAP feature importance for the final model on the 2018 test period.

The largest mean absolute SHAP values in 2018 were observed for int rate, term, home ownership, dti, and open acc. Other features with notable contributions included annual inc, sub grade, revol bal, revol util, and loan amnt. The temporal stability of these rankings was evaluated by comparing the 2017 and 2018 global SHAP rankings. Table 7 summarizes the results.

Table 7: Temporal stability of global SHAP feature rankings.
<table><tr><td>Measure</td><td>Result</td></tr><tr><td>Spearman rank correlation</td><td>0.9959</td></tr><tr><td>Spearman p-value</td><td> $4 . 1 8 \times 1 0 ^ { - 1 8 }$ </td></tr><tr><td>Top-5 overlap</td><td>100%</td></tr><tr><td>Top-10 overlap</td><td>90%</td></tr><tr><td>Top-15 overlap</td><td>100%</td></tr></table>

The Spearman correlation of 0.9959 indicates that the global feature ranking was very similar between the two periods. The top-5 and top-15 sets were identical, while the top-10 sets had 90% overlap. This result describes stability of the model’s feature-importance ranking; it does not imply that the relationships between the features and default are causal.

## 6.6. Feature Drift Analysis

Feature distribution changes were measured using the Population Stability Index (PSI). The 2007–2016 training period was compared with the 2017 and 2018 periods. Table 8 presents selected features with their PSI values.

Table 8: Selected feature drift measurements using PSI.
<table><tr><td>Feature</td><td>2017 PSI</td><td>2018 PSI</td></tr><tr><td>revol_util</td><td>0.0967</td><td>0.3305</td></tr><tr><td>int_rate</td><td>0.0741</td><td>0.1531</td></tr><tr><td>revol_bal</td><td>0.0200</td><td>0.0915</td></tr><tr><td>loan_amnt</td><td>0.0267</td><td>0.0600</td></tr><tr><td>installment</td><td>0.0259</td><td>0.0439</td></tr><tr><td>dti</td><td>0.0024</td><td>0.0392</td></tr><tr><td>pub_rec</td><td>0.0000</td><td>0.0332</td></tr><tr><td>total_acc</td><td>0.0099</td><td>0.0216</td></tr><tr><td>open_acc</td><td>0.0038</td><td>0.0175</td></tr><tr><td>inq_last_6mths</td><td>0.0091</td><td>0.0155</td></tr><tr><td>annual_inc</td><td>0.0057</td><td>0.0097</td></tr><tr><td>delinq-2yrs</td><td>0.0000</td><td>0.0089</td></tr></table>

Figure 7 shows the PSI values for the selected features.

![](images/4edf4411c2d17434144dffea18f2500cad1c6c8f76e5859cbd142985388e0a8c.jpg)  
Figure 7: Population Stability Index for selected features between the training period and later evaluation periods.

The largest shift was observed for revol util, whose PSI increased from 0.0967 in 2017 to 0.3305 in 2018. Interest rate also showed a noticeable increase, from 0.0741 to 0.1531. These values indicate that the distributions of some input variables changed over time. The drift analysis does not by itself establish model degradation or the cause of the observed distribution changes. It is therefore considered together with the temporal performance and error analyses.

## 6.7. Error Analysis

At the selected classification threshold of 0.23, the 2018 test set produced the confusion matrix shown in Table 9.

Table 9: Confusion matrix and error rates on the 2018 test period.
<table><tr><td colspan="2">Predicted Non-default</td><td>Predicted Default</td></tr><tr><td colspan="2">Actual Non-default</td><td>31,102 16,342</td></tr><tr><td colspan="2">Actual Default</td><td>3,118 5,749</td></tr><tr><td rowspan="3"></td><td>Measure Value</td><td></td></tr><tr><td>Non-default error rate 34.44%</td><td></td></tr><tr><td>Default error rate 35.16%</td><td></td></tr></table>

The model produced 16,342 false positives and 3,118 false negatives. The false-positive group contained, on average, larger loan amounts, higher interest rates, higher DTI, and higher revolving utilization than the true-negative group. The false-negative group generally had lower interest rates and lower DTI than the true-positive group. These patterns show that some observations are dificult to classify using the available features. The error analysis is descriptive and does not imply that any individual feature causes the prediction error.

## 6.8. Subgroup Robustness

Subgroup analysis was performed for employment length, home ownership, loan term, and verification status. The purpose was to examine whether model performance was reasonably consistent across diferent groups. The subgroup analysis uses the same 2018 test predictions and the same classification threshold used for the main evaluation. The complete subgroup results are reported in the corresponding evaluation table included with the research artifacts. The subgroup results are descriptive and should not be interpreted as a formal fairness assessment. The analysis is limited to the available group definitions and the observed test-set sample sizes.

## 6.9. Policy Retrieval Evaluation

The policy retrieval component was evaluated using a labeled set of 28 queries covering all nine policy sections. Each query was associated with an expected policy section. The evaluation measured whether the expected section appeared in the top-k retrieved results, for BM25, FAISS, and a reciprocalrank-fusion hybrid of the two. The agent-routing component was evaluated on 45 labeled requests covering four route types (decision, policy, risk, and SQL). Table 10 reports the retrieval metrics together with the SQL and agent-routing evaluations.

Table 10: Evaluation of policy retrieval, SQL analytics, and agent routing.
<table><tr><td>Component</td><td>Metric</td><td>Result</td></tr><tr><td>Policy RAG (BM25)</td><td>Hit@1 / Hit@2 / Hit@3 / Hit@5 0.714 / 0.857 / 0.857 / 1.000</td><td></td></tr><tr><td>Policy RAG (BM25)</td><td>MRR</td><td>0.818</td></tr><tr><td>Policy RAG (FAISS)</td><td>Hit@1 / Hit@2 / Hit@3 / Hit@5</td><td>0.929 / 1.000 / 1.000 / 1.000</td></tr><tr><td>Policy RAG (FAISS)</td><td>MRR</td><td>0.964</td></tr><tr><td>Policy RAG (Hybrid)</td><td>Hit@1 / Hit@2 / Hit@3 / Hit@5</td><td>0.821 / 1.000 / 1.000 / 1.000</td></tr><tr><td>Policy RAG (Hybrid)</td><td>MRR</td><td>0.911</td></tr><tr><td>SQL Agent</td><td>SQL exact match</td><td>1.000</td></tr><tr><td>SQL Agent</td><td>Execution success</td><td>1.000</td></tr><tr><td>SQL Agent</td><td>Result match</td><td>1.000</td></tr><tr><td>Agent Routing</td><td>Correct cases</td><td>43/45 (0.956)</td></tr></table>

The retrieval component was evaluated on a 28-query labeled set covering all nine policy sections. FAISS achieved the strongest ranking quality (Hit@1 = 0.929, MRR = 0.964), reaching Hit@2 = 1.0. BM25 reached Hit@1 = 0.714 and did not reach Hit@3 = 1.0, needing the full top-5 to retrieve the expected section for every query (MRR = 0.818). The hybrid reciprocal-rank-fusion method fell between the two on Hit@1 (0.821) and MRR (0.911) while matching FAISS from Hit@2 onward. All three methods reached Hit@5 = 1.0. The SQL benchmark contained six cases and achieved exact match, execution success, and result match values of 1.0. The agent-routing evaluation contained 45 predefined cases covering decision, policy, risk, and SQL routes and achieved an overall accuracy of 0.956 (macro F1 = 0.96). Per-route performance was strongest for policy routing (precision = recall = 1.00) and weakest for the risk route, where precision was 0.91 because some SQL-route cases were misclassified as risk; recall for SQL routing was correspondingly 0.90. These results should be interpreted within the size and design of the evaluation sets. The 28-query policy benchmark and the 45-case routing benchmark are larger than a handful of cases but remain controlled evaluations drawn from a single policy document and a predefined case set, and should not be treated as evidence of large-scale retrieval or routing performance in production settings.

## 6.10. Decision-Intelligence Evaluation

The final decision-intelligence component was evaluated using one representative applicant from the 2018 test period. The applicant had a raw model probability of 0.9004 and a calibrated probability of 0.6108. Since the calibrated probability was above the classification threshold of 0.23, the predicted class was default. The corresponding risk level was High Risk according to the predefined risk ranges. The final decision-support package contained the calibrated prediction, raw prediction, classification threshold, model information, four risk-raising SHAP features, three risk-reducing SHAP features, two policyevidence items, and four analytical context fields. The validation checks for the final decision-intelligence output passed. This evaluation verifies that the required evidence components can be assembled for a representative case. It does not measure the reliability of the decision-intelligence workflow across a larger population of applicants.

## 7. Discussion

The experiments show that CredWise can combine credit-risk prediction, probability calibration, explainability, policy retrieval, SQL analytics, and controlled agent workflows into a single decision-support system. The results also show that no single metric is suficient to describe the system. Prediction performance, probability quality, explanation stability, feature drift, and componentlevel behaviour provide diferent information.

## 7.1. Predictive Performance and Calibration

The final XGBoost model achieved a ROC-AUC of 0.7109, PR-AUC of 0.2993, and F1-score of 0.3714 on the 2018 test period at the selected threshold of 0.23. These results indicate useful ranking performance, but also show that the model does not perfectly separate default and non-default cases. This is expected in credit-risk prediction, where many factors afecting borrower out comes may not be available in the dataset. The PR-AUC result is particularly relevant because the positive class is imbalanced. Reporting both ROC-AUC and precision-recall measures provides a more complete view of classification performance (Davis & Goadrich, 2006; Lessmann et al., 2015). The feature ablation experiment showed that Feature Set B performed better than Feature Set A in the tested setup. ROC-AUC increased from 0.6985 to 0.7198, recall from 0.6370 to 0.6799, and F1-score from 0.4182 to 0.4366. (This comparison uses a random split and is therefore not directly comparable in absolute terms to the temporal-split results in Table 3.) The additional features were int rate, installment, and sub grade. This suggests that these variables contain useful predictive information in this dataset, but the result does not imply that they cause default outcomes. Probability calibration reduced the Brier score from 0.2157 to 0.1273 and ECE from 0.2862 to 0.0585. Since calibration is applied after model prediction, it does not change the underlying XGBoost model o its feature ranking. The result therefore indicates improved probability qual ity rather than improved underlying classification performance. For context, the Brier score of a constant predictor based on the 2018 test-set default rate is approximately 0.1327, giving a Brier skill score of approximately 0.040 for the calibrated model. Thus, the calibrated probabilities are only modestly bet ter than the prevalence baseline, even though the improvement over the raw XGBoost probabilities is substantial. (Niculescu-Mizil & Caruana, 2005; Platt, 1999; Guo et al., 2017). Using both discrimination and calibration measures is useful when model probabilities are used as part of decision support.

## 7.2. Temporal Stability and Feature Drift

The global SHAP rankings for 2017 and 2018 had a Spearman correlation of 0.9959. The top-5 and top-15 feature sets had 100% overlap, while the top-10 sets had 90% overlap. This indicates that the relative importance of the main features was highly similar across the two evaluated periods. However, this result does not establish stability under future conditions. SHAP values describe model feature contributions and should not be interpreted as causal efects (Lundberg & Lee, 2017; Bussmann et al., 2021). The feature drift analysis showed that the input distributions changed between the training period and later periods. The largest observed PSI was for revol util, reaching 0.3305 in 2018, followed by int rate with a PSI of 0.1531. These results indicate temporal changes in the input data, but PSI alone does not establish the reasons for those changes or prove model degradation. The combination of temporal performance, SHAP stability, and feature drift therefore provides a broader view of model behaviour (Gama et al., 2014).

## 7.3. Error and Subgroup Behaviour

The 2018 test set contained 16,342 false positives and 3,118 false negatives at the selected threshold. Diferences between these groups were observed in variables including loan amount, interest rate, income, DTI, and revolving utilization. The presence of both error types shows that the model does not perfectly separate the two outcome classes. The subgroup analysis examined employment length, home ownership, loan term, and verification status. Its purpose was to check performance variation across the selected groups rather than to provide a complete fairness evaluation. The study does not include all potentially relevant protected attributes or a full fairness framework. Therefore, the subgroup results should be treated as a robustness check within the available test data. These findings also support the human-review design of CredWise. Model probabilities should be treated as one source of evidence together with explanations, policy information, and other available information rather than as an unquestionable decision.

## 7.4. Retrieval, Analytics, and Agentic Components

The policy retrieval component achieved Hit@5 = 1.0 for BM25, FAISS, and the hybrid method on the 28-query labeled evaluation set, with FAISS reaching the highest MRR (0.964). This shows that the expected policy section was retrieved for each query in the controlled evaluation. However, the small evaluation set does not support generalization to larger or more diverse policy collections (Lewis et al., 2020). The SQL benchmark passed all six predefined cases, including expected query outputs and the tested safety checks. These results validate the implemented prototype for the tested scenarios, but do not establish safety for all possible SQL inputs or production databases. The agent-routing evaluation contained 45 predefined cases covering decision, policy, risk, and SQL routes and achieved an overall accuracy of 0.956 (43/45 cases; macro F1 = 0.96). The result confirms the behaviour of the tested routing logic but does not establish general agent reliability. Agent systems can introduce routing, tool-use, retrieval, and generation errors, making component-level evaluation important (Wang et al., 2023; Liu et al., 2023). The final decisionintelligence evaluation successfully combined model prediction, SHAP evidence, policy evidence, and analytical context into one structured output. Its role is to organize available evidence rather than produce an independent credit-risk prediction.

## 7.5. Human-in-the-Loop Decision Support and Relation to Previous Work

CredWise is designed so that the system does not make the final lending decision. Instead, it produces a decision-support package for human review. Keeping prediction, retrieved policy evidence, explanations, and analytics as separate evidence sources allows the reviewer to inspect the basis of the generated output. This follows the broader principle of maintaining appropriate human control in AI-assisted workflows (Amershi et al., 2019; Bussmann et al., 2021; B¨ucker et al., 2022). CredWise also builds on earlier work, which studied multi-agent and retrieval-based workflows for enterprise customer support (Tiwari & Kumar, 2026). The present work applies related system ideas to credit-risk decision support, with additional emphasis on model explanations, probability calibration, policy evidence, and financial analytics. An earlier study of local LLM judges found that high internal consistency did not necessarily imply agreement with human ratings (Tiwari, 2026). This motivates the use of predefined component-level tests and human review rather than treating automated agent outputs as automatically reliable. Overall, the results address the seven research questions by showing that the final model provides useful temporal test performance, Feature Set B improves the tested predictive metrics, calibration improves probability quality, SHAP rankings remain highly similar across the evaluated periods, and measurable feature drift exists between the training and later periods. The controlled retrieval, SQL, and agent evaluations also produced the expected outputs, while the decision-intelligence layer successfully combined the available evidence into a human-reviewable output.

## 8. Limitations

This study has several limitations that should be considered when interpreting the results. The experiments rely on a single historical Lending Club dataset and a temporal evaluation window ending in 2018 (2007–2016 training, 2017 validation, 2018 test). While this provides a more realistic evaluation than a random split, it does not show how the model would behave under future changes in borrower populations, lending policies, or economic conditions, and the results may not directly transfer to other institutions or portfolios. The policy retrieval, SQL, and agent-routing components were each evaluated using small, predefined test sets – 28 labeled queries for retrieval, six cases for SQL, and 45 cases for agent routing. These benchmarks confirmthe implemented behaviour for the tested scenarios but are too limited to support claims of general retrieval accuracy, SQL robustness, or agent reliability under a broader range of queries and inputs. The decision-intelligence layer was also evaluated on a single representative applicant, so the result demonstrates component integration for that case but does not establish robustness across a larger applicant population. The explainability and drift analyses are descriptive rather than causal. SHAP values quantify feature contributions to the model output, and PSI quantifies distributional change; neither establishes why a feature is important or why a distribution shifted, and the observed temporal stability of SHAP rankings does not guarantee stability under future data changes. The subgroup analysis, covering employment length, home ownership, loan term, and verification status, is a robustness check rather than a complete fairness evaluation, since it does not cover all potentially relevant protected or sensitive attributes. Finally, CredWise is an academic research prototype rather than a production lending system. It does not provide regulatory, legal, or financial advice, and the reported experiments do not establish suitability for autonomous lending decisions; the final decision is intentionally left to a human reviewer. Future work can address these limitations through evaluation on additional datasets and time periods, larger and more diverse retrieval/SQL/agent benchmarks, and a broader analysis of robustness, fairness, and deployment-related risks.

## 9. Conclusion

This paper presented CredWise, an academic research prototype for controlled credit-risk decision support that combines an XGBoost credit-risk model with probability calibration, SHAP-based explainability, policy retrieval, SQL analytics, and controlled agent-based workflows within a single evidence-first architecture. Using a temporal evaluation design (2007–2016 training, 2017 validation, and 2018 untouched test) over 1,345,310 loan records, the calibrated model achieved a ROC-AUC of 0.7109 and an F1-score of 0.3714 on the final test period. Sigmoid calibration reduced the Brier score from 0.2157 to 0.1273 and the expected calibration error from 0.2862 to 0.0585. These results indicate that the model provides useful ranking performance and that calibration improves the quality of the predicted probabilities. The explanation and drift analyses together highlight an important observation of this study: model behaviour can remain highly stable at the feature-importance level, with a Spearman correlation of 0.9959 between the 2017 and 2018 global SHAP rankings, even as the underlying input distributions shift. The largest observed shifts were found for revol util and int rate. This distinction between explanation stability and data drift suggests that temporal monitoring can be useful in addition to one-time model validation. The supporting components – policy retrieval, SQL analytics, and agent routing – were each evaluated independently against pre defined test cases rather than assessed only through the correctness of a final generated response. This component-wise evaluation strategy, combined with the decision-intelligence layer’s ability to assemble model, policy, and analytical evidence into a single structured output, allows the diferent sources of evidence to remain traceable when they are combined for review. Importantly, CredWise does not automate the lending decision itself. The system is deliberately con strained to produce decision-support evidence, with the final judgment left to a human reviewer. This human-in-the-loop design, together with the separation of model, policy, and analytical evidence, is central to the auditability goals of the framework. The reported results should be interpreted within the scope of a single historical dataset, a limited temporal window, and small controlled benchmarks for the retrieval, SQL, and agent components, as discussed in Section 8. Broader claims about real-world deployment, generalization to other lending institutions, or long-term robustness would require evaluation on addi tional datasets, longer time horizons, and larger benchmark sets. Future work along these directions, together with a more comprehensive fairness analysis, would help clarify the conditions under which a system such as CredWise could support, rather than replace, human credit-risk decision-making.

## Data Availability

The study uses the publicly available Lending Club historical loan dataset (accepted 2007 to 2018Q4.csv), obtained from Kaggle (https://www.kaggle. com/datasets/wordsforthewise/lending-club), subject to that platform’s licensing and redistribution terms. The CredWise implementation, trained model artifacts, and research evaluation scripts will be made available in a public repository upon publication.

## Declaration of Generative AI and AI-assisted Technologies

During the preparation of this manuscript, the author used ChatGPT (OpenAI) and Claude (Anthropic) for manuscript organization, language improvement, and editing. The author reviewed and revised all AI-assisted content, independently verified the reported results and references, and takes full responsibility for the final content of the manuscript.

## Acknowledgements

The author thanks the Indian Institute of Technology Kharagpur for providing the academic environment and computational resources for this research.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT authorship contribution statement

Aakash Kumar Tiwari: Conceptualization, Methodology, Software, Data curation, Formal analysis, Investigation, Visualization, Writing – original draft, Writing – review & editing.

## Declaration of competing interest

The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

Amershi, S., Weld, D., Vorvoreanu, M., Fourney, A., Nushi, B., Collisson, P., Suh, J., Iqbal, S., Bennett, P., Inkpen, K., Teevan, J., Kikin-Gil, R., & Horvitz, E. (2019). Guidelines for human-ai interaction. In Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems. doi:10.1145/3290605.3300233.

Arrieta, A. B., Diaz-Rodriguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., Garcia, S., Gil-Lopez, S., Molina, D., Benjamins, R., Chatila, R., & Herrera, F. (2020). Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai. Information Fusion, 58 , 82–115. doi:10.1016/j.inffus.2019.12.012.

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78, 1–3. doi:10.1175/1520-0477-78.1.1.

Bussmann, N., Giudici, P., Marinelli, D., & Papenbrock, J. (2021). Explainable machine learning in credit risk management. Computational Economics, 57, 203–216. doi:10.1007/s10614-020-10042-0.

B¨ucker, M., Szepannek, G., Gosiewska, A., & Biecek, P. (2022). A holistic approach to interpretability in financial lending: Models, visualizations, and summary-explanations. Decision Support Systems, 152, 113647. doi:10.1016/j.dss.2021.113647.

Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). doi:10.1145/ 2939672.2939785.

Cormack, G. V., Clarke, C. L. A., & Buttcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research

and Development in Information Retrieval (pp. 758–759). doi:10.1145/ 1571941.1572114.

Davis, J., & Goadrich, M. H. (2006). The relationship between precision-recall and roc curves. In Proceedings of the 23rd International Conference on Machine Learning (pp. 233–240). doi:10.1145/1143844.1143874.

Efron, B., & Tibshirani, R. (1986). Bootstrap methods for standard errors, confidence intervals, and other measures of statistical accuracy. Statistical Science, 1 , 54–75. doi:10.1214/ss/1177013815.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29 , 1189–1232. doi:10.1214/aos/ 1013203451.

Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. ACM Computing Surveys, 46 , 1–37. doi:10.1145/2523813.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning (pp. 1321–1330). PMLR volume 70.

Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: A review. Journal of the Royal Statistical Society: Series A (Statistics in Society), 160 , 523–541. doi:10.1111/j.1467-985X. 1997.00078.x.

Johnson, J., Douze, M., & Jegou, H. (2021). Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7 , 535–547. doi:10.1109/TBDATA. 2019.2921572.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-t. (2020). Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods

in Natural Language Processing (pp. 6769–6781). doi:10.18653/v1/2020.   
emnlp-main.550.

Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. European Journal of Operational Research, 247 , 124–136. doi:10. 1016/j.ejor.2015.05.030.

Lessmann, S. et al. (2024). Advancing credit risk modelling with machine learning: A comprehensive review of the state-of-the-art. Engineering Applications of Artificial Intelligence, 137 , 109082. doi:10.1016/j.engappai. 2024.109082.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.-t., Rocktaschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems. volume 33.

Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., Gu, Y., Ding, H., Men, K., Yang, S., Deng, X., Zeng, A., Du, Z., Zhang, C., Shen, S., Zhang, T., Su, Y., Sun, H., Huang, M., Dong, Y., & Tang, J. (2023). Agentbench: Evaluating llms as agents. In arXiv preprint arXiv:2308.03688 .

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems. volume 30.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In Proceedings of the 22nd International Conference on Machine Learning (pp. 625–632). doi:10.1145/1102351.1102430.

Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In Advances in Large Margin Classifiers (pp. 61–74). MIT Press.

Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (pp. 3982–3992). doi:10.18653/v1/D19-1410.

Tiwari, A. K. (2026). When consistency does not mean reliability: Evaluating local llm judges against human ratings. doi:10.48550/arXiv.2609.13824. arXiv:2609.13824.

Tiwari, A. K., & Kumar, S. (2026). Shopease: A generative ai-based multiagent framework for intelligent enterprise customer support using hybrid retrieval-augmented generation. doi:10.48550/arXiv.2609.13856. arXiv:2609.13856.

Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei, Z., & Wen, J.-R. (2023). A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432, .

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., & Cao, Y. (2023). React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.