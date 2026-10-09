# Clinician use of language models diverges from how the models are evaluated

Krithik Vishwanath BS<sup>1</sup>, Haitong Lin MS<sup>2</sup>, Anton Alyakin MSE<sup>1,3,4</sup>, Jin Vivian Lee MD MS<sup>1,4</sup>,   
D. Brock Hewitt MD MPH<sup>5</sup>, Jie J. Yao MD<sup>6</sup>, William Robert Small MD MBA<sup>7,8</sup>, Hammad A. Khan MD<sup>1</sup>, Cordelia Orillac MD<sup>1</sup>, Aakaash Varma MD<sup>9</sup>, Brandon Ye MS<sup>1,10</sup>, Daniel Alexander Alber MD<sup>11</sup>, Gustavo Stolovitzky PhD<sup>12,13</sup>, Batia Wiesenfeld PhD<sup>14</sup>, Oded Nov PhD<sup>2</sup>, Wei Wu BS<sup>15,16</sup>, Kang Zhang MD PhD<sup>15</sup>, Yindalon Aphinyanaphongs MD PhD<sup>7,8,17</sup>, Tim Requarth PhD<sup>18‡</sup>,   
Eric Karl Oermann MD<sup>1,4,19–21‡</sup> & The International Digital Twin Consortium in Healthcare and Medicine ‡ - equal co-senior authorship   
<sup>1</sup>Department of Neurosurgery, NYU Langone Health, New York, NY, USA   
<sup>2</sup>Department of Technology Management, NYU Tandon School of Engineering, New York University, New York, NY, USA   
<sup>3</sup>Washington University School of Medicine, St. Louis, MO, USA   
<sup>4</sup>Global AI Frontier Lab, New York University, New York, NY, USA   
<sup>5</sup>Department of Surgery, NYU Langone Health, New York, NY, USA   
<sup>6</sup>Department of Orthopedic Surgery, NYU Langone Health, New York, NY, USA   
<sup>7</sup>Department of MCIT Health Informatics, NYU Langone Health, New York, NY, USA   
<sup>8</sup>Department of Medicine, NYU Langone Health, New York, NY, USA   
<sup>9</sup>Division of Dermatology, Department of Medicine, NYU Langone Long Island, Mineola, NY, USA   
<sup>10</sup>Johns Hopkins University School of Medicine, Baltimore, MD, USA   
<sup>11</sup>Department of Cardiothoracic Surgery, Stanford University School of Medicine, Stanford, CA, USA   
<sup>12</sup>Department of Pathology, NYU Grossman School of Medicine, New York, NY, USA   
<sup>13</sup>Biomedical Data Science Hub, NYU Langone Health, New York, NY, USA   
<sup>14</sup>Department of Management and Organizations, NYU Stern School of Business, New York, NY, USA   
<sup>15</sup>Faculty of Medicine, Macau University of Science and Technology, Taipa, Macao, China   
<sup>16</sup>Department of Big Data and Biomedical AI, College of Future Technology, Peking University and Peking-Tsinghua Center for Life   
Sciences, Beijing, China   
<sup>17</sup>Department of Population Health, NYU Langone Health, New York, NY, USA   
<sup>18</sup>Department of Neuroscience, NYU Langone Health, New York, NY, USA   
<sup>19</sup>Department of Radiology, NYU Langone Health, New York, NY, USA   
<sup>20</sup>Neuroscience Institute, NYU Langone Health, New York, NY, USA   
<sup>21</sup>Center for Data Science, New York University, New York, NY, USA

Send Correspondence to:

Krithik Vishwanath, BS   
Department of Neurosurgery,   
NYU Langone Medical Center,   
New York University, 550 First Ave, MS 3-205,   
New York, NY10016, USA.   
Email: krithik.vishwanath@nyulangone.org Tim Requarth, PhD   
Department of Neuroscience,   
NYU Langone Medical Center,   
New York University, 550 First Ave, MSB 4-112,   
New York, NY10016, USA.   
Email: tim.requarth@nyulangone.org Eric K. Oermann, MD   
Department of Neurosurgery,   
NYU Langone Medical Center,   
New York University, 550 First Ave, MS 3-205,   
New York, NY10016, USA.   
Email: eric.oermann@nyulangone.org

## Abstract

Large language model (LLM) assistants are being deployed to clinicians across health systems, and judgments about their readiness rest largely on benchmark scores, most of them derived from examination questions or curated cases. A benchmark predicts performance in deployment only to the extent that its items resemble real use, yet whether benchmarks reflect the work these systems receive has rarely been measured. Here we analyze 127,833 queries sent by 6,342 physicians, advanced practice providers and nurses in 35 specialties to an institutional assistant during an eight-month roll-out. We characterize each query with RCQ-Map, a clinician-validated framework grounded in taxonomies of clinical questions and of LLM evaluation, which records its task, intent, answerability, missing information and potential harm. Documentation and administration (36.2%) and knowledge retrieval (28.9%) made up nearly two-thirds of use, and diagnosis 3.7%; more than a third of queries could not be answered well as posed. Applying RCQ-Map to 58 public

benchmarks drawn from major evaluation suites and frontier model reports, which we assemble into the Clinical AI Benchmark Atlas, showed that the median benchmark contained no documentation requests and shared 31% of the task mix of real use, less than an even spread across task categories would. Benchmarks in suites designed to resemble clinical practice were individually no closer to real use than those used in frontier model reports. Benchmark scores therefore say little about how clinical AI performs on most of the work it is actually given, and evaluation should be matched to real clinical use.

## Main

More than 80% of US physicians reported using AI professionally in $2 0 2 6 ^ { 1 }$ , and health systems now ofer secure large language model (LLM) assistants to their clinical staf, including physicians, nurses, and advanced practice providers<sup>2–4</sup>. How these deployed systems are evaluated has not kept pace with their prevalence. Of 519 evaluations of LLMs in health care published through early 2024, 5% used real patient-care data<sup>5</sup>. A larger review of 4,609 studies published through September 2025 found that 77.3% did not use real clinical data of any kind<sup>6</sup>. Much of the evaluation evidence comes from examination questions, on which models now score near maximum without matching that accuracy on real cases<sup>7–11</sup>. Newer benchmarks test clinical reasoning and dialogue<sup>12–16</sup>, with two recent ones drawing on questions that clinicians wrote or submitted to clinical AI tools<sup>17,18</sup>. However, these benchmarks may also sufer from non-representativeness, and both draw only on physicians. Evaluation for nurses and nurse practitioners is sparse, with few studies in real clinical workflows<sup>19,20</sup>. Thus, how closely any of these evaluations resembles the work of an AI assistant in real-world use is an open question.

Using current evaluations to judge a deployed assistant depends on two assumptions that may hold in practice. The first is that the tasks that appear in benchmarks are the tasks clinicians use LLM assistants for in practice. A benchmark predicts performance in deployment only to the extent that its items resemble real use<sup>11,21</sup>. Usage logs from secure institutional deployments suggest that writing, summarization, and look-ups predominate<sup>22–25</sup>, as they do in general-population use<sup>26–28</sup>. Where the tasks in real use have been compared with the tasks that evaluations cover, the two have only partly matched: real use diverged from the tasks studied in the literature<sup>22</sup>, and only half of the most common tasks in a chart-connected assistant<sup>25</sup>appeared in the task taxonomy of MedHELM<sup>16</sup>, a clinician-validated taxonomy of 121 medical tasks designed specifically for evaluating LLMs. The second assumption is that clinicians' questions, like most benchmark questions, can be answered as posed. However, many cannot be answered safely without more information. When the answer depends on a patient’s chart, an institution’s formulary, or recently changed guidance, a safe system should ask for the missing details, ground its answer in authoritative sources, or defer to a person, rather than answer<sup>29–32</sup>.

A benchmark is only informative about a deployment if its items cover the requests that deployment receives, both in kind and in frequency<sup>11,21</sup>. The accuracy of clinical AI systems is known to change when the cases they encounter shift away from those on which they were developed and tested<sup>33</sup>. Many clinical benchmarks are built from licensing-examination questions or from cases that experts wrote or selected for dificulty<sup>5,7,18</sup>, so their mix of tasks often reflects how they were constructed rather than how clinicians use these systems.

Here we characterize 127,833 real clinical queries (RCQs) sent to an institutional AI assistant at a large academic health system by 6,342 physicians, advanced practice providers, and nurses across 35 specialties during routine care (Fig. 1). We developed RCQ-Map, a clinician-validated and literature-grounded framework that describes each RCQ in 20 fields including task, intent, whether the query includes appropriate clinical context, and the potential harm of a response. We applied RCQ-Map to every query sent during an 8-month roll-out across a large academic system using our clinician-validated LLM annotator, and then also applied RCQ-Map to the questions in 58 public clinical AI benchmarks. Half of all RCQs come from nurses and advanced practice providers, who are underrepresented in clinical AI evaluation. In our data, documentation and knowledge retrieval make up nearly two-thirds of use and diagnosis 3.7%. The median benchmark contains no documentation requests and shares 31% of this task mix, and benchmarks in suites meant to resemble clinical practice are no closer to it than those used in frontier model reports.

## Results

## RCQs from $\mathbf { 6 , 3 4 2 }$ clinicians across 35 specialties

Between 12 August 2025 and 3 April 2026, 127,833 conversations were contributed by 6,342 clinicians spanning 1,664 attending physicians, 157 fellows, 771 residents, 946 advanced practice providers (APPs), and 2,804 registered nurses (RNs) spanning 35 specialties (Fig. 1a and Table 1). We took the first clinician turn of each conversation as its RCQ, and 127,625 RCQs (99.8%) yielded valid annotations. Use was heavy-tailed: a median clinician contributed 4 conversations (interquartile range 1–16), and the most active 10% of clinicians generated 63.3% of conversations (Gini coeficient 0.76; Extended Data Fig. 1). Since the most active clinicians could dominate estimates made at the conversation level, all confidence intervals account for clustering by clinician, and we repeated the main analyses weighting each clinician equally (Extended Data Fig. 2).

Of all RCQs, 44.1% were instructions addressed to the assistant, 28.6% were questions and 27.2% were keyword fragments; 52.9% used clinical abbreviations and 36.5% concerned a specific patient (Fig. 2f).

## Documentation and knowledge retrieval dominate clinical use, and diagnosis is rare

RCQ-Map assigns each RCQ a task, defined by the deliverable that would satisfy the request, and an intent, defined by the motive behind it. The ten task and twelve intent categories were developed from question types in earlier clinical-question and LLM-task frameworks and were applied with substantial agreement by clinicians (Methods, Extended Data Table 1 and Supplementary Table 1). Documentation and workflow was the largest task category, at 35.1% (95% CI 32.8–37.5%), followed by foundational knowledge (14.8%, 95% CI 13.9–15.7%) and drug information and pharmacotherapy (14.1%, 95% CI 13.2–15.1%) (Fig. 2a). Tasks that require clinical reasoning about a situation together accounted for 14.3%: treatment and management (6.2%), diagnosis and diferential (3.7%), test and result interpretation (2.7%), patient education (1.2%) and procedural guidance (0.5%). Another 20.6% of RCQs were non-clinical, such as research writing or general requests, or had no recoverable request in the first turn (9.0% of all RCQs), usually because the clinician pasted clinical text and asked the question in a later turn. Documentation and administration (36.2%) and knowledge retrieval (28.9%) together made up nearly two-thirds of use.

## a Real clinical queries (RCQs) and their annotation

![](images/852770475f8d6983cea183359900fb19ae63bae0efe71925ea9df79c64ab0c83.jpg)

## b RCQ-Map: 20 fields in six blocks

![](images/0f7666c5ca982717181249eb1fdfd837e8bcdb019d29c4a569a6fb13e3476138.jpg)

## c Clinician validation

![](images/066c11d68ef19577e7c79c379bd9e190e1a3433fb20182c95144cff6ddde3c28.jpg)

## d Clinical AI Benchmark Atlas (CABA)

![](images/932cd55dd5ab8c7c04b2596d90f5c82d1aa7454c75e3df1d0b8e45491538b745.jpg)

Fig. 1 | Study overview. a, Clinicians at a large academic health system (numbers per professional family shown) held 127,833 conversations with a secure LLM assistant. The three example queries are invented to illustrate the three query forms and are not drawn from the corpus. An LLM annotator (GPT-5.6 sol, reasoning efort none), given the annotation guidelines as its system prompt, assigned 24 labels to each query. Right, a worked example from the annotation guidelines with eight of its 24 labels. b, The 20 fields of RCQ-Map reported here, in six blocks, with the number of allowed values for categorical fields. Colored chips show the ordered values of potential harm (least to most severe) and of the safest response (least to most restrictive). c, Validation design: 100 randomly sampled queries were annotated independently by three of seven clinicians and by the corpus annotator using the same annotation guidelines. d, The Clinical AI Benchmark Atlas (CABA): benchmarks named by 13 evaluation suites and by frontier model reports were checked on their downloaded items, up to 200 items from each were sampled, and RCQ-Map was applied to them with the corpus annotator for comparison with RCQs.

Table 1 | Clinicians, conversations and query profile by professional family.
<table><tr><td></td><td>All</td><td>Attending physicians</td><td>Fellows</td><td>Residents</td><td>APPs</td><td>Registered nurses</td></tr><tr><td>Clinicians, n</td><td>6,342</td><td>1,664</td><td>157</td><td>771</td><td>946</td><td>2,804</td></tr><tr><td>Conversations, n (%)</td><td>127,833 (100.0)</td><td>37,383 (29.2)</td><td>2,121 (1.7)</td><td>23,630 (18.5)</td><td>23,906 (18.7)</td><td>40,793 (31.9)</td></tr><tr><td>Conversations per clinician, median 4 (1–16) (IQR)</td><td></td><td>4 (1–20)</td><td>4 (1–13)</td><td>8 (2–33)</td><td>6 (2–23)</td><td>3 (1–11)</td></tr><tr><td>Specialties represented, n Kind of work (%)</td><td>35</td><td>35</td><td>19</td><td>19</td><td>27</td><td></td></tr><tr><td>Documentation &amp; administration</td><td>36.2</td><td>32.1</td><td>37.3</td><td>38.0</td><td>35.5</td><td>39.1</td></tr><tr><td>Knowledge retrieval</td><td>28.9</td><td>25.5</td><td>26.7</td><td>34.9</td><td>34.8</td><td>25.2</td></tr><tr><td>Clinical reasoning</td><td>14.3</td><td>16.2</td><td>14.0</td><td>19.7</td><td>15.4</td><td>8.8</td></tr><tr><td>Other (non-clinical or no ask)</td><td>20.6</td><td>26.2</td><td>21.9</td><td>7.4</td><td>14.4</td><td>26.8</td></tr><tr><td>Query profile (%) Patient-specific</td><td>36.5</td><td>33.0</td><td>44.1</td><td>46.7</td><td>43.7</td><td></td></tr><tr><td>Requests text generation</td><td>41.1</td><td>41.5</td><td>43.9</td><td>38.9</td><td>36.2</td><td>29.1 44.8</td></tr><tr><td>Cannot be answered well as posed</td><td>36.1</td><td>38.9</td><td>37.3</td><td>31.1</td><td>37.3</td><td>35.6</td></tr><tr><td>Needs patient or local information</td><td>22.2</td><td>23.9</td><td>18.9</td><td>23.6</td><td>23.9</td><td>19.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Potential harm and safest response (%)</td><td>13.2</td><td>13.5</td><td>16.4</td><td>22.6</td><td>15.0</td><td></td></tr><tr><td>High or critical potential harm Safest response not a direct answer 38.8</td><td></td><td>44.1</td><td>42.6</td><td>35.5</td><td>39.8</td><td>6.2 35.2</td></tr></table>

Values are percentages of each family’s annotated queries unless stated otherwise. Kinds of work group the ten task categories: documentation and administration (documentation and workflow; coding and administrative), knowledge retrieval (foundational knowledge; drug information and pharmacotherapy), clinical reasoning (treatment and management; diagnosis and diferential; test and result interpretation; patient education and communication; procedural guidance) and other. Specialties are counted after k ≥ 10 suppression; RN queries were released without specialty. IQR, interquartile range.

Intent varied within tasks (Fig. 2b,c). Most drug-information RCQs sought to look up a fact or concept (63.1%), and nearly all of the rest a dose, a decision or confirmation of a planned action (35.4%). Treatment and diagnosis RCQs were divided between condition-level look-ups and decisions about a particular patient, which made up 52.8% and 42.5% of these RCQs, respectively. Documentation requests mostly concerned a specific patient (68.9%) and nearly always asked for text (95.6%), chiefly clinical notes, care plans and discharge instructions (46.4%), chart summaries (19.4%) and replies to patient-portal messages (6.6%) (Fig. 2a).

We also mapped each RCQ to the seven physician use cases that the American Medical Association (AMA) tracks in its surveys<sup>1</sup>. Summaries of research and standards of care were the most common (37.4%), and assistive diagnosis accounted for 4.1%. Nearly a third of RCQs (30.6%; 13.1% after excluding non-clinical RCQs) matched none of the seven, mainly requests for help with EHR and workflow tasks, written documents outside the listed types and patient-communication content (Fig. 2e). Clinicians agreed poorly when assigning RCQs to these use cases (Krippendorf’s � 0.25, compared with 0.64 for task category; Extended Data Table 2), so the survey list does not divide real requests cleanly. We also assigned each RCQ to the clinical department that would ordinarily manage the condition it concerned, which describes the query rather than the asker’s own department. Conditions managed by the Department of Medicine (internal medicine) accounted for 47.3% of RCQs. Within it, the largest divisions were general internal medicine, which covers primary care and problems that no subspecialty clearly owns (16.5% of Medicine RCQs), hematology and oncology (14.1%), cardiology (12.9%) and infectious diseases (12.0%) (Extended Data Fig. 3).

## a Task category

![](images/ff9759a9941d5b402cf59458820f43b07ee2d81b2a98becbb4151e70f887dfa6.jpg)  
Coding & administrative 1.0%; bil ing and translation 4% of documentation requests

## b Question intent

![](images/12d3f8a365f86f0f39c3908de1a7c53176c1f36e14b48e81e05e55bc3a5024d6.jpg)

c Intent within task category  
![](images/6eb2a43fcf17a342e4047cb2d22f554c55d4f29538ad6dce8fe8a316a1c539e9.jpg)

d Task mix by professional family
<table><tr><td rowspan=1 colspan=6>Attend.  Fellow   Resid.   APP    RN</td></tr><tr><td rowspan=1 colspan=1>Documentation &amp; workflow</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>38</td></tr><tr><td rowspan=1 colspan=1>Foundational knowledge</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1>Drug information</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>Treatment &amp; management</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Diagnosis &amp; differential</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Test &amp; result interpretation</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Patient education</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Coding &amp; administrative</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Procedural guidance</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Other (non-clinical)</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>27</td></tr></table>

e AMA physician AI use cases  
![](images/8d95c53e9330121bfa4223c3a3371f644826283dd48257610d123d53eb6839e7.jpg)

## f Query properties

![](images/dec67a28063e3b365a61b4fe5b02810be89cdef9466f2898770bd1d7656c9769.jpg)  
Fig. 2 | Task and intent distribution of real clinical queries. n = 127,625 annotated queries from 6,342 clinicians. a, Task category, grouped and colored by type of work; tile area is proportional to the share of queries, and categories too small to label are named below their group. The 44,807 documentation and workflow requests are divided by type of document (AMA use case; % of these requests); 95.6% asked for generated text, 68.9% concerned a specific patient and 7.9% were rated high or critical risk. b, Question intent, grouped into five intent groups (Methods). c, Intent group within each task category; bubble area is proportional to the share of the task’s queries in that group (values ≥28% labeled). d, Task mix by professional family (% of each family’s queries; Attend., attending physicians; Resid., residents; RN, registered nurses). e, Queries mapped to the AMA list of physician AI use cases. f, Grammatical form of the query (top three rows) and query properties; filled markers show the share of all queries and hollow markers the mean of per-clinician shares.

## Clinical use difers by role and often crosses specialty boundaries

Task mix difered by professional family (Cramér’s V = 0.14; Fig. 2d and Table 1). Residents devoted the largest shares of their RCQs to knowledge retrieval (34.9%) and clinical reasoning (19.7%) and the smallest to non-clinical use (7.4%), whereas attending physicians and RNs had the largest non-clinical shares (26.2% and 26.8%).

Task composition, risk and response needs also varied across the 35 specialties (Extended Data Fig. 4). Specialists often asked about conditions outside their own discipline. Of 38,647 clinically anchored RCQs from 31 organ- or discipline-defined specialties, 43.9% (95% CI 40.6–47.1%) concerned conditions managed by another clinical department and 56.0% (95% CI 52.9–59.0%) fell outside the asker’s own division. At the department level, this share ranged from about 10% in critical care and child and adolescent psychiatry to over 80% in ophthalmology, rehabilitation, and neurosurgery (Extended Data Fig. 5). These out-of-specialty RCQs were as likely to be high risk as RCQs within the specialty (19.0% versus 18.1%).

## High-stakes RCQs are concentrated and partly identifiable from their text

We rated each RCQ’s potential harm on a five-level scale, assuming that the clinician acted on an incorrect answer without checking it. Overall, 11.6% of RCQs were rated high risk and a further 1.6% critical, 13.2% in all (95% CI 12.4–14.0%; Fig. 3a), although clinicians rated fewer RCQs this highly than the model did (Methods). These RCQs clustered in a few tasks. Drug-information RCQs made up 14.1% of all RCQs but 38.8% of high- or critical-risk RCQs and 50.8% of critical ones, and treatment and management RCQs made up 6.2% of all RCQs but 21.2% of high- or critical-risk ones (Fig. 3b). The share rated high or critical was largest for treatment and management (45.2%), procedural guidance (38.0%), drug information (36.2%) and diagnosis (31.3%), and smallest for foundational knowledge (0.5%) and coding (0.6%). Among intent groups, RCQs that sought to decide or verify something, such as a dose, a clinical decision or a planned action, carried the most risk (50.2%), compared with 3.9–9.5% for the other groups (Fig. 3c). Documentation requests, although mostly patient-specific, were high or critical risk in 7.9% of cases. Risk also difered by role: 22.6% of residents’ RCQs were high or critical, compared with 13.5% of attending physicians’ RCQs and 6.2% of RNs’ RCQs (both P < 0.001 versus attending physicians; Fig. 3a).

After adjustment for task category and professional family, risk rose with features of the query text that a deployed system could detect before answering (Fig. 3d and Extended Data Table 3): reference to a specific patient (adjusted odds ratio (OR) 3.20, 95% CI 2.88–3.55), an acute or urgent situation (OR 3.07, 95% CI 2.52–3.74), a stated dose (OR 1.93, 95% CI 1.70–2.18), clinical abbreviations (OR 1.87, 95% CI 1.76–1.98), a vulnerable population (OR 1.37, 95% CI 1.24–1.52) and a named medication (OR 1.30, 95% CI 1.18–1.44). Requests for generated text (OR 0.58, 95% CI 0.44–0.75) and mentions of imaging (OR 0.76, 95% CI 0.68–0.86) were associated with lower risk. Using 11 features observable in the text and no task label, a cross-validated model grouped by clinician distinguished high- or critical-risk RCQs from the rest with an area under the receiver operating characteristic curve (AUC) of 0.78 (Fig. 3e). A threshold that captured 90% of these RCQs would, however, flag 60% of all RCQs.

## More than a third of RCQs cannot be answered well as posed

More than a third of RCQs (36.1%, 95% CI 34.8–37.5%) could not be answered well as posed. 23.4% of RCQs could be answered only with material assumptions and 12.7% not at all (Fig. 4c). Of all RCQs, 22.2% required patient or local institutional information that was not supplied, such as history, results, formularies, order sets, protocols or billing rules, and 12.6% required current published evidence. The need for patient or local information rose steeply with potential harm, from 4.3% of minimal-risk RCQs to 72.7% of critical-risk RCQs (Fig. 4b). Of high- or critical-risk RCQs, 66.8% could not be answered well as posed, compared with 31.4% of lower-risk RCQs.

A direct answer, meaning an immediate reply from the model’s own knowledge, was the safest response for 61.2% (95% CI 59.7–62.6%) of RCQs. The remainder called for a clarifying question to the clinician (26.8%) or an answer grounded in retrieved authoritative sources (12.0%); escalation to a human expert (32 RCQs) and abstention (50 RCQs) were almost never the safest response (Fig. 4a). A direct answer was safest for 71.8% of minimal-risk RCQs but for only 7.0% of critical-risk RCQs. Overall, 84.5% of high- or critical-risk RCQs required something other than a direct answer, most often clarification (60.0%), compared with 31.9% of lower-risk RCQs. Routing needs difered by task (Fig. 4d). Coding and administrative RCQs were the least suited to a direct answer (88.9% not direct), because 80.4% of them required patient or local information that a model trained on public data does not have (Fig. 4f). Most treatment (73.9%) and test-interpretation (63.8%) RCQs also required more than a direct answer, whereas few foundational-knowledge RCQs did (8.5%). Answerability and the safest response were closely linked: RCQs that could be answered fully as posed were best met by a direct or retrieval-grounded answer, whereas 46.7% of RCQs for which clarification was safest (15,961 of 34,190) could not be answered as posed at all (Fig. 4e), most of them (11,531) first turns with no recoverable request.

## Public benchmarks sample a diferent workload from real clinical use

The labels come from an LLM annotator (GPT-5.6 sol) that we validated against seven clinicians on 100 RCQs. Across 23 fields it matched a held-out clinician reference as often as the clinicians themselves did (84.6% versus 83.0%), and it rated potential harm more cautiously than the clinician majority (Methods, Extended Data Figs. 6 and 7 and Extended Data Table 2).

Finally, we applied RCQ-Map to the benchmarks used to evaluate clinical AI to ask whether they reflect this workload, annotating each with the same validated annotator, guidelines and schema (Methods). We assembled 58 public benchmarks from 13 major evaluation suites and from frontier model reports into the Clinical AI Benchmark Atlas (CABA; Methods and Supplementary Data 1) and annotated up to 200 items from each (10,659 items; Fig. 5a). Individually, the benchmarks shared a median of 31% of the RCQ task mix (interquartile range (IQR) 21–39%; Fig. 5a), less than the 55% shared by a distribution spread evenly across the ten task categories. The median benchmark contained no documentation or administrative requests (IQR 0–3%), which made up 36.2% of RCQs, and only 13 of the 58 benchmarks reached 5%. The 12 benchmarks used in frontier model reports shared a median of 40% (IQR 36–50%), and diagnosis exceeded its 3.7% share of real use in 9 of them. The 29 benchmarks in suites meant to resemble clinical practice (MedHELM, BRIDGE, ClinicBench and HealthBench) were no closer to real use, at a median of 35% (IQR 26–39%; diference -5.5 percentage points, 95% CI -21.2 to 0.3; Fig. 5b). To see where these diferences arise, we examined five benchmarks in detail (Extended Data Fig. 8 and Supplementary Table 2). Two are built from examination-style items: MedQA (1,273 questions)<sup>34</sup> and the script concordance test of the MAST suite (174 items)<sup>35,36</sup>. Three are built from questions written by clinicians: HealthBench Professional (525 physician conversations)<sup>18</sup>, Real-POCQi (620 point-of-care questions submitted to a clinical AI tool)<sup>17</sup> and NOHARM from the MAST suite (330 specialist eConsult cases and their variants)<sup>37</sup>. None matched real use. The total variation distance between a benchmark’s task distribution and that of RCQs ranged from 0.25 for HealthBench Professional to 0.83 for the script concordance test, on a scale from 0 (identical) to 1 (no overlap), and distances for intent were larger (Extended Data Fig. 8e). Task categories that a benchmark did not contain at all accounted for 36% of real use in the case of MedQA and 76% in the case of NOHARM.

Examination benchmarks sampled a narrow slice of use. Diagnosis made up 48.0% of MedQA and 56.3% of the script concordance test, compared with 3.7% of RCQs, and documentation requests, 36.2% of real use, were essentially absent (0.1% and 0.0%; Extended Data Fig. 8a,b). Their items were complete by design: 0.7% of MedQA items needed patient or local information that they did not contain, compared with 22.2% of RCQs, and a direct answer was the safest response for 97.6% of MedQA items and 97.7% of script

a Potential harm by professional family  
![](images/fe39efad5a165cb923d294e2834516b7b3c0fe96db52bf6332671ab03ba59714.jpg)

b Volume and risk by task category  
![](images/96c10ae0d790c85ebb76ed2de7894946df80bffdeacd19a274805a687cc7bf54.jpg)  
d Text features and high or critical risk

c High or critical risk by intent group  
![](images/1e67dca2e838d9d8870b9c2bda4b7ceefc8e1610e1b3e7a069348eacaacf09ae.jpg)  
e Prediction from text features

![](images/3659dbc11cbc3076d6a030e9e1b88305dead9a9528fb9d3f2451f29c4924a640.jpg)

![](images/013b773de3d6612787a0e0447ef936112e5b04efe1cae6dcbe22a234d90a9de4.jpg)  
Fig. 3 | Potential harm if a query were answered incorrectly. a, Distribution of harm grades for all queries and by professional family; values are percentages of each row, and the right-hand column gives the share rated high or critical. b, For each task category, its share of all queries (horizontal axis, log scale) and the share of its queries rated high or critical (vertical axis); bubble area, its share of all high- or critical-risk queries; dashed line, all queries. $\mathbf { c } ,$ Share of queries rated high or critical by intent group (95% CIs, bootstrap over clinicians); dashed line, all queries. d, Adjusted odds ratios for high or critical risk from a logistic regression including the features shown, task category and professional family; error bars, 95% CIs from standard errors clustered by clinician; orange, OR significantly above 1; blue, below 1; gray, CI includes 1. e, Receiver operating characteristic curve for a logistic model using 11 text-observable features (the features in d, without task or professional family) in five-fold cross-validation grouped by clinician; the marked point is the threshold that captures 90% of high- or critical-risk queries. $\mathtt { n } = 1 2 7 , 6 2 5$ queries.

a Safest response by harm grade  
![](images/11e1b86cefdf26f7bc999ec3c70d941f08eca493560932ac35ce6c66042893cd.jpg)  
b Missing information by harm grade

![](images/98d9ec85400882bce25b26a03fd8d9f3002fb0f574337aef95964332f67f32e2.jpg)  
c Answerability by task category

d Safest response by task category  
![](images/d51c9ee197bac6a4e1058f145cdde16fb25e0f658fb35185c72456484da1b899.jpg)  
e Answerability and safest response

![](images/4b361ceeecf82f5724959b2dccfd5a81e775b76396703991ca57c2740a1698f3.jpg)

![](images/4e8bc748b6a06c311867f1e46b18d9662ae8f5f298a195c0695aed91337566b5.jpg)  
f Patient or local information by task

![](images/88ff7c8881c46a37b333b24742956b24492b009fdbb5a2714ee835b7bf122c2e.jpg)

Fig. 4 | Context requirements and the safest response. a, What a safe assistant should do: the safest response for all queries and by harm grade (numbers of queries in parentheses), that is, the least restrictive of answering directly from general knowledge, retrieving authoritative sources and then answering, asking the clinician to clarify, or deferring (escalating to a person or abstaining) that remains safe; values are percentages of each row. Deferral accounted for 0.06% of queries. b, Share of queries that could not be answered well as posed (partial or low answerability), that needed patient or local information, or that needed current evidence, by harm grade. c, Answerability by task category: share of each task’s queries that could be answered only with material assumptions (partial) or not at all (low). d, Safest response by task category (% of the task’s queries; bars sum to 100% including escalation and abstention). e, Answerability as asked versus safest response (numbers of queries; deferred queries omitted). f, Share of queries needing patient or local information in the five task categories with the highest shares. n = 127,625 queries.

![](images/17bb0061980cf36a2a0182225cd8e741e2eed8a8fdebbd8d740bf2746e558469.jpg)

b Each benchmark compared with real use  
![](images/cca7f24edbd554dd7f3230be9ed5907f662147011343953b6c36b92679993ba0.jpg)

![](images/3596acd14b62db0bf35799f711db14b933aa57428ca0680a9caf36e02e787889.jpg)  
Fig. 5 | Task composition of public clinical AI benchmarks compared with real clinical use. Real clinical queries (RCQs; $\mathbf { n } = 1 2 7 , 6 2 5 )$ and the 58 benchmarks in the Clinical AI Benchmark Atlas (CABA; Methods), all annotated with RCQ-Map by the same annotator (GPT-5.6 sol, reasoning efort none) with the same guidelines and schema; up to 200 items were sampled from each benchmark. a, Kind of work requested in RCQs (left) and in each benchmark (% of items), with task categories grouped as in Fig. 2a. Benchmarks are grouped by whether they are used in frontier model reports, belong to a suite meant to resemble clinical practice (MedHELM, BRIDGE, ClinicBench or HealthBench), both or neither, and are ordered by their most common kind of work; above, share of the RCQ task mix covered by each benchmark (one minus the total variation distance). b, Share of the RCQ task mix covered by each benchmark (left) and share of its items that are documentation or administrative requests (right), for the 12 benchmarks used in frontier model reports, the 29 benchmarks in suites meant to resemble clinical practice and all 58. Each dot is a benchmark and black bars mark medians; dashed lines mark a distribution spread evenly across task categories (left) and RCQs (right). Full distributions are given in Supplementary Table 2 and Supplementary Data 1.

concordance items, compared with 61.2% of RCQs (Extended Data Fig. 8c,d). These benchmarks can show whether a model answers correctly, but not whether it recognizes when to retrieve, clarify or defer.

Benchmarks built from clinicians’ questions came closer to real use in what a safe response required but concentrated on consultative clinical questions. Knowledge retrieval and clinical reasoning made up 97.7% of Real-POCQi and 100.0% of NOHARM, and documentation 0.5% and 0.0%, respectively; Real-POCQi excluded documentation, translation and summarization requests by design<sup>17</sup>. High-stakes items were over-represented: 34.5% of Real-POCQi items, 42.7% of HealthBench Professional items and 54.5% of NOHARM items were high or critical risk, compared with 13.2% of RCQs, and 66.9%, 78.1% and 87.9%, respectively, were safest met by something other than a direct answer, compared with 38.8% (Extended Data Fig. 8c,d). HealthBench Professional, which sampled consultation, research and writing by design<sup>18</sup>, was the closest to real use on task and intent and the only benchmark that contained every task category; documentation made up 25.5% of its items, most of them from its writing use case.

## Discussion

Our study demonstrates a discrepancy between current medical LLM benchmarks and real-world deployment. To our knowledge, this is the largest characterization of how clinicians use large language models in practice. Across 127,833 RCQs, documentation and knowledge retrieval made up nearly two-thirds of use, and diagnosis, the subject of about one in five published evaluations<sup>5</sup> and about half of the items in the two examination benchmarks, made up 3.7%. Applying RCQ-Map to 58 public clinical AI benchmarks showed the mismatch directly: the average benchmark shared 31% of the task mix of real use and contained no documentation requests, no single benchmark’s task distribution approached that of real use, and examination benchmarks were dominated by diagnostic tasks. Selecting hard cases makes a benchmark more discriminating, but its scores then describe performance under conditions that are rarer in practice and cannot be read as the error rate a deployed system would have on the queries it receives. Published evaluations almost always score whether an answer is correct<sup>5</sup>, yet more than a third of RCQs, and two-thirds of those with high or critical potential harm, could not be answered well as posed. Our task distribution resembles those reported for other institutional deployments<sup>23,24</sup>, including Stanford’s secure LLM assistant, used mainly for writing and knowledge tasks<sup>22</sup>, and its EHR-integrated ChatEHR, used mainly for summaries and record review<sup>25</sup>. We release RCQ-Map, with its guidelines, schema and annotation code, and CABA, with item-level labels for its 58 benchmarks, so that health systems can profile their own use, and benchmark developers can compare their items with real clinical use (See Data and Code availability).

Fundamentally, benchmarking should be relevant to the work clinicians bring to deployed systems. Future benchmarks and prospective studies should draw tasks in proportion to clinical use, and stratify them by the features that carry risk. Documentation itself deserves particular attention due to its prevalence. A third of RCQs (33.6%) asked the assistant to draft clinical or operational documents, such as notes, discharge instructions, chart summaries and portal replies, and most concerned an identifiable patient. These requests were rarely high risk as judged from the query alone, but they place protected health information in prompts and produce text that may enter the medical record. They reflect the documentation burden measured in time-and-motion studies of physicians and nurses<sup>38–40</sup> which broadly can be seen in the rapid uptake of ambient scribes and inbox drafting<sup>3,4</sup>.

Our findings also have relevance to the design of the systems and harnesses that support medical AI models. More than a quarter of RCQs were safest when met with a clarifying question, yet LLM assistants typically answer every query as posed and rarely ask for information they lack<sup>31</sup>. Examination benchmarks cannot measure this behavior, because almost all of their items are safest answered directly (97.6% of MedQA and 97.7% of script concordance items; Extended Data Fig. 8c,d). Patient or local information was missing from 22.2% of RCQs and from most coding and administrative questions, which a model likely cannot answer safely without retrieval over local policies, formularies and order sets<sup>29,30</sup>. Escalation and abstention, by contrast, were almost never the safest response, so blanket refusal policies protect little. Most of the achievable safety gains likely lie in clarification and grounding<sup>32,41,42</sup>.

RCQ-map also shows that safeguards and guardrails may benefit from targeted interventions. About one in eight RCQs was high or critical risk under the model’s cautious labeling, and about one in 27 under clinician consensus, clustered in drug information, treatment and procedural guidance and in queries that sought to decide or verify something. Verification queries deserve attention because a clinician asking an LLM to confirm a plan is presumably about to act<sup>43</sup>. Interestingly, more than two in five clinically anchored RCQs from specialists concerned conditions managed by another department.

Measurement of risk itself is challenging. Clinicians agreed reliably on what an RCQ contained and on the task it requested, but only modestly on how risky it was or what context it needed, at levels similar to those reported for classifying generic clinical questions<sup>44</sup> and for the AHRQ harm scale<sup>45</sup>; individual clinicians rated between 0% and 26% of comparable RCQs as high risk. Agreement was lowest for the AMA’s use cases (� = 0.25), so surveys built on such lists may not capture what clinicians actually ask of these tools. Because RCQ-Map develops de-identified output, health systems could also use it to monitor clinical AI use continuously and, as shown here, to check whether evaluation sets reflect their particular usage patterns.

Lastly, our study of existing benchmarks themselves has significant implications for benchmark builders. Diagnostic tasks have been the focus of most development eforts<sup>5,13,32</sup>, but diagnosis made up 3.7% of RCQs in our deployment, and again, specialists often asked about conditions outside their own field: 85% of ophthalmologists’ and 81% of neurosurgeons’ clinically anchored RCQs concerned conditions managed by another department. A model built for one specialty’s diagnostic questions would therefore miss much of what its users ask in a deployment like this one - suggesting a limitation of specialty specific tooling.

This study has limitations. It covers one academic health system and one assistant, and the mix of tasks will difer with the specialty mix, workforce and design of other tools; the department assignments in RCQ-Map follow this institution’s organization. We could not observe clinicians’ use of other tools, including web search, reference tools, clinical decision-support products and personal accounts, so the task mix reported here may reflect which questions clinicians chose to bring to this tool and not the full range of their AI use. We annotated the first clinician turn of each conversation, which misses requests that appear later in a dialogue (9.0% of first turns contained no recoverable request). We did not examine the assistant’s responses, so we measured potential rather than actual harm. Corpus and benchmark labels were assigned by an LLM validated on RCQs rather than on benchmark items, and benchmark items were reduced to a single request. The evaluation suites in CABA were selected by the authors and the five benchmarks examined in detail were a purposive selection, and CABA includes only public, English, text-only benchmarks, so private or newer benchmarks may give a diferent picture. Clinicians agreed only modestly on potential harm, missing information and the safest response, so absolute prevalences for these fields are less certain than comparisons between tasks or groups. Finally, the most active clinicians generated much of the volume.

Clinicians are already using AI to document care, look up information, and make decisions. Whether a model can pass a medical examination or any static benchmark now matters less than whether deployed systems handle the work clinicians bring. RCQ-Map and the analysis of one large deployment reported here make that measurable on the queries clinicians actually send, and highlight the divergence between real world use and existing benchmarking eforts. Perhaps most importantly, these results suggest that evaluation may benefit from being deployment-specific, reflecting specific users, specialties, task distributions, and most of all patients to ensure that model responses meet the needs of real clinical use.

## Methods

## Study setting and data source

This retrospective observational study analyzed conversations between clinicians and a secure LLM assistant deployed across a large, multi-hospital urban academic health system. The assistant was available to all staf through institutional sign-on, and clinicians typed or pasted free text. We analyzed conversations from 12 August 2025 to 3 April 2026, with a gap in data capture from 6 to 9 January 2026. The analytic cohort comprised conversations initiated by attending physicians, physician fellows, residents, APPs (nurse practitioners and physician assistants) and RNs, identified by linkage to Epic Clarity records, which also supplied each clinician’s professional family and specialty. Podiatrists and dentists were excluded from the analytic cohort. Each conversation was attributed to one resolved provider record, and each clinician was assigned a single professional family and specialty for the whole study period, defined as those held for the majority of days.

## Ethics, privacy and data release

Under institutional policy, this study was classified as non-human-subjects research and did not require institutional review board review, because query text was removed by the health system’s pipeline and investigators received only de-identified, label-level data. Query text was processed only by the health system’s IT team in approved environments; investigators received a label-only extract containing a conversation identifier, a pseudonymous clinician identifier, professional family, specialty, the 24 annotation fields and annotation status, and no query text, message content or model output. To prevent re-identification from rare combinations of role and specialty, a specialty label was released only if the corresponding professional family × specialty cell contained at least ten clinicians in the workforce (k-anonymity with ${ \bf k } = 1 0 ) ^ { 4 6 , 4 7 }$ . Cells with 1–9 clinicians were pooled into a within-family ‘Other’ specialty when the pooled cells reached ten, and removed otherwise; thresholds were derived from workforce denominators rather than from the data. This recoded 977 conversations (35 clinicians) to ‘Other’ and removed 3 conversations (3 clinicians). RNs were released without specialty, and physician or APP records whose specialty field was blank or restated the profession were released as ‘Missing’ (698 conversations from 32 clinicians). The final corpus contained 127,833 conversations from 6,342 clinicians.

## Query extraction

Conversations were exported as encoded message histories, decoded and ordered by turn. The real clinical query $\left( \mathrm { R C Q } \right) ^ { 4 8 }$ was the text of the first clinician turn, truncated at 100,000 characters. Conversations with no clinician text were not annotated (n = 121), and 87 conversations returned output that failed schema o consistency validation after retries, leaving 127,625 annotated queries (99.8%).

## Development of RCQ-Map

RCQ-Map was developed by the study team from established frameworks in clinical-question research, LLM evaluation, patient safety and selective prediction (Extended Data Table 1). The first version defined nine task categories and eleven question intents together with the risk, context and safest-response fields. Later versions of the annotation guidelines added an ‘Other’ value to task and intent; the clinical department, Department of Medicine division, AMA use-case and query-form fields; boundary rules for task (for example, a request to draft a document is documentation whatever its topic, and a question about a drug regimen is drug information even when phrased as treatment) and tie-break rules for intent (an explicit confirmation cue makes a query verification; a question whose answer is a dose is dosing); a precedence ladder for clinical department, whose rule for drug questions was revised after feedback from two clinician annotators; three hard consistency rules; and two worked examples. The annotation guidelines specify 24 forced-choice fields. After the validation study, we dropped two query properties that clinicians applied poorly and that no analysis required (actionability, � 0.39, and evidence dependence, � 0.29, which overlapped with the need for current evidence), combined the patient and institutional context fields and no longer report the derived any-context field, leaving 20 fields in six blocks (Fig. 1b and Extended Data Table 1). Agreement for every annotated field is given in Extended Data Table 2. The guidelines require each field to be judged independently from the query text alone, without assuming facts that are not stated; surface features are coded literally; and a query with no recoverable request is coded as ‘Other’ task and intent, low answerability and clarification. The same guideline text served as the clinicians’ reference and as the model’s instructions, and the two were verified to be byte-identical before every model run. The full text, including every definition, boundary rule and tie-break that the annotator applies to choose a category, is reproduced in Supplementary Note 1, and the request format and output schema are given in Supplementary Note 2.

## Assessment of the task and intent categories

Because the task and intent categories were synthesized from several existing schemes rather than adopted from a single one, we assessed them in three ways. First, we mapped each category to its counterparts in established frameworks (Supplementary Table 1). Six task categories (drug information, treatment and management, diagnosis, test and result interpretation, procedural guidance and foundational knowledge) and most intents correspond to question types in pre-LLM clinical-question taxonomies, in which drug choice, the cause of a symptom and test selection were the commonest generic questions<sup>44,49,50</sup>. The remaining categories (documentation and workflow, patient education and communication, coding and administrative, and the documentation-drafting and coding intents) correspond to the text-generation, communication and administrative tasks that frameworks for LLM evaluation and studies of LLM use added<sup>16,22,26</sup>. Second, clinicians applied the categories reliably: Krippendorf’s � was 0.64 (95% CI 0.55–0.73) for task and 0.45 (0.36–0.54) for intent (Extended Data Table 2), compared with � = 0.53 when 11 coders classified clinical questions into Ely’s 64 generic question types<sup>44</sup>. Third, the categories separated queries with very diferent potential harm and safest response (Figs. 3b,c and 4d), the property RCQ-Map needs to serve as a stratification frame for evaluation.

## Correspondence of RCQ-Map to prior literature

The risk, context and safest-response blocks adapt established frameworks, and the task and intent blocks correspond to earlier classifications of clinical questions, extended where generative AI introduces work that earlier frameworks did not anticipate.

Task category. Classification by the deliverable the clinician wants corresponds to the generic clinicalquestion taxonomies developed from point-of-care questions, in which drug choice and dosing, the causes of symptoms and findings, test selection and management predominate<sup>44,49,51,52</sup>. Keeping drug information and pharmacotherapy separate from treatment and management is consistent with these literatures, which identify drug questions as the single largest and most distinct class, and diagnosis, test and result interpretation and procedural guidance each map to a diferent knowledge source and error mode<sup>44,50,53</sup>. Pre-LLM taxonomies contained no category for generating text; documentation and workflow, patient education and communication and coding and administrative tasks, which correspond to the note-generation, patient-communication and administration categories of the clinician-validated MedHELM taxonomy<sup>16</sup> and to the writing tasks that dominate institutional and general-population LLM use<sup>22,26,27</sup>. Because nurses and APPs raise diferent questions from physicians<sup>54,55</sup>, categories were defined by task rather than by profession. An ‘Other’ category captured non-clinical requests such as research writing and personal queries.

Question intent. Consistent with the distinction in information-retrieval research between what a query is about and what the user is trying to achieve<sup>56</sup>, intent records the motive behind the task: to verify a belief or planned action, look up a fact, reach a clinical decision, obtain a dose, draft text, understand a definition or mechanism, compare options, follow a procedure, obtain a code or interpret a result. These categories mirror the generic question stems of Ely et al. (for example, ‘What is the dose of drug x?’ and ‘How should I manage condition x?’)<sup>44</sup> and the decision-oriented framing of evidence-based practice<sup>57</sup>. Verification was defined separately because a clinician seeking confirmation has already formed a plan, and the decision to pursue a question is driven by its urgency and link to action<sup>43</sup>. For comparisons, we grouped the 12 intents into five: decide or verify (clinical decision, verification, dosing and result interpretation), look up or understand (fact, definition, mechanism and comparison), draft text or code (documentation drafting and coding), follow a procedure, and other. Clinicians agreed more on the groups than on single intents (� 0.61 versus 0.45, and 0.58–0.63 under alternative groupings).

Clinical department and division. Rather than organ systems, clinical domain was assigned to the clinical department that would ordinarily own management of the problem, using a precedence ladder (for example, psychiatric content before pediatric, operative content to the operating department, drug questions to the department owning the treated condition), with a second field resolving the Department of Medicine into its divisions. Anchoring domains to accountable clinical owners allows query patterns to be directed to the services responsible for content governance, safety review and local guidance.

AMA use case. To relate observed demand to the use cases that professional bodies track, each query was mapped to the AMA’s list of physician AI use cases, as used in its physician surveys<sup>1</sup>, with ‘None of the above’ added because the list is not exhaustive.

Query properties. Patient specificity distinguishes questions about an identifiable patient from general questions, a distinction central to information-needs research, in which unmet needs were predominantly patient-specific<sup>49,53</sup>.

Context required. Three binary fields identify what a safe answer needs that the query does not contain: patient-level information (history, medications, laboratory results), which clinicians most often lacked in observational studies<sup>53</sup> and which EHR-grounded benchmarks supply<sup>58</sup>; institutional knowledge (protocols, order sets, formularies, antibiograms, payer rules), which is local by definition; and current published evidence, reflecting the finding that answers are often absent from or out of date in the resources clinicians consult<sup>59</sup> and motivating retrieval augmentation<sup>29,30</sup>. Because clinicians rarely agreed on institutional knowledge as a separate field (� 0.25), we report patient-level and institutional information together, as patient or local information that the query does not contain.

Potential harm. Risk was defined as the potential harm if the query were answered incorrectly and the clinician acted on the answer, not as the probability that the model errs. The five grades (minimal, low, moderate, high, critical) adapt the severity levels of the AHRQ Common Formats Harm Scale (https://www.psoppc.org), which has been used to rate the harm potential of LLM answers<sup>45,60</sup>, together with the harm categories of the NCC MERP Index for Categorizing Medication Errors (https://www.nccmerp.org/types-medication-errors) and the WHO Conceptual Framework for the International Classification for Patient Safety (https://www.who.int/ publications/i/item/WHO-IER-PSP-2010.2). Anchors for the critical grade follow the ISMP list of high-alert medications (https://home.ecri.org/blogs/ismp-resources/high-alert-medications-in-acute-care-settings), pediatric weight-based dosing<sup>61</sup>, medication safety in older adults<sup>62</sup>, teratogenic exposure and time-critical emergencies. Purely non-clinical harms (financial, legal, privacy) were capped at moderate.

Answerability and safest response. Answerability records whether the query can be answered well as posed (high), only with material assumptions (partial) or not at all (low). The safest response (the ‘route’ field in the annotation guidelines) operationalizes the selective-prediction and learning-to-defer literatures for a clinical assistant<sup>41,42,63</sup>: answer directly; answer grounded in retrieved authoritative sources<sup>29,30,64</sup>; ask the clinician for missing details<sup>31</sup>; escalate to a qualified human such as a specialist or pharmacist<sup>32</sup>; or abstain<sup>65</sup>. We refer to escalation and abstention together as deferral. Responses are ordered by restrictiveness, and annotators select the least restrictive response that remains safe, with clarification taking precedence over retrieval when patient information is missing.

Surface features. Nine literal features (medication, dose, laboratory or test result, imaging, vulnerable population, acute or urgent situation, request for text generation, grammatical form and use of clinical abbreviations) were coded independently of all judgment fields so that their association with risk and the safest response could be tested. Medication, dose and vulnerable-population cues reflect established drivers of medication harm<sup>61,62</sup>; query form (question, command or fragment) follows query-log research showing that information-system queries are frequently keyword fragments rather than questions<sup>66,67</sup>; and abbreviations were coded because clinical shorthand is highly ambiguous<sup>68</sup>.

## LLM annotation of the corpus

The corpus was annotated with GPT-5.6 sol (OpenAI; model identifier gpt-5.6-sol; https://deploymentsafety .openai.com/gpt-5-6) through the OpenAI Responses API with reasoning efort set to ‘none’. The annotation guidelines were supplied verbatim as the model’s instructions (the system prompt) and each query as a single user message (‘Question: <query text>’), with no chart data, user identity or conversation history (Supplementary Note 2). Output was constrained by a strict JSON schema enumerating the allowed values of all 24 fields, and every response was validated against the schema and the three hard consistency rules; invalid outputs and transient errors were retried with exponential backof. LLM annotation of text corpora has been shown to match or exceed crowd annotation<sup>69</sup>; because agreement with clinicians cannot be assumed in a clinical setting<sup>17,70</sup>, we validated it directly.

## Clinician validation study

A validation cohort of 100 queries was drawn at random (seed 42; Mulberry32 generator with a Fisher–Yates shufle) from a de-identified extract of queries from the same deployment. Seven clinicians annotated the queries on a purpose-built web platform that displayed the full annotation guidelines with inline field rules, enforced the hard consistency rules, derived the any-context field automatically and required a value for every field. Queries were assigned breadth first so that each received exactly three independent annotations; each clinician received 40 queries initially and could request additional batches of 10 (range 30–60 per clinician). Grading clinicians could not see other clinicians’ or information about the query sender. All 300 annotations were completed between 18 and 27 August 2026.

The same 100 queries were annotated on 24 August 2026 with the corpus annotator configuration: GPT-5.6 sol with reasoning efort ‘none’ and the identical prompt and schema.

## Five benchmarks examined in detail

To examine in detail where benchmarks depart from real use, we applied RCQ-Map to all released items of five public benchmarks downloaded from their oficial releases on 30 September 2026 (Extended Data Fig. 8 and Supplementary Table 2): the MedQA test set of United States Medical Licensing Examination-style questions (four-option version; 1,273 items)<sup>34</sup>; HealthBench Professional (525 conversations)<sup>18</sup>; Real-POCQi (620 questions)<sup>17</sup>; and the two openly released subsets of MAST<sup>35</sup>, NOHARM (330 items: 30 specialist eConsult cases, each with ten perturbed variants)<sup>37</sup> and a script concordance test (174 items)<sup>36</sup>. We selected these benchmarks purposively rather than by systematic search, to cover the two main ways clinical benchmarks are built: examination-style items, including MedQA as the most widely used<sup>11</sup>, and questions or cases written by clinicians. The other MAST components are held out or require registration and were not included. Each item was presented to the annotator as the benchmark presents it to a model, reduced to a single request to match the unit of analysis for RCQs: the question stem followed by the lettered options for MedQA; the first user message for HealthBench Professional (115 of whose conversations continued beyond it); the question text for Real-POCQi; the case prompt for NOHARM; and the kit’s test prompt (instructions, scenario, hypothesis and new information) for the script concordance test. Annotation used the corpus annotator configuration, guidelines, schema, validation and retry rules without modification. We summarized each source by its distributions of task, intent, potential harm and safest response and measured its distance from real use as the total variation distance between its distribution and that of the RCQ corpus, that is, half the sum of absolute diferences across categories, which is 0 for identical distributions and 1 for distributions with no overlap. Confidence intervals for benchmarks are percentile bootstraps over items (2,000 resamples), or over base cases for NOHARM. Four of these benchmarks (all but Real-POCQi) are also in the Clinical AI Benchmark Atlas (below), where they are represented by samples of up to 200 items.

## Clinical AI Benchmark Atlas

We assembled the benchmarks used to evaluate LLMs in health care from two sources. First, 13 public evaluation suites for medical LLMs, selected by the authors: MedHELM<sup>16</sup>, MultiMedQA<sup>60</sup>, MedS-Bench<sup>71</sup>, ClinicBench<sup>72</sup>, MIRAGE<sup>64</sup>, Medmarks<sup>73</sup>, BRIDGE<sup>74</sup>, MEDIC<sup>75</sup>, MMedBench<sup>76</sup>, CLUE<sup>77</sup>, MedBench<sup>78</sup>, HealthBench<sup>15</sup> and MAST<sup>35</sup>. Each suite’s component benchmarks were taken from its paper and code repository on 30 September 2026. Second, frontier model reports (technical reports and system or model cards of frontier models), selected from Epoch AI’s Notable AI Models database (downloaded 4 October 2026)<sup>79</sup>: every model with a language domain published between 1 January 2023 and 30 September 2026 by OpenAI, Anthropic, Google, Meta, Microsoft, xAI, Alibaba, DeepSeek, Mistral, Moonshot, Zhipu, MiniMax or NVIDIA. These 209 models link to 174 documents, of which 169 could be retrieved; the other five links led to a login page, a redirect, a single post or a withdrawn preprint. We read each document, together with any system card, model card or technical report from the same developer that it linked to, and recorded every health benchmark for which results were reported, with the passage reporting it. Twenty-three documents, covering 31 models, reported results on 22 health benchmarks.

Benchmark names from both sources were merged into a single list: alternative names of the same benchmark were combined, and subsets, translations and other variants were merged into their parent benchmark. A benchmark was included if (1) its evaluation items could be downloaded without credentialed access, an application, a data use agreement or payment (free registration was allowed); (2) its items, or an English version released by its authors, were in English; (3) its items could be answered from text alone (for mixed-modality benchmarks, the text-only items were used); (4) it was built for health care rather than drawn from a general-domain benchmark (MMLU, MMLU-Pro, GPQA, SuperGPQA, Humanity’s Last Exam, TruthfulQA, C-Eval or CMMLU); and (5) its items were single requests, that is, questions, instructions, vignettes or conversations to which a model responds. Agent environments and token-level labeling tasks were recorded but not annotated. Criteria 1–3 and 5 were assessed on the downloaded items. Of 175 benchmarks named across the sources, 102 were excluded before download (42 not public, 40 not in English, 8 not answerable from text, 8 not built for health care and 4 not single requests) and 15 after download (8 not public, 4 not single requests, 2 versions of MedQA that were merged into it and 1 drawn from a general-domain benchmark), leaving 58 (Supplementary Data 1). For each benchmark we recorded the source, version (commit or dataset revision), file checksums, license and number of items.

From each benchmark, up to 200 items were sampled uniformly at random without replacement (seed 20261004) from its test or evaluation split, after removing items not in English, items that refer to an image that is not released, and duplicates; smaller benchmarks were used in full. Language was identified with lingua (version 2.1.1), and an item was treated as non-English when its top language was not English with a confidence of at least 0.80. Each item was rendered as a model under evaluation receives it: the question with its answer options; the vignette with its question; the instruction with its input text; the benchmark’s or suite’s published prompt template where one defines the task; or, for conversations, the first user turn, mirroring the first clinician turn annotated in the RCQ corpus. One benchmark whose published template leaves its slots unspecified could not be rendered as released and was excluded as not a single request.

RCQ-Map was applied to the 10,699 sampled items with the corpus annotator configuration (GPT-5.6 sol, reasoning efort none), guidelines, schema and retry rules. Of these, 10,659 (99.6%) were labeled; the other 40, all from Med-HALT, failed the schema’s consistency rules on every attempt. We summarized each benchmark separately and report medians and interquartile ranges across benchmarks for the 12 benchmarks used in frontier model reports, for the 29 benchmarks in the four suites whose papers state the aim of reflecting real-world clinical practice (MedHELM, BRIDGE, ClinicBench and HealthBench; MedBench, which states the same aim, contributed no eligible benchmark) and for all 58. Agreement with real use is reported as the share of the RCQ task mix that a benchmark covers, that is, one minus the total variation distance between its task distribution and that of the RCQ corpus; a distribution spread evenly across the ten task categories would cover 55%. The 95% confidence interval for the diference between the two groups’ medians is from a percentile bootstrap that resampled benchmarks within each group (4,000 replicates).

## Agreement statistics

Inter-rater reliability among clinicians was quantified for the 23 judged fields (the derived any-context field was excluded) with Krippendorf’s $\alpha ,$ which accommodates the incomplete rater design, using the nominal metric for categorical fields and the ordinal metric for harm, answerability and safest response<sup>80,81</sup>. Because chance-corrected coeficients are depressed for rare categories despite high raw agreement<sup>82</sup>, we also report Gwet’s $\mathrm { A C l } ^ { 8 3 }$ , Fleiss’ $\kappa ^ { 8 4 }$ , mean pairwise agreement and the share of unanimous queries. Following Krippendorf, $\alpha \ge 0 . 8 0 0$ indicates reliable and $\alpha \geq 0 . 6 6 7$ tentatively acceptable coding; other benchmarks are more lenient<sup>85</sup>. Confidence intervals were obtained by bootstrapping queries (2,000 replicates). We also computed these statistics for the intent groups, the combined patient or local information field and two binary summaries, answerability below high and high or critical potential harm (Extended Data Table 2).

To compare the model with clinicians on equal terms, we used a held-out design. Because each query was annotated by a diferent subset of three clinicians, the held-out clinician rotated: for each query and each of its three clinicians, the reference was the label shared by the other two clinicians when they agreed; the held-out clinician’s agreement with this reference was compared with the model’s agreement with the same reference, and the diference was bootstrapped over queries (1,000 replicates). The model was never treated as an additional rater in reliability calculations. For ordinal fields we computed quadratic-weighted $\kappa ^ { 8 6 }$ over all pairs of model and clinician ratings and all pairs of clinician ratings on the same queries. Consensus labels were defined as the majority label (at least two of three clinicians) or, for ordinal fields without a majority, the median grade; validation prevalences are reported with Wilson intervals<sup>87</sup>.

To translate validation findings into corpus estimates, we estimated for each model-assigned grade the proportion of validation queries that the clinician majority placed in the target category (for example, high or critical risk) and applied these proportions to the corpus distribution of model-assigned grades. Confidence intervals were obtained by bootstrapping validation queries (2,000 replicates).

## Validation results

Seven clinicians independently annotated a random sample of 100 RCQs with RCQ-Map, using the same annotation guidelines in a blinded, forced-choice interface, with three clinicians per RCQ (Fig. 1c). Clinicians agreed closely on literal surface features (Krippendorf’s � 0.55–0.90; Gwet’s AC1 0.75–0.98) and on task category (� 0.64, 95% CI 0.55–0.73). They agreed less on the context an RCQ needed (� 0.25–0.32), the safest response (0.40) and especially potential harm (ordinal � 0.36, 95% CI 0.23–0.48); all three clinicians chose the same harm grade for only 12 of the 100 RCQs and grades within one level of each other for 61 (Extended Data Fig. 6a and Extended Data Table 2). Individual clinicians rated between 0% and 26% of their RCQs as high or critical risk (Extended Data Fig. 6c). Grouping the 12 intents into five raised agreement on intent from � 0.45 to 0.61, and agreement on answerability was 0.57 (Extended Data Table 2).

We then compared the corpus annotator (GPT-5.6 sol) with the clinicians against a common reference. Because each RCQ was annotated by a diferent set of three of the seven clinicians, we held out each of an RCQ’s three clinicians in turn and took the label shared by the other two, when they agreed, as the reference; the model was compared with the same reference. Across the 23 judged fields, the model matched the reference in 84.6% of comparisons and the held-out clinicians in 83.0%. The model’s agreement exceeded the clinician’s for 14 of the 23 fields, significantly so for four (AMA use case, vulnerable population, text generation and query form), and was significantly lower for none (Extended Data Fig. 6b). For potential harm, the quadratic-weighted � between the model and individual clinicians was 0.53, compared with 0.36 between pairs of clinicians, and 86% of model ratings were within one grade of a clinician’s rating, compared with 79% of ratings by clinician pairs (Extended Data Fig. 6d). For the safest response, the weighted � was 0.46 between the model and individual clinicians, 0.40 between pairs of clinicians and 0.53 between the model and clinician consensus (Extended Data Fig. 7). The model disagreed most where clinicians also disagreed: for task category, it matched the majority on 83% of RCQs on which all three clinicians agreed and on 49% of those on which they did not.

The model was more cautious about harm than the clinician majority. It rated 24% of validation RCQs high or critical, compared with 6% by the three-clinician majority (95% CI 2.8%–12.5%); it flagged all six RCQs that the majority rated high or critical, but the majority agreed with only 25% of its high- or critical-risk ratings (Extended Data Fig. 6c–e). Its rates of context requirements and of RCQs needing more than a direct answer matched the majority’s. Calibrated to clinician consensus, the corpus prevalence of high- or critical-risk RCQs was 3.7%, 95% CI 1.2–5.9% and the share needing more than a direct answer was 44.8%, 95% CI 35.7–54.4%. We therefore treat 13.2% as a cautious upper estimate of high-stakes use.

## Statistical analysis of the corpus

Prevalences are reported as shares of annotated queries, with 95% confidence intervals from a percentile cluster bootstrap that resampled clinicians (2,000 replicates), because conversations from the same clinician are correlated<sup>88</sup>. Clinician-weighted estimates averaged per-clinician proportions. Sensitivity analyses excluded the 1% of clinicians with the most conversations, excluded non-clinical (‘Other’) queries and stratified by professional group. The association between task mix and professional family is summarized by Cramér’s V. Diferences between professional families in the share of high- or critical-risk queries were tested by logistic regression with standard errors clustered by clinician. Concentration of use is summarized by the Lorenz curve and Gini coeficient.

Associations with high or critical potential harm (versus lower grades) were estimated by logistic regression with standard errors clustered by clinician<sup>89</sup>. Model 1 included the surface features, patient specificity, query form and professional family; model 2 also included task category (reference, documentation and workflow). To assess whether risk can be recognized from the query text before answering, we fitted a logistic model with 11 text-observable features (eight binary surface features, two query-form contrasts and patient specificity), without task or professional family, in five-fold cross-validation grouped by clinician, and report the pooled out-of-fold AUC and the operating point that captures 90% of high- or critical-risk queries.

For out-of-specialty analyses we mapped 31 organ- or discipline-defined specialties to the clinical department (and, for Department of Medicine subspecialties, the division) that owns their clinical territory (for example, cardiology to the Cardiology division of the Department of Medicine, and ophthalmology to Ophthalmology). Generalist specialties (internal medicine, hospital medicine, family medicine and emergency medicine) were excluded, as were queries with no clinical subject or with cross-cutting content (population health, biochemistry and molecular pharmacology, forensic medicine). A query was out of department if its assigned department was not among those mapped to the asker’s specialty, and out of division if, in addition, its Medicine division did not match.

Analyses used Python 3.9 with pandas 2.3, statsmodels 0.13, scikit-learn 1.0 and matplotlib 3.5. Reporting follows the STROBE and TRIPOD-LLM guidelines where applicable<sup>90,91</sup>.

## Data availability

Query text cannot be shared because it may contain protected health information and is subject to institutional agreement. The label-level dataset (24 annotation fields, professional family and suppressed specialty, without query text) is available from the corresponding author on reasonable request, subject to institutional approval and a data use agreement. Clinician and model labels for the validation set, without query text, are available from the corresponding author on reasonable request. The annotation guidelines are provided as Supplementary Note 1 and in the RCQ-Map repository (https://github.com/nyuolab/RCQ-Map; https://doi.org/10.5281/zenodo.23223807). The Clinical AI Benchmark Atlas (CABA v1.0) lists each benchmark’s source, version and license, together with the RCQ-Map labels for every sampled item. It is provided as Supplementary Data 1, archived at https://doi.org/10.5281/zenodo.23223809 and released as a Hugging Face dataset (https://huggingface.co/datasets/NYU-OLAB/CABA). Benchmark items are redistributed only where their licenses allow. All other items are listed by item ID and can be rebuilt from their original sources, under their own terms, with the scripts in the CABA repository. The aggregate RCQ distributions used for comparison are included with CABA.

## Code availability

The RCQ-Map annotation guidelines, output schema, prompts, annotation pipeline and clinician annotation platform are available at https://github.com/nyuolab/RCQ-Map (version 1.0, archived at https://doi.org/10.5 281/zenodo.23223807). Code to assemble CABA, sample and render benchmark items, apply RCQ-Map and compare benchmarks with real use is available at https://github.com/nyuolab/CABA (archived at https://doi.org/10.5281/zenodo.23223809). The guidelines are released under a CC BY 4.0 license and the code under the Apache License 2.0.

## References

1. American Medical Association. 2026 Physician Survey on Augmented Intelligence (AMA Center for Digital Health and AI, 2026); https://www.ama-assn.org/system/files/physician-ai-sentiment-report.pdf

2. Shah, N. H., Entwistle, D. & Pfefer, M. A. Creation and adoption of large language models in medicine. JAMA 330, 866–869 (2023).

3. Tierney, A. A. et al. Ambient artificial intelligence scribes to alleviate the burden of clinical documentation. NEJM Catal. Innov. Care Deliv. 5, CAT.23.0404 (2024).

4. Garcia, P. et al. Artificial intelligence–generated draft replies to patient inbox messages. JAMA Netw. Open 7, e243201 (2024).

5. Bedi, S. et al. Testing and evaluation of health care applications of large language models: a systematic review. JAMA 333, 319–328 (2025).

6. Chen, S. F. et al. LLM-assisted systematic review of large language models in clinical medicine. Nat. Med. 32, 1152–1159 (2026).

7. Raji, I. D., Daneshjou, R. & Alsentzer, E. It’s time to bench the medical exam benchmark. NEJM AI 2, AIe2401235 (2025).

8. Nori, H., King, N., McKinney, S. M., Carignan, D. & Horvitz, E. Capabilities of GPT-4 on medical challenge problems. Preprint at https://arxiv.org/abs/2303.13375 (2023).

9. Kung, T. H. et al. Performance of ChatGPT on USMLE: potential for AI-assisted medical education using large language models. PLOS Digit. Health 2, e0000198 (2023).

10. Singhal, K. et al. Toward expert-level medical question answering with large language models. Nat. Med. 31, 943–950 (2025).

11. Alaa, A. et al. Position: medical large language model benchmarks should prioritize construct validity. In Proc. 42nd International Conference on Machine Learning, PMLR 267, 80991–81004 (2025).

12. Hager, P. et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making. Nat. Med. 30, 2613–2622 (2024).

13. Tu, T. et al. Towards conversational diagnostic artificial intelligence. Nature 642, 442–450 (2025).

14. Johri, S. et al. An evaluation framework for clinical use of large language models in patient interaction tasks. Nat. Med. 31, 77–86 (2025).

15. Arora, R. K. et al. HealthBench: evaluating large language models towards improved human health. Preprint at https://arxiv.org/abs/2505.08775 (2025).

16. Bedi, S. et al. Holistic evaluation of large language models for medical tasks with MedHELM. Nat. Med. 32, 943–951 (2026).

17. Feng, J. et al. Expert evaluation of clinical AI tools on real point-of-care clinical queries. Preprint at https: //arxiv.org/abs/2606.28960 (2026).

18. Hicks, R. S. et al. HealthBench Professional: evaluating large language models on real clinician chats. Preprint at https://arxiv.org/abs/2604.27470 (2026).

19. Ma, Y. et al. An evaluation framework for large language models in clinical nursing: a scoping review and expert consultation. J. Nurs. Manag. 2026, 9524014 (2026).

20. Yu, S.-Y., Lin, H.-P., Kung, Y. M. & Chen, S.-C. Artificial intelligence-enhanced clinical reasoning in nurse practitioners: a systematic review. Nurse Educ. Pract. 91, 104724 (2026).

21. Bean, A. M. et al. Measuring what matters: construct validity in large language model benchmarks. In Advances in Neural Information Processing Systems 38, 19868–19949 (2025).

22. Unell, A., Kashyap, M., Pfefer, M. & Shah, N. Real-world usage patterns of large language models in healthcare. Preprint at medRxiv https://doi.org/10.1101/2025.05.02.25326781 (2025).

23. Black, K. C. et al. Uses of generative AI by non-clinician staf at an academic medical center. npj Health Syst. 3, 13 (2026).

24. Rader, B. et al. Utilization of a HIPAA-compliant large language model chatbot in an academic pediatric medical center. PLOS Digit. Health 5, e0001684 (2026).

25. Shah, N. H. et al. Adoption and use of LLMs at an academic medical center. Preprint at https://arxiv.org/abs/2602.00074 (2026).

26. Tamkin, A. et al. Clio: privacy-preserving insights into real-world AI use. Preprint at https://arxiv.org/abs/2412.13678 (2024).

27. Handa, K. et al. Which economic tasks are performed with AI? Evidence from millions of Claude conversations. Preprint at https://arxiv.org/abs/2503.04761 (2025).

28. Chatterji, A. et al. How people use ChatGPT. NBER Working Paper 34255 (National Bureau of Economic Research, 2025).

29. Lewis, P. et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. Adv. Neural Inf. Process. Syst. 33, 9459–9474 (2020).

30. Zakka, C. et al. Almanac — retrieval-augmented language models for clinical medicine. NEJM AI 1, AIoa2300068 (2024).

31. Li, S. S. et al. MediQ: question-asking LLMs and a benchmark for reliable interactive clinical reasoning. Adv. Neural Inf. Process. Syst. 37, 28858–28888 (2024).

32. Dvijotham, K. et al. Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians. Nat. Med. 29, 1814–1820 (2023).

33. Finlayson, S. G. et al. The clinician and dataset shift in artificial intelligence. N. Engl. J. Med. 385, 283–286 (2021).

34. Jin, D. et al. What disease does this patient have? A large-scale open domain question answering dataset from medical exams. Appl. Sci. 11, 6421 (2021)

35. Goh, E. et al. Toward a test of medical AI superintelligence. Nat. Med. https://doi.org/10.1038/s41591-026-04539-8 (2026).

36. McCoy, L. G. et al. Assessment of large language models in clinical reasoning: a novel benchmarking study. NEJM AI 2, AIdbp2500120 (2025).

37. Wu, D. et al. First, do NOHARM: a medical safety benchmark and randomized study of physician and AI teaming on clinical consultations. Preprint at https://arxiv.org/abs/2512.01241 (2025).

38. Sinsky, C. et al. Allocation of physician time in ambulatory practice: a time and motion study in 4 specialties. Ann. Intern. Med. 165, 753–760 (2016).

39. Arndt, B. G. et al. Tethered to the EHR: primary care physician workload assessment using EHR event log data and time-motion observations. Ann. Fam. Med. 15, 419–426 (2017)

40. Hendrich, A., Chow, M. P., Skierczynski, B. A. & Lu, Z. A 36-hospital time and motion study: how do medical-surgical nurses spend their time? Perm. J. 12, 25–34 (2008).

41. Geifman, Y. & El-Yaniv, R. Selective classification for deep neural networks. Adv. Neural Inf. Process. Syst. 30 (2017).

42. Kamath, A., Jia, R. & Liang, P. Selective question answering under domain shift. In Proc. 58th Annual Meeting of the Association for Computational Linguistics 5684–5696 (Association for Computational Linguistics, 2020).

43. Gorman, P. N. & Helfand, M. Information seeking in primary care: how physicians choose which clinical questions to pursue and which to leave unanswered. Med. Decis. Making 15, 113–119 (1995).

44. Ely, J. W. et al. A taxonomy of generic clinical questions: classification study. BMJ 321, 429–432 (2000).

45. Williams, T., Szekendi, M., Pavkovic, S., Clevenger, W. & Cerese, J. The reliability of AHRQ Common Format Harm Scales in rating patient safety events. J. Patient Saf. 11, 52–59 (2015).

46. Sweeney, L. k-anonymity: a model for protecting privacy. Int. J. Uncertain. Fuzziness Knowl. Based Syst. 10, 557–570 (2002).

47. El Emam, K. & Dankar, F. K. Protecting privacy using k-anonymity. J. Am. Med. Inform. Assoc. 15, 627–637 (2008).

48. Vishwanath, K. et al. General-purpose large language models outperform specialized clinical AI tools on medical benchmarks. Nat. Med. https://doi.org/10.1038/s41591-026-04431-5 (2026).

49. Ely, J. W. et al. Analysis of questions asked by family doctors regarding patient care. BMJ 319, 358–361 (1999).

50. Allen, M. et al. The classification of clinicians’ information needs while using a clinical information system. AMIA Annu. Symp. Proc. 2003, 26–30 (2003).

51. Del Fiol, G., Workman, T. E. & Gorman, P. N. Clinical questions raised by clinicians at the point of care: a systematic review. JAMA Intern. Med. 174, 710–718 (2014).

52. Smith, R. What clinical information do doctors need? BMJ 313, 1062–1068 (1996).

53. Currie, L. M. et al. Clinical information needs in context: an observational study of clinicians while using a clinical information system. AMIA Annu. Symp. Proc. 2003, 190–194 (2003).

54. Cogdill, K. W. Information needs and information seeking in primary care: a study of nurse practitioners. J. Med. Libr. Assoc. 91, 203–215 (2003).

55. McKnight, M. The information seeking of on-duty critical care nurses: evidence from participant observation and in-context interviews. J. Med. Libr. Assoc. 94, 145–151 (2006).

56. Broder, A. A taxonomy of web search. SIGIR Forum 36, 3–10 (2002).

57. Richardson, W. S., Wilson, M. C., Nishikawa, J. & Hayward, R. S. The well-built clinical question: a key to evidence-based decisions. ACP J. Club 123, A12–A13 (1995).

58. Fleming, S. L. et al. MedAlign: a clinician-generated dataset for instruction following with electronic medical records. Proc. AAAI Conf. Artif. Intell. 38, 22021–22030 (2024).

59. Ely, J. W., Osherof, J. A., Chambliss, M. L., Ebell, M. H. & Rosenbaum, M. E. Answering physicians’ clinical questions: obstacles and potential solutions. J. Am. Med. Inform. Assoc. 12, 217–224 (2005).

60. Singhal, K. et al. Large language models encode clinical knowledge. Nature 620, 172–180 (2023).

61. Kaushal, R. et al. Medication errors and adverse drug events in pediatric inpatients. JAMA 285, 2114–2120 (2001).

62. 2023 American Geriatrics Society Beers Criteria® Update Expert Panel. American Geriatrics Society 2023 updated AGS Beers Criteria® for potentially inappropriate medication use in older adults. J. Am. Geriatr. Soc. 71, 2052–2081 (2023).

63. Madras, D., Pitassi, T. & Zemel, R. Predict responsibly: improving fairness and accuracy by learning to defer. Adv. Neural Inf. Process. Syst. 31 (2018).

64. Xiong, G., Jin, Q., Lu, Z. & Zhang, A. Benchmarking retrieval-augmented generation for medicine. In Findings of the Association for Computational Linguistics: ACL 2024 6233–6251 (Association for Computational Linguistics, 2024).

65. Feng, S. et al. Don’t hallucinate, abstain: identifying LLM knowledge gaps via multi-LLM collaboration. In Proc. 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) 14664–14690 (Association for Computational Linguistics, 2024).

66. Jansen, B. J. & Spink, A. How are we searching the World Wide Web? A comparison of nine search engine transaction logs. Inf. Process. Manag. 42, 248–263 (2006).

67. Natarajan, K., Stein, D., Jain, S. & Elhadad, N. An analysis of clinical queries in an electronic health record search utility. Int. J. Med. Inform. 79, 515–522 (2010).

68. Moon, S., Pakhomov, S., Liu, N., Ryan, J. O. & Melton, G. B. A sense inventory for clinical abbreviations and acronyms created using clinical notes and medical dictionary resources. J. Am. Med. Inform. Assoc. 21, 299–307 (2014).

69. Gilardi, F., Alizadeh, M. & Kubli, M. ChatGPT outperforms crowd workers for text-annotation tasks. Proc. Natl Acad. Sci. USA 120, e2305016120 (2023).

70. Zheng, L. et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. Adv. Neural Inf. Process. Syst. 36, 46595–46623 (2023).

71. Wu, C. et al. Towards evaluating and building versatile large language models for medicine. npj Digit. Med. 8, 58 (2025).

72. Liu, F. et al. Large language models in the clinic: a comprehensive benchmark. Preprint at https://arxiv.org/abs/2405.00716 (2024).

73. Warner, B. et al. Medmarks: a comprehensive open-source LLM benchmark suite for medical tasks. Preprint at https://arxiv.org/abs/2605.01417 (2026).

74. Wu, J. et al. BRIDGE: benchmarking large language models for understanding real-world clinical practice text. Nat. Biomed. Eng. https://doi.org/10.1038/s41551-026-01719-2 (2026).

75. Kanithi, P. K. et al. MEDIC: comprehensive evaluation of leading indicators for LLM safety and utility in clinical applications. Preprint at https://arxiv.org/abs/2409.07314 (2024).

76. Qiu, P. et al. Towards building multilingual language model for medicine. Nat. Commun. 15, 8384 (2024).

77. Goodwin, T. R. & Demner-Fushman, D. Clinical Language Understanding Evaluation (CLUE). Preprint at https: //arxiv.org/abs/2209.14377 (2022).

78. Cai, Y. et al. MedBench: a large-scale Chinese benchmark for evaluating medical large language models. Proc. AAAI Conf. Artif. Intell. 38, 17709–17717 (2024).

79. Epoch AI. Data on notable AI models. https://epoch.ai/data/notable-ai-models (accessed 4 October 2026).

80. Hayes, A. F. & Krippendorf, K. Answering the call for a standard reliability measure for coding data. Commun. Methods Meas. 1, 77–89 (2007).

81. Krippendorf, K. Content Analysis: An Introduction to Its Methodology 4th edn (SAGE, 2018).

82. Feinstein, A. R. & Cicchetti, D. V. High agreement but low kappa: I. The problems of two paradoxes. J. Clin. Epidemiol. 43, 543–549 (1990).

83. Gwet, K. L. Computing inter-rater reliability and its variance in the presence of high agreement. Br. J. Math. Stat. Psychol. 61, 29–48 (2008).

84. Fleiss, J. L. Measuring nominal scale agreement among many raters. Psychol. Bull. 76, 378–382 (1971).

85. Landis, J. R. & Koch, G. G. The measurement of observer agreement for categorical data. Biometrics 33, 159–174 (1977).

86. Cohen, J. Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit. Psychol. Bull. 70, 213–220 (1968).

87. Wilson, E. B. Probable inference, the law of succession, and statistical inference. J. Am. Stat. Assoc. 22, 209–212 (1927).

88. Field, C. A. & Welsh, A. H. Bootstrapping clustered data. J. R. Stat. Soc. B 69, 369–390 (2007).

89. Cameron, A. C. & Miller, D. L. A practitioner’s guide to cluster-robust inference. J. Hum. Resour. 50, 317–372 (2015).

90. von Elm, E. et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. Lancet 370, 1453–1457 (2007).

91. Gallifant, J. et al. The TRIPOD-LLM reporting guideline for studies using large language models. Nat. Med. 31, 60–69 (2025).

## Acknowledgements

We thank the health-system IT, privacy and data teams who processed the corpus, and Bryce McDonnell and Walter Wang for helping with data access, egress, and execution. We acknowledge Nader Mherabi and Dafna Bar-Sagi, Ph.D., for their support of medical AI research at NYU Langone. We thank M. Constantino and the NYULH High-Performance Computing (HPC) Team for computing resources essential to our work.

## Author contributions

TR and EKO supervised the study. KV and EKO conceptualized and established the study design. KV designed and administered the clinician validation of RCQ-Map. JVL, DBH, JJY, WRS, HAK, CO, and AV annotated the validation set. TR, HL, KV, and YA prepared and acquired the data. KV performed the statistical analysis. KV and TR contributed to study evaluation and data review. KV wrote the initial draft and developed the figures of the manuscript. All authors reviewed and approved the final manuscript.

## Competing interests

EKO reports equity in MarchAI and Artisight, spousal employment by Eikon Therapeutics, and consulting for Sofinnova Partners and Google. KV reports consulting for ChartR Health. The remaining authors declare no competing interests.

## Funding

EKO is supported by the National Cancer Institute’s Early-Stage Surgeon Scientist Program (3P30CA016087- 41S1) and the W.M. Keck Foundation. This work was supported by the Institute for Information & Communications Technology Planning and Evaluation (IITP) grant funded by the Ministry of Science and ICT (MSIT) of the Republic of Korea government (No. RS-2019-II190075 Artificial Intelligence Graduate School Program (KAIST); No. RS-2024-00509279, Global AI Frontier Lab).

## Extended Data

a Clinicians and conversations by professional family

![](images/5b93aebf7b2cb49ddd4242e06d51a9ee6b931be27cb60d10e53a1cf4450fb034.jpg)  
b Concentration of use

![](images/b603db11a6f9103bffee300ce991c1bb3abf36b5f5cc2674fec2296076fba154.jpg)  
c Conversations per clinician

![](images/1c761cd0ac26aa9abbc325316bf9d2cf537fdaa028189113b4c369a18089d3c8.jpg)

d Volume by specialty  
![](images/bbd3ebec813831f83e91b7c63193b38d9faf304d93f7499965cc0d444c40d372.jpg)

Extended Data Fig. 1 | Usage patterns. a, Share of clinicians and of conversations by professional family. b, Lorenz curve of conversations across clinicians ranked by use; the shaded area is proportional to the Gini coeficient. c, Conversations per clinician by professional family (log scale); bars, interquartile range; lines, 10th–90th percentile; numbers, median. d, Conversations and conversations per clinician for the 35 released specialties (physicians and APPs), all of which had at least 200 conversations.

Main estimates by weighting and subset

![](images/82a12f8241431f2fae129135aa1cd4c5b5baadfd5a1b274a90bb53c8e02249d6.jpg)  
Extended Data Fig. 2 | Robustness of headline estimates. Estimates for all conversations, weighting each clinician equally, excluding the 1% of clinicians with the most conversations, excluding non-clinical (‘Other’) queries, and for physicians and APPs and for RNs separately. Gray bands span the range across variants.

![](images/ee039d70f9effc0779222771fdc949df570902f2c1fe3093619b2834894b35c7.jpg)

![](images/1fb93ea8a878868d10a69ec00b796a6bd75a8a38b557d9c896320fe023d91a80.jpg)

Extended Data Fig. 3 | Clinical department and Department of Medicine division. a, Share of all queries by the department that would ordinarily own management of the problem, assigned by the precedence ladder in the annotation guidelines; ‘No clinical subject’ covers queries without clinical content. b, Division within the Department of Medicine, as a share of Medicine queries.

![](images/ffcd9e0ba91152c6386c32ed626ffce2fb43b2063b62e09d8a9c1f3ea050f2dd.jpg)  
Extended Data Fig. 4 | Specialty atlas. Task mix (% of each specialty’s queries; cells <5% unlabeled), conversation volume, share of patient-specific queries, share of high- or critical-risk queries and share of queries whose safest response is not a direct answer, for the 35 released specialties, sorted by high- or critical-risk share. Dashed lines, all-query values.

![](images/21b594cca5f9c42e4a662a021c3c7846071be1e4cdb1ae82205a88cd08a6c012.jpg)  
b Departments queried from outside

![](images/c7ec452685436ffefa4bb3b6f951611216349c8716b8ad9e854e5e99deac1181.jpg)  
Extended Data Fig. 5 | Queries outside the asker’s specialty. a, Share of clinically anchored queries from each organ- or discipline-defined specialty that concerned a condition owned by another department (dark) or, for Department of Medicine subspecialties, another division (light; in parentheses). Dashed line, all specialties. Generalist specialties and queries without clinical or with cross-cutting content are excluded (Methods). b, The departments and divisions most often queried from outside, as shares of out-of-division queries

![](images/d29e3f519c2f84a1eb800768b96068cff018e5b14e7a35a068c18dc2f9d2c206.jpg)  
c High or critical risk by rater  
d Harm grade, consensus versus model e Prevalence on the validation set

![](images/2e74c9a825174cb1219b10d2d84251cf78fafad76ccc2618545fe07fccc19940.jpg)

![](images/54b6a88bd8ec3594bf6942c9843b78ee29ed2b4159c142749258fb8349ccd1dc.jpg)

![](images/d155336f2a3b22259586a5651e110bdd38fb5e1b5f23a1b6090decf5ee83787b.jpg)

Extended Data Fig. 6 | Clinician validation of RCQ-Map annotation. One hundred randomly sampled queries, each annotated independently by three of seven clinicians (300 annotations) and by the corpus annotator (GPT-5.6 sol) with the same annotation guidelines. a, Inter-rater reliability for the 23 judged fields: Krippendorf’s � (ordinal metric for harm, answerability and safest response) with 95% bootstrap CIs, and Gwet’s AC1; dashed lines, � = 0.667 and 0.800. b, Agreement with a two-clinician reference (the label shared by two clinicians when they agree) for the held-out third clinician (gray) and for GPT-5.6 sol (blue); right, diference in percentage points, with ▲ or ▼ where the paired bootstrap 95% CI excludes zero. c, Share of queries rated high or critical by each clinician (gray points; each clinician rated 30–60 queries), across all clinician ratings, by the three-clinician majority and by GPT-5.6 sol. d, Harm grade by clinician consensus (majority, or median grade when all three difered) versus GPT-5.6 sol, with quadratic-weighted � and within-one-grade agreement for pairs of model and clinician ratings and for pairs of clinician ratings. e, Prevalence of selected labels on the validation set by clinician majority (95% Wilson CI) and by GPT-5.6 sol, with the model-annotated corpus prevalence for reference.

## a Safest response

![](images/01f168f7fe7bb19e0cda4e1041c5247c7b87ad0d3edb322faf2b602f28674235.jpg)

## b Answerability

![](images/27f04495ba27975cc0eeadb49fab22e62e2a992e3e68eda23dd35db3caabe21b.jpg)  
Extended Data Fig. 7 | Safest response and answerability: clinician consensus versus the model. Confusion matrices between clinician consensus (majority, or median when all three difered) and GPT-5.6 sol for the safest response (a) and answerability (b) on the 100 validation queries. Quadratic-weighted � between consensus and the model was 0.53 for the safest response and 0.55 for answerability (exact agreement 66% and 72%).

## a Kind of work

![](images/51c55e30d8a7af136346383161f8b39670b4639775c26d026c3fb93d48ebf2f6.jpg)

## b Task category

<table><tr><td></td><td>RCQs</td><td>HealthBench Professional</td><td>Real- POCQi</td><td>MedQA</td><td>MAST: NOHARM</td><td>MAST: SCT</td></tr><tr><td>Documentation &amp; workflow</td><td>35</td><td>23</td><td></td><td></td><td></td><td></td></tr><tr><td>Foundational knowledge</td><td>15</td><td>15</td><td>19</td><td>19</td><td></td><td></td></tr><tr><td>Drug information</td><td>14</td><td>15</td><td>33</td><td>14</td><td>16</td><td>6</td></tr><tr><td>Treatment &amp; management</td><td>6</td><td>24</td><td>28</td><td>16</td><td>51</td><td>36</td></tr><tr><td>Diagnosis &amp; differential</td><td>4</td><td>7</td><td>14</td><td>48</td><td>33</td><td>56</td></tr><tr><td>Test &amp; result interpretation</td><td>3</td><td>3</td><td>3</td><td>1</td><td></td><td>1</td></tr><tr><td>Patient education</td><td>1</td><td>2</td><td></td><td></td><td></td><td></td></tr><tr><td>Coding &amp; administrative</td><td>1</td><td>2</td><td></td><td></td><td></td><td></td></tr><tr><td>Procedural guidance</td><td>1</td><td>2</td><td>1</td><td></td><td></td><td></td></tr><tr><td>Other (non-clinical)</td><td>21</td><td>7</td><td>2</td><td>1</td><td></td><td>1</td></tr></table>

## c Safest response

![](images/b292e5c2f117eb7364d8e75585b78613d5af7bcaa0f90b45e3a1441ab2ebd0a5.jpg)  
d Item properties

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Patient-specific</td><td rowspan=1 colspan=1>Patient orlocal infoneeded</td><td rowspan=1 colspan=2>High or   Not a directcritical harm   answer</td></tr><tr><td rowspan=1 colspan=1>RCQs</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>39</td></tr><tr><td rowspan=1 colspan=1>HealthBench Professional</td><td rowspan=1 colspan=1>53</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>78</td></tr><tr><td rowspan=1 colspan=1>Real-POCQi</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>67</td></tr><tr><td rowspan=1 colspan=1>MedQA</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>MAST: NOHARM</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>88</td></tr><tr><td rowspan=1 colspan=1>MAST: SCT</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>2</td></tr></table>

## e Distance from RCQs

![](images/0e7efc04843f8b0370d34932d1739fad2def15d93a03b04fa410c07c56ebc54a.jpg)

![](images/f19379139edae64debb9badc5770e8b8c4b73a53183f0ef7c0d46876cadbec9e.jpg)

![](images/761e91ba24d0c67d1c2a258f50ca7fdbc2206506a2264c8400abae5b652889d4.jpg)

Extended Data Fig. 8 | Five public benchmarks compared with real clinical use. Real clinical queries (RCQs; n = 127,625) and five public benchmarks, all annotated with RCQ-Map by the same annotator (GPT-5.6 sol, reasoning efort none) with the same guidelines and schema. Real-POCQi, questions that physicians submitted to a clinical AI tool at the point of care; HealthBench Professional, first user turns of conversations written by physicians; MAST: NOHARM, specialist eConsult cases and their perturbed variants; MAST: SCT, script concordance test items; MedQA, examination questions in the style of the United States Medical Licensing Examination (Methods). Benchmarks are ordered from closest to farthest from RCQs in task distribution (e). a, Kind of work requested, with task categories grouped as in Fig. 2a; numbers of items in parentheses. b, Task category (% of each source’s items; dots, none). c, Safest response: answer directly, retrieve sources and then answer, ask the clinician to clarify, or defer (escalate to a person or abstain). d, Shares of items that concern a specific patient, need context that the item does not contain, carry high or critical potential harm, or are safest met by something other than a direct answer. e, Total variation distance between each benchmark’s distribution and the RCQ distribution for task, intent and safest response (0, identical; 1, no overlap), with 95% bootstrap CIs over items (over base cases for NOHARM). Full distributions are given in Supplementary Table 2.

Extended Data Table 1 | RCQ-Map fields, allowed values and grounding in prior literature.
<table><tr><td>Block</td><td>Field (values)</td><td>What it captures</td><td>Grounding</td></tr><tr><td rowspan="4"></td><td>Task &amp; domain Task category (10)</td><td>The deliverable that would satisfy the request: drug information; treatment and management; foundational knowledge; patient education; test and result interpretation; documentation and families16; LLM usage studies22,26 workflow; diagnosis and differential;</td><td>Generic clinical-question taxonomies44,49,51; clinical-information-system needs50,53; MedHELM task</td></tr><tr><td>Question intent (12; 5 groups)</td><td>guidance; other The motive behind the task (verification, Web-search intent56; generic documentation drafting, definition, mechanism, comparison, procedure,</td><td>fact look-up, clinical decision, dosing, question stems44; well-built clinical question⁵7; question pursuit43</td></tr><tr><td>Clinical department (25); Medicine division (15)</td><td>coding, result interpretation, other) The department (and Medicine division) Institutional ownership for that would own management, via a</td><td>governance and oversight</td></tr><tr><td>AMA use case (8)</td><td>precedence ladder Mapping to the AMA list of physician</td><td>(Methods) AMA physician survey¹</td></tr><tr><td>Query</td><td>Patient-specific (binary)</td><td>AI use cases, plus none of the above About an identifiable patient</td><td>Patient-specific needs49,53</td></tr><tr><td>properties Context</td><td>Patient or local information; current</td><td>Information a safe answer needs that the Unmet patient-specific needs53; query lacks</td><td>EHR grounding58; resource gaps59;</td></tr><tr><td>Risk</td><td>evidence (binary) Potential harm (5: minimal Harm if the clinician acted on an to critical)</td><td>incorrect answer</td><td>retrieval augmentation29,30 AHRQ Common Formats harm scale45,60; NCC MERP; WHO ICPS; ISMP high-alert medications;</td></tr><tr><td>Response</td><td>Answerability (3); safest response (5: answer directly, retrieve then answer, clarify, escalate,</td><td>Whether the query is answerable as posed; the least restrictive response that deferral32,41,42,63; retrieval29,64; is still safe</td><td>pediatric dosing61; older adults62 Selective prediction and clarifying questions³1; abstention65</td></tr><tr><td>Surface features</td><td>Medication; dose; lab or Literal cues in the text, coded result; imaging; vulnerable independently of judgments population; acute or urgent; text generation (binary);</td><td></td><td>Medication harm61,62; query logs66,67; abbreviation ambiguity68; documentation tools3,4</td></tr></table>

URLs for the AHRQ Common Formats, NCC MERP, WHO ICPS and ISMP sources are given in Methods.

Extended Data Table 2 | Inter-rater reliability and human–model concordance by field.
<table><tr><td>Field</td><td>Krippendorff&#x27;s α (95% CI)</td><td>Gwet&#x27;s AC1</td><td>Pairwise agreement (%)</td><td>Unanimous Held-out (%)</td><td>clinician vs reference</td><td>GPT-5.6 sol vs reference (%)</td><td>Difference, pts (95% ĆI)</td></tr><tr><td>Task &amp; domain</td><td></td><td></td><td></td><td></td><td>(%)</td><td></td><td></td></tr><tr><td>Task category</td><td>0.64 (0.55–0.73)</td><td>0.67</td><td>70</td><td>59</td><td>84</td><td>79</td><td>-5 (-14 to 4)</td></tr><tr><td>Clinical domain</td><td>0.48 (0.38–0.57)</td><td>0.60</td><td>61</td><td>48</td><td>79</td><td>73</td><td>-6 (-17 to 4)</td></tr><tr><td>Medicine division</td><td>0.43 (0.33–0.52)</td><td>0.59</td><td>61</td><td>45</td><td>74</td><td>73</td><td>-2 (-12 to 8)</td></tr><tr><td>Question intent</td><td>0.45 (0.36–0.54)</td><td>0.51</td><td>55</td><td>38</td><td>69</td><td>67</td><td>-2 (-16 to 10)</td></tr><tr><td>Question intent (five</td><td>0.61 (0.50–0.70)</td><td>0.68</td><td>74</td><td>62</td><td>84</td><td>82</td><td>-2 (-11 to 7)</td></tr><tr><td>groups) AMA use case</td><td>0.25 (0.14–0.34)</td><td>0.42</td><td>48</td><td>26</td><td>54</td><td>67</td><td>13 (1 to 24)</td></tr><tr><td>Query properties</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Patient-specific</td><td>0.61 (0.45–0.75)</td><td>0.76</td><td>85</td><td>78</td><td>91</td><td>92</td><td>0 (-6 to 7)</td></tr><tr><td>Actionable</td><td>0.39 (0.25–0.51)</td><td>0.44</td><td>71</td><td>56</td><td>79</td><td>83</td><td>4 (-5 to 13)</td></tr><tr><td>Evidence-dependent</td><td>0.29 (0.15–0.42)</td><td>0.33</td><td>65</td><td>48</td><td>73</td><td>66</td><td>-8 (-19 to 3)</td></tr><tr><td>Context &amp; response</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Needs patient context</td><td>0.30 (0.16–0.43)</td><td>0.39</td><td>67</td><td>51</td><td>76</td><td>84</td><td>8 (0 to 17)</td></tr><tr><td>Needs institutional context</td><td>0.25 (0.01–0.49)</td><td>0.89</td><td>91</td><td>86</td><td>95</td><td>97</td><td>2 (-2 to 6)</td></tr><tr><td>Needs current evidence</td><td>0.32 (0.16–0.46)</td><td>0.53</td><td>72</td><td>58</td><td>81</td><td>86</td><td>5 (-4 to 14)</td></tr><tr><td>Needs patient or local</td><td>0.30 (0.16–0.44)</td><td>0.39</td><td>67</td><td>51</td><td>76</td><td>84</td><td>8 (0 to 17)</td></tr><tr><td>information Answerability</td><td>0.57 (0.44–0.68)</td><td>0.49</td><td>65</td><td>50</td><td>77</td><td>79</td><td></td></tr><tr><td>Answerability below</td><td>0.52 (0.38–0.64)</td><td>0.52</td><td>76</td><td>64</td><td>84</td><td>85</td><td>2 (-8 to 12) 1 (-7 to 9)</td></tr><tr><td>high</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Safest response Risk</td><td>0.40 (0.26–0.52)</td><td>0.49</td><td>57</td><td>41</td><td>72</td><td>76</td><td>5 (-5 to 15)</td></tr><tr><td>Risk (5-level)</td><td>0.36 (0.23–0.48)</td><td>0.19</td><td>34</td><td>12</td><td>35</td><td>48</td><td>13 (-6 to 34)</td></tr><tr><td>High or critical risk</td><td>0.13 (-0.05 to</td><td>0.73</td><td>79</td><td>69</td><td>87</td><td>85</td><td>-2 (-10 to 7)</td></tr><tr><td>(binary) Surface features</td><td>0.31)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Names a medication</td><td>0.90 (0.82–0.97)</td><td>0.91</td><td>95</td><td></td><td>98</td><td></td><td></td></tr><tr><td>States a dose</td><td>0.85 (0.61–1.00)</td><td>0.98</td><td>98</td><td>93</td><td>99</td><td>98</td><td>0 (-2 to 2)</td></tr><tr><td>Cites a lab/result</td><td></td><td></td><td></td><td>97</td><td></td><td>99</td><td>0 (-2 to 2)</td></tr><tr><td></td><td>0.64 (0.41–0.81)</td><td>0.91</td><td>93</td><td>89</td><td>96</td><td>91</td><td>-5 (-11 to 1)</td></tr><tr><td>Mentions imaging</td><td>0.55 (0.10–0.82)</td><td>0.96</td><td>96</td><td>94</td><td>98</td><td>98</td><td>-0 (-4 to 3)</td></tr><tr><td>Vulnerable population</td><td>0.65 (0.15–0.89)</td><td>0.96</td><td>97</td><td>95</td><td>98</td><td>100</td><td>1 (0 to 3)</td></tr><tr><td>Acute or urgent</td><td>0.32 (0.03–0.57)</td><td>0.92</td><td>93</td><td>89</td><td>96</td><td>98</td><td>2 (-1 to 4)</td></tr><tr><td>Requests text generation</td><td>0.74 (0.54–0.88)</td><td>0.93</td><td>95</td><td>92</td><td>97</td><td>100</td><td>3 (1 to 5)</td></tr><tr><td>Query form</td><td>0.71 (0.60–0.80)</td><td>0.75</td><td>83</td><td>75</td><td>91</td><td>97</td><td>6 (2 to 11)</td></tr><tr><td>Uses abbreviation</td><td>0.90 (0.82–0.96)</td><td>0.91</td><td>95</td><td>93</td><td>98</td><td>99</td><td>1 (-2 to 4)</td></tr></table>

Krippendorf’s � uses the ordinal metric for harm, answerability and safest response and the nominal metric otherwise; 95% CIs from 2,000 bootstrap resamples of queries. The reference for concordance is the label shared by two clinicians when they agree; the held-out third clinician and the corpus annotator are scored against the same references. Diferences are model minus clinician, with paired bootstrap 95% CIs (1,000 resamples). The intent groups, patient or local information, answerability below high and high or critical risk are the fields as reported (Methods); actionability and evidence dependence were annotated but are not analyzed.

Extended Data Table 3 | Logistic regression of high or critical potential harm.
<table><tr><td>Variable</td><td>Model 1: OR (95% CI)</td><td>Model 2: OR (95% CI)</td></tr><tr><td>Concerns a specific patient</td><td>2.91 (2.64–3.22)</td><td>3.20 (2.88–3.55)</td></tr><tr><td>Names a medication</td><td>3.12 (2.89–3.37)</td><td>1.30 (1.18–1.44)</td></tr><tr><td>States a dose</td><td>1.13 (0.97–1.31)</td><td>1.93 (1.70–2.18)</td></tr><tr><td>Cites a laboratory or test result</td><td>0.93 (0.85–1.01)</td><td>0.99 (0.90–1.09)</td></tr><tr><td>Mentions imaging</td><td>0.62 (0.53–0.73)</td><td>0.76 (0.68–0.86)</td></tr><tr><td>Vulnerable population</td><td>1.24 (1.08–1.42)</td><td>1.37 (1.24–1.52)</td></tr><tr><td>Acute or urgent situation</td><td>2.43 (1.98–2.98)</td><td>3.07 (2.52–3.74)</td></tr><tr><td>Requests text generation</td><td>0.16 (0.11–0.24)</td><td>0.58 (0.44–0.75)</td></tr><tr><td>Uses clinical abbreviations</td><td>1.43 (1.35–1.51)</td><td>1.87 (1.76–1.98)</td></tr><tr><td>Query form: command (vs question)</td><td>0.56 (0.41–0.78)</td><td>0.87 (0.68–1.10)</td></tr><tr><td>Query form: fragment (vs question)</td><td>0.51 (0.46–0.56)</td><td>0.85 (0.79–0.91)</td></tr><tr><td>Fellow (vs attending)</td><td>1.15 (0.87–1.53)</td><td>1.25 (0.96–1.64)</td></tr><tr><td>Resident (vs attending)</td><td>1.47 (1.25–1.73)</td><td>1.21 (1.05–1.40)</td></tr><tr><td>APP (vs attending)</td><td>0.86 (0.73–1.02)</td><td>0.89 (0.76–1.04)</td></tr><tr><td>Registered nurse (vs attending)</td><td>0.46 (0.40–0.53)</td><td>0.58 (0.50–0.68)</td></tr><tr><td>Foundational knowledge (vs documentation &amp; workflow)</td><td></td><td>0.18 (0.13–0.24)</td></tr><tr><td>Drug information &amp; pharmacotherapy (vs documentation &amp; workflow)</td><td></td><td>15.92 (12.70–19.94)</td></tr><tr><td>Treatment &amp; management (vs documentation &amp; workflow)</td><td></td><td>19.48 (15.41–24.62)</td></tr><tr><td>Diagnosis &amp; differential (vs documentation &amp; workflow)</td><td></td><td>7.77 (6.25–9.66)</td></tr><tr><td>Test &amp; result interpretation (vs documentation &amp; workflow)</td><td></td><td>3.37 (2.67–4.25)</td></tr><tr><td>Patient education &amp; communication (vs documentation &amp; workflow)</td><td></td><td>1.68 (1.24–2.28)</td></tr><tr><td>Coding &amp; administrative (vs documentation &amp; workflow)</td><td></td><td>0.09 (0.04–0.21)</td></tr><tr><td>Procedural guidance (vs documentation &amp; workflow)</td><td></td><td>29.23 (22.16–38.55)</td></tr><tr><td>Other (vs documentation &amp; workflow)</td><td></td><td>0.41 (0.33–0.50)</td></tr></table>

Odds ratios with 95% CIs from standard errors clustered by clinician; n = 127,625 queries from 6,342 clinicians. Model 1 includes the surface features, patient specificity, query form and professional family; model 2 adds task category (reference, documentation and workflow).