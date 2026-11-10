# The Deception Archive as a repository of standardized datasets for large-scale verbal deception detection research

Caterina Borgese<sup>1</sup>, Riccardo Loconte<sup>2</sup>, Lucca Pfründer<sup>2</sup>, Bennett Kleinberg<sup>2,3</sup>

<sup>1</sup> Department of Health Sciences, 'Magna Graecia' University of Catanzaro, Catanzaro, Italy

<sup>2</sup> Department of Methodology and Statistics, Tilburg University, Tilburg, Netherlands

<sup>3</sup>Department of Security and Crime Science, University College London, London, UK

Abstract: Computational advances in natural language processing and machine learning have substantially expanded the use of automated approaches in verbal deception research. However, existing verbal deception datasets remain fragmented across disciplines, as they are often difficult to identify, access, and reuse, limiting cross-study comparisons and the development of robust computational models. We present the Deception Archive, a curated and standardized repository of publicly available verbal deception datasets built upon a large-scale systematic review. Eligible datasets were retrieved, manually inspected, and standardized following a structured data curation workflow. The current repository comprises 41 publicly available datasets containing 387,995 truthful or deceptive statements. The archive represents a heterogeneous repository (e.g., in data size, topic of investigation, groundtruth, etc.) and a living resource for integrating newly available deception datasets over time. By providing centralized access to standardized datasets and accompanying metadata, the Deception Archive facilitates dataset discovery, cross-dataset validation, and reproducible evaluation of computational models.

Corresponding author: Riccardo Loconte, r.loconte@tilburguniversity.edu

## BACKGROUND & SUMMARY

The ability to distinguish truthful from deceptive communication has long been a central challenge across psychology, communication studies, and, more recently, computational sciences. Deception occurs in both everyday and high-stakes contexts, including online marketplaces, employment interviews, and forensic and legal settings. In many cases, inaccurate credibility assessment can have dire legal, financial, and societal consequences, including misleading criminal investigations and wrongful convictions, financial fraud, and the spread of misinformation<sup>1</sup>. Consequently, understanding deceptive communication has become a longstanding research objective across disciplines.

Despite its relevance, decades of research have shown that accurately distinguishing truthful from deceptive statements remains a challenging task for human judges, who typically perform only slightly above the chance level when they base their judgments on intuition<sup>2,3</sup>. In the search for more substantiated approaches to understand and detect deception, theoretical frameworks grounded in verbal cues have emerged as the most promising because deception is expected to affect how information is generated, organized, and communicated<sup>4–7</sup>. For example, the Cognitive Load Theory posits that fabricating information imposes greater cognitive demands than telling the truth because deceivers must suppress the truth, construct an alternative account, and maintain internal consistency<sup>8</sup>. These additional cognitive demands are expected to influence the structure, complexity, and amount of detail contained in verbal reports<sup>1,9</sup>. Reality Monitoring theory proposes that truthful accounts, being derived from genuine memories, contain richer perceptual, contextual, and sensory information than fabricated recollections of events<sup>10</sup>. Similarly, the Verifiability Approach suggests that deceivers strategically regulate the information they disclose by avoiding details that could be independently verified while maintaining overall plausibility<sup>11,12</sup>. Although these frameworks differ in their underlying assumptions, they converge on the idea that deception affects verbal communication and leaves traces in language in the form of verbal cues that can, in turn, be used to predict deception.

This reliance on verbal cues has fuelled an interest among deception research in natural language processing and machine learning research. Rather than relying on manually coded linguistic cues or subjective human judgments, automated approaches have enabled the analysis of large collections of truthful and deceptive statements by extracting linguistic features and have led to predictive modelling approaches adapted from supervised machine learning. Importantly, recent evidence suggests that automated approaches can outperform human judges in deception detection. Together, the availability of linguistic features and the scalability of automated approaches made verbal communication particularly well-suited for computational analysis, leading to a rapid increase in automated verbal deception detection research in the past 25 years<sup>13–15</sup> .

As the field of automated verbal deception detection has grown and computational approaches have become increasingly data-intensive, the primary challenge has gradually shifted from developing analytical techniques to ensuring the availability of suitable datasets. Computational models require large, diverse, and well-documented corpora for training, benchmarking, and external evaluation<sup>13</sup>. Consequently, the quality of the deception detection models depends not only on the learning algorithms employed but also on the properties of the datasets on which they were trained and evaluated<sup>13</sup>. More broadly, a growing emphasis on open science and research reproducibility has further highlighted the importance of transparent, well-documented, and reusable research datasets, particularly in dataintensive research.

Despite the growing number of datasets on verbal deception published over the past two decades<sup>15,16</sup>, identifying and accessing these resources remains difficult. Existing datasets differ in characteristics (e.g., language, data collection procedures, size), are dispersed across scientific areas, and are available in non-standardized formats. As a result, researchers seeking to identify appropriate datasets, reproduce previous work, or develop new computational models invest considerable effort in locating, retrieving, and understanding existing resources. Such fragmentation limits reproducibility and, ultimately, the scientific progress within verbal deception research.

To address these challenges, we present the Deception Archive, a curated repository of datasets on verbal deception compiled from a large-scale systematic review of the computational verbal deception literature<sup>15</sup>. In the present article, we describe its development, including the structured data curation workflow, and further provide a descriptive overview of the key properties of the included datasets. The Deception Archive is a transparent and standardized research resource to support future computational, psychological, and linguistic investigations of verbal deception. The Deception Archive is publicly available in an interactive, searchable interface via https://deception-archive.net; all data underlying the repository are available through Open Science Framework (OSF; link: https://osf.io/x5wpr/overview?view\_only=b1a69771e2cf4150be10bebf8b3c3391) and Zenodo (DOI: removed for peer review).

## METHODS

## Construction of the Deception Archive

The construction of the Deception Archive followed a systematic workflow consisting of four phases: (1) selection of candidate datasets, (2) dataset retrieval, (3) dataset curation and standardization, and the final (4) repository construction. Figure 1 provides an overview of the complete data curation protocol.

## Phase 1: Selection of Candidate Datasets

The identification of candidate datasets was based on the corpus of publications included in a recent systematic review and meta-analysis<sup>15</sup>, which systematically identified and synthesized studies applying computational methods to verbal deception detection. The review was conducted according to PRISMA guidelines and screened an initial 29,023 publications from both peer-reviewed literature and preprints. Screening and eligibility assessment resulted in a final set of 289 eligible records. For those, a comprehensive set of record-level variables was extracted, including methodological characteristics, computational approaches, and dataset-related information (Table 1). Out of all 289 initial records, 10 records were excluded because it was unclear whether the datasets had been newly collected by the authors or reused from previous studies, 20 records were excluded because no information on the ground-truth labels was available, and 13 records were excluded because they were not documented in an academic publication, leaving 246 final eligible records. From this pool of records, we then identified 134 unique datasets for retrieval. Among these, 56 were unique datasets reused across multiple publications, and 78 represented newly developed datasets.

![](images/7f73673421e4a1bc68255976ab1277eccff183ef43e4eb0d23dbc45ed505f32a.jpg)  
Figure 1. Overview of the Deception Archive development workflow

## Phase 2: Dataset Retrieval

For each of the 134 datasets retained, a systematic dataset retrieval procedure was used to locate the corresponding datasets. All related publications were first manually inspected for information regarding dataset availability, including direct download links, supplementary materials, or references to external repositories. If no information enabling dataset retrieval was available within the publication, additional searches were conducted across publicly available repositories (such as Open Science Framework, GitHub, or Kaggle), and project websites. A total of 46 datasets were retrieved using this procedure.

For the remaining 88 datasets (i.e., from papers published by 82 corresponding authors), for which no downloadable dataset could be identified through either the publication or additional searches, we contacted the corresponding authors by email to request access to the original data. If the contacted authors did not respond within 14 days, a follow-up reminder was sent two weeks after the initial request. Overall, 11 authors (13.4%) responded to the data request, of whom nine replied after the initial email and two after the reminder. Among those who responded, three provided access to their datasets, four indicated that the original data were no longer available, three reported that the datasets could not be shared because of data-sharing restrictions, and one provided an alternative download link that ultimately proved inaccessible. Combining the 46 datasets retrieved from publicly available sources with the three datasets obtained after contacting the authors resulted in a final collection of 49 datasets, which formed the basis for the subsequent dataset processing and standardization stage.

## Phase 3: Dataset Processing and Standardization

Each retrieved dataset was manually inspected and processed to harmonize the substantial heterogeneity in file formats, data structures, and variable names. During this process, the accompanying documentation and, where necessary, the corresponding publication were consulted to correctly interpret the dataset structure, variable definitions, and annotation scheme. Each retrieved dataset was first transformed into a standardized format consisting of six core variables: (i) a unique dataset identifier to ensure traceability across the repository; (ii) a participant identifier to preserve information on whether multiple statements originated from the same individual within a dataset; (iii) the original textual content and (iv) veracity label provided by the dataset creators without modification; (v) a standardized binary veracity label (i.e., truthful vs. deceptive) to facilitate cross-dataset analyses, and (vi) the language of the statement text content. Additional dataset-level variables describing dataset characteristics were also reported, including study design (i.e., within- or between-subjects), level of ground-truth (following previous definitions<sup>15</sup>), truthful/deceptive proportion, format of statements (i.e., typed vs. transcribed), macro topic (e.g., Spam) and sub-topic (e.g., email phishing), type of deception (e.g., fabrication, embedded lies, mixed, etc.), and source of the data (e.g., online experiment). The metadata fields extracted and curated for each dataset are summarized in Table 1.

During the standardization process, some retrieved datasets were found to lack the information required to be harmonized within the common data structure of the Deception Archive. When missing or ambiguous information was identified, the corresponding authors were contacted to resolve the issue. Datasets were excluded only when these efforts failed to provide the information necessary for reliable standardization. Specifically, datasets were excluded when the original textual data were unavailable (n=2 datasets), the ground-truth labels were missing or could not be interpreted unambiguously (n=2), and the accompanying documentation was insufficient to reliably reconstruct the original dataset structure (n=1), or the dataset represented a duplicate of included datasets (n=1). Furthermore, two additional large-scale datasets, namely the Amazon Review Dataset<sup>17</sup> and the Yelp Dataset<sup>18</sup>, were not retrieved because their exceptionally large size made them unsuitable for inclusion within the standardized curation workflow adopted for the Deception Archive. Applying these criteria resulted in the exclusion of eight datasets. Consequently, the final version of the Deception Archive comprised 41 standardized datasets<sup>19–59</sup> harmonized within a common data structure and ready-to-use.

## Phase 4: Repository construction

The standardized datasets were subsequently integrated into the Deception Archive and organized according to a common data structure, with each dataset accompanied by curated metadata describing its main characteristics. An interactive dashboard was implemented to allow researchers to explore the available datasets, inspect their characteristics, and access the corresponding data for download and subsequent computational analyses. The repository was also designed to accommodate the integration of newly available datasets over time, allowing the Deception Archive to be continuously expanded as additional resources become available.

## Ethics statement

The Deception Archive compiles previously collected and published datasets involving human participants; no new data were collected for the present study. Ethical review and participant consent for the original data collections were obtained by the authors of the respective source studies, as documented in the corresponding publications. The datasets included in the Archive were either publicly released by their creators or shared with us for this purpose.

Table 1. Metadata fields included in the Deception Archive
<table><tr><td rowspan=1 colspan=1>Variable</td><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>Example</td></tr><tr><td rowspan=1 colspan=2>Core variables</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Participant ID</td><td rowspan=1 colspan=1>Participant identifier preserving within-datasetlinkage</td><td rowspan=1 colspan=1>HIP_2022_1</td></tr><tr><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>Original unmodified statement</td><td rowspan=1 colspan=1>&quot;Concerts are my most favoritething, and my boyfriend knew it.[...]&quot;</td></tr><tr><td rowspan=1 colspan=1>Original label</td><td rowspan=1 colspan=1>Original annotation provided by datasetauthors</td><td rowspan=1 colspan=1>Imagined; recalled</td></tr><tr><td rowspan=1 colspan=1>Standardized label</td><td rowspan=1 colspan=1>Binary truthful/deceptive label</td><td rowspan=1 colspan=1>Truthful; deception</td></tr><tr><td rowspan=1 colspan=1>Language</td><td rowspan=1 colspan=1>Language of the statement</td><td rowspan=1 colspan=1>English</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Dataset-level variables</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Dataset ID</td><td rowspan=1 colspan=1>Unique dataset identifier</td><td rowspan=1 colspan=1>HIP_2022</td></tr><tr><td rowspan=1 colspan=1>Publication Year</td><td rowspan=1 colspan=1>Year in which the dataset waspublished/reported</td><td rowspan=1 colspan=1>2022</td></tr><tr><td rowspan=1 colspan=1>Within or betweendesign</td><td rowspan=1 colspan=1>Whether data were collected within- orbetween-subjects.</td><td rowspan=1 colspan=1>Between/ Within subject</td></tr><tr><td rowspan=1 colspan=1>Ground truth</td><td rowspan=1 colspan=1>Reliability level of the deception labelsassigned to the dataset</td><td rowspan=1 colspan=1>Clear but not verifiable</td></tr><tr><td rowspan=1 colspan=1>Documented inacademic outlet</td><td rowspan=1 colspan=1>Whether the dataset is described in a peer-reviewed publication or other academic source</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>Number ofstatements</td><td rowspan=1 colspan=1>Total number of textual records included inthe dataset</td><td rowspan=1 colspan=1>5535</td></tr><tr><td rowspan=1 colspan=1>Truthful / deceptiveproportion</td><td rowspan=1 colspan=1>Distribution of truthful and deceptivestatements</td><td rowspan=1 colspan=1>1.0</td></tr><tr><td rowspan=1 colspan=1>Format</td><td rowspan=1 colspan=1>Format of the verbal material</td><td rowspan=1 colspan=1>Typed</td></tr><tr><td rowspan=1 colspan=1>Macro topic</td><td rowspan=1 colspan=1>Main topic or application domain representedin the dataset.</td><td rowspan=1 colspan=1>Past events</td></tr><tr><td rowspan=1 colspan=1>Sub-topic</td><td rowspan=1 colspan=1>Specific topic represented within the macrotopic</td><td rowspan=1 colspan=1>Past memorable experiences</td></tr><tr><td rowspan=1 colspan=1>Type of deception</td><td rowspan=1 colspan=1>Main deception type represented in thedataset.</td><td rowspan=1 colspan=1>Fabrication</td></tr><tr><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>Experimental or naturalistic origin and studydesign</td><td rowspan=1 colspan=1>Online experiment</td></tr></table>

## Data Records

The data underlying the Deception Archive are hosted on Zenodo (DOI: removed for peer review) and are also available through the OSF (link: https://osf.io/x5wpr/overview?view\_only=b1a69771e2cf4150be10bebf8b3c3391). The deposit contains: (i) 41 standardized dataset files in CSV format (UTF-8 encoded), one per dataset, named by the dataset identifier listed in the metadata (e.g., [BLC\_2019.csv]); (ii) a dataset-level metadata table ([metadata.csv]) containing the fields described in Table 1; and (iii) a README file documenting the folder structure and variable definitions. Each standardized dataset file contains the variables described in Table 1. The standardized veracity label is coded [0 = truthful, 1 = deceptive]. The Archive is released under a CC BY 4.0 license.

## Data Overview

Across these datasets, the Archive contains a total of 387,995 statements, with substantial variability in dataset size, ranging from 62 to 108,688 statements (M = 9,463.29; Median = 1,492.00; SD = 25,704.07). This wide range reflects the considerable heterogeneity of existing deception corpora, encompassing both relatively small experimental datasets and large-scale data collections in online settings.

The distribution of dataset characteristics is summarized in Supplementary Table S1. Most datasets are in English (n = 29, 70.7% of datasets, accounting for 95.5% of all statements), contain typed rather than transcribed text (n = 31, 75.6% of datasets), and originate from experimental or quasi-experimental designs, with fabrication as the predominant deception type (n = 30, 73.2% of datasets) and “clear but non-verifiable” as the most common ground-truth level (n = 22, 53.7% of datasets; see Supplementary Table S2 for definitions). At the same time, the Archive spans nine languages beyond English and a broad range of application domains, from consumer reviews and personal narratives to forensic interviews and digital scams.

## Technical Validation

The quality of the Deception Archive was assessed throughout the curation workflow described in the Methods. Each retrieved dataset was manually inspected, and the mapping from the original annotations to the standardized binary labels was verified against the accompanying documentation and, where necessary, the original publication. Ambiguities were resolved by contacting the corresponding authors, and datasets that could not be reliably standardized were excluded.

## USAGE NOTES

The Deception Archive reflects the availability and accessibility of verbal deception datasets at the time of compilation: many datasets described in the literature were no longer retrievable, and datasets lacking the information required for reliable standardization were excluded (see Methods). Researchers who wish to contribute new or previously unavailable datasets to future releases of the Archive are invited to contribute their dataset by reaching out the authors or add their dataset directly on https://deceptionarchive.net/contribute.

When using the Deception Archive, two limitations should be acknowledged. First, the Deception Archive includes only datasets that we could successfully retrieve, interpret, and standardize based on the available materials. Consequently, some datasets described in the literature could not be incorporated because the necessary data or documentation were unavailable. Second, the current version of the Deception Archive focuses on datasets originating from the automated verbal deception detection literature. Other valuable verbal deception datasets, including those developed outside this research area (e.g., manually annotated verbal cues of deception), are not currently included but could be incorporated in future releases. Taken together, while the Deception Archive represents the largest concentrated repository of verbal deception research datasets based on the hitherto most extensive systematic review in the field, it is far from a comprehensive collection of all datasets used for studying verbal deception. Importantly, the open structure of the Deception Archive allows researchers to contribute additional datasets over time, supporting its continued expansion and development as a shared resource for verbal deception research.

For computational reuse, several properties of the standardized data should be taken into account. The participant identifier allows statements from the same individual to be kept within the same partition when constructing training and test splits, avoiding data leakage in within-subjects datasets. The standardized binary label supports cross-dataset analyses, whereas the original labels preserve finergrained distinctions where available. Users should further account for the unbalanced truthful/deceptive proportions in some datasets, and for the dominance of English-language material when generalizing across datasets. When individual datasets are used, the original source publications should be cited alongside the Archive.

## Data Availability

The data underlying the Deception Archive are publicly available on Zenodo (DOI: removed for peer review) and through the Open Science Framework (OSF; https://osf.io/x5wpr/overview?view\_only=b1a69771e2cf4150be10bebf8b3c3391) under a CC BY 4.0 license. The deposited files are described in the Data Records section. An interactive, searchable interface is available at https://deception-archive.net.

## Code Availability

No custom code is required to access or use the datasets.

## References

1. Vrij, A., Fisher, R., Mann, S. & Leal, S. A cognitive load approach to lie detection. Journal of Investigative Psychology and Offender Profiling 5, 39–43 (2008).

2. Bond, C. F. & DePaulo, B. M. Accuracy of Deception Judgments. Personality and Social Psychology Review 10, 214–234 (2006).

3. Hartwig, M. & Bond, C. F. Why do lie-catchers fail? A lens model meta-analysis of human lie judgments. Psychol. Bull. 137, 643–659 (2011).

4. Amado, B. G., Arce, R., Fariña, F. & Vilariño, M. Criteria-Based Content Analysis (CBCA) reality criteria in adults: A meta-analytic review. Int. J. Clin. Health Psychol. 16, 201–210 (2016).

5. Gancedo, Y., Fariña, F., Seijo, D., Vilariño, M. & Arce, R. Reality Monitoring: A Metaanalytical Review for Forensic Practice. https://journals.copmadrid.org/ejpalc 13, 99–110 (2021).

6. Vrij, A., Fisher, R. P., Blank, H., Leal, S. & Mann, S. A. A cognitive approach to elicit verbal and nonverbal cues to deceit. Preprint at https://researchportal.port.ac.uk/en/publications/acognitive-approach-to-elicit-verbal-and-nonverbal-cues-to-decei/ (2016).

7. Vrij, A. & Granhag, P. A. Eliciting cues to deception and truth: What matters are the questions asked. J. Appl. Res. Mem. Cogn. 1, 110–117 (2012).

8. Walczyk, J. J., Harris, L. L., Duck, T. K. & Mulay, D. A social-cognitive framework for understanding serious lies: Activation-decision-construction-action theory. New Ideas Psychol. 34, 22–36 (2014).

9. Vrij, A., Granhag, P. A. & Porter, S. Pitfalls and opportunities in nonverbal and verbal lie detection. Psychological Science in the Public Interest, Supplement 11, 89–121 (2010).

10. Johnson, M. K. & Raye, C. L. Reality monitoring. Psychol. Rev. 88, 67–85 (1981).

11. Nahari, G., Vrij, A. & Fisher, R. P. Exploiting liars’ verbal strategies by examining the verifiability of details. Legal Criminol. Psychol. 19, 227–239 (2014).

12. Vrij, A. & Nahari, G. The Verifiability Approach. Evidence-Based Investigative Interviewing 116–133 (2019) doi:10.4324/9781315160276-7.

13. Constancio, A. S., Tsunoda, D. F., de Fátima Nunes Silva, H., da Silveira, J. M. & Carvalho, D. R. Deception detection with machine learning: A systematic review and statistical analysis. PLoS One 18, e0281323 (2023).

14. Hauch, V., Blandón-Gitlin, I., Masip, J. & Sporer, S. L. Are computers effective lie detectors? A meta-analysis of linguistic cues to deception. Pers. Soc. Psychol. Rev. 19, 307–342 (2015).

15. Loconte, R., Festor, J., Fatjanova, Z., Bolkvadze, M. & Kleinberg, B. A persistent accuracy ceiling in automated verbal deception detection. https://arxiv.org/abs/2610.12118v1 (2026).

16. Markowitz, D. M. Verbal and Linguistic Analysis of Deceptive Statements: A 20-Year Explication (2005-2025). https://doi.org/10.31234/OSF.IO/3HE2A\_V1 (2026) doi:10.31234/OSF.IO/3HE2A\_V1.

17. McAuley, J. & Leskovec, J. Hidden factors and hidden topics: Understanding rating dimensions with review text. RecSys 2013 - Proceedings of the 7th ACM Conference on Recommender Systems 165–172 (2013) doi:10.1145/2507157.2507163.

18. Open Dataset | Yelp Data Licensing. https://business.yelp.com/data/resources/open-dataset/.

19. \*Skalicky, S., Duran, N. D. & Crossley, S. A. Please, Please, Just Tell Me: The Linguistic Features of Humorous Deception. Dialogue & Discourse 11, 128–149 (2020).

20. \*Salvetti, F., Lowe, J. B. & Martin, J. H. A Tangled Web: The Faint Signals of Deception in Text - Boulder Lies and Truth Corpus (BLT-C). 3510–3517 Preprint at https://aclanthology.org/L16-1558/ (2016).

21. \*Soldner, F., Pérez-Rosas, V. & Mihalcea, R. Box of Lies: Multimodal Deception Detection in Dialogues. NAACL HLT 2019 - 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies - Proceedings of the Conference 1, 1768–1777 (2019).

22. \*Verhoeven, B. & Daelemans, W. CLiPS Stylometry Investigation (CSI) corpus: A Dutch corpus for the detection of age, gender, personality, sentiment and deception in text. http://www.clips.uantwerpen.be/datasets/csi-corpus/.

23. \*Hirschberg, J. et al. Distinguishing deceptive from non-deceptive speech. 9th European Conference on Speech Communication and Technology 1833–1836 (2005) doi:10.21437/INTERSPEECH.2005-580.

24. \*Pérez-Rosas, V. & Mihalcea, R. Cross-cultural Deception Detection. 52nd Annual Meeting of the Association for Computational Linguistics, ACL 2014 - Proceedings of the Conference 2, 440–445 (2014).

25. \*Fornaciari, T. & Poesio, M. Identifying fake Amazon reviews as learning from crowds. 14th Conference of the European Chapter of the Association for Computational Linguistics 2014, EACL 2014 279–287 (2014) doi:10.3115/V1/E14-1030.

26. \*Fornaciari, T., Cagnina, L., Rosso, P. & Poesio, M. Fake opinion detection: how similar are crowdsourced datasets to real data? Lang. Resour. Eval. 54, 1019–1058 (2020).

27. \*Fornaciari, T. & Poesio, M. DeCour: a corpus of DEceptive statements in Italian COURts. 1585–1590 Preprint at http://www.lrec-conf.org/proceedings/lrec2012/pdf/377\_Paper.pdf (2012).

28. \*Ott, M., Choi, Y., Cardie, C. & Hancock, J. T. Finding Deceptive Opinion Spam by Any Stretch of the Imagination. 309–319 Preprint at https://aclanthology.org/P11-1032/ (2011).

29. \*Capuozzo, P., Lauriola, I., Strapparava, C., Aiolli, F. & Sartori, G. DecOp: A Multilingual and Multi-domain Corpus For Detecting Deception In Typed Text. 1423–1430 Preprint at https://aclanthology.org/2020.lrec-1.178/ (2020).

30. \*Velutharambath, A., Wührl, A. & Klinger, R. Can Factual Statements Be Deceptive? The DeFaBel Corpus of Belief-based Deception. 2708–2723 Preprint at https://aclanthology.org/2024.lrec-main.243/ (2024).

31. \*Velutharambath, A., Sassenberg, K. & Klinger, R. What if Deception Cannot be Detected? A Cross-Linguistic Study on the Limits of Deception Detection from Text. Computational Linguistics https://doi.org/10.1162/COLI.a.614 (2025) doi:10.1162/COLI.a.614.

32. \*Sap, M. et al. Quantifying the narrative flow of imagined versus autobiographical stories. Proc. Natl. Acad. Sci. U. S. A. 119, e2211715119 (2022).

33. \*Kleinberg, B. & Verschuere, B. How humans impair automated deception detection performance. Acta Psychol. (Amst). 213, 103250 (2021).

34. \*Li, J., Ott, M., Cardie, C. & Hovy, E. Towards a General Rule for Identifying Deceptive Opinion Spam. 52nd Annual Meeting of the Association for Computational Linguistics, ACL 2014 - Proceedings of the Conference 1, 1566–1576 (2014).

35. \*Ibraheem, S., Zhou, G. & DeNero, J. Putting the Con in Context: Identifying Deceptive Actors in the Game of Mafia. NAACL 2022 - 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Proceedings of the Conference 158–168 (2022) doi:10.18653/V1/2022.NAACL-MAIN.11.

36. \*de Ruiter, B. & Kachergis, G. The Mafiascum Dataset: A Large Text Corpus for Deception Detection. https://arxiv.org/abs/1811.07851v3 (2018).

37. \*Lloyd, E. P. et al. Miami University deception detection database. Behav. Res. Methods 51, 429–439 (2019).

38. \*Monaro, M., Maldera, S., Scarpazza, C., Sartori, G. & Navarin, N. Detecting deception through facial expressions in a dataset of videotaped interviews: A comparison between human judges and machine learning models. Comput. Human Behav. 127, (2022).

39. \*Ott, M., Cardie, C. & Hancock, J. T. Negative Deceptive Opinion Spam. 497–501 Preprint at https://aclanthology.org/N13-1053/ (2013).

40. \*Pérez-Rosas, V. & Mihalcea, R. Experiments in Open Domain Deception Detection. Conference Proceedings - EMNLP 2015: Conference on Empirical Methods in Natural Language Processing 1120–1125 (2015) doi:10.18653/V1/D15-1133.

41. \*Pérez-Rosas, V., Abouelenien, M., Mihalcea, R. & Burzo, M. Deception detection using reallife trial data. ICMI 2015 - Proceedings of the 2015 ACM International Conference on Multimodal Interaction 59–66 (2015) doi:10.1145/2818346.2820758.

42. \*Banerjee, R., Feng, S., Kang, J. S. & Choi, Y. Keystroke Patterns as Prosody in Digital Writings: A Case Study with Deceptive Reviews and Essays. EMNLP 2014 - 2014 Conference on Empirical Methods in Natural Language Processing, Proceedings of the Conference 1469– 1473 (2014) doi:10.3115/V1/D14-1155.

43. \*Soldner, F., Kleinberg, B. & Johnson, S. D. Confounds and overestimations in fake review detection: Experimentally controlling for product-ownership and data-origin. PLoS One 17, e0277869 (2022).

44. \*Pérez-Rosas, V. et al. Verbal and Nonverbal Clues for Real-life Deception Detection. Conference Proceedings - EMNLP 2015: Conference on Empirical Methods in Natural Language Processing 2336–2346 (2015) doi:10.18653/V1/D15-1281.

45. \*Hsiao, S. W. & Sun, C. Y. LoRA-like Calibration for Multimodal Deception Detection using ATSFace Data. Proceedings - 2023 IEEE International Conference on Big Data, BigData 2023 2163–2172 (2023) doi:10.1109/BigData59044.2023.10386630.

46. \*Sarzynska-Wawer, J., Pawlak, A., Szymanowska, J., Hanusz, K. & Wawer, A. Truth or lie: Exploring the language of deception. PLoS One 18, e0281179 (2023).

47. \*Kleinberg, B., van der Toolen, Y., Vrij, A., Arntz, A. & Verschuere, B. Automated verbal credibility assessment of intentions: The model statement technique and predictive modeling. Appl. Cogn. Psychol. 32, 354–366 (2018).

48. \*Hayat, U., Saeed, A., Vardag, M. H. K., Ullah, M. F. & Iqbal, N. Roman Urdu Fake Reviews Detection Using Stacked LSTM Architecture. SN Comput. Sci. 3, (2022).

49. \*Boumber, D. A., Qachfar, F. Z. & Verma, R. Domain-Agnostic Adapter Architecture for Deception Detection: Extensive Evaluations with the DIFrauD Benchmark. 5260–5274 Preprint at https://aclanthology.org/2024.lrec-main.468/ (2024).

50. \*Loconte, R. & Kleinberg, B. Examining embedded lies through computational text analysis. Scientific Reports 2025 15:1 15, 26482- (2025).

51. \*Kim, S., Lee, S., Park, D. & Kang, J. Constructing and evaluating a novel crowdsourcingbased paraphrased opinion spam dataset. 26th International World Wide Web Conference, WWW 2017 827–836 (2017) doi:10.1145/3038912.3052607.

52. \*Spyridis, Y., Younes, J.-P., Deeb, H. & Argyriou, V. Empowering Prior to Court Legal Analysis: A Transparent and Accessible Dataset for Defensive Statement Classification and Interpretation. https://arxiv.org/abs/2405.10702v1 (2024).

53. \*Cormack, G. & Lynam, T. TREC 2005 Spam Track Overview.

54. \*Peskov, D. et al. It Takes Two to Lie: One to Lie, and One to Listen. Proceedings of the Annual Meeting of the Association for Computational Linguistics 3811–3854 (2020) doi:10.18653/V1/2020.ACL-MAIN.353.

55. \*Baldivas, R. I. A. et al. LegalEye: Multimodal Court Deception Detection Across Multiple Languages. Behavioral Sciences 2025, Vol. 15, Page 1707 15, 1707 (2025).

56. \*Himdi, H. & Alhayan, F. Optimized ensemble stacking approaches to detect Arabic phishing email. Journal of Engineering Research 14, 776–788 (2026).

57. \*Patra, C., Giri, D., Kundu, B., Maitra, T. & Wazid, M. Rhetorical Structure Theory-based machine intelligence-driven deceptive phishing attack detection scheme. Journal of Information Security and Applications 94, 104184 (2025).

58. \*Sharma, N. et al. Gradient Boosting Decision Trees for Real-Time Phishing Attack Prevention in Cybersecurity. Proceedings - IEEE 10th International Conference on Smart Structures and Systems, ICSSS 2025 https://doi.org/10.1109/ICSSS66939.2025.11346380 (2025) doi:10.1109/ICSSS66939.2025.11346380.

59. \*Mendes, P., Maia, E. & Praça, I. MeAJOR Corpus: A Multi-Source Dataset for Phishing Email Detection. https://arxiv.org/abs/2507.17978v2 (2025).

## Supplementary Information

Supplementary Table S1. Distribution of dataset-level characteristics included in the Deception Archive
<table><tr><td>Variable</td><td>Value</td><td>n (%)</td></tr><tr><td>Language</td><td>English Italian</td><td>29 (70.7%) 3 (7.3%)</td></tr><tr><td></td><td>Dutch</td><td>1(2.4%)</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>German</td><td>1(2.4%)</td></tr><tr><td></td><td>Chinese</td><td>1(2.4%)</td></tr><tr><td></td><td>Polish</td><td>1(2.4%)</td></tr><tr><td></td><td>Roman Urdu</td><td>1(2.4%)</td></tr><tr><td></td><td>Arabic</td><td>1(2.4%)</td></tr><tr><td></td><td>Bilingual</td><td>2(4.9%)</td></tr><tr><td></td><td>Multilingual</td><td>1(2.4%)</td></tr><tr><td>Format</td><td>Typed</td><td>31(75.6%)</td></tr><tr><td></td><td>Transcribed</td><td>9 (22.0%)</td></tr><tr><td></td><td>Typed / Transcribed</td><td>1 (2.4%)</td></tr><tr><td>Collection methodology</td><td>Online experimental</td><td>14(34.1%)</td></tr><tr><td></td><td>Online naturalistic</td><td>7 (17.1%)</td></tr><tr><td></td><td>Offline naturalistic</td><td>7 (17.1%)</td></tr><tr><td></td><td>Online quasi-experimental</td><td>5 (12.2%)</td></tr><tr><td></td><td>Offline experimental</td><td>5 (12.2%)</td></tr><tr><td></td><td>Mixed</td><td>3 (7.3%)</td></tr><tr><td>Ground truth</td><td>Clear but non-verifiable</td><td>22(53.7%)</td></tr><tr><td></td><td>Directly inferred</td><td>8 (19.5%)</td></tr><tr><td></td><td>Clear and verifiable</td><td>7 (17.1%)</td></tr><tr><td></td><td>Indirectly inferred</td><td>3 (7.3%)</td></tr><tr><td></td><td>Mixed</td><td>1 (2.4%)</td></tr><tr><td>Type of deception</td><td>Fabrication</td><td>30(73.2%)</td></tr><tr><td></td><td>Mixed deception types</td><td>9 (22.0%)</td></tr><tr><td></td><td>Embedded lies</td><td>1 (2.4%)</td></tr><tr><td></td><td>Exaggeration / minimization</td><td>1 (2.4%)</td></tr></table>

Supplementary Table S2. Ground-truth classification and definitions
<table><tr><td>Level</td><td>Description</td></tr><tr><td>Clear and verifiable</td><td>Ground truth is fully transparent and can be independently verified for each statement.</td></tr><tr><td>Clear but not</td><td>Ground truth is experimentally assigned but individual statements cannot be</td></tr><tr><td>verifiable Directly inferred</td><td>externally verified.</td></tr><tr><td>Indirectly inferred</td><td>Ground truth is inferred from strong external indicators.</td></tr><tr><td>Mixed</td><td>Ground truth is based on weaker proxy indicators or subjective judgments. Dataset combines statements with different ground-truth reliability levels.</td></tr></table>

Note. Ground truth (GT) refers to the degree of confidence that the truthfulness of statements is established through empirical evidence rather than inference. To characterize the reliability of deception labels across datasets included in the Deception Archive, each dataset was assigned to one of the five ground-truth levels described below.