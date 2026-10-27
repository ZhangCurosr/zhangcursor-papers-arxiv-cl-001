# MATCHA: A Chinese Multi-Disorder Mental Health Dataset with Cross-Lingual Analyses on Social Media

Bingrui Jin<sup>1,\*</sup>, Baihan Li<sup>1,\*</sup>, Zitong Li<sup>1,\*</sup>, Kunyao Lan<sup>1</sup>, Shimeng Li<sup>1</sup>, Kenny Q. Zhu<sup>2</sup>, Mengyue Wu<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University, Shanghai, China <sup>2</sup>University of Texas at Arlington, Arlington, TX, USA {suzy\_jin, LBH0612, mengyuewu}@sjtu.edu.cn <sup>\*</sup>Equal contribution. Correspondence: mengyuewu@sjtu.edu.cn

## Abstract

Mental health research on social media has been dominated by English-language datasets and DSM-5-centered resources, limiting crosslingual and cross-cultural understanding of online mental health expression. We introduce MATCHA, the first Chinese multidisorder mental health dataset that links disorder-specific Weibo Super-Topic communities with users’ broader public timelines. MATCHA covers seven DSM-5 disorders and a control group, comprising 46,025 cohortlevel users and 1.8 million posts. The dataset distinguishes users with explicit selfdisclosure evidence from broader disorderrelated participants through an LLM-assisted annotation pipeline, and is released in a rigorously de-identified, controlled-access format. Using MATCHA, we analyze behavioral, linguistic, symptom-related, and life-eventrelated patterns across disorders, and conduct cross-lingual comparisons with Twitter-STMHD. Our analyses show that Weibo Super-Topics function as concentrated spaces for affective and health-related expression, while cross-platform diferences between Weibo and Twitter often exceed cross-disorder variation. MATCHA provides a new resource for community-aware and cross-lingual mental health NLP research.

## 1 Introduction

Social media platforms are now major channels for expressing emotions, sharing life experiences, and seeking peer support. The resulting usergenerated content captures language and behavior continuously, with high temporal resolution and near-real-time availability, providing a complementary evidence stream to surveys and clinical records. In English-language research, resources such as SMHD (Cohan et al., 2018) and

Twitter–STMHD (Suhavi et al., 2022) enable systematic analyses across multiple disorders and support detection, early identification, and risk modeling. Culture and language shape how mental-

![](images/0cd013bf7db65eee0ffcf5a70fa3fc9a1a59ab0ebaefd5fa8ac36a58c402b8ca.jpg)  
Figure 1: Overview of the MATCHA Dataset Collection and LLM -assisted Annotation Pipeline.

health experiences are expressed and interpreted (Kleinman, 1982). This motivates mental-health NLP resources that are sensitive to language and community context while remaining explicit about their relationship to major diagnostic taxonomies, such as DSM-5 and ICD-11 (American Psychiatric Association, 2013; World Health Organization, 2022).

Sina Weibo (often referred to as the “Twitter of China”<sup>1</sup>) and its unique Super-Topic mechanism provide an ideal setting to address this research gap. As one of China’s largest social platforms, Weibo reported about 591 million monthly active users and 261 million daily active users in the first quarter of 2025 (Weibo Corporation, 2025), reflecting its massive influence. Super-Topics are themed sub-communities that aggregate posts around specific interests, including health and mental well-being. This structure enables researchers tojointly analyze community-level discussions and individual-level posting histories, ofering a natural context to identify users interested in disorderrelated topics via corresponding Super-Topics. As a result, Weibo allows the organic, communityaware collection of mental health content at scale and supports systematic cross-lingual comparisons between Chinese and English-language social media environments.

We introduce MATCHA (Multi-disorder mental heAlth daTa in CHinA), a community-driven Chinese Weibo dataset for multi-disorder mental health research. Covering seven DSM-5 disorders and a control group, it links public user timelines with posts from disorder-specific Super-Topics, followed by de-identification and aggregation. A key feature is its LLM-assisted user stratification into Identified Positives (IP; explicit CEP-defined self-disclosure evidence, including eligible caregiver reports) and Disorder-Related Users (DRU; users discovered through participation in disorder-related Super-Topics, without diagnostic inference). Figure 1 shows the overview of the MATCHA dataset collection and annotation pipeline.

Homepage Posts provide a complementary, cue-reduced setting for studying whether cohortrelated signals extend beyond disorder-centered communities. Unlike Super-Topic Posts, where disorder-specific terminology and clinical cues are naturally concentrated, Homepage Posts capture users’ broader public-timeline language. This design allows us to ask whether user-level cohortrelated signals remain detectable in users’ broader public timelines, where explicit disorder and clinical cues are substantially less concentrated.

Although MATCHA is a Chinese corpus, its alignment with Twitter-STMHD (Suhavi et al., 2022) enables a descriptive cross-platform comparison. MATCHA is aligned with the self-reportoriented, user-level design of Twitter-STMHD while retaining its distinct Weibo Super-Topic candidate-selection and CEP annotation procedure. Twitter-STMHD uses pattern-based anchor tweets for user collection, whereas MATCHA uses disorder-specific Weibo Super-Topics to identify candidate users. To this end, our contributions are as follows:

• We introduce MATCHA, the first multi-disorder Chinese Weibo dataset covering 7 disorders, over 46k cohort-level users, and 1.8M posts, linking disorder-specific Super-Topic discourse with users’ broader public timelines. This dual-source design supports both communitycentered analysis and cue-reduced evaluation beyond disorder-specific communities. We further train DSM-5-grounded symptom and life-event classifiers to support scalable analysis.

• We establish a multi-level analytical framework that links community dynamics with individual linguistic and behavioral patterns, moving beyond conventional user- or post-level analysis of mental health discourse.

• We conduct a cross-lingual comparison with Twitter-STMHD, revealing divergent patterns in symptom expression and online behavior, while providing resources for cross-lingual mental health research.

## 2 Datasets

MATCHA covers seven DSM-5 mental health disorders—attention-deficit/hyperactivity disorder (ADHD), autism spectrum disorder (ASD), bipolar disorder (BD), depressive disorders (DEP), generalized anxiety disorder (GAD), obsessive– compulsive disorder (OCD), and schizophrenia (SZ)—along with a control group. The detailed distribution across diagnostic categories is illustrated in Figure 2. After deduplication within each cohort, the eight cohorts comprise 46,025 cohortlevel users and 1,834,699 unique posts. Because users and posts may occur in multiple disorder cohorts, these aggregated cohort-level counts are not globally deduplicated across cohorts.

## 2.1 Data Collection

Data was collected using Weibo’s oficial APIs under approved research protocols. We identified 1–3 active Super-Topics for each disorder and retrieved posts from two source streams. Super-Topic Posts (STP) are posts retrieved directly from disorderspecific Super-Topic community streams, whereas Homepage Posts (HP) are posts collected from the public personal timelines of users discovered through those communities. Thus, STP and HP are distinguished by retrieval source rather than by manual content categorization. The two sources serve complementary roles. STP captures concentrated disorder-centered discourse, where disorderspecific terminology and clinical cues are naturally more frequent, whereas HP captures users’ broader public-timeline language and provides a less cuesaturated setting for examining whether cohortrelated signals extend beyond disorder-centered discourse. STP and HP have slightly diferent temporal coverage, but 99.5% of disorder-cohort HP fall within the STP date range; detailed collection statistics are provided in Appendix A. We also constructed a control group by selecting users discussing neutral topics (e.g., “breakfast,” “shopping”) using 20 everyday keywords.

![](images/88d240508f696480dd15f1442fdbdfd873a9b35cb8131a92318f07162d97fa06.jpg)  
<sup>Share</sup> <sup>of</sup> <sup>users Share</sup> <sup>of</sup> <sup>posts</sup>Figure 2: Composition of MATCHA across disorder and control cohorts. Panel (a) reports cohort-level user counts and Panel (b) reports within-cohort deduplicated post counts. Across the eight cohorts, MATCHA contains 46,025 cohort-level users and 1,834,699 withincohort unique posts. Because cohorts are not mutually exclusive, these aggregated counts may include users or posts appearing in more than one cohort.

## 2.2 Data Annotation

Our annotation pipeline consists of two stages: LLM-assisted annotation for high-confidence mental health evidence identification, followed by human annotation for symptom and life-event labeling.

LLM-Assisted Data Annotation. Since participation in a Super-Topic does not necessarily imply a clinical diagnosis, we implement a strict LLMassisted annotation protocol to identify posts containing explicit mental health evidence (Gilardi et al., 2023). Specifically, we use a locally deployed Qwen3 model (Yang et al., 2025) to classify a post as a Clinical Evidence Post (CEP) if it contains one or more of the following signals related to the disorder: (1) an explicit medical diagnosis, (2) mention of prescription medication, (3) a direct self-disclosure or family declaration, or (4) evidence of hospitalization or ongoing treatment. For early-onset disorders such as ADHD and ASD, family-member reports are retained, whereas for other disorders only first-person disclosures are considered. Users with at least one CEP are labeled as Identified Positives (IP). To assess validity, we conducted a blinded manual audit of 200 posts: 100 LLM-positive and 100 LLM-negative posts, balanced across the seven disorders. Three researchers independently annotated every post under the CEP guidelines without access to the LLM prediction, and majority vote served as the reference. This audit evaluates agreement with the operational Clinical Evidence Post criterion rather than clinical diagnostic accuracy. Against the majority reference, the LLM achieved 88.0% precision, 83.0% recall, and 85.4% F1. Human reliability was high (87.3% average pairwise agreement; average pairwise Cohen’s $\kappa = 0 . 7 4 7 ;$ Fleiss’ $\kappa =$ 0.746; 81.0% unanimous agreement). On the unanimously labeled subset, LLM precision was 95.8%. The pipeline exhibits a slightly conservative error profile: its 12 false positives were concentrated in caregiver-oriented or educational/informational posts, especially for ASD, whereas its 18 false negatives more often involved implicit self-reference, weak medication mentions, or ADHD/ASD familymember reports.

Human Annotation. Mental-health-related expressions on social media are often implicit and context-dependent, making reliable annotation challenging. To support symptom and life-event annotation, we recruited 21 annotators with backgrounds in psychology, medicine, linguistics, or related disciplines. Annotators received DSM-5- based guideline training and completed pilot annotations before formal labeling.

The team annotated 13,723 unique Weibo posts across 38 symptom and 12 life-event categories following guidelines adapted from prior mentalhealth studies (Zhang et al., 2022; Lv et al., 2025). For posts with multiple independent annotations, labels were aggregated before classifier training. Among multiply annotated posts, macro-averaged Cohen’s κ across the symptom and life-event labels was 0.72. Detailed annotation and classifierpreparation statistics are provided in Appendix B.

## 2.3 Dataset Structure

The final dataset includes:

• Super-Topic Posts (STP): Posts from users in disorder-related Weibo Super-Topics.

• Homepage Posts (HP): Public personaltimeline posts from users discovered through disorder-related Super-Topics.

• Identified Positives (IP): A subset of DRUs with at least one Clinical Evidence Post (CEP), namely explicit textual self-disclosure evidence under the operational CEP criterion, including eligible ADHD/ASD family reports. IP is not a clinically verified diagnosis.

• Disorder-Related Users (DRU): Users who posted in disorder-related Super-Topics, irrespective of self-disclosure.

• Control Group: Users discussing neutral topics, with posts sampled using everyday keywords.

STP and HP therefore serve complementary rather than competing roles. STP captures concentrated, community-centered disorder discourse, whereas HP provides a broader and less cuesaturated public-timeline setting for examining whether cohort-related linguistic and behavioral signals persist beyond disorder-specific communities.

## 3 MATCHA Cross-Disorder Analysis

In this section, we present a multi-dimensional exploratory analysis of MATCHA, integrating behavioral, linguistic, and mental-state signals to characterize disorder-related communication on Weibo. We analyze (1) posting behaviors, (2) linguistic features via LIWC, and (3) symptom and life-event signals based on re-trained classifiers.

## 3.1 Posting Behavior

We first analyze how users interact with the platform across disorders and user types.

User Engagement and Posting Volume Identified Positive (IP) users, though a minority, contribute disproportionately to STP. As shown in Table 1, their per-user activity is 1.7–14.7 times higher than non-IP users. This is especially pronounced in ASD, where IPs (under 9% of STP users) generate nearly 59% of posts. Across all disorders, HP volumes substantially exceed STP volumes. However, STP/HP ratios vary across disorders, ranging from 0.18 for ASD to 0.06 for schizophrenia, suggesting diferent levels of engagement with disorder-specific communities.

Table 1: Posting activity comparison between CEPdefined IP and non-IP users within the Super-Topic Post (STP) population across disorders. IP denotes explicit textual self-disclosure evidence under the operational CEP criterion, not clinically verified diagnosis.
<table><tr><td>Disorder IP users [%]</td><td></td><td>IP posts [%]</td><td>STP/IP STP/non-IP</td></tr><tr><td>ADHD</td><td>23.5</td><td>34.5</td><td>7.6 4.4</td></tr><tr><td>ASD</td><td>8.9</td><td>58.9</td><td>50.1 3.4</td></tr><tr><td>BD</td><td>18.8</td><td>43.6 7.6</td><td>2.3</td></tr><tr><td>DEP</td><td>19.4</td><td>50.3 12.5</td><td>3.0</td></tr><tr><td>GAD</td><td>28.6</td><td>61.7 10.3</td><td>2.6</td></tr><tr><td>OCD</td><td>23.3</td><td>53.6 10.3</td><td>2.7</td></tr><tr><td>SZ</td><td>18.5</td><td>38.1 5.7</td><td>2.1</td></tr></table>

Temporal Patterns All disorder cohorts show higher user-normalized evening posting than the original control cohort (+0.5 to +6.3 percentage points), with the largest increases for DEP, BD, and GAD. Night-time elevation is not uniform: BD, DEP, and OCD have night ratios above the control baseline, whereas the remaining cohorts have lower ratios. Table 2 summarizes a usernormalized full-cohort measure; its estimand differs from the pooled-post hourly distributions used for the cross-platform correlation analysis.

Table 2: User-normalized evening and night-time posting metrics for the full disorder cohorts, computed on the merged STP+HP corpus after note-ID deduplication. Evening is each user’s 20:00–23:00 posting share averaged over users; ∆Evening is relative to the original control cohort; Night Ratio is the 22:00–05:00 share divided by the 10:00–17:00 share.
<table><tr><td>Disorder</td><td>Evening [%]</td><td>∆Evening [pp]</td><td>Night Ratio</td></tr><tr><td>ADHD</td><td>21.9</td><td>0.5</td><td>0.50</td></tr><tr><td>ASD</td><td>22.8</td><td>1.4</td><td>0.57</td></tr><tr><td>BD</td><td>27.2</td><td>5.8</td><td>1.18</td></tr><tr><td>DEP</td><td>27.7</td><td>6.3</td><td>1.20</td></tr><tr><td>GAD</td><td>26.8</td><td>5.4</td><td>0.72</td></tr><tr><td>OCD</td><td>26.2</td><td>4.8</td><td>0.94</td></tr><tr><td>SZ</td><td>26.5</td><td>5.1</td><td>0.79</td></tr><tr><td>Control</td><td>21.4</td><td>0.0</td><td>0.88</td></tr></table>

## 3.2 Linguistic Patterns

We conduct LIWC analysis on original Chinese Weibo posts using SC-LIWC (Gao et al., 2013). English LIWC2015 (Pennebaker et al., 2015) is applied to locally translated versions of the same posts as an auxiliary analysis. Because the Chinese and English analyses use diferent lexica and languages, we interpret Panel B as an auxiliary analysis rather than a numerically equivalent crosslingual measurement.

Table 3: LIWC comparisons between CEP-defined identified-positive (IP) users and final control users on Weibo. IP denotes explicit self-disclosure evidence under the operational Clinical Evidence Post criterion, rather than a clinically verified diagnosis. Panel A reports results from a Chinese LIWC lexicon applied to the original Chinese posts. Panel B reports selected categories from the English LIWC applied to LLM-translated posts; all source– disorder translation cohorts achieved exact note-ID coverage. All values are percentage-point diferences in category frequency (%) between the disorder group and the control group, reported separately for Super-Topic Posts (STP) and Homepage Posts (HP).  
Panel A: Chinese LIWC on original Chinese Weibo text
<table><tr><td rowspan="2">Cat.</td><td colspan="6">STP</td><td colspan="7">HP</td></tr><tr><td>ADHD</td><td>ASD</td><td>BD</td><td>DEP</td><td>GAD</td><td>OCD</td><td>SZ</td><td>ADHD</td><td>ASD</td><td>BD</td><td>DEP</td><td>GAD OCD</td><td>SZ</td></tr><tr><td colspan="10">Affective Processes</td><td></td><td></td><td></td><td></td></tr><tr><td>Positive emotion</td><td>-0.31-0.36</td><td></td><td>0.32</td><td>0.37</td><td>-0.51</td><td>-0.39 -0.34</td><td></td><td>0.47 -0.09</td><td>0.38</td><td>0.50</td><td>0.45</td><td>0.31</td><td>0.29</td></tr><tr><td>Negative emotion</td><td>0.67</td><td>0.08</td><td>2.39</td><td>2.50</td><td>2.84</td><td>3.21</td><td>2.25</td><td>0.18 0.20</td><td>0.94</td><td>1.14</td><td>0.91</td><td>0.95</td><td>0.70</td></tr><tr><td>Anxiety</td><td>0.11</td><td>0.03</td><td>0.47</td><td>0.52</td><td>1.80</td><td>1.79</td><td>0.46</td><td>0.05 0.05</td><td>0.15</td><td>0.19</td><td>0.34</td><td>0.36</td><td>0.13</td></tr><tr><td>Sadness</td><td>0.11</td><td>0.02</td><td>0.79</td><td>0.89</td><td>0.41</td><td>0.76</td><td>0.83</td><td>0.04 0.06</td><td>0.33</td><td>0.45</td><td>0.21</td><td>0.25</td><td>0.27</td></tr><tr><td>Anger</td><td>0.15</td><td>-0.04</td><td>0.42</td><td>0.40</td><td>0.16</td><td>0.31</td><td>0.36</td><td>0.02 -0.01</td><td>0.16</td><td>0.17</td><td>0.13</td><td>0.12</td><td>0.13</td></tr><tr><td colspan="10">Cognitive Processes</td><td colspan="3"></td></tr><tr><td>Cognitive processes</td><td>7.71</td><td>3.68</td><td>10.23</td><td>10.10</td><td>10.17</td><td>12.24</td><td>8.20</td><td>3.18 3.65</td><td>3.91</td><td>3.96</td><td>3.41</td><td>4.20</td><td>1.88</td></tr><tr><td>Insight</td><td>2.18</td><td>1.07</td><td>3.69</td><td>3.52</td><td>3.77</td><td>4.34</td><td>2.79</td><td>1.05 0.91</td><td>1.40</td><td>1.48</td><td>1.46</td><td>1.69</td><td>0.95</td></tr><tr><td>Discrepancy</td><td>1.22</td><td>0.44</td><td>1.94</td><td>1.90</td><td>1.43</td><td>2.26</td><td>1.73</td><td>0.53</td><td>0.56 0.77</td><td>0.89</td><td>0.68</td><td>0.83</td><td>0.43</td></tr><tr><td>Tentative</td><td>1.31</td><td>0.48</td><td>1.97</td><td>1.85</td><td>1.56</td><td>2.25</td><td>1.82</td><td>0.41</td><td>0.56 0.77</td><td>0.82</td><td>0.63</td><td>0.82</td><td>0.37</td></tr><tr><td colspan="10">Social Processes</td><td colspan="3"></td></tr><tr><td>Social processes</td><td>4.04</td><td>3.44</td><td>1.39</td><td>1.66</td><td>-0.93</td><td>0.19</td><td>1.06</td><td>1.76 2.73</td><td>0.97</td><td>1.04</td><td>0.33</td><td>0.68</td><td>1.10</td></tr><tr><td>Friends</td><td>-0.11</td><td>-0.05</td><td>0.06</td><td>0.07</td><td>-0.01</td><td>-0.00 0.04</td><td>-0.01</td><td>-0.02</td><td>0.02</td><td>0.01</td><td>0.02</td><td>0.00</td><td>0.00</td></tr><tr><td>Family</td><td>2.38</td><td>1.53</td><td>0.08</td><td>0.18</td><td>-0.09</td><td>-0.13</td><td>0.32</td><td>0.68 1.21</td><td>0.09</td><td>0.07</td><td>0.08</td><td>0.06</td><td>0.24</td></tr><tr><td>Humans</td><td>2.83</td><td>1.91</td><td>0.84</td><td>0.76</td><td>0.19</td><td>0.61</td><td>0.45</td><td>0.66 1.58</td><td>0.41</td><td>0.39</td><td>0.19</td><td>0.32</td><td>0.30</td></tr></table>

Panel B: EN translation Same column order as above: STP(7)  HP(7).
<table><tr><td colspan="13"></td></tr><tr><td>I</td><td>1.82</td><td>0.30</td><td>7.40</td><td>7.50</td><td>6.31</td><td>Linguistic Dimensions 6.16 5.51</td><td>1.49</td><td>0.62</td><td>3.41</td><td>3.72</td><td>3.29</td><td>3.31</td><td>2.08</td></tr><tr><td>We</td><td>0.23</td><td>0.37</td><td>-0.23</td><td>-0.20</td><td>-0.32</td><td>-0.20 -0.25</td><td>0.17</td><td>0.17</td><td>-0.03</td><td>-0.07</td><td>-0.10</td><td>-0.06</td><td>0.04</td></tr><tr><td>You</td><td>-0.30</td><td>-0.63</td><td>-0.22</td><td>-0.16</td><td>-0.44</td><td>-0.14 -0.26</td><td>0.13</td><td>-0.24</td><td>0.24</td><td>0.46</td><td>0.16</td><td>0.23</td><td>0.17</td></tr><tr><td>They</td><td>0.73</td><td>0.23</td><td>-0.00</td><td>0.13</td><td>-0.06</td><td>0.15 0.18</td><td>0.11</td><td>0.35</td><td>-0.00</td><td>-0.01</td><td>-0.05</td><td>0.00</td><td>-0.04</td></tr><tr><td>She/He</td><td>1.31</td><td>1.94</td><td>-0.35</td><td>-0.26</td><td>-0.66</td><td>-0.61 -0.15</td><td>0.81</td><td>0.74</td><td>-0.18</td><td>-0.25</td><td>-0.30</td><td>-0.20</td><td>-0.15</td></tr><tr><td>Verbs</td><td>4.80</td><td>2.32</td><td>7.79</td><td>8.24</td><td>7.00</td><td>7.11 7.06</td><td>2.93</td><td>2.27</td><td>3.62</td><td>3.85</td><td>3.64</td><td>3.78</td><td>2.11</td></tr><tr><td>Adverbs</td><td>1.56</td><td>0.75</td><td>3.61</td><td>4.25</td><td>3.64</td><td>3.58 2.80</td><td>1.15</td><td>0.97</td><td></td><td>1.77 1.91</td><td>2.31</td><td>2.34</td><td>0.56</td></tr><tr><td></td><td colspan="9">Time Orientations</td><td></td><td></td><td></td></tr><tr><td>Past focus</td><td>1.56</td><td>1.90</td><td>1.48</td><td>1.28</td><td>1.71</td><td>0.86 0.96</td><td>0.48</td><td>0.53</td><td>0.54</td><td>0.34</td><td>0.42</td><td>0.47</td><td>0.16</td></tr><tr><td>Present focus</td><td>2.50</td><td>0.35</td><td>5.50</td><td>6.31</td><td>4.62</td><td>5.50 5.54</td><td>2.45</td><td>1.42</td><td>2.70</td><td>3.22</td><td>2.67</td><td>2.96</td><td>1.89</td></tr><tr><td>Future focus</td><td>-0.04-0.09</td><td></td><td>0.17</td><td>0.24</td><td>0.07</td><td>0.17 -0.08</td><td>0.05</td><td>0.04</td><td>0.13</td><td>0.15</td><td>0.11</td><td>0.12</td><td>-0.00</td></tr></table>

For each LIWC category c, we compute its category frequency within a group as:

$$
f _ { c } = \frac { N _ { c } } { N _ { \mathrm { t o t a l } } } ,\tag{1}
$$

where $N _ { c }$ is the number of tokens assigned to category c, and $N _ { \mathrm { t o t a l } }$ is the total number of tokens in the corresponding group. We report the percentage-point diferences between CEPdefined Identified Positive (IP) users and controls for both Super-Topic Posts (STP) and Homepage

Posts (HP). Table 3 summarizes the results. Panel A reports Chinese LIWC results on original Chinese Weibo posts, while Panel B reports selected categories from English LIWC computed on translated text.

Primary findings from Chinese LIWC. Across disorders, we observe consistent afective and cognitive deviations in STP relative to the control group. Most disorder groups exhibit lower positive emotion and higher negative emotion usage in STPs. In parallel, cognitive process categories (including insight, discrepancy, and tentative) are substantially elevated across disorders, indicating stronger introspection, uncertainty, and self-reflection in disorder-related discussions.

Role of English LIWC. We additionally apply English LIWC to Chinese posts translated by a locally deployed LLM. English LIWC provides auxiliary estimates for narrative perspective and temporal orientation. It should not be treated as an independent replication of the Chinese LIWC results, because the selected English categories and lexical coverage difer from Panel A.

Linguistic dimensions and narrative perspective. Panel B shows cross-group diferences in pronoun, verb, and adverb categories. ADHD and ASD exhibit relatively stronger third-person pronoun signals (e.g., she/he), a pattern compatible with caregiver-mediated narration for early-onset conditions. In contrast, the other disorder cohorts show higher first-person-pronoun use (e.g., I) relative to controls, especially in STPs. Verbs and adverbs are also elevated across the reported cohorts. These are descriptive lexical patterns and do not identify the author or speaker of an individual post.

Temporal orientation. The Chinese LIWC inventory used here does not provide directly comparable tense-focus categories. We therefore report past, present, and future focus only as auxiliary English-LIWC measures. Present-focus is elevated across the reported cohorts, especially in STPs; past-focus is elevated for several cohorts, whereas future-focus diferences are small.

## 3.3 Exploratory Psychiatric Signal Analysis

We further analyze symptom-related and life-eventrelated signals using DSM-aligned bilingual classifiers trained on Weibo and Twitter data. These signals are treated as aggregate proxy indicators rather than clinical diagnoses. Detailed classifier construction and evaluation are provided in Appendix B.

To improve cross-lingual consistency and platform-specific robustness, we re-trained separate English and Chinese classifiers for Twitter and Weibo data using updated bilingual annotation procedures. Detailed classifier construction and evaluation results are provided in Appendix B. For all analyses below, per-post density for a given signal family C is:

$$
\begin{array} { r } { \mathrm { D e n s i t y } ( C ) = \frac { \sum _ { c \in C } { \# \mathrm { l a b e l s } ( c ) } } { \# \mathrm { p o s t s } } , } \end{array}
$$

computed within each disorder × post type.

Context Amplifies Mental Health Signals Figure 3 shows that STP densities exceed HP densities for both signal families in every disorder, with the magnitude of the diference varying by disorder and signal family. Super-topic Posts (STPs) show higher densities for both symptoms and life events, suggesting that STPs provide a more informationdense context for mental-health-related discourse than HPs.

![](images/fca4920e428acb7eed7b6d6513df219a904667a4564f16852b5112fa9d5d6806.jpg)

(a) Symptom labels per post  
![](images/7e517ec5272c6044268f21afb6b53be65763e93484105cabce2cb8fb1f502901.jpg)  
(b) Life-event labels per post  
Figure 3: Proxy-signal density, defined as the expected number of classifier-positive labels per post, for CEP-defined IP users’ STP and HP across disorders. The dashed line denotes the corresponding HP-control baseline. Classifier outputs are aggregate proxies rather than clinical assessments.

Disorder-Specific Content Profiles The magnitude of the STP–HP gap varies by disorder and signal family. Symptom density is highest in OCD and GAD STP, whereas life-event density is relatively high in ADHD and GAD STP. We interpret these classifier-derived diferences descriptively; they do not identify individual clinical status or establish the source of a cohort-level pattern.

## 3.4 Cue-Reduced Evaluation with Homepage Posts

We further examine whether cohort-related signals persist beyond disorder-centered communities. HP-only achieves an average Macro-F1 of 0.842 (AUROC 0.932), compared with 0.953 (0.995) for STP-only, while explicit disorder or clinical cues occur substantially less often in HP than STP (4.64% vs. 19.74%). After masking these cues, Macro-F1 decreases by 0.009 for HP and 0.017 for STP; the average reduction is larger for STP. These results position HP as a complementary cuereduced setting in which cohort-related signals remain detectable beyond disorder-specific communities. This benchmark predicts cohort membership rather than clinical diagnosis. Full experimental settings and per-disorder results are provided in Appendix D.

Across analyses, a consistent regularity is context amplification: STP concentrate substantially more classifier-derived proxy signal than HP, although the magnitude of this diference varies by disorder and proxy-signal family. This pattern supports the interpretation that STP ofer an information-dense window into mental-healthrelated discourse, whereas HP reflects broader everyday public-timeline language.

## 4 Cross-lingual Analysis

We compare mental health expression between Weibo (MATCHA) and Twitter-STMHD(Suhavi et al., 2022) across shared disorder categories. To ensure balanced comparison, we restrict the comparison to BD, DEP, GAD, and OCD, the four cohorts satisfying the shared-data and label-mapping requirements. ADHD, ASD, and SZ are not included in these cross-platform figures.

## 4.1 Linguistic Pattern Comparison

We extracted LIWC word frequencies from Twitter-STMHD using the same procedure as for MATCHA. To align with the self-reported STMHD setting, we use only CEP-defined Identified Positive users from MATCHA and merge each user’s Super-Topic and homepage posts. For each LIWC category c, we compute a cohort-control log ratio within each platform:

$$
\mathbf { L R } ( c ) = \log { \frac { f _ { \mathrm { d i s o r d e r } , c } + \varepsilon } { f _ { \mathrm { c o n t r o l } , c } + \varepsilon } }\tag{2}
$$

where $\varepsilon \ = \ 1 0 ^ { - 6 }$ Correlations over the 73 LIWC categories show stronger within-platform than cross-platform similarity: disorder groups are closer to other groups in the same language than to their disorder counterparts across platforms (Figure 4a). This pattern suggests that platform- and language-associated variation can exceed disorderspecific variation in LIWC signals.

We further inspect representative LIWC categories with large cohort-control deviations (Figure 5). Both platforms exhibit elevated afective expression, but MATCHA shows stronger increases in negative emotion, anxiety, sadness, and cognitive-processing categories. MATCHA users also display stronger present-focus patterns, whereas Twitter-STMHD exhibits relatively greater past-focus. In addition, life-related categories such as leisure and ingest are lower in MATCHA than in Twitter-STMHD. These comparisons are descriptive within-platform cohort– control contrasts rather than direct raw-frequency comparisons.

## 4.2 Disorder Presentation Comparison

We next compare normalized posting-hour, symptom, and life-event distributions. The Chinese and English classifiers are trained separately with bilingual annotations, so all results are interpreted as aggregate proxy trends rather than individual-level clinical claims.

Across these three views, groups cluster primarily by platform/language rather than disorder type. Within-platform correlations are generally higher than cross-platform correlations. This structure indicates substantial platform- and languageassociated variation in online disorder presentation, potentially reflecting the combined efects of platform afordances, community structure, sampling, and cultural context.

Figure 4c summarizes the life-event correlation pattern. As with the other cross-platform measures, these correlations are descriptive and may reflect diferences in sampling, label mapping, classifier behavior, language, and platform context.

Because classifier design, platform sampling, and label mapping difer across platforms, these symptom-composition comparisons are descriptive and should not be interpreted as evidence for clinical or causal cross-cultural mechanisms.

Figure 6 summarizes the average categorywise symptom-prevalence diferences (MATCHA minus Twitter–STMHD) across the four shared disorder cohorts. Avoidance is omitted because its English classifier estimates were unstable.

Summary Overall, the cross-platform analyses reveal larger between-platform/language differences than cross-disorder diferences across several measured dimensions, but these patterns should be interpreted descriptively because the present design cannot isolate cultural efects from platform, language, sampling, demographic, and community-level factors.

## 5 Related Work

Self-Report-Based Mental Health Social Media Datasets Foundational datasets such as SMHD (Cohan et al., 2018) and Twitter-STMHD (Suhavi et al., 2022) link social media users to DSM-5 disorders through explicit self-reporting. SMHD identifies diagnosed Reddit users via high-precision patterns while removing mentalhealth-related content to reduce topical bias, whereas STMHD extends this framework to Twitter with multi-year timelines across eight disorders. However, both primarily rely on explicit self-disclosure and may overlook users expressing symptoms indirectly, particularly in high-stigma contexts. Our work preserves their DSM-5 and user-level framework while incorporating disorder-related community participants identified through Super-Topic participation, with a separate LLM-assisted procedure used to identify explicit self-disclosure evidence. Instead of de-topicalization, we leverage Weibo Super Topics as organic community spaces for analyzing support and disorder-related discourse.

![](images/0f68fe0eb90020303f418fface1ba3847fbdf0ba6ad481bd53d8cb7a77caec4f.jpg)

![](images/cbd869812a699a6779f6e10489759e2c44956d0ac39a0c1681fb39e309695f82.jpg)

![](images/5443881a22c79953f4f1b3c60e746bec38aa87d150f6eabe46b337790a8bf1ae.jpg)

![](images/383238159a1a5794b8b0f9eb4fd14a94b624e9420d05dd6dea066d2262a97e23.jpg)  
Figure 4: Cross-lingual Pearson correlation matrices for (a) LIWC feature log ratios, (b) posting hours, (c) life-event distributions, and (d) symptom distributions. Cross-platform symptom correlations in (d) are lower on average than within-platform correlations, indicating substantial platform- and language-associated variation in symptom composition. These correlations are descriptive and do not identify causal cultural efects. Numbers in the upperright corner of each cell denote statistical significance levels: $1 = p < 0 . 0 0 1 , 2 = p < 0 . 0 1$ , and $3 = p < 0 . 0 5$

![](images/f6801633102ca081f1855961007d5aac44b0c80bd7e62cd5d1f013c44d304cc3.jpg)  
Figure 5: Cross-platform diferences in core LIWC features across BD, DEP, GAD, and OCD. Bars show within-platform cohort–control frequency diferences (percentage points) for CEP-defined MATCHA IP cohorts and frozen Twitter-STMHD cohorts.

![](images/bb61b97a15c96469d7adc50a536ec7842c6940b9ae77194002c5604fc3a75979.jpg)  
Figure 6: Cross-platform symptom-category distribution diferences between MATCHA and Twitter-STMHD. Avoidance is excluded from STMHD due to unstable English classifier behavior.

Chinese Social Media Mental Health Corpora In the Chinese setting, WU3D (Wang et al., 2022) provides a professionally annotated Weibo dataset for depression detection and serves as a benchmark for supervised classification. However, WU3D focuses on depression and does not provide MATCHA’s dual STP/HP retrieval design or its operational CEP-based IP definition. MATCHA instead covers multiple disorder cohorts and supports descriptive cross-platform comparison with English social-media data.

Early Detection and Shared Task Initiatives Resources such as RSDD (Yates et al., 2017), CLPsych 2015 (Coppersmith et al., 2015), and CLEF eRisk (Losada et al., 2017) established benchmarks for mental-health detection and earlyrisk prediction in English social media. These resources use platform-specific self-report or clinician-provided proxy labels; their collection and topical-filtering choices difer by resource. In contrast, we study disorder-related community participants on Chinese social media without treating community participation as evidence of diagnosis.

## 6 Conclusion

We present MATCHA, a large-scale Chinese mental health social media dataset covering seven DSM-5 disorders and a control group. By linking Weibo Super-Topic communities with userlevel timelines, MATCHA provides an ethically de-identified corpus that connects clinical concepts with real-world mental health discourse. Our analyses show that Super-Topics function as high-density contexts for disorder-related expression, featuring elevated negative afect, richer lifeevent and symptom signals, and higher user activity. Cross-lingual comparison further reveals that platform- and language-level diferences between Weibo and Twitter are more pronounced than cross-disorder variations, highlighting substantial variation associated with linguistic and platform environments in online mental health expression. Some observed diferences may also reflect classifier transfer efects and residual crosslingual annotation mismatches.

Our cue-reduced evaluation further shows that Homepage Posts retain cohort-related signals despite containing substantially fewer explicit disorder cues than Super-Topic Posts, highlighting the complementary roles of community-centered and broader timeline discourse. Together, MATCHA supports descriptive analysis, cross-platform comparison, and downstream cohort-level modeling of mental-health-related social-media language.

## Limitations

Several limitations should be considered when interpreting MATCHA and the analyses presented in this work.

First, MATCHA is constructed from public

Weibo Super-Topic communities and therefore does not represent the broader population of individuals with mental health conditions in China. Users who actively participate in disorder-related communities are likely more willing to disclose symptoms, seek support, or engage in mentalhealth-related discussion, which may bias linguistic and behavioral patterns toward highly engaged online populations.

Second, although the dataset is organized around DSM-5 disorder categories, online selfdisclosure and community participation do not constitute clinical diagnosis. Our Identified Positive (IP) labels rely on explicit textual evidence of diagnosis, treatment, self-disclosure, or eligible family reports, rather than verified medical records, and the symptom/life-event analyses are based on classifier-derived proxy signals intended only for aggregate-level comparisons. The cue-reduced benchmark likewise predicts cohort membership rather than clinical diagnosis and should not be interpreted as inferring mental disorders from ordinary social-media posts.

Third, cross-lingual comparisons between Weibo and Twitter-STMHD remain afected by substantial cultural, linguistic, and platformspecific diferences. Variations in moderation policies, posting norms, demographic composition, and discourse conventions may contribute to observed diferences independently of mentalhealth status. In addition, LIWC categories are not perfectly aligned across Chinese and English, even when translation-based robustness checks are applied.

Fourth, the dataset currently focuses on seven DSM-5 disorders and does not cover the full spectrum of psychiatric or neurodevelopmental conditions. Comorbidity, symptom overlap, and temporal disorder progression are also not explicitly modeled in the current release.

Finally, despite rigorous de-identification and controlled-access release procedures, mentalhealth social-media data inherently involve privacy and ethical risks. Future work should continue exploring stronger privacy-preserving release mechanisms, culturally adaptive annotation frameworks, and clinically informed evaluation protocols.

## Ethical Considerations

All data were collected from publicly accessible posts on Weibo through the platform’s oficial APIs under an approved research application, in accordance with Weibo’s terms of service. No private, deleted, or restricted content was accessed. The data collection process was non-intrusive and did not interfere with normal user activities. Individual informed consent was not sought from Weibo users because this study used only publicly accessible posts retrieved through the platform’s oficial APIs under an approved research application. To mitigate privacy risks, we de-identified the data and release only controlled-access, aggregate statistics and derived features.

Upon publication, the MATCHA dataset is available through a controlled-access mechanism for academic, non-commercial research only. The released data will not contain raw user identifiers or direct personal information; instead, they will consist of de-identified, aggregated statistics and derived features. The accompanying license will explicitly prohibit any attempt at user re-identification or deanonymization. Documentation, the Data Use Agreement, and controlled-access request instructions are available at https://github.com/Baihan-12/ MATCHA-Mental-Health-Dataset.

To protect user privacy, all data underwent strict de-identification, including the removal or hashing of usernames, user IDs, and URLs. For crosslingual analysis, Chinese posts were translated solely for analytical purposes, which additionally reduces the risk of identifying individual writing styles. All processing was conducted using locally deployed large language models in ofline environments, with no external data transmission, further minimizing privacy and security risks.

Potential risks include privacy leakage, reidentification attempts, stigmatization, and inappropriate use of cohort-level or classifier-derived signals for individual-level clinical inference. We mitigate these risks through controlled access, de-identification, explicit prohibitions on reidentification, and by restricting all reported analyses to aggregate-level research purposes.

## Acknowledgements

This work has been supported by the China NSFC Project (No. 62572320).

## References

American Psychiatric Association. 2013. Diagnostic and Statistical Manual ofMental Disorders, 5th edition. American Psychiatric Association.

Siyuan Chen, Meilin Wang, Minghao Lv, Zhiling Zhang, Qianqian Ju, Dejiyangla, Yujia Peng, Kenny Q. Zhu, and Mengyue Wu. 2024. Mapping long-term causalities in psychiatric symptomatology and life events from social media. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5472–5487, Mexico City, Mexico. Association for Computational Linguistics.

Arman Cohan, Bart Desmet, Andrew Yates, Luca Soldaini, Sean MacAvaney, and Nazli Goharian. 2018. SMHD: a large-scale resource for exploring online language usage for multiple mental health conditions. In Proceedings ofthe 27th International Conference on Computational Linguistics, pages 1485– 1497, Santa Fe, New Mexico, USA. Association for Computational Linguistics.

Glen Coppersmith, Mark Dredze, Craig Harman, Kristy Hollingshead, and Margaret Mitchell. 2015. CLPsych 2015 shared task: Depression and PTSD on Twitter. In Proceedings of the 2nd Workshop on Computational Linguistics and Clinical Psychology: From Linguistic Signal to Clinical Reality, pages 31–39, Denver, Colorado. Association for Computational Linguistics.

Rui Gao, Bibo Hao, He Li, Yusong Gao, and Tingshao Zhu. 2013. Developing simplified chinese psychological linguistic analysis dictionary for microblog. In Brain and Health Informatics, pages 359–368, Cham. Springer International Publishing.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy ofSciences, 120(30):e2305016120.

Arthur Kleinman. 1982. Neurasthenia and depression: A study of somatization and culture in china. Culture, Medicine and Psychiatry, 6(2):117–190.

David E. Losada, Fabio Crestani, and Javier Parapar. 2017. CLEF 2017 erisk overview: Early risk prediction on the internet (experimental foundations). In Working Notes of CLEF 2017 – Conference and Labs of the Evaluation Forum, volume 1866 of CEUR Workshop Proceedings, Dublin, Ireland. CEUR-WS.org.

Minghao Lv, Siyuan Chen, Haoan Jin, Minghao Yuan, Qianqian Ju, Yujia Peng, Kenny Q. Zhu, and Mengyue Wu. 2025. Tracking life’s ups and downs: Mining life events from social media posts for mental health analysis. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6950–6965, Vienna, Austria. Association for Computational Linguistics.

James W. Pennebaker, Ryan Boyd, Kayla Jordan, and Kate Blackburn. 2015. The Development and Psychometric Properties of LIWC2015. University of Texas at Austin.

Suhavi, Asmit Kumar Singh, Udit Arora, Somyadeep Shrivastava, Aryaveer Singh, Rajiv Ratn Shah, and Ponnurangam Kumaraguru. 2022. Twitter-STMHD: An extensive user-level database of multiple mental health disorders. In Proceedings of the International AAAI Conference on Web and Social Media, volume 16, pages 1182–1191. AAAI Press.

Yiding Wang, Zhenyi Wang, Chenghao Li, Yilin Zhang, and Haizhou Wang. 2022. Online social network individual depression detection using a multitask heterogenous modality fusion approach. Information Sciences, 609:727–749.

Weibo Corporation. 2025. Weibo announces first quarter 2025 unaudited financial results. Technical report, Weibo Corporation. Accessed on October 7, 2025.

World Health Organization. 2022. International classification of diseases, 11th revision (icd-11). Technical report, World Health Organization. Accessed on October 7, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Andrew Yates, Arman Cohan, and Nazli Goharian. 2017. Depression and self-harm risk assessment in online forums. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 2968–2978, Copenhagen, Denmark. Association for Computational Linguistics.

Zhiling Zhang, Siyuan Chen, Mengyue Wu, and Kenny Zhu. 2022. Symptom identification for interpretable detection of multiple mental disorders on social media. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9970–9985, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

## A Additional Dataset Collection Details

Temporal coverage. The main STP collection spans May 13, 2014–January 14, 2025. Disordercohort HP extend further back in time, with the earliest retrieved post dated April 10, 2010, because homepage retrieval provides historical public-timeline context for users identified through Super-Topic communities. Control HP span October 17, 2009–December 29, 2024.

Despite this broader nominal range, the temporal mismatch is small in practice: 1,551,872 of 1,559,045 disorder-cohort HP (99.5%) fall within the STP date range. Of the 7,173 posts outside this range, 7,070 precede May 13, 2014, reflecting the longer historical reach of homepage retrieval, while only 103 occur after January 14, 2025. The latter comprise 103 additional GAD homepage posts from 10 users, extending to September 30, 2025, without corresponding additional STP collection. Restricting HP to the STP date range retains 30,843 of 30,851 users (>99.9%). Thus, nearly all HP data overlap temporally with STP, making it unlikely that the broader nominal HP range alone accounts for the observed diferences between the two sources.

Homepage-post retrieval. The released data are consistent with homepage retrieval of up to approximately 100 recent public-timeline posts per user at a retrieval point. Some users have more than 100 unique HP in the final corpus, consistent with repeated retrieval and subsequent post-ID deduplication. Consequently, the final number of unique HP per user is not strictly capped at 100. Across disorder cohorts, users have a median of 36 HP (IQR: 5– 96; mean: 50.5); 10.0% have exactly 100 posts and 4.0% have more than 100. The earliest retrieved HP precedes a user’s first observed STP by a median of 222 days (IQR: 37–714 days), reflecting the broader historical timeline context captured by HP.

Super-Topic search. For each disorder, we queried one to three public Weibo Super-Topics using common Chinese disorder names and terminology variants. Table 4 records the disorder cohorts and their corresponding Super-Topic terms. It documents the initial collection configuration; final corpus counts arise only after downstream user expansion and post-ID deduplication and are therefore not expected to equal the final corpus counts reported in the main paper. The Super-Topic mapping identifies public discussion communities and does not establish that any participant has a clinical diagnosis.

Control anchor search. The control cohort was constructed through a separate everyday-topic search pipeline rather than through a Super-Topic. We used 20 everyday-life keywords: 上班 (shangban, ‘going to work’), 早餐 (zaocan, ‘breakfast’), 聊 天 (liaotian, ‘chatting’), 购 物 (gouwu, ‘shopping’), 健 身 (jianshen, ‘fitness’), 音 乐 (yinyue, ‘music’), 旅行 (lvxing, ‘travel’), 逛街 (guangjie, ‘shopping or strolling’), 火锅 (huoguo, ‘hot pot’), 小 吃 (xiaochi, ‘snacks’), 生 日 (shengri, ‘birthday’), 假期 (jiaqi, ‘holiday’), 手机 (shouji, ‘mobile phone’), 游戏 (youxi, ‘games’), 电话 (dianhua, ‘telephone’), 分享 (fenxiang, ‘sharing’), 天气 (tianqi, ‘weather’), 晚 上 (wanshang, ‘evening’), 公交 (gongjiao, ‘public bus’), and 路上 (lushang, ‘on the road’). This search produced 3,992 raw anchor posts before downstream user expansion and deduplication.

## B Life Event and Symptom Classifiers

To support cross-lingual psychiatric signal analysis, we constructed separate English and Chinese classifiers for symptom-related and life-eventrelated categories. Our approach builds upon prior DSM-aligned psychiatric signal frameworks, including PsySym for symptom categories and prior work on psychiatric life-event analysis (Chen et al., 2024; Lv et al., 2025; Zhang et al., 2022).

Annotation. The merged annotation corpus contains 13,723 unique posts labeled by a team of 21 annotators across 38 symptom and 12 life-event categories. For each label, annotators assigned 1 if the post mentioned the category and the author was currently experiencing it or still afected by it, 0 if the category was mentioned but was not clearly attributable to the author’s current experience, and −1 if the category was not mentioned. For posts with multiple independent annotations, labels were aggregated before classifier training. All annotators were fairly compensated at or above the local living wage, with rates determined prior to annotation.

Annotator recruitment and compensation. We recruited 21 annotators through social-media recruitment. All annotators were based in China and had backgrounds in psychology, medicine, linguistics, or related fields. Annotators received a total of RMB 13,179.79 for 22,153 completed annotation records, corresponding to RMB 0.595 per record. The expected annotation time was approximately one minute per record, yielding an estimated compensation rate of RMB 35.70 per hour. Rates were determined before annotation and were at or above the applicable local living-wage benchmark.

Inter-annotator agreement. Inter-annotator agreement was assessed on the 6,612 posts with at least two independent annotators and valid 50-dimensional label vectors. Cohen’s κ was computed per label from pairwise annotator comparisons, including only instances for which both annotators judged the category to be mentioned (0 or 1); cells coded as not mentioned (−1) were excluded. One label lacked suficient variance in this subset and was excluded. The macro-average over the 49 evaluable labels was κ = 0.72.

Data preparation. Since our classifiers target relevance, i.e., whether a symptom or life event is mentioned in a post, both 0 and 1 were treated as positive votes during majority-voting aggregation across annotators, while −1 was treated as a negative vote. The aggregated label therefore reflects whether annotators reached majority consensus on relevance.

For symptom classification, we applied a disease mask: on symptom dimensions atypical for a post’s disorder, aggregated 0 labels were set to −1 and excluded from loss computation, avoiding penalization for plausible false negatives. For lifeevent classification, no disease mask was applied. Instead, −1 labels were treated as negative votes during aggregation, such that cells where the majority vote was −1 received a final label of 0 (no event).

To mitigate class imbalance, we applied a negative downsampling ratio of 10:1 per label column, masking excess negative samples to −1 and excluding them from loss computation. Posts were split into train, validation, and test sets at 70%/10%/20% using iterative stratified splitting, ensuring suficient positive examples for rare labels in both validation and test sets.

Classifier training. We conducted grid search over two model families for each task.

For symptom classification:

• Chinese-MentalBERT (∼102M trainable parameters): learning rate $\in \{ 1 \times 1 0 ^ { - 4 } , 2 \times$

Table 4: Disorder cohorts and their corresponding public Weibo Super-Topics. Chinese terms are accompanied by pinyin transliterations and English glosses.
<table><tr><td>Cohort</td><td>Weibo Super-Topic</td></tr><tr><td>DEP</td><td>抑郁症超话 (yiyuzheng chaohua, &#x27;Depression Super-Topic&#x27;)</td></tr><tr><td>DEP</td><td>抑郁症患者超话 (yiyuzheng huanzhe chaohua, &#x27;Depression Patient Super-Topic&#x27;)</td></tr><tr><td>DEP</td><td>抑郁超话 (yiyu chaohua, Depression Super-Topic&#x27;)</td></tr><tr><td>SZ</td><td>精神分裂症超话 (jingshen fenliezheng chaohua, Schizophrenia Super-Topic&#x27;)</td></tr><tr><td>GAD</td><td>焦虑症超话 (jiaolvzheng chaohua, Anxiety Disorder Super-Topic&#x27;)</td></tr><tr><td>ADHD</td><td>多动症超话 (duodongzheng chaohua, &#x27;Hyperactivity Disorder Super-Topic&#x27;)</td></tr><tr><td>ASD</td><td>自闭症超话(zibizheng chaohua, Autism Super-Topic&#x27;)</td></tr><tr><td>BD</td><td>躁郁症超话(zaoyuzheng chaohua, Manic-Depressive Disorder Super-Topic&#x27;)</td></tr><tr><td>BD</td><td>双相情感障碍超话 (shuangxiang qinggan zhangai chaohua, &#x27;Bipolar Affective Disorder Super-Topic&#x27;)</td></tr><tr><td>OCD</td><td>强迫症超话 (qiangpozheng chaohua, Obsessive-Compulsive Disorder Super-Topic&#x27;)</td></tr></table>

$1 0 ^ { - 4 } \}$ , batch size $\in \ \left\{ 1 6 , 3 2 \right\}$ , max length ∈ {64, 128, 192}.

• Qwen3-Embedding-8B (frozen backbone, ∼49.2K trainable parameters): learning rate $\in \{ 5 \times 1 0 ^ { - 5 } , 1 \times 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } \}$ , batch size $\in \{ 8 , 1 6 \}$ , max length ∈ {256, 512}.

Chinese-MentalBERT was selected based on validation macro-AUC. The final symptom checkpoint used learning rate $1 \times 1 0 ^ { - 4 }$ , batch size 32, and maximum length 128, and achieved validation macro-AUC 0.9502.

For life-event classification, Qwen3- Embedding-8B was selected based on validation macro- $. \mathrm { A U C } .$ . The final checkpoint used learning rate $5 \times 1 0 ^ { - 5 }$ , batch size 8, and maximum length 256, and achieved validation macro-AUC 0.9542.

In both tasks, −1 labels were masked from the binary cross-entropy loss during training. Positive class weights were set to $\sqrt { n _ { \mathrm { n e g } } / n _ { \mathrm { p o s } } }$ per label to further mitigate imbalance. Final checkpoints were selected based on validation macro-AUC and reevaluated on held-out test sets.

English classifiers. For cross-lingual comparison, we re-trained English classifiers on Twitter data to mitigate potential domain shift from the original Reddit-based models. Candidate annotations were obtained by applying existing English symptom and life-event classifiers to Twitter-STMHD, followed by verification using Qwen3-8B. The resulting dataset was matched in scale to the Chinese annotation set and used to train updated English classifiers based on mental-bert-base-uncased, following the same training procedure as the Chinese classifiers.

Annotation statistics. Table 8 and Table 9 provide detailed per-label annotation statistics for the

Table 5: Annotation statistics for the Chinese training data. The 22,153 records are shared post–annotator records annotated for both task families; the two columns are not additive.
<table><tr><td></td><td>Symptom</td><td>Life Event</td></tr><tr><td>Number of labels</td><td>38</td><td>12</td></tr><tr><td>Annotated posts</td><td>13,723</td><td>13,723</td></tr><tr><td>Annotation records</td><td>22,153</td><td>22,153</td></tr><tr><td>Median positive rate</td><td>2.26%</td><td>1.47%</td></tr><tr><td>Positive rate range</td><td>0.78-16.39%</td><td>0.26–43.94%</td></tr><tr><td>Most frequent label</td><td>Depressed Mood</td><td>Health</td></tr><tr><td>Most frequent rate</td><td>16.39%</td><td>43.94%</td></tr><tr><td>Least frequent label</td><td>Impulsivity</td><td>Legal</td></tr><tr><td>Disease mask</td><td>Yes</td><td>No</td></tr><tr><td>Negative downsampling</td><td>10:1</td><td>10:1</td></tr><tr><td>Train / Val / Test</td><td>70/10/20</td><td>70/10/20</td></tr></table>

Chinese symptom and life-event datasets, respectively. We report both the number of strictly positive annotations (label value 1) and the total number of relevance annotations (0+1) prior to majority-vote aggregation. Because the classification objective focuses on whether a symptom or life event is mentioned in a post, both 0 and 1 are treated as relevance-positive signals during aggregation, while −1 denotes absence of mention. The tables further illustrate the substantial class imbalance present across both tasks, motivating the use of disease masking, negative downsampling, and class-balanced loss weighting during classifier training.

Classifier usage. The resulting classifiers were used only for aggregate-level proxy signal analysis in the main paper. All symptom- and life-eventrelated results should therefore be interpreted as distributional trends rather than clinical assessments or individual-level diagnoses.

Table 6: Chinese classifier evaluation on a held-out test set (n = 2,745). Per-label decision thresholds were selected on the validation set and applied once to the heldout test set.
<table><tr><td>Task</td><td>Macro AUC</td><td>Micro AUC</td><td>Macro F1</td></tr><tr><td>Symptom</td><td>0.9374</td><td>0.9494</td><td>0.5579</td></tr><tr><td>Life Event</td><td>0.9533</td><td>0.9673</td><td>0.6449</td></tr></table>

Table 7: English classifier evaluation results on heldout test sets.
<table><tr><td>Task</td><td>Macro AUC</td><td>Micro AUC</td><td>Macro F1</td></tr><tr><td>Symptom</td><td>0.9910</td><td>0.9942</td><td>0.8852</td></tr><tr><td>Life Event</td><td>0.9769</td><td>0.9767</td><td>0.7669</td></tr></table>

## C Psychiatric Signal Taxonomy

## C.1 Symptom Taxonomy

To support interpretable and cross-lingual psychiatric signal analysis, we constructed a bilingual symptom taxonomy aligned with DSM-5-oriented clinical concepts. The taxonomy covers afective, cognitive, behavioral, somatic, and trauma-related symptom dimensions adapted from prior psychiatric NLP frameworks (Zhang et al., 2022). Each Chinese symptom category was manually aligned with its English counterpart to improve semantic consistency across platforms and languages.

Tables 10 and 11 present the complete symptom taxonomy used in our analyses.

## C.2 Life-Event Taxonomy

In addition to symptom-related signals, we annotated major life events frequently associated with mental-health experiences on social media. The life-event taxonomy was adapted from prior work on psychiatric life-event modeling (Lv et al., 2025; Chen et al., 2024) and manually aligned across Chinese and English settings.

The taxonomy spans multiple dimensions of daily life, including health, relationships, education, work, finances, relocation, and identityrelated experiences. Table 12 presents representative categories and example sub-events.

## D Homepage Posts as a Cue-Reduced Testbed

## D.1 Experimental Setup

To evaluate the complementary value of Homepage Posts (HP), we formulate seven binary disorder-versus-control cohort-classification tasks covering ADHD, ASD, BD, DEP, GAD, OCD, and SZ. We compare three input settings: STPonly, using posts retrieved from disorder-specific Super-Topics; HP-only, using users’ broader public homepage timelines; and STP+HP, combining both sources.

All experiments use user-level stratified train/validation/test splits and are repeated with five random seeds. We use TF–IDF features with logistic regression and report Macro-F1 and AUROC. To ensure deterministic construction of the STP+HP condition, user IDs are sorted before aggregation and PYTHONHASHSEED is fixed across runs.

## D.2 Cue Prevalence and Cohort Classification

Table 13 reports the prevalence of explicit disorder and clinical cues at the post level. Disorderrelated cues are substantially more concentrated in STP than HP: 13.88% of STP contain a disordername cue, compared with 3.72% of HP. Considering either disorder or clinical cues, the prevalence is 19.74% for STP and 4.64% for HP.

Despite the lower cue prevalence, HP remains predictive of cohort membership. Under the original-text condition, HP achieves an average Macro-F1 of 0.842, while STP achieves 0.953. This performance gap is partly expected because STP is collected directly from disorder-specific communities, where explicit topical cues are more concentrated.

## D.3 Cue-Masking Analysis

We next test whether classification performance is primarily driven by explicit terminology. We apply the same masking procedure symmetrically to STP and HP under three conditions: A, original text; B, disorder-name cues masked; and C, disorder-name and clinical cues masked.

Table 14 summarizes the seven-disorder average results. After masking disorder and clinical cues, STP Macro-F1 decreases from 0.953 to 0.936 $( \Delta = - 0 . 0 1 7 )$ , whereas HP decreases from 0.842 to 0.833 $( \Delta = - 0 . 0 0 9 )$ . Thus, the performance reduction for STP is approximately twice that of HP.

Table 15 reports the per-disorder comparison between the original condition (A) and the most restrictive masking condition (C). The magnitude of the masking-related decrease difers across disorders; the largest STP decreases occur for ADHD (0.043), ASD (0.030), and OCD (0.018).

Table 8: Per-label annotation statistics for symptom labels. Positive counts correspond to value 1, while mention counts include both 0 and 1, i.e., instances in which the corresponding category was mentioned in the post.
<table><tr><td>#</td><td>Label</td><td>Positive (1)</td><td>Positive %</td><td>Mentioned (0+1)</td><td>Mentioned %</td></tr><tr><td>1</td><td>Irritable Mood</td><td>372</td><td>2.71</td><td>499</td><td>3.64</td></tr><tr><td>2</td><td>Anxious Mood</td><td>1,498</td><td>10.92</td><td>1,835</td><td>13.37</td></tr><tr><td>3</td><td>Autonomic Symptoms</td><td>834</td><td>6.08</td><td>899</td><td>6.55</td></tr><tr><td>4</td><td>Cardiovascular Symptoms</td><td>740</td><td>5.39</td><td>786</td><td>5.73</td></tr><tr><td>5</td><td>Catatonic Behavior</td><td>197</td><td>1.44</td><td>244</td><td>1.78</td></tr><tr><td>6</td><td>Fatigue or Loss of Energy</td><td>550</td><td>4.01</td><td>622</td><td>4.53</td></tr><tr><td>7</td><td>Depressed Mood</td><td>2,249</td><td>16.39</td><td>2,645</td><td>19.27</td></tr><tr><td>8</td><td>Gastrointestinal Symptoms</td><td>531</td><td>3.87</td><td>591</td><td>4.31</td></tr><tr><td>9</td><td>Genitourinary Symptoms</td><td>134</td><td>0.98</td><td>185</td><td>1.35</td></tr><tr><td>10</td><td>Hyperactivity / Agitation</td><td>176</td><td>1.28</td><td>250</td><td>1.82</td></tr><tr><td>11</td><td>Impulsivity</td><td>107</td><td>0.78</td><td>282</td><td>2.05</td></tr><tr><td>12</td><td>Inattention</td><td>248</td><td>1.81</td><td>355</td><td>2.59</td></tr><tr><td>13</td><td>Indecisiveness</td><td>128</td><td>0.93</td><td>151</td><td>1.10</td></tr><tr><td>14</td><td>Respiratory Symptoms</td><td>657</td><td>4.79</td><td>703</td><td>5.12</td></tr><tr><td>15</td><td>Suicidal Ideation</td><td>783</td><td>5.71</td><td>899</td><td>6.55</td></tr><tr><td>16</td><td>Worthlessness / Guilt</td><td>642</td><td>4.68</td><td>756</td><td>5.51</td></tr><tr><td>17</td><td>Avoidance of Trauma Cues</td><td>120</td><td>0.87</td><td>176</td><td>1.28</td></tr><tr><td>18</td><td>Compensatory Behavior</td><td>159</td><td>1.16</td><td>187</td><td>1.36</td></tr><tr><td>19</td><td>Compulsive Behavior</td><td>154</td><td>1.12</td><td>258</td><td>1.88</td></tr><tr><td>20</td><td>Diminished Emotional Expression</td><td>320</td><td>2.33</td><td>387</td><td>2.82</td></tr><tr><td>21</td><td>Reckless / Self-Destructive Behavior</td><td>151</td><td>1.10</td><td>227</td><td>1.65</td></tr><tr><td>22</td><td>Mood / Energy Fluctuation</td><td>345</td><td>2.51</td><td>412</td><td>3.00</td></tr><tr><td>23</td><td>Social Anxiety / Avoidance</td><td>342</td><td>2.49</td><td>475</td><td>3.46</td></tr><tr><td>24</td><td>Fear of Weight Gain</td><td>182</td><td>1.33</td><td>202</td><td>1.47</td></tr><tr><td>25</td><td>Fear of Negative Evaluation</td><td>127</td><td>0.93</td><td>157</td><td>1.14</td></tr><tr><td>26</td><td>Flight of Ideas</td><td>179</td><td>1.30</td><td>212</td><td>1.54</td></tr><tr><td>27</td><td>Intrusive Symptoms</td><td>138</td><td>1.01</td><td>152</td><td>1.11</td></tr><tr><td>28</td><td>Anhedonia / Loss of Interest</td><td>386</td><td>2.81</td><td>457</td><td>3.33</td></tr><tr><td>29</td><td>More Talkative</td><td>115</td><td>0.84</td><td>182</td><td>1.33</td></tr><tr><td>30</td><td>Obsession</td><td>408</td><td>2.97</td><td>564</td><td>4.11</td></tr><tr><td>31</td><td>Panic / Fear</td><td>450</td><td>3.28</td><td>528</td><td>3.85</td></tr><tr><td>32</td><td>Pessimism</td><td>302</td><td>2.20</td><td>338</td><td>2.46</td></tr><tr><td>33</td><td>Poor Memory</td><td>236</td><td>1.72</td><td>275</td><td>2.00</td></tr><tr><td>34</td><td>Sleep Disturbance</td><td>866</td><td>6.31</td><td>971</td><td>7.08</td></tr><tr><td>35</td><td>Musculoskeletal Symptoms</td><td>579</td><td>4.22</td><td>660</td><td>4.81</td></tr><tr><td>36</td><td>Other Somatic Symptoms</td><td>556</td><td>4.05</td><td>644</td><td>4.69</td></tr><tr><td>37</td><td>Sensory Symptoms</td><td>482</td><td>3.51</td><td>522</td><td>3.80</td></tr><tr><td>38</td><td>Weight and Appetite Change</td><td>504</td><td>3.67</td><td>593</td><td>4.32</td></tr></table>

Table 9: Per-label annotation statistics for life-event labels.
<table><tr><td colspan="5"># Label Pos. Pos. [%] Mentioned Mentioned [%]</td></tr><tr><td>1 Bereavement</td><td>90</td><td>0.66</td><td>196</td><td>1.43</td></tr><tr><td>2 Health</td><td>6,030</td><td>43.94</td><td>7,409</td><td>53.99</td></tr><tr><td>3 Relationship Change</td><td>303</td><td>2.21</td><td>658</td><td>4.79</td></tr><tr><td>4 Family Member Change</td><td>46</td><td>0.34</td><td>199</td><td>1.45</td></tr><tr><td>5 Relocation</td><td>116</td><td>0.85</td><td>228</td><td>1.66</td></tr><tr><td>6 Education</td><td>236</td><td>1.72</td><td>431</td><td>3.14</td></tr><tr><td>7 Legal</td><td>35</td><td>0.26</td><td>224</td><td>1.63</td></tr><tr><td>8 Lifestyle Change</td><td>370</td><td>2.70</td><td>562</td><td>4.10</td></tr><tr><td>9 Social Event</td><td>106</td><td>0.77</td><td>213</td><td>1.55</td></tr><tr><td>10 Occupation</td><td>334</td><td>2.43</td><td>524</td><td>3.82</td></tr><tr><td>11 Financial</td><td>175</td><td>1.28</td><td>365</td><td>2.66</td></tr><tr><td>12 Identity</td><td>54</td><td>0.39</td><td>194</td><td>1.41</td></tr></table>

The tiered masking results in Table 14 further show that most of the performance reduction occurs when disorder-name cues are removed. Masking additional clinical terminology produces little further change. This pattern is strongest for STP and substantially weaker for HP.

Together, these results support the complementary role of HP as a cue-reduced evaluation setting. STP contains more concentrated explicit disorder cues and achieves higher absolute classification performance, but is also more sensitive to cue removal. In contrast, HP contains substantially fewer explicit cues while retaining comparable performance after masking. We therefore interpret STP and HP as distinct discourse settings rather than redundant feature sources: STP captures community-centered explicit discourse, whereas HP supports evaluation of broader publictimeline cohort-related signals.

We emphasize that this benchmark predicts membership in MATCHA’s community-derived cohorts rather than clinical diagnosis. The results should therefore not be interpreted as evidence that mental disorders can be clinically inferred from ordinary social-media posts.

Table 10: Complete bilingual symptom taxonomy used in MATCHA (Part 1).
<table><tr><td>English Label</td><td>Semantic Gloss</td></tr><tr><td>Irritable Mood</td><td>Persistent irritability, anger, frustration, low emotional tolerance</td></tr><tr><td>Anxious Mood</td><td>Excessive worry, panic anticipation, fear of catastrophic outcomes</td></tr><tr><td>Autonomic</td><td>Sweating, accelerated heart rate, dry</td></tr><tr><td>Symptoms Cardiovascular</td><td>mouth, shortness of breath Chest pain, palpitations, cardiovascular</td></tr><tr><td>Symptoms Catatonic Behavior</td><td>discomfort Motor rigidity, stupor, mutism,</td></tr><tr><td>Decreased Energy</td><td>abnormal movement behavior Exhaustion, low energy, persistent</td></tr><tr><td>Tiredness Fatigue Depressed Mood</td><td>tiredness</td></tr><tr><td>Gastrointestinal</td><td>Sadness, hopelessness, emptiness, emotional numbness</td></tr><tr><td>Symptoms</td><td>Nausea, stomach pain, digestive discomfort</td></tr><tr><td>Genitourinary Symptoms</td><td>Urinary discomfort, reproductive-system-related symptoms</td></tr><tr><td>Hyperactivity</td><td>Restlessness, pacing, inability to</td></tr><tr><td>Agitation Impulsivity</td><td>remain still, agitation</td></tr><tr><td></td><td>Impulsive decisions, poor behavioral inhibition</td></tr><tr><td>Inattention</td><td>Distractibility, forgetfulness, difficulty</td></tr><tr><td>Indecisiveness</td><td>concentrating Difficulty making decisions, hesitation,</td></tr><tr><td>Respiratory</td><td>uncertainty</td></tr><tr><td>Symptoms</td><td>Shortness of breath, choking sensations, breathing difficulty</td></tr><tr><td>Suicidal Ideas</td><td>Thoughts of death, suicidal ideation,</td></tr><tr><td>Worthlessness and</td><td>suicide planning or attempts Self-blame, guilt, negative</td></tr><tr><td>Guilty Avoidance of</td><td>self-evaluation, shame Avoidance of trauma-related memories</td></tr><tr><td>Stimuli</td><td>or external reminders</td></tr><tr><td>Compensatory</td><td></td></tr><tr><td></td><td>Purging, excessive exercise,</td></tr><tr><td></td><td></td></tr><tr><td>Weight Gain Compulsions</td><td>Behaviors to Prevent compensatory eating behaviors Checking, cleaning, counting,</td></tr></table>

## E LLM Annotation Prompt for Identified Positive (IP) Classification

This section details the prompts used by the LLM. For ADHD and ASD, evidence referring to an affected family member (e.g., a child) is eligible. For BD, DEP, GAD, OCD, and SZ, only evidence referring to the post author is eligible; familymember references do not satisfy the CEP criterion.

## System Instruction.

You are a classifier for disease-related posts. You should output only a single integer result.

## User Prompt (translated from Chinese).

Table 11: Complete symptom taxonomy used in MATCHA (Part 2).
<table><tr><td>English Label</td><td>Semantic Gloss</td></tr><tr><td>Diminished Emotional</td><td>Reduced facial expression, monotone speech, emotional flattening</td></tr><tr><td>Expression Do Things Easily Get Painful</td><td>Risk-taking, self-harm, dangerous impulsive behavior</td></tr><tr><td>Consequences Drastical Shift in Mood and Energy</td><td>Abrupt changes in mood and energy levels</td></tr><tr><td>Fear About Social Situations Fear of Gaining</td><td>Fear or avoidance of conversations, strangers, or public performance</td></tr><tr><td>Weight Fears of Being</td><td>Fear of weight gain, body-image disturbance, restrictive eating Fear of criticism, embarrassment, or</td></tr><tr><td>Negatively Evaluated</td><td>social judgment</td></tr><tr><td>Flight of Ideas</td><td>Rapid topic shifts, racing thoughts, disorganized thinking</td></tr><tr><td>Intrusion Symptoms</td><td>Flashbacks, intrusive traumatic memories, trauma-related nightmares</td></tr><tr><td>Loss of Interest or Motivation</td><td>Loss of motivation, social withdrawal, inability to experience pleasure</td></tr><tr><td>More Talkative</td><td>Pressured speech, excessive talking, difficult-to-interrupt speech</td></tr><tr><td>Obsession</td><td>Intrusive thoughts, contamination fears, symmetry obsessions</td></tr><tr><td>Panic Fear</td><td>Panic attacks, acute fear responses, intense anxiety episodes</td></tr><tr><td>Pessimism</td><td>Hopelessness, negative expectations</td></tr><tr><td>Poor Memory</td><td>about the future Forgetfulness, impaired recall,</td></tr><tr><td>Sleep Disturbance</td><td>memory difficulties Insomnia, hypersomnia, nightmares,</td></tr><tr><td>Somatic Muscle</td><td>reduced sleep need Muscle pain, body aches, stiffness,</td></tr><tr><td>Somatic Symptoms</td><td>movement discomfort General unexplained physical</td></tr><tr><td>Others Somatic Symptoms</td><td>discomfort or bodily complaints Heat sensations, numbness, tingling,</td></tr><tr><td>Sensory</td><td>dizziness Weight and Appetite Binge eating, appetite loss, significant</td></tr></table>

You will receive a post written by a user in the {disease} Super-Topic.

Please determine whether the post contains any of the following information:

• The eligible person has been diagnosed with the disease;

• The eligible person is taking prescribed medication for the disease;

• The eligible person explicitly states that they have the disease;

• The eligible person is hospitalized or receiving treatment (e.g., follow-up visits) for the disease.

If any of the above conditions are met, output 1;   
otherwise, output 0.

Please carefully distinguish whether the post describes a real event that actually happened to the eligible person under the disorder-specific rule

Table 12: Representative life-event categories in MATCHA.
<table><tr><td>Category</td><td>Representative Sub-events</td></tr><tr><td>Health</td><td>Hospitalization, medication use, surgery, self-harm, recovery from addiction</td></tr><tr><td>Relationship</td><td>Breakup, divorce, marriage, family</td></tr><tr><td>Changes</td><td>conflict, parenting difficulties</td></tr><tr><td>Career</td><td>Job loss, promotion, workplace discrimination, unemployment</td></tr><tr><td>Education</td><td>Examinations, school transition, academic stress, dropping out</td></tr><tr><td>Financial</td><td>Debt, bankruptcy, major financial difficulty, home purchase</td></tr><tr><td>Relocation</td><td>Moving cities, immigration,</td></tr><tr><td>Identity</td><td>homelessness, leaving parents&#x27; home Coming out, gender transition,</td></tr><tr><td>Death</td><td>religious or political identity change Death of spouse, parent, child, friend, or pet</td></tr><tr><td>Lifestyle Change</td><td>Changes in habits, social activities, recreation, military service</td></tr><tr><td></td><td>New Birth in Family Childbirth, adoption, becoming a parent or grandparent</td></tr><tr><td>Legal</td><td>Arrest, lawsuit, incarceration, loss of driver&#x27;s license</td></tr><tr><td>Societal</td><td>Pandemic, war, natural disaster, major political events</td></tr></table>

Table 13: Post-level prevalence of explicit disorder and clinical cues across sources.
<table><tr><td>Source</td><td>Posts Dis. [%] Clin. [%] Either [%]</td><td></td><td></td></tr><tr><td>HP-control</td><td>153,704</td><td>0.17</td><td>0.42</td></tr><tr><td>STP</td><td>169,508</td><td>13.88</td><td>8.22 19.74</td></tr><tr><td>HP</td><td>1,559,045</td><td>3.72</td><td>1.31 4.64</td></tr></table>

above. If the post does not explicitly mention a confirmed diagnosis, you should not assume the user or their family member has the disease.

## Examples.

• “I was diagnosed with XX disease last year.”  explicit diagnosis  output 1.

• “I took my meds again today, hope the next hospital visit goes well.” medication and treatment output 1.

• “The hospital reminds you to pay attention to the following points for patients with XX disease...” informational content only output 0.

• “I’ve been having headaches and crying a lot recently, what should $I d { \bar { o } } ? $ symptoms only, no confirmed diagnosis output 0.

• “I really suspect that I have XX disease.”  selfsuspected, not confirmed  output 0.

## Example of Model Reasoning (translated from Chinese).

The post describes self-perceived changes and coping behaviors but does not include any explicit mention of medical diagnosis, medication, hospitalization, or confirmed self-identification as a patient. The analysis therefore relies strictly on explicit linguistic cues rather than inference or speculation, and the correct output is 0.

Output Format. The model must output only one digit (0 or 1), without any explanation or additional text.

## F Additional Cross-lingual Analysis Details

For the LIWC comparison, all categories are normalized within platform by contrasting disorder users against matched controls. Positive log ratios therefore indicate categories that are relatively more frequent among disorder users, while negative values indicate categories that are relatively less frequent. The main text reports representative categories with the largest and most interpretable deviations.

For the symptom and life-event comparison, the Chinese and English classifiers are trained independently to reduce direct transfer bias across languages. Remaining diferences may still reflect annotation granularity, label-inventory mismatch, and platform-specific discourse conventions. Accordingly, all cross-lingual results are interpreted as aggregate distributional trends rather than clinical or individual-level claims.

Table 14: Cue-masking performance. Panel A compares original and fully masked text; Panel B reports tiered masking results. Values are mean ± standard deviation over five seeds.
<table><tr><td colspan="6">Panel A: Original versus fully masked text</td></tr><tr><td>Source</td><td> $\mathrm { F 1 _ { o r i g } }$ </td><td> $\mathrm { F l } _ { \mathrm { m a s k e d } }$ </td><td>∆F1</td><td> $\mathbf { A U R O C } _ { \mathrm { o r i g } }$ </td><td> $\mathrm { { A U R O C } _ { \mathrm { { m a s k e d } } } }$ </td></tr><tr><td>STP</td><td> $0 . 9 5 3 \pm 0 . 0 0 6$ </td><td> $0 . 9 3 6 \pm 0 . 0 0 7$ </td><td>-0.017</td><td>0.995</td><td>0.992</td></tr><tr><td>HP</td><td> $0 . 8 4 2 \pm 0 . 0 1 2$ </td><td> $0 . 8 3 3 \pm 0 . 0 1 2$ </td><td>-0.009</td><td>0.932</td><td>0.927</td></tr><tr><td colspan="6">Panel B: Tiered cue masking Condition</td></tr><tr><td></td><td>A</td><td>B</td><td>C</td><td>∆(A-B)</td><td> $\Delta ( \mathrm { A } { - } \mathrm { C } )$ </td></tr><tr><td>HP-only</td><td> $0 . 8 4 2 \pm 0 . 0 1 2$ </td><td> $0 . 8 3 3 \pm 0 . 0 1 2$ </td><td> $0 . 8 3 3 \pm 0 . 0 1 2$ </td><td>0.009</td><td>0.009</td></tr><tr><td>STP-only</td><td> $0 . 9 5 3 \pm 0 . 0 0 6$ </td><td> $0 . 9 3 7 \pm 0 . 0 0 7$ </td><td> $0 . 9 3 6 \pm 0 . 0 0 7$ </td><td>0.016</td><td>0.017</td></tr><tr><td>STP+HP</td><td> $0 . 8 5 8 \pm 0 . 0 0 9$ </td><td> $0 . 8 4 8 \pm 0 . 0 1 0$ </td><td> $0 . 8 4 7 \pm 0 . 0 1 0$ </td><td>0.010</td><td>0.010</td></tr></table>

Table 15: Per-disorder Macro-F1 before and after masking disorder-name and clinical cues. ∆ denotes the absolute decrease from the original condition.
<table><tr><td rowspan="2"></td><td colspan="3">HP</td><td colspan="3">STP</td></tr><tr><td>Disorder Original</td><td>Masked</td><td>∆</td><td>Original</td><td>Masked</td><td>∆</td></tr><tr><td>ADHD</td><td>0.858</td><td>0.847</td><td>0.011</td><td>0.953</td><td>0.910</td><td>0.043</td></tr><tr><td>ASD</td><td>0.874</td><td>0.856</td><td>0.018</td><td>0.967</td><td>0.937</td><td>0.030</td></tr><tr><td>BD</td><td>0.850</td><td>0.849</td><td>0.001</td><td>0.952</td><td>0.949</td><td>0.003</td></tr><tr><td>DEP</td><td>0.789</td><td>0.783</td><td>0.006</td><td>0.956</td><td>0.952</td><td>0.004</td></tr><tr><td>GAD</td><td>0.808</td><td>0.805</td><td>0.002</td><td>0.977</td><td>0.972</td><td>0.005</td></tr><tr><td>OCD</td><td>0.845</td><td>0.835</td><td>0.010</td><td>0.969</td><td>0.951</td><td>0.018</td></tr><tr><td>SZ</td><td>0.868</td><td>0.856</td><td>0.012</td><td>0.897</td><td>0.883</td><td>0.014</td></tr><tr><td>Avg.</td><td>0.842</td><td>0.833</td><td>0.009</td><td>0.953</td><td>0.936</td><td>0.017</td></tr></table>