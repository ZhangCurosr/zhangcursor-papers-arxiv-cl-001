# Detecting LLM-Assisted Vietnamese Writing via Keystrokes under Behavioral Manipulation

Thanh Dong\* Bucknell University Lewisburg, PA, USA tpd010@bucknell.edu

An Ngo\* Bucknell University Lewisburg, PA, USA axn001@bucknell.edu

Minh Dau Bucknell University Lewisburg, PA, USA mtcd001@bucknell.edu

Rajesh Kumar   
Bucknell University   
Lewisburg, PA, USA   
rajesh.kumar@bucknell.edu

Abstract—We study the robustness of keystroke dynamics for detecting large language model (LLM)-assisted writing. We introduce a Vietnamese keystroke dataset capturing realistic writing modes, including bona fide composition, transcription, and paraphrasing. We also define a behaviorally grounded threat model in which users deliberately alter typing patterns. To implement the threat model, we create behaviorally manipulated variants of the data designed to evade keystroke-based detection. We evaluate four keystroke modeling approaches: temporal and rhythmic representations, and sequential representations modeled with a one-dimensional convolutional neural network (1D-CNN) and TypeNet, under user-independent and contextindependent settings. The results show that sequential models outperform feature-based approaches in most cases and that keystroke signals encode discriminative information about the writing process. However, detection is not uniformly robust: transcription is reliably identified, while paraphrasing and adversarially manipulated samples are frequently misclassified as bona fide when not explicitly modeled. To address this, we incorporate adversarial training using behaviorally manipulated data, which substantially improves separability and robustness. These results suggest that keystroke-based detection depends critically on exposure to diverse writing behaviors, and that strong performance under limited conditions does not generalize to realistic or adversarial settings without targeted modeling.

Index Terms—Keystroke Dynamics, LLM-Assisted Writing Detection, Adversarial Robustness, Human vs. LLM

## I. INTRODUCTION

Recent advances in large language models (LLMs) such as ChatGPT [1], Claude [2], Gemini [3], and LLaMA [4] have made it increasingly easy to generate high-quality text. This capability allows students to generate, paraphrase, or transcribe responses with minimal effort, raising significant concerns about academic integrity, particularly in online and take-home assessment settings [5]–[8]. Text-based approaches have also explored stylometric features for detecting LLM-assisted writing, demonstrating that lexical and grammatical characteristics can provide useful discriminative signals [9]. However, such approaches analyze the final written output rather than the behavioral process through which it is produced. Traditional plagiarism detection systems based on textual similarity can also become less effective when LLM-generated content is substantially paraphrased or rewritten [10], [11].

Prior studies further indicate that human evaluators struggle to reliably distinguish between human-written and artificial intelligence (AI)-generated text under such conditions, with performance approaching random guessing [8], [12].

Rather than replacing conventional plagiarism detection systems, we view keystroke-based analysis as a complementary layer of academic integrity assessment. In a practical deployment, submissions would first be screened using similaritybased plagiarism detectors, which remain effective for identifying copied material from known or indexed sources. Submissions that pass this initial screening but remain suspicious because they may have been generated or heavily paraphrased using LLMs can then be analyzed using keystroke dynamics, which examine the writing process rather than the final text.

These limitations motivate a shift from analyzing the final written output to examining the process through which the text is produced [5]–[8]. Keystroke dynamics capture fine-grained behavioral signals during text production, including typing speed, pauses, and revision patterns [5]–[8]. Such signals reflect underlying cognitive and motor processes involved in writing, including planning, translation, and editing [13]. Unlike textual features, which can be manipulated through paraphrasing or rewriting, keystroke signals arise from realtime interaction between cognition and motor execution, making them inherently difficult to replicate without altering the writing process. Altering the writing process and corresponding keystrokes is non-trivial [14]. Prior work has demonstrated systematic differences between bona fide and AI-assisted writing in these behavioral patterns, suggesting that keystroke dynamics provide a reliable signal for detecting LLM-assisted content [5]–[8].

Despite this promise, existing work remains limited in several important respects [8]. First, most approaches consider simplified problem settings, typically distinguishing only between bona fide and assisted writing [5]–[8]. In practice, LLM assistance can take different forms: a user may directly transcribe an LLM-generated response or paraphrase it while composing the final answer. These interaction modes differ in cognitive demand and consequently produce distinct behavioral signatures in keystroke dynamics [6], [8]. Importantly, paraphrasing may reduce textual evidence of LLM assistance while preserving behavioral differences in the writing process.

Second, the problem becomes adversarial at the behavioral level when detection relies on keystroke signals. In this setting, users may attempt to evade detection not by modifying the textual output [15], but by deliberately altering how they type. This includes introducing artificial pauses, revisions, or variations in typing speed to mimic natural writing behavior. While prior work [6] has explored adversarial settings using synthetic statistical perturbations, such approaches do not fully capture behaviorally grounded evasion, where users consciously adapt their typing process during composition. This distinction separates task-level variation (transcription vs. paraphrasing) from strategic behavioral manipulation aimed at bypassing keystroke-based detectors.

Third, the applicability of keystroke-based detection across languages remains poorly understood [6]. Existing studies focus primarily on English [5], [7], [16] and, more recently, Korean [6]. It remains unclear whether these methods generalize to languages with different typing characteristics, input methods, and orthographic structures.

Vietnamese provides a particularly informative testbed because its TELEX input method composes diacritics through multi-keystroke sequences, resulting in typing dynamics that differ substantially from English and Korean [17]–[19]. This allows us to evaluate whether existing keystroke-based detection methods generalize across fundamentally different input mechanisms. Thus, Vietnamese provides a distinct setting for evaluating keystroke-based detection, while generalization to other languages remains to be investigated.

To bridge these gaps, we develop and evaluate a behavioral detection framework that complements conventional textbased plagiarism detection for Vietnamese LLM-assisted writing. In particular, this work makes the following contributions:

• A Vietnamese keystroke dynamics dataset for analyzing AI-assisted writing across multiple realistic scenarios, publicly available for research on plagiarism detection and keystroke-based authentication.

• A comparative evaluation of temporal, rhythmic, and deep sequential modeling approaches, including a onedimensional convolutional neural network (1D-CNN) and TypeNet.

• An analysis of detection performance across diverse writing conditions, including synthetically generated behavioral manipulations designed to simulate adversarial evasion strategies.

• An evaluation of adversarial training for improving robustness against these synthetically generated behavioral manipulations.

The remainder is organized as follows: Section II reviews related work, Section III presents the approach, Section IV reports results, and Section V concludes.

## II. RELATED WORK

Keystroke dynamics has been widely studied as a behavioral signal for user authentication, identification, and fraud detection [20]–[28]. These works show that typing patterns encode stable temporal characteristics that can be used to distinguish users in different contexts. Beyond identity modeling, keystroke behavior has also been explored as a proxy for cognitive processes and as complementary information for writing assessment [29]. Banerjee et al. [30] study keystroke patterns as an analog to prosody in speech to capture the realtime writing process for deception detection. They collect a dataset of truthful and deceptive texts, extract features related to editing behavior and temporal dynamics such as pauses and typing speed, and incorporate them into a classification framework. Their results show that keystroke-based features improve deception detection performance, indicating that differences in cognitive load between truthful and deceptive writing are reflected in measurable typing patterns.

Agarwal et al. [16] design a keystroke-based system to distinguish genuine students from impostors in remote assessments by modeling typing behavior as a biometric signal. The approach uses duration and latency features and evaluates detection performance under strict constraints such as zero false accusations. Their results show that keystroke dynamics can effectively identify contract cheating, supporting the use of behavioral signals for verifying authorship in academic settings. More recently, Crossley et al. [7] used keystroke logs to distinguish authentic writing from transcribed text. Their model leverages rhythmic features such as pauses, revisions, and writing bursts to capture differences in writing processes and achieves high classification accuracy. Similarly, Deane et al. [31] used keystroke dynamics to distinguish original composition from nonoriginal text production, further demonstrating that differences in the writing process can provide signals beyond the final text. Zhang et al. [32] further examined natural writing and copy typing using keystroke logs and deep learning models, showing distinct behavioral patterns between the two writing processes. However, these studies focus on transcription, copy typing, or nonoriginal text production rather than direct interaction with LLMs during composition.

Recent work explicitly studies LLM-assisted writing rather than treating transcription or authorship mismatch as a proxy [5], [6], [8], [33]. These studies share a common goal of distinguishing bona fide writing from interactions with generative models, but differ in how they model LLM usage and behavioral variability. Early work [5] formulates the problem as binary classification between bona fide and ChatGPTassisted writing using deep sequential models such as TypeNet [34]. While this demonstrates that keystroke dynamics capture differences in writing processes, the formulation assumes a single mode of LLM assistance and does not distinguish between behaviors such as transcription and paraphrasing.

Subsequent work from Roh et al. [6] separates ChatGPT interaction into transcription and paraphrasing modes and incorporates cognitive variation using Bloom’s Taxonomy. Their results show that paraphrasing produces more variable and complex keystroke patterns than transcription, highlighting the role of cognitive load in shaping behavior. Mehta et al. [8] further extend this line of work by scaling the dataset, introducing explicit paraphrasing conditions, and benchmarking keystroke-based models against text-based detectors and human evaluators. They also introduce a deception-oriented threat model and evaluate robustness under adversarial settings. In their formulation, an adversary generates forged keystroke sequences by transferring timing statistics from prior user data to new text, simulating attacks such as user-specific and pooled timing-based mimicry. Their results show that such attacks significantly degrade model performance and that adversarial training improves robustness to these perturbations.

TABLE I: Comparison of keystroke-based approaches across language (EN: English, KR: Korean, VI: Vietnamese), representation (rhythmic, temporal, deep), classifiers (RF: random forest, ML: classical machine learning), threat assumptions, and writing modes (B: bona fide, T: transcribe LLM, P: paraphrase LLM, FT/FP: adversarial variants).
<table><tr><td>Work</td><td>Lang</td><td>Cognition</td><td>Modeling</td><td>Classifier</td><td>Threat</td><td>Modes</td></tr><tr><td>Crossley et al. [7]</td><td>EN</td><td>Implicit</td><td>Rhythmic</td><td>RF</td><td>None</td><td>{B, T}</td></tr><tr><td>Kundu et al. [5]</td><td>EN</td><td>None</td><td>Deep</td><td>TypeNet</td><td>None</td><td>{B, T}</td></tr><tr><td>Roh et al. [6]</td><td>KR</td><td>Explicit</td><td>Rhythmic, Temporal</td><td>ML</td><td>None</td><td>{B, T, P}</td></tr><tr><td>Mehta et al. [8]</td><td>EN</td><td>Implicit</td><td>Temporal, Deep</td><td>TypeNet, ML</td><td>Statistical</td><td>{B, T, P}</td></tr><tr><td>Ours</td><td>VI</td><td>Explicit</td><td>Rhythmic, Temporal, Deep</td><td>ML, 1D-CNN, TypeNet</td><td>Behavioral</td><td>{B, T, P, FT, FP}</td></tr></table>

However, the proposed threat model operates primarily at the level of statistical timing synthesis and does not fully capture how users may adapt their typing behavior during realworld typing. In practice, adversaries may introduce higherlevel behavioral strategies, such as modifying pause patterns, revision behavior, or writing rhythm, which are not explicitly modeled in their framework. As a result, while the study provides an important first step toward evaluating deception, it does not cover the full space of behaviorally grounded adversarial strategies.

As summarized in Table I, prior approaches are limited in modeling diversity, adversarial realism, and linguistic diversity, with existing studies focusing primarily on English [5], [8] and Korean [6]. The lack of language diversity raises questions about whether keystroke-based signals generalize across typologically distinct languages, such as Vietnamese, which exhibits different lexical segmentation and input dynamics. This work addresses these gaps by introducing a Vietnamese keystroke dataset with multiple writing modes, including transcription and paraphrasing, as well as a behaviorally grounded threat model for realistic adversarial behavior. In addition, we incorporate adversarial training and evaluate across temporal, rhythmic, and sequence-based representations.

## III. METHODOLOGY

## A. Task definition

Detecting varying levels of LLM intervention in the writing process can be formulated as a multi-class classification problem. A user produces a text response t through a sequence of keystroke events $k ,$ where the keystroke sequence captures the underlying writing behavior. The objective is to infer the type of writing process that generated the response, distinguishing between genuine composition and different forms of assisted or deceptive writing.

Each sample is represented as a pair (t, k), where t denotes the final text and $k \stackrel { \cdot } { = } \{ ( k _ { i } , t _ { i } ^ { \downarrow } , t _ { i } ^ { \uparrow } ) \} _ { i = } ^ { \tilde { N } }$ denotes the corresponding keystroke sequence. Here, $k _ { i }$ represents the key identity at event $\dot { i } , t _ { i } ^ { \downarrow }$ denotes the timestamp at which the key is pressed (key-down event), and $t _ { i } ^ { \uparrow }$ denotes the timestamp at which the key is released (key-up event). These events encode finegrained temporal dynamics of the writing process, including key hold times and inter-key latencies.

Each sample is associated with a label $y \in \mathcal { D } ,$ , where $\mathcal { Y } ~ = ~ \{ B , T , P , F T , F P \}$ . The label B denotes bona fide writing, where the text is independently composed without external assistance. The remaining classes correspond to different forms of deviation from genuine writing. The labels T and P denote text-level interventions in which the user relies on LLM-generated content. $T$ is for direct transcription (T) and P is for paraphrasing (P) without any intentional alteration to the typing process.

The labels $F T$ and FP denote behavioral-level interventions in which, while transcribing or paraphrasing, the user intentionally modifies keystroke behavior to mimic a bona fide writing process or to fool the detectors. These modifications include changes in typing speed, pause structure, and revision patterns, with the goal of deceiving the detectors by producing keystroke traces that resemble genuine composition.

We learn a classifier $f _ { \theta } : \mathcal { X }  \mathcal { Y }$ , where X denotes the space of keystroke sequences. The model operates on representations derived solely from k.

## B. Detection framework overview

The proposed framework assumes that conventional textbased plagiarism screening has already been performed. Consequently, our detector is designed to analyze submissions that exhibit little textual similarity to existing sources but may nevertheless involve substantial LLM assistance. In this deployment scenario, keystroke dynamics provides a complementary behavioral analysis after the initial text-based screening stage.

In a practical deployment, keystrokes could be captured within a web-based assessment interface, as in our datacollection system, without requiring background software. Collection would be limited to the assessment session and would require appropriate disclosure and privacy safeguards.

The detection framework consists of six stages: data collection, preprocessing, adversarial generation, behavioral modeling and representation, training, and evaluation. Each stage is described in detail below.

1) Data collection: Following Institutional Review Board (IRB) approval from the university, a total of 45 Vietnamese participants were recruited, including 19 university students and 26 high school students, with 11 females and 34 males (mean age 17.66 years). The study consisted of two phases separated by at least two weeks; 30 participants completed both phases. All participants provided informed consent prior to participation, with appropriate procedures for minor participants.

Participants completed the study on personal devices to preserve typing familiarity. Data were collected in real-world environments, including classrooms, cafes, and residences,´ introducing natural variations in typing conditions. One author was present for all sessions to resolve technical issues and ensure adherence to the protocol.

The protocol included three writing modes following prior work [6]: bona fide composition (B), transcription of LLMgenerated responses (T), and paraphrasing of LLM-generated content (P). Prompts were carefully crafted to include questions of varying cognitive demand, and responses were constrained to 90–150 words [6]. This range provides sufficiently long keystroke sequences for behavioral analysis while limiting participant fatigue and maintaining comparable response lengths across conditions. For T and P, participants first generated LLM responses that satisfied the length constraint and then performed the task.

Keystroke events were recorded using a web-based interface, capturing key identity and key-down/key-up timestamps. For T and $\mathrm { \mathrm { P } } ,$ a reference field displays the LLM response, while a separate input field records user-generated text, ensuring complete keystroke traces. Copy-paste was disabled because the study targets exam settings and the proposed method requires observing the writing process. Handling pasted and subsequently edited text is left for future work, as it introduces varying levels of user modification that are not captured in the current protocol.

All responses were subjected to automated validation. Lexical validity required at least 85% of tokens to appear in a Vietnamese lexicon. Transcription responses were required to achieve $\geq 9 5 \%$ similarity to the source text. Paraphrased responses were required to exhibit $\geq 1 5 \%$ lexical divergence. In addition, responses were evaluated using the Gemini 2.5 Flash application programming interface (API) [35], which scores relevance, coherence, depth, and grammatical quality on a 0–3 scale per dimension; a minimum total score of 7 (out of 12) was required for inclusion. Participants were informed of these criteria prior to the study to replicate a real exam environment, and no responses failed the validation thresholds. These criteria served as quality-control measures to remove invalid responses, ensure close adherence in transcription, and require sufficient modification in paraphrasing.

2) Data cleaning and preprocessing: Raw keystroke logs were normalized to ensure consistency across input systems and to remove noise prior to representation learning. Vietnamese text is typically entered using TELEX, a diacritic input method. Since the dataset was collected from heterogeneous devices, including Windows laptops and MacBooks, discrepancies arise in recorded keystroke sequences. On macOS,

TELEX is natively supported and produces keystroke logs that closely reflect user-intended inputs. In contrast, most Windows users rely on Unikey, which introduces intermediate virtual keystrokes when composing diacritics.

For example, typing “chao” requires the sequence\` c–h–a– $f { - } o ,$ , where $f$ transforms a into a. Crucially, the same lexical\` output can also be produced via the alternative ordering c–h– $a { - } o { - } f ,$ reflecting a non-canonical keystroke permutation characteristic of this input method, with admissible variations in ordering across user input strategies. Under Unikey, additional transient keystrokes are generated between a and $f ,$ with patterns varying across software versions. These artifacts do not correspond to user-level actions and distort temporal and sequential features.

We identify recurring Unikey-specific patterns and remove virtual keystrokes through an automated filtering procedure, yielding sequences that retain only user-intended inputs.

Following normalization, all keystroke sequences undergo a unified two-stage noise filtering procedure applied across all modeling approaches. Let $K ( \bar { x } ) \bar { = } \{ ( k _ { i } , t _ { i , } ^ { \downarrow } , t _ { i } ^ { \uparrow } ) \} _ { i = 1 } ^ { N }$ , with derived features including key hold time $h _ { i } = t _ { i } ^ { \uparrow } { - } t _ { i } ^ { \downarrow }$ , inter-key interval $\Delta _ { i } = t _ { i + 1 } ^ { \downarrow } - t _ { i } ^ { \downarrow }$ , and release-to-keydown time (RKDT).

In the first stage, timing values are constrained to a physiologically plausible range. Any $h _ { i } , \Delta _ { i }$ , or related timing feature outside [50, 5000] ms is removed, eliminating artifacts such as accidental key holds or system-level logging delays.

In the second stage, outliers are removed using an interquartile range (IQR) filter applied to each timing feature distribution. Specifically, values outside $[ Q 1 - 2 \cdot I Q R , Q 3 + 2 \cdot I Q R ]$ are discarded to suppress anomalous pauses while preserving natural variability.

3) Threat model and adversarial sample generation: We consider the manipulation of keystroke dynamics during LLMassisted writing. In this setting, a user produces an LLMassisted response by typing, while deliberately altering their typing behavior to evade detection.

Let $( t , k , y )$ denote a sample as defined in Section III-A, where $\dot { k } ~ = ~ \{ ( k _ { i } , t _ { i } ^ { \downarrow } , t _ { i } ^ { \uparrow } ) \} _ { i = 1 } ^ { N }$ is the keystroke sequence and $y \in \mathcal { Y } = \{ B , T , P \}$ is the writing mode. Keystroke events are represented as paired keydown/keyup events, from which key hold times $h _ { i } = t _ { i } ^ { \uparrow } - t _ { i } ^ { \downarrow }$ and inter-key intervals $\Delta _ { i } = t _ { i + 1 } ^ { \downarrow } - t _ { i } ^ { \downarrow }$ are derived. The detector $f _ { \theta } : \mathcal { X }  \mathcal { V }$ operates only on k.

For samples with $y \in \{ T , P \}$ , we assume that users may deliberately modify their typing behavior during transcription and paraphrasing to evade detection. Such manipulation produces a keystroke sequence $\hat { k }$ during typing that deviates from natural transcription and paraphrasing patterns and more closely resembles bona fide writing. The objective is to induce misclassification as bona fide, i.e., $f _ { \theta } ( \boldsymbol { \hat { k } } ) = B$

We assume a zero-knowledge black-box setting in which the adversary has no access to model parameters or features, but is aware that transcription and paraphrasing produce regular, low-variance timing patterns with limited revisions. These differences define the key behavioral gap between LLM-assisted and bona fide writing: transcription and paraphrasing exhibit reduced temporal variability, minimal revisions, and weaker pause structure [6], [7]. The adversarial transformations are designed to explicitly target these deficiencies.

The manipulation operates on the temporal structure of k. The adversary perturbs $\{ h _ { i } , \Delta _ { i } \}$ during typing to alter higherorder statistics, including variance, burst structure, pause distribution, and revision frequency.

In practice, we approximate this behavior by constructing adversarial samples through a transformation $\mathcal { A } : \mathcal { X }  \mathcal { X }$ applied to recorded keystroke sequences. For each $( t , k , y )$ with $y \in \{ T , P \}$ , we generate $\hat { k } = \boldsymbol { \mathcal { A } } ( k ) , \quad k \in \mathcal { X } , \hat { k } \in \mathcal { X }$ and assign new labels $F T$ and FP to the behavioral-level interventions defined in Section III-A. The transformation A operates on $\{ h _ { i } , \Delta _ { i } \}$ and consists of three components: temporal variability (velocity scaling), revision behavior (revision injection), and cognitive planning signals (boundaryaware pausing). To induce temporal variability, typing speed is modulated by scaling $\Delta _ { i }$ with a factor $\overline { { v \sim U ( 0 . 7 5 , 1 . 1 0 ) } }$ , resampled at word boundaries with probability 0.15, increasing variance and disrupting the near-constant rhythm observed in transcription, which lacks natural motor variability.

The revision behavior is manipulated by deleting and retyping segments, which are introduced with probability 0.10 at word or sentence boundaries, ensuring at least two revision episodes per sequence. Each revision deletes 2–5 words via backspace events, preceded by a hesitation delay sampled from U(600, 1400) ms and followed by a re-planning delay sampled from U(900, 2600) ms. Subsequent backspace events occur with short inter-key intervals (60–110 ms), modeling rapid motor execution following error realization. Retyped text is produced with reduced latency $( \Delta _ { i }  0 . 7 \Delta _ { i } , h _ { i }  0 . 8 5 h _ { i } )$ reflecting motor fluency during repetition.

To accomplish boundary-aware pausing, additional delays were inserted into $\Delta _ { i }$ at linguistic boundaries with probabilities 0.20 (word), 0.45 (clause), 0.70 (sentence), and 0.85 (line). Pause durations were sampled from Weibull distributions (mean ≈ 750–1750 ms, shape $k \in [ 2 . 1 , 2 . 4 ] )$ , with a longtail mixture (probability $0 . 4 \mathrm { - } 0 . 6 ,$ multiplier 2.5–3.0) to model occasional extended planning pauses. Pauses were not injected when $\Delta _ { i } > 1 5 0 0$ ms.

4) Behavioral representation and modeling: Each preprocessed sequence $\tilde { k }  { \mathrm { ~ \textrm ~ { ~ ~ } ~ } }  { \mathrm { i s } }$ mapped to $\begin{array} { r l r l r l r l } { z } & { { } } & { = } & { } & { \{ \phi _ { \mathrm { t e m p } } ( \tilde { k } ) , \ } & { { } \ \phi _ { \mathrm { r h y t h m } } ( \tilde { k } ) , \ } & { } & { \phi _ { \mathrm { s e q } } ( \tilde { k } ) \} } \end{array}$ , capturing complementary behavioral signals at motor, structural, and sequential levels.

The temporal representation $\phi _ { \mathrm { t e m p } } ( \tilde { k } )$ captures low-level typing dynamics through key hold time (KHT), key interval time (KIT), and release-to-keydown time (RKDT), defined as $h _ { i } = t _ { i } ^ { \uparrow } - t _ { i } ^ { \downarrow } , \Delta _ { i } = t _ { i + 1 } ^ { \downarrow } - t _ { i } ^ { \downarrow }$ , and $r _ { i } = t _ { i + 1 } ^ { \downarrow } - t _ { i } ^ { \uparrow }$ , respectively. Features are computed over the top-50 most frequent digraphs [36] to reduce sparsity and focus on stable transitions, and summarized using $\{ \mu , \sigma , \sigma / \mu ,$ , range, max, median}, yielding a fixed-length representation (with dimensionality determined by the number of digraph-feature combinations) suitable for tabular models.

The rhythmic representation $\phi _ { \mathrm { r h y t h m } } ( { \tilde { k } } )$ consists of 107 features derived from pause, burst, and revision statistics [6], [7]. These include binned pause durations; inter-word and inter-sentence pauses; pauses preceding deletions; P-bursts (production bursts), R-bursts (revision bursts), and deletion bursts; and distributional descriptors such as moments, quantiles, counts, and entropy. These features are complementary to temporal statistics and capture long-range structural patterns not encoded in local timing signals.

The sequential representation $\phi _ { \mathrm { s e q } } ( \ddot { k } )$ models keystroke $\mathrm { { d y } \mathrm { { - } } }$ namics as an ordered signal with events encoded as $\begin{array} { r l } { x _ { i } } & { { } = } \end{array}$ $( \Delta _ { i } , c _ { i } )$ , where $c _ { i }$ denotes the key category. To avoid leaking textual content and prevent models from exploiting lexical information, keys are mapped into five classes: characters, digits, space, backspace, and special keys, while preserving structural cues such as word boundaries and deletions. Sequences are segmented using sliding windows of length $L = 1 0 0$ (keydown events) with a stride of 50, selected to balance temporal resolution and contextual coverage.

A 1D-CNN is used to model local temporal dependencies. To handle heterogeneous keystroke data, continuous timing and discrete keycodes are first processed by a multilayer perceptron (MLP) and an embedding layer, respectively. Their outputs are concatenated and fed into a secondary MLP fusion layer to create a unified, high-dimensional representation before entering the convolutional stack. This sequence is then fed into three stacked convolutional layers with a kernel size of 3, capturing trigram-level keystroke patterns that reflect shortrange motor dependencies [37]. The convolutional stack is followed by global max pooling, which provides invariance to sequence length and captures salient timing anomalies, and by a two-layer fully connected classification head that maps learned representations to class probabilities. We additionally employ TypeNet [34], an established two-layer long shortterm memory (LSTM) network trained with cross-entropy, as a recurrent baseline; while the 1D-CNN captures local temporal dependencies, the LSTM models longer-range sequential structure across keystrokes. The sequential representation for TypeNet is modeled as $x _ { i } = ( t _ { i + 1 } ^ { \downarrow } - t _ { i } ^ { \downarrow } , t _ { i } ^ { \uparrow } - t _ { i } ^ { \downarrow } , t _ { i + 1 } ^ { \downarrow } -$ $t _ { i } ^ { \uparrow } , \ t _ { i + 1 } ^ { \uparrow } - t _ { i } ^ { \uparrow } , \ c _ { i } )$ , following the original TypeNet design [34]. Given z, classifiers $f \in { \mathcal { F } }$ are trained under each evaluation setting. Temporal and rhythmic representations are modeled using eXtreme Gradient Boosting (XGBoost), while $\phi _ { \mathrm { s e q } }$ is modeled using 1D-CNN and recurrent architectures. For sequence models, predictions are computed at the window level and aggregated via mean pooling across windows, followed by class selection based on maximum probability to obtain the final prediction $\hat { y } .$

5) Training and evaluation: Models are trained and evaluated under two complementary generalization settings, userindependent evaluation (UIE) and context-independent evaluation (CIE), π ∈ {UIE, CIE}, capturing variability across users and prompts.

In the user-independent evaluation (UIE), training and testing are performed on disjoint sets of users. This setting evaluates whether $f _ { \theta }$ generalizes to unseen individuals, reflecting deployment scenarios where no prior behavioral data is available. In the context-independent evaluation (CIE), training and testing are performed on disjoint sets of prompts, with overlap allowed in users. This setting isolates generalization across tasks, ensuring that predictions rely on stable behavioral signals rather than prompt-specific artifacts.

![](images/3ba38ad3ca3092522c7e36b710bd707c884c0e492e30c2228b58a92cd5c8e93e.jpg)  
Fig. 1: Confusion Matrix for User-Independent Evaluations. Cells corresponding to explicitly trained classes utilize a white-to blue gradient. Cells corresponding to unseen attack vectors (zero-shot evaluation) utilize a white-to-orange gradient to highlight vulnerabilities.

Within each setting, four training configurations are defined to examine the effect of exposure to assisted writing and adversarial behavior. Let $\mathcal { Y } = \{ B , T , P , F T , F P \}$ . For each configuration $m \in \{ M 2 , M 3 , M 4 , M 5 \}$ , training is restricted to a subset $\mathcal { V } _ { t r a i n } ^ { m } \subseteq \mathcal { V }$ , while evaluation is performed over all classes.

M2 uses {B, T} and tests whether exposure to transcription alone is sufficient to detect more complex behaviors. M3 uses {B, T, P} and evaluates whether modeling paraphrasing improves robustness to LLM-assisted writing. M4 uses $\{ B , T , P , F T \}$ and assesses zero-shot adversarial generalization by exposing the model to one form of behavioral manipulation. M5 uses all classes and represents the fully supervised setting in which all behaviors are observed during training.

To obtain robust estimates, 3-fold cross-validation is applied in both settings. In UIE, users are partitioned into three disjoint groups, with two used for training and one for testing. In CIE, prompts are partitioned into three groups with balanced cognitive load, and models are trained on two groups and evaluated on unseen prompts. This design preserves variability while preventing overlap between training and testing distributions.

Hyperparameters are optimized using nested crossvalidation within each training fold to avoid data leakage. For each outer split, the training data is further divided into inner training and validation subsets, and model parameters are finetuned.

Performance is evaluated using macro-averaged F1 (Macro-F1), False Acceptance Rate (FAR), and False Rejection Rate (FRR). Macro-F1 is the unweighted mean of the per-class F1 scores, where F1 is the harmonic mean of precision and recall, computed over all five classes; it therefore weights each writing mode equally regardless of the number of samples. For FAR and FRR, bona fide samples (B) are treated as the positive class, while all assisted or adversarial samples are treated as negative. FAR measures the proportion of assisted responses incorrectly classified as bona fide, while FRR measures the proportion of bona fide responses incorrectly rejected. All metrics are computed from confusion matrices pooled over the three folds.

![](images/ef60929a869153810e7d9424d2dbd641ebba7f538584c46f5e44682edf9f51cd.jpg)  
Fig. 2: Confusion matrices for context-independent evaluation. Cells corresponding to explicitly trained classes utilize a white to-blue gradient. Cells corresponding to unseen attack vectors (zero-shot evaluation) utilize a white-to-orange gradient to highlight vulnerabilities.

TABLE II: M5 results (%), pooled over three folds. Macro-F1 averages the per-class F1; FAR/FRR treat bona fide (B) as positive. Best per row in bold.
<table><tr><td></td><td>Metric</td><td>Temporal</td><td>Rhythmic</td><td>1D-CNN</td><td>TypeNet</td></tr><tr><td rowspan="3">UIE</td><td>Macro-F1 ↑</td><td>58.8</td><td>67.7</td><td>85.6</td><td>80.8</td></tr><tr><td>FAR↓</td><td>13.3</td><td>12.8</td><td>6.7</td><td>7.0</td></tr><tr><td>FRR↓</td><td>45.3</td><td>40.0</td><td>16.0</td><td>18.7</td></tr><tr><td rowspan="3">CIE</td><td>Macro-F1 ↑</td><td>56.9</td><td>71.2</td><td>79.3</td><td>76.3</td></tr><tr><td>FAR↓</td><td>13.1</td><td>10.8</td><td>9.7</td><td>6.2</td></tr><tr><td>FRR↓</td><td>39.6</td><td>32.9</td><td>15.6</td><td>32.0</td></tr></table>

## IV. RESULTS AND DISCUSSION

We evaluate the effectiveness of keystroke dynamics in distinguishing bona fide writing from multiple forms of LLMassisted behavior under both standard and adversarial conditions. Across all experimental settings, keystroke signals provide a strong and consistent basis for discrimination, particularly when modeled using sequential architectures. Table II summarizes the performance of all four models in the M5 configuration.

Model comparison and overall performance Sequential models, especially the 1D-CNN, consistently outperform feature-based approaches across both user-independent and context-independent evaluations. The 1D-CNN achieves a false acceptance rate (FAR) of 6.7% in the user-independent setting, while maintaining 84.0% classification accuracy for bona fide samples in the fully supervised M5 setting. In contrast, temporal feature-based representations achieve 54.7% and 60.4% bona fide accuracy under UIE and CIE, respectively, compared with 84.0% and 84.4% for the 1D-CNN. These differences highlight a key finding: discriminative information in keystroke dynamics is not adequately captured by summary statistics alone but is distributed across temporal structure. Sequential models can exploit this structure, particularly local dependencies such as pause-burst patterns and revision behavior.

Separability across writing modes A more detailed analysis shows clear and consistent separability between transcription and bona fide writing. Under the 1D-CNN, transcription is classified with accuracy above 93% in the user-independent setting, with misclassification into the bona fide class typically below 5%. This demonstrates that transcription is behaviorally distinct in the learned keystroke representations, consistent with reduced variability and limited revision activity. In contrast, paraphrasing presents a more challenging case. Under the 1D-CNN, when paraphrasing is not observed during training (M2), 73.3% of paraphrased samples are classified as bona fide in the user-independent setting. When paraphrasing is included during training (M3), classification accuracy increases to 72.0%, although 16.0% of paraphrased samples are still misclassified as bona fide. Transcription remains substantially easier to distinguish, with 94.7% classification accuracy under the same M3 configuration.

This behavior reflects a fundamental property of the task. Paraphrasing shares key behavioral signals associated with genuine writing, including timing variability and revision patterns, thereby reducing separability and increasing overlap with bona fide distributions.

Adversarial behavior and robustness Behaviorally manipulated samples derived from paraphrasing and transcription further stress the detection framework. In zero-shot settings without adversarial exposure, these samples are frequently misclassified as bona fide. For example, under the 1D-CNN in the user-independent M2 configuration, approximately 98.7% of adversarial samples are assigned to the bona fide class, indicating that such manipulations can closely align assisted writing with the statistical characteristics of genuine composition.

At the same time, these results highlight an important strength of the framework. When adversarial samples are incorporated during training, performance improves substantially. In the M5 setting of 1D-CNN UIE, adversarial classes are classified with accuracy above 85%, corresponding to improvements exceeding 80 percentage points relative to the zero-shot case. This demonstrates that adversarial modeling provides an effective defense: exposure to behaviorally grounded variations enables the model to recover separability and maintain robust performance under realistic threat conditions. Importantly, this improvement reflects adaptation to observed behavioral patterns, emphasizing that robustness depends on coverage of realistic writing behaviors rather than static model capacity.

Role of feature representations Temporal features degrade significantly under cognitive variation, with reduced accuracy and increased confusion between bona fide and assisted writing. Rhythmic features provide partial improvement by capturing higher-level structure such as pauses and bursts, but still exhibit overlap between paraphrasing and bona fide writing. Sequential models maintain more stable performance across most classes, indicating that effective detection requires modeling keystroke behavior across multiple temporal scales. This supports the use of sequence-based representations as a primary modeling approach for this task.

Generalization across users and contexts These trends remain consistent across both evaluation settings. In the userindependent setting, models perform well on unseen individuals, indicating that the learned representations capture task-dependent behavior rather than user-specific traits. In the context-independent setting, performance remains relatively strong on unseen prompts, particularly for sequential models, suggesting that the learned signal is not entirely prompt-

specific.

Error structure and behavioral interpretation The confusion matrices reveal a consistent and interpretable error structure aligned with behavioral similarity. Transcription remains well separated from bona fide writing, while paraphrasing and adversarial samples cluster closer to the bona fide class when not explicitly modeled. This pattern reflects differences in cognitive demand: transcription involves lower cognitive effort and remains distinct, whereas paraphrasing and behavioral manipulation increase variability and move closer to genuine writing behavior.

Overall, the results show that keystroke-based detection captures differences in writing behavior at the level of cognitive effort and interaction. While paraphrasing and adversarial manipulation reduce separability by mimicking behavioral signals, the framework remains effective when trained on sufficiently diverse data.

Ethical considerations Keystroke dynamics constitutes behavioral data and should therefore be collected with informed consent, appropriate privacy safeguards, and limited data retention. Predictions from the proposed framework should not be treated as definitive evidence of academic misconduct, but rather as an additional signal that may support further review.

Limitations The study has several limitations. The dataset includes 45 participants, with 30 returning for the second phase, and is limited to Vietnamese writing. In addition, the adversarial samples are behaviorally motivated transformations rather than data collected from users actively attempting to evade detection. Future work will evaluate the framework with larger and more diverse populations, additional languages, and human-generated adversarial behaviors.

## V. CONCLUSION AND FUTURE WORK

We investigated keystroke dynamics as a behavioral signal to detect LLM-assisted writing under both realistic and adversarial conditions. Our results show that keystroke signals capture meaningful differences in the writing process, particularly for transcription. Sequential models provide consistent gains over feature-based approaches across both user-independent and context-independent settings. At the same time, the findings expose clear limitations. Detection is not uniform across writing behaviors: paraphrasing remains difficult to distinguish from bona fide writing, and behaviorally manipulated samples were misclassified when not represented in the training set. These patterns indicate that performance is driven less by inherent separability and more by coverage of writing behaviors during training.

We further demonstrated that training with behaviorally grounded adversarial samples substantially improves robustness to the modeled adversarial behaviors, enabling models to recover strong separability for adversarial samples that are frequently misclassified as bona fide in zero-shot settings. This suggests that keystroke-based detection should be viewed as an evolving system rather than a static classifier. In practice, we envision this framework as a complementary second layer following conventional similarity-based plagiarism detection systems rather than as a replacement. Behavioral analysis is particularly valuable for submissions that exhibit little textual overlap despite possible LLM assistance.

In the future, we plan to scale the dataset to larger, more diverse populations to validate its stability across age groups, devices, and typing proficiency levels. We also plan to study more realistic adversarial settings, including adaptive users who iteratively respond to detector feedback, to better approximate deployment conditions. Finally, we plan to investigate complementary behavioral signals, such as the acoustic characteristics of keystrokes and cursor movements, to determine whether they provide additional information for detecting LLM-assisted writing.

## REFERENCES

[1] OpenAI, “Chatgpt,” https://chat.openai.com/, 2024, [Accessed 25-01- 2026].

[2] Anthropic, “Claude 3 opus,” https://www.anthropic.com/news/claude-3- family, 2024, [Accessed 25-01-2026].

[3] G. Team, “Gemini: A family of highly capable multimodal models,” 2025. [Online]. Available: https://arxiv.org/abs/2312.11805

[4] A. e. a. Grattafiori, “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[5] D. Kundu, A. Mehta, R. Kumar, N. Lal, A. Anand, A. Singh, and R. Shah, “Keystroke dynamics against academic dishonesty in the age of llms,” IEEE IJCB, 2024.

[6] D. H. Roh, R. Kumar, and A. Ngo, “Llm-assisted cheating detection in korean language via keystrokes,” in 2025 IEEE IJCB, 2025.

[7] S. Crossley, Y. Tian, J. Choi, L. Holmes, and W. Morris, “Plagiarism detection using keystroke logs,” in Proceedings of the 17th International Conference on Educational Data Mining, 2024, pp. 476–483. [Online]. Available: https://educationaldatamining.org/ edm2024/proceedings/2024.EDM-short-papers.47/

[8] A. Mehta, R. Kumar, A. Singla, K. Bisht, Y. K. Singla, and R. R. Shah, “Detecting llm-assisted academic dishonesty using keystroke dynamics,” IEEE Transactions on Biometrics, Behavior, and Identity Science, pp. 1–1, 2026.

[9] R. Kumar, N. Siddiqui, and A. Fuchsberger, “Detecting gpt-assisted writing using interpretable stylometric features,” in Proceedings of the Hawaii International Conference on System Sciences (HICSS), 2027.

[10] T. Kumarage, G. Agrawal, P. Sheth, R. Moraffah, A. Chadha, J. Garland, and H. Liu, “A survey of ai-generated text forensic systems: Detection, attribution, and characterization,” arXiv preprint arXiv:2403.01152, 2024.

[11] S. M. Alzahrani, N. Salim, and A. Abraham, “Understanding plagiarism linguistic patterns, textual features, and detection methods,” IEEE TSMC: Applications and Reviews, 2012.

[12] D. Cooke, A. Edwards, S. Barkoff, and K. Kelly, “As good as a coin toss: Human detection of ai-generated content,” Commun. ACM, 2025. [Online]. Available: https://doi.org/10.1145/3729417

[13] D. Condrey, “Detecting cognitive signatures in typing behavior for non-intrusive authorship verification,” 2026. [Online]. Available: https://arxiv.org/abs/2603.00177

[14] H. Khan, U. Hengartner, and D. Vogel, “Mimicry attacks on smartphone keystroke authentication,” ACM TOPS, 2020.

[15] R. Jemama and R. Kumar, “How well do llms imitate human writing style?” in 2025 IEEE 16th Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON), 2025.

[16] N. Agarwal, N. F. Danielsen, P. K. Gravdal, and P. Bours, “Contract cheat detection using biometric keystroke dynamics,” in 2022 20th International Conference on Emerging eLearning Technologies and Applications (ICETA), 2022, pp. 15–21.

[17] Wikipedia contributors, “Telex (input method),” https://en.wikipedia.org/ wiki/Telex (input method), 2024.

[18] W. contributors, “Vietnamese language and computers,” https://en. wikipedia.org/wiki/Vietnamese language and computers, 2024.

[19] Wikipedia contributors, “Keyboard layout (vietnamese input methods),” https://en.wikipedia.org/wiki/Keyboard layout, 2024.

[20] R. Joyce and G. Gupta, “Identity authentication based on keystroke latencies,” Commun. ACM, 1990.

[21] J. Leggett and G. Williams, “Verifying identity via keystroke characterstics,” International Journal of Man-Machine Studies, 1988.

[22] F. Monrose and A. D. Rubin, “Keystroke dynamics as a biometric for authentication,” FGCS, 2000.

[23] R. Kumar, V. V. Phoha, and A. Serwadda, “Continuous authentication of smartphone users by fusing typing, swiping, and phone movement patterns,” in IEEE BTAS, 2016.

[24] R. Kumar, P. P. Kundu, and V. V. Phoha, “Continuous authentication using one-class classifiers and their fusion,” in IEEE ISBA, 2018.

[25] J. Roth, X. Liu, and D. Metaxas, “On continuous user authentication via typing behavior,” IEEE Transactions on Image Processing, 2014.

[26] J. Roth, X. Liu, A. Ross, and D. Metaxas, “Investigating the discriminative power of keystroke sound,” IEEE TIFS, 2015.

[27] A. Kuruvilla, R. Daley, and R. Kumar, “Spotting fake profiles in social networks via keystroke dynamics,” in IEEE-CCNC, 2024.

[28] D. Kim, J. S. Valacich, J. L. Jenkins, D. W. Wilson, M. Kumar, and P. Weisgarber, “Are you, you? seamlessly fighting identity fraud with keystroke dynamics,” Information Systems Research, 2026.

[29] N.-J. Schaller, D. M. Melanchthon, T. Jansen, O. Koller, and A. Hor-¨ bach, “KEYSCORE — keystroke-enhanced automated essay scoring,” in Proceedings of the 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026), E. Kochmar, B. Alhafni, S. Banno, M. Bexte, J. Burstein, A. Horbach, R. Laarmann-Quante,\` A. Tack, V. Yaneva, and Z. Yuan, Eds. San Diego, California, USA: Association for Computational Linguistics, Jul. 2026.

[30] R. Banerjee, S. Feng, J. S. Kang, and Y. Choi, “Keystroke patterns as prosody in digital writings: A case study with deceptive reviews and essays,” in EMNLP. Association for Computational Linguistics, 2014. [Online]. Available: https://aclanthology.org/D14-1155

[31] P. Deane, M. Zhang, J. Hao, and C. Li, “Using keystroke dynamics to detect nonoriginal text,” Journal of Educational Measurement, 04 2025.

[32] M. Zhang, L. Feng, X. He, C. Li, and M. Zhu, “Disentangling copy typing and natural writing behaviors using keystroke logs and deep learning models,” Assessing Writing, 2026.

[33] M. W. Asher, G. Gold, E. Chen, and P. F. Carvalho, “Chatbots are undermining crowdsourced research in the behavioral sciences: Detecting artificial intelligence–assisted cheating with a keystroke-based tool,” Advances in Methods and Practices in Psychological Science, 2026.

[34] A. Acien, A. Morales, J. V. Monaco, R. Vera-Rodriguez, and J. Fierrez, “Typenet: Deep learning keystroke biometrics,” IEEE TBIOM, 2022.

[35] G. DeepMind, “Introducing gemini 2.5: Advanced llm,” https://blog. google/technology/google-deepmind/gemini-model-thinking-updatesmarch-2025/, 2024.

[36] T. Sim and R. Janakiraman, “Are digraphs good for free-text keystroke dynamics?” in 2007 IEEE CVPR. IEEE.

[37] S. Kiranyaz, O. Avci, O. Abdeljaber, T. Ince, M. Gabbouj, and D. J. Inman, “1d convolutional neural networks and applications: A survey,” 2019. [Online]. Available: https://arxiv.org/abs/1905.03554