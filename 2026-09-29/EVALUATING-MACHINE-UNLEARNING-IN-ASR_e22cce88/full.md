# EVALUATING MACHINE UNLEARNING IN ASR

Diogo Dinis<sup>1,2</sup>, Francisco Teixeira<sup>1</sup>, Bhiksha Raj<sup>3</sup>, Alberto Abad<sup>1,2</sup>, Isabel Trancoso<sup>1,2</sup>

<sup>1</sup> INESC-ID, <sup>2</sup>Instituto Superior Tecnico, Universidade de Lisboa, Portugal ´ <sup>3</sup>LTI, Carnegie Mellon University, Pittsburgh, PA, USA

## ABSTRACT

Machine unlearning (MU) offers a path to compliance with ”right to be forgotten” regulations. While MU has received increasing attention for speech tasks, it remains largely unexplored for Automatic Speech Recognition (ASR). In this work, we investigate whether existing MU algorithms and evaluation tools are suitable for ASR. We apply several MU techniques to an ASR model, evaluating privacyutility trade-offs for single-subject unlearning, then assess the best algorithm under sequential and simultaneous unlearning. Results show that gradient ascent-based algorithms achieve strong utilityprivacy trade-offs, whereas more complex approaches over-unlearn samples, making them easier to identify as unlearned. This suggests standard privacy evaluations based on simple Membership Inference attacks are insufficient to reliably assess unlearning success, motivating improved evaluation methods for MU in ASR. Finally, we show that both sequential and simultaneous unlearning yield worse privacy and utility than single-subject unlearning, underscoring the need for unlearning constructions better suited to these settings.

Index Terms— Machine unlearning, speech recognition, membership inference

## 1. INTRODUCTION

The widespread deployment of large-scale deep learning systems has raised serious concerns over the privacy of training data subjects. These concerns arise in part from the demonstrated vulnerability of such models to membership inference [1], model inversion [2] and data extraction attacks [3], which can enable the extraction of information, or even full retrieval of training data samples (or subjects). There is also a growing need to ensure compliance with data protection regulations worldwide, such as the European Union’s General Data Protection Regulation (GDPR) [4], or California’s Consumer Protection Act (CCPA) [5]. Among other protections, these regulations enshrine the “right to erasure” (commonly known as the “right to be forgotten”) [4], granting individuals the right to request the deletion of their personal data. While compliance with this protection is straightforward for stored data, this is not the case for trained model weights, which contain patterns or information related to training data that cannot be easily removed.

Machine unlearning (MU) techniques seeking to remove the influence of specific training samples from an already trained model have emerged as a potential practical alternative to full retraining [6], not only to ensure privacy, but also as a way to minimise biases [7] or remove learned information or classes [8]. Following the seminal work of Cao and Yang [6], early efforts on MU have largely focused on image classification models [9, 8]. More recently, the growing prominence of generative models has led to an increasing number of studies on unlearning in large language models (LLMs) [10]. In contrast, MU for speech-based models has only recently begun to attract sustained attention. Prior research has largely focused on classification tasks, including speaker identification [11], keyword spotting [11], speech emotion recognition [12, 13], and spoken language understanding [14, 15, 16]. Beyond these, MU for text-to-speech synthesis (TTS) has also started to garner interest [17, 18].

However, a significant gap remains in MU research concerning Automatic Speech Recognition (ASR). This gap is particularly notable given that ASR systems are among the most widely deployed speech-based models and, consequently, among the most likely targets of privacy attacks and user data deletion requests, along with TTS models. In addition, ASR models have been shown to memorize and leak training data under certain conditions [19, 20]. Architectural improvements have further compounded these vulnerabilities by enabling models to be prompted, thereby expanding their attack surface. As ASR model architectures progressively move towards Speech Language Models (SLMs), which couple LLMs with pre-trained speech encoders to perform recognition, these risks are likely to increase [21]. Notable efforts include the works of Liu [22] and Shamsian et al. [23]. The former evaluates Gradient Ascent [22] unlearning over synthetic canaries – artificially created samples injected into the training data to act as privacy tracking devices – providing relevant insights into memorisation in LLM-based ASR. However, the fact that canaries come from a synthetic distribution may overestimate the ability of the MU algorithm to remove the influence of data samples, as the canaries and training data will belong to different distributions. Shamsian et al.’s work, on the other hand, compares several MU methods over several tasks, including ASR, but striving to obtain an error as high as possible on the forget set, which, as we argue below, is different from our definition of MU.

In this work, we aim to further explore the application and adaptation of MU techniques to an end-to-end ASR model. We evaluate unlearning in terms of privacy and utility, following a narrow interpretation of unlearning: the unlearned model should behave similarly to a “gold standard” model trained from scratch without the data to be forgotten (the forget set). This implies that it should not be possible to distinguish between a model’s behaviour for an unlearned sample from a test sample drawn from the same data distribution.

Our results show that gradient ascent-based algorithms achieve strong privacy-utility trade-offs under standard metrics. On the other hand, more complex algorithms match these trade-offs in simple MI attacks, but leave unlearned members identifiable to unlearninginformed MI attacks, indicating that na¨ıve MI evaluations are insufficient to assess unlearning success. Moreover, the Earth mover’s distance between forget-loss distributions of the unlearned and retrained (gold-standard) models remains high, suggesting unlearning does not necessarily move models in the desired direction. Finally, sequential and simultaneous unlearning both yield worse privacyutility trade-offs than single-subject unlearning, underscoring the need for constructions better suited to these settings.

The main contributions of this work are as follows:

• We conduct the first in-depth exploration of MU applied to

ASR following a narrow unlearning formulation;

• We provide a reproducible experimental setup (including a codebase) which includes five MU methods;

• We thoroughly evaluate privacy through membership inference attacks (MIAs) in unlearning-unaware and unlearninginformed settings, and show that several MU algorithms do not hold the same privacy guarantees across the two;

• We assess unlearning under sequential and simultaneous multi-subject scenarios, showing that both settings degrade privacy and utility relative to single-subject unlearning.

## 2. MACHINE UNLEARNING IN ASR

## 2.1. Problem statement

Let $f _ { \theta } : \mathcal { X } \ \to \ \mathcal { y }$ be an ASR model, parametrised by $\theta \quad =$ $\mathcal { A } ( f , \mathcal { D } _ { t r a i n } )$ , where A is a randomised supervised learning algorithm, and $\mathcal { D } _ { t r a i n } ~ = ~ \{ ( x _ { i } , y _ { i } , s _ { i } ) \} _ { i = 1 } ^ { N }$ a training set with $N$ triples of speech recordings $x _ { i } ~ \in ~ { \mathcal { X } } ,$ , ground-truth transcriptions $y _ { i } \in { \mathcal { D } } ,$ and speaker identities $s _ { i } \in { \cal S } ,$ , drawn from a data distribution $\mathcal { D } _ { \cdot }$ . Given a deletion request from a speaker $s _ { f } \in \mathcal S ,$ , let $\mathcal { D } _ { f } \ : = \ : \{ ( x _ { i } , y _ { i } , s _ { i } ) \in \mathcal { D } _ { t r a i n } : s _ { i } = s _ { f } \}$ be the subset of speech data to be forgotten, and $\mathcal { D } _ { r } \subseteq \mathcal { D } _ { t r a i n } \ \backslash \ \mathcal { D } _ { f }$ the set of data to be retained, commonly called the “forget set” and “retain set”, respectively. Informally, the goal of a MU algorithm U is to produce $\theta ^ { \mathcal { U } }$ as close as possible to $\bar { \theta } ^ { \mathcal { G } } = \mathcal { A } ( f , \mathcal { D } _ { t r a i n } \setminus \mathcal { D } _ { f } )$ , with $\bar { \theta ^ { \mathcal { G } } }$ being the “gold standard” model, re-trained without the forget $\mathrm { s e t } ^ { 1 }$ . An unlearned ASR model that is close to a model re-trained from scratch should not have an uncharacteristically high Word Error Rate (WER) and loss for the forgotten subject, but instead WER and loss values distributionally close to unseen data from the same distribution.

## 2.2. Machine Unlearning paradigms

There are two main types of unlearning techniques: exact and approximate. Exact Unlearning is a branch of MU where algorithms are designed to provide formal guarantees that $\theta ^ { \mathcal { U } }$ was not trained on $\mathcal { D } _ { f }$ . These techniques introduce changes in the model’s architecture and training process itself, such that, to unlearn a sample, it is only necessary to re-train a small part of the model at a much lower cost than full re-training. Since the resulting model will not have been trained on $\mathcal { D } _ { f } .$ , it can be said that it has exactly forgotten it. For instance, the “Sharded, Isolated, Sliced, and Aggregated” (SISA) [9] algorithm splits the model into multiple replicas, each trained on a disjoint subset of $\mathcal { D } _ { t r a i n }$ , ensuring that the influence of any given set is confined to a single replica. This way, the retraining process for $\mathcal { D } _ { f }$ is reduced to the smaller replicas trained on its constituents, and maintains the guarantee of the absence of $\mathcal { D } _ { f }$ in $\theta ^ { \mathcal { U } }$

Approximate Unlearning methods, on the other hand, have the goal of reducing the influence of $\mathcal { D } _ { f }$ on θ, without retraining $f _ { \theta } ,$ , and without requiring specialised pre-training mechanisms. Instead, $\mathsf { A p - }$ proximate Unlearning algorithms correspond to some form of posthoc adaptation of pre-trained models, often in the form of gradient ascent on data from the forget set in combination with finetuning on the retain set, to ensure model utility is kept. In this case, however, existing algorithms do not provide exact guarantees of unlearning, and unlearning success in terms of privacy must be evaluated empirically. Nevertheless, while exact methods provide formal unlearning guarantees, their reliance on specific training algorithms makes them unsuitable for existing deployed models.

## 2.3. Evaluation

Machine unlearning algorithms need to be evaluated at two levels: utility, to ensure that the unlearning process kept the model’s performance; and unlearning success. Evaluating utility is straightforward, as the model’s performance metrics are usually well established beforehand. In contrast, evaluating unlearning success depends heavily on the unlearning objective. For privacy, unlearning success is most commonly measured in terms of a membership inference (MI) attacker’s success in correctly identifying unlearned samples as part of the training set [1]. Membership inference attacks act as a proxy for how much a model has memorised or overfitted to a sample. Other possible measures of unlearning success include data extraction attacks that attempt to retrieve information about the forget set [3], and statistical indistinguishability tests between unlearned and retrained models [25]. Nevertheless, data extraction attacks are not well developed for all applications, whereas statistical indistinguishability tests can become computationally prohibitive for large models [26].

## 2.4. Proposed Machine Unlearning methods for ASR

In this work, we adapt and evaluate the following approximate MU methods for ASR, given their wider applicability:

Baseline – Finetuning: This method performs gradient descent on $\mathcal { D } _ { r }$ to reduce the influence of $\mathcal { D } _ { f }$ on the model, deliberately overfitting $\mathcal { D } _ { r }$ to induce catastrophic forgetting in $\mathcal { D } _ { f }$

Baseline – CF-k: Unlike simple finetuning, this method, “Catastrophically forgetting the last k layers” [27], focuses only on finetuning the last k layers, freezing the preceding layers.

NegGrad and NegGrad+: NegGrad [8], or Gradient Ascent, is the most common unlearning method. The model is finetuned with $\mathcal { D } _ { f }$ using the reverse of the gradient direction, effectively moving the model’s weights in the direction of increasing loss for these samples. A popular extension of this method, NegGrad+ [28], mitigates catastrophic forgetting by additionally finetuning on $\mathcal { D } _ { r }$

SCRUB: “SCalable Remembering and Unlearning unBound” [28], or SCRUB, uses a teacher-student setup, with the original model as a teacher. This method uses three different losses: a distillation loss, instantiated as the Kullback-Leibler (KL) divergence, aiming to maximise the similarity between the student and the teacher on the retain set, $\mathcal { D } _ { r } ;$ a task loss, which is applied only to $\mathcal { D } _ { r } ;$ and the negative KL divergence, which aims to minimise the similarity between the teacher and student on $\mathcal { D } _ { f }$

ASU: Attention Smoothing Unlearning [29] also uses a teacherstudent setup, using the original model as a teacher, increasing the Softmax temperature in the teacher’s attention layers, smoothing the attention distributions, producing less confident outputs on $\mathcal { D } _ { f }$ . The student is trained to minimise the KL divergence between its and the smoothed teacher’s outputs on $\mathcal { D } _ { f }$ . Our adaptation of this method for ASR targets the attention layers of the ASR model’s decoder.

## 3. EXPERIMENTAL SETUP

## 3.1. Model selection and implementation

For the experiments in this work, we selected a state-of-the-art, open-source, pre-trained E-Branchformer $[ 3 0 ] ^ { 2 }$ to ensure a fully transparent pipeline and, most importantly, strict traceability of the datasets and data partitions used for training. This model was trained with the LibriSpeech ASR recipe from ESPnet [31], which uses the full 960 hours of training data from LibriSpeech [32].

Table 1: Data used for model training, unlearning and utility evaluation and data partitions for MI evaluation. #Spk. and #Utt. correspond to the average over the partitions for all 10 forget subjects.
<table><tr><td>Partition</td><td>#Spk.</td><td>#Utt.</td><td>Avg. Dur. (s)</td><td>Source (LibriSpeech)</td></tr><tr><td colspan="5">Unlearning &amp; Utility Eval.</td></tr><tr><td>forget</td><td>1</td><td>105</td><td>11.9</td><td>train-clean-100</td></tr><tr><td>retain</td><td>250</td><td>28,434</td><td>12.7</td><td>train-clean-100</td></tr><tr><td>test-clean</td><td>40</td><td>2,620</td><td>7.4</td><td>test-clean</td></tr><tr><td>test-other</td><td>33</td><td>2,939</td><td>6.5</td><td>test-other</td></tr><tr><td>test</td><td>73</td><td>5,559</td><td>7.0</td><td>test-clean &amp; test-other</td></tr><tr><td colspan="5">Membership Inference Eval.</td></tr><tr><td>members_train</td><td>219.8</td><td>736.6</td><td>11.9</td><td>train-clean-100</td></tr><tr><td>non-members_train</td><td>29.9</td><td>82.9</td><td>11.8</td><td>test-clean &amp; test-other</td></tr><tr><td>members_eval</td><td>1</td><td>105</td><td>11.9</td><td>train-clean-100</td></tr><tr><td>non-members_eval</td><td>29.4</td><td>82.0</td><td>11.9</td><td>test-clean &amp; test-other</td></tr></table>

Retrained Model: In order to obtain a gold standard, we retrained the full model from scratch. Although such retraining would preferably be done once for each of the subjects to be unlearned, to minimise computational costs, we opted to retrain a single model using the original model’s training set (LibriSpeech’s “train-960”), and excluding the 10 forget subjects at the same time.

All experiments were performed on a single computation node with an Intel(R) Xeon(R) Gold 6348 CPU and 3 NVIDIA RTX A6000 GPUs. We make our codebase openly available on GitHub<sup>3</sup>.

## 3.2. Data

Three partitions of LibriSpeech were used in our experiments: “train-clean-100”, “test-clean”, and “test-other”. Each was broken into different subsets for unlearning and evaluation. We designed our experiments as speaker-level unlearning tasks. For a target speaker s<sub>f</sub> , the forget set, $\mathcal { D } _ { f }$ contains all of that speaker’s utterances in “train-clean-100”, while the retain set, $\mathcal { D } _ { r }$ includes all utterances from the remaining subjects in that set. While MU implementations often set $\mathcal { D } _ { r }$ as $\mathcal { D } _ { t r a i n } \setminus \mathcal { D } _ { f }$ , we used “trainclean-100” as a representative subset of the model’s full training set (LibriSpeech’s “train-960”). In total, 10 pairs of forget/retain sets were generated, corresponding to 10 speakers to be forgotten. We reserved an additional speaker, disjoint from the previous ten, for hyperparameter search. The aforementioned data partitions are detailed in the first half of Table 1.

## 3.3. Hyperparameter selection

To better compare the chosen unlearning methods, we standardised the unlearning process to 10 epochs with a fixed batch size of 8. We determined the remaining hyperparameters through a Bayesian search using the Optuna [33] library’s default algorithm, a Treestructured Parzen Estimator sampler. Considering $\mathcal { L } _ { M } ( S )$ as the distribution of per-utterance losses of model M over set $S ,$ our objective was defined as the minimisation of the Earth Mover’s Distance (EMD), between the loss distributions of the forget set and the pre-unlearning test set, i.e., $\mathrm { E M D } ( \mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { f } ) , \mathcal { L } _ { \theta } ( \mathcal { D } _ { t e s t } ) )$ .

## 3.4. Utility evaluation

To evaluate utility, we compute the mean WER and model loss values, corresponding to the Connectionist Temporal Classification (CTC) loss from the model’s encoder output and the cross-entropy (CE) loss computed over the decoder’s output, for the forget, retain, and test sets, for each speaker to forget. We include the loss, as a direct measure of the alignment between the model’s behaviour on the forget set, and its performance on the retain and test sets. We use the model’s default decoding configuration, except for the beam size, which is set to 5.

## 3.5. Privacy evaluation

To evaluate unlearning success in terms of privacy, we created data partitions and implemented two types of MI attacks: simple and informed. Both are evaluated in terms of Area Under the Curve (AUC) and Equal Error Rate (EER). We additionally include the $\mathrm { E M D } ( \mathcal { L _ { \theta \upsilon } } ( \mathcal { D } _ { f } ) , \mathcal { L _ { \theta \upsilon } } ( \mathcal { D } _ { r } ) )$ , $\mathrm { E M D } ( \mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { f } ) , \mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { t e s t } ) )$ and $\operatorname { E M D } ( \mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { f } ) , \mathcal { L } _ { \theta ^ { \mathcal { G } } } ( D _ { f } ) )$ , representing how close the loss distribution over the forget set is to the corresponding model’s retain and test sets, as well as to the forget set’s loss distribution on the re-trained model. These metrics are used to gain a more in-depth understanding of the results obtained for the MI attackers.

Attack partitions: MI attacks require observed and non-observed data (members and non-members). The attacker is trained on utterances from the D<sub>r</sub> as positives (members train) and from $\mathcal { D } _ { t e s t }$ as negatives (non-members train), and subsequently evaluated on $\mathcal { D } _ { f }$ as positives (members test) against held-out samples from $\mathcal { D } _ { t e s t }$ as negatives (non-members test). The evaluation asks whether the attack recognises the forgotten sample as a member, which, for a perfectly unlearned model, should yield an AUC of 50%. After observing in initial experiments that mean utterance duration varies substantially between the original train and test sets, making duration a confounding factor for MI attacks, MI partitions were matched by duration. Further details of the partitions are provided in the second half of Table 1.

Simple Attacker: employs a Random Forest (RF) classifier trained with the target models’ encoder’s CTC loss and decoder’s crossentropy loss [34]. The RF is trained on utterances from members train and non-members train, and evaluated on members eval and non-members eval. However, this attack has no reference for how the model behaves with regard to an unlearned sample. As such, if the unlearned samples’ losses diverge from the retain samples’ losses (e.g., by having a very high loss), this attacker might fail to recognise them as belonging to the original training set.

Informed Attacker: follows the simple attacker, but attempts to address its na¨ıve construction. Specifically, the informed MI attacker leverages all unlearned models to train the RF classifier in a leaveone-out fashion – i.e., for each forget subject, we leverage the remaining 9 other subjects’ unlearned model losses as an “\`ınformed” members train, along with the losses for the same models over nonmembers train, to train the RF; and then use the target forget and test subject samples’ losses computed on the unlearned model for evaluation. This way, the classifier is able to learn the behaviour of a model with regard to forget samples after unlearning.

## 4. RESULTS

The results for the application of the MU algorithms of Section 2.2 to ASR can be found in Table 2.

Utility: We observe that NegGrad, NegGrad+, SCRUB and AttSmooth have the closest WER values to $\theta ^ { \mathcal { G } }$ , for all sets. On the other hand, in all cases, the two baseline methods (finetune and CF-k) cause noticeable overfitting to the retain set and markedly degrade performance in both test partitions. In terms of mean loss distributions, the algorithms behave in the same way as for the WER for all sets except for the $\mathcal { D } _ { f } .$ , wherein both versions of NegGrad and SCRUB highly degrade the loss.

Privacy: Using the simple MIA attack, results behave similarly to those observed for utility. Finetune and CF-k have the worst performances in terms of privacy, whereas the remaining methods present strong privacy improvements. The best performances are achieved by NegGrad+ and AttSmooth. Note that, while NegGrad, NegGrad+ and SCRUB have higher losses, this does not translate into worse privacy results. This comes from the fact that the simple MI attack is trained to distinguish between train and test losses, which means that, while these MU algorithms may increase $\mathcal { D } _ { f } \mathbf { \bar { s } }$ loss by a large amount, $\mathcal { D } _ { f } \ '$ s loss distribution may still be closer to the test set than to $\mathcal { D } _ { r }$ . This is validated by the EMD columns reported in the table, where, in all cases that the EMD between $\mathcal { L } _ { \theta ^ { U } } ( D _ { f } )$ and $\mathcal { L } _ { \theta ^ { U } } ( D _ { t e s t } )$ is smaller than the EMD between $\mathcal { L } _ { \theta ^ { U } } ( D _ { f } )$ and $\mathcal { L } _ { \theta ^ { U } } ( D _ { r } )$ , the MI classifier has very poor performance.

Table 2: Main utility and MIA results. WER and Loss should be close to the re-trained model. AUC and EER should be close to 50%. The EMD between $\mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { f } ) , \mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { t e s t } )$ and $\mathcal { L } _ { \theta ^ { \mathcal { G } } } ( D _ { f } )$ should be low, while EMD between $\mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { f } )$ and $\mathcal { L } _ { \boldsymbol { \theta } ^ { U } } ( \mathcal { D } _ { r } )$ should be high.
<table><tr><td></td><td colspan="2">Forget (Df)</td><td colspan="2">Retain (Dτ)</td><td colspan="2">Test-Clean</td><td colspan="2">Test-Other</td><td colspan="3">EMD (vs  $\mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { f } ) )$ </td><td colspan="2">MIA</td><td colspan="2">IMIA</td></tr><tr><td>Method</td><td>WER (%)</td><td>Loss</td><td>WER (%)</td><td>Loss</td><td>WER (%)</td><td>Loss</td><td>WER (%)</td><td>Loss</td><td> $( \mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { r } ) )$ </td><td> $( \mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { t e s t } ) )$ </td><td> $( \mathcal { L } _ { \theta ^ { \mathcal { G } } } ( D _ { f } ) )$ </td><td>AUC (%)</td><td>EER (%)</td><td>AUC (%)</td><td>EER (%)</td></tr><tr><td>Original</td><td>0.40</td><td>1.22</td><td>0.46</td><td>1.19</td><td>1.89</td><td>3.04</td><td>4.00</td><td>5.09</td><td>0.31</td><td>3.17</td><td>3.26</td><td>67.48</td><td>36.45</td><td>一</td><td>1</td></tr><tr><td>Re-trained</td><td>2.01</td><td>4.51</td><td>0.41</td><td>1.58</td><td>2.18</td><td>3.22</td><td>5.02</td><td>5.33</td><td>3.03</td><td>2.56</td><td></td><td>51.97</td><td>49.70</td><td>1</td><td>一</td></tr><tr><td>Finetune</td><td>1.84</td><td>4.46</td><td>0.16</td><td>0.76</td><td>3.54</td><td>6.17</td><td>9.80</td><td>13.24</td><td>4.28</td><td>5.06</td><td>1.73</td><td>68.01</td><td>36.16</td><td>58.45</td><td>43.46</td></tr><tr><td>CF-k</td><td>1.45</td><td>3.78</td><td>0.14</td><td>0.77</td><td>3.01</td><td>5.49</td><td>6.11</td><td>9.54</td><td>3.49</td><td>3.66</td><td>1.10</td><td>63.87</td><td>39.86</td><td>56.81</td><td>46.08</td></tr><tr><td>NegGrad</td><td>2.01</td><td>6.62</td><td>0.51</td><td>2.89</td><td>1.99</td><td>4.24</td><td>4.21</td><td>6.30</td><td>4.28</td><td>3.58</td><td>3.00</td><td>52.13</td><td>47.67</td><td>53.95</td><td>47.09</td></tr><tr><td>NegGrad+</td><td>2.36</td><td>10.36</td><td>0.55</td><td>1.61</td><td>2.03</td><td>3.41</td><td>4.25</td><td>5.57</td><td>9.79</td><td>7.58</td><td>8.91</td><td>50.18</td><td>49.35</td><td>52.69</td><td>46.60</td></tr><tr><td>SCRUB AttSmooth</td><td>1.94</td><td>6.48</td><td>0.42</td><td>0.97</td><td>1.94</td><td>2.97</td><td>4.08</td><td>5.01</td><td>6.13</td><td>4.19</td><td>4.23</td><td>46.00</td><td>51.78</td><td>68.98</td><td>35.87</td></tr><tr><td></td><td>2.09</td><td>3.83</td><td>0.13</td><td>0.58</td><td>2.03</td><td>3.41</td><td>4.32</td><td>5.72</td><td>3.19</td><td>3.14</td><td>2.41</td><td>52.09</td><td>50.29</td><td>87.57</td><td>20.00</td></tr><tr><td>NegGrad Seq.</td><td>3.82</td><td>15.11</td><td>2.93</td><td>12.60</td><td>4.77</td><td>9.80</td><td>8.84</td><td>13.16</td><td>0.86</td><td>3.39</td><td>2.78</td><td>59.39</td><td>42.02</td><td>62.30</td><td>41.07</td></tr><tr><td>NegGrad+ Seq.</td><td>2.01</td><td>4.70</td><td>0.75</td><td>2.07</td><td>3.39</td><td>3.84</td><td>6.81</td><td>6.28</td><td>2.69</td><td>2.76</td><td>3.69</td><td>55.58</td><td>45.20</td><td>56.32</td><td>44.77</td></tr><tr><td>NegGrad Sim.</td><td>0.55</td><td>3.16</td><td>0.45</td><td>3.03</td><td>2.66</td><td>4.01</td><td>5.80</td><td>5.95</td><td>0.61</td><td>3.43</td><td>2.80</td><td>65.28</td><td>37.40</td><td>67.45</td><td>38.51</td></tr><tr><td>NegGrad+ Sim.</td><td>3.32</td><td>14.91</td><td>1.22</td><td>4.63</td><td>4.00</td><td>5.79</td><td>7.53</td><td>8.45</td><td>10.13</td><td>8.69</td><td>10.18</td><td>51.57</td><td>48.35</td><td>58.47</td><td>43.80</td></tr></table>

Informed attacker: Regarding the informed attacker’s results, we observe a very large privacy degradation for both SCRUB and AttSmooth, with the informed MI attack achieving an AUC of 87.57% for AttSmooth, compared to the 52.09% obtained by the simple MI attacker. To understand this large discrepancy, we analysed the classifier’s decision boundary for AttSmooth. We found that while the simple attacker is able to rely on the combination of both losses to identify members and non-members, the informed attacker, trained with unlearned samples, relies much more strongly on higher CE losses to predict members (unlearned samples). This is likely due to the fact that smoothing is only applied in decoder layers, making this loss much higher for unlearned samples.

If we consider only MI and WER, NegGrad+ has the strongest privacy and utility trade-off, closely followed by NegGrad. Nevertheless, NegGrad is the more computationally efficient alternative, as it only requires gradient ascent on $\mathcal { D } _ { f }$ , contrary to NegGrad+, that also finetunes on $\mathcal { D } _ { r }$ . However, we also note that the lowest $\operatorname { E M D } ( \mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { f } ) , \mathcal { L } _ { \theta ^ { \mathcal { G } } } ( \mathcal { D } _ { f } ) )$ ) are obtained by finetuning and CF-k. This means that, while some of the tested unlearning methods prevent successful MIAs, they are not yielding models that behave exactly like the re-trained model, which is the true goal of unlearning.

Sequential and simultaneous unlearning: In a real-world scenario, deletion requests might accumulate or come in batches. We therefore evaluated the two gradient ascent-based methods by unlearning subjects sequentially and simultaneously, using the hyperparameters selected for single-subject unlearning. The results are presented at the end of Table 2. Under the sequential approach, NegGrad severely degrades the model, with the retain WER rising to 2.93% against 0.41% on the retrained model. Although its MIA AUC of 59.39% is the highest among the multi-subject runs, the low EMD between $\mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { f } )$ and $\mathcal { L } _ { \theta ^ { U } } ( \mathcal { D } _ { r } )$ (0.86) provides no evidence of forget-set memorisation, instead indicating convergence of the forget and retain distributions. NegGrad+ improves on this metric through the retain finetuning, and its forget-set metrics are the closest to $\mathbf { \bar { \theta } } ^ { \mathcal { G } }$ of all four multi-subject runs. Even so, it still degrades test performance, with a Test-Other WER of 6.81% against 4.25% and 5.02% on the single-subject case and retrained model, respectively, and a rise in MIA AUC from 50.18% to 55.58% when compared to the singlesubject case. Under simultaneous unlearning, the two methods fail in opposite directions. NegGrad fails to unlearn, with a forget WER of 0.55% and loss of 3.16, close to the original model (0.40% and 1.22), as does the MIA AUC (65.28%), suggesting that sharing the gradient ascent over ten speakers dilutes the per-speaker signal. Neg-Grad+, by contrast, over-unlearns, reaching a forget loss of 14.91 and the largest distance to $\theta ^ { \mathcal { G } }$ in the table, with an EMD of 10.18. Overall, the single-subject hyperparameters do not transfer reliably to multi-subject unlearning.

Limitations: Even though the current work focuses on a single ASR model trained on a single dataset, we consider that this is sufficient to validate the conclusions that can be taken from this paper. Nevertheless, we believe that future work should focus on extending the experiments conducted in this paper to other model architectures and datasets. Membership inference attacks were also limited to simple loss classification, which had a weak original performance (∼67%). More complex constructions exist in the literature [35, 36] and should be explored in the context of MU for ASR.

## 5. CONCLUSIONS

In this paper, we focused on studying and evaluating MU algorithms for ASR. Overall, our simple MI-based evaluation results identified four algorithms (NegGrad, NegGrad+, SCRUB and AttSmooth) with strong trade-offs between privacy and utility. However, under unlearning-informed MI attacks, SCRUB and AttSmooth were found to produce loss patterns that allowed the identification of forgotten samples. Results for sequential and simultaneous unlearning, which mimic real-world conditions, were also found to degrade both privacy and utility. In addition, our results showed an inversion between EMD and MIA: the methods most resistant to MIA were farthest from the gold standard in terms of forget-set loss distribution, while those closest to it were most vulnerable. Resistance to MIAs therefore does not immediately imply absence of information about the forget data in the unlearned models. We therefore consider that MU-informed MIA constructions are necessary for a thorough evaluation of privacy, while distributional measures should be included in evaluation setups to further validate similarity to gold standard models. This study highlights MU as a compelling and critical open problem. By benchmarking MU algorithms and exposing flaws in standard evaluation metrics, this work tried to provide a stepping stone for future research in MU for ASR and other speech tasks.

## 6. ACKNOWLEDGMENTS

This work was supported by national funds through Fundac¸ao para a ˜ Ciencia e a Tecnologia, I.P. (FCT) under projects UID/50021/2025,ˆ UID/PRR/50021/2025 and CMU-Portugal project https://do i.org/10.54499/2024.14611.CMU (LeaF).

## 7. REFERENCES

[1] R. Shokri, M. Stronati, C. Song, et al., “Membership inference attacks against machine learning models,” in Proc. IEEE SP. 2017, pp. 3–18, IEEE Computer Society.

[2] S. Hidano, T. Murakami, S. Katsumata, et al., “Model Inversion Attacks for Prediction Systems: Without Knowledge of Non-Sensitive Attributes,” in Proc. PST, 2017, pp. 115–11509.

[3] N. Carlini, F. Tramer, E. Wallace, et al., “Extracting Train-\` ing Data from Large Language Models,” in USENIX Security, 2021, pp. 2633–2650.

[4] European Parliament and Council, “On the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC (General Data Protection Regulation),” 2016, Pages: 1–88 Volume: L 119.

[5] State of California, “The california consumer privacy act of 2018 (CCPA),” 2018, California Civil Code § 1798.100 et seq.

[6] Y. Cao and J. Yang, “Towards Making Systems Forget with Machine Unlearning,” in IEEE Symposium on Security and Privacy, 2015, pp. 463–480.

[7] D. Liu, Y. Liu, G. Jin, et al., “Mitigating biases in language models via bias unlearning,” in Proc. EMNLP. 2025, pp. 4160– 4178, ACL.

[8] A. Golatkar, A. Achille, and S. Soatto, “Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks,” in Proc.CVPR, 2020, pp. 9301–9309.

[9] L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, et al., “Machine Unlearning,” in 2021 IEEE Symposium on Security and Privacy (SP), 2021, pp. 141–159.

[10] V. Dorna, A. R. Mekala, W. Zhao, et al., “OpenUnlearning: Accelerating LLM unlearning via unified benchmarking of methods and metrics,” in Proc. NeurIPs, 2026.

[11] J. Cheng and H. Amiri, “Speech unlearning,” in Interspeech, 2025, pp. 3209–3213.

[12] O. C. Phukan, Girish, M. M. Akhtar, et al., “Towards machine unlearning for paralinguistic speech processing,” in Interspeech, 2025, pp. 4473–4477.

[13] Z. Ren, R. A. Rammohan, K. Scheck, et al., “Machine Unlearning in Speech Emotion Recognition via Forget Set Alone,” 2025, arXiv.

[14] A. Koudounas, C. Savelli, F. Giobergia, et al., ““Alexa, can you forget me?” machine unlearning benchmark in spoken language understanding,” in Interspeech, 2025, pp. 1768–1772.

[15] C. Savelli, A. Koudounas, F. Giobergia, et al., “UnSLU-BENCH+: Extended Machine Unlearning Benchmark for Spoken Language Understanding,” IEEE TASLP, vol. 34, pp. 1892–1902, 2026.

[16] A. Singh and V. K. Kurmi, “Selective capability unlearning in end-to-end spoken language understanding,” 2026.

[17] T. Kim, J. Kim, D. C. Kim, et al., “Do not mimic my voice : Speaker identity unlearning for zero-shot text-to-speech,” in Proc. ICML, 2025.

[18] M. Lee, E. Shin, and J. Lee, “Erasing your voice before it’s heard: Training-free speaker unlearning for zero-shot text-tospeech,” in Proc. ICASSP, 2026, pp. 17627–17631.

[19] V. Shejwalkar, O. Thakkar, and A. Narayanan, “Quantifying Unintended Memorization in BEST-RQ ASR Encoders,” in Interspeech 2024, 2024, pp. 2905–2909.

[20] L. Wang, O. Thakkar, and R. Mathews, “Unintended memorization in large asr models, and how to mitigate it,” in Proc. ICASSP, 2024, pp. 4655–4659.

[21] M. Zufle and J. Niehues, “When helpful context leaks: Privacy¨ risks in domain-adapted asr,” 2026.

[22] Z. Liu, “Unlearning LLM-based speech recognition models,” in Interspeech, 2025, pp. 3214–3218.

[23] A. Shamsian, E. Shaar, A. Navon, et al., “Go Beyond Your Means: Unlearning with Per-Sample Gradient Orthogonalization,” 2026.

[24] E. Triantafillou, A. I. Humayun, M. Ribero, et al., “Is your algorithm unlearning or untraining?,” 2026, arXiv:2604.07962 [cs.LG] version: 1.

[25] C. Zhang, M. Li, F. Liu, et al., “Unlearning evaluation through subset statistical independence,” in Proc. ICLR, 2026.

[26] E. Triantafillou, P. Kairouz, F. Pedregosa, et al., “Are we making progress in unlearning? findings from the first neurips unlearning competition,” arXivpreprint arXiv:2406.09073, 2024.

[27] S. Goel, A. Prabhu, A. Sanyal, et al., “Towards Adversarial Evaluations for Inexact Machine Unlearning,” 2023, arXiv.

[28] M. Kurmanji, P. Triantafillou, J. Hayes, et al., “Towards unbounded machine unlearning,” in Proc. NeurIPs. 2023, NIPS ’23, pp. 1957–1987, Curran Associates Inc.

[29] S. Z. Zade, X. Zhou, S. Liu, et al., “Attention Smoothing Is All You Need For Unlearning,” in Proc. ICLR, 2025.

[30] K. Kim, F. Wu, Y. Peng, et al., “E-Branchformer: Branchformer with Enhanced Merging for Speech Recognition,” in IEEE SLT Workshop, 2022, pp. 84–91.

[31] S. Watanabe, T. Hori, S. Karita, et al., “ESPnet: End-to-end speech processing toolkit,” in Proc. Interspeech, 2018, pp. 2207–2211.

[32] V. Panayotov, G. Chen, D. Povey, et al., “Librispeech: An ASR corpus based on public domain audio books,” in Proc. ICASSP, 2015, pp. 5206–5210.

[33] T. Akiba, S. Sano, T. Yanase, et al., “Optuna: a next-generation hyperparameter optimization framework,” in Proc. SIGKDD. 2019, Kdd ’19, pp. 2623–2631, ACM.

[34] F. Teixeira, K. Pizzi, R. Olivier, et al., “Exploring features for membership inference in ASR model auditing,” CSL, vol. 95, pp. 101812, 2026.

[35] J. Tao and R. Shokri, “Information-theoretic membership inference for granular quantification of memorization,” in Proc. ICLR, 2026.

[36] N. Jebreel, M. Khalil, D. Sanchez, et al., “Revisiting the´ lira membership inference attack under realistic assumptions,” 2026.