# Lost but not erased: Finding traces of a forgotten language in neural speech models

Peter Plantinga, Charlotte Moore, Peter W. Donhauser, Krista Byers-Heinlein, & Denise Klein

## Abstract

International adoptees retain phonological traces of a birth language they can no longer speak or comprehend, a persistence typically attributed to a biologically-timed critical period. We asked whether it could instead reflect the ordinary dynamics of learning, using automatic speech recognition models that simulate the international adoptee experience without maturational confounds. Models were trained on one language and then abruptly switched to a second. We found that traces of the first language persisted throughout second-language training, but mainly in the lowest, pre-phonemic layers. These traces were functional, as models with early exposure re-learned their lost first language 14% faster than naive models; this advantage held even against models adopted early from a related language and disappeared when the earliest layers were substituted from a non-adopted model. We argue that these critical-period effects reflect entrenchment of foundational representations rather than a maturational loss of plasticity, and that experience plays a central role in critical periods in language acquisition.

## Introduction

International adoptees offer a natural experiment for understanding the effects of early language exposure on subsequent brain development. These individuals abruptly lose access to their pre-adoption language early in life, then rapidly acquire their post-adoption language, typically catching up with their peers within a few years<sup>1</sup>. In adulthood, they exhibit linguistic performance on par with lifelong monolingual speakers of the post-adoption language, with no conscious recollection of their pre-adoption language<sup>2</sup>.

This apparent completeness of adoptees’ birth language loss is at odds with critical periods in language acquisition<sup>3,4</sup>, often described as a maturationally-based window of heightened neural plasticity in which early experience should leave a lasting trace. Indeed, careful experimentation has revealed that the pre-adoption language is not fully lost. For example, Mandarin-to-French international adoptees show brain activation patterns that better match speakers with continuous Mandarin access than French monolinguals<sup>5,6</sup>. Adoptees also show faster relearning of the pre-adoption language: Korean-born international adoptees raised in Sweden and the Netherlands have shown accelerated relearning of Korean phonological contrasts<sup>2,7</sup>. This faster relearning is a modern instance of the classic savings effect first described by Ebbinghaus<sup>8</sup>: material that appears to have been forgotten is reacquired more quickly than it was initially learned. Together, these findings show that traces of the phonology (sounds) of the pre-adoption language persist despite years of post-adoption language exposure and no conscious memory of the lost language.

Yet, other research has demonstrated that adoptee advantages do not extend to all aspects of language. Korean-Swedish adoptees’ advantage on the pre-adoption language held only for phonology, and adoptees scored below non-adopted Swedish speakers on higher-complexity tasks like Korean grammaticality judgments<sup>2</sup>. A similar asymmetry appears in artificial systems, even though they lack biological maturation: deep neural networks initially trained on blurred images showed persistent deficits in classification accuracy, but not when the images were instead flipped vertically, a manipulation that preserves fine-grained spatial statistics<sup>9</sup>. These results suggest that entrenchment of early-learned information is not uniform across a learning system but is concentrated in its foundational, low-level representations.

Complex perceptual domains are represented hierarchically in both biological and artificial systems: successive processing stages extract low-level features, such as visual textures or acoustic spectro-temporal edges, and progressively combine them into higher-level categories such as objects or words<sup>10,11</sup>. This hierarchy could explain the asymmetry observed above. Because higher-level categories are built from lower-level features, revising a lower-level representation mid-stream could destabilize everything computed on top of them, whereas a high-level category could be revised with comparatively little cost, since few downstream computations depend on it.

This hierarchical explanation has direct empirical support in adult language acquisition. Finn and colleagues found that learning a constructed language with unfamiliar sounds required adults to recruit more general auditory processing networks rather than relying cleanly on native-language networks, concluding that adult language learning difficulties stem in part from the rigidity of phonetic neural scaffolding<sup>12</sup>. Under this view, low-level representations of sounds resist mid-stream changes, likely because revising these foundational structures risks destabilizing the higher-level representations that depend on them. We propose that critical-period effects can arise from this pressure to preserve a stable statistical foundation, without requiring an appeal to biological maturation.

This proposal remains untested, however, because no existing study has isolated the mechanism: human studies show the phenomenon but cannot rule out biological maturation as its cause, while artificial-network studies have focused on behavioral metrics in computer vision<sup>9,13</sup> or high-level text modeling<sup>14</sup> without localizing the critical-period effects. By contrast, this study begins to isolate the mechanisms of critical-period effects by simulating the adoptee experience in deep speech models and investigating the internal representations of the speech encoder. Using layer-wise representational analyses, we directly test where in the network language traces persist, providing mechanical evidence in favor of statistical explanations for critical-period effects, rather than biologically-timed maturational schedules.

More specifically, we have used state-of-the-art automatic speech recognition (ASR) models<sup>15</sup>, which must learn phonological structure from raw speech input much as human infants do. Because the system was artificial, we could precisely control factors such as the learning rate throughout training, eliminating the maturational confound that limits interpretation of human studies. We previously validated this modeling framework for bilingual phonological development, demonstrating the emergence of parallel phonological systems in simultaneous bilinguals and cross-language transfer in sequential bilinguals<sup>16</sup>.

3. RSA

Here we extended this approach to model international adoptees: each model was first trained on a pre-adoption language $( L _ { p r e } )$ before training was abruptly switched to a post-adoption language $( L _ { p o s t } )$ , mimicking the international adoptee experience. Whereas previous studies of humans examined adoptees initially exposed to a tonal language that later acquired a non-tonal language, we trained adoptee models on non-tonal languages that are more closely related (French, German, and English), thus examining the generalizability of international adoptee effects.

Our adoptee models $( M _ { a } )$ reproduced the behavioral profile of human international adoptees, rapidly losing proficiency in $L _ { p r e }$ , while acquiring native-like proficiency in $L _ { p o s t }$ . They also showed similar advantages in identifying $L _ { p r e }$ phonemes that human international adoptees show in forced-choice tasks<sup>7</sup>. Examining their representational geometry revealed that traces of $L _ { p r e }$ persisted selectively in the model’s earliest layers, mirroring the neural traces of a lost birth language documented in human international adoptees<sup>5,6</sup>, and this persistence carried functional consequences for how quickly the lost language could be relearned, echoing relearning advantages documented behaviorally in human adoptees<sup>2,7</sup>. Varying the duration of initial training revealed an optimal window for trace preservation: with extended pre-adoption exposure, representational traces peaked before plateauing and slowly declining. Finally, swapping the earliest layers with a non-adopted model removed a significant fraction of the adoptee re-learning advantage. Together, our findings offer a mechanistic account of how early experience can leave lasting, consequential structure in a learning system without any change in its capacity to learn.

## Analytic Approach and Results

Our overall method is presented in Figure 1, consisting of four steps: 1. model training, 2. similarity matrix generation, 3. representational similarity analysis (RSA), and 4. trajectory analysis.

1. Model Training  
![](images/a716aede5bfec0b755cb2d6cf5d581fe3bc09ba9592652f250eb41aa0e70c11c.jpg)

2. Similarity Matrix  
![](images/07fc16e07a9925d7c6a9654ea4a45984f431f072e66485128c6ab187cc45a7c6.jpg)

4. Trajectory Analysis  
![](images/c0bd1b6364352b17e89cb472e8c5ee182373ebae836ef58ad95979fac6002f6d.jpg)  
Figure 1. Schematic of the modeling and analysis pipeline. We trained automatic speech recognition (ASR) models under an abrupt language switch and found traces of the lost language by comparing adoptee models to single-language models using representational similarity analysis (RSA). Step 1 (model training): We trained models under different language schedules: adoptee models $( M _ { a } )$ were trained on a pre-adoption language $( L _ { p r e } )$ and then a post-adoption language $( { \cal L } _ { p o s t } ) ,$ and single-language models $( M _ { p r e } , M _ { p o s t } )$ were trained on one language each. Dots on the training trajectory mark saved checkpoints. Step 2 (similarity matrix generation): Within a model, and at a given layer, we computed pairwise cosine similarity $( \angle )$ of representations for a fixed set

of held-out $L _ { p r e }$ samples to form a representational similarity matrix. Step 3 (RSA): We compared the similarity matrices of two models using Spearman's rank correlation (⍴), quantifying how similarly they organize the same samples. Step 4 (trajectory analysis): We tracked RSA over the course of training to determine whether traces of the pre-adoption language remained stable throughout extended post-adoption training.

Across all experiments, we trained encoder–decoder ASR models on French, English, and German speech and abruptly switched their training language to simulate international adoption (see Methods). Importantly, we did not explicitly train the model on the phonology of any language, similar to other models where phonology emerged from word-level training<sup>16,17</sup>. We switched the language after a consistent number of updates across experiments: after models reached basic proficiency in its pre-adoption language, measured by word error rate (WER, the proportion of words transcribed incorrectly; lower scores are better) but long before final convergence. For most experiments we use 8.4k updates, after which French WER was 34.2%, German 13.8%, and English 27.6%. Throughout, we compared these adoptee models $( M _ { a } )$ against baseline models trained on a single language $( M _ { p r e }$ and $M _ { p o s t } )$

## Behavioral Links

We first asked whether our artificial adoptee models match the behavioral profile of human international adoptees. Throughout training, we tracked proficiency in each language using next-token prediction accuracy: the model's ability to predict the next linguistic unit of an utterance from the audio and the preceding text, where each unit is a subword token from a data-driven tokenizer. We used next-token accuracy rather than WER here because it provides a fast-to-compute and low-variance view of accuracy, rather than relying on slow search algorithms that can fail for entire utterances well into training. To track the loss of $L _ { p r e }$ specifically, we paired the adoptee model’s evolving encoder with the original pre-adoption decoder (from $M _ { p r e } )$ after the switch, letting us measure how accessible $L _ { p r e }$ remained as $L _ { p o s t }$ training proceeded.

![](images/b66d05aaf89ce156d4d2d923715a1fd23025f9fdb1c5a2c13bd8a0e7d4315393.jpg)

b  
![](images/20de5f6a99c660bf5ee472388ea28aa1a089ca6edc49023b2a83ed5fff5c8180.jpg)  
Figure 2. The adoptee model rapidly loses its pre-adoption language while gaining a new one, mirroring human international adoptees. Figures show next-token identification accuracy of an adoptee model $( M _ { a } )$ at two timescales. In both panels, the y-axis is next-token prediction accuracy, the model’s ability to predict the next subword token from the audio and the preceding text. Lines show the mean across 3 seeds, bands around all lines indicate 95% CIs. Panel a: Shortly after the switch, $M _ { a }$ accuracy on the pre-adoption language $( L _ { p r e }$ German, in blue) fell sharply, while its accuracy on the post-adoption language $( L _ { p o s t } ;$ French, in red) rose over the

same period. The x-axis shows the number of training updates in the post-adoption language. Panel b: Over the full course of training, M (red) reached the same $L _ { p o s t }$ (French) accuracy as compared to a model only trained on French $( M _ { p o s t } ,$ yellow). The x-axis shows the number of updates since training began in any language; the vertical dotted line marks the onset of $L _ { p o s t }$ training for $M _ { a }$

The adoptee model showed a sharp decline in its $L _ { p r e }$ (German) recognition accuracy, but gained proficiency in $L _ { p o s t }$ (French) over the same timespan, mirroring the typical pattern in human international adoptees who rapidly acquire the adopted language. Recognition accuracy in $L _ { p r e }$ declined over 4000 updates to 25.9% (see Figure 2), approaching the accuracy obtained when the decoder was given randomly shuffled inputs instead of a real audio representation (25.4%), indicating that the decoder acted as a next-token predictor (i.e. ignored the audio entirely). At the same time, $L _ { p o s t }$ accuracy rose rapidly, reaching 80% after only 3000 updates post-switch, 30% fewer steps than it took $M _ { p o s t }$ to reach the same level. With the total number of updates (on any language) held fixed, $M _ { a }$ and $M _ { p o s t }$ converged to less than 0.1% accuracy difference (93.6% vs. 93.7%). This pattern aligns with observed behavior in human international adoptees, who lose conscious access to their birth language and in most cases<sup>18</sup> gain proficiency similar to their monolingual peers in the adopted language<sup>1,19</sup>.

A second well-documented aspect of human international adoptee behavior is an enhanced ability to discriminate phonemic contrasts from their birth language<sup>2</sup>. We asked whether $M _ { a }$ shows an analogous advantage in its representations of $L _ { p r e }$ phonemes. Because human experiments ask participants to identify phonemes with little explicit feedback<sup>20</sup>, we used a linear probe<sup>21</sup>, trained for 300 steps. We froze each model and trained a simple linear classifier on the internal hidden states of each layer to predict the corresponding time-aligned phoneme (see Methods). Above-baseline probe accuracy therefore reflects phonemic information already present in the frozen representations rather than new phonemic learning.

a  
![](images/b3187303b5b10a12eef3c940b71ab124f0ad20cbd14dde49dff31709f3c99d64.jpg)

b  
![](images/0c9216b3b034a9d79d85b7e89adac79f244e9db130db5bc2490e02fb0dbb2a84.jpg)  
Figure 3. The adoptee model retained a phoneme-discrimination advantage for its pre-adoption language in early processing (pre-phonemic) layers. Linear phoneme probe classification accuracy over the layers of the encoder. Panel a: Absolute accuracy of a linear probe trained to classify pre-adoption phonemes $( L _ { p r e } ;$ German) across representations from single-language models (German, French) and adoptee models $( M _ { a } =$ German→French; $M _ { c t r l } = \mathsf { E n g l i s h {  } F r e n c h } )$ . The x-axis is the encoder layer index (1-12); probe accuracy is the percentage of frames correctly labeled with their aligned phoneme. Shaded regions denote functional layer groupings (Pre-phonemic, Phonemic, and Post-phonemic). Confidence bands represent 95% intervals across model training runs. Panel b: Normalized probe accuracy for $M _ { a }$ and $M _ { c t r l }$ aggregated by layer group across six

language switch conditions. Accuracy is expressed relative to single-language benchmarks where $M _ { p r e } = 1 0 0 \%$ (maximum) and $M _ { p o s t } = 0 \%$ (baseline). Connected points represent a family of models like the example given in panel a, evaluated on phonemes from the pre-adoption language of the adoptee model.

Figure 3a shows the probe’s accuracy across encoder layers for an example language switch direction (i.e. German → French). For all four compared model families, probe accuracy peaked in the middle layers (5-8), establishing them as the models’ “phonemic layers.” Both $M _ { a }$ and $M _ { c t r l }$ showed an advantage over the $M _ { p o s t }$ baseline at the earlier pre-phonemic layers $( p = 0 . 0 0 1 1$ and $p = 0 . 0 1 9$ respectively, one-sided Mann-Whitney U), as well as at the phonemic layers for $M _ { a }$ but not $M _ { c t r l } \left( p = 0 . 0 1 3 \right.$ and $p = 0 . 0 5 7 )$ , and not for the post-phonemic layers either $( p > 0 . 9$ for both). This pattern extended across language pairs (Figure 3b) where it can be seen that the advantage of $M _ { a }$ over $M _ { p o s t }$ was largest at pre-phonemic layers and shrank as depth increased. Furthermore, the probe was more accurate for $M _ { a }$ than $M _ { c t r l }$ at the pre-phonemic and phonemic layers $( p = 0 . 0 1 6$ for both, one-sided Wilcoxon signed-rank) but not at post-phonemic layers $( p = 0 . 2 8 )$

These data reveal two distinct sources of early-exposure advantage. First, the strong phonetic identification ability of $M _ { c t r l }$ in Figure 3a and the highlighted portion of Figure 3b is most naturally read as positive forward transfer, in which early exposure to any Germanic phonology deposits acoustic-phonetic features that also serve languages in the same family, such as spectral edges or formant ranges. Second, the consistent $M _ { a } .$ -over- $. M _ { c t r l }$ increment from Figure 3b indicates an advantage attributable to $L _ { p r e }$ exposure specifically, over and above the family-level transfer that $M _ { c t r l }$ also enjoys. Together these results suggest that the adoptee models retain low-level, language-family-specific patterns from $L _ { p r e }$ most strongly at the foundational levels of representation, rather than intact higher-level phonemic categories.

## Representational Traces of Lost Language

Neuroimaging studies have shown identifiable traces of early language experience when stimuli from a lost language are presented to international adoptees<sup>5</sup>. To look for an analogous trace in our models, we used representational similarity analysis $( { \mathsf { R S A } } ) ^ { 2 2 }$ to compare how models with different language histories organize their internal representations. RSA is well suited here since it captures whether two models represent the same stimuli similarly without requiring that the representations be in the same space, handling geometric differences that arise due to random initialization and data ordering (e.g. rotations and translations). We computed RSA using 500 held-out $L _ { p r e }$ speech samples, comparing each model’s representational geometry layer by layer (see Methods).

Figure 4a shows RSA scores across model pairs for models with different training schedules. We bracketed the range with two bounds: an upper bound from pairs of same-language models trained on different seeds (high similarity) and a lower, cross-language bound from $M _ { p o s t } - M _ { p r e }$ pairs (low similarity). These baselines represent the level of architectural similarity we would expect from two monolinguals who either speak the same language or two different languages. Against these bounds, we tracked the similarity of $M _ { a }$ to both $M _ { p r e }$ and $M _ { p o s t }$ over the course of $L _ { p o s t }$ training.

![](images/88155d7ecedb672bbe9fd6a21009746d8b9f1fbeeece86cd8d9ab7b3db06961f.jpg)

![](images/2cd41679f524f81bf40ce15bf505cbf21ea82b713bcad20252dcab24741c10d0.jpg)  
Figure 4. Representational traces of the lost language that persist after behavioral loss are concentrated in the lowest layers, and peak during an optimal pre-training window. Representational similarity analysis (RSA) of adoptee models $( M _ { a } )$ against single-language models, computed on held-out pre-adoption language $( L _ { p r e } )$ samples. Confidence bands show 95% confidence intervals across 3 seeds. Panel a: For a German → French adoptee model given 8.4k steps of $L _ { p r e }$ training, the similarity of $M _ { a }$ and $M _ { c t r l }$ and single-language $M _ { p r e }$ and $M _ { p o s t }$ models over the course of post-adoption training $( L _ { p o s t } )$ , averaged across the pre-phonemic layers (1-4) The control model was an English → French adoptee model. The x-axis is updates of $L _ { p o s t }$ training. Horizontal grey lines mark similarity scores for two monolingual models trained on the same language (high similarity, dashed line) and a lower, cross-language similarity baseline (low-similarity, dotted line). $M _ { a } \mathbf { \ ' } \mathbf { s }$ similarity to $M _ { p o s t }$ approaches this upper bound, whereas its similarity to $M _ { p r \epsilon }$ stabilized above the cross-language baseline, indicating a persistent trace of the lost language. Panel b: The representational trace, defined as $M _ { a } { } ^ { \circ }$ similarity to $M _ { p r e }$ above the cross-language baseline, as a function of the number of $L _ { p r \epsilon }$ training steps. The solid red line shows the trace for the pre-phonemic layers (1-4), the average of the four language switch conditions shown in light grey; the heavy dashed grey line shows the average for these four language switch conditions across the other encoder layers (5-12), showing that the trace is concentrated in the lowest layers.

Immediately after language switch, $M _ { a }$ representations rapidly approached $M _ { p o s t }$ representations, confirming robust adaptation to the new language. The crucial finding is that well after accuracy on $L _ { p r e }$ had fallen to chance (4k steps, see Figure 2), $M _ { a }$ looked more similar to $M _ { p r e }$ than a $M _ { p o s t }$ model without $L _ { p r e }$ exposure. Moreover, this residual similarity did not fade, as even after new-language training equivalent to thousands of hours of experience, the representational geometry stabilized rather than continuing to converge on $M _ { p o s t }$ . As can be seen in Figure 4a, after around 15k updates the $\mathsf { R S A } ( M _ { a , } M _ { p r e } )$ curve flattened, averaging 8.3% (95% $C | = 2 . 9 \%$ to 13.7%) over the last 5k steps. The control model’s score was also significantly more similar than chance (mean 6.3%, 95% CI 2.4% to 10.2%). As with the phoneme probe, this significant control trace indicates that part of the representational similarity reflects language family-level transfer, i.e. Germanic phonetic information rather than German transfer specifically; $M _ { a } \mathbf { \ ' }$ numerically higher score is consistent with an additional $L _ { p r \epsilon }$ -specific increment, but the overlapping intervals mean RSA does not, on its own, cleanly separate the two components. That separation is achieved by the relearning experiment below.

To determine where and when these traces formed, we evaluated representational similarity across encoder layers and initial exposure lengths (Figure 4b). As in the phoneme probe, traces were sharply localized to the earliest, pre-phonemic layers (layers 1–4), reaching a relative RSA of 4% to 8%, whereas the remaining 8 higher-level encoder layers showed virtually no residual trace (remaining near zero baseline). Crucially, the magnitude of these pre-phonemic traces did not scale continuously with $L _ { p r e }$ training. Instead, relative RSA

nearly doubled between 3.4k and 5.0k pretraining updates (rising from \~4% to \~8%), reached a clear peak around 5.0k–6.7k steps, and began a slow tail-off by 8.4k steps. This non-linear pattern was consistent across 3 out of 4 individual language switch pairs (Figure 4b, light grey lines), while the English → French $M _ { a }$ models show consistently stronger representational traces with more pre-training.

Taken together, these results show that representational traces of the lost language persist through extended new-language training, peaking after a limited pre-training duration rather than accumulating indefinitely.

## Savings on Relearning

Finally, we asked whether these traces serve a functional purpose or are merely residual imprints from early language experiences. We investigated this by testing how quickly adoptee models could relearn their lost language.

a  
![](images/6fe217fd94ff7c788d370499b2f6ccaa6886aebcd134250f8e28be81ba461eba.jpg)

b  
![](images/675f12607364dadb8c4d0ceacd78c2a3900870a1d159829e0d13eec6f02c0edf.jpg)  
Figure 5. The adoptee model relearned its lost language faster than control models, but lost this advantage when the earliest layers are artificially excised. Relearning of pre-adoption language (here, German) by models with various language backgrounds. Panel a: Next token accuracy scores for next-token prediction as a function of number of learning updates on $L _ { p r e }$ (here, German). X-axis shows the number of $L _ { p r e }$ updates the model has experienced in the new learning schedule. Y-axis shows next-token accuracy. Red represents $M _ { a } ,$ models exposed to German for 8.4k updates and then abruptly switched to French. Blue represents $M _ { c t r l } ,$ adoptee models whose previous language experience do not include German (instead, English). Cyan represents $M _ { p o s t }$ models that have also never been exposed to German. Horizontal dotted line marks the fixed accuracy threshold (70%) used to define a target for comparison; vertical lines show the point at which each model crosses the threshold. Color bands represent 95% CIs. Panel b: Relative advantage of $M _ { a }$ over $M _ { p o s t }$ in learning speed when portions of the encoder have been exchanged between the two models. Speed is measured in steps to 70% accuracy when relearning the pre-adoption language $L _ { p r e } .$

As shown in Figure 5a, $M _ { a }$ relearned $L _ { p r e }$ faster than both $M _ { p o s t }$ models and a control model $( M _ { c t r l } )$ that was adopted from a third language, reaching a fixed 70% accuracy threshold in 14.3% fewer steps than $M _ { p o s t }$ (95% CI 12.3% to 16.2%) and 12.8% fewer steps than $M _ { c t r l }$ (95% CI 10.9% to 14.8%). Because $M _ { c t r l }$ shares $M _ { a } { } ^ { \bullet }$ adoption history but lacks early $L _ { p r e }$ experience specifically, this advantage cannot be attributed to a generic benefit of having learned another language early $\mathsf { o n } ;$ it depends on prior exposure to $L _ { p r e }$ . This is the savings effect<sup>8</sup>, previously demonstrated in deep networks for computer vision<sup>23</sup>, and indicates that relearning is facilitated by the retained phonological traces of the lost language.

To test what portion of the network drives the relearning advantage, we conducted a mechanistic localization experiment with pre-phonemic, phonemic, and post-phonemic portions of the encoder, as shown in Figure 5b. We spliced different sets of layers between $M _ { a }$ and $M _ { p o s t } ,$ and tested the effect on relearning speed, isolating the effect of the representations from any general effect of the splicing operation by only comparing the relative performance between the models. Splicing the pre-phonemic layers consistently resulted in the smallest gap between the models, confirming that these layers serve a critical role in supporting the relearning of the pre-adoption language. The differential contribution of early versus late network layers suggests that the adoptee advantage is driven primarily by representations analogous to early-developing auditory–phonological systems rather than later-emerging higher-order linguistic representations.

## General Discussion

Human language learners show a surprising combination of flexibility and entrenchment in language acquisition, as exemplified by populations like international adoptees who demonstrate lasting neural traces of a forgotten language without any overt knowledge of it<sup>5,6</sup>. We used a neural network model trained on raw human speech data to investigate the neural traces left behind by previous language experience. Our results reproduced the same effects observed in international adoptees: models showed a complete behavioral loss of the pre-adoption language and eventual native-like proficiency in the post-adoption language. Of note, representational traces of the pre-adoption language in the model’s geometry persisted, particularly at the lowest levels of representation. These traces in turn conferred a functional advantage, accelerating re-learning of the pre-adoption language phonemic contrasts relative to models with equivalent language experience but no early exposure.

These results support a hierarchical account in which early learning persists only when it establishes foundational representations. Consistent with this view, the persistent representational trace and its causal contribution to relearning was confined to the lowest, pre-phonemic layers, while the decodable phoneme advantage extended only slightly higher, into the phonemic layers, and vanished entirely at the post-phonemic and higher levels. We propose that early-formed low-level representations become entrenched and are reused, in modified form, when input changes, whereas higher-level representations built upon them are more readily overwritten. This interpretation is consistent with evidence that prior learning alters the neural mechanisms supporting subsequent learning: for example, rats trained on one spatial navigation task no longer require the same brain plasticity to learn a second task<sup>24</sup>, and early language experience shapes neural responses to later phonological working memory tasks<sup>6</sup>. Rather than simply demonstrating that early language experience leaves a lasting trace, our findings identify the type of representation that persists and the stage of processing at which it exerts its influence. Although deep neural network layers are not direct analogues of brain regions, they provide a computational means of localizing the relearning advantage to early processing stages, suggesting that early language exposure leaves durable low-level representations that facilitate later relearning.

A central theme across our analyses is that behavioral performance underestimates what a learning system retains. Prior work exploring critical period effects for artificial learners has largely tracked behavioral trajectories following early deprivation <sup>9</sup>. By instead tracking

internal speech representations after a sudden language change, we observed structures that model behavior did not reveal. The representations of our adoptee model always remained distinguishable from those of a monolingual of either language, despite near-identical behavioral proficiency (93.6% versus 93.7% accuracy). This finding bolsters the relationship between our model and human adoptees, who show neural traces even when phonemic discrimination behaviour shows no advantage<sup>5,6</sup>. This distinction may also explain why phonemic discrimination tasks used with human Korean-Swedish adoptees <sup>2</sup> detect a birth-language advantage at all: such tasks can be completed using any acoustic information that distinguishes the relevant sounds, including pre-phonemic cues. These tasks may therefore draw on the same low-level traces we find in our models, rather than on intact phonemic-level representations per se. Moreover, we found advantages for English-French adoptee models later tested on German, suggesting that in humans, low-level phonetic encoding skills should transfer across closely related languages<sup>25</sup>, which would be consistent with the broader second language phonology literature<sup>26,27</sup>. Techniques that are sensitive to lower-level auditory encoding (e.g. frequency-following responses) may be well suited to test these predictions.

While our model demonstrates several connections between speech models and human international adoptee learners, several limitations should be noted. First, our models do not develop like humans do. The network’s size, depth and learning rate were determined a priori rather than arrived at through maturation. We therefore cannot speak to how biological maturation might interact with these learning dynamics in humans. Our results constrain what a maturational account must explain, rather than eliminating a role for biology. Second, we directly tested low-level phonological processing, while our account predicts that higher-level representations such as grammar should show weaker latent preservation. An important future direction will be to integrate these aspects of language into a single artificial language model.

There are other, more practical limitations. Our models underwent a total and abrupt cessation of $L _ { p r e }$ input after the language switch. In most real-world cases of language attrition, including but not limited to international adoption, speakers continue to have some contact with their birth language, whether through occasional conversations, media, daily routines, or internal experiences such as dreams or recollections<sup>28</sup>. Exploring the effects of small, punctuated episodes of language re-exposure on model behavior and representations could strengthen the connection to human experience and help determine the minimum amount of continued input needed to maintain or strengthen latent representations. Finally, we did not exhaustively explore the space of modelling decisions, including network size and data scale and set of languages. We note, however, that the central phenomenon does not appear to be architecture- or task-specific as parallel results have been observed in convolutional vision models<sup>9</sup>. Systematically mapping how these effects vary with scale, objective, architecture, and languages remains an important direction for consolidating this evidence base.

Our results sharpen the question of the role of biology in development, rather than diminishing it. Because experience is intrinsic to every learning process and can by itself generate critical period effects, the experiential account should serve as the null hypothesis. On this view, a maturational claim about any given effect should have to show that the effect exceeds what learning dynamics alone produce<sup>29</sup>. Our findings give that null hypothesis

concrete empirical force in a naturalistic domain, showing that the lasting effects of early language exposure observed in international adoptees need not be attributed entirely to a biologically-determined critical period, because a system with fixed plasticity already reproduces them. Any biological account of these effects cannot simply point to their persistence as evidence of a maturational window: it must specify what maturation adds beyond learning dynamics alone.

More broadly, our findings reframe what early language exposure accomplishes. Even when it leaves no measurable skill, early exposure can create durable latent structures that persist long after exposure has ceased and provide a basis for future learning. Because this structure is a product of learning dynamics rather than biological maturation, it may be possible to shape it through learning conditions, including the order in which material is presented. This idea is already exploited in machine learning as curriculum learning<sup>30</sup>, and is relevant to educational programs such as early immersion and heritage-language maintenance. More fundamentally, our results suggest that the entrenchment of early learning, long taken as a signature of biological maturation, may instead reflect a general property of how learning systems build representations from the bottom up, with foundational structure becoming fixed as later learning comes to depend on it. Critical period effects, in this view, can arise from the order in which experience is organized over time, without requiring a biological window.

## Methods

## Model Training

All models were trained and tested on spectrograms derived from the Common Voice dataset version 22.0<sup>31</sup>, a publicly available repository of human speech from thousands of speakers across dozens of languages. We extracted about 1,000 hours from three of the CommonVoice languages with the most data: English, French, and German, and held out 2% (20 hours) for both validation and test sets. For each language in our dataset, we trained a separate SentencePiece unigram tokenizer<sup>32</sup> with a vocabulary size of 5,120 tokens, chosen to balance dictionary size and encoding length.

On this data, we trained ASR models using a state-of-the-art off-the-shelf CommonVoice ASR recipe from the SpeechBrain toolkit<sup>33</sup>, which follows an encoder-decoder formula with Conformer architecture, consisting of 12 encoder and 6 decoder layers for a total of 109.3 million parameters. The encoder ingests 80-dimensional mel-spectral features via an intervening CNN to reduce dimensionality along both time and frequency by a factor of 4, and outputs a multidimensional representation of the utterance for further processing. The decoder module receives the encoded representation, along with its own previous predictions, to transform the audio into words and sentences in an autoregressive manner. The recipe achieved near state-of-the-art performance for the CommonVoice dataset when no pretraining was used, getting a WER of 6.17% on our German test set, and 9.97% on our French test set.

One crucial detail that we changed from the off-the-shelf recipe is removing late-training learning rate decay. While standard recipes for ASR include a warmup period and a

cooldown period, we included only the linear warmup period (12k updates), needed for convergence, but excluded any kind of cooldown period in middle or later stages of training. We did this to ensure that the findings of neural traces cannot be attributed to a kind of “freezing” effect, where rapid learning rate decay removes the potential of later updates to affect the geometry enough to remove traces of the early learning experiences. This does come at a small performance cost, as adding one epoch of training with learning rate cooldown to the end of training improved WER by close to 1% absolute (6.17% → 5.34% for German), but this ensured our results were generalizable effects of the learning process itself, and not a byproduct of a specific learning rate schedule. This mirrors the results of Achille and colleagues<sup>9</sup> who found that the sensitive period effects were reduced but not eliminated when learning rate cooldown was removed.

To simulate the adoptee experience, we trained each model on a pre-adoption language $( L _ { p r e } )$ and then abruptly switched the training data and tokenizer to a post-adoption language $( L _ { p o s t } )$ after 2-5 epochs (3.4k-8.4k updates). When switching the language, we kept all trained parameters of both the encoder and decoder for the new language training, changing only the tokenizer and data. This approach retains the greatest possible chance of saving representational geometry across the model’s representational hierarchy.

For each language pair and pretraining length we trained at least three models, each with a different random weight initialization and data ordering. Confidence bands in the figures represent 95% CIs over seeds, computed using a non-parametric bootstrapping algorithm. Statistical tests are over random seeds except for one test over language switch directions in the phonemic linear probe. False discovery rate correction (Benjamini-Hochberg) applied to the p-values listed in the paper does not change any conclusion at a 0.05 significance level.

## Representational Similarity Analysis

Once the training was complete, the second step was to construct a representational similarity matrix for each model over a set of 500 stimuli randomly selected from the held-out data from $L _ { p r e }$ . Choosing the pre-adoption language allowed us to see if samples from the forgotten language were still able to activate sub-networks of the adoptee model above the level of a model that had never heard utterances from $L _ { p r e }$ before.

For each of the 500 speech stimuli from $L _ { p r e , }$ we constructed a single-vector representation of the sample by averaging over time the internal model representations for a given layer when presented with the sample. Then, we compared all pairs of stimulus vector representations using cosine similarity to construct the similarity matrix. The diagonal and replicated half of the matrix were discarded before correlating matrices to avoid artificially inflating similarity scores.

In the third step, we compared similarity matrices using Spearman’s rank test to estimate the degree to which two models hold similar or different internal representations for the given set of stimuli. The full set of comparisons we conducted are as follows:

1. RS $\backslash ( M _ { p o s t } , M _ { p o s t } )$ — The maximum expected correlation comes from separate models that are trained on the same language: we compared pairs of models with different random initializations—but both trained using $L _ { p o s t }$ —to provide a topline comparison for whether the adoptee model fully adapts to the second domain.

2. RS $\mathsf { A } ( M _ { p r e } , M _ { p o s t } )$ —The minimum expected correlation comes from two models that have entirely non-overlapping training data coming from separate domains. These models can still exhibit some similarity from overlap in the task being performed: for example, some phonemes sound very similar across languages.

3. RS $\mathsf { a } ( M _ { a } , M _ { p r e } )$ —The question we were investigating was whether the adoptee models’ similarity with models from the pre-adoption domain would remain above a model that had never seen data from that first domain. We compared the correlation here against the $\mathsf { R S A } ( M _ { p r e } , M _ { p o s t } )$ correlation score listed above to determine if higher-than expected correlation could be found, a result we term “representational traces” of the first seen language.

4. RS $\mathbf { \partial } \cdot \mathsf { A } ( M _ { a } , M _ { p o s t } ) -$ Due to the recent experience of both $M _ { a }$ and $M _ { p o s t }$ on the same language, ${ \sf R S A } ( M _ { a } , M _ { p o s t } )$ should nearly reach the topline ${ \sf R S A } ( M _ { p o s t } , M _ { p o s t } )$ level, but due to the distinct early language experiences of $M _ { a }$ some representational differences could remain, which we expected to be of roughly the same magnitude as the $\mathsf { R S A } ( M _ { a } , M _ { p r e } )$ score from the baseline.

Finally, we calculated the amount of representational traces remaining and a 95% CI around this value by averaging the RSA scores over the last 5k steps to arrive at a stable estimate of representational traces per random initial seed (5 for $M _ { a }$ and 7 for $M _ { p o s t } )$ , using a student’s t-distribution for the small-sample CI estimate.

For the experiment on the duration of pretraining, we use a slightly different formulation of representational traces based on a headroom-normalized comparison against only the baseline without consideration for the topline placement (i.e. $( \mathsf { R S A } ( M _ { a } , M _ { p r e } ) - \mathsf { R S A } ( M _ { p r e }$ $M _ { p o s t } ) / \left( 1 - \mathsf { R S A } ( M _ { p r e } , M _ { p o s t } ) \right)$ . This is due to some language switches having a very small baseline-topline gap in the early layers causing the variance to be too high to extract recognizable patterns.

## Phonemic Linear Probe

To determine if the adoptee model retained a phoneme-learning advantage in a similar manner as international adoptees, we trained a linear probe (i.e. an affine transformation) to predict, for each frame, the corresponding aligned phoneme. To conduct this experiment, we relied on the Montreal Forced Aligner $( M F A ) ^ { 3 4 }$ toolkit with MFA-trained language-specific acoustic models (version 3.0.0) to generate estimates of phoneme timings based on a comparison of the raw audio with the ground-truth transcripts. We limited the phoneme inventory to a set of 51 overlapping phonemes by collapsing allophones over the three languages. For each frame, we then selected the phoneme that occupied the majority of the frame as the label during linear probe training.

For each model, we conducted the probe experiment by first freezing all the weights of the model encoder, and applying a separate linear probe to the outputs of each layer, trained with a cross-entropy loss. The probe trained quickly with only 300 updates at a constant 0.001 learning rate needed to reach an accuracy plateau. A total of 5 models out of 93 were removed as outliers due to very poor phonemic probe performance (<50% accuracy at the middle layer), likely due to an inability to distinguish one or more crucial common phonemes.

The statistical tests determining if linear probe scores exceeded the baseline were conducted over 6 seeds for the adoptee model and 4 seeds for the control model (2 excluded as outliers), by averaging the normalized scores at each layer into an overall score for each model and layer bin combination. Then a non-parametric one-sided Mann-Whitney U test was performed to test if the model scores exceeded the baseline. We further aggregated scores over language switch directions, resulting in six conditions per model (adoptee or control) per layer bin. Then we tested whether the direction of the effect over the 6 language pairs was significant using a one-sided Wilcoxon signed-rank test.

## Savings on Relearning

To test for a “savings effect” in all of our models, we (re-)trained our fully-converged models on $L _ { p r e } .$ . We used an accelerated warmup schedule (1000 updates to full learning rate) as the models were already well-structured and did not require a long warmup to converge. We measured validation performance every 100 steps and used linear interpolation between the two nearest points to find the number of training steps needed to reach our arbitrary accuracy threshold.

To localize the relearning speed advantage in the adoptee model, we spliced a sequence of layers from each $M _ { a }$ with a corresponding $M _ { p o s t }$ and vice-versa. After splicing, we froze the entire model except the first and last fully-connected portions of the spliced layers, and adapt on 2000 steps of the $L _ { p o s t }$ to compensate for rotations or translations of the representations between models. Then, language re-learning proceeded as before. Confidence intervals of the adoptee advantage were computed using bootstrapping over model seeds.

## Data Availability

The CommonVoice dataset is available through the Mozilla Foundation.

## Code Availability

Code is available at https://github.com/pplantinga/bilingual\_networks.

## References

1. Roberts, J. A. et al. Language development in preschool-age children adopted from China. J. Speech Lang. Hear. Res. JSLHR 48, 93–107 (2005).

2. Hyltenstam, K., Bylund, E., Abrahamsson, N. & Park, H.-S. Dominant-language replacement: The case of international adoptees. Biling. Lang. Cogn. 12, 121–140 (2009).

3. Johnson, J. S. & Newport, E. L. Critical period effects in second language learning: the influence of maturational state on the acquisition of English as a second language. Cognit. Psychol. 21, 60–99 (1989).

4. Lenneberg, E. H. The Biological Foundations of Language. Hosp. Pract. 2, 59–67 (1967).

5. Pierce, L. J., Klein, D., Chen, J.-K., Delcenserie, A. & Genesee, F. Mapping the unconscious maintenance of a lost first language. Proc. Natl. Acad. Sci. 111, 17314–17319 (2014).

6. Pierce, L. J., Chen, J.-K., Delcenserie, A., Genesee, F. & Klein, D. Past experience shapes ongoing neural patterns for language. Nat. Commun. 6, 10073 (2015).

7. Choi, J., Broersma, M. & Cutler, A. Early phonology revealed by international adoptees’ birth language retention. Proc. Natl. Acad. Sci. 114, 7307–7312 (2017).

8. Ebbinghaus, H. Über Das Gedächtnis. (1885).

9. Achille, A., Rovere, M. & Soatto, S. Critical learning periods in deep neural networks. (2019).

10. Creutzfeldt, O. D. & Nothdurft, H. C. Representation of complex visual stimuli in the brain. Naturwissenschaften 65, 307–318 (1978).

11. Krizhevsky, A., Sutskever, I. & Hinton, G. E. ImageNet Classification with Deep Convolutional Neural Networks. in Advances in Neural Information Processing Systems vol. 25 (Curran Associates, Inc., 2012).

12. Finn, A. S., Hudson Kam, C. L., Ettlinger, M., Vytlacil, J. & D’Esposito, M. Learning language with the wrong neural scaffolding: the cost of neural commitment to sounds. Front. Syst. Neurosci. 7, (2013).

13. Kleinman, M., Achille, A. & Soatto, S. Critical Learning Periods for Multisensory Integration in Deep Networks. in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 24296–24305 (2023). doi:10.1109/CVPR52729.2023.02327.

14. Constantinescu, I., Pimentel, T., Cotterell, R. & Warstadt, A. Investigating Critical Period Effects in Language Acquisition through Neural Language Models. Trans. Assoc. Comput. Linguist. 13, 96–120 (2025).

15. Gulati, A. et al. Conformer: Convolution-augmented Transformer for Speech Recognition. Preprint at https://doi.org/10.48550/arXiv.2005.08100 (2020).

16. Moore, C., Donhauser, P. W., Klein, D. & Byers-Heinlein, K. Efficient neural encoding as revealed by bilingualism. Proc. Natl. Acad. Sci. 122, e2513768122 (2025).

17. Magnuson, J. S. et al. EARSHOT: A minimal neural network model of incremental human speech recognition. Cogn. Sci. 44, e12823 (2020).

18. Delcenserie, A. & Genesee, F. The acquisition of accusative object clitics by IA children from China: Evidence of early age effects? J. Child Lang. https://doi.org/10.1017/s0305000913000500 (2013) doi:10.1017/s0305000913000500.

19. Snedeker, J., Geren, J. & Shafto, C. L. Starting Over: International Adoption as a Natural Experiment in Language Development. Psychol. Sci. 18, 79–87 (2007).

20. Laurence, S. & Margolis, E. The Poverty of the Stimulus Argument. Br. J. Philos. Sci. 52, 217–276 (2001).

21. Alain, G. & Bengio, Y. Understanding intermediate layers using linear classifier probes. Preprint at https://doi.org/10.48550/arXiv.1610.01644 (2018).

22. Kriegeskorte, N., Mur, M. & Bandettini, P. A. Representational similarity analysis - connecting the branches of systems neuroscience. Front. Syst. Neurosci. 2, (2008).

23. Mayo, D. et al. Multitask Learning Via Interleaving: A Neural Network Investigation. Proc. Annu. Meet. Cogn. Sci. Soc. 45, (2023).

24. Bannerman, D. M., Good, M. A., Butcher, S. P., Ramsay, M. & Morris, R. G. Distinct components of spatial learning revealed by prior training and NMDA receptor blockade. Nature 378, 182–186 (1995).

25. Abdullah, B., Zaitova, I., Avgustinova, T., Möbius, B. & Klakow, D. How Familiar Does That Sound? Cross-Lingual Representational Similarity Analysis of Acoustic Word Embeddings. in Proceedings of the Fourth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP (eds Bastings, J. et al.) 407–419 (Association for Computational Linguistics, Punta Cana, Dominican Republic, 2021). doi:10.18653/v1/2021.blackboxnlp-1.32.

26. Smith, L. C. & Baker, W. Learning L3: why learning French first is better than learning German first. in Proceedings from the XVI International Congress of Phonetic Sciences 1629–1632 (2007).

27. Major, R. C. Transfer in second language phonology: A review. Phonol. Second Lang. Acquis. 63–94 (2008).

28. Schrauf, R. W. & Rubin, D. C. Internal languages of retrieval: The bilingual encoding of memories for the personal past. Mem. Cognit. 28, 616–623 (2000).

29. Müller, H. M. Childhood Language Acquisition. in Introduction to Neurolinguistics (ed. Müller, H. M.) 41–60 (Springer, Berlin, Heidelberg, 2025). doi:10.1007/978-3-662-71906-0\_4.

30. Bengio, Y., Louradour, J., Collobert, R. & Weston, J. Curriculum learning. in Proceedings of the 26th Annual International Conference on Machine Learning 41–48 (Association for Computing Machinery, New York, NY, USA, 2009). doi:10.1145/1553374.1553380.

31. Ardila, R. et al. Common Voice: A Massively-Multilingual Speech Corpus. Preprint at https://doi.org/10.48550/arXiv.1912.06670 (2020).

32. Kudo, T. & Richardson, J. SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. Preprint at https://doi.org/10.48550/arXiv.1808.06226 (2018).

33. Ravanelli, M. et al. SpeechBrain: A General-Purpose Speech Toolkit. Preprint at https://doi.org/10.48550/arXiv.2106.04624 (2021).

34. McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M. & Sonderegger, M. Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi. in Interspeech 2017 498–502 (ISCA, 2017). doi:10.21437/Interspeech.2017-1386.

## Author Information

These authors contributed equally: Peter Plantinga, Charlotte Moore.

These authors jointly supervised this work: Krista Byers-Heinlein, Denise Klein.

## Affiliations

McGill University Department of Neurology and Neurosurgery Peter Plantinga, Charlotte Moore, Denise Klein

Concordia University Department of Psychology Charlotte Moore, Krista Byers-Heinlein

## Ernst Strüngmann Institut

Peter Donhauser

Centre for Research on the Brain, Language, and Music Peter Plantinga, Charlotte Moore, Krista Byers-Heinlein, Denise Klein

## Mila Quebec Artificial Intelligence Institute

Peter Plantinga

## Contributions

All authors were involved in conceptual development. P.P. and C.M. designed the experimental protocols. P.P. and P.D. wrote the experimental code. P.P. ran the experiments and generated the figures. K.B.-H. and D.K. provided expert guidance and feedback. All authors participated in writing and editing the manuscript.

Corresponding Author Peter Plantinga