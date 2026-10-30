# A Novel Approach to Evaluating Evaluation Metrics for Multi-Output Structured Prediction

Akshay Vyas, Angelo Pimienta, Nicholas Ruozzi

The University of Texas at Dallas akshay.vyas, angelomatthew.pimienta, nicholas.ruozzi@utdallas.edu

## Abstract

In multi-output structured prediction tasks, while only one ground truth label may be provided in the training data, multiple equally valid outputs may be possible, making reliable evaluation a persistent challenge. We postulate that human evaluators implicitly use task-specific invariants, e.g., object boundaries in colorized images or named entities in translations, to judge if an output is acceptable. Under this assumption, we introduce a notion of approximate task-specific invariants and use them as diagnostic tools to evaluate a variety of existing metrics for vision and language tasks. We use these task invariants as part of a framework to systematically test metric reliability by encouraging domain-relevant invariants in model outputs via an augmented loss function. In our experiments, we observe that enforcing invariants with an augmented loss yields substantial improvements in popular distributional metrics while more traditional metrics change only marginally. Through this invariants-driven evaluation, we expose where standard metrics fail to detect meaningful differences, and we highlight the conditions under which distributional metrics succeed or still fall short.

## Code —

https://github.com/akshay25vyas/invariants-metric-eval Proceedings version — https://doi.org/10.1609/aaai.v40i31.39808

## Introduction

Machine learning models trained for multi-output vision and language tasks often face an evaluation paradox: there may exist many plausible outputs for a given input, e.g., different colorizations of a photo or multiple valid translations of a sentence, yet traditional evaluation metrics assume a single ground-truth output. In vision, pixel-level metrics such as mean squared error (MSE) or peak signal-to-noise ratio (PSNR) treat any deviation from the single ground-truth image as an error, even if the deviation is a valid alternative, e.g., a shirt colored blue instead of green. These metrics can also be poor proxies for perceptual quality, e.g., an image can have low MSE yet look unnatural to humans. In language, n-gram overlap metrics such as Bilingual Evaluation Understudy Score (BLEU), while standard, can be brittle: a translation that uses synonyms or different phrasing will get a low BLEU score despite conveying the same meaning.

Overall, when trained models output one of many acceptable answers, the above metrics can often over penalize legitimate variation while under penalizing more human perceptible errors, which can lead trained models to produce labels that may have low error with respect to the chosen metric but be easily identified as poor by human evaluators.

Recently, distributional/learned metrics have emerged as attempts to address the above issues. Distributional metrics compare the distribution of labels produced by a given model to the distribution of the ground truth labels. For images, the Frechet Inception Distance (FID) (Heusel et al.´ 2017) computes the Wasserstein-2 distance between multivariate normal distributions fit to model generated labels and ground-truth labels in a feature space extracted from the Inception network. As only distributions are considered, these metrics allow for diversity beyond pixel agreement. For text, BERTScore leverages pre-trained language model embeddings to measure semantic similarity between the model-generated text and ground-truth reference. Distributional metrics can better handle paraphrasing or varying image styles, allowing them to be less influenced by superficial differences between the predicted and the ground-truth labels. However, an open question remains: Do these distributional metrics capture the variations that human evaluators expect, or do they have blind spots of their own? For example, if a translation preserves meaning but shifts tone, will Frechet BERT Distance (FBD) register the change? Does´ FID drop when an image’s colors violate object boundaries, or could it be fooled by overall color statistics? These questions highlight the need for a systematic study of metric sensitivity to meaningful variations in output.

The main postulate in this work is that human evaluators implicitly or explicitly apply domain or task invariants when judging neural network outputs. Invariants are properties that should remain (approximately) unchanged under valid transformations of the output. For example, a human evaluator for the image colorization task might expect that the addition of color does not alter the image’s structural details, e.g., the model colorized image should respect the same object boundaries and depth cues as the original ground-truth color image. If neural network colorization causes “color bleeding” across object edges, humans might immediately flag the colorization as incorrect, even if the colors appear plausible overall. Similarly, when assessing machine translation, bilingual speakers might expect the translation to preserve named entities and sentiment, e.g., a translated sentence that flips the sentiment or mistranslates a proper name violates one of these invariants. In image captioning, an accurate caption should reflect the image’s content and relationships, e.g., who is doing what to whom. So, properties like semantic roles and scene relationships act as invariants, e.g., if the image shows a cat on a couch, predicted captions should maintain that subject-object relation.

While the above invariants are natural for their respective domains, many existing metrics do not explicitly account for these kinds of invariants. Further, existing work has observed that neural networks trained for structured prediction tasks do not necessarily produce predictions that enforce the task constraints, e.g., neural networks trained to predict shortest paths need not produce a connected path, a natural invariant for path problems (Ahmed et al. 2022). We observe the same difficulties for the multi-output structured prediction tasks described above: task invariants are not automatically preserved through neural network training (even when training with large data sets). However, as violation of these invariants can result in poor-quality labels that are readily identified as such by human evaluators, our aim is to understand to what extent natural task invariants are captured by existing metrics. We propose a framework to evaluate the evaluation metrics themselves using approximate invariants as a probe. Rather than artificially introducing invariant violations, we systematically compare outputs from standard baseline models (which can significantly violate the task-specific invariants) with invariant-regularized models explicitly trained to respect approximate invariants, e.g., segmentation consistency in image colorization, named entities in translation, etc. We consider three modeling tasks: image colorization, machine translation, and image captioning. For each task, we first define one or more approximate task invariants (see Table 1 for a list of possible invariants). We then train models that are both invariant aware, where we enforce the invariants with an auxiliary loss function, and invariant agnostic. We evaluate the different models via a battery of metrics including standard metrics and distributional metrics and observe how the performance changes when the invariant is or is not enforced. Our contributions are as follows.

• We introduce a novel evaluation framework that uses approximate task invariants as a diagnostic lens for evaluating metrics that provides an alternative to human evaluators.

• We apply our framework across vision and language domains (image colorization, image captioning, and machine translation) with distinct invariants and evaluation challenges. This comprehensive study demonstrates the versatility of our approach and provides insights into metrics ranging from pixel-level (MSE) and token-level (BLEU) to distributional (FID) and embedding-based (BERTScore).

• Our experiments show that distributional metrics, e.g.,

<table><tr><td>Main Task</td><td>Invariant Task(s)</td></tr><tr><td>Image Colorization</td><td>Image Segmentation, Depth Estimation, Edge De- tection</td></tr><tr><td>Machine Translation (NMT)</td><td>Named Entity Recognition (NER), Sentiment Analysis, Paraphrase Detection</td></tr><tr><td>Image Captioning</td><td>Semantic Role Labeling (SRL), Scene Graph Con- sistency, Named Entity Recognition (NER)</td></tr><tr><td>Text Summarization</td><td>Named Entity Recognition (NER), Sentiment Analysis, Coreference Resolution</td></tr><tr><td>Video Prediction</td><td>Optical Flow Estimation, Object Tracking, Action Recognition</td></tr><tr><td>Natural Language Inference</td><td>Paraphrase detection, Sentiment Analysis</td></tr></table>

Table 1: Tasks and corresponding invariants. BOLD tasks are considered in this work.

FID, FBD, etc., are generally more sensitive to invariantrelated differences, and enforcing invariants in outputs can lead to significant performance gains (15% − 30%) in sensitive metrics. In constrast, we find that traditional metrics, e.g., BLEU, PSNR, etc., are largely unchanged when training with the selected invariant losses.

## Related Work

While the aim in this work is to provide alternatives to human evaluators, one branch of related work focuses on metrics that better align with human judgment by incorporating semantic information or learning directly from human evaluations. METEOR (Banerjee and Lavie 2005) extended BLEU by accounting for synonyms and stemming, CIDEr (Vedantam, Lawrence Zitnick, and Parikh 2015) reweights n-grams by TF-IDF to reflect their salience, and SPICE (Anderson et al. 2016) incorporates semantic propositional knowledge. SPICE was shown to achieve a higher correlation with human caption preferences than BLEU or CIDEr (Anderson et al. 2016). Hasler and Suesstrunk (2003) introduced Colorfulness to measure vibrancy in images.

Learned metrics based on neural network embeddings have also gained traction. BERTScore (Zhang et al. 2020) leverages pretrained language model embeddings to judge text similarity, yielding significantly improved correlation with human assessments of translation quality. In vision, the Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al. 2018) metric measures image differences in a deep feature space. These learned or “feature-based” metrics effectively compute distances in a feature space extracted from trained models on large datasets, making them sensitive to high-level features.

Distributional metrics evaluate the distribution of outputs as a whole, rather than scoring individual output-reference pairs. This approach arose prominently in image generation research. For example, FID (Heusel et al. 2017) has become a popular approach for assessing the realism of images produced by GANs. For language tasks, Xiang et al. (2021) introduce Frechet Bert Distance (FBD)´

and argue that it better aligns high human judgment than BLEU, METEOR, and ROUGE-L.

Distributional metrics also have important limitations. Chong and Forsyth (2020) showed that FID is a biased estimator and proposed a corrective fix. Parmar, Zhang, and Zhu (2022) showed that compression and resizing have a significant effect on FID. Jayasumana et al. (2024) show that FID is sensitive to complex image distortions and does not behave monotonically as the distortions increase. They propose using maximum mean discrepancy (MMD) calculated over CLIP embeddings instead as MMD is unbiased, CLIP embeddings are richer than Inception features, and MMD better aligns with human evaluators in their experiments.

Problem or domain invariants have historically been used to improve generalization, e.g., data augmentation for rotation equivariance in computer vision tasks. Some neural architectures are built on invariants, e.g., convolutional layers enforce translation equivariance symmetry, and further research has aimed at hard-coding other symmetries, such as rotation equivariance, into convolutional neural networks (Cohen and Welling 2016; Worrall et al. 2017). However, to the best of our knowledge, the more complex, approximate invariants, including the types we formally define herein, have not been treated in any systematic way.

Another related line of work has explored enforcing certain desired properties through multi-task learning or auxiliary losses. In image colorization, researchers have added auxiliary segmentation or depth prediction tasks to the model so that the colorization network jointly learns to respect object boundaries or 3D structure. Similarly, in machine translation, one might regularize a model to preserve named entities or sentiment by jointly training on a related classification task. These auxiliary loss approaches, e.g., (Misra et al. 2016; Ruder 2017), can act as a form of regularization. However, the approaches in prior work require extra supervision, e.g., segmentation labels, or complex multi-task training setups. Contrast this with the approach taken here in which no additional labels are needed – only access to a pretrained DNN for a desired invariant is required. There has also been progress in explicitly modeling the multi-moda nature of outputs, e.g., using probabilistic frameworks or adversarial generative models that can produce multiple different valid outputs (Zhu et al. 2017; Zhao, Ma, and Ermon 2018). These methods address the inherent ambiguity by aiming to capture the full distribution of outputs, but they often rely on adversarial training and do not guarantee that specific semantic properties are preserved in each output.

## Multi-Output Tasks and Invariants

We consider a variant of the standard supervised structured learning task: for data observations $x ^ { ( 1 ) } , \cdot \cdot \cdot , x ^ { ( M ) } \in \mathcal { X }$ our aim is to produce corresponding labels $y ^ { ( 1 ) } , \ldots , y ^ { ( M ) }$ that are assumed to be taken from a (structured) set $\mathcal { V } .$ However, we do not assume that there is a one-to-one mapping between the data observations and the labels. Specifically, we assume that there exists a function $f : \mathcal { X } \stackrel { \cdot } {  } 2 ^ { \mathcal { Y } }$ such that for all $m \in \{ 1 , \ldots , M \} , y ^ { ( m ) } \in \overset { } { f } ( x ^ { ( m ) } )$ . Note that if f maps each element of X to a unique label, $\mathrm { i . e . , } | f ( x ) | = \bar { 1 } .$ then this is equivalent to a standard supervised learning task whose aim is to find $f ,$ but more generally, there could exist $x \in \mathcal { X }$ with $| f ( x ) | > 1$ . In the general case, our aim will be to produce one valid labeling for each given data observation (note that one could also consider fitting distributions over possible labels, but we do not consider that task here).

As an example, consider the image colorization task which seeks to transform grayscale images into colored variants. Inherently, the task is ill-posed: a single grayscale image can be mapped to multiple, plausible colorized images. In this task, images are often represented in CIELAB color space as a triple $( L , a , b )$ . To map this problem into the above framework, we can choose X to be a set of grayscale images in Lab color space, $\mathcal { V }$ to be a set of color images in Lab color space, and $f$ can be defined as the function that maps grayscale images to a set of valid/realistic colorizations. Similarly, consider the task of captioning images: We can define X as the set of all uncaptioned real-world images, $\mathcal { V }$ to be a set of captions, and $f$ can be defined as the function that takes an uncaptioned image to a set of valid captions.

In both of the above examples, there is not necessarily a unique label $y \in \mathcal { V }$ for every $x \in { \mathcal { X } } .$ , nor is it clear that there is even a notion of a “best” label $y \in \mathcal { V }$ as $\mathrm { \hbar \mathrm { \hbar \mathfrak { c s t } } ^ { \mathrm { * } } }$ in both of these instances could be subjective. This creates difficulty when using standard approaches to learning/evaluation in these settings: for a given $x \in \mathcal { X }$ , there could be $y , y ^ { \prime } \in \mathcal { D }$ with $y \ne y ^ { \prime }$ that are both “good” label choices, but they could be far apart from each other under typical notions of distance, e.g., mean squared error, on $\mathcal { N } .$

## Approximate Task Invariants

Consider the image colorization task for which a DNN is trained to produce a colored image in Lab space given a grayscale image. This DNN, when applied to an unseen test image, say of a pedestrian crossing the street, may actually produce a “good” colorization, but that colorization may be evaluated poorly under a squared error metric if the colors deviate significantly from the ground truth, e.g., the pedestrian’s shirt is colored red by the DNN in contrast to blue in the ground truth. Different colorizers may produce different colors for the pedestrian’s shirt, and unless the pedestrian’s shirt is somehow unique, even a human evaluator might not know which color is correct. However, if a particular colorizer assigns a color to the pedestrian’s shirt that then bleeds into the neighboring pixels, then a human evaluator may observe that this colorizer has made a mistake.

In the above example, the human evaluator only noticed the error because an unexpected artifact was observed. We can view this as a segmentation error - colors should not spill across object boundaries, or put another way, we expect the segmentations of the ground truth and the colorizerproduced image to (approximately) match. We call this observation that the outputs of “good” colorizers should have the same segmentations as the ground-truth images an approximate task invariant for the colorization problem. Formally, we will define invariants as follows.

Definition 0.1. Let $g : \mathcal { Y }  \mathcal { Z } . \mathrm { ~ A ~ }$ labeling problem, as defined above, is g-invariant if for all $x \in \chi$ and all $y , y ^ { \prime } \in f ( x ) , g ( y ) \overset { \cdot } { = } g ( y ^ { \prime } )$ , i.e., for all data observations x, g applied to any valid labeling of the data observation x yields the same result.

We will typically consider $\mathcal { Z }$ to be a subset of $\mathbb { R } ^ { n }$ for some $n .$ Ideally, $\bar { \mathcal Z }$ should be a space on which similarities/differences are easier to detect than on $\mathcal { V } .$ . From the definition of g-invariant, we have that if $| | g ( y ) - g ( y ^ { \prime } ) | | > 0$ , then there does not exist an x $\in \mathcal { X }$ such that $y , y ^ { \prime } \in { \hat { f } } ( x )$ , i.e., we can use $| | g ( y ) - g ( y ^ { \prime } ) | |$ as a surrogate measure for how close $y ^ { \prime }$ is to being in the set $f ( x )$ . Not all invariants, as defined above, are useful for classification: constant functions, $\mathrm { e . g . }$ $g ( y ) = 0$ , trivially satisfy this definition. Note also that the definition makes no restrictions on $y \not \in f ( x )$ . That ${ \mathrm { i s } } ,$ the invariant is a property of all “good” labels, but some labels that are not “good” can map to the same element of $\mathcal { Z }$ . As a result, these invariants should be thought of as desirable properties for “good” labelings, but they place no restrictions on invalid labels. However, the invariants that will be most useful in practice should distinguish “good” labels from “poor” labels as much as possible.

The intuition behind considering invariants of this form is that they can act like regularizers. That is, suppose we trained a DNN model for a labeling task, and on input x with ground truth label $y \in f ( x )$ it produces the label $y ^ { \prime } .$ If $y \ne y ^ { \prime } .$ , the output DNN should still be considered good if $y ^ { \prime } \in f ( x )$ or if $y ^ { \prime }$ is at least close to some member of $f ( x )$ . If the labeling task is g-invariant, then we can add an auxiliary loss to the objective to penalize model weights for which $| | g ( y ) - g ( y ^ { \prime } ) | |$ is large. This has two benefits: (1) it allows some deviation from $| | y - y ^ { \prime } | |$ in order to better match the specified invariant, and (2) even if $| | y - y ^ { \prime } | |$ is small, the penalty encourages the model to produce labels that approximately preserve the invariant, which hopefully increases the realism of the resulting predictions.

While true invariants might be difficult to obtain, we can still apply this approach with approximate invariants:

Definition 0.2. Let $g : \mathcal { Y }  \mathcal { Z } .$ A labeling problem, as defined above, is an ϵ-approximate g-invariant for some $\epsilon > 0 \mathrm { i } 1$ f for all $x \in \mathcal { X }$ and all $y , y ^ { \prime } \in \bar { f ( x ) } , | | g ( y ) - g ( y ^ { \prime } ) | | \leq$ ϵ for an appropriately chosen norm on $\mathcal { Z } , \mathrm { i . e . . }$ , for all data observations $x , g$ applied to any valid labeling of the data observation x yields almost the same result.

Consider again the problem of image colorization. A reasonable (approximate) invariant for this task is that “good” colorizations should be almost indistinguishable to a welltrained image segmentor. Similarly, good colorizations of the same gray scale images should have similar depth maps. In the invariant framework above, a DNN trained for image segmentation or depth estimation, while not a perfect invariant for the colorization task as neither is likely to be error free, can be used to define a function g such that the colorization task is approximately g-invariant. This is the approach that we take in this work.

## Invariants as Regularization

Existing work has observed that enforcing task constraints can yield noticeable performance improvements in practice (Ahmed et al. 2022). Motivated by this observation, our aim, then, is (1) to understand to what extent natural task invariants are preserved by/reflected in existing metrics and (2) to determine whether or not enforcing task invariants has any performance benefits for multi-output structured prediction tasks. To encourage models to better align with invariants, we take a common approach: we add an additional penalty to the loss function.

Assume that we are fitting a DNN, represented by $f _ { \theta } ^ { \prime } ,$ which takes as input a data observation x and attempts to produce the ground truth label $y .$ . We also assume that we are given an approximate invariant $g$ for this labeling task. With this notation, our approach implements an augmented loss function, $L ,$ given by

$$
L ( \theta | x , y ) = L _ { T } ( f _ { \theta } ^ { \prime } ( x ) , y ) + \alpha L _ { I } ( g ( f _ { \theta } ^ { \prime } ( x ) ) , g ( y ) ) ,
$$

where $L _ { T }$ is a task-specific loss measuring the error between the DNN’s prediction and the ground truth label $y , \theta$ is the vector of parameters for the DNN, $L _ { I }$ is the invariant loss for invariant $^ { g , }$ measuring the error between the invariant map applied to the ground truth label $y$ and the invariant map applied to the output of the DNN, and α is a hyperparameter controlling the trade-off between these two losses. Note that the invariant $g$ is treated as fixed, and although it will be chosen to be a DNN in our formulation, its parameters are frozen, $\mathrm { i . e . , }$ they are not learned as part of the framework. As an example, for image colorization with a segmentation invariant, we select $g$ to be a frozen semantic segmentation network and $L _ { I }$ to be the cross-entropy between the segmentation masks of the generated and ground-truth color images.

## Experimental Evaluation

The aim of training with invariants is to encourage the trained model to produce predictions that, while they may not perfectly match the ground truth, deviate from the ground truth in ways that are approximately preserved by the invariant. The question that we seek to answer in this section, then, is to what extent invariance preservation is reflected in existing metrics. In particular, we are interested in understanding (1) to what extent each metric is sensitive to differences with respect to the selected invariants and (2) to what degree enforcing invariants yields performance improvements in practice.

We illustrate our framework by applying it to the image colorization, machine translation, and image captioning tasks. All models were trained from scratch to ensure that the same training process was used for both the baseline models and the baseline+invariant models. Note that in all experimental settings the weights of the invariant networks are frozen during training. Full implementation details for the baselines can be found in the original papers, and all architectures were obtained directly from the authors of these works. All systems were trained using either one NVIDIA A100 GPU or one V100 GPU. In Table 2, we summarize the choice of baselines, invariants, data sets, losses, and metrics for each experimental setting. Full experimental details can be found in the appendix.

## Sensitivity of Metrics to Invariants

Ideally, we would want a model for the selected tasks that produces labels that align well with the ground truth, both in terms of the loss used for training and the chosen invariant loss. However, different evaluation metrics may be more or less sensitive to changes with respect the invariant loss while the training loss remains stable. This could happen for a variety of reasons. For example, consider the FID metric that has gained popularity as an evaluation metric for colorization models. The loss fits a multivariate Gaussian distribution over a feature space extracted from the Inception network. As this feature space is much lower dimensional than the input images, some information is inherently lost. For example, it could be that two colorized images with the same MSE to the ground truth, but different invariant losses, map to the same Inception features. If this occurs, the differences being captured by the invariant loss have been forgotten in the feature space used for evaluation. In this instance, we would say that the Inception features are not sensitive to the selected invariant.

![](images/afd5f5934fafb2139d8072d0b315b082488c7b3d5e19214b308234b9d0f67786.jpg)  
Figure 1: Effectiveness of invariant-based regularization in improving image colorization. Left panel: Comparison using segmentation invariant. Right panel: Comparison using depth invariant. Each panel shows the colorized outputs (top row) and corresponding invariant masks (bottom row). Invariant-regularized models significantly preserve critical structural details, such as object boundaries and spatial depth cues, compared to baseline models that exhibit noticeable artifacts and inaccuracies.

<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Image Colorization</td><td rowspan=1 colspan=1>German to English Neural Machine Translation</td><td rowspan=1 colspan=1>Image Captioning</td></tr><tr><td rowspan=1 colspan=1>Base Networks</td><td rowspan=1 colspan=1>ChromaGAN (Vitoria, Raad, and Ballester 2020),ColorFormer (Ji et al. 2022), and DDColor (Kanget al. 2023)</td><td rowspan=1 colspan=1>BiBERT (Xu, Van Durme, and Murray 2021) andBi-SumCut (Gao et al. 2022)</td><td rowspan=1 colspan=1>SmallCap (Ramos et al. 2023) and ClipCap(Mokady, Hertz, and Bermano 2021)</td></tr><tr><td rowspan=1 colspan=1>Loss   Function(LT)</td><td rowspan=1 colspan=1>Pixel-reconstruction MSE, L1-loss, Perceptual-loss, GAN-loss, Colorfulness-loss</td><td rowspan=1 colspan=1>Label Smoothed Cross Entropy: Cross-entropyloss measuring predicted vs target word probabil-ity differences with label smoothing for better gen-eralization</td><td rowspan=1 colspan=1>Cross Entropy Loss: Token-level classificationloss between predicted and ground truth captionsequences</td></tr><tr><td rowspan=1 colspan=1>Invariant   Net-works</td><td rowspan=1 colspan=1>Depth: MiDaS (Ranftl et al. 2020)Segmentation: Side Adapter Network (SAN)(Xu et al. 2023)</td><td rowspan=1 colspan=1>NER: Hugging Face&#x27;s (Wolf et al. 2019)Sentiment: Hugging Face&#x27;s distilbert-base-uncased-finetuned-sst-2-english</td><td rowspan=1 colspan=1>NER: Hugging Face&#x27;s (Wolf et al. 2019)Scenegraph: sngparser (Wu et al. 2019)(Schuster et al. 2015)</td></tr><tr><td rowspan=1 colspan=1>Invariant   Loss(L1)</td><td rowspan=1 colspan=1>SegmentationSANLoss: Dice loss between pre-dicted vs GT SAN segmentation masks preventscolor bleeding DepthMidasLoss: SILog loss be-tween predicted vs GT MiDaS depth maps pre-serves spatial structure</td><td rowspan=1 colspan=1>NER Loss: KL divergence between predicted vsGT BERT-NER token classifications for entitypreservation and Sentiment Loss: KL divergencebetween predicted vs GT DistilBERT sentimentlogits</td><td rowspan=1 colspan=1>NER Loss: KL divergence between predicted vsGT BERT-NER token classifications for entitypreservation. Scene Graph Loss: Symmetric dif-ference ratio of entities and relations between pre-dicted vs GT scene graphs</td></tr><tr><td rowspan=1 colspan=1>Metrics</td><td rowspan=1 colspan=1>Frechet Inception Distance (FID), Frechet DinoV2Distance (FDD), CLIP maximum mean discrep-ancy distance (CMMD), Learned Perceptual Im-age Patch Similarity (LPIPS), Colorfulness (CF),Peak Signal to Noise Ratio (PSNR)</td><td rowspan=1 colspan=1>FBD-4(fbd of average of last 4 hidden lay-ers), Frechet Bert Distance(FBD) (Xiang et al.2021), BERTScore (Zhang et al. 2020), BLEU-4(Papineni et al. 2002), METEOR (Banerjee andLavie 2005), COMET (Rei et al. 2020)</td><td rowspan=1 colspan=1>FBD-4(fbd of average of last 4 hidden layers),FBD (Xiang et al. 2021), BERTScore (Zhang et al.2020), CIDEr (Vedantam, Lawrence Zitnick, andParikh 2015), BLEU-4(Papineni et al. 2002), ME-TEOR (Banerjee and Lavie 2005)</td></tr><tr><td rowspan=1 colspan=1>Data Sets</td><td rowspan=1 colspan=1>ImageNet (Deng et al. 2009), COCO (Lin et al.2014), and ADE20K (Zhou et al. 2019) (Zhouet al. 2017)</td><td rowspan=1 colspan=1>IWSLT 2014 (Cettolo, Girardi, and Federico2012) and WMT-14 (Luong and Manning 2015)</td><td rowspan=1 colspan=1>COCO (Lin et al. 2014) and flickr30k (Plummeret al. 2015)</td></tr></table>

Table 2: Experimental configurations for the three selected tasks.

As our primary supposition in this work is that invariants are properties that should be approximately preserved by “good” labels, it is critical that the evaluation metrics be sensitive to the selected invariant(s). Our first set of experiments is designed to assess the sensitivity of a variety of evaluation metrics to a selected set of invariants for each of our three chosen tasks. To this end, we trained a variety of architectures both with and without an invariant loss and evaluated them using a variety of metrics on validation data.

(a) Image Colorization [ImageNet]  
![](images/aa186b1eb0ddcf69498572b37a416b7de7553c2f6145086b140ec6dcd8a56ba1.jpg)

(b) Machine Translation [IWSLT14]  
![](images/fe20ef77e814ba56240f597f85fc84f852ef4aa91e52b11030f1a079d3d3f194.jpg)

(c) Image Captioning [COCO-Stuff]  
![](images/c972da5eaf39fc2bfd17167d1dbf051ecc4728c8ecde1a0aec68ec35553deb69.jpg)  
Figure 2: Performance improvement of architectures trained with invariants versus training without invariants, evaluated across common metrics for three tasks. (a) Image Colorization: Base models are ChromaGAN (CG), ColorFormer (CF), and DDColor (DD), each augmented with Segmentation (+S), Depth (+D), or both (+SD). (b) Machine Translation: Base models are Bi-SimCut (BS) and BiBERT (BB), augmented with NER (+N), SeAM (+Se), or both (+NS). (c) Image Captioning: Base models are SmallCap (SC) and ClipCap (CC), augmented with NER (+N), SceneGraph (+SG), or both (+NS). Distributional metrics (FDD, FBD, FID, CMMD) show larger improvements than traditional metrics (PSNR, BLEU-4, METEOR), indicating higher sensitivity to invariant-based training.

Figure 2 illustrates the best possible performance improvement (compared against the baseline with no invariant loss during training) for each invariant/metric pair obtained via hyperparameter search on the indicated validation data sets for our chosen tasks. Note that only α, the hyperparameter for the invariant loss was tuned here, all other settings for the architectures remained untouched from the baseline. For the image captioning task, even a small invariant penalty during training results in ±1% performance improvement / degradation for METEOR, BLEU-4, and CIDEr. FBD and BertScore show essentially no differences with respect to training with/without an invariant penalty. This suggests that these metrics are not very sensitive to the selected invariants for this task. However, FBD-4, constructed from the last 4 layers of the BERT network, shows 5% − 13% improvement across models/invariants. For machine translation, METEOR, BLEU-4, CIDEr, and BERTScore all show a < 1% improvement while FBD-4 improves 2% − 5% and FBD improves 13% − 15%. A similar trend can be observed for the colorization task: PSNR, a traditional metric, shows a < 2% improvement, LPIPS shows moderate 4% − 7% improvement, FID shows 7% − 16% improvement, and FDD shows a more dramatic 3%−33% improvement. In all cases, metrics derived from richer feature embeddings tend to show increased sensitivity to training with invariants.

From these observations, we can conclude that, while training with an invariant penalty does clearly result in different predictions, many of the standard error metrics for evaluating task performance are almost insensitive to these changes compared with distributional metrics and metrics derived from richer feature spaces. Put another way, if we had valdiated a model using the insensitive metrics, we essentially would not be able to tell whether or not the selected model respects the invariant. This suggests that the natural invariants considered here are not being automatically enforced through the standard train/validate pipeline.

## The Effect of Invariants on Task Performance

Above, we observed that distributional metrics seem to be more sensitive to training with invariants than their traditional counterparts when we consider the best performance that can be obtained on validation data. This sensitivity is important if we care about preserving the invariants in practice, but high sensitivity alone does not guarantee that a metric is necessarily “good” for evaluating task performance. In particular, models that perform the best along one metric could perform poorly on all other metrics. In this case, high performance was achieved along one metric at the expense of the other metrics. In this section, we explore whether or not training with an invariant penalty yields performance improvements more broadly. That is, does incorporating task invariants as above and validating with an invariant-sensitive metric yield practical performance gains?

<table><tr><td rowspan="2">Method</td><td rowspan="2">#P</td><td colspan="6">ImageNet (val50k)</td><td colspan="6">COCO-Stuff(val5k)</td></tr><tr><td>|FDD↓ FID↓</td><td></td><td>CMMD↓</td><td>LPIPS↓</td><td>CF↑</td><td>PSNR↑</td><td>FDD↓</td><td>FID↓</td><td>CMMD↓</td><td>LPIPS↓</td><td>CF↑</td><td>PSNR↑</td></tr><tr><td>ChromaGAN</td><td>173.9M</td><td>42.36</td><td>5.81</td><td>0.597</td><td>0.236</td><td>27.41</td><td>23.65</td><td>94.90</td><td>16.73</td><td>0.715</td><td>0.233</td><td>27.89</td><td>23.66</td></tr><tr><td>+Seg</td><td>173.9M</td><td>29.38</td><td>5.44</td><td>0.679</td><td>0.236</td><td>29.36</td><td>23.76</td><td>81.07</td><td>15.50</td><td>0.603</td><td>0.233</td><td>29.56</td><td>23.82</td></tr><tr><td>+Depth</td><td>173.9M</td><td>28.55</td><td>5.27</td><td>0.565</td><td>0.231</td><td>28.73</td><td>23.83</td><td>79.28</td><td>15.29</td><td>0.659</td><td>0.229</td><td>28.95</td><td>23.91</td></tr><tr><td>+Seg+Depth</td><td>173.9M</td><td>28.86</td><td>5.06</td><td>0.599</td><td>0.231</td><td>28.69</td><td>23.83</td><td>80.00</td><td>14.58</td><td>0.597</td><td>0.229</td><td>29.99</td><td>23.82</td></tr><tr><td>ColorFormer</td><td>44.8M</td><td>21.16</td><td>2.00</td><td>0.318</td><td>0.137</td><td>27.57</td><td>24.91</td><td>59.38</td><td>9.03</td><td>0.377</td><td>0.134</td><td>28.30</td><td>24.70</td></tr><tr><td>+Seg</td><td>44.8M</td><td>15.43</td><td>1.71</td><td>0.273</td><td>0.130</td><td>28.29</td><td>24.92</td><td>52.52</td><td>8.49</td><td>0.370</td><td>0.134</td><td>29.21</td><td>24.67</td></tr><tr><td>+Depth</td><td>44.8M</td><td>17.45</td><td>1.82</td><td>0.302</td><td>0.129</td><td>28.16</td><td>24.95</td><td>55.31</td><td>8.80</td><td>0.385</td><td>0.133</td><td>28.83</td><td>24.73</td></tr><tr><td>+Seg+Depth</td><td>44.8M</td><td>15.66</td><td>1.69</td><td>0.263</td><td>0.131</td><td>28.52</td><td>24.85</td><td>52.74</td><td>8.45</td><td>0.376</td><td>0.135</td><td>29.39</td><td>24.61</td></tr><tr><td>DDColor</td><td>55M</td><td>29.60</td><td>4.68</td><td>0.436</td><td>0.179</td><td>52.84</td><td>21.22</td><td>71.58</td><td>12.08</td><td>0.484</td><td>0.177</td><td>50.88</td><td>21.43</td></tr><tr><td>+Seg</td><td>55M</td><td>28.65</td><td>4.31</td><td>0.423</td><td>0.174</td><td>51.14</td><td>21.57</td><td>69.70</td><td>11.48</td><td>0.465</td><td>0.172</td><td>50.10</td><td>21.73</td></tr><tr><td>+Depth</td><td>55M</td><td>26.95</td><td>4.20</td><td>0.418</td><td>0.174</td><td>51.63</td><td>21.54</td><td>69.69</td><td>11.22</td><td>0.462</td><td>0.172</td><td>50.27</td><td>21.73</td></tr><tr><td>+Seg+Depth</td><td>55M</td><td>26.85</td><td>4.17</td><td>0.419</td><td>0.175</td><td>52.12</td><td>21.40</td><td>69.35</td><td>11.20</td><td>0.465</td><td>0.172</td><td>50.67</td><td>21.63</td></tr></table>

Table 3: Quantitative comparison of image colorization methods. ↑ (↓) indicates higher (lower) values are better; Bold values in FDD column of ImageNet indicate the model selection criterion. The Seg architecture is SAN, and depth architecture is Midas.
<table><tr><td rowspan="2">Model</td><td rowspan="2">#P</td><td colspan="6">COCO-Stuff (val5k)</td><td colspan="6">Flickr30k (val1k)</td></tr><tr><td>|FBD-4↓ FBD↓</td><td>BERTScore↑</td><td></td><td>CIDEr↑</td><td>BLEU-4↑</td><td>METEOR↑</td><td>FBD-4↓</td><td>FBD↓ BERTScore↑</td><td></td><td>CIDEr↑</td><td>BLEU-4↑</td><td>METEOR↑</td></tr><tr><td>SmallCap</td><td>7M</td><td>62.60</td><td>41.90</td><td>0.9224</td><td>118.49</td><td>36.68</td><td>27.90</td><td>58.45</td><td>39.12</td><td>0.9189</td><td>112.30</td><td>34.82</td><td>26.45</td></tr><tr><td>+NER</td><td>7M</td><td>58.82</td><td>41.89</td><td>0.9223</td><td>116.29</td><td>36.07</td><td>27.91</td><td>55.20</td><td>39.08</td><td>0.9187</td><td>110.15</td><td>34.25</td><td>26.48</td></tr><tr><td>+SceneGraph</td><td>7M</td><td>54.12</td><td>41.90</td><td>0.9218</td><td>115.47</td><td>35.68</td><td>27.76</td><td>51.85</td><td>39.15</td><td>0.9182</td><td>109.42</td><td>33.89</td><td>26.32</td></tr><tr><td>+NER+SceneGraph</td><td>7M</td><td>56.52</td><td>41.89</td><td>0.9226</td><td>117.48</td><td>36.52</td><td>27.95</td><td>53.78</td><td>39.11</td><td>0.9191</td><td>111.68</td><td>34.71</td><td>26.51</td></tr><tr><td>ClipCap</td><td>43M</td><td>48.20</td><td>38.45</td><td>0.9301</td><td>125.80</td><td>39.15</td><td>29.20</td><td>44.60</td><td>36.25</td><td>0.9267</td><td>119.45</td><td>37.28</td><td>27.85</td></tr><tr><td>+NER</td><td>43M</td><td>45.60</td><td>38.42</td><td>0.9299</td><td>123.45</td><td>38.52</td><td>29.22</td><td>42.15</td><td>36.21</td><td>0.9265</td><td>117.30</td><td>36.65</td><td>27.88</td></tr><tr><td>+SceneGraph</td><td>43M</td><td>42.85</td><td>38.44</td><td>0.9295</td><td>122.30</td><td>38.01</td><td>29.05</td><td>39.85</td><td>36.28</td><td>0.9260</td><td>116.52</td><td>36.14</td><td>27.72</td></tr><tr><td>+NER+SceneGraph 43M</td><td></td><td>44.70</td><td>38.41</td><td>0.9303</td><td>124.90</td><td>38.98</td><td>29.25</td><td>41.25</td><td>36.24</td><td>0.9269</td><td>118.85</td><td>37.12</td><td>27.91</td></tr></table>

Table 4: Performance on image captioning benchmarks. Bold values in FBD-4 column of COCO-Stuff indicate the model selection criterion.

![](images/f007dce10e7a1955c8b1cd0d4cdd3a5c24cfa8668737ffab1ba5cdf66f10440c.jpg)

<table><tr><td colspan="3">Ground Truth</td><td colspan="2">SmallCap+NER+SceneGraph</td><td colspan="3">SmallCap</td></tr><tr><td colspan="3">Caption: A cat standing on a couch in a cluttered living room.</td><td colspan="3">Caption: A black cat standing on a couch in a living room.</td><td colspan="3">Caption: A black cat standing on top of a bed.</td></tr><tr><td colspan="3">Entities: Head Span</td><td colspan="3">Entities: Modifiers Head Span Modifiers</td><td colspan="3">Entities: Head Span Modifiers a, black</td></tr><tr><td>cat couch living room</td><td colspan="2">a cat a a couch a a cluttered a, clut- living room tered</td><td>cat couch living room</td><td>a black cat a couch a living a</td><td>a, black cat a bed</td><td colspan="3">a black cat a bed a</td></tr><tr><td colspan="3">Relations: Subject Relation</td><td colspan="3">room Relations:</td><td colspan="3">Relations:</td></tr><tr><td colspan="3">cat on in</td><td colspan="3">Subject Relation Object</td><td colspan="3">Subject Relation Object</td></tr><tr><td colspan="3">cat</td><td colspan="3">cat on couch cat in living</td><td colspan="3">cat on bed</td></tr></table>

Figure 3: Image captions produced by models trained with/without scenegraph and NER invariants for an image of a cat (left).

For each of our tasks, we trained a variety of architectures with different invariant losses and report the resulting performance across common task metrics. The best model for each architecture/invariant combination was selected using the most sensitive metric for each task from the previous section, i.e., FDD for colorization, FBD-4 for image captioning, and FBD for machine translation, on a validation data set. The results of our investigation can be found in Tables 3 and 4. As was noted above, the best performing models each perform better with respect to the metric used for model selection. However, the more detailed results in the tables illustrate that the performance improvements are not accompanied by significant degredation in the other metrics. For example, in Table 3, Colorformer+Seg achieves the best FDD score across all architectures and yields performance improvements on nearly all metrics across the two validation data sets. Contrast this with the results in Table 4 in which dramatic improvements in FBD-4 are met with minor ±2% differences in nearly all other metrics. A similar observation can be made for FBD in machine translation.

<table><tr><td rowspan="2">Model</td><td rowspan="2">#P</td><td colspan="6">IWSLT-14</td><td colspan="6">WMT-14</td></tr><tr><td>|FBD↓ FBD-4↓</td><td>BERTScore↑</td><td></td><td>BLEU-4↑</td><td>METEOR↑</td><td>COMET↑</td><td>FBD↓ FBD-4↓</td><td>BERTScore↑</td><td></td><td>BLEU-4↑</td><td>METEOR↑</td><td>COMET↑</td></tr><tr><td>Bi-SimCut</td><td>65M</td><td>9.87</td><td>102.15</td><td>0.8871</td><td>37.83</td><td>0.6412</td><td>0.5743</td><td>15.24</td><td>135.42</td><td>0.8952</td><td>34.47</td><td>0.5836</td><td>0.6124</td></tr><tr><td>+NER</td><td>65M</td><td>8.42</td><td>101.89</td><td>0.8867</td><td>36.95</td><td>0.6358</td><td>0.5739</td><td>14.18</td><td>134.76</td><td>0.8948</td><td>33.82</td><td>0.5794</td><td>0.6119</td></tr><tr><td>+SeAM</td><td>65M</td><td>8.51</td><td>103.14</td><td>0.8868</td><td>37.02</td><td>0.6361</td><td>0.5737</td><td>14.26</td><td>136.23</td><td>0.8949</td><td>33.89</td><td>0.5798</td><td>0.6117</td></tr><tr><td>+NER+SeAM</td><td>65M</td><td>8.38</td><td>102.67</td><td>0.8869</td><td>37.18</td><td>0.6373</td><td>0.5741</td><td>14.09</td><td>135.58</td><td>0.8950</td><td>34.06</td><td>0.5809</td><td>0.6121</td></tr><tr><td>BiBERT</td><td>110M</td><td>8.99</td><td>100.24</td><td>0.8930</td><td>38.20</td><td>0.6597</td><td>0.5899</td><td>13.76</td><td>133.18</td><td>0.9012</td><td>35.24</td><td>0.6023</td><td>0.6287</td></tr><tr><td>+NER</td><td>110M</td><td>7.71</td><td>100.03</td><td>0.8925</td><td>37.35</td><td>0.6538</td><td>0.5894</td><td>12.84</td><td>132.91</td><td>0.9008</td><td>34.52</td><td>0.5976</td><td>0.6282</td></tr><tr><td>+SeAM</td><td>110M</td><td>7.82</td><td>101.27</td><td>0.8926</td><td>37.41</td><td>0.6536</td><td>0.5890</td><td>12.93</td><td>134.17</td><td>0.9009</td><td>34.58</td><td>0.5974</td><td>0.6279</td></tr><tr><td>+NER+SeAM 110M</td><td></td><td>7.68</td><td>100.80</td><td>0.8927</td><td>37.58</td><td>0.6548</td><td>0.5897</td><td>12.76</td><td>133.75</td><td>0.9010</td><td>34.74</td><td>0.5987</td><td>0.6284</td></tr></table>

Table 5: Performance on machine translation. Bold values indicate the model selection criterion.

The tables also illustrate the effect of using more than one invariant penalty during training. Recall that, as the invariant networks are frozen during training, the only additional training overhead comes from evaluating these networks. In colorization, ColorFormer+Seg+Depth acheives a lower FDD than ColorFormer+Depth, but is slightly worse than ColorFormer+Seg. A similar observation can be made for ClipClap+NER+Scenegraph for image captioning. In machine translation, BiBERT+NER+SeAM achieves the best overall FBD score. Overall, while adding two potentially conflicting losses can lead to optimization challenges, the results still remain competitive while encouraging preservation of more than one invariant.

Qualitatively, we have found that training with invariants can help correct errors that violate the invariants. In Figure 1, we provide an example of colorization outputs when trained with/without a depth or a segmentation invariant. In both cases we observe that better aligned outputs from the invariant networks correlate with better colorizations. In Figure 3, an example of image captioning with an NER+SceneGraph invariant shows that a better caption is produced and the SceneGraph generated from the ground truth image better matches the SceneGraph from the NER+SceneGraph invariant model (the invaraint-trained model correctly identifies the couch).

Additional qualitative results for all of the tasks can be found in the appendix.

## Conclusion

In summary, our experimental results illustrate that (1) traditional metrics, e.g., PSNR, BLEU-4, etc., show minimal sensitivity to invariants, (2) distributional metrics, e.g., FID, FDD, and FBD variants, show substantial improvements when invariants are enforced without significant loss with respect to traditional metrics, and (3) richer feature representations, e.g., DinoV2 in FDD, show even greater sensitivity to invariants than less rich representations. Critically, our metric evaluation framework does not depend on human evaluators, which makes this a flexible approach that extends beyond the tasks considered here. Furthermore, our results suggest that future metric design should favor richer feature encoders, e.g., DinoV2 rather than Inception for images, and consider incorporating explicit invariant-violation penalties.

## Limitations and Future Work

Our framework relies on approximate task invariants derived from pretrained models, which may introduce biases or fail in low-resource domains where such models are unavailable. Moreover, we do not directly validate the hypothesis that enforcing approximate invariants leads to improved human evaluation; instead, we position invariants as a scalable diagnostic when human studies are impractical. Finally, our empirical study focuses on three domains (image colorization, machine translation, and image captioning). Extending invariants-as-probes to other multi-output structured tasks and integrating them directly into metric design are interesting directions for future work.

## Acknowledgements

This work was supported in part by NSF grant 2327245.

## References

Ahmed, K.; Teso, S.; Chang, K.-W.; Van den Broeck, G.; and Vergari, A. 2022. Semantic Probabilistic Layers for Neuro-symbolic Learning. In Proceedings of the 36th International Conference on Neural Information Processing Systems (NeurIPS). Red Hook, NY, USA: Curran Associates Inc.

Anderson, P.; Fernando, B.; Johnson, M.; and Gould, S. 2016. SPICE: Semantic Propositional Image Caption Evaluation. In Proceedings of the 14th European Conference on Computer Vision (ECCV), 382–398. Berlin, Heidelberg: Springer-Verlag.

Banerjee, S.; and Lavie, A. 2005. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. In Proceedings of the ACL workshop on intrinsic and extrinsic evaluation measures for ma-

chine translation and/or summarization, 65–72. Kerrville, TX, USA: Association for Computational Linguistics.

Cettolo, M.; Girardi, C.; and Federico, M. 2012. Wit3: Web Inventory of Transcribed and Translated Talks. In Proceedings of the Conference of European Association for Machine Translation (EAMT), 261–268. Switzerland: European Association for Machine Translation.

Chong, M. J.; and Forsyth, D. 2020. Effectively Unbiased FID and Inception Score and Where to Find Them. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 6069–6078. Los Alamitos, CA, USA: IEEE Computer Society.

Cohen, T.; and Welling, M. 2016. Group Equivariant Convolutional Networks. In Proceedings of the 33rd International Conference on International Conference on Machine Learning (ICML), 2990–2999. JMLR.org.

Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and Fei-Fei, L. 2009. Imagenet: A Large-scale Hierarchical Image Database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 248–255. Los Alamitos, CA, USA: IEEE Computer Society.

Gao, P.; He, Z.; Wu, H.; and Wang, H. 2022. Bi-SimCut: A Simple Strategy for Boosting Neural Machine Translation. In Carpuat, M.; de Marneffe, M.-C.; and Meza Ruiz, I. V., eds., Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics (NAACL): Human Language Technologies, 3938–3948. Kerrville, TX, USA: Association for Computational Linguistics.

Hasler, D.; and Suesstrunk, S. E. 2003. Measuring Colorfulness in Natural Images. In Human Vision and Electronic Imaging (HVEI) VIII, volume 5007, 87–95. International Society for Optics and Photonics, Bellingham, WA, USA: SPIE.

Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; and Hochreiter, S. 2017. GANs Trained by a Two Time-scale Update Rule Converge to a Local Nash Equilibrium. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NeurIPS), 6629–6640. Red Hook, NY, USA: Curran Associates Inc.

Jayasumana, S.; Ramalingam, S.; Veit, A.; Glasner, D.; Chakrabarti, A.; and Kumar, S. 2024. Rethinking FID: Towards a Better Evaluation Metric for Image Generation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9307–9315. Los Alamitos, CA, USA: IEEE Computer Society.

Ji, X.; Jiang, B.; Luo, D.; Tao, G.; Chu, W.; Xie, Z.; Wang, C.; and Tai, Y. 2022. ColorFormer: Image Colorization via Color Memory Assisted Hybrid-attention Transformer. In Proceedings of the 17th European Conference on Computer Vision (ECCV), 20–36. Berlin, Heidelberg: Springer-Verlag.

Kang, X.; Yang, T.; Ouyang, W.; Ren, P.; Li, L.; and Xie, X. 2023. DDcolor: Towards photo-realistic image colorization via dual decoders. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 328–338. Los Alamitos, CA, USA: IEEE Computer Society.

Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Dollar, P.; and Zitnick, C. L. 2014. Microsoft´ COCO: Common Objects in Context. In Proceedings of the 13th European Conference on Computer Vision (ECCV), 740–755. Berlin, Heidelberg: Springer-Verlag.

Luong, M.-T.; and Manning, C. D. 2015. Stanford Neural Machine Translation Systems for Spoken Language Domains. In Proceedings of the 12th International Workshop on Spoken Language Translation: Evaluation Campaign, 76–79. Kerrville,TX, USA: Association for Computational Linguistics.

Misra, I.; Shrivastava, A.; Gupta, A.; and Hebert, M. 2016. Cross-Stitch Networks for Multi-Task Learning. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (ICCV), 3994–4003. Los Alamitos, CA, USA: IEEE Computer Society.

Mokady, R.; Hertz, A.; and Bermano, A. H. 2021. Clipcap: Clip Prefix for Image Captioning. arXiv:2111.09734.

Papineni, K.; Roukos, S.; Ward, T.; and Zhu, W.-J. 2002. BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Associationfor Computational Linguistics (ACL), 311–318. Kerrville, TX, USA: Association for Computational Linguistics.

Parmar, G.; Zhang, R.; and Zhu, J.-Y. 2022. On Aliased Resizing and Surprising Subtleties in GAN Evaluation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Los Alamitos, CA, USA: IEEE Computer Society.

Plummer, B. A.; Wang, L.; Cervantes, C. M.; Caicedo, J. C.; Hockenmaier, J.; and Lazebnik, S. 2015. Flickr30k entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models. In 2015 IEEE/CVF International Conference on Computer Vision (ICCV), 2641– 2649. Los Alamitos, CA, USA: IEEE Computer Society.

Ramos, R.; Martins, B.; Elliott, D.; and Kementchedjhieva, Y. 2023. Smallcap: Lightweight Image Captioning Prompted with Retrieval Augmentation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2840–2849. Los Alamitos, CA, USA: IEEE Computer Society.

Ranftl, R.; Lasinger, K.; Hafner, D.; Schindler, K.; and Koltun, V. 2020. Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer. IEEE Transactions on Pattern Analysis and Machine Intelli gence (TPAMI).

Rei, R.; Stewart, C.; Farinha, A. C.; and Lavie, A. 2020. COMET: A Neural Framework for MT Evaluation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2685–2702. Kerrville, TX, USA: Association for Computational Linguistics.

Ruder, S. 2017. An Overview of Multi-task Learning in Deep Neural Networks. arXiv:1706.05098.

Schuster, S.; Krishna, R.; Chang, A.; Fei-Fei, L.; and Manning, C. D. 2015. Generating Semantically Precise Scene Graphs from Textual Descriptions for Improved Image Retrieval. In Proceedings of the Fourth Workshop on Vision

and Language, 70–80. Kerrville, TX, USA: Association for Computational Linguistics.

Vedantam, R.; Lawrence Zitnick, C.; and Parikh, D. 2015. CIDEr: Consensus-based Image Description Evaluation. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4566–4575. Los Alamitos, CA, USA: IEEE Computer Society.

Vitoria, P.; Raad, L.; and Ballester, C. 2020. ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution. In The IEEE Winter Conference on Applications of Computer Vision (WACV), 2445–2454. Los Alamitos, CA, USA: IEEE Computer Society.

Wolf, T.; Debut, L.; Sanh, V.; Chaumond, J.; Delangue, C.; Moi, A.; Cistac, P.; Rault, T.; Louf, R.; Funtowicz, M.; et al. 2019. Huggingface’s Transformers: State-of-the-Art Natural Language Processing. arXiv:1910.03771.

Worrall, D. E.; Garbin, S. J.; Turmukhambetov, D.; and Brostow, G. J. 2017. Harmonic Networks: Deep Translation and Rotation Equivariance. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 5028– 5037. Los Alamitos, CA, USA: IEEE Computer Society.

Wu, H.; Mao, J.; Zhang, Y.; Jiang, Y.; Li, L.; Sun, W.; and Ma, W.-Y. 2019. Unified Visual-semantic Embeddings: Bridging Vision and Language with Structured Meaning Representations. In 2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 6609–6618. Los Alamitos, CA, USA: IEEE Computer Society.

Xiang, J.; Liu, Y.; Cai, D.; Li, H.; Lian, D.; and Liu, L. 2021. Assessing Dialogue Systems with Distribution Distances. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, 2192–2198. Kerrville, TX, USA: Association for Computational Linguistics.

Xu, H.; Van Durme, B.; and Murray, K. 2021. BERT, mBERT, or BiBERT? A Study on Contextualized Embeddings for Neural Machine Translation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP). Kerrville, TX, USA: Association for Computational Linguistics.

Xu, M.; Zhang, Z.; Wei, F.; Hu, H.; and Bai, X. 2023. Side Adapter Network for Open-vocabulary Semantic Segmentation. IEEE Transactions on Pattern Analysis & Machine Intelligence (TPAMI), 15546–15561.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 586–595. Los Alamitos, CA, USA: IEEE Computer Society.

Zhang, T.; Kishore, V.; Wu, F.; Weinberger, K. Q.; and Artzi, Y. 2020. BERTScore: Evaluating text generation with BERT. In International Conference on Learning Representations (ICLR). Appleton, WI, USA: International Conference on Learning Representations.

Zhao, S.; Ma, H.; and Ermon, S. 2018. Bias and Generalization in Deep Generative Models: An Empirical Study. In Proceedings ofthe 32nd International Conference on Neural Information Processing Systems, 10815–10824. Red Hook, NY, USA: Curran Associates Inc.

Zhou, B.; Zhao, H.; Puig, X.; Fidler, S.; Barriuso, A.; and Torralba, A. 2017. Scene Parsing through ADE20K Dataset. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 633–641. Los Alamitos, CA, USA: IEEE Computer Society.

Zhou, B.; Zhao, H.; Puig, X.; Xiao, T.; Fidler, S.; Barriuso, A.; and Torralba, A. 2019. Semantic Understanding of Scenes Through the ADE20k Dataset. International Journal ofComputer Vision, 127(3): 302–321.

Zhu, J.-Y.; Zhang, R.; Pathak, D.; Darrell, T.; Efros, A. A.; Wang, O.; and Shechtman, E. 2017. Toward Multimodal Image-to-image Translation. In Proceedings of the 31st International Conference on Neural Information Processing Systems, 465–476. Red Hook, NY, USA: Curran Associates Inc.

## Appendix

This appendix accompanies the paper published in the AAAI-26 proceedings and is not part of the proceedings version.

## Qualitative Results

## Image Colorization

![](images/441ea77f08d0f6c2aa97e01180c0ded28bb3a7dbf881757969db9e4879a15534.jpg)  
Figure 4: Effectiveness of invariant-based regularization in improving image colorization. Left panel: Comparison using segmentation invariant. Right panel: Comparison using depth invariant. Each panel shows the colorized outputs (top row) and corresponding invariant masks (bottom row). Invariant-regularized models significantly preserve critical structural details, such as object boundaries and spatial depth cues, compared to baseline models that exhibit noticeable artifacts and inaccuracies.

![](images/6c33253b97544c8f4f4c6c6b7686c62972d44f7059f0efc5069802e125f6df9c.jpg)  
Figure 5: Effectiveness of invariant-based regularization in improving image colorization. Left panel: Comparison using segmentation invariant. Right panel: Comparison using depth invariant. Each panel shows the colorized outputs (top row) and corresponding invariant masks (bottom row). Invariant-regularized models significantly preserve critical structural details, such as object boundaries and spatial depth cues, compared to baseline models that exhibit noticeable artifacts and inaccuracies.

Machine Translation
<table><tr><td rowspan=10 colspan=10>Example 1Source Sentence:              &quot;heinrich , auf der anderen seite , wird zum geburtstermin geboren , er ist also am 1 marz 1980 geboren ,&quot;&quot;henry , by contrast , is born at term , so he&#x27;s born on march 1 , 1980 .&quot;0.99992           0.00000 0.00003 0.00001 0.00001 0.00001 0.00001B-LOC&quot;henry , on the other hand , is born at the time of birth , so he was born on march 1 , 1980 .&quot;0.999930     B-MIS   I-MIS   B-PER  I-PER  B-ORG  I-ORG  B-LOC  I-LOC&quot;and tom , on the other hand , is born at the time of birth , so he was born on march 1 , 1980 .&quot;0.999960    B-MIS  I-MIS  B-PER  I-PER                      B-LOC  I-LOC</td></tr><tr><td rowspan=9 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.99992</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00000</td><td rowspan=1 colspan=1>0.00003</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=1>"henry , or</td><td rowspan=1 colspan=1>the other </td><td rowspan=1 colspan=1>and , is bo</td><td rowspan=1 colspan=1>n at the tin</td><td rowspan=1 colspan=1>e of birth ,</td><td rowspan=1 colspan=1>so he was</td><td rowspan=1 colspan=1>born on ma</td><td rowspan=1 colspan=1>rch 1 , 198</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.99993</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00000</td><td rowspan=1 colspan=1>0.00002</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=1>“and tom ,</td><td rowspan=1 colspan=1>on the othe</td><td rowspan=1 colspan=1>r hand , is</td><td rowspan=1 colspan=1>orn at the</td><td rowspan=1 colspan=1>ime of birt</td><td rowspan=1 colspan=1>1 , so he w</td><td rowspan=1 colspan=1>s born on 1</td><td rowspan=1 colspan=1>narch 1 , 1</td><td rowspan=1 colspan=1>80."</td></tr><tr><td rowspan=1 colspan=1>0.99996</td><td rowspan=1 colspan=1>0.00000</td><td rowspan=1 colspan=1>0.00000</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00000</td><td rowspan=1 colspan=1>0.00000</td><td rowspan=1 colspan=1>0.00000</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00000</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=10 colspan=9>Example 2&quot;da gab&#x27;s die kernforschungsanlage leopoldshafen .&quot;“there was the leopoldshafen nuclear research plant .&quot;0.99512                                         0.00366           0.01567B-MIS  I-MIS  B-PER  I-PER  B-ORG  I-ORG  B-LOC  I-LOC&quot;there was the nuclear research plant in albertz .&quot;0.99929                                                              0.00015B-ORG  I-ORG  B-LOC  I-LOC&quot;there was the nuclear research facility .&quot;0.99994 0.00001  0.00000 0.00001  0.00000 0.00001  0.00001  0.00001 0.00001B-MIS  I-MIS  B-PER  I-PER            I-ORG  B-LOC  I-LOC</td></tr><tr><td rowspan=9 colspan=1>Source Sentence:Target Sentence:Target Logits:BiBERT+NER Pred Sentence:BiBERT+NER Logits:BiBERT Pred Sentence:BiBERT Logits:</td></tr><tr><td rowspan=1 colspan=1>0.99512</td><td rowspan=1 colspan=1>0.00295</td><td rowspan=1 colspan=1>0.00220</td><td rowspan=1 colspan=1>0.00265</td><td rowspan=1 colspan=1>0.00175</td><td rowspan=1 colspan=1>0.00366</td><td rowspan=1 colspan=1>0.00625</td><td rowspan=1 colspan=1>0.01567</td><td rowspan=1 colspan=1>0.01369</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=1>there was</td><td rowspan=1 colspan=1>the nuclear</td><td rowspan=1 colspan=1>research p</td><td rowspan=1 colspan=1>ant in albe</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.99929</td><td rowspan=1 colspan=1>0.00009</td><td rowspan=1 colspan=1>0.00004</td><td rowspan=1 colspan=1>0.00006</td><td rowspan=1 colspan=1>0.00004</td><td rowspan=1 colspan=1>0.00008</td><td rowspan=1 colspan=1>0.00010</td><td rowspan=1 colspan=1>0.00015</td><td rowspan=1 colspan=1>0.00015</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=1>there was</td><td rowspan=1 colspan=1>the nuclear</td><td rowspan=1 colspan=1>research f</td><td rowspan=1 colspan=1>cility ."</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.99994</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00000</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00000</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td><td rowspan=1 colspan=1>0.00001</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=10>Example 3</td></tr><tr><td rowspan=9 colspan=1>Source Sentence:Target Sentence:Target Logits:BiBERT+NER Pred Sentence:BiBERT+NER Logits:BiBERT Pred Sentence:BiBERT Logits:</td><td rowspan=9 colspan=9>&quot;vor vier jahren starteten ray kurzweil und ich hier bei ted eine neue universitat , die “singularity university&quot; heißt .&quot;“four years ago here at ted , ray kurzweil and i started a new university called singularity university .&quot;0.99928 0.00007 0.00003 0.00008 0.00004 0.00007 0.00009 0.00017 0.00017I-MIS  B-PER                      I-ORG  B-LOC  I-LOC&quot;four years ago , ray kurzweil and i started a new university here at ted called singularity university .&quot;0.99948                                         0.000060    B-MIS  I-MIS  B-PER  I-PER  B-ORG  I-ORG  B-LOC  I-LOC&quot;four years ago , ray kurzweil and i , here at ted , started a new university called singular university .&quot;0.99957 0.00005           0.00009  0.00005            0.00007 0.000070                                                                           I-LOC</td></tr><tr><td rowspan=1 colspan=1>0.99928</td><td rowspan=1 colspan=1>0.00007</td><td rowspan=1 colspan=1>0.00003</td><td rowspan=1 colspan=1>0.00008</td><td rowspan=1 colspan=1>0.00004</td><td rowspan=1 colspan=1>0.00007</td><td rowspan=1 colspan=1>0.00009</td><td rowspan=1 colspan=1>0.00017</td><td rowspan=1 colspan=1>0.00017</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=1>“four years</td><td rowspan=1 colspan=1>ago , ray k</td><td rowspan=1 colspan=1>urzweil an</td><td rowspan=1 colspan=1>l i started a</td><td rowspan=1 colspan=1>new unive</td><td rowspan=1 colspan=1>sity here a</td><td rowspan=1 colspan=1>ted called</td><td rowspan=1 colspan=1>singularity</td><td rowspan=1 colspan=1>university</td></tr><tr><td rowspan=1 colspan=1>0.99948</td><td rowspan=1 colspan=1>0.00006</td><td rowspan=1 colspan=1>0.00003</td><td rowspan=1 colspan=1>0.00008</td><td rowspan=1 colspan=1>0.00004</td><td rowspan=1 colspan=1>0.00006</td><td rowspan=1 colspan=1>0.00010</td><td rowspan=1 colspan=1>0.00008</td><td rowspan=1 colspan=1>0.00008</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=1>“four years</td><td rowspan=1 colspan=1>ago , ray k</td><td rowspan=1 colspan=1>urzweil an</td><td rowspan=1 colspan=1>l i , here at</td><td rowspan=1 colspan=1>ted , starte</td><td rowspan=1 colspan=1>a new uni</td><td rowspan=1 colspan=1>versity call</td><td rowspan=1 colspan=1>ed singular</td><td rowspan=1 colspan=1>university</td></tr><tr><td rowspan=1 colspan=1>0.99957</td><td rowspan=1 colspan=1>0.00005</td><td rowspan=1 colspan=1>0.00002</td><td rowspan=1 colspan=1>0.00009</td><td rowspan=1 colspan=1>0.00005</td><td rowspan=1 colspan=1>0.00005</td><td rowspan=1 colspan=1>0.00007</td><td rowspan=1 colspan=1>0.00007</td><td rowspan=1 colspan=1>0.00005</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=10 colspan=10>Example 4&quot;er forderte octavio auf , sein stabschef zu werden .&quot;“he asked octavio to be his chief of staff .&quot;0    B-MIS  I-MIS  B-PER  I-PER  B-ORG  I-ORG  B-LOC  I-LOC&quot;he asked octavio to be his chief .&quot;I-MIS                                 I-ORG            I-LOC&quot;he asked octavio to become his chief residency .&quot;I-MIS             I-PER  B-ORG  I-ORG  B-LOC  I-LOC</td></tr><tr><td rowspan=9 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.99958</td><td rowspan=1 colspan=1>0.00003</td><td rowspan=1 colspan=1>0.00002</td><td rowspan=1 colspan=1>0.00017</td><td rowspan=1 colspan=1>0.00005</td><td rowspan=1 colspan=1>0.00004</td><td rowspan=1 colspan=1>0.00004</td><td rowspan=1 colspan=1>0.00004</td><td rowspan=1 colspan=1>0.00003</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=1>he asked</td><td rowspan=1 colspan=1>ctavio to b</td><td rowspan=1 colspan=1>e his chief</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.99926</td><td rowspan=1 colspan=1>0.00005</td><td rowspan=1 colspan=1>0.00003</td><td rowspan=1 colspan=1>0.00028</td><td rowspan=1 colspan=1>0.00010</td><td rowspan=1 colspan=1>0.00008</td><td rowspan=1 colspan=1>0.00007</td><td rowspan=1 colspan=1>0.00007</td><td rowspan=1 colspan=1>0.00006</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr><tr><td rowspan=1 colspan=1>he asked</td><td rowspan=1 colspan=1>ctavio to b</td><td rowspan=1 colspan=1>ecome his</td><td rowspan=1 colspan=1>hief reside</td><td rowspan=1 colspan=1>ncy ."</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.99946</td><td rowspan=1 colspan=1>0.00005</td><td rowspan=1 colspan=1>0.00003</td><td rowspan=1 colspan=1>0.00005</td><td rowspan=1 colspan=1>0.00002</td><td rowspan=1 colspan=1>0.00008</td><td rowspan=1 colspan=1>0.00006</td><td rowspan=1 colspan=1>0.00014</td><td rowspan=1 colspan=1>0.00011</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>B-MIS</td><td rowspan=1 colspan=1>I-MIS</td><td rowspan=1 colspan=1>B-PER</td><td rowspan=1 colspan=1>I-PER</td><td rowspan=1 colspan=1>B-ORG</td><td rowspan=1 colspan=1>I-ORG</td><td rowspan=1 colspan=1>B-LOC</td><td rowspan=1 colspan=1>I-LOC</td></tr></table>

Table 6: Comparative analysis of Machine Translation Models with and without Named Entity Recognition (NER) Invarian Regularization. Each example shows the original German source sentence, the target English translation, and predicted translations from two models: BiBERT+NER (with NER regularization) and BiBERT (baseline). The tables include detailed logit distributions across named entity categories. The results illustrate how the NER regularization significantly preserves the correct named entities and their classifications, whereas the baseline model frequently omits or misclassifies named entities, highlight ing the practical advantages of invariant-based regularization.

![](images/fc8e104d1761027e55ce2b770de24d1e4240e2a31f6ff344b67597c9d91e9fa5.jpg)

<table><tr><td colspan="2">Ground Truth</td><td colspan="2">SmallCap+NER+SceneGraph</td><td colspan="3">SmallCap</td></tr><tr><td colspan="2">Caption: A man riding skis down a snow covered slope.</td><td colspan="2">Caption: A man riding skis down a snow covered slope.</td><td colspan="3">Caption: A person on skis in the snow.</td></tr><tr><td colspan="2">Entities: Head Span Modifiers</td><td colspan="2">Entities: Head</td><td colspan="3">Entities:</td></tr><tr><td rowspan="2">man skis slope</td><td rowspan="2">a man a skis a snow cov- a, covered</td><td rowspan="2">Span man a man skis skis slope</td><td rowspan="2">Modifiers a a snow cov- a, covered</td><td>Head Span person</td><td>a person</td><td>Modifiers a</td></tr><tr><td>skis skis snow</td><td>the snow</td><td>the</td></tr><tr><td>Relations: Subject</td><td>ered slope Relation Object</td><td>Relations: Subject Relation</td><td>ered slope</td><td>Relations:</td><td></td><td></td></tr><tr><td rowspan="2">man man</td><td rowspan="2">riding skis down slope</td><td rowspan="2">man riding down</td><td rowspan="2">Object skis slope</td><td rowspan="2">Subject</td><td rowspan="2">Relation</td><td rowspan="2">Object skis</td></tr><tr><td>person on skis in</td></tr></table>

Figure 6: Comparison of image captions produced by models trained with and without scenegraph and NER invariants for an image of a person skiing (left).  
![](images/a8b2dbebd46e3dbbf6294b2db2499035a0022c73735c34ad4c53cb98eb001f35.jpg)

<table><tr><td colspan="3">Ground Truth</td><td colspan="2">SmallCap+NER+SceneGraph</td><td colspan="3">SmallCap</td></tr><tr><td colspan="3">Caption: An orange and white cat lay- ing on top of a brown shoe.</td><td colspan="2">Caption: a brown and white cat laying on top of a shoe.</td><td colspan="3">Caption: a cat sniffing a pair of shoes on the floor.</td></tr><tr><td colspan="3">Entities:</td><td colspan="2">Entities:</td><td colspan="3">Entities:</td></tr><tr><td>Head Span cat</td><td>an orange and</td><td>Modifiers an, or-</td><td>Head Span cat white cat</td><td>Modifiers a brown and a, brown,</td><td>Head cat pair</td><td>Span a cat a pair</td><td>Modifiers a a</td></tr><tr><td>shoe</td><td>white cat a brown shoe a, brown</td><td>ange, white</td><td>shoe a shoe</td><td>white a</td><td>shoes floor</td><td>shoes the floor</td><td>the</td></tr><tr><td>Relations:</td><td colspan="2">Relation Object</td><td colspan="2">Relations: Subject Relation Object</td><td>Relations:</td><td></td><td></td></tr><tr><td colspan="3">Subject</td><td colspan="2"></td><td>Subject</td><td>Relation</td><td>Object</td></tr><tr><td colspan="3"></td><td colspan="2"></td><td>cat pair</td><td>sniffing of</td><td>pair shoes</td></tr></table>

Figure 7: Comparison of image captions produced by models trained with and without scenegraph and NER invariants for an image of a cat with a shoe (left).  
![](images/a50bd10164e28f7c830187fadf21cce529019bb282e4f777b8ad37548166c11e.jpg)

<table><tr><td colspan="3">Ground Truth</td><td colspan="2">SmallCap+NER+SceneGraph</td><td colspan="2">SmallCap</td></tr><tr><td colspan="3">Caption: A brown sheep standing on top of a lush green field.</td><td colspan="2">Caption: a sheep standing on top of a lush green field.</td><td colspan="2">Caption: a sheep that is standing in the grass.</td></tr><tr><td colspan="2">Entities: Head Span Modifiers</td><td>Entities: Head Span</td><td>Modifiers</td><td>Entities: Head</td><td>Span</td><td>Modifiers</td></tr><tr><td rowspan="2">sheep field</td><td colspan="2" rowspan="2">a brown sheep a, brown a lush green a, lush, field green</td><td>sheep a sheep a field a lush green a, lush, field</td><td>sheep that grass</td><td>a sheep that the grass</td><td>a</td></tr><tr><td>green Relations:</td><td colspan="3">the</td></tr><tr><td colspan="3">Relations: Subject Relation</td><td>Subject Relation Object</td><td colspan="2">Relations: Subject Relation that</td><td>Object</td></tr><tr><td colspan="3">Object</td><td></td><td colspan="3">in grass</td></tr></table>

Figure 8: Comparison of image captions produced by models trained with and without scenegraph and NER invariants for an image of a sheep in a field (left).

## Differential Performance of FBD Variants Across Tasks

Our experiments reveal an intriguing phenomenon: FBD-4 computed using token averaging performs well for machine translation but poorly for image captioning, while FBD-4 computed using CLS tokens shows the opposite pattern. This divergent behavior can be attributed to fundamental differences in how BERT encodes linguistic information for these distinct tasks.

## Task-Specific Information Distribution in BERT Layers

Machine Translation: The translation task requires preserving fine-grained linguistic features across languages. Lower and middle BERT layers capture syntactic structures and local dependencies that are crucial for translation quality. The average of all tokens across the last 4 layers effectively captures both syntactic patterns (from earlier layers) and semantic coherence (from later layers). Translation necessitates maintaining word order, grammatical structures, and morphological variations, which are distributed across multiple token representations.

Image Captioning: In contrast, image captioning requires mapping visual concepts to linguistic descriptions, making global semantic representation more important than fine-grained syntactic details. The CLS token in later layers aggregates scene level semantic information, which aligns better with the holistic nature of image-to-text mapping. Caption generation is less dependent on preserving exact syntactic structures compared to translation.

## Information Aggregation Patterns

The effectiveness of different aggregation strategies reflects the nature of information preservation required by each task: Token-Average (FBD-4) for Machine Translation:

• Averaging all tokens preserves local context and word-level nuances critical for translation

• This approach maintains information about positional relationships and local dependencies

• The invariants in translation (named entities, sentiment) are often localized to specific tokens, making token-level averaging more sensitive to these changes

CLS-Token (FBD-4-CLS) for Image Captioning:

• The CLS token acts as a global sentence representation, capturing overall semantic content

• For captions, the relationship between visual content and textual description is more about semantic correspondence than token-level precision

• Scene graphs and entity recognition in images map better to global semantic representations than to individual token positions

## Invariant Sensitivity Mechanisms

The effectiveness of each metric variant correlates with how invariants manifest in each task. Translation invariants (NER, sentiment) are embedded within specific token sequences, making token-averaging more sensitive to their preservation. In contrast, caption invariants (scene graphs, visual entities) are better captured by global representations that abstract away from specific token positions.

## Theoretical Formulation

The differential performance of FBD variants can be formalized as follows:

For machine translation:

$$
\mathrm { F B D - } 4 _ { \mathrm { a v g } } = D \left( \mu \left( h _ { 1 : n } ^ { L - 3 : L } \right) , \mu \left( h _ { 1 : n } ^ { \prime L - 3 : L } \right) \right)\tag{1}
$$

For image captioning:

$$
\mathrm { F B D - } 4 _ { \mathrm { c l s } } = D \left( h _ { \mathrm { C L S } } ^ { L - 3 : L } , h _ { \mathrm { C L S } } ^ { \prime L - 3 : L } \right)\tag{2}
$$

where h represents hidden states, L is the last layer, n is sequence length, and D is the Frechet distance.´

The superior performance of token-averaging for translation reflects the need to preserve fine-grained linguistic features, while CLS-based aggregation excels for captioning due to its alignment with global semantic mapping from visual to textual domains.

## Implications for Metric Design

This phenomenon aligns with recent findings in the literature. Tenney et al. (2019) showed that BERT’s linguistic knowledge follows a classical NLP pipeline, with syntactic information in middle layers and semantic information in upper layers. Rogers et al. (2020) demonstrated that CLS tokens in final layers best capture sentence-level semantic similarity.

Our findings highlight a crucial insight: the optimal aggregation strategy for distributional metrics depends on the task’s information requirements. This task-specific optimization of distributional metrics underscores the importance of considering the underlying information distribution when designing evaluation metrics for multi-output structured prediction tasks. Rather

than applying metrics uniformly across tasks, researchers should carefully consider how information is encoded and what aspects of the output are most critical for task performance.

This observation has broader implications for the field: as we develop more sophisticated evaluation metrics, we must move beyond one-size-fits-all approaches and design metrics that align with the specific information patterns and invariant structures of each task.

## Experimental Details

All the training parameters of baseline model were kept same including learning rate.

## Image Colorization

General Colorization Training Details

• ChromaGAN

– Input image size: 224×224

– We reimplemented ChromaGAN in PyTorch to ensure fair comparison with our PyTorch-based models, achieving identi cal validation performance (FID = 5.81 on 50k images) to the original TensorFlow pretrained model.

– All ChromaGAN results reported in our experiments correspond to the original TensorFlow pretrained model, while TaskNet models are implemented in PyTorch.

– Our PyTorch implementation maintains architectural fidelity and parameter count consistency with the original Chroma-GAN design.

• ColorFormer

– Input image size: 256×256

– We used same pytorch implementation of basicSR repository and used by ColorFormer

– We added invariant loss files in train colorformer.yml file

## • DDColor

– Input image size: 256×256

– We used same pytorch implementation of basicSR repository and used by DDColor

– We added invariant loss files in train colorformer.yml file

– We used DDColor-tiny model for all our experiments and retrained the model.

Table 7: Hyperparameter α Settings for Image Colorization Experiments. α represents the multiplier to invariant loss added to total training loss during colorization training.
<table><tr><td>Model</td><td>Dataset</td><td>Best α</td><td>Batch Size</td><td>Epochs / Iterations</td><td>Hyperparameters Tried (α)</td></tr><tr><td>ChromaGAN</td><td>ImageNet</td><td></td><td>10</td><td>5 epochs</td><td></td></tr><tr><td>ChromaGAN + Seg</td><td>ImageNet</td><td>0.02</td><td>10</td><td>15 epochs</td><td>0.0002, 0.002, 0.02, 0.05, 0.2</td></tr><tr><td>ChromaGAN + Depth</td><td>ImageNet</td><td>0.0001</td><td>10</td><td>15 epochs</td><td>0.0001, 0.001, 0.005, 0.01</td></tr><tr><td>ChromaGAN + Seg + Depth</td><td>ImageNet</td><td>0.02,0.0001</td><td>10</td><td>15 epochs</td><td>(0.02,0.0001),(0.02,0.005),(0.01,0.001)</td></tr><tr><td>ColorFormer</td><td>ImageNet</td><td></td><td>16</td><td>200,000 iterations</td><td></td></tr><tr><td>ColorFormer + Seg</td><td>ImageNet</td><td>0.05</td><td>16</td><td>600,000 iterations</td><td>0.0001, 0.001, 0.01, 0.05, 0.1, 0.2</td></tr><tr><td>ColorFormer + Depth</td><td>ImageNet</td><td>0.01</td><td>16</td><td>600,000 iterations</td><td>0.0001, 0.001, 0.01, 0.05, 0.1, 0.2</td></tr><tr><td>ColorFormer + Seg + Depth</td><td>ImageNet</td><td>0.05,0.01</td><td>16</td><td>600,000 iterations</td><td>(0.05,0.01),(0.01,0.001),(0.05,0.001)</td></tr><tr><td>DDColor</td><td>ImageNet</td><td></td><td>16</td><td>400,000 iterations</td><td></td></tr><tr><td>DDColor + Seg</td><td>ImageNet</td><td>0.5</td><td>16</td><td>800,000 iterations</td><td>0.05, 0.1, 0.5, 2, 5</td></tr><tr><td>DDColor + Depth</td><td>ImageNet</td><td>0.02</td><td>16</td><td>800,000 iterations</td><td>0.001, 0.01, 0.02, 0.05, 0.1</td></tr><tr><td>DDColor + Seg + Depth</td><td>ImageNet</td><td>2,0.02</td><td>16</td><td>800,000 iterations</td><td>(2,0.02),(0.5,0.02),(0.1,0.01)</td></tr></table>

## Invariant Network and Loss Details

• Side Adapter Network: It is an open vocabulary semantic segmentor network, which segments images based on the vo cabulary given to it. For enhanced segmentation, ImageNet classes were incorporated, resulting in a total of 1171 classes. Given the predicted output pred and the ground truth $\texttt { g t }$ , the Dice Loss for multi-class segmentation is calculated using the following formula:

$$
{ \mathrm { D i c e L o s s } } ( { \mathrm { p r e d } } , { \mathrm { g t } } ) = 1 - { \frac { 2 \times \sum _ { i = 1 } ^ { C } p _ { i } \cdot g _ { i } + \epsilon } { \sum _ { i = 1 } ^ { C } p _ { i } ^ { 2 } + \sum _ { i = 1 } ^ { C } g _ { i } ^ { 2 } + \epsilon } }\tag{3}
$$

where $C$ represents the total number of classes (1171 in this case), $p _ { i } ( { \tt p r e d } )$ and $g _ { i } ( \mathsf { g t } )$ denote the predicted and ground truth probabilities for the $i ^ { t h }$ class, respectively, and ϵ is a small constant added for numerical stability.

• MiDaS: We used the ”MiDaS small” version from the PyTorch hub. MiDaS takes an RGB image as input, so during training, we convert the colorizer’s Lab output to RGB. The loss function used is the silog loss.

Let M be the number of pixels in the image. Let $d = d \left( \theta \right) \in \mathbb { R } ^ { M }$ be the disparity prediction output by the depth estimator and $d = d \ast \in \mathbb { R } ^ { M }$ be the ground truth disparity. Then the silog loss can be written as:

$$
L _ { s i l o g } \left( z , z * \right) = \operatorname* { m a x } _ { \mathbf { s } } \frac { 1 } { 2 M } \sum _ { i = 1 } ^ { M } \left( \log \left( e ^ { s } z _ { i } \right) - \log \left( z * _ { i } \right) \right)\tag{4}
$$

## Machine Translation

## General Machine Translation Training Details

## • BiBERT

– Model architecture: BERT-based bidirectional encoder with cross-attention

– Hidden dimension: 768, Number of layers: 12

– Vocabulary size: 37,000 shared BPE tokens

– Maximum sequence length: 150 tokens

– We used the official BiBERT implementation with PyTorch

– Training conducted on 1 NVIDIA V100 GPUs with mixed precision

– Gradient accumulation steps: 4

– Warmup steps: 4,000

## • Bi-SimCut

– Model architecture: Transformer with simple data augmentation

– Hidden dimension: 512, Number of layers: 6

– Uses bidirectional training with cut-off augmentation

– Dropout rate: 0.3

– Label smoothing: 0.1

– Training conducted on 1 NVIDIA V100 GPUs

Table 8: Hyperparameter α Settings for Neural Machine Translation Experiments
<table><tr><td>Model</td><td>Dataset</td><td>Best α</td><td>Batch Size</td><td>Epochs</td><td>Hyperparameters Tried (α)</td></tr><tr><td>BiBERT</td><td>IWSLT-14</td><td></td><td>4096 tokens</td><td>75</td><td></td></tr><tr><td>BiBERT + NER</td><td>IWSLT-14</td><td>100</td><td>4096 tokens</td><td>75</td><td>10, 50, 100, 500, 1000</td></tr><tr><td>BiBERT + SeAM</td><td>IWSLT-14</td><td>50</td><td>4096 tokens</td><td>75</td><td>10, 50, 100, 500, 1000</td></tr><tr><td>BiBERT + NER + SeAM</td><td>IWSLT-14</td><td>100,50</td><td>4096 tokens</td><td>75</td><td>(100,50), (50,50), (100,100)</td></tr><tr><td>Bi-SimCut</td><td>IWSLT-14</td><td></td><td>4096 tokens</td><td>75</td><td></td></tr><tr><td>Bi-SimCut + NER</td><td>IWSLT-14</td><td>200</td><td>4096 tokens</td><td>75</td><td>50, 100, 200, 500, 1000</td></tr><tr><td>Bi-SimCut + SeAM</td><td>IWSLT-14</td><td>100</td><td>4096 tokens</td><td>75</td><td>50, 100, 200, 500, 1000</td></tr><tr><td>Bi-SimCut + NER + SeAM</td><td>IWSLT-14</td><td>200,100</td><td>4096 tokens</td><td>75</td><td>(100,50), (50,50), (200,100)</td></tr></table>

## Invariant Network and Loss Details

• NER (Named Entity Recognition) Network: We use Hugging Face’s dslim/bert-large-NER model which identifies four entity types: Person (PER), Organization (ORG), Location (LOC), and None (O). The NER loss is computed as KL divergence between predicted and ground truth entity distributions:

$$
L _ { N E R } = \sum _ { i = 1 } ^ { N } \sum _ { \substack { c \in \{ P E R , O R G , L O C , O \} } } p _ { i } ^ { c } \log { \left( \frac { p _ { i } ^ { c } } { q _ { i } ^ { c } } \right) }\tag{5}
$$

where $N$ is the sequence length, $p _ { i } ^ { c }$ is the ground truth probability for class c at position $i ,$ and $q _ { i } ^ { c }$ is the predicted probability.

• Sentiment Analysis (SeAM) Network: We use Hugging Face’s distilbert-base-uncased-finetuned-sst-2-english for binary sentiment classification. The sentiment loss ensures translations preserve the emotional tone:

$$
L _ { S e A M } = D _ { K L } ( P _ { s e n t } ^ { s r c } | | P _ { s e n t } ^ { t g t } )\tag{6}
$$

where $P _ { s e n t } ^ { s r c }$ and $P _ { s e n t } ^ { t g t }$ are the sentiment distributions for source and translated sentences respectively.

## Training Procedure

• All models trained for 75 epochs on IWSLT-14 and 300k steps on WMT-14

• Learning rate schedule: inverse square root with 8,000 warmup steps

• Optimizer: Adam with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 8$

• Batch size: 4,096 tokens per GPU

• Checkpoint selection based on validation BLEU score

• For invariant models, checkpoint selection based on validation FBD score

## Image Captioning

## General Image Captioning Training Details

## • SmallCap

– Architecture: Lightweight transformer with retrieval augmentation

– Vision encoder: CLIP ViT-B/32

– Text decoder: GPT-2 small (117M parameters)

– Uses retrieval from training set to provide relevant examples

– Maximum caption length: 40 tokens

– Beam size: 5 for inference

## • ClipCap

– Architecture: CLIP vision encoder with GPT-2 language model

– Mapping network: 8-layer transformer to project CLIP features

– Text decoder: GPT-2 medium (345M parameters)

– Prefix length: 10 tokens

– Does not use additional retrieval mechanisms

Table 9: Hyperparameter α Settings for Image Captioning Experiments
<table><tr><td>Model</td><td>Dataset</td><td>Best α</td><td>Batch Size</td><td>Epochs</td><td>Hyperparameters Tried (α)</td></tr><tr><td>SmallCap</td><td>COCO</td><td></td><td>32</td><td>10</td><td></td></tr><tr><td>SmallCap + NER</td><td>COCO</td><td>0.5</td><td>32</td><td>10</td><td>0.1, 0.3, 0.5, 0.7, 1.0</td></tr><tr><td>SmallCap + SceneGraph</td><td>COCO</td><td>1.0</td><td>32</td><td>10</td><td>0.5, 1.0, 2.0, 5.0</td></tr><tr><td> $\mathrm { S m a l l C a p + N E R + S G }$ </td><td>COCO</td><td>0.5, 1.0</td><td>32</td><td>10</td><td>(0.5,1.0), (0.3,1.0), (0.5,2.0)</td></tr><tr><td>ClipCap</td><td>COCO</td><td>一</td><td>64</td><td>10</td><td></td></tr><tr><td> $\mathrm { C l i p C a p + N E R }$ </td><td>COCO</td><td>0.3</td><td>64</td><td>10</td><td>0.1, 0.3, 0.5, 0.7, 1.0</td></tr><tr><td> $\mathrm { C l i \bar { p } C a \bar { p } + S c e n e G r a p h }$ </td><td>COCO</td><td>0.7</td><td>64</td><td>10</td><td>0.5, 0.7, 1.0, 2.0</td></tr><tr><td> $\mathrm { C l i p C a p + N E R + S G }$ </td><td>COCO</td><td>0.3, 0.7</td><td>64</td><td>10</td><td>(0.3,0.7), (0.3,1.0), (0.5,0.7)</td></tr></table>

## Invariant Network and Loss Details

• Named Entity Recognition (NER): Same BERT-large-NER model as in MT, adapted for caption text. Ensures proper nouns and entities in images are correctly identified in captions.

• Scene Graph Parser: Uses the Stanford Scene Graph Parser to extract structured representations. The scene graph loss compares entities and relationships:

$$
L _ { S G } = \lambda _ { e } L _ { e n t i t i e s } + \lambda _ { r } L _ { r e l a t i o n s }\tag{7}
$$

where:

$$
L _ { e n t i t i e s } = \frac { | E _ { g t } \triangle E _ { p r e d } | } { | E _ { g t } \cup E _ { p r e d } | }\tag{8}
$$

$$
L _ { r e l a t i o n s } = \frac { | R _ { g t } \triangle R _ { p r e d } | } { | R _ { g t } \cup R _ { p r e d } | }\tag{9}
$$

$E _ { g t } , E _ { p r e d }$ are ground truth and predicted entity sets, $R _ { g t } , R _ { p r e d }$ are relation sets, and $\triangle$ denotes symmetric difference.

## Training Procedure

• Trained for 10 epochs on COCO and 15 epochs on Flickr30k

• Learning rate: 5e-5 with cosine annealing

• Optimizer: AdamW with weight decay 0.01

• Batch size: 32 for SmallCap, 64 for ClipCap

• Teacher forcing ratio: 1.0 (always use ground truth during training)

• Gradient clipping: max norm 1.0