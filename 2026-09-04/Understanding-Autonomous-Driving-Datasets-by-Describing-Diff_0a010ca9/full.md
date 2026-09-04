# Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language

Julian Truetsch<sup>∗†</sup>, Felix Hauser<sup>∗†</sup>, Christoph Stiller<sup>†</sup>, Fellow, IEEE, and Frank Bieder<sup>∗†</sup>

Abstract—Understanding the composition of large-scale autonomous driving datasets is essential for safety, robustness, and reliable operation across domains. For example, domain shift between locations could lead to the operating environment being misaligned with the training data, resulting in potentially dangerous performance degradation. Yet, existing data analysis pipelines largely rely on metadata, predefined labels, or manual inspection, which provide limited semantic insight or do not scale. This paper studies set difference captioning: given two subsets of images, the goal is to produce a natural-language hypothesis describing differences between the target and reference set. Building on a two-stage formulation, we adapt the method to autonomous driving by focusing on object-centric patches derived from object detection, which simplifies aggregation and enables attribution of differences to specific object instances or categories. To evaluate this setting in-domain, we introduce a new benchmark, AD-Diff Bench. Low-concentration experiments assess the suitability of set-difference-captioning approaches to sparse, realworld differences. We restrict our experiments to open-weight models to support reproducibility and ease of deployment. The proposed benchmark and analysis provide a step towards practical, human-interpretable dataset introspection for autonomous driving datasets. Our implementation and benchmark dataset are available at https://github.com/KIT-MRT/AD-Diff

Index Terms—autonomous driving, benchmark, dataset introspection, domain shift, set difference captioning, vision language model

## I. INTRODUCTION

O <sup>VER</sup> <sup>the</sup> <sup>last</sup> <sup>decade,</sup> <sup>autonomous</sup> <sup>driving</sup> <sup>research</sup> <sup>has</sup>shifted more and more towards machine learning ap- shifted more and more towards machine learning approaches, relying on deep-learning-based system components for detection, tracking, and increasingly also for decisionmaking and trajectory planning. Many in the community believe that the field’s future lies in fully end-to-end autonomous driving systems that are completely data-driven, without any engineered heuristics or model-based parts. Consequently, data quantity and quality become primary factors in achieving high system reliability and robustness.

Much effort has been put into answering the question of how to effectively collect, label, and curate [1], [2] the large and diverse datasets that are required for a data-driven approach to autonomous driving. Complete “data engines” have been proposed to automate large parts of a continuous record-labelretrain-deploy pipeline [3], [4].

Yet, very little research addresses tools to help developers and stakeholders understand dataset composition – essential for adequate system performance. Without such data introspection tools, decision-making regarding dataset composition and system operation will always rely either on guesswork or on costly, poorly scalable human labor.

In this paper, we adapt a set-difference-captioning tool for the autonomous driving research community. Given two potentially very large image sets, the goal of set difference captioning is to generate natural-language descriptions of their differences. In realistic deployments, fleets of autonomous vehicles collect data over diverse sensor configurations, locations, and time periods, yielding large volumes of raw data accompanied by metadata (e.g., timestamps, GNSS positions, data source, etc.) and partial semantic labels (e.g., bounding boxes). However, analysis based on metadata and available annotations is constrained by the predefined label space and therefore provides only limited insight.

A tool that generates natural-language difference descriptions between two image sets enables pairwise comparisons of subsets. This supports scalable dataset introspection by revealing distribution shifts, gaps, trends, and biases, including subtle differences that are not readily apparent to human inspection. Such differences are directly relevant to safety and reliability when deploying and validating autonomous driving systems across domains. They can indicate changes in object appearance, infrastructure, or environmental conditions that are not captured by existing labels and, therefore, can affect downstream model behavior.

Natural language provides a flexible and expressive summary format that complements fixed metadata and label taxonomies. Crucially, the method emphasizes relative descriptions: comparing a target set against a reference set highlights salient properties and suppresses incidental variation. In contrast, unreferenced set descriptions are either trivial (e.g., “there are cars”) or become unmanageably exhaustive when applied to high-dimensional image data, where the space of potentially describable properties grows rapidly with dataset size and diversity. A reference set is therefore essential to determine which properties are meaningful and relevant for decisions, as opposed to expected or irrelevant background variation.

The proposed approach enables more automated data pipelines while preserving a human-in-the-loop interface for safety-relevant decisions in increasingly automated data collection and management systems. Our key contributions are:

• Object-centric set difference captioning. Building on prior work on set difference captioning, we adapt the task to the autonomous driving domain by introducing objectcentric set difference captioning, a formulation tailored to safety-relevant objects and long-tail visual concepts in driving data.

• Set difference captioning benchmark for autonomous driving. We curate a benchmark dataset and evaluation protocol for object-centric set difference captioning, termed AD-Diff Bench, which we will release publicly to support reproducible comparisons.

• First study of high-sparsity differences. We study set difference captioning in the high-sparsity regime, clarifying its real-world applicability to distribution shifts and safety-relevant dataset diagnostics.

## II. RELATED WORK

Prior work on dataset introspection spans diverse approaches, yet often relies on predefined categories or additional annotations [5], or targets narrow aspects such as identifying challenging subsets [6], [7], limiting generality. In contrast, we are interested in open-ended, human-interpretable descriptions of differences between subsets. We therefore consider set difference captioning, i.e., generating natural language descriptions that characterize how two image sets differ.

1) Vision Language Models: To bridge the gap from images to natural language descriptions, we rely on Vision Language Models (VLMs). Modern VLMs couple high-capacity vision encoders with Large Language Models (LLMs), enabling compositional scene reasoning. Early models like Flamingo [8] and BLIP-2 [9] demonstrated strong few-shot capabilities by combining frozen image encoders with LLMs. LLaVA [10] introduced visual instruction tuning, providing VLMs with reasoning capabilities similar to those of LLMs. The emergence of high-performing open-weight VLMs like Qwen3-VL [11] and InternVL-3.5 [12] now enables practical deployment of VLM-based systems without relying on proprietary APIs, which is particularly relevant for autonomous driving applications where data privacy and offline operation are important considerations.

2) Difference and Change Captioning: Prior work on difference captioning [8], [13]–[15] and change captioning [16]– [18] has focused on describing differences between pairs of images. However, scaling these methods to thousands of images would require quadratic comparisons and yield fragmented descriptions, which is a crucial limitation for datasetlevel analysis.

3) Set Difference Captioning: In the text domain, the D3 framework [19] introduced a proposer-ranker approach to describe differences between text distributions using LLMs. Based on this prior work in the text domain, Dunlap et al. [20] formalized the task of describing differences between sets of images. They introduced VisDiff, a two-stage algorithm combining a caption-based proposer using BLIP-2 and GPT-4 with a CLIP-based ranker. Their work demonstrated the utility of set difference captioning for understanding dataset shifts, comparing model behaviors, and analyzing generative models. GS-CLIP [21] proposed a related approach using CLIP features to retrieve differences from a predefined text bank, though it is limited to a fixed vocabulary. Our work builds directly on the VisDiff framework but adapts it to the autonomous driving domain using state-of-the-art open-weight models and introduces an object-centric formulation better suited to complex traffic scenes.

4) Domain Shift Datasets: To benchmark set difference captioning, we require image-sets with ground-truth difference descriptions. Since even subtle dataset shift can significantly reduce accuracy [22], many benchmarks have been proposed to evaluate classifier robustness to domain shift [23]–[27]. Of these only MetaShift [25] includes explicit ground-truth annotations for domain-shift and enough image-sets for a meaningful evaluation of set difference captioning. VisDiff-Bench [20] is specifically designed to evaluate set-difference captioning. It consists of 150 image-set pairs with 100 webscraped images per set, and additionally 37 image-set pairs from ImageNetR [24] and ImageNet\* [28]. MetaShift and VisDiffBench contain, however, only few images from road environments. We fill this gap by introducing AD-Diff Bench, the first set-difference captioning benchmark for autonomous driving, consisting solely of domain-relevant images.

## III. SET DIFFERENCE CAPTIONING FOR AUTONOMOUS DRIVING

Our approach to understanding autonomous driving datasets is based on descriptions of the differences between two subsets of the data. The specific question that one wants to answer (e.g., “How are cars in country A different from country B?”, “Which conditions contribute to accidents/close-calls?”) determines the criteria by which the data is split into two subsets (location and accident label, respectively, for the examples). Once the subsets are defined, the task is reduced to a set difference captioning task [20]. In this section, we first define the task of set difference captioning before aligning this more general task definition to the specific requirements in the autonomous driving domain, introducing the task of object-centric set difference captioning.

## A. Set Difference Captioning

The goal of the set difference captioning task (Fig. 1) is to generate natural-language descriptions of the differences between two image sets A and B. We follow prior work [20] by defining a correct difference description as a description that is more true for set A than for set B. We refer to set A as the target set and to set B as the reference set. The ground truth difference descriptions in our benchmark are therefore descriptions of the target set A. A difference description is considered correct if it is semantically equivalent to the ground truth difference description.

![](images/b7cef0a73f179367272e1952999a18ac59fc6177794ce7388697f926aebbcd3c.jpg)

![](images/3cd26ec7dc8709ac05b309d21fcc8a362ab41f5c4aa80759d8710385f6765be3.jpg)  
Fig. 2. Splits of AD-Diff Bench. Example set-pairs for web-scraped, annotation-filtered, and CLIP-filtered split. Images in the annotation-filtered and CLIP-filtered split are extracted from KITTI, nuImages, and Waymo Open Dataset. Each row shows three images from each set together with the groundtruth set description. The ground-truth set-pair difference description is the description of the target set, marked in green.

TABLE I  
NUMBER OF SETS AND IMAGES IN EACH SPLIT OF AD-DIFF BENCH.
<table><tr><td>Dataset</td><td># Paired Sets</td><td># Images Per Set</td></tr><tr><td>Web-scraped</td><td>60+60+60</td><td>100</td></tr><tr><td>Annotation-filtered</td><td>60</td><td>10-8000</td></tr><tr><td>CLIP-filtered</td><td>80</td><td>100</td></tr></table>

![](images/c8758d7e1f9d869b3725e64ef974db3945c2a5a626c81890423867c5a315160b.jpg)  
Fig. 1. Object-centric set difference captioning. Given two image datasets A and B, we find difference descriptions between the objects in the two datasets. The output is a natural-language description of a concept that is more true for the target set A than for the reference set B. In this example, A and B are images from the nuImages and KITTI dataset, respectively.

## B. Object-centric Set Difference Captioning

Directly applying set difference captioning to camera images from autonomous driving datasets will, in most cases, not result in useful difference descriptions. Any single image typically contains many traffic participants and other objects; difference descriptions between complete images are therefore prone to missing relevant differences. Furthermore, descriptions would accumulate information about different objects in one or multiple images in natural language (e.g., “3 more people with red shirts”), making automated aggregation across sets difficult.

We therefore introduce a modification of the original task that we term object-centric set difference captioning. The goal of this task is to caption the difference between two sets of objects observed in the sensor data. Generally, but not necessarily, the objects in both sets are of the same class, as differences between objects of different classes are rarely informative – there is no point in “comparing apples to oranges”. This object-centric approach resolves aggregation issues and makes it straightforward to ensure that no objects are ignored during difference captioning. If required, it is also easier with this approach to attribute difference descriptions to specific object instances.

To adapt existing set-difference captioning methods to the object-centric setting, we first select one image in which the object appears, if it is tracked across multiple time steps or across cameras. We then extract an image patch centered on the object from the corresponding camera image, as shown in Fig. 1. The object is localized using pre-trained 2D detection models or bounding box annotations, if available. These image patches can be partitioned into subsets based on dataset annotations or metadata and then compared using the same procedure as in general set-difference captioning. Some differences between sets are only identifiable given additional context from the image. Therefore, we enlarge the image patches by 50 % from the raw bounding box annotations. The object of interest is marked with a red bounding box painted on the image patch, allowing the set-difference-captioning approach to locate the correct object even in crowded scenes.

## IV. BENCHMARK

The only existing benchmark for set difference captioning of images [20] targets general-purpose use. The domain shift to camera images recorded from autonomous vehicles is therefore high. We introduce AD-Diff Bench, a benchmark for objectcentric set difference captioning, consisting of images from the autonomous driving domain and taking requirements for real-world use in this application into account. This section first introduces the datasets we collected for AD-Diff Bench before explaining techniques to test whether set captioning approaches also work on sparse or noisy sets, which are common in the real world.

## A. Set Differences Dataset

To compare set-difference-captioning approaches with respect to their applicability as a development tool for real autonomous driving systems, a sufficiently large dataset of domain-relevant image set pairs with ground truth difference descriptions must be collected first. We consider images of traffic scenes, road users, and any object that could reasonably appear on public roads as domain-relevant. To make the benchmark challenging enough, all images of an image set pair display traffic-relevant objects of the same object category, with the difference between the two image sets being subtle enough to be interesting in the context of a set difference captioning task.

At the same time, differences across the dataset should be as diverse as possible, as set captioning approaches should be applicable to any possible semantic difference between image set pairs. To this end, we first manually craft a list of setdifferences we want to include in the dataset, with each set difference being defined by target-captions or filter criteria for the target image set (A) and the reference image set (B) of each image set pair. Based on this, we use three different approaches to collect the images for the image sets, resulting in the three datasets of image set pairs with ground truth difference annotations that make up AD-Diff Bench. Table I summarizes the three datasets of our benchmark, Fig. 2 shows example images for each split.

1) Web-scraped: Similarly to VisDiffBench [20], a benchmark for general image set difference captioning, our first dataset is collected by querying Bing image search for the predefined target-captions. Each image set consists of the top 100 results retrieved from Bing image search for its target-caption. The ground-truth difference description of the image set pair is then the search query of the target image set (A). We manually filter out image set pairs that do not display the intended targetdifference or that prominently display differences other than the ground-truth difference. We collect 180 image set pairs, grouped by expert knowledge into three difficulty levels and 6 domain-relevant object categories: pedestrians, cars and light trucks, micromobility (bicycles, e-scooters, etc.), large vehicles, traffic control devices, and misc; with 10 image set pairs per object category and difficulty level. The resulting dataset is very diverse in both images and difference-descriptions, including both common and rare (corner-case) representations of road users and traffic-relevant objects.

2) Annotation-filtered: The web-scraped dataset is very diverse but shows a significant domain shift away from images recorded by autonomous vehicles (camera system, perspective, sampling bias). To be able to draw confident conclusions regarding the applicability of set-difference-captioning approaches to data from the autonomous driving domain, we additionally create a dataset of image set pairs sampled from three different autonomous driving datasets: KITTI [29], nuImages [30], and Waymo Open Perception [31], [32].

In line with the object-centric approach, both image sets of each pair consist of image patches featuring objects of the same class, cut from the bounding box annotations of the three autonomous driving datasets. The filter criteria that determine which images are assigned to which image set are defined based on the fine-grained annotations for sub-classes and attributes contained in these datasets, resulting in very clean and well-defined image sets. Difference descriptions are manually defined from the filtered attributes.

The image set pairs of this dataset vary in image set size from 10 to 8000 images, testing not only the accuracy but also the scalability of set-difference-captioning approaches.

We provide two sub-splits of this dataset: the main split annotation-filtered-60, including all 60 image set pairs of this split, and the subset annotation-filtered-39, which is primarily intended for experiments with the concentration parameter that we will introduce in Section IV-B.

3) CLIP-filtered: While curating image sets based on already available annotations is convenient and results in very accurate and clean set-pairs, diversity is limited by the predefined categories and attributes. To obtain more set-pairs with low domain shift from data collected by deployed autonomous vehicles, we reuse the image patches extracted from the autonomous driving datasets for the annotation-filtered dataset, but use CLIP instead of annotations to filter the image patches into subsets. For each image set, we use OpenClip [33] to filter all image patches of the respective object-category for the 100 best matching image patches given the predefined keyword (highest cosine similarity between the image-embedding of the image patch and the text-embedding of the keyword). Keywords are based on both descriptions of the target set and the reference set of the web-scraped dataset. Sets containing too many incorrect matches are discarded by manual inspection. The 80 pairs of image sets of this dataset consist of the keyword ground-truth description, the CLIP-filtered image patches for the target set, and 100 image patches randomly sampled from the same category for the reference set.

## B. Set Difference Captioning of Diluted Sets

The image sets in AD-Diff Bench are relatively clean, i.e., every image set contains primarily images that match the corresponding set description. Sets of images collected in the real world are, however, rarely clean and usually contain images of objects with many different appearances that do not all fit a singular set description.

The VisDiffBench set difference captioning benchmark [20] introduced the concept of purity to account for this mismatch between reality and the benchmark. Purity is a parameter that can be set for each run of the benchmark to a value between 0 and 1 to test the performance of set captioning approaches on noisy sets, i.e., sets with partially incorrect labels. A purity value of 1 indicates a completely clean, i.e., unchanged set pair. Lower values indicate that some portion of the images has been swapped between the two sets, with half the images being swapped for a value of zero, leading to completely shuffled sets. A set-difference-captioning approach that works even for low purity values would, for example, be able to detect, given sets of images of pedestrians from two different cities, that pedestrians wear hats more often in one of the cities. A set-difference-captioning approach that works only for purity values close to one will only detect such a difference if hats are only ever worn in one of the two cities.

Sparse differences between sets are equally relevant and common. The difference between two sets is sparse if the correct difference description is only true for a small fraction of the images in each set. If there were, for example, a new fad for pink-colored cars, which were previously nearly nonexistent, then the difference observed between an image set recorded after the trend has been established and an image set recorded years earlier would be a sparse difference. Not all cars are colored pink with the new trend; non-pink cars are still much more common. Images fitting the difference description appear in this case only in one of the sets but are rare even in this set. This is particularly important for the development of safe and reliable autonomous driving systems, which requires identifying even very rare potential hazards before they become relevant.

To test the capability of set-difference-captioning approaches to detect sparse differences, we introduce the concept of dilution to AD-Diff Bench. A set pair is diluted by adding additional images from a third set to both sets. Dilution is measured by the parameter concentration; the more diluted the set, the lower the concentration. An image set S of $n _ { S }$ images is diluted to a concentration of $\begin{array} { r } { c = \frac { n _ { S } } { n _ { S } + n _ { D } } } \end{array}$ , with $n _ { D }$ being the number of dilution images sampled from the dilution set D and added to the set S. Each set pair in the annotation-filtered split of AD-Diff Bench is assigned a dilution set that is selected to be similar enough to the target and reference set to make high dilution experiments challenging. For some set-pairs, the dilution set can overlap or be identical to the reference set, but there is never any overlap between the target set and the dilution set.

We consider the accurate detection of even very sparse differences to be of paramount importance for a set difference captioning tool when used for autonomous driving applications. To illustrate this, consider that out of the 784 444 annotations for the class vehicle in the nuImages [30] dataset, only 42 correspond to the sub-class ambulance. This corresponds to a concentration of $c \approx 5 . 4 \times 1 0 ^ { - 5 }$ . This value is tiny, but correctly identifying ambulances is without a doubt still important for autonomous vehicles.

## C. Evaluation

To evaluate whether the generated difference captions should be considered correct, we employ the open-weight LLM gpt-oss-120b [34] as a judge. gpt-oss is tasked with evaluating whether the ground-truth difference description from the dataset is semantically equivalent to the difference description proposed by the evaluated approach. The prompt given to gpt-oss includes exact scoring rules for edge cases where the decision between clearly semantically equivalent and not equivalent is not obvious. Based on these scoring rules, gpt-oss can give a score of 0 (no match), 0.5 (partial match), and 1 (perfect match). The highest score among the top-N ranked hypotheses for each set pair is averaged across all set pairs to compute the top-N accuracy (Acc@N).

To evaluate whether the assessment of gpt-oss is reliable, we manually labeled over 1000 hypotheses on VisDiffBench [20] using the same scoring rules that we provide to gpt-oss in the prompt. We find that gpt-oss assigns the same score as the human annotator for 80.3 % of the hypotheses, with a low mean absolute error of 0.104.

## V. APPROACH

Our general approach follows [20] with the distinction between separate proposer and ranker submodules, illustrated in Fig. 3. We consider two datasets, the target set A and the reference set $B ,$ for which we seek additional insights. Since comparing large datasets directly using vision-language models is still not feasible, the proposer takes a subset $S _ { A } \subseteq A , S _ { B } \subseteq B$ of the two input datasets and generates a list of difference hypotheses. This is followed by the ranker, which compares these hypotheses against the full datasets A, B and assigns a score to each hypothesis. The final explanation for perceived differences between the two datasets is then the top ranked hypothesis. For transparency and reproducibility, we use publicly available, open-weight state-of-the-art models.

## A. Proposer

We employ one of three methods to generate difference hypotheses: image-based, caption-based, or feature-based. To ensure a fair comparison, all three proposer methods use Qwen3-VL-30B-A3B-Instruct as the VLM and LLM. We perform three generation rounds. In each round, we independently sample 20 images from each set A and B and generate ten difference hypotheses.

1) Image-based: Since modern VLMs accept multiple images as input, we just feed the subsets $S _ { A } , S _ { B }$ directly into the model together with the prompt.

2) Caption-based: In a first step, we generate captions for all images in the sampled subsets $S _ { A } , S _ { B }$ with the help of the VLM. We then compile the captions into a new textonly prompt and reprocess them with the LLM to generate the hypothesis.

3) Feature-based: We start by extracting visual embeddings from the VLM for all images in $S _ { A }$ and $S _ { B }$ , and compute the mean embedding for each subset. Using feature arithmetic, we take the difference between the mean embeddings and use it as a direction that captures the shift from A to B. We combine this difference representation with prompt text embeddings and pass the result through the VLM decoder to produce the hypothesis. To enable averaging, all embeddings must have the same shape, which we achieve by scaling all images to a size of $2 2 4 \times 2 2 4$

## B. Ranker

Since the hypotheses generated from the proposer have high variance due to the subsampling of data, a second filtering step is required. To this end, [20] introduced three ranking methods. We focus on the feature-based ranker as it substantially outperformed the alternatives and is the only method that produces continuous values. Given hypothesis $h ,$ this ranker computes $R ( x , h )$ for all images $x \in A \cup B$ , where $R ( x , h )$ is the cosine similarity between the image embedding of x and the text embedding of h. Interpreting R as a binary classifier score, the resulting AUROC is used as the final score for the hypothesis. We perform ranking using a state-of-the-art SigLIP 2 Giant model [35].

![](images/75780a6f6dba6d4c58186a54080c58321fb8159ef293f7e0d9678c4434e91afb.jpg)  
Fig. 3. Object-centric set difference captioning using the two-stage, image-based approach. First, object-centric image patches are extracted from the target image set and reference image set, resulting in the target and reference patch sets A and B, respectively. Then, the proposer predicts difference hypotheses based on subsets of image patches $S _ { A }$ and $S _ { B }$ sampled from A and B. Lastly, the ranker evaluates the difference hypotheses from the proposer on the complete datasets A and ${ \bf \bar { \theta } } _ { B , }$ calculating a score for each hypothesis. Adapted from [20]

TABLE II  
RESULTS OF METHODS ON THE THREE SPLITS OF AD-DIFF BENCH, AVERAGED OVER THREE RUNS.
<table><tr><td rowspan="2">Approach</td><td rowspan="2">Proposer</td><td colspan="2">Web-scraped</td><td colspan="2">Annotation-filtered-60</td><td colspan="2">CLIP-filtered</td></tr><tr><td>Acc@1</td><td>Acc@5</td><td>Acc@1</td><td>Acc@5</td><td>Acc@1</td><td>Acc@5</td></tr><tr><td rowspan="3">Two-stage</td><td>Image-based</td><td>0.73</td><td>0.88</td><td>0.56</td><td>0.78</td><td>0.64</td><td>0.80</td></tr><tr><td>Caption-based</td><td>0.70</td><td>0.83</td><td>0.60</td><td>0.83</td><td>0.63</td><td>0.81</td></tr><tr><td>Feature-based</td><td>0.33</td><td>0.45</td><td>0.20</td><td>0.28</td><td>0.41</td><td>0.55</td></tr><tr><td>Single-stage</td><td>Image-based</td><td>0.64</td><td>0.85</td><td>0.53</td><td>0.70</td><td>0.49</td><td>0.62</td></tr></table>

## C. Single-stage

In addition to the two-stage pipeline, we propose a singlestage variant that unifies the proposer and ranker in a single inference step. This is feasible because modern VLMs can jointly process a substantially larger number of input images. After subsampling 100 images per dataset, they are given to the VLM together with a prompt detailing the task to find differences among the datasets, as well as ranking them. With this setup, the VLM generates difference hypotheses directly in one pass.

## VI. RESULTS

We present experimental results addressing four questions: (A) How do the three dataset splits compare? (B) Is a twostage approach necessary for reliable set difference captioning? (C) Which proposer performs best? (D) Can current methods handle real-world sparse and noisy set differences? (E) Can we use object-centric set difference captioning to get new insights about dataset composition?

## A. Comparison between Dataset Splits

As can be seen in Table II, accuracy scores are lowest for most approaches on the annotation-filtered split and highest on the web-scraped split. The image patches in the two splits based on autonomous driving datasets, annotation-filtered and CLIP-filtered splits, are generally of much lower resolution than those of the web-scraped split and are often further degraded by image noise, motion blur, or underexposure, making these two splits inherently more difficult than the web-scraped split. It is also possible that the VLMs used in our experiments perform worse on autonomous driving data, regardless of resolution or image quality, due to the larger domain shift from their (primarily web-scraped) training data. Regardless of which factor contributes more to the observed performance difference, the fact that data from autonomous driving datasets is generally more challenging for set-difference-captioning approaches highlights the need for a dedicated, domain-specific benchmark.

The annotation-filtered split is harder than the CLIP-filtered split not due to the images, but due to the differences described. A significant amount of differences in the annotationfiltered split are hard to recognize even for humans (e.g., “parked car” vs “car stopped with driver inside”), while the CLIP-filtered split features mostly differences that significantly affect appearance. The sets of the annotation-filtered split are, however, much cleaner than in the web-scraped and CLIPfiltered sets.

We furthermore validate the difficulty levels of the webbased split (easy, medium, hard) by comparing the accuracy of the image-based two-stage approach on these sub-splits. As expected, the acc@1 is highest for the easy split and lowest for the hard split $( 0 . 8 4 > 0 . 7 5 > 0 . 5 9 )$ .

## B. Two-stage or Single-stage Approach?

Comparing the two-stage image-based approach to the single-stage image-based approach in Table II, the two-stage approach results in much more accurate difference captions for the web-scraped and CLIP-filtered splits, but it performs similarly to the single-stage approach on the annotationfiltered split. We attribute this to a greater robustness of the two-stage approach with respect to higher sparsity of the target and reference sets. The feature-based ranker can effectively aggregate even very sparse differences across the complete dataset, which the single-stage approach cannot correctly rank from the small samples it sees.

This hypothesis is supported by the dilution experiments described in Section VI-D. The web-scraped and CLIPfiltered datasets are sparser than the annotation-filtered split, containing images that do not fit either the target or the reference set description due to the inherent limitations of search engines and CLIP-based similarity search, compared to manual annotations. In this case, the single-stage approach is significantly outperformed by the two-stage approach, but not for the very clean annotations of the annotation-filtered split. In general, the two-stage approach seems to be the more robust choice for sparse datasets from the real world. The single-stage approach might be a good choice if the goal is to quickly sample difference hypotheses, as it can be orders of magnitude faster for large datasets.

## C. Comparison between different Proposers

Comparing the different proposers in Table II, the featurebased proposer performs significantly worse than both the image-based and caption-based proposers. This is in line with previous results [20], where the feature-based proposer was able to compete with the alternatives for style differences, but not for semantic differences. Unlike the results of the VisDiff [20] approach on VisDiffBench, we did not observe a significant advantage of the caption-based approach over the image-based approach. The accuracy of these two proposers is about the same across all three splits of our dataset. We attribute this to the recent advancements in VLMs. Unlike VisDiff’s single-image VLM – requiring all images to be downscaled and composited into one – modern VLMs accept multiple images natively, making the image-based approach now the simpler and marginally more efficient choice.

## D. Noisy and Diluted Sets

Fig. 4 shows the results of our dilution experiments on the annotation-filtered-39 split of AD-Diff Bench. This split is a sub-split of the annotation-filtered-60 split, filtered to include only the image-set pairs with a dilution set large enough to achieve a concentration value below 0.001.

As expected, lower concentration values result in lower accuracy scores for all approaches. Interestingly, the accuracy remains stable until a concentration of about 0.5 (every second image is from the dilution set), before it drops rapidly for smaller concentrations. None of the tested approaches is able to reliably describe set differences for very low concentrations.

Accuracy drops much more rapidly for the single-stage approach compared to two-stage approaches. This indicates that the ranker of the two-stage approach can successfully aggregate even very sparse differences across the complete dataset, which the single-stage approach cannot correctly rank from the small sample-size it sees.

Purity has a similar influence on accuracy as concentration does, with accuracy being fairly stable up to a value of 0.5, but then approaching zero for low purity values. The exception is the single-stage proposer, which performs very poorly for low concentration values but performs well for low purity values. Even at a purity value of 0 it is still able to correctly predict the set difference in 35 % of cases. This is surprising, as the images of both sets are completely shuffled for purity 0, with the two sets being statistically identical. However, the two original sets can still be identifiable to the proposer as clusters of specific image representations within the shuffled set. These differences are averaged out in the ranker, which averages similarity scores between hypothesis descriptions and images across each subset. The ranker is therefore not able to robustly rank the ”correct” hypothesis highest if both image sets are on average identical. However, the fact that the onestage approach can retrieve the original hypothesis should not be considered an advantage, as such a difference description would be considered a false positive in any real scenario. Two image sets featuring distinct clusters of representations within the set, but no statistical differences between the sets, are identical sets from the perspective of set difference captioning.

This also highlights the advantage of accuracy for low concentration over accuracy for low purity as a metric for the realworld performance of set-difference-captioning approaches, as our results seem to indicate that it cannot be cheated as easily as low purity experiments.

## E. Application Example

Fig. 5 shows an application example in which we compare pedestrian image patches from two different cities, Singapore Queenstown and Boston Seaport. One possible application scenario is the expansion of a vehicle fleet’s operational design domain (ODD) from Boston to Singapore. The image patches are extracted from nuImages pedestrian bounding box annotations and assigned to their respective set using the location metadata available in nuImages.

We use the two-stage approach with an image-based proposer to propose and rank hypotheses for difference descriptions. As we do not have ground-truth labels for set differences in this application example, we employ a combination of dataset annotations and external sources, such as public statistics, to validate the correctness of the generated hypotheses (see Fig.5). We are able to validate all of the top-8 hypotheses for this example, demonstrating that our proposed approach is able to generate useful difference descriptions in real-world settings with sparse and diluted, a priori unknown differences.

In this specific example, hypotheses 1, 3, 5, 6, and 8 all relate to significantly higher construction activity in Singapore and the resulting presence of construction workers near roads. In the context of the example application, this could not only prompt us to evaluate how robustly our perception

![](images/4701a584c1170d9bf4fdbf41d70492e2d37d62eceabc91cc9e3397ae1357ef26.jpg)

Fig. 4. Results for noisy and diluted sets. Results of purity and concentration experiments averaged over three runs on the annotation-filtered-39 split Solid lines show top-1 accuracy (Acc@1) and dashed lines top-5 accuracy (Acc@5). Full-view plots are shown left (purity) and right (concentration), with zoomed-in plots for very low purity (top) and concentration (bottom) values stacked in the middle.  
![](images/216e112fa74f94666faa6ff8037bcd15c9a06105f502b9e4aae194827bb896c1.jpg)

![](images/88978a9846cc8facce40e37a287bcb8bbc2b80110b949095a84945e14f41c6bc.jpg)

# Difference Hypothesis AUROC ↓ Supporting Evidence   
1 people wearing hard hats 0.602 Fraction of persons labeled construction\_worker in nuImages: Singapore 12.4 %, Boston 3.4 %   
Govt. statistics, employed in construction: Singapore 12.1 % (2019) [36], Boston 2.6 % (2017) [37]   
2 outdoor scenes with greenery 0.585 Large parks (Kent Ridge Park, HortPark, etc.) in Singapore Queenstown; no large parks in Boston Seaport   
3 construction workers 0.584 See hypothesis #1   
4 people in daylight conditions 0.574 Images recorded 8pm – 6am according to nuImages annotations: Singapore 0.6 %, Boston 3.4 %   
5 people in high-visibility vests 0.551 See hypothesis #1   
6 people wearing work vests 0.547 See hypothesis #1   
7 people wearing t-shirts 0.542 Average yearly temperature: Singapore $2 7 . 8 ^ { \circ } \mathrm { C }$ [38], Boston $1 1 . 0 ^ { \circ } \mathrm { C }$ [39]   
8 people in construction zones 0.541 See hypothesis #1

Fig. 5. Application example. Geographic variation in pedestrian appearance and visual context. Comparison of pedestrian image sets from two cities using the proposed two-stage approach with an image-based proposer. The target set contains pedestrian patches from Singapore Queenstown, while the reference set contains pedestrian patches from Boston Seaport. Object-centric image patches were extracted from the nuImages dataset and assigned to each set using location annotations. The generated difference hypotheses are ranked by the AUROC score computed by the ranker. For each hypothesis, we additionally provide supporting evidence based on both nuImages annotations and external sources such as public statistics to assess its plausibility and correctness.

system detects construction workers (which may have been comparatively rare in our training dataset), but also whether the higher density of construction sites introduces additional hazards that must be considered before deploying the fleet in Singapore.

## VII. SUMMARY AND CONCLUSIONS

This paper studies object-centric set difference captioning as a practical dataset introspection tool for autonomous driving. By operating on image patches extracted from object detections, the approach avoids confounding scene-level variation and enables aggregation across large collections. To support in-domain evaluation, we introduce a benchmark with three dataset splits (web-scraped, annotation-filtered, and CLIPfiltered) and controlled purity and concentration to model noisy and sparse set differences.

Experiments indicate that autonomous driving data remains substantially more challenging than web imagery, motivating domain-specific evaluation. Across methods, a two-stage proposer-to-ranker pipeline consistently provides the most robust behavior under sparsity and noise, as the ranker can aggregate weak signals over the full datasets. Among proposers, the image-based and caption-based variants achieve similar accuracy and significantly outperform feature-based proposers.

Accuracy, however, drops sharply for very low concentration and low purity settings, so reliably identifying rare, safetycritical differences remains an open challenge. Future work should therefore focus on improving robustness to extreme sparsity.

## REFERENCES

[1] R. Greer, B. Antoniussen, A. Møgelmose, and M. Trivedi, “Languagedriven active learning for diverse open-set 3d object detection,” in Proc. Winter Conf. Appl. of Comput. Vision (WACV), 2025, pp. 980–988.

[2] K. Kim, V. Kakani, and H. Kim, “Automatic pruning and quality assurance of object detection datasets for autonomous driving,” Electron., vol. 14, no. 9, p. 1882, 2025.

[3] M. Liang et al., “Aide: An automatic data engine for object detection in autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vision and Pattern Recognition (CVPR), June 2024, pp. 14 695–14 706.

[4] L. Li et al., “Data-centric evolution in autonomous driving: A comprehensive survey of big data system, data mining, and closedloop technologies,” arXiv preprint arXiv:2401.12888, 2024. [Online]. Available: https://arxiv.org/abs/2401.12888

[5] A. Wang, A. Narayanan, and O. Russakovsky, “Revise: A tool for measuring and mitigating bias in visual datasets,” in Eur. Conf. Comput. Vision (ECCV), 2020.

[6] S. Eyuboglu et al., “Domino: Discovering systematic errors with crossmodal embeddings,” in Int. Conf. Learning Representations (ICLR), 2022.

[7] G. d’Eon, J. d’Eon, J. R. Wright, and K. Leyton-Brown, “The spotlight: A general method for discovering systematic errors in deep learning models,” in ACM Conf. Fairness, Accountability, and Transparency (FAccT), 2021.

[8] J.-B. Alayrac et al., “Flamingo: A visual language model for few-shot learning,” in Advances in Neural Inform. Process. Syst. (NeurIPS), 2022.

[9] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” arXiv preprint arXiv:2301.12597, 2023.

[10] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” arXiv preprint arXiv:2304.08485, 2023.

[11] Qwen Team, “Qwen2.5-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[12] W. Wang et al., “Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025. [Online]. Available: https://arxiv.org/abs/2508. 18265

[13] B. Li et al., “Mimic-it: Multi-modal in-context instruction tuning,” arXiv preprint arXiv:2306.05425, 2023.

[14] B. Li, Y. Zhang, L. Chen, J. Wang, J. Yang, and Z. Liu, “Otter: A multi-modal model with in-context instruction tuning,” arXiv preprint arXiv:2305.03726, 2023.

[15] L. Yao, W. Wang, and Q. Jin, “Image difference captioning with pretraining and contrastive learning,” in AAAI Conf. Artificial Intell. (AAAI), 2022.

[16] S. Chang and P. Ghamisi, “Changes to captions: An attentive network for remote sensing change captioning,” IEEE Trans. Image Process. (TIP), 2023.

[17] H. Kim, J. Kim, H. Lee, H. Park, and G. Kim, “Viewpoint-agnostic change captioning with cycle consistency,” in IEEE/CVF Int. Conf. Comput. Vision (ICCV), 2021.

[18] D. H. Park, T. Darrell, and A. Rohrbach, “Robust change captioning,” in IEEE/CVF Int. Conf. Comput. Vision (ICCV), 2019.

[19] R. Zhong, C. Snell, D. Klein, and J. Steinhardt, “Describing differences between text distributions with natural language,” in Int. Conf. Mach. Learning (ICML), 2022.

[20] L. Dunlap et al., “Describing differences in image sets with natural language,” in IEEE/CVF Conf. Comput. Vision and Pattern Recognition (CVPR), 2024.

[21] Z. Zhu, W. Liang, and J. Zou, “Gs-clip: A framework for explaining distribution shifts in natural language,” in Int. Conf. Mach. Learning (ICML) DataPerf Workshop, 2022.

[22] A. Torralba and A. Efros, “Unbiased look at dataset bias,” in IEEE/CVF Conf. Comput. Vision and Pattern Recognition (CVPR), 2011.

[23] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar, “Do imagenet classifiers generalize to imagenet?” in Int. Conf. Mach. Learning (ICML), 2019.

[24] D. Hendrycks et al., “The many faces of robustness: A critical analysis of out-of-distribution generalization,” in IEEE/CVF Int. Conf. Comput. Vision (ICCV), 2021.

[25] W. Liang and J. Zou, “Metashift: A dataset of datasets for evaluating contextual distribution shifts and training conflicts,” in Int. Conf. Learning Representations (ICLR), 2022.

[26] P. W. Koh et al., “Wilds: A benchmark of in-the-wild distribution shifts,” in Int. Conf. Mach. Learning (ICML), 2021.

[27] T. Sun et al., “Shift: A synthetic driving dataset for continuous multitask domain adaptation,” in Proc. IEEE/CVF Conf. Comput. Vision and Pattern Recognition (CVPR), June 2022, pp. 21 371–21 382.

[28] J. Vendrow, S. Jain, L. Engstrom, and A. Madry, “Dataset interfaces: Diagnosing model failures using controllable counterfactual generation,” arXiv preprint arXiv:2302.07865, 2023. [Online]. Available: https://arxiv.org/abs/2302.07865

[29] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” Int. J. of Robotics Res. (IJRR), 2013.

[30] Motional. (2020) nuimages dataset. [Online]. Available: https://www. nuscenes.org/nuimages

[31] P. Sun et al., “Scalability in perception for autonomous driving: Waymo open dataset,” in IEEE/CVF Conf. Comput. Vision and Pattern Recognition (CVPR), 2020.

[32] “Waymo open dataset: An autonomous driving dataset,” 2019-2025. [Online]. Available: https://www.waymo.com/open

[33] G. Ilharco et al., “Openclip,” Jul. 2021. [Online]. Available: https://doi.org/10.5281/zenodo.5143773

[34] OpenAI, “gpt-oss-120b & gpt-oss-20b model card,” arXiv preprint arXiv:2508.10925, 2025. [Online]. Available: https://arxiv.org/abs/2508. 10925

[35] M. Tschannen et al., “Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,” arXiv preprint arXiv:2502.14786, 2025.

[36] Ministry of Manpower, Manpower Research and Statistics Department, Singapore Manpower Statistics in Brief 2020, 2020. [Online]. Available: https://stats.mom.gov.sg/Pages/Manpower-Statistics-In-Brief-2020.aspx

[37] Boston Planning & Development Agency, Research Division, Boston’s Economy 2019, 2019. [Online]. Available: https://www.boston.gov/sites/default/files/file/2023/02/ Economy-Report 2019 Online-Version 9-9-19.pdf

[38] National Environment Agency, Meteorological Service Singapore. [Online]. Available: https://www.weather.gov.sg/ climate-climate-of-singapore/

[39] NOAA National Centers for Environmental information, “Climate at a glance: City time series,” 2026. [Online]. Available: https://www.ncei. noaa.gov/access/monitoring/climate-at-a-glance/city/time-series