# Scaling E-Commerce Attribute Extraction with Parallel Decoding

Nikhita Vedula Dushyanta Dhyani Bryan Wang Shervin Malmasi

Amazon.com, Inc.

{veduln, dhyanidd, brywan, malmasi}@amazon.com

## Abstract

Customers rely on specific product attributes to compare products and make purchasing decisions, but e-commerce catalogs are messy and unstructured, making it difficult to identify which attributes matter most and extract them at scale. Standard Attribute Value Extraction (AVE) systems treat all attributes equally, producing large, inconsistent attribute sets that do not reflect the factors consumers use to differentiate products. We introduce a two-stage LLM pipeline that first discovers a compact, ranked schema of purchase-discriminative attributes for each product category, then extracts their values from catalog text using a fine-tuned compact LLM (Qwen3-4B) with Hyper-Parallel Decoding (HPD). This pipeline achieves 85% extraction accuracy, on par with the foundational LLM it was distilled from, while reducing inference costs by 92% over foundational LLMs, enabling production-scale use for prod uct discovery and catalog enrichment. The resulting category-level structured representations effectively constitute automatically constructed product knowledge bases, providing consistent, comparable attributes across varied product categories that can ground downstream knowledge-intensive applications.

## 1 Introduction

Attribute Value Extraction is a core capability for e-commerce systems. Structured product attributes power search, filtering, comparison, recommendation, and product quality workflows (Yang et al., 2022; Brinkmann et al., 2023). However, e-commerce catalogs are highly messy as product information is spread across titles, descriptions, bullets, and semi-structured fields; sellers use inconsistent terminology; and values are often missing, implicit, duplicated, or expressed in non-standard forms (Khandelwal et al., 2023; Yang et al., 2022). Recent work has increasingly explored generative approaches to AVE, where encoder-decoder models and LLMs extract attribute values directly from product text (Blume et al., 2023; Shinzato et al., 2023; Khandelwal et al., 2023). These methods handle heterogeneous inputs and flexible output schemas, but deploying them at e-commerce scale introduces two major challenges. First, defining high-quality schemas manually across thousands of product categories is expensive and difficult to maintain (Huang et al., 2025; Xu et al., 2023). Second, extracting values over massive catalogs is computationally costly, especially when products are added daily and schemas evolve with downstream needs (Chen et al., 2023; Yang et al., 2024; Zhang et al., 2024).

We focus on extracting customer-relevant, purchase-discriminative attributes for each product category. Unlike standard AVE formulations that target broad attribute coverage or extract values for predefined attributes (Yang et al., 2022; Xu et al., 2023), our goal is to identify the attributes most useful for differentiating products within a category, such as wattage for blenders, capacity for storage devices, or noise cancellation for headphones, and ensure all products in the same category share the same schema, making their structured representations directly comparable. Therefore, our pipeline automatically induces a domain ontology (the category-level schemas) and populates it at scale, constructing a structured product knowledge base without manual schema engineering.

We propose a scalable two-stage LLM pipeline. In the first stage, we automatically discover category-specific schemas from representative catalog examples using a large, foundational LLM (Claude), then standardize semantically similar attributes across categories. In the second stage, we extract values for the discovered schemas from product catalog text using a fine-tuned small language model with Hyper-Parallel Decoding (HPD). HPD exploits the conditional independence of attribute values to decode multiple values in parallel (Glavas et al., 2026b), substantially improving throughput.

We evaluate the system on a large proprietary e-commerce catalog spanning thousands of categories. Our schema discovery stage produces attributes judged correct or relevant in 89.6% of cases, and our fine-tuned value extraction model achieves approximately 85% extraction accuracy, matching the foundational LLM teacher model despite a >100× reduction in model parameters. The proposed system reduces inference cost by 92%, making repeated catalog-scale processing practical. Our main contributions are:

(i) A two-stage pipeline that automatically discovers compact, category-level attribute schemas and extracts their values at scale, requiring no manual schema engineering.

(ii) A practical, large-scale use case combining knowledge distillation into a compact LLM (Qwen3-4B) with Hyper-Parallel Decoding, processing hundreds of millions of products across multiple marketplaces while achieving 92% cost reduction over a foundational LLM baseline.

(iii) A comprehensive evaluation covering schema quality, extraction accuracy, quantity understanding, determinism, and efficiency, with detailed cost and throughput analysis.

## 2 Related Work

Product attribute-value extraction. Attribute Value Extraction (AVE) for e-commerce recovers structured product attributes from noisy catalog content to support search, recommendation, comparison, and catalog enrichment (Yang et al., 2022; Brinkmann et al., 2023). Prior work has formulated AVE as sequence tagging (Zheng et al., 2018) or question answering over product context (Wang et al., 2020). These approaches improve extraction for predefined attributes, but typically assume available schemas and can become costly across many attributes and categories.

Generative AVE. Recent work has explored generative AVE, using encoder-decoder or LLM-based models to produce attribute values directly from product text (Blume et al., 2023; Shinzato et al., 2023; Khandelwal et al., 2023; Brinkmann et al., 2023). These methods handle heterogeneous inputs and flexible output formats, but production-scale deployment still requires scalable schema construction and efficient extraction over large catalogs. Our work targets both challenges through automatic schema discovery and low-cost generative extraction.

Schema discovery and knowledge base construction. Open-world attribute mining and product schema modeling aim to discover or maintain product attributes with limited human supervision (Xu et al., 2023; Huang et al., 2025). More broadly, automated knowledge base construction encompasses the extraction, integration, and maintenance of structured knowledge from unstructured sources (Weikum et al., 2021), with recent work leveraging LLMs to reshape the classical pipeline of ontology engineering, knowledge extraction, and knowledge fusion (Bian, 2025). In the e-commerce domain, Hongwimol et al. (2026) present a multiagent LLM framework that induces product types and attribute keys on demand and consolidates them into a globally consistent product knowledge graph, while Peshevski et al. (2025) automate ontology creation and KG population from unstructured product descriptions without predefined schemas. Our work shares the goal of automated product KB construction but differs in emphasis: rather than maximizing broad attribute coverage, we seek compact, purchase-discriminative category-level schemas and prioritize extreme extraction efficiency through parallel decoding, enabling deployment at scales of hundreds of millions of products.

Efficient and scalable extraction. Scalability is a central challenge for AVE because real-world products are associated with many attributes. Some QA-style approaches repeatedly process the same product context for different target attributes (Chen et al., 2023). Intra-prompt parallel decoding (Glavas et al., 2026a) addresses this commoncontext bottleneck by stacking multiple questions into a single prompt and decoding their answers simultaneously through attention mask manipulation. Recent work improves efficiency by caching context representations and using lightweight attributecontext interactions (Yang et al., 2024), while “lifelong” AVE addresses the need to adapt to evolving products, categories, and attributes (Zhang et al., 2024). Our extraction stage is complementary: we specialize a small language model for structured value extraction and use Hyper-Parallel Decoding (HPD) to decode multiple conditionally independent attribute values from the same product context in parallel (Glavas et al., 2026b).

## 3 Scalable Schema-Guided Attribute Value Generation

An e-commerce catalog contains billions of products, each with a unique set of attributes and values. Extracting these attribute-value pairs for each product in isolation can produce a large number of semantically similar but differently named attributes across the catalog, making them both unmanageable and difficult to use downstream. We propose a two-stage pipeline where we first identify a compact set of key purchase-discriminative attributes for each product category, and subsequently extract values corresponding to each attribute for every product in that category. Note that the choice of categorization granularity is important: a broad categorization with few categories will result in extremely generic attributes, while an overly granular categorization can result in a massive attribute set size increase.

## 3.1 Problem Statement

Let P denote a large-scale e-commerce product catalog, categorized into $| \tau |$ high-level categories. Each product $p \in \mathcal { P }$ has catalog context $\mathbf { x } _ { p }$ comprising its title, description, and other unstructured product details, and is associated with a category $t ( p ) \in \mathcal T$ . We seek to produce, for each product $\underline { { p } } .$ a structured representation ${ \phi } ( p ) ~ = ~ \{ ~ ( a _ { i } , ~ v _ { i } ) \} _ { i = 1 } ^ { N }$ where $\mathcal { A } _ { t ( p ) } = \{ a _ { 1 } , . . . , a _ { N } \}$ is a schema of $N$ purchase-discriminative attributes defined at the category level, and each $v _ { i }$ is the value of attribute $a _ { i }$ extracted from $\mathbf { x } _ { p }$ if available. This decomposes into two sub-problems:

Attribute Schema Discovery: For each category $t \in \mathcal T$ , identify a schema $\mathcal { A } _ { t } = \{ a _ { 1 } , . . . , a _ { N } \}$ of attributes that are most informative for consumer purchase decisions within that category.

Attribute Value Generation: For each product $p \in { \mathcal { P } } .$ , given the schema $\boldsymbol { \mathcal { A } } _ { t ( \boldsymbol { p } ) }$ and the catalog context $\mathbf { x } _ { p } .$ , extract the value $v _ { i }$ of each attribute $a _ { i } \in \mathcal { A } _ { t ( p ) }$ from $\mathbf { x } _ { p }$ . The key design constraint is category-level schema sharing: all products belonging to the same category t share the same attribute schema $\boldsymbol { A } _ { t }$ , ensuring that $\phi ( p )$ and $\phi ( p ^ { \prime } )$ are comparable for any two products $p , p ^ { \prime }$ with $t ( p ) = t ( p ^ { \prime } )$

## 3.2 Stage 1: Attribute Schema Discovery

Category-Specific Attributes Specification For each category, we randomly select k representative products that satisfy a minimum threshold of customer engagement metrics, ensuring a balanced selection of quality and diversity. We then use the Claude Sonnet 4 LLM to identify M key attributes for the category from the title and description of each product. The prompts are tuned to also extract each attribute’s description, data type (numerical, categorical, single-value, multi-valued, or free-form text), and a list of standardized values the attribute can take for the given category.

Global Attributes Specification While each product has its unique set of attributes (e.g., noise canceling ability for headphones or storage capacity for a hard disk), certain attributes are universally applicable across all products. We supplement the category-specific attributes with the following set of global attributes: total pack size, total weight, total volume, per product weight, per product volume.

Attribute Name Standardization While defining a fixed set of M attributes per category constrains the local schema, running Stage 1 across $P$ categories can produce $M \times P$ attributes (which can range from a few to tens of thousands) with semantic redundancy (e.g., Water Resistance, Waterproof rating, $I P$ Water Rating). We develop an LLM-based clustering approach to standardize these. We embed all attribute names using a Qwen3-8B embedding model and apply hierarchical agglomerative clustering to group semantically similar attributes. Each cluster is then assigned a canonical name via one of two methods: a) Heuristic: replacing all cluster members with the most frequent representative name, provided it satisfies a minimum similarity threshold; or b) LLM-based: using an LLM to generate a standardized name while preserving domain-specific context.

## 3.3 Stage 2: Parallel Attribute Value Generation

Once a standardized attribute set has been generated, the next step is to extract values for those attributes from all catalog products. A straightforward approach is to prompt a foundational LLM with the predefined attributes and product catalog information. While state-of-the-art LLMs can accomplish this with high accuracy, scaling to hundreds of millions of products in a cost-effective way poses a significant challenge. Even at an optimistic throughput of 100K products per hour, processing a complete catalog would take multiple weeks, which does not scale for a real-world e-commerce service.

![](images/14eb1b1dd5f107ebdad517affb5b637b58fdbf89af51c83f9b97b5068548e446.jpg)  
Figure 1: Overview of our two-stage pipeline for scalable schema-guided attribute value generation. Stage 1 discovers a compact schema of N purchase-discriminative attributes per category using a large foundational LLM. Stage 2 extracts attribute values from product catalog text using a fine-tuned compact LLM with Hyper-Parallel Decoding, generating all N values simultaneously.

Given a product context and a fixed schema of N attributes, standard autoregressive decoding generates the output JSON sequentially, requiring approximately $\sum _ { i = 1 } ^ { \bullet } K _ { i }$ decoding steps, where $K _ { i }$ is the token length of the value for attribute i.

Hyper-Parallel Decoding (HPD): Efficiently Scaling Inference Throughput and Cost In schema-guided AVE, many attribute values are conditionally independent given the same product context. HPD (Glavas et al., 2026b) exploits this structure by decoding values for multiple attributes in parallel, reducing decoding steps from $\textstyle \sum K _ { i }$ to approximately max $K _ { i }$ . Standard autoregressive decoding generates attribute-value pairs sequentially, attending to all previously generated values before producing the next. HPD observes that conditioned on the same product context $x _ { p } ,$ the values of different attributes are conditionally independent: extracting a blender’s wattage should not causally depend on having first extracted its color. This conditional independence allows HPD to extract values in parallel during decoding.

The mechanism works by constructing a JSON skeleton template with attribute keys and blank value fields, creating position ID “gaps” of size $K _ { \mathrm { m a x } }$ at each value location. Special BOV (beginning-of-value) tokens mark where generation should occur. During the first inference step, the model outputs next-token probabilities for the entire input, and HPD selects N probabilities at the BOV positions to generate the first token of each value simultaneously. Subsequent tokens are appended to the end of the sequence but assigned position IDs that logically place them in the previously created gaps, maintaining key-value cache functionality. A position-based causal attention mask ensures tokens only attend to appropriate context despite being arranged out of order in physical memory. In subsequent steps, only the N new tokens are passed to the model with the cached keys/values, generating the t-th token for all values in parallel until completion. HPD further stacks J documents in a single prompt, decoding $J \times N$ tokens per step; combined with batch inference (batch size b), this yields $b \times J \times N$ tokens per step, achieving large efficiency gains.

HPD requires custom fine-tuning for parallel decoding. BOV tokens are added to the vocabulary, and a “block $\mathbf { I D } ^ { \prime \prime }$ tensor tracks which inference step each token belongs to (block 0 for the prompt, block 1 and above for generated tokens). Position IDs are manipulated during training to match the inference configuration, and a custom 2D attention mask enforces that prompt tokens only attend to the prompt while generated tokens attend to themselves and lower block IDs. This enables the model to match autoregressive quality while maintaining ${ \sim } 1 0 \times$ speedup during inference.

Attribute Value Standardization A key challenge in large-scale attribute extraction is ensuring that extracted values are consistent and comparable across products, categories, and marketplaces. Without standardization, the same attribute can produce highly variable surface forms (e.g., “30 oz”, “30 ounces”, “30oz”, “thirty ounces”) that are difficult to use in downstream applications requiring exact matching or aggregation. We address this at multiple levels: Stage 1 defines expected data types and standardized categorical values for each attribute; Stage 2 prompts instruct the model to adhere to type requirements, output numerical values with standard units, select from predefined categorical options, and produce comma-separated lists for multi-valued attributes; and a strictly structured JSON output format further reduces variability across runs.

We additionally experiment with Constrained Hyper-Parallel Decoding at inference time, where token-level constraints force the model to generate only valid data types and standardized categorical values. For categorical attributes, a logit mask is applied at each decoding step to permit only tokens corresponding to valid options from the Stage 1 schema; for numerical attributes, a regex expression enforces the correct number-unit format (e.g., allowing only digit and unit tokens in sequence). In practice, we find that the combination of careful prompt engineering and knowledge distillation from a foundational LLM during fine-tuning already yields high standardization compliance, with constrained decoding affecting fewer than 1% of extracted values while providing a formal guarantee of format adherence.

## 4 Experiments and Results

## 4.1 Experimental Setup

Product Catalog and Attribute Schema Inventory. We perform experiments with our pipeline on a large proprietary e-commerce catalog containing more than 30M products spanning a few thousand product categories. Each product is represented by catalog text and metadata, including title, description, and other structured or semi-structured product details. Stage 1 produces category-level key attribute schemas, which are then standardized across categories. The resulting schema inventory contains several thousand unique attributes. Across the evaluated catalog, these attributes correspond to tens of millions of observed attribute values, including numerical, categorical, and free-form values.

Training data generation. To train the Stage 2 extraction model, we construct a supervised dataset of 100K product-schema examples sampled across product categories. For each example, we prompt the Claude Sonnet 4 LLM with the product catalog context and corresponding Stage 1 attribute schema to produce a structured JSON object containing the extracted value for each attribute, or null when absent. These outputs serve as training targets for HPD fine-tuning.

Models and HPD configuration. We fine-tune Qwen3-4B and Qwen3-8B models for purchasediscriminative attribute value generation using Hyper-Parallel Decoding. Each product is paired with a fixed category-level schema, and the model generates one value slot per attribute in structured JSON format. We use up to 20 attributes per product with $K _ { \mathrm { m a x } } = 1 0 0$ tokens and 6 input products per prompt. Fine-tuning is performed on an AWS EC2 p4de.24xlarge instance with 8 A100 GPUs.

At inference time, HPD decodes values for all attributes in parallel, rather than generating the JSON output strictly left-to-right. We compare against a foundational LLM baseline accessed via cloudhosted batched API inference, which processes requests asynchronously with variable queue wait times.

Evaluation. We comprehensively evaluate both stages using automated and human evaluation. For Stage 1, we use Claude Sonnet 4.5 as an LLM judge to classify discovered attributes as correct/relevant, irrelevant, or too vague/too specific. For Stage 2, we use the same LLM judge to evaluate extraction quality on 20K products, classifying outputs as: correct extraction, correct null (attribute absent and model correctly returns null), incorrect extraction, missed extraction (value present but model returns null), or ungrounded extraction. We additionally perform human evaluation on 500 products. Finally, to assess determinism, we run the complete pipeline five times on 50 products from diverse categories and compare outputs across runs.

Attribute Schema Discovery Quality We observe that 89.6% of the discovered key attribute names are classified as “correct” or relevant by the LLM judge with respect to the category, 4.0% are deemed irrelevant, and 6.4% are considered either too vague or too specific for the given category (e.g., “Performance” is overly vague as it lacks discriminative specificity, while “Bluetooth Codec

<table><tr><td>Classification</td><td>% of Attributes</td></tr><tr><td>Correct / Relevant</td><td>89.6%</td></tr><tr><td>Too vague or too specific</td><td>6.4%</td></tr><tr><td>Irrelevant</td><td>4.0%</td></tr></table>

Table 1: Stage 1 attribute schema quality evaluated by an LLM judge across a few thousand product categories (20 attributes per category).

Support” is too specific for a broad “Audio Equipment” category). Table 1 summarizes the schema quality evaluation. This assessment is conducted at the category level across a few thousand categories, covering 20 key attributes per category.

Attribute Value Generation Quality As shown in Table 2, the accuracy of key attribute extraction is very similar for both fine-tuned models, at approximately 85%. Both Qwen3 models finetuned with HPD achieve extraction accuracy on par with or slightly exceeding the foundational teacher LLM, demonstrating that knowledge distillation combined with task-specific fine-tuning can fully close the quality gap despite a >100× reduction in model parameters. Notably, HPD introduces no quality degradation while providing significant speed gains.

Of the remaining approximately 15%, the LLM judge flags 4.6% as potentially ungrounded extractions (values not directly verifiable from the explicit input text), 3.5% as missed extractions, and 2% as incorrect extractions. Manual investigation of 50 such cases, corroborated by our 500- product human evaluation, reveals that approximately half of ungrounded cases are reasonable inferences from implicit context (e.g., inferring “plastic” from “BPA-free”), bringing the effective ungrounded extraction rate to approximately 2.3%. Additionally, fewer than 2% involve format violations where the model does not conform to Stage 1 type constraints. The remaining 3-4% include ambiguous cases due to catalog noise, where the LLM judge cannot make a definitive determination, and are excluded from evaluation.

Interestingly, the larger Qwen3-8B model performs slightly worse than Qwen3-4B. Our practical constraints - processing several hundred million products with daily incremental and monthly full catalog refreshes - require sustained high throughput on commodity GPU instances, where models exceeding 8B parameters significantly reduce per-GPU batch capacity. We therefore focus on models at or below 8B parameters and rely on knowledge distillation to close any quality gap.

Determinism and Coverage. To assess reliability, we run both stages five times on 50 products from diverse categories. For both schema discovery and value extraction, approximately 93% of outputs are identical or semantically equivalent across runs, with variations primarily in subjective attributes such as “special features.” Overall pipeline coverage is 99.997%: attributes are successfully generated for all products except those lacking catalog information or with incorrect category assignments (0.003% of products).

Human Evaluation We perform human evaluation on 500 products spanning diverse categories, assessing correctness of extracted values against input catalog text for both categorical and numerical attributes. Human judges rate 85% of extractions as correct, consistent with the LLM judge evaluation, with Qwen3-4B again performing slightly better than Qwen3-8B. Overall, the LLM judge and human annotators agree on 91% of generated attribute values. Of the errors, approximately 6% are incorrect extractions; the remainder are primarily data type violations and, for numerical quantity attributes, cases where the attribute information is present only in product images rather than the catalog text provided as input.

Efficiency and Throughput Table 3 compares the cost and throughput of our HPD-based extraction against both standard autoregressive decoding on the same model and the foundational LLM baseline using cloud-hosted batched API inference. HPD achieves an average throughput of ∼159K products per hour per instance. Running inference on 30M products using 5 parallel instances takes approximately 36 hours at a total cost of ∼\$5,000. Standard autoregressive decoding on the same Qwen3-4B model yields approximately 18K products per hour per instance (∼9× lower throughput), requiring over 330 hours on the same hardware at a cost of ∼\$46K.

The foundational LLM baseline, accessed via cloud-hosted batched API inference with 10 parallel batch jobs, achieves a nominal throughput of ∼100K products per hour excluding queue wait times. In practice, variable queue delays add significant additional latency, making the effective wallclock time substantially longer than the reported 300 hours. The total API cost for 30M products is

<table><tr><td>Model</td><td>Total Correct</td><td>Correct Extractions</td><td>Correct Null Extractions</td><td>Incorrect Extractions</td><td>Missed Extractions</td></tr><tr><td>Foundational LLM Baseline</td><td>84.80%</td><td>54.30%</td><td>30.50%</td><td>1.30%</td><td>4.00%</td></tr><tr><td>Qwen3-4B (AR)</td><td>84.60%</td><td>51.80%</td><td>32.80%</td><td>2.00%</td><td>3.80%</td></tr><tr><td>Qwen3-4B (HPD)</td><td>85.40%</td><td>52.40%</td><td>33.00%</td><td>1.70%</td><td>3.50%</td></tr><tr><td>Qwen3-8B (HPD)</td><td>84.40%</td><td>49.60%</td><td>34.80%</td><td>2.60%</td><td>3.50%</td></tr></table>

Table 2: Stage 2 key attribute value generation quality for fine-tuned models, evaluated by an LLM judge (Claude Sonnet 4.5) on a sample of 20K products. Bold indicates the best result per column (highest for correct metrics, lowest for incorrect/missed).

<table><tr><td></td><td>Qwen3-4B (HPD)</td><td>Qwen3-4B (AR)</td><td>Foundational LLM</td></tr><tr><td>Throughput (products/hr)</td><td>159K</td><td>18K</td><td>100K†</td></tr><tr><td>Parallel instances / jobs</td><td>5</td><td>5</td><td>10</td></tr><tr><td>Time for 30M products</td><td>36 hrs</td><td>333 hrs</td><td>300+ hrs†</td></tr><tr><td>Inference cost Cost reduction vs.</td><td>$5K</td><td>$46K</td><td>$63K</td></tr><tr><td>Foundational LLM</td><td>92%</td><td>27%</td><td></td></tr></table>

<sup>†</sup> Excludes variable queue wait times, which add significant latency.

Table 3: Efficiency comparison for 30M products between HPD inference, standard autoregressive (AR) inference, and foundational LLM batched API inference.

∼\$63,000, representing a 92% cost reduction with HPD. Fine-tuning the Qwen3-4B model requires 8 hours on a single instance with 8 A100 GPUs, at a one-time cost of ∼\$350.

Effect of Attribute Schema Size on Parallelism. HPD decodes all N attribute values simultaneously, generating J × N tokens per inference step. Since decoding steps remain fixed at $K _ { \mathrm { m a x } }$ regardless of N, throughput in attribute-values per hour scales approximately linearly with schema size. We validate this by varying $N \in \{ 1 0 , 1 2 , 1 6 , 1 8 , 2 0 \}$ and observe that product throughput remains approximately constant while attribute-values extracted per hour increases proportionally. Meanwhile, the foundational LLM baseline cost scales linearly with N, yielding progressively larger cost reductions from approximately 84% at N = 10 to 92% at N = 20, making our choice of N = 20 nearoptimal for both parallelism and category coverage.

## 5 Practical Use Case Evaluation

We employ our full two-stage pipeline to process product catalogs across two major marketplaces, extracting key attributes for hundreds of millions of products. Large-scale inference is distributed across 4–20 A100 GPU instances, each processing products independently to achieve linear throughput scaling. The pipeline is containerized for reproducibility across instance types.

Given that new products appear in the catalog daily and product content is frequently updated, we plan a tiered refresh strategy. New products are processed on a daily or weekly cadence, while the full catalog is re-processed monthly to capture updates. Schema discovery (Stage 1) is refreshed less frequently as category-level attributes are relatively stable; schemas are versioned per refresh cycle and validated against the previous version before deployment. For Stage 2, outputs that fail JSON parsing or violate data type constraints are caught by automated post-processing validation and re-processed in the subsequent batch.

The extracted attribute-value pairs are published to an internal large-scale knowledge base and made queryable by downstream use cases. These structured representations can enable several applications: (i) product comparison via shared categorylevel schemas; (ii) catalog enrichment, where extracted values fill gaps in incomplete product listings and augment sparse catalog metadata; (iii) catalog quality improvement by surfacing inconsistencies in product details; and (iv) personalized product understanding through compact structured inputs to recommendation and search systems.

We additionally validate our outputs through live seller interviews across two marketplaces, where sellers evaluate the extracted attributes for quality and usefulness in differentiating products and informing customer purchase decisions. Sellers provide positive feedback, confirming that the extracted attributes reflect factors they consider when positioning their products relative to competitors.

## 5.1 Sample System Output

We illustrate the end-to-end pipeline output for the product category COFFEE\_MAKER. Stage 1 discovers the following schema of purchase-discriminative attributes:

Given this schema and the catalog text of a specific product (title, description, and bullet points), Stage 2 extracts the following structured attribute-

<table><tr><td>Attribute</td><td>Data Type</td><td>Standardized Values</td></tr><tr><td>Brew Type</td><td>categorical</td><td>Manual, Semi-Automatic, Automatic, Super-Automatic, Pod</td></tr><tr><td>Pump Pressure</td><td>numerical</td><td>(bars)</td></tr><tr><td>Heating System</td><td>categorical</td><td>Thermocoil, Single Boiler, Dual Boiler, Thermoblock</td></tr><tr><td>Temperature Control</td><td>categorical</td><td>PID, Thermostat, Digital</td></tr><tr><td>Grinder Type</td><td>categorical</td><td>Conical Burr, Flat Burr, Blade, None</td></tr><tr><td>Tank Capacity</td><td>numerical</td><td>(oz)</td></tr><tr><td>Milk System</td><td>categorical</td><td>Steam Wand, Auto Frother, None</td></tr><tr><td>Build Material</td><td>categorical</td><td>Stainless Steel, Plastic, Aluminum</td></tr></table>

Table 4: Stage 1 output: discovered attribute schema for COFFEE\_MAKER.

value pairs:

{"Brew Type": "Semi-Automatic",   
"Pump Pressure": "15 bars",   
"Heating System": "Thermocoil",   
"Temperature Control": "PID",   
"Grinder Type": "Conical Burr",   
"Tank Capacity": "67 oz",   
"Milk System": "Steam Wand",   
"Build Material": "Stainless Steel"}

All products in the COFFEE\_MAKER category share this same schema, making their structured representations directly comparable. For example, a budget pod machine would yield {“Brew Type”: “Pod”, “Heating System”: “Thermoblock”, “Grinder Type”: “None”, “Build Material”: “Plastic”, ...}, enabling direct attribute-level comparison.

## 6 Conclusion

We presented a scalable two-stage pipeline for extracting purchase-discriminative attributes from large e-commerce catalogs. The first stage automatically discovers compact, category-level schemas using a foundational LLM, eliminating manual schema engineering across thousands of categories. The second stage extracts attribute values at scale using a fine-tuned Qwen3-4B model with HPD, which exploits the conditional independence of attribute values to decode them simultaneously. The system achieves 85% extraction accuracy, on par with the foundational teacher LLM, while reducing inference costs by 92%. Future directions include incorporating multimodal inputs such as product images to capture visually conveyed attributes, and extending the pipeline to multilingual catalogs.

## Limitations

Our system extracts attributes exclusively from textual catalog content such as product titles, descriptions, and bullet points. Product information that is conveyed only through images, such as visual design details, color variations, or quantities shown in packaging photos, is not captured by our current approach. Incorporating multimodal inputs to address this gap is left for future work. The current pipeline has been evaluated only on Englishlanguage catalogs. Extending the system to non-English marketplaces would require multilingual fine-tuning and potentially different schema discovery prompts, which we have not yet explored. The conditional independence assumption underlying HPD assumes that extracting the value of one attribute should not depend on previously extracted values. While this holds for most attributes in practice, certain attributes may exhibit correlations (e.g., product weight and volume, or material and durability). Our empirical results suggest this does not significantly impact extraction quality, but a deeper investigation into correlated attribute groups is warranted. Finally, we evaluated our fine-tuned models exclusively using the Qwen3 model family in 4B and 8B parameter sizes. Other model architectures or larger model sizes may yield different qualityefficiency tradeoffs, and the optimal model choice may vary across domains or catalog characteristics.

## References

Haonan Bian. 2025. Llm-empowered knowledge graph construction: A survey. arXiv preprint arXiv:2510.20345.

Ansel Blume, Nasser Zalmout, Heng Ji, and Xian Li. 2023. Generative models for product attribute extraction. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 575–585, Singapore. Association for Computational Linguistics.

Alexander Brinkmann, Roee Shraga, and Christian Bizer. 2023. Extractgpt: Exploring the potential of large language models for product attribute value extraction. arXiv preprint arXiv:2310.12537.

Wei-Te Chen, Keiji Shinzato, Naoki Yoshinaga, and Yandi Xia. 2023. Does named entity recognition truly not scale up to real-world product attribute extraction? In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 152–159, Singapore. Association for Computational Linguistics.

Theodore Glavas, Nikhita Vedula, Dushyanta Dhyani, Antonios Valkanas, Yilun Zhu, and Shervin Malmasi. 2026a. Intra-prompt parallel decoding for common-context question answering. Preprint, arXiv:2609.05707.

Theodore Glavas, Nikhita Vedula, Dushyanta Dhyani, Yilun Zhu, and Shervin Malmasi. 2026b. Breaking the autoregressive chain: Hyper-parallel decoding

for efficient llm-based attribute value extraction. In Findings of the Association for Computational Linguistics: ACL 2026, pages 36792–36808.

Pollawat Hongwimol, Haoning Shang, Chutong Wang, Zhichao Wan, Yi Gao, Yuanming Li, Lin Gui, Wenhao Sun, and Cheng Yu. 2026. AutoPKG: An automated framework for dynamic e-commerce productattribute knowledge graph construction. In Findings of ACL.

Yunhan Huang, Klevis Ramo, Andrea Iovine, Melvin Monteiro, Sedat Gokalp, Arjun Bakshi, Hasan Turalic, Arsh Kumar, Jona Neumeier, Ripley Yates, Rejaul Monir, Simon Hartmann, Tushar Manglik, and Mohamed Yakout. 2025. AttributeForge: An Agentic LLM Framework for Automated Product Schema Modeling.

Anant Khandelwal, Happy Mittal, Shreyas Kulkarni, and Deepak Gupta. 2023. Large scale generative multimodal attribute extraction for e-commerce attributes. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 5: Industry Track), pages 305–312, Toronto, Canada. Association for Computational Linguistics.

Dimitar Peshevski, Riste Stojanov, and Dimitar Trajanov. 2025. Ai agent-driven framework for automated product knowledge graph construction in ecommerce. In Proceedings ofthe 1st GOBLIN Workshop on Knowledge Graph Technologies.

Keiji Shinzato, Naoki Yoshinaga, Yandi Xia, and Wei-Te Chen. 2023. A unified generative approach to product attribute-value identification. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 6599–6612, Toronto, Canada. Association for Computational Linguistics.

Qifan Wang, Li Yang, Bhargav Kanagal, Sumit Sanghai, D. Sivakumar, Bin Shu, Zac Yu, and Jon Elsas. 2020. Learning to extract attribute value from product via question answering: A multi-task approach. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

Gerhard Weikum, Xin Luna Dong, Simon Razniewski, and Fabian Suchanek. 2021. Machine knowledge: Creation and curation of comprehensive knowledge bases. Foundations and Trends in Databases.

Liyan Xu, Chenwei Zhang, Xian Li, Jingbo Shang, and Jinho D. Choi. 2023. Towards open-world product attribute mining: A lightly-supervised approach. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12223–12239, Toronto, Canada. Association for Computational Linguistics.

Li Yang, Qifan Wang, Jianfeng Chi, Jiahao Liu, Jingang Wang, Fuli Feng, Zenglin Xu, Yi Fang, Lifu Huang, and Dongfang Liu. 2024. Eave: Efficient product attribute value extraction via lightweight sparse-layer

interaction. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 1491– 1505, Miami, Florida, USA. Association for Computational Linguistics.

Li Yang, Qifan Wang, Zac Yu, Anand Kulkarni, Sumit Kumar Sanghai, Bin Shu, Jon Elsas, and Bhargav Kanagal. 2022. Mave: A product dataset for multi-source attribute value extraction. In Proceedings ofthe Fifteenth ACM International Conference on Web Search and Data Mining, pages 1256–1265.

Tao Zhang, Chenwei Zhang, Xian Li, Jingbo Shang, Hoang Nguyen, and Philip Yu. 2024. Stronger, Lighter, Better: Towards Life-Long Attribute Value Extraction for E-Commerce Products. In Findings of the Association for Computational Linguistics: ACL 2024, pages 8631–8643, Bangkok, Thailand. Association for Computational Linguistics.

Guineng Zheng, Subhabrata Mukherjee, Xin Luna Dong, and Feifei Li. 2018. Opentag: Open attribute value extraction from product profiles. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1049–1058.