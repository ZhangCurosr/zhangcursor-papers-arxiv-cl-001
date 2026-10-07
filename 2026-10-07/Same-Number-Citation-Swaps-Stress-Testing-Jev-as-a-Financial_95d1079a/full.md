# Same-Number Citation Swaps: Stress-Testing Jev as a Financial Evidence Judge

Chuhong Xu Sofia University chuhong.xu@sofia.edu

Ruiyang Xu Northeastern University xu.r@northeastern.edu

Bo Su   
Indiana University   
subo@iu.edu

Shimeng Dai Michigan State University daishime@msu.edu

Ziyao Chen University of California, San Diego cziyao@ucsd.edu

Xinyu Qiu Northeastern University qiu.xiny@northeastern.edu

Abstract—Financial reports repeat values across periods, metrics and accounting lines, allowing an LLM-generated calculation to be numerically correct while citing the wrong financial role. We evaluate what probabilistic evidence verification adds beyond number matching using Jev as a source-support verifier for GPT-4.1-mini calculation traces. A signed-number-at-pointer baseline explains most recovery over exact quotation checks. To isolate the remaining role-recognition problem, we hold operands and arithmetic fixed, move citations between same-number cells, and retain controls that express equivalent facts. These contrasts reveal both wrong-role citations that pass and valid alternative citations that are withheld. Explicit column labels improve selected wrong-role decisions while also lowering support for some equivalent evidence. A constructed follow-up on 36 new source pages, labeled by a non-author reviewer, extends this evaluation and exposes the same tradeoff between detecting role errors and retaining valid citations. The contribution is a controlled evaluation that identifies what a probabilistic financial verifier distinguishes when numerical matching is held fixed. For LLM-based financial assistants, it makes numerical correctness, cited-role support and acceptance outcomes separately assessable.

Index Terms—counterfactual citation perturbation, evidence attribution verification, financial document question answering, Jev, LLM-as-a-judge, probabilistic source verification, selective acceptance and abstention, tabular numerical reasoning, typed decision models

## I. INTRODUCTION

An LLM-based financial research assistant produces a numerical calculation and citations for its operands. Verifying the output requires checking both the calculation and the cited cells’ period, metric and accounting role. In our GPT-4.1- mini records, 27 of 192 traces have executed outputs that match the answer key yet cite unsupported roles. A literal quotation check already catches all 27, making it essential to compare probabilistic verification against deterministic citation checks. We ask what the verifier adds once numerical citation matching is in place, and whether it retains valid evidence while detecting role errors.

FinQA supplies numerical reasoning questions over financial reports [1]. Its answers come with gold programs, but a number and its cited basis still have to be assessed separately. FinanceBench pairs financial answers with supporting evidence [2]. Financial reports repeat the same value across years, metrics and accounting categories, so a value found at a citation may belong to the wrong role, while a different citation may express the same fact. We ask: what does probabilistic financial evidence verification add beyond numerical matching, and which citation changes reveal its limits?

We use Jev to verify GPT-4.1-mini calculation traces, each containing operands, an operation, source pointers and an answer. We resolve each pointer into the supplied excerpt and ask Jev whether the cited evidence supports the calculation. This source-support probability concerns a different event from numerical answer selection.

The experiments build on one another. Natural traces give performance against retained source labels, and a signednumber-at-pointer rule measures the contribution of number matching. Same-number swaps then hold numerical agreement fixed to test financial-role recognition, with equivalent-fact controls distinguishing harmless citation changes from genuine role errors. Explicit column rendering probes selected failures while preserving the source. A follow-up repeats the design on new pages selected using source information alone, with non-author source review.

The evaluation measures incremental value as supported acceptances and joint passes relative to signed-number matching, financial-role discrimination at fixed numerical agreement, and representation sensitivity under explicit column rendering. Equivalent-fact controls measure retention of valid evidence. The 192 natural records are LLM outputs; the 36-question follow-up uses program-derived records. Historical assistant labels and returned reviewer labels remain distinct throughout. Together, these comparisons characterize a probabilistic verification component for financial assistants.

Section IV reports the original, prospectively specified release endpoint. Component corrections did not improve fixedgate released answers on this bank; the later source-support evaluation addresses the evidence-verification component.

## II. FINANCIAL RECORDS AND MEASUREMENT TARGETS

## A. Source population and calculation traces

The bank contains 96 FinQA test questions with singleoperation calculations. The source is fixed to FinQA revision 0f16e286 [1]. Of 1,147 test records, 243 pass the structural screen, spanning 175 source pages. Source-only assistant review of the first 144 hash-ranked pages accepts 115. We additionally exclude a known cross-year duplicate of a reserved repair-pilot question, then take the first 96 remaining accepted questions. They cover 96 pages and 88 company/year reports. Retained screening decisions and reasons define the study population: selected single-operation questions from the benchmark.

Before collection, we committed the sample, prompts, execution rules, original release policies and analysis specification. Each question receives one Jev numerical selection and two GPT-4.1-mini traces from identical menu-free inputs. Jev selects among five numerical candidates: the rounded key, one numerical neighbor and three arithmetic distractors, assigned to neutral IDs. A fixed rule consults a trace only when the maximum Choice probability is strictly below 0.8; this selector role defines the bank’s original release endpoint (Section IV), while the later verifier role studied in Section III evaluates cited evidence. All 288 requests succeed on their first attempt. Both draws are retained, yielding the 192 natural records. Later source-probability and citation experiments were designed after inspecting these records: retrospective extensions of a prospectively collected bank.

External calculation is established in PAL [3]. Checking an operand’s financial role requires linking that operand to its source context. Financial QA agents also use external calculation [4].

Each trace requests two decimal operands, their source pointers and quotations, one arithmetic operation, a stated answer, an output unit and a justification. A bounded executor applies predefined addition, subtraction, multiplication or division to the supplied operands at their emitted scale, then rounds to the frozen question precision. Missing-operand inference, value rescaling and generated-code execution are outside its scope. Stated and executed numbers are paired readings of the same trace. The model receives the question and supplied financial excerpt; the numerical key, gold reasoning program and source annotations are withheld.

## B. Numerical correctness, evidence support and acceptance

Numerical correctness means matching the frozen rounded FinQA-derived key. Mismatches and unavailable outputs are separate outcomes. We assess numerical correctness and source support as separate dimensions: source support concerns the cited operands’ financial roles, while numerical correctness also depends on arithmetic and output scaling.

Cited-role support asks whether the actual cited cells, rows, headers or text spans provide the operands in the roles required by the question: metric, entity or cohort, period, status and contextual sign. It also includes task-role completeness; for example, a cash-spent calculation requires evidence beyond share counts. Arithmetic order and final unit conversion are assessed separately from the cited roles. Assistant review assigns supported, unsupported, ambiguous or unassessable labels and retains a source-linked reason. An additional fullexcerpt judgment asks whether the expressed calculation is recoverable elsewhere in the supplied evidence. Cited-role support remains anchored to the actual emitted pointer.

The natural-record source labels come from unblinded assistant review of all 192 traces, with documented adjudications and two unresolved records retained. Support is defined by the cited source context; numerical outcomes and later verifier probabilities are separate readouts. Independent human confirmation of these natural-record labels remains incomplete. The returned non-author review in Section III-D applies exclusively to the new constructed contrasts.

Literal correspondence requires the emitted quotation and numerical operand span to agree with the location addressed by the pointer. Combined with available execution and matching output units, this forms the literal/unit acceptance rule. Acceptance is a decision under a stated rule, distinct from source support and from numerical correctness. At a high acceptance cutoff, a withheld citation can still carry a support probability above 0.5. A joint pass requires both supported cited roles and an executed key match.

Evidence binding refers here to resolving a citation to its actual source location and presenting that location’s context. The initial verifier input includes the complete source table, visible row/column indices, the table’s header row and the cited row. Thus column identity is already recoverable. The later intervention repeats the selected column’s header explicitly alongside the citation, varying its representation while holding source availability fixed.

## C. Models and reporting units

All Jev calls return typesafe/jev-1.13-20260917 through Open-Router with the TypeSafe provider. Numerical answer selection uses Choice; source verification uses one independently specified binary Noul event. Choice and Noul probabilities are analyzed as distinct measurement targets. Analyses use the exact returned probabilities, preserving their original scale; the interface’s native confidence statistic is excluded from accuracy estimation.

GPT-4.1-mini uses the OpenAI provider, disabled provider fallback, temperature 0, top-p 1 and a 1,024-token output limit for traces. The prompt requests JSON, and decoding is unconstrained. The returned openai/gpt-4.1-mini identifier is a serving alias; it does not pin immutable model weights. Jev selection and menu-free trace generation have different tasks and output budgets, defining a workflow comparison between their assigned roles.

Natural-record summaries retain all traces and question identities. Verifier and original-release resampling runs use 2,000 and 20,000 draws, respectively. Question bootstrap intervals quantify variation across sampled questions while retaining both draws. Annotation uncertainty and future serving variation lie outside their coverage. Constructed citation studies report contrasts and source questions separately; repeated identical inputs are counted as technical calls within the same semantic case.

## III. VERIFICATION EXPERIMENTS

A. Natural records: discrimination and a stronger numerical control

The natural records contain both numerical and citation errors (Table I). Of 192 executed outputs, 170 match the key, 21 do not, and one is unavailable. Source review finds 153 supported traces, 37 unsupported, one ambiguous and one unassessable. Twenty-seven key-matching traces on 14 questions cite unsupported roles even though their derivations are recoverable elsewhere in the excerpt. Literal correspondence detects all 27, demonstrating why numerical correctness and cited-role support require separate assessment. Conversely, literal correspondence rejects 33 supported traces. The combined literal/unit rule accepts two unsupported-role traces; both are numerical mismatches.

We next ask whether a probabilistic source verifier can improve this acceptance tradeoff. Every natural-record, citationswap and rendering request uses the same fixed Noul instruction, beginning: "Does the cited evidence support the proposed record’s operand values and financial roles for the quantity requested in the question?" The instruction requires support at the actual cited locations for the requested metric or entity, period, sign and source unit, allowing harmless paraphrases and using surrounding context. It explicitly excludes rejection based solely on a wrong arithmetic operator, operand order, final rounding or missing final display-unit conversion. The same instruction is present in all 590 saved evaluation requests, including the fresh-source follow-up in Section III-D.

Only the allowlisted source and record state varies. Jev receives the full supplied excerpt, question, requested unit and precision, operation, operands, justification, executed value and deterministically resolved citations. The numerical key, source labels, checker verdicts and original stated-answer field are absent. This input defines the source-support target while leaving numerical-correctness cues available to the model. We wrote the predicate after inspecting the retained labels and tested the input format in four development checks; all 192 natural-record requests then succeeded. The experiment evaluates a new probability target retrospectively on known records.

On the 190 binary labels, the source probabilities have Brier score 0.03378 and AUROC 0.98587. Question-bootstrap 95% intervals from 2,000 resamples are [0.01814, 0.05260] and [0.96356, 0.99938], respectively. The binary support rate of 0.8053 is close to the mean probability of 0.7975. Calibration also requires agreement within probability ranges: in the largest fixed probability bin, 146 records have mean probability 0.9375 and observed support rate 0.9932. A preinference sensitivity excluding six task-role boundary records from questions 467, 576 and 991 retains 184 binary records, with Brier 0.03132 and AUROC 0.98551. Original labels remain unchanged.

At the inclusive 0.8 source-probability cutoff, all 27 keymatching traces with unsupported cited roles are withheld (maximum probability 0.79), as are nine of the ten unsupported traces with numerical mismatches. Thus no correctnumber/unsupported-role trace is accepted at that cutoff in this natural bank. Section III-B tests role recognition with numerical agreement held fixed.

To measure the increment beyond number matching, we add a post hoc zero-call control, the relaxed pointer rule: remove exact-quotation matching while still requiring each signed operand value at its actual cited location. Matching is confined to the string at the emitted table-cell or text-span pointer, using the operand’s original sign and scale. Whole-row list pointers are ineligible. Numerical spans are matched as complete decimal values, ignoring thousands separators while preserving accounting parentheses as a negative sign. The shared release guards require an available execution and exact equality of the emitted and requested unit strings. This guard checks unit labels; scaling correctness is a separate numerical question. Both rules operate on the source and emitted record, with answer keys and support labels excluded.

Table II retains all four inclusive cutoffs listed before verifier inference: 0.5, 0.8, 0.9 and 0.95. New releases require available execution and matching output units in addition to the probability gate. The OR policy rescues records rejected by the original literal/unit rule. Unresolved labels remain in the full denominator rather than becoming supported records.

This relaxed pointer rule recovers 31 of the 33 supported literal failures and 24 of the 26 that also match the key, with no additional unsupported acceptance and one ambiguous acceptance.

At 0.8, Jev accepts 146 records with 142 joint passes, compared with 154 accepted and 141 joint passes for the relaxed rule. The tradeoff is fewer accepted records for one additional joint pass. The two Jev-only joint passes both belong to question 218, where a negative benefit is used as a positive magnitude in a subtraction. This contextual-sign case identifies the specific distinction from strict signed-number matching. As a hard 0/1 classifier, the relaxed rule also has lower Brier score on these labels (4/190 = 0.02105), although its erroneous certainties have infinite unregularized log loss. These comparisons motivate testing role recognition while holding numerical matches fixed.

Paired decision differences. At the same inclusive 0.8 cutoff and with identical release guards, both rules accept 144 traces, Jev alone accepts two, the numerical rule alone accepts ten, and both withhold 36. Their joint-pass counts in these groups are 140, two, one and zero, respectively. The net gain of one joint pass is therefore two recoveries on question 218 minus one withheld valid trace on question 583, draw 1. Removing both question-218 traces leaves 190 records: Jev has 140 joint passes versus 141 for the numerical rule. This leave-onequestion-out sensitivity locates the observed increment in a single contextual-sign case.

A. Numerical outputs. The 96 questions are retained in each draw. These are all-trace component results, not the selected-gate release outcomes in Table III.
<table><tr><td>Trace draw</td><td>Output</td><td>Matches / mismatches / unavailable</td></tr><tr><td>0</td><td>Stated number</td><td>79 /  16 /  1</td></tr><tr><td>0</td><td>Executed number</td><td>84 / 11 / 1</td></tr><tr><td>0</td><td>Execution + lexical/unit acceptance</td><td>58 / 3 / 35</td></tr><tr><td>1</td><td>Stated number</td><td>80 / 16 / 0</td></tr><tr><td>1</td><td>Executed number</td><td>86 /  10 / 0</td></tr><tr><td>1</td><td>Execution + lexical/unit acceptance</td><td>59 / 2 / 35</td></tr></table>

B. Cited-role and literal-check diagnostics. All draw cells retain the scheduled denominator. The question column counts at least one qualifying draw; its rows need not be mutually exclusive. Labels are assistant judgments.
<table><tr><td>Measurement</td><td>Draw 0</td><td>Draw 1</td><td>All draws</td><td>Questions</td></tr><tr><td>Cited roles supported</td><td>74/96</td><td>79/96</td><td>153/192</td><td>79/96</td></tr><tr><td>Cited roles unsupported</td><td>20/96</td><td>17/96</td><td>37/192</td><td>20/96</td></tr><tr><td>Cited roles ambiguous</td><td>1/96</td><td>0/96</td><td>1/192</td><td>1/96</td></tr><tr><td>Cited roles unassessable</td><td>1/96</td><td>0/96</td><td>1/192</td><td>1/96</td></tr><tr><td>Key-matching executed number, unsupported cited roles</td><td>14/96</td><td>13/96</td><td>27/192</td><td>14/96</td></tr><tr><td>Supported cited roles, literal correspondence fails</td><td>15/96</td><td>18/96</td><td>33/192</td><td>19/96</td></tr><tr><td>Literal/unit accepted, unsupported cited roles</td><td>2/96</td><td>0/96</td><td>2/192</td><td>2/96</td></tr></table>

TABLE II

SOURCE-PROBABILITY GATES ON THE EXISTING 192 TRACES. ALL THRESHOLDS ARE INCLUSIVE. SUPPORTED/UNSUPPORTED/UNRESOLVED REFER TO RETAINED ASSISTANT SOURCE-ROLE LABELS; JOINT PASSES ADDITIONALLY MATCH THE NUMERICAL KEY.
<table><tr><td>Acceptance rule</td><td>Accepted / 192</td><td>Supported</td><td>Unsupported</td><td>Unresolved</td><td>Joint passes</td></tr><tr><td>Literal/unit rule</td><td>122</td><td>120</td><td>2</td><td>0</td><td>117</td></tr><tr><td>Relaxed pointer rule (post hoc)</td><td>154</td><td>151</td><td>2</td><td>1</td><td>141</td></tr><tr><td>Jev source gate, 0.5</td><td>159</td><td>152</td><td>6</td><td>1</td><td>143</td></tr><tr><td>Jev source gate, 0.8</td><td>146</td><td>145</td><td>1</td><td>0</td><td>142</td></tr><tr><td>Jev source gate, 0.9</td><td>126</td><td>125</td><td>1</td><td>0</td><td>123</td></tr><tr><td>Jev source gate, 0.95</td><td>87</td><td>87</td><td>0</td><td>0</td><td>87</td></tr><tr><td>Literal OR Jev, 0.5</td><td>161</td><td>153</td><td>7</td><td>1</td><td>143</td></tr><tr><td>Literal OR Jev, 0.8</td><td>151</td><td>149</td><td>2</td><td>0</td><td>143</td></tr><tr><td>Literal OR Jev, 0.9</td><td>144</td><td>142</td><td>2</td><td>0</td><td>137</td></tr><tr><td>Literal OR Jev, 0.95</td><td>135</td><td>133</td><td>2</td><td>0</td><td>130</td></tr></table>

Seven of the ten rule-only acceptances have supported cited roles but mismatched numerical answers, spanning four questions. They explain the difference between ten such accepted traces under the rule and three under Jev. Withholding these records can improve accepted-answer precision while leaving the joint-pass count unchanged. The pattern links lower source probabilities to broader calculation defects. Because arithmetic, explanations and other input features vary together in these natural traces, their individual contributions remain unresolved.

## B. Same-number citations: isolating financial role

Systematic tabular-reasoning probes use evidence perturbations to examine model behavior [5]. We apply this logic to a source-support verifier while holding operands and arithmetic fixed. TabDSR also uses table-data perturbations [6]. The present intervention changes the citation pointer and then the presentation of its selected column.

In each matched contrast, arm A retains the original citation and arm B points one operand to a different cell containing the same numerical value. Both arms use exact cell quotations and a neutral justification. The question, complete source, operand values, operation and executed result remain fixed. The changed pointer and its resolved context are the intended difference. Both arms therefore satisfy the same signednumber-at-pointer criterion.

Source review before inference separates clear wrong-role changes from equivalent facts and unresolved boundaries. Equivalent-fact labels require support from the source context as well as equal values. For example, an ending balance and the next period’s opening balance may represent the same carried-forward amount. Original proposed classes and subsequent pre-inference adjudications are both saved. Categories remain fixed throughout inference.

Same value, wrong period (question 906). The question asks for Intel’s cumulative return relative to the Dow Jones U.S. Technology Index in 2010. The record divides Intel’s 157 by the index’s 191. Arm A cites /table/2/3, headed "2010"; arm B cites /table/2/4, headed "2011". Both index cells contain "\$ 191", and the executed ratio remains 0.82. Numerical matching is preserved, while the swapped citation supplies the following year’s index fact. The citation contract requires the period requested by the question.

Different headings, equivalent fact (question 535). For December 2012 repurchases, arm A cites 102,400 total shares at /table/3/1; arm B cites the same count under publicly announced plans or programs at /table/3/3. The preamble says the quarter’s repurchases were "pursuant to our publicly announced stock repurchase program", and note (1) attaches the totalpurchase column to that program. This source statement, rather than equality alone, supports the equivalent-purchase-set label. Both arms retain the December average-price operand. The judgment concerns cited roles; fee completeness is a separate calculation question. Both examples include the assistant judgments and their source-based rationales for inspection.

The prepared bank contains 42 evaluation calls over 12 questions; a source-only supplement adds 12 calls over three questions, with question 930 shared. Together they comprise 15 distinct contrasts over 14 questions: six clear wrongrole contrasts over five questions, six equivalent-fact controls and three boundary contrasts. Repeated identical payloads inherited from the natural traces contribute unequal numbers of technical calls. Fig. 1 averages technical calls within each arm, using the contrast as the semantic reporting unit. All 54 evaluation and eight format-check calls succeed.

Two clear wrong-role B means fall below 0.8; four remain accepted, including wrong-period and wrong-status citations near 0.95. Two of six equivalent-fact B means also fall below 0.8. Boundary cases retain their unresolved status. High aggregate discrimination on natural records therefore coexists with missed role errors and sensitivity to equivalent citations in these constructed cases.

A post hoc structural split suggests a concrete followup. Seven contrasts change only the column; their B-minus-A shifts range from -0.045 to zero. Eight change the row, with or without the column; six B means fall below 0.8. Row changes also alter the surrounding numbers and text, so this split motivates a representation test. Column identity is already available through visible pointer indices and the complete table; the next experiment tests the effect of repeating it explicitly in the citation display.

## C. Diagnosing selectedfailures through explicit column labels

We reuse all seven column-only contrasts, including four wrong-role contrasts on three questions, two equivalent-fact controls and the accounting/parser boundary. For each A/B arm, the enriched input adds only column\_header=source.table[0] [column\_index] to every table citation lookup. The header is copied verbatim. Removing the added fields restores the original serialized state exactly. Source content, prompt, model instructions, annotations and numerical values remain fixed.

The inputs and readout are saved before these new calls. Original and enriched renderings are recollected together in one randomized interleaved schedule (seed 80930), with two repetitions of each arm/rendering combination. This schedule supplies contemporaneous original controls for the enriched rendering. All 56 evaluation and four format calls succeed. The seven contrasts cover six source questions and 26 unique evaluation inputs. The two contrasts on question 930 share an A input, producing four technical A calls per rendering for that shared input.

Fig. 2 shows every cell mean. The interaction is the enriched B-minus-A difference minus the original B-minus-A difference. A negative interaction indicates greater separation between the original and swapped citations; whether that separation is desirable depends on the semantic stratum.

For the four selected wrong-role contrasts, all eight B calls are accepted under the original rendering and withheld under the explicit rendering at 0.8. All corresponding A calls remain accepted in both conditions. Enriched A means range from 0.950 to 0.970, whereas enriched B means range from 0.115 to 0.620. The largest observed identical-input range is 0.05 in this repeat check. These matched observations support a representation effect on the tested period and status errors.

The equivalent controls reveal the cost to valid evidence. Both enriched B means are 0.785; one of their four individual calls is accepted at the inclusive cutoff. Question 583 also has an original A mean below 0.8, and enrichment lowers its A mean further, producing a positive interaction. Explicit labels thus change support for valid as well as wrong citations. The accounting/parser case retains its unresolved label. Coverage is limited to the seven column-only contrasts; the earlier rowchanging opening/closing-balance controls were not rerun.

On these selected cases, explicit column labels increase sensitivity to period and status mismatches while lowering support for some equivalent facts. This representation-sensitive tradeoff motivates the fresh-source test that follows.

## D. Fresh source pages with a returned non-author review

To test beyond previously observed failures, we construct a separate bank from the pinned FinQA train split. After excluding 178 pages used by earlier local experiments, a structural screen yields 116 eligible questions on 91 pages: both operands must be leading signed numbers in table cells, with another same-value cell for at least one operand. A fixed hash ranking selects 36 questions on 36 pages from 35 reports, one question per page. A source-only rule chooses the original citations and the first alternative same-number cell in rowmajor order. The follow-up uses program-derived constructed records to test controlled citation changes. These pages are new to our local experiments; natural model-error frequencies and training-data overlap are outside this sampling design.

Before inference we save the inputs, sample and original/explicit-column schedule. All 288 evaluation calls (two arms, two renderings, two repeats per question) and four format checks succeed. The prompt, model, column-field intervention and cutoffs are unchanged. One non-author reviewer returns labels for all 36 contrasts, declaring unaided reading of the supplied transcripts and no access to model outputs, author labels or the blinded X/Y mapping. Review took place after collection under the reviewer’s declared blindness. The reviewer assesses support for each complete citation set, including its unchanged operand. The labels reflect one reviewer per case; inter-human agreement remains unmeasured.

![](images/9624214e63850d1843b19e14fa540211e99cddaf6739404fa8f84b121c861fda.jpg)  
Fig. 1. Same-number citation contrasts under the original rendering. A retains the original citation; B cites a different same-number cell. Shading separates wrong roles, equivalent facts and boundaries. Two wrong-role and two equivalent-fact B means fall below the inclusive 0.8 cutoff. Question 930 contributes two contrasts. Means aggregate technical repeats within each arm; counts describe the constructed contrasts.

![](images/f5f471b412e7e5aefa65aa0bbc80df8887a30a9f86bac2ed01f2044ef39ef5d7.jpg)  
Fig. 2. Explicit-column intervention: all two-call A/B means and interactions. Interaction is explicit (B minus A) minus original (B minus A). Negative values indicate greater separation; wrong roles call for separation, whereas equivalent facts call for support in both arms. Boundary labels remain unresolved. Means are descriptive; question 930 shares a technically repeated A input.

The returned labels identify 15 contrasts with A supported and B unsupported, 19 with both supported, and two with neither supported. Eight contrast labels differ from the corrected assistant judgments (64 of 72 arm labels agree); all disagreements and original rationales remain saved. Fig. 3

uses the returned labels as specified by the follow-up protocol;   
Tables 1–2 and Figures 1–2 retain their historical labels.

The follow-up quantifies the tradeoff between wrong-role separation and supported-evidence coverage. Explicit columns remove the two unsupported-B acceptances, but supported-A withholding rises from five to six; paired separation increases by one contrast. Among equivalent controls, most pairs still have at least one arm withheld. With numerical agreement held fixed, these contrasts reveal both successful role separation and withholding of valid evidence.

Continuous scores qualify this policy-level interpretation. Among the 53 reviewer-supported arms, 25 are withheld at 0.8 under original rendering and 24 under explicit rendering; respectively 16 and 13 of those still have means at least 0.5. Means below 0.5 occur on 9 and 11 supported arms. In the 15 A-only-supported contrasts, mean A exceeds mean B in 11 versus 14 cases, but median A-minus-B decreases from 0.500 to 0.440. Thus ordering, separation magnitude and acceptance coverage are distinct readouts. The saved per-case ledger reports all 288 probabilities, two-call means and paired differences; Fig. 3 reports the complete original cutoff set.

All six column-only contrasts remain visible: four reviewerlabeled wrong roles and two equivalents. Explicit headers preserve accepted A while withholding B on two wrong-role examples (444 and 4296); the other two wrong-role pairs have both arms withheld in both renderings. Equivalent case 1687 changes from both accepted to B withheld, while equivalent case 3921 remains accepted. These three threshold-changing examples illustrate a local benefit and its specificity cost within the full six-contrast subgroup. Case 444’s issued/outstanding distinction also admits a contextual-equivalence reading. Post hoc exclusion of this case removes the net gain: paired separation is 8/14 in both renderings. Excluding all eight author/reviewer disagreements likewise gives 8/14 in both renderings, with both-equivalent acceptance at 3/12 versus 4/12. These sensitivity analyses retain the original labels.

All 36 review cases reproduce the question and source text in the verifier inputs, but the reviewer can inspect both arms and the whole packet. A source-only check finds one rationale, for case 1185, explicitly borrowing another case’s year-column order. Excluding it gives paired separation 8/14 versus 9/14 and leaves the 19 equivalent pairs unchanged. A broader post hoc header-quality exclusion covers 1185, 2484 (duplicated year labels), and 3521 and 2829 (data rows reused as headers); separation is 6/12 versus 9/12 and both-equivalent acceptance is 5/18 versus 6/18. Case 4296 remains because its own prose supplies the year order. These source-quality sensitivities retain the existing review, with the full 36-case analysis remaining primary.

The four semantic concerns recorded before inference remain a sensitivity, even though the reviewer labels all four as equivalent. Excluding them leaves the negative readout unchanged and equivalent B withholding at 9/15 versus 8/15. Excluding headerless case 3521 changes negative paired separation to 7/14 versus 9/14; its original A mean 0.805 becomes 0.795, so its lost acceptance is a small cutoff crossing.

The largest identical-input repeat range is 0.18. The caselevel results expose sensitivity to role interpretation and nearthreshold means; they describe this constructed bank rather than population rates.

## IV. VERIFICATION AND ANSWER-RELEASE OUTCOMES

Cascade analysis emphasizes the conditional quality of both the primary model and its fallback [7]. The verifier experiments measure support probabilities on saved traces. Table III evaluates answer release on the same question bank under the original consultation rule introduced in Section II, retaining its prospectively specified endpoint and full denominator.

For this original task, Jev chooses among five numerical candidates: the rounded key, a one-sided neighbor and three arithmetic distractors assigned to neutral IDs by a fixed hash. The neighbor gap is the larger of one rounding unit and a rounded 5% of the absolute key; actual relative gaps span 4.08%–100%. These transformed questions define a FinQAderived selection task. Trace generation uses menu-free inputs.

The inherited policy consults the saved trace only when the maximum Choice probability is strictly below 0.8, otherwise retaining Jev’s answer. Stated and executed fallback are compared using the same trace; a secondary policy requires literal/unit acceptance or abstains. This consultation rule concerns numerical selection and is distinct from the inclusive source-support gates in Table II. The later Noul verifier and column-rendering intervention are not inserted into this original policy.

Jev matches 95 of 96 keys. All 12 consulted questions already have correct primary answers; the sole primary error has probability exactly 0.8 and is outside the strict gate. Thus the observed gate has no correction opportunity in this bank. Execution corrects 13 stated outputs on seven questions across the complete trace bank, but none of those corrections activates the gate. Two execution regressions share a separate question involving output-unit scaling.

Executed fallback yields 93 and 94 key matches across the two trace draws. The prospectively specified mean withinquestion difference is -1.5625 percentage points, with a descriptive 95% question-bootstrap interval of [-4.1667, 0] from 20,000 resamples. The interval includes zero, so a zero population effect remains compatible with this descriptive estimate. A post hoc sensitivity accepting both readings of a temporal ambiguity changes the point difference to -1.04 points without replacing the original scoring. The original fixed-gate comparison provides a measured workflow boundary for the component findings: no release benefit was observed on this bank, and the later source verifier was evaluated separately.

## V. IMPLICATIONS AND RELATION TO PRIOR WORK

## A. Incremental information and failure conditions

The natural-record comparison identifies the value of a strong numerical baseline: signed-number matching explains most recovery over exact quotation checks, while probabilistic verification offers different coverage and joint-pass tradeoffs across cutoffs. Same-number swaps then expose citation distinctions that numerical agreement alone cannot resolve. Jev responds to some wrong roles but misses others, and the acceptance policy also withholds some equivalent citations. The incremental-information question therefore requires both unsupported-role detection and supported-evidence retention.

(a) Acceptance outcomes at the inclusive 0.8 cutoff  
Original rendering Explicit column  
![](images/05cc71ac3d133071b14543702fff2eebeb251f41f385241f26a563fd555c04d8.jpg)

(b) Accept A, withhold B; only A supported  
![](images/e268dd689af901c056c594ca5e6456ee15b058e23ba27736c5c1d75bb28b3337.jpg)

(c) Accept both; both supported  
![](images/3df0399da0822ea369cb4305dab135949651a673f1552345b4f52e2fa0d93d25.jpg)  
Fig. 3. Fresh-source outcomes under one reviewer’s declared blinded labels. Decisions use two-call means and inclusive cutoffs. (a) All six outcomes at 0.8; overlapping measurements within disjoint strata of 15 A-only-supported, 19 both-supported and 2 neither-supported pairs. (b) Accept A, withhold B among A-only-supported pairs. (c) Accept both among both-supported pairs. Cutoffs were fixed before collection; lines guide the eye. A is rule-selected; B changes one same-number pointer. Fractions describe this constructed bank.

TABLE III  
PROSPECTIVELY SPECIFIED FINANCIAL RELEASE COMPARISON.  
Each cell gives key matches / mismatches / unavailable among all 96 questions. The same primary response is used in both trace draws; the mean is computed within question. The gate is strictly below 0.8.
<table><tr><td colspan="3"></td><td rowspan="2">Mean net match change (pp)</td><td rowspan="2">Consulted questions</td></tr><tr><td>Release policy</td><td></td><td>Trace draw 0 Trace draw 1</td></tr><tr><td>Jev alone</td><td> $9 5 \mathrm { ~ / ~ 1 ~ / ~ 0 ~ }$ </td><td> $9 5 \mathrm { ~ / ~ 1 ~ / ~ 0 ~ }$ </td><td>+0.00</td><td>0</td></tr><tr><td>Gate → stated number</td><td> $9 3 \mathrm { ~ / ~ } 3 \mathrm { ~ / ~ } 0$ </td><td> $9 4 / 2 / 0$ </td><td>-1.56</td><td>12</td></tr><tr><td>Gate → executed number</td><td> $9 3 \mathrm { ~ / ~ } 3 \mathrm { ~ / ~ } 0$ </td><td> $9 4 / 2 / 0$ </td><td>-1.56 -7.81</td><td>12 12</td></tr><tr><td>Gate → execution + lexical/unit acceptance</td><td> $8 8 \mathrm { ~ / ~ 1 ~ / ~ 7 ~ }$ </td><td> $8 7 \ / \ 1 \ / \ 8$ </td><td></td><td></td></tr></table>

The row/column response difference motivates a controlled test of representation sensitivity. Making column identity explicit changes the tested period/status decisions, while equivalent controls distinguish justified rejection from withholding of valid evidence. The fresh-source follow-up retains this tradeoff under a non-author reviewer and reveals withholding of supported original citations at the chosen cutoff. Together, these comparisons identify sensitivity to the supplied representation and its acceptance-coverage consequences.

For financial research assistants, verification must check the link between an LLM-generated operand and documentary evidence. Tool execution checks the proposed arithmetic; citation verification additionally assesses the operand’s financial role. A verifier evaluation should test both rejection of unsupported roles and retention of equivalent evidence, such as carriedforward balances. Our matched inputs hold arithmetic fixed and evaluate emitted records, making cited-role support a distinct target alongside numerical correctness and answer release. Section IV retains the original fixed-gate release comparison as the workflow-level endpoint.

## B. Closest precedents and scope of contribution

JEV-as-a-Judge is a direct precedent for probabilistic judging and escalation [8]. Our experiments examine what a judge distinguishes in cited financial evidence. BudgetVerify studies the allocation of verification effort in financial QA [9]. Equalnumber and equivalent-fact controls add a complementary evaluation axis: whether a verifier distinguishes unsupported roles from valid alternative citations.

Hume’s practitioner report probes Jev’s contextual behavior [10]. This provides a black-box precedent for examining sensitivity to the supplied context. Sun et al. study option-name binding in typed decision models [11]. Here the controlled variation concerns cited financial roles at fixed numerical values.

VeriFin combines financial evidence grounding with symbolic verification [12]. Our target is the probability assigned to support at a specified citation. FinCARDS aligns financial entities, metrics, periods and numerical spans for evidence reranking [13]. Checking the role of an already cited operand complements these grounding and reranking stages.

FinRank likewise studies evidence-grounded financial question answering and retrieval over SEC filings [14]. The present study holds the available source fixed and varies a citation or its rendering. This controlled comparison connects numerical baselines, financial-role contrasts and equivalent-fact controls within the evaluation of a probabilistic verifier.

## VI. LIMITATIONS AND REPRODUCIBILITY

The natural bank is selected for single-operation calculations and spans overlapping company/year reports. The primary selector’s 95/96 key matches leave little room for the original gate to improve accuracy. Source PDFs, training contamination and historical information cutoffs are outside the evaluation’s scope. Historical labels come from unblinded assistant review. The fresh-source extension adds one returned non-author review, with blindness and method declared in its form; this is a single-reviewer assessment without inter-human reliability estimates or consensus adjudication. Temporal interpretation, malformed year headers, price-role completeness and source transcription defects require annotation judgment. One returned rationale explicitly infers missing year-column order from another supplied case, a context difference from the verifier’s per-question input; Section III-D reports its exclusion and a separate header-quality sensitivity. These analyses show dependence on label and source-quality choices; semantic label validity remains a separate requirement.

The historical extensions are exploratory: the source verifier was specified after inspecting natural records, the relaxed deterministic baseline was added after inspecting verifier results, and the column intervention revisits observed failures in constructed citation swaps. Inputs and assistant labels were saved before each collection within this exploratory sequence. The 36-question follow-up avoids selection by verifier outcomes, while its structurally filtered, program-derived population limits generalization. Four negative rendering contrasts span only three source questions, and question 930 shares an A input. The two equivalent B means in the selected column-rendering study sit close to the cutoff, making these decisions sensitive to small probability changes. Counts describe the selected cases; failure prevalence in financial reports remains unmeasured.

The experiments evaluate one returned Jev identity and one generative alias. Technical repeats and two trace draws characterize observed serving variability; future serving behavior requires separate assessment. Source probabilities target cited-role support against the stated annotation source. Trading performance, live-agent improvement and the model’s internal reasoning are outside the evaluated endpoints.

Saved records retain source revisions, selection decisions, prompts, serialized inputs, model/provider identities, exact probabilities, technical-repeat groups, costs and source-review reasons. Original endpoints and historical experiments retain their own records. Recorded costs, rounded to five decimal places, are USD 0.14781 for the 288-call natural-bank collection, USD 0.01669 for the 196-call source-verifier extension, USD 0.00488 for 62 swap-study calls including format checks, USD 0.00481 for the 60-call rendering experiment, and USD 0.02417 for the 292-call fresh-source follow-up. These charges document observed provider billing; the experiments use different tasks and call budgets. Offline numerical and inputinvariant checks support numerical reproducibility; semantic assessment relies on the retained source review. A portable inspection packet retains exact instructions, source contexts, paired inputs, individual responses, label reasons and the complete natural-record disagreement table. The blinded packet and exact returned form preserve the follow-up review and declaration. A follow-up ledger retains all 288 probabilities, original-cutoff summaries and post hoc label/header sensitivities. The inspection artifacts are retained privately; public access remains pending.

## VII. CONCLUSION

For LLM-generated financial calculations, probabilistic evidence verification should be assessed by what it adds beyond numerical matching and by the citation changes under which it fails. On the retained natural records, a signed-numberat-pointer rule explains most recovery over exact quotation checks. Matched-number swaps reveal a separate requirement: distinguishing unsupported financial roles from alternative citations to equivalent facts. At the stated cutoff, Jev accepts some of the former and withholds some of the latter despite strong aggregate discrimination on natural records.

The fresh-source follow-up, with a separate reviewer, extends this evaluation to new pages and highlights the tradeoff between detecting unsupported roles and retaining valid citations. The contribution is a controlled evaluation combining numerical baselines, same-number swaps and equivalent-fact controls, with column rendering used to probe representation sensitivity. For LLM-based financial assistants, the evaluation separates numerical correctness, cited-role support and answer-release quality.

## ACKNOWLEDGMENT

CodeX(OpenAI) was used for language editing, experiment running, and assistance in script development. The authors reviewed the manuscripts, verified the reported results and citations, and take responsibility for the final text.

[1] Z. Chen et al., “FinQA: A dataset of numerical reasoning over financial data,” in Proc. 2021 Conf. Empirical Methods Natural Language Processing, 2021, pp. 3697–3711, doi: 10.18653/v1/2021.emnlp-main.300.

[2] P. Islam, A. Kannappan, D. Kiela, R. Qian, N. Scherrer, and B. Vidgen, “FinanceBench: A new benchmark for financial question answering,” arXiv:2311.11944v1, 2023. [Online]. Available: https://arxiv.org/abs/ 2311.11944v1

[3] L. Gao et al., “PAL: Program-aided language models,” in Proc. 40th Int. Conf. Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202, 2023, pp. 10764–10799. [Online]. Available: https: //proceedings.mlr.press/v202/gao23f.html

[4] N. Tan, Z. Seng, L. Zhang, Y.-C. Shih, D. Yang, and A. Salunkhe, “Improved LLM agents for financial document question answering,” arXiv:2506.08726v3, Jan. 2026, doi: 10.48550/arXiv.2506.08726.

[5] V. Gupta, R. A. Bhat, A. Ghosal, M. Shrivastava, M. Singh, and V. Srikumar, “Is my model using the right evidence? Systematic probes for examining evidence-based tabular reasoning,” Transactions of the Association for Computational Linguistics, vol. 10, pp. 659–679, 2022, doi: 10.1162/tacl\_a\_00482.

[6] C. Jiang, F. Yu, H. Chen, W. Lu, and J. Zeng, “TabDSR: Decompose, sanitize, and reason for complex numerical reasoning in tabular data,” in Findings of the Association for Computational Linguistics: EMNLP 2025, 2025, pp. 3172–3196, doi: 10.18653/v1/2025.findings-emnlp.169.

[7] W. Jitkrittum, N. Gupta, A. K. Menon, H. Narasimhan, A. S. Rawat, and S. Kumar, “When does confidence-based cascade deferral suffice?” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 9891–9906, doi: 10.52202/075280-0431.

[8] Y. Li, Y. Miao, R. Krishnan, and R. Padman, “JEV-as-a-Judge: Accept when confident, escalate when unsure,” arXiv:2609.26550v3, Sep. 29, 2026. [Online]. Available: https://arxiv.org/abs/2609.26550v3

[9] J. Jenq and H. Shen, “BudgetVerify: Budget-tiered verification for financial QA,” arXiv:2609.33052v1, Sep. 27, 2026. [Online]. Available: https://arxiv.org/abs/2609.33052v1

[10] A. Hume, “Jev’s architecture unmasked,” Sep. 17, 2026. Accessed: Sep. 30, 2026. [Online]. Available: https://archerhume.com/posts/ jevs-architecture-unmasked

[11] Y. Sun, J. Xu, J. Shi, and Z. Yang, “Type-safe is not error-free: Typed decision models follow the option name, not the definition bound to it,” arXiv:2609.26758v2, Sep. 23, 2026. [Online]. Available: https://arxiv. org/abs/2609.26758v2

[12] B. Hall, S. Shome, and W. Eiers, “VeriFin: A neurosymbolic framework for verifying LLM-generated financial claims,” arXiv:2608.10213v1, Aug. 2026, doi: 10.48550/arXiv.2608.10213.

[13] Y. Zhou, F. Zhang, Y. Chen, H. Zhang, P. Nakov, and Z. Xie, “FinCARDS: Card-based analyst reranking for financial document question answering,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 24836–24852, doi: 10.18653/v1/2026.findings-acl.1244.

[14] S. Mansouri, D. Saad, M. Wahrenburg, M. Weissel, and F. Woebbeking, “FinRank: An evidence-grounded benchmark for financial question answering and retrieval over SEC filings,” arXiv:2608.07400v1, Aug. 2026, doi: 10.48550/arXiv.2608.07400.