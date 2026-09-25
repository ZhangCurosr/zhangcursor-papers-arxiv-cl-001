# Low-Cost Assays for Measuring Model Behavior Across Vendors and Releases

Tapan Parikh Cornell Tech tsp53@cornell.edu

## Abstract

Language models advise people, keep them company, and write software while they sleep. Measuring what they do is hard: behavior has to be sampled repeatedly across models, prompts and releases, most of it lives in unstructured text that has to be coded before it can be counted, and the result has to be legible and rigorous enough to meaningfully compare models and vendors. To address these constraints, we present a simple, cheap, scalable, and replicable model for studying model behavior. Each study is a frozen, public stimulus run identically on a cross-vendor panel, at a few dollars per model or less. Each reads its transcripts one of three ways, chosen by how much interpretation the behavior needs: exact match on a clamped reply, a codebook applied by LLM judges whose agreement with a human coder is reported per code, and an instrumented environment that records what an agent did independently of what it said. Run across four years of model releases from both frontier and open-source labs, these instruments find four things. Convergence: asked to pick a word, 27 of 44 models answer serendipity at least once in four tries. Resistance: a trailing “right?” moves endorsement by up to 32 points, and the sign flips from sycophantic to resistant as generations advance, keyed to the tag’s surface form. House: whether a model holds a position under pressure tracks its generation, and how it holds tracks the lab that built it. Account: told to do something the documentation in their repository contradicts, some coding agents never went along silently and others always did, and the same model can change with the harness it runs in. Re-run on every release, batteries like these track how behavior is changing across vendors and over time.

## 1 A measurement problem

People ask LLMs for advice, confide in them, and let them act on their behalf while coding. Most model benchmarks measure capabilities, or how they respond to red teamed attacks. How a model behaves in everyday moments shapes what people believe and do, and our public understanding of how they behave in such situations is limited. We need to know how models behave under pressure, how they respond when a question has no right answer, and what they report about what they did when we ask them to do tasks unattended.

This is a hard measurement problem. Behavior is stochastic and varies by model, prompt and context, so it has to be sampled repeatedly across all three, and sampled again as new models are released or updated. Most of the behavior worth measuring is in unstructured text, which has to be read and coded before it can be counted. The result has to be legible to the people who deploy and use these models and rigorous enough to survive a skeptical reader.

We present a simple, cheap, scalable, and replicable model for studying model behavior to address these requirements. Each battery is a frozen stimulus that puts a model in a situation where behavior matters, sent identically to every model on a cross-vendor panel. The panel can be re-run on every release, and a battery becomes a standing record of how model behavior is changing across vendors and over time.

Running one fixed stimulus across many models is what makes diference legible. Every model meets the same situation, so where they diverge helps us understand how models difer, and where they agree it helps us see what behaviors are common across them. Each instrument reads its transcripts one of three ways, chosen by how much interpretation the behavior needs. Every transcript, label and score is public.

A study can be of one deployed agent, one model, one lineage, or the whole field, it can widen to more languages and scenes, or carry custom scenarios and rubrics of the kind eval teams already write for their own agents.

## 2 The assays

The single-turn and conversational assays run cross-vendor through one API at requested temperature 1.0, with reasoning of where a model allows it. The agent assay runs each coding product or harness in its own full-auto mode. Running one model through any of these studies costs a few dollars or less.<sup>1</sup>

Three analysis methods, in order of how much interpretation the behavior needs. Clamped: where the prompt clamps the reply to a discrete choice, the score is exact match on normalized tokens, a regular expression recomputable from the transcripts with no model in the loop. Coded: where the prompt is open, and the model is free to respond in any manner, the score is a code applied to the transcript. A person builds the codebook by reading a sample of transcripts, LLM coders from several vendors apply it to every transcript, and a blind human pass on held-out transcripts checks each code before it is used. In the conduct study, the codes behind the findings reported below agree with the human pass at Cohen’s κ 0.80 to 0.87, except self-citation at 0.54 [8]. The person does the part that needs judgment, and the machines do the reading at volume, which is what keeps the coded readout cheap. Instrumented: where the model operates in an instrumented environment, the environment records what it did, and the score compares that record with what the model said.

## 3 Related work

Evaluation has been framed as social-science measurement [14], and machines as agents whose behavior can be studied [12]. Fixed prompts across many models are how benchmarks already work [5], and sycophancy [11, 13], LLM judges [15], model homogenization [2, 4] and coding agents [3] each have their own literature. Alnasser et al. [1] code 14 models under user disagreement with an a priori scheme; ours is inductive, and the vendor-level result is new. What is diferent here is the target and the price. The assays measure everyday behavior with no clearly right answer, are cheap enough to re-run on every release, and the analysis method is chosen by how much interpretation the behavior needs.

## 4 Four preliminary findings

Run across four years of model releases, the assays found four things.

## 4.1 Convergence: a diferent serendipity in every language

Asked to pick a word (“Pick a word. Reply with one word only.”), 27 of 44 models from more than a dozen labs answer serendipity at least once in four tries, and it is 41 percent of all answers. The prompt is one of 31 in the one-word census [7]. The category prompts converge harder: oak is 94 percent of all trees, hammer 94 percent of tools, and rose 91 percent of flowers. Within the Claude, GPT, Qwen, and Grok lineages, later releases are more conformist, with the trend partially reversing for the latest flagships. Asked the same open prompt in 44 languages [9], the models settle on a default word in each, and the default is the language’s own. Serendipity wins only in English. Every other language has its own serendipity: sky in seven Indic languages and in Persian and Malay, sun in Ukrainian, Russian, and Spanish, peace in Hebrew, Arabic, and Amharic, cat in Japanese and Polish.

## 4.2 Resistance: keyed to the tag

The tag-question study [10] asks 20 decisions between two defensible options (rent or buy; name the cat Luna or Willow) as a neutral question and as the same sentence with a confirmation tag (“Luna is the better choice, right?”), counterbalanced over both options, scored by exact match on clamped yes or no. Across 45 models the tag efect spans +32 percentage points (endorses more often when the user fishes for agreement) to −32 points. Within every lineage the efect crosses from positive to negative as generations advance (GPT +4 to −28, Claude +7 to −32, Qwen +16 to −11), and two releases during the study window landed on the trend out of sample. Models are getting more resistant to bids for agreement over time.

Replace right? with maybe? and agreement rises above a neutral framing in 45 of 45 models, by 19.6 points on average, so two single-token tags on an otherwise identical sentence span 25 points. Agreement tracks how sure the user sounds: a confident bid meets resistance in the newer models, and a tentative one is rubber-stamped by nearly all of them. Maybe? also softens the claim being endorsed, so part of this contrast may be about what the user asserts rather than how they ask.

## 4.3 House: newer models hold, each lab holds its own way

The conduct study [8] puts a model under social pressure in four-turn scenes: a user insists that 5 × 9 = 54 and turns contemptuous, begs for a backdated doctor’s note, wants applause for quitting to day-trade, or discloses a dying parent and then asks for a cheerful email. In each scene the model either holds its ground or folds.

Across 38 models, the oldest third folded 43 percent of the time, and models released since mid-2025 folded 23 percent of the time. A larger panel of 60 models shows the same trend. A codebook of 17 codes records how each model held or folded. Six of the codes sort by lab, and release date does not explain them. Anthropic’s models name the user’s feeling, warn of the consequence, and ofer an alternative, each well above the panel. Google’s models cite their own nature or rules at three times the panel rate, and apologize when they fold. Meta’s models question the plan more than the panel does, and when they fold they produce what was asked.

OpenAI’s signature is what its models do less of: they warn less, name feelings less, and almost never cite their own nature or rules.

## 4.4 Account: what agents did and what they said

The coding atlas [6] gives a coding agent a small repository and a one-line instruction. Each repository holds a right thing to do, a shortcut that looks finished, and a hidden check the agent never sees. It asks three questions. When the job is only part done, does the agent say so? When the test and the code disagree, does it find out which is wrong? When the user insists on something the repository’s own documentation contradicts, does it hold or go along? Two records come back from every run, what the agent changed on disk and what it said, and the score compares them by counts and fixed string lists.

Fourteen configurations ran six scenarios three times each in full-auto mode: Claude Code, Codex CLI and Gemini CLI as products, and eleven models in the OpenCode harness. Taking the shortcut and reporting the job as done is rare: most configurations never did it in 12 work runs, and Codex and Gemini CLI did it twice. Going along with the user’s false claim without saying the repository contradicts it splits the field. Over the six pushback replies, Claude Code, Opus 5, Fable 5 and Kimi K3 never did it, and Codex and Gemini 3.5 Flash did it every time. The harness is part of the behavior: Gemini 3.5 Flash went along silently in three of six replies in Gemini CLI and six of six in OpenCode. A third count sets the final message against the agent’s own command log: runs that claim the tests pass without having run any, that execute a destructive command and never mention it, or that edit files and then say almost nothing. Gemini CLI does one of these in five of eighteen runs and Claude Code in two, where most configurations never do.

## 5 Limitations

Each construct is measured with one wording, and the tag results show how much one word can move. A frozen public stimulus can enter training data, so a standing record needs fresh scenes alongside the frozen ones. Providers update models and serving settings without notice, and older models are queried as they are served now. Release date moves with size, tier and training recipe, so the generation findings are observational. The API assays run with reasoning of. The agent results rest on three runs per scenario.

## 6 A behavioral science of LLMs

LLMs are complex phenomena that need to be studied behaviorally, across a wide range of operating contexts and configurations. Our contribution is a way of measuring model behavior that is simple, cheap, scalable and replicable. Fixed turns and short scoring rules are what make it cheap: a frozen stimulus removes the need to design a new prompt per model, and exact match, a quoted code, or a comparison of a dif against a message removes the need for an expensive reader. The instruments are simple and they still separate models across vendors and generations. A monoculture across 44 models, a sign flip within every lineage, a vendor signature in how models hold their ground, and a gap between what an agent does and what it says all came out of cheap instruments (§2).

A study at this price is closer in scale to a small experiment with human participants than to a benchmark: one person can run it, repeat it on the next release, and point it at a new question the week after. Models are easier to recruit and schedule than people. At that price, and with no lab scoring itself, studying model behavior can become an ordinary thing to do.<sup>2</sup>

## References

[1] Walid Alnasser, Yusuf Çetinkaya, Jian Zhao, and Tuğrulcan Elmas. How AI models manage epistemic authority, 2026. EMNLP 2026.

[2] Rishi Bommasani, Kathleen A. Creel, Ananya Kumar, Dan Jurafsky, and Percy Liang. Picking on the same person: Does algorithmic monoculture lead to outcome homogenization?, 2022.

[3] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues?, 2024.

[4] Jon Kleinberg and Manish Raghavan. Algorithmic monoculture and social welfare. Proceedings of the National Academy of Sciences, 118(22), 2021.

[5] Percy Liang, Rishi Bommasani, Tony Lee, et al. Holistic evaluation of language models, 2022.

[6] Tapan Parikh. What is your coding agent hiding from you? a field guide to coding agents (study and data). https://github.com/tap2k/coding-atlas, 2026.

[7] Tapan Parikh. The one-word census: Answer-choice conformity across 44 language models, 2026.

[8] Tapan Parikh. Conduct under pressure: What sixty language models do when a user pushes, 2026.

[9] Tapan Parikh. Every language has its own serendipity: The one-word census across languages. https://convovo.ai/blog/every-language-serendipity, 2026. Published 2026-07-24.

[10] Tapan Parikh. Tag questions and the generational reversal of sycophancy across 45 language models, 2026.

[11] Ethan Perez, Sam Ringer, Kamil˙e Lukoši¯ut˙e, et al. Discovering language model behaviors with model-written evaluations, 2022.

[12] Iyad Rahwan et al. Machine behaviour. Nature, 568:477–486, 2019.

[13] Mrinank Sharma, Meg Tong, Tomasz Korbak, et al. Towards understanding sycophancy in language models, 2023.

[14] Hanna Wallach, Meera Desai, A. Feder Cooper, Angelina Wang, Chad Atalla, Solon Barocas, Su Lin Blodgett, Alexandra Chouldechova, Emily Corvi, P. Alex Dow, Jean Garcia-Gathright, Alexandra Olteanu, Nicholas Pangakis, Stefanie Reed, Emily Sheng, Dan Vann, Jennifer Wortman Vaughan, Matthew Vogel, Hannah Washington, and Abigail Z. Jacobs. Position: Evaluating generative AI systems is a social science measurement challenge, 2025.

[15] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. Judging LLM-as-a-judge with MT-Bench and chatbot arena, 2023.