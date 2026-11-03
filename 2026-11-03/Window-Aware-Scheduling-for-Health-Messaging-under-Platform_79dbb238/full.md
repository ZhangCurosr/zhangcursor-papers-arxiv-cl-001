# Window-Aware Scheduling for Health Messaging under Platform Delivery Constraints: Evidence from an AI-Powered WhatsApp Maternal and Infant Health Chatbot in Nigeria

Adepoju Jeremiah Moyondafoluwa Omolaiye Samuel Adewuyi Thanni Okenla AbdulQahhar Abimbola Lily-Aimas Love Adereni Abiodun HelpMum Africa

## Abstract

WhatsApp lets a business send free-form messages only within 24 hours of a user’s last message; outside this customer service window, only pre-approved templates are delivered. Health chatbots inherit this rule, but its cost is rarely measured. We analyse 98,607 messages sent to 3,696 mothers by a WhatsApp maternal health chatbot in Nigeria, with a largely separate export as replication. No free-form message failed with the window open (0 of 35,500), and failure jumped by 93.0 percentage points (95% CI 87.0 to 99.1) at exactly 24 hours. Two flows produced 66.1% of failures, including an ofer to speak with a medical professional that, as built, reached 0.3 in 100 mothers whose window had closed. We formalise window-aware scheduling as a decision problem with an action of known zero value. Model-based ofline evaluation, with bounds for unconfirmed delivery, estimates that sending the ofer as an approved template would reach 81.5 to 86.8 in 100 such mothers, against 16.6 for waiting and 11.6 for a reply prompt. In a calibrated simulation, tabular Q-learning converges to the same template rule and does not beat it in any of 13 stress tests, and opting out does not rise with message volume. The main gain comes from respecting the platform rule, and it needs no learning. All outcomes are deliveries, not health outcomes.

## 1. Introduction

Nigeria accounted for an estimated 28.5% of the world’s maternal deaths in 2020 (World Health Organization, 2023). Mobile messaging has been tested widely as a way to reach pregnant women with health information: systematic reviews of maternal and newborn mHealth in low- and middle-income countries report improvements in some care outcomes, with evidence of mixed quality (Lee et al., 2016; Sondaal et al., 2016), a cluster-randomised trial in Zanzibar found that text messages increased skilled attendance at delivery among urban women (Lund et al., 2012), and text-message trials in Kenya improved antiretroviral adherence (Lester et al., 2010; Pop-Eleches et al., 2011). WHO guidance on digital health includes targeted communication to clients through mobile devices (World Health Organization, 2019). Some now run on WhatsApp, which is widely used in health research (Manji et al., 2021), and many take the form of chatbots, a class of tools reviewed by Laranjo et al. (2018).

WhatsApp imposes a delivery rule that most programmes inherit without measuring. A business may send free-form messages only within 24 hours of the user’s last message. Outside this window, only templates approved in advance by Meta are delivered, and a free-form message is rejected (Meta for Developers, 2026b). The rule matters because many important health messages are proactive. A reminder, a follow-up or an ofer to speak with a clinician is sent by a scheduler, often when the mother has not written for days. If the scheduler ignores the window, these messages fail silently: they are logged as sent and never arrive.

We measure that loss and evaluate how to sched ule around it, using the logs of a WhatsApp maternal health chatbot run by a non-profit organisation in Nigeria. We treat the window as a hard, observable constraint on which actions are available. This connects a practical delivery problem to bandits with changing action sets (Kanade et al., 2009; Kleinberg et al., 2010) and to adaptive interventions that depend on when a person can be reached (Nahum-Shani et al., 2018; K¨unzler et al., 2019). It also changes the learning problem: the platform fixes the value of one action at zero whenever the window is closed, and the window is visible before each decision.

Research questions. RQ1: How much proactive messaging fails because of the window, and is the efect sharp enough to identify from logs alone? RQ2: Which flows produce the loss, and why? RQ3: Which window-aware policies recover it: approved templates, waiting for the mother to write, or prompting a reply that reopens the window? RQ4: When today’s choice changes tomorrow’s window, does reinforcement learning improve on simple rules?

Contributions. We make four contributions. First, we measure the message loss attributable to the 24-hour window, identified by a sharp discontinuity and replicated in a second export (Section 6.1). Second, we diagnose its source: two flows produce two thirds of failures, and one of them blocks a clinician referral by design (Section 6.2). Third, we formulate window-aware scheduling with a knownzero action and state when waiting or a reply prompt can beat a template (Section 4). Fourth, we evaluate policies ofline by content type, with explicit delivery bounds and stated identification assumptions, and show in a calibrated simulation that Q-learning converges to the template rule (Section 6.3 and Section 6.4).

Figure 1 summarises the first finding. The broader lesson is that a learning method is only as useful as the structure left after known constraints are encoded; here, encoding one platform rule removes almost all of the loss.

## 2. Related Work

Maternal messaging and reach. Large programmes show that reach often limits impact. India’s Kilkari voice service lost much of its intended exposure to unanswered and short calls (Bashingwa et al. 2021), and South Africa’s MomConnect documented gaps in receipt and engagement (Barron et al., 2018; LeFevre et al., 2018). In Nigeria, a voice-messaging trial in Lagos found efects on antenatal care use (Osanyin et al., 2022), and SMS tools are being piloted for breastfeeding support (Musti, 2025); in Kenya, a cluster-randomised trial tested a digital platform across the maternal and newborn care continuum (Vatsa et al., 2024). Engagement with digital health tools commonly declines over time (Eysenbach, 2005). These studies measure exposure on SMS and voice channels, not the delivery rules of a messaging platform.

Chatbots and WhatsApp in health services. Reviews of conversational agents in health care describe many systems with limited or mixed evaluation (Laranjo et al., 2018; Milne-Ives et al., 2020), and users can be hesitant about chatbot-led health services (Nadarzynski et al., 2019). WhatsApp is widely used in health research (Manji et al., 2021). A diabetes chatbot in Cape Town reported messages “queued up in the system and eventually dropped” and no provider error logs (Mash et al., 2022). A MomConnect trial protocol notes the template requirement as a design constraint (IDinsight, 2023), and practitioners report high blocking of marketing templates and rising costs (Hughey, 2026; Amazon-Brown, 2026). We found no peer-reviewed study that quantifies how much health messaging the 24-hour window blocks.

Timing, receptivity and adaptive interventions. Work on just-in-time adaptive interventions studies when people are available and receptive (Nahum-Shani et al., 2018; K¨unzler et al., 2019; Mishra et al., 2021), and studies of phone notifications show that attention to a message depends on its content and the user’s context (Pielot et al., 2014; Mehrotra et al., 2016). Micro-randomised trials estimate the efect of prompts sent at diferent moments (Klasnja et al., 2015, 2019), and contextual bandits, a method also used for news recommendation (Li et al., 2010), and reinforcement learning personalise which prompt to send and when (Tewari and Murphy, 2017; Yom-Tov et al., 2017; Liao et al., 2020; Trella et al., 2022; Mandyam et al., 2024). In maternal health, restless bandits allocate scarce health-worker calls in a large programme in India (Mate et al., 2022; Wang et al., 2023; Verma et al., 2026), and bandits have been used to choose call times (Lalan et al., 2024; Dasgupta et al., 2025). These studies treat availability as a probabilistic state learned from behaviour. On WhatsApp, availability for free-form messages is a hard state set by the platform and visible in the log.

Changing action sets and ofline evaluation. Sleeping bandits model actions available only in some rounds, revealed before each choice (Kanade et al., 2009; Kleinberg et al., 2010); bandits with knapsacks add budget constraints (Badanidiyuru et al., 2018); Lattimore and Szepesv´ari (2020) give an overview.

![](images/ab728683e69b1f6f84f1b69e7863116b50af799d08eb199b95d95958b3f1a517.jpg)

![](images/85b03ffb9ba4881f2395fd792019c1a43a67076ce43cd53f61cfae140d9bc139.jpg)  
Figure 1: Bot-message failure by hours since the mother’s last message h, main export (observed). (a) Failure rate by bin with Wilson 95% intervals, among the 65,193 messages with an earlier message observed (the 33,414 without one have no h; Table 2). (b) 2-hour bins and the local linear fit on each side of 24 hours (bandwidth 24 hours, 95% band); the template jump is 1.0 point (CI −0.7 to 2.7).

Our setting is a special case in which availability is known from the window and the reward of an unavailable action is exactly zero. Of-policy evaluation by importance sampling (Precup et al., 2000; Li et al., 2011) or doubly robust estimators (Dud´ık et al., 2011; Jiang and Li, 2016; Thomas and Brunskill, 2016; Mandyam et al., 2026) needs logged action probabilities, and guidelines for reinforcement learning in health, motivated by applications such as sepsis treatment (Komorowski et al., 2018), warn that evaluation from observational logs can mislead without them (Gottesman et al., 2019). Our logs have none, so, as in other of-policy work on clinical logs (Ji et al., 2025), we state the assumptions our model based values need.

Gap and contribution. Prior work on maternal messaging measures reach on SMS and voice, and work on timing treats availability as something to learn. We measure a reach loss caused by an observable platform rule, show that it is identified by a discontinuity in routine logs, and show that once the rule is encoded, the remaining decision is simple enough that learning does not improve on a rule.

## 3. Setting, Data and Cohort

## 3.1. The service

The service runs in English on the WhatsApp Business Platform through a messaging provider. Mothers receive scheduled content and can ask questions at any time. The main flows are: weekly pregnancy guides, sent as approved templates; an ofer to speak with a medical professional, a free-form interactive message with Yes and No buttons chained about a second after the weekly guide; a free-form error follow-up (“Please let me know if you ever have any questions”), sent about five minutes after the ofer in the same weekly chain; vaccination, nutrition and broadcast templates; and free-form conversational replies. Mothers can opt out of individual streams.

## 3.2. Platform rules during the study

Throughout the study period, free-form messages could be delivered only inside the window, which opens or resets only when the user messages the business (Meta for Developers, 2026b). Templates could be sent at any time, subject to approval and, for marketing templates, per-user limits (Meta for Developers, 2026a). Service conversations became free from

Table 1: The two exports and the analysis cohorts (observed counts). <sup>a</sup>Bot-initiated freeform messages with 7 days of follow-up. <sup>b</sup>Delivered (not failed) bot-initiated messages. Obs.: earlier message observed; none obs.: no earlier message observed.
<table><tr><td></td><td>Main</td><td>Replication</td></tr><tr><td>Bot messages</td><td>98,607</td><td>23,138</td></tr><tr><td>free-form</td><td>50,574</td><td>12,266</td></tr><tr><td>templates</td><td>48,033</td><td>10,872</td></tr><tr><td>Messages from mothers</td><td>29,297</td><td>7,586</td></tr><tr><td>Mothers</td><td>3,696</td><td>927</td></tr><tr><td>Period</td><td>2023 to 2026</td><td>2023 to 2026</td></tr><tr><td>Mothers in both exports</td><td>131</td><td>131</td></tr><tr><td>Failed bot messages</td><td>21.8%</td><td>21.0%</td></tr><tr><td>No delivery receipt</td><td>3.0%</td><td>3.6%</td></tr><tr><td>Decision pointsª</td><td>19,827</td><td>3,457</td></tr><tr><td>mothers</td><td>2,230</td><td>396</td></tr><tr><td>closed, obs.</td><td>4,225</td><td>367</td></tr><tr><td>closed, none obs.</td><td>8,526</td><td>2,072</td></tr><tr><td>Burden sampleb</td><td>49,256</td><td>10,789</td></tr></table>

November 2024, templates moved to per-message pricing in July 2025, and from 1 October 2026 inwindow service messages became chargeable after 1,000 free per month (Meta for Developers, 2026b). Almost all our data predate any charge for in-window messages. Published utility rates for Nigeria difer between sources (US\$0.0101 and US\$0.0067 per message; Techpoint Africa, 2026; Ominiflow, 2026); we use the higher rate.

## 3.3. Data and cohort

The operator supplied two exports of conversation logs (Table 1). Each row holds a mother’s message, a bot message or both, with timestamps, flow, message type and final status (read, delivered, sent, failed or unknown). The exports contain no demographic fields. Recorded pregnancy weeks for guide recipients have a median of 33, so most mothers in the data are late in pregnancy or have recently given birth.

Selection and truncation. Both exports contain only mothers in contact in the export month, so mothers who had already left cannot appear. In the main export, each history is also left-truncated. For the 131 mothers in both exports, the main export contains every row of the replication export inside its own period but none of 1,973 earlier bot rows and 1,106 earlier inbound rows. Retained histories are contiguous and recent, and the selection rule is undocumented. Results therefore describe mothers still on the send list at the end of each export, not all mothers who ever enrolled.

Cohorts. The window analysis uses all bot messages. The policy evaluation uses bot-initiated freeform messages with at least 7 days of follow-up before the end of the export (decision points). The burden analysis uses delivered bot-initiated messages. The replication export is analysed with the same code and no changes.

## 4. Problem Formulation

For bot message i sent at $t _ { i } ,$ let $\tau _ { i }$ be the mother’s most recent message at or before $t _ { i } ,$ taken from every row with a message from her, and $h _ { i } = t _ { i } - \tau _ { i }$ in hours. We write $W _ { i }$ for the true window state, which a live scheduler knows, and $\tilde { W } _ { i } = \mathbf { 1 } \{ h _ { i } \leq 2 4 \}$ for the state observed in the export, with ${ \ddot { W } } _ { i } ~ = ~ 0$ when no earlier message is observed. Histories are contiguous, so $\tilde { W } _ { i } = W _ { i }$ whenever an earlier message is observed; without one, the window may still be open (Appendix B).

A scheduler wants to deliver a proactive message at time t with context x and window state W. It can send free-form now (F), send an approved tem plate now (T), wait and send free-form when she next writes within H hours (D), or add a reply button to the scheduled template and send the follow-up only if she replies within 24 hours $( T ^ { + } )$ . A message delivered u hours late is worth $e ^ { - \delta u }$ , with $\delta = \ln 2 / \ell$ for a half-life ℓ of 6 hours (urgent) or 48 hours (routine). A template costs $^ { c , }$ as a share of a message’s value. The platform fixes P(delivered | $F , W = 0 ) = 0 ;$ we also take P(delivered | $F , W = 1 ) = 1$ , as observed (0 of 35,500 failed). W is known before acting, and x denotes the other features available to a learner, which exclude h. Propositions 1 to 3 concern one proactive message with actions F, T, D or nothing; Proposition 4 concerns a follow-up chained to a scheduled template. Proofs are in Appendix A.

We measure a policy by its expected value: discounted deliveries minus template costs. In-window free-form messages cost nothing during the study period. Write $p _ { T } ( x )$ for the probability that a template sent with the window closed is delivered.

Proposition 1 (Reach ceiling) A policy that always sends free-form achieves at most $P ( W = 1 )$ . A window-aware policy that sends F when W = 1 and, when $W = 0$ , the best of T, D and sending nothing achieves exactly

$$
P ( W = 0 ) \mathbb { E } \big [ \operatorname* { m a x } \{ p _ { T } ( x ) - c , \ V _ { D } ( x ) , \ 0 \} \ | \ W = 0 \big ]
$$

more, where $V _ { D } ( x )$ is the value of waiting.

Proposition 2 (Known-zero action) A learner that observes W and masks $F$ when $W = 0$ incurs no regret or exploration from that action. $I f \left( x _ { t } , W _ { t } \right)$ are drawn independently across rounds, a learner whose policy depends on x and its history but not on W has expected regret of at least gn after n rounds against the window-aware oracle. The constant g is positive whenever, on a set of contexts with positive probability, both window states can occur and the best action difers between them.

Proposition 3 (When to wait) Suppose the mother next writes after a time R that is exponential with rate $\lambda ,$ and a free-form message sent when she writes is delivered. The value of waiting up to H hours, with no fallback, is

$$
V _ { D } = \mathbb { E } \big [ e ^ { - \delta R } \mathbf { 1 } \{ R \leq H \} \big ] = \frac { \lambda } { \lambda + \delta } \big ( 1 - e ^ { - ( \lambda + \delta ) H } \big ) ,
$$

and waiting beats a template if and only if $V _ { D } \geq$ $p _ { T } ( x ) \ - \ c .$ For any distribution of R, $\begin{array} { r l } { V _ { D } } & { { } \leq } \end{array}$ $P ( R \leq H )$ , and in the exponential case also $V _ { D } \ \leq$ $\lambda \operatorname* { m i n } ( H , 1 / \delta )$

Proposition 4 (Prompting a reply) Suppose a reply button on the scheduled template draws a reply within 24 hours with probability q(x) per scheduled send (including the chance that the template is not delivered), after a lag L, and the follow-up then sent in the open window costs $c _ { s } \geq 0$ . Then $T ^ { + }$ beats $\textit { T i f }$ and only if

$$
q ( x ) \big ( \mathbb { E } [ e ^ { - \delta L } \mid r e p l y ] - c _ { s } \big ) \geq p _ { T } ( x ) - c ,
$$

and $q ( x ) \geq p _ { T } ( x ) - c$ is a necessary condition.

So the window should be encoded, not learned, and neither waiting nor a reply prompt can be worth more than the chance that the mother writes within the deadline (Figure 4 in Appendix A).

## 5. Methods

We keep three kinds of result apart: observed outcomes from the logs, modelled counterfactuals for actions not taken, and simulated outcomes from a calibrated model, and, as in observational health studies that separate prediction from causal claims (Noroozizadeh et al., 2025), we make causal claims only where the design supports them (the discontinuity). Full definitions are in Appendix B to Appendix F.

## 5.1. Identifying the loss

We report failure by format and window with Wilson intervals (Wilson, 1927), and estimate the jump at 24 hours with a local linear probability model (Imbens and Lemieux, 2008; Lee and Lemieux, 2010; Cattaneo et al., 2020),

$$
Y _ { i } = \alpha + \beta \mathbf { 1 } \{ h _ { i } > 2 4 \} + \gamma r _ { i } + \eta r _ { i } \mathbf { 1 } \{ h _ { i } > 2 4 \} + \varepsilon _ { i } ,
$$

where $Y _ { i } = 1$ if message i failed, $r _ { i } = h _ { i } - 2 4$ and $\beta$ is the jump, with bandwidths 6 to 72 hours and standard errors clustered by mother (Cameron and Miller, 2015). We report conventional intervals across band widths rather than bias-corrected intervals (Calonico et al., 2014); with a jump close to the maximum possible, the choice does not afect the conclusion. We add placebo cut-ofs with the same model, the same model for templates, a diference in discontinuities between formats (Grembi et al., 2016), and Fisher’s exact test between 18 and 30 hours. Messages with no earlier message observed have no $h _ { i }$ and are excluded; because $h _ { i } \geq 0$ , bandwidths above 24 hours widen the window only to the right. Messages are scheduled by flow, not by the time of the mother’s last message, which supports continuity of other determinants of failure at 24 hours.

## 5.2. Mechanism

We count failures by flow and examine the timing of the medical-professional ofer relative to the mother’s last message. Meta error codes are not in the exports. We infer them from format (re-engagement error 131047 for failed free-form messages, undeliverable 131026 for failed templates), and treat the inferred split as descriptive.

## 5.3. Return to the chat

Conversations are runs of the mother’s own messages, split at gaps over 12 hours. Kaplan-Meier curves (Kaplan and Meier, 1958), censored at the end of the export, give time to the next conversation, and a Cox model (Cox, 1972) with a mother-cluster bootstrap relates return to the number of earlier conversations.

## 5.4. Ofline policy evaluation

Decision points are bot-initiated free-form messages with 7 days of follow-up. A logistic model, cross-fitted in two folds by mother (Chernozhukov et al., 2018), gives template delivery ${ \hat { p } } _ { T } ( x )$ , trained on closedwindow templates. Features are whether an earlier message was observed, log(1 + earlier messages), log(1 + h) with h capped at 60 days and set to the cap when no earlier message is observed, weekend and evening. We fit it twice: with “not failed” as the outcome, an upper bound that counts unconfirmed sends as delivered, and with confirmed delivery, a lower bound. With the window closed, the logged policy is what the chatbot did; the template rule sends a template; the wait rule waits up to 7 days; the threshold rule applies Proposition 3 with a return rate estimated from a cross-fitted return model; and a hindsight bound takes the better of the template’s expected value and waiting’s realised value. No approved template existed for the main contents afected, so template values are a design counterfactual. We therefore report results by content type and add a suppress option for content whose value is doubtful when late. Each of 200 bootstrap replicates resamples mothers in both the decision points and the template training data and refits both models. A replay with linear Thompson sampling (Agrawal and Goyal, 2013; Chapelle and Li, 2011) compares a learner that masks F when the window is closed, one that sees the window but does not mask, and one that cannot see it.

Identification assumptions. The modelled template values rest on four assumptions, which we state rather than test. The first is transport: a new approved template for the ofer or follow-up would be delivered like existing closed-window templates with the same features, which we check against the observed delivery of the scheduled weekly-guide template sent to the same mothers. The second is positivity: closed-window templates, 45,538 of them in the training data, cover the feature range of the decision points. The third is that waiting does not change return, so under the wait rule a mother writes again when the logs say she did. The fourth is that value does not depend on format, so a message delivered as a template is worth the same as the free-form version. It is the weakest: a template can carry the same buttons, but whether mothers tap them, and whether Meta approves it as utility, is untested, so our outcome is reach, not connection to a clinician. Importance-weighted and doubly robust estimators (Li et al., 2011; Dud´ık et al., 2011; Jiang and Li, 2016) are not available without logged propensities, so none of our ofline values is design-unbiased. They are model-based estimates, reported with bounds.

## 5.5. Simulation and Q-learning

To let today’s action change tomorrow’s window, we simulate mothers for 26 weeks. Gaps between conver sations follow a two-component exponential mixture fitted to the censored gaps. Each week the chatbot sends the guide as a template and wants to deliver the medical ofer. Free-form is delivered only with the window open; templates are delivered at the scheduled guide’s observed rate; and a delivered template draws a genuine reply (excluding opt-out taps) at the observed rate, which reopens the window. In the logs, opting out removes a mother from one stream, not the service: of 265 first opt-outs with 30 days of follow-up, 259 (97.7%) were followed by further proactive messages. The main setting therefore leaves the guide and ofer stream untouched by opt-outs; a global model in which each delivered message ends all contact with probability $\rho = 1 . 6 9 \%$ gives an upper bound on burden.

We compare the logged behaviour, waiting, $T ^ { + }$ a hybrid (button, then template after 6 hours), the template rule, the threshold rule and tabular Q-learning (Watkins and Dayan, 1992; Sutton and Barto, 2018). Q-learning sees time since her last message (five bands) and conversations in the last 28 days (three bands). Its weekly reward is the guide delivered plus the discounted ofer value, minus 0.05 per delivered message, with step size $n ( s , a ) ^ { - 0 . 6 }$ (Even-Dar and Mansour, 2003), $\gamma = 0 . 9 7$ and exploration decaying from 0.3 to 0.01 over 150,000 training mothers. We validate the simulator against log statistics it was not fitted to, and run 13 single-input stress tests, retraining Q-learning each time. Nothing was tuned to favour any strategy. With weekly decisions and no fatigue state, sending in the main setting has no future cost, so the template rule is optimal there by construction; the test is whether replies that restart the return process, or exit under the global model, change this (Appendix D).

Table 2: Failure by format and window state, main export (observed). Unconfirmed: status “sent” or “unknown”. None obs.: no earlier message observed.
<table><tr><td>Format</td><td>Window</td><td>n</td><td>Failed</td><td>Unconf.</td></tr><tr><td rowspan="2">Free-form</td><td>Open</td><td>35,500</td><td>0.0%</td><td>0.1%</td></tr><tr><td>Closed</td><td>6,267</td><td>98.6%</td><td>0.0%</td></tr><tr><td rowspan="4">Template</td><td>None obs.</td><td>8,807</td><td>90.9%</td><td>0.0%</td></tr><tr><td>Open</td><td>1,690</td><td>0.1%</td><td>0.0%</td></tr><tr><td>Closed</td><td>21,736</td><td>3.3%</td><td>4.3%</td></tr><tr><td>None obs.</td><td>24,607</td><td>26.6%</td><td>7.8%</td></tr></table>

## 5.6. Burden and send time

A learner could beat a fixed rule if sending more messages made mothers more likely to quit, a dy namic the rule ignores. We test whether opting out or reading within 24 hours changes with the messages a mother received in the previous 7 days, comparing each mother with herself (mother, flow and month fixed efects, errors clustered by mother) and, because opting out cuts later volume and biases withinmother estimates downward, with a pooled discretetime model for her first opt-out. Within batched broadcasts, we compare reading by send time with campaign fixed efects; batch order is undocumented, so this is an association. Models were fitted with statsmodels (Seabold and Perktold, 2010).

## 6. Results

## 6.1. The window explains most of the loss

Of 98,607 bot messages, 21,456 (21.8%) failed. Failures followed the window (Table 2, Figure 1). No free-form message failed with the window open, and only one of all failures occurred with it open, a template. Templates failed at 3.3% when an earlier message was observed and 26.6% when not, consistent with numbers no longer on WhatsApp rather than the window.

Between 18 and 30 hours, 0 of 114 free-form messages failed before 24 hours and 72 of 79 after (Fisher’s exact test, $p = 2 . 1 \times 1 0 ^ { - 4 5 } )$ . The jump at 24 hours is 93.0 percentage points (95% CI 87.0 to 99.1), and between 89.1 and 95.1 across bandwidths from 6 to 72 hours (Figure 6 and Table 5 in Appendix B). Templates show none (1.0 point, CI −0.7 to 2.7); the diference in discontinuities is 92.0 points (CI 85.6 to 98.4). With the same local linear model, no placebo cut-of shows a positive jump; every interval includes zero, the one at 48 hours only just (−14.1, CI −28.1 to 0.0), and template placebos are all null (Appendix B).

Replication. In the replication export, 0 of 9,546 free-form messages failed with the window open, and between 18 and 30 hours 0 of 19 failed before 24 hours against 18 of 18 after (Appendix G).

## 6.2. Two flows produce two thirds of the loss

The error follow-up (9,525 failures, 44.4% of all) and the medical-professional ofer (4,654, 21.7%) account for 66.1% of failures (Figure 7 in Appendix B). Both send free-form content to mothers whose window has closed.

The ofer fails by design. It is chained a second after the weekly guide, a template, and a template does not open the window. Of 4,869 ofers, 95.6% failed: all 2,080 sent with the window closed and 2,574 of 2,583 with no earlier message observed, but none of 206 sent with it open (Figure 5 in Ap pendix B). Where an earlier message was observed, the median time since it was 181 hours. In practice, an ofer to connect a pregnant woman with a clinician reached almost no one.

Mothers return slowly. Across 9,278 conversations, the median time to the next one was 51 days. After a first conversation, 13.6% of mothers wrote again within 7 days and 36.3% within 30 days (Figure 8 in Appendix C). Waiting for the window to reopen therefore rarely works.

## 6.3. Scheduling-policy evaluation

The evaluation uses 19,827 decisions from 2,230 mothers; 12,751 were sent with the window closed or no earlier message observed (4,225 and 8,526): 64.4% error follow-ups, 30.8% medical ofers and 4.8% other content (Table 3). Template values are modelled.

Medical ofer. As a template, the ofer would reach an estimated 81.5 to 86.8 in 100 mothers, against 0.3 as logged, 11.6 (CI 10.5 to 13.0) for a reply button and 16.6 for waiting (Figure 2). The replybutton value counts any message from the mother within 24 hours of the ofer; the guide had no button, so this understates a button (Appendix C). Applying the scheduled-guide template’s observed delivery instead of the model gives 81.5 to 88.2, so the result does not depend on the model.

Table 3: Closed-window decisions by content, main export, per 100 (95% intervals from 200 mother-cluster bootstrap replicates that refit the delivery models). Logged values are observed; wait values replay the logs under the assumption that waiting does not change return; template values are modelled.
<table><tr><td>Content</td><td>n</td><td>Logged</td><td>Template, confirmed</td><td>Template, not failed</td><td>Wait 7 days</td></tr><tr><td>Medical offer</td><td>3,933</td><td>0.3 (0.2 to 0.5)</td><td>81.5 (79.8 to 83.0)</td><td>86.8 (85.7 to 87.9)</td><td>16.6 (14.8 to 18.1)</td></tr><tr><td>Error follow-up</td><td>8,209</td><td>1.5 (1.2 to 2.0)</td><td>80.9 (79.3 to 82.5)</td><td>86.4 (85.2 to 87.5)</td><td>17.2 (15.5 to 18.6)</td></tr><tr><td>Other content</td><td>609</td><td>99.5 (98.8 to 100)</td><td>67.5 (65.2 to 69.6)</td><td>75.0 (73.1 to 77.2)</td><td>88.8 (84.1 to 92.8)</td></tr></table>

![](images/dfff0c2680f691066523691274bf52351c6e869cc8829cdff4f6f2f6cd6134e0.jpg)  
Figure 2: Options for the medical-professional offer per 100 closed-window decisions (n = 3,933), main export, with 95% intervals (right). Logged: observed. Wait: offline replay of observed returns. Reply proxy: any message within 24 hours (no button was deployed). Template: hypothetical approved-template counterfactual (modelled, open markers), bounded by confirmed and not-failed delivery. Deliveries, not clinician contacts.

Error follow-up. No template existed for it, it would probably not qualify as a utility template, and its value days later is doubtful. The realistic comparator is suppression, so we do not count follow-up deliveries as a gain for the template rule.

Other content. 603 of 609 had no earlier message observed and were delivered as logged. They come from onboarding and menu flows in which the mother’s opening message is missing from the export, the clearest sign that the “no earlier message observed” group contains open windows.

Pooled results. The template rule delivers 86.0 per 100 closed-window decisions (80.4 confirmed only), against 5.8 logged and 20.4 for waiting. The threshold rule never waits, so it equals the template rule; the hindsight bound reaches 87.7. Waiting overtakes the template only when a template costs 75% to 85% of a message’s value, depending on the halflife (Figure 9). Cross-fitted delivery predictions are within 7.5 points of observed delivery in every decile (Table 7). On the 4,225 decisions with an earlier message observed, where the window state is exact, the template reaches 97.3 to 99.0 per 100, waiting 28.8 and the logged policy 0.1 (Table 8).

Learning on the logs. In the replay, a learner that masks F has regret 1.4 per 100 decisions, one that sees the window without masking 1.5, a window-blind learner 2.3 and the logged behaviour 49.7 (Figure 10). The blind learner almost never sends free-form into a closed window (0.3%); its extra regret comes from sending templates when the window is open (50% of those decisions), as Proposition 2 predicts.

## 6.4. Simulation: learning converges to the rule

Under the logged behaviour, the simulator reproduces outcomes it was not fitted to: ofer failure is 98.7% simulated against 99.4% observed. It slightly understates open windows (1.5% against 3.1% of ofers) and overstates later returns (43.2% against 37.8% within 30 days), both of which favour waiting (Figure 11).

The template rule delivers 90.3 ofers per 100 due, against 1.3 for the logged behaviour, 10.2 for waiting and 18.5 for a reply button (Figure 3, Table 10). The hybrid delivers 91.7 but loses about half their value to delay (discounted value 51.2). After about 10,000 training mothers, Q-learning sends free-form when the window is open and a template in every other state. Its reward then equals the template rule’s (44.6 per mother) to within simulation noise, and it never finds a better policy. Across 13 single-input stress tests, the template rule, the threshold rule and Q-learning tie (largest gap 0.05 reward units) and beat every other strategy; the closest challenger is the hybrid for routine content, 1.4 units behind (Figure 13). Under the global opt-out bound, delivering the ofer raises the share who leave by week 26 from 33.5% to 55.1%, and the template rule still has the highest reward. Allowed to suppress the ofer, Q-learning matches the template rule until exit per message reaches eight times the bound (Table 11). Each extra ofer costs about one delivered message, US\$0.010 at the rate used here.

![](images/12b1647df3ef600c46d903ddfea32d6efdc24fb0c52c72fdf3b0ba38e7b9aa62.jpg)  
Figure 3: Simulated strategies over 26 weeks: (a) ofers delivered and (b) discounted value per 100 ofers due, main setting (no mother leaves); (c) share who leave by week 26 under the global opt-out model, an upper bound, not observed attrition. Means with 95% intervals over 8 seeds of 3,000 mothers. The template rule, threshold rule and Q-learning reach the same policy.

## 6.5. Burden and send time

Across 49,256 delivered messages with 785 opt-outs, within-mother estimates were negative: per log unit of templates in the previous week, opting out within 24 hours changed by −2.3 points (−1.6 per doubling of one plus the count), controlling for recency (Table 12). These are biased downward, because opting out cuts later volume; a pooled model of the first optout shows no association (odds ratio 0.94 per log unit, CI 0.75 to 1.17). We find no rise, not a fall. Reading within 24 hours fell by 2.2 points per doubling, from about 60%. Within 17 batched broadcasts, messages sent after 22:00 were read within 6 hours 23.0 points less often than morning sends (CI −30.9 to −15.2), and by 19.5 to 23.8 points when each campaign is dropped; 193 of the 266 night messages come from one campaign. Prior reading and contact were balanced (joint p = 0.86 and 0.36). Reading eventually did not difer, so late sends mainly wait until morning (Figure 14).

## 7. Discussion

## 7.1. The rule is the problem, and the fix

The window explains most failed messages, with a jump of about 93 points at 24 hours, replicated in a second export. One of the two flows behind the loss is a clinician referral that, as built, could reach only mothers who had written in the previous day. Once a suitable template exists, a template reaches 81 to 87 in 100 mothers whose window has closed, while waiting or prompting a reply reaches 12 to 17. Proposition 3 explains why: waiting can deliver no more often than the mother writes within the deadline, which happened for about 20 in 100 closed-window decisions.

## 7.2. Implications for machine learning in health

A Q-learning scheduler with the same choices learns the template rule and stops: the platform rule makes the most important part of the decision known in advance. When a rule fixes the value of an action and the rule is observable, the action needs neither exploration nor estimation, and a learner that cannot see the rule pays a constant cost in every round (Proposition 2).

A learner can beat a fixed rule only where the problem has structure the rule ignores. We tested the most plausible candidate, burden that builds with volume, and found none. Learning may still help with other decisions, such as which content to send or at what hour. We recommend removing every action whose value a platform fixes before learning, then asking what structure is left. This follows guidelines that ask what a learned policy does diferently from current practice (Gottesman et al., 2019).

## 7.3. Practical deployment considerations

What follows are design recommendations, not tested interventions. The scheduler should check the window before every proactive free-form message and send a template, or nothing, when it is closed. The medical ofer needs its own utility template; Meta decides the category and may recategorise templates (Hughey, 2026), so approval is a risk to plan for. The error follow-up can simply be suppressed when the window is closed. Operators can monitor failures by flow and hours since the mother’s last message. Costs are small: about US\$0.47 per mother over six months in simulation, against US\$0.24 as logged, at US\$0.0101 per message and counting in-window service messages as charged, as under pricing from October 2026 (Meta for Developers, 2026b). Billed as marketing (an indicative US\$0.0516; Ominiflow, 2026), the ofer would raise this to about US\$1.43 and face per-user limits; the non-profit exemption for service messages (Meta for Developers, 2026b) barely changes it.

## 7.4. Limitations

Both exports contain only mothers still in contact, with left-truncated histories selected by an undocumented rule; for shared mothers, 12.2% of messages labelled “no earlier message observed” went to mothers who had written. Error codes are inferred, but the window efect is identified from timing. Template results assume a new template delivered like existing ones, and 3% of messages lack a receipt, so we report bounds. The nearly deterministic logged policy ruled out importance weighting. The simulator is cal ibrated to the same logs and gives learning little to exploit, so it shows that the obvious dynamics do not change the answer, not that no richer dynamics could. Whether a template ofer leads mothers to a clinician as often as a free-form one would is untested. All outcomes are deliveries, reads or replies, and nothing here shows that clinical outcomes improved. Finally, the evidence comes from one service in one country and one language.

## 7.5. Ethical considerations

The loss falls unevenly. Mothers who rarely write, whose phones are of or whose numbers no longer work are the least likely to receive proactive messages, and they may most need a referral; a template rule helps the first group only, and 26.6% of templates to mothers with no earlier message observed still failed. More messages could also add burden if opting out ended all contact. The analysis used deidentified logs (salted hashes, no names or message text, aggregates only), and no mother was contacted for this study.

## 8. Conclusion

In a Nigerian maternal health chatbot, WhatsApp’s 24-hour window explains most failed messages; approved templates would recover most of the loss, and reinforcement learning rediscovers the simple rule. The first step is not a smarter algorithm but checking the window.

## Data and Code Availability

The data are conversation logs from a WhatsApp maternal health chatbot operated by a non-profit organisation in Nigeria. They contain phone numbers, names and the text of health questions, and are not publicly available; access depends on the operator’s data-sharing terms. Analysis code, tests, aggregate result tables and figures are provided as a code archive under the MIT licence, with a README describing the pipeline, a reproducibility manifest (software versions, random seeds, sample counts and checksums) and unit tests that run without the pri vate data.

## Institutional Review Board (IRB)

The research does not require IRB approval. It is a retrospective secondary analysis of de-identified logs collected during routine operation of the service; no mother was contacted and no message was sent for this study.

## Acknowledgments

We thank HelpMum Africa for developing and operating the chatbot service and supporting this research.

## References

Shipra Agrawal and Navin Goyal. Thompson sampling for contextual bandits with linear payofs. In Proceedings of the 30th International Conference on Machine Learning, 2013.

Isabelle Amazon-Brown. Meta wants to charge for WhatsApp messages: Chatbot builders beware. ICTworks, 2026. 21 July 2026. https://www.ictw orks.org/meta-wants-to-charge-for-whatsap p-messages/.

Ashwinkumar Badanidiyuru, Robert Kleinberg, and Aleksandrs Slivkins. Bandits with knapsacks. Journal of the ACM, 65(3), 2018.

Peter Barron, Joanne Peter, Amnesty E. LeFevre, et al. Mobile health messaging service and helpdesk for South African mothers (MomConnect): History, successes and challenges. BMJ Global Health, 3(Suppl 2):e000559, 2018.

Jean Juste Harrisson Bashingwa, Diwakar Mohan, Sara Chamberlain, et al. Assessing exposure to Kilkari: A big data analysis of a large maternal mobile messaging service across 13 states in India. BMJ Global Health, 6(Suppl 5):e005213, 2021.

Sebastian Calonico, Matias D. Cattaneo, and Rocio Titiunik. Robust nonparametric confidence intervals for regression-discontinuity designs. Econometrica, 82(6):2295–2326, 2014.

A. Colin Cameron and Douglas L. Miller. A practitioner’s guide to cluster-robust inference. Journal of Human Resources, 50(2):317–372, 2015.

Matias D. Cattaneo, Nicol´as Idrobo, and Roc´ıo Titiunik. A Practical Introduction to Regression Discontinuity Designs: Foundations. Cambridge University Press, 2020.

Olivier Chapelle and Lihong Li. An empirical evaluation of Thompson sampling. In Advances in Neural Information Processing Systems 24, 2011.

Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased ma chine learning for treatment and structural parameters. The Econometrics Journal, 21(1):C1–C68, 2018.

David R. Cox. Regression models and life-tables. Journal of the Royal Statistical Society: Series B, 34(2):187–220, 1972.

Arpan Dasgupta, Mizhaan Prajit Maniyar, Awad hesh Srivastava, Sanat Kumar, Amrita Mahale, Aparna Hegde, Arun Suggala, Karthikeyan Shanmugam, Milind Tambe, and Aparna Taneja. Learning to call: A field trial of a collaborative bandit algorithm for optimizing call timing in mobile ma ternal health. In Proceedings of the 10th Machine Learning for Healthcare Conference, volume 298 of Proceedings of Machine Learning Research, 2025.

Miroslav Dud´ık, John Langford, and Lihong Li. Doubly robust policy evaluation and learning. In Proceedings of the 28th International Conference on Machine Learning, 2011.

Eyal Even-Dar and Yishay Mansour. Learning rates for Q-learning. Journal of Machine Learning Research, 5:1–25, 2003.

Gunther Eysenbach. The law of attrition. Journal of Medical Internet Research, 7(1):e11, 2005.

Omer Gottesman, Fredrik Johansson, Matthieu Komorowski, Aldo Faisal, David Sontag, Finale Doshi-Velez, and Leo Anthony Celi. Guidelines for reinforcement learning in healthcare. Nature Medicine, 25(1):16–18, 2019.

Veronica Grembi, Tommaso Nannicini, and Ugo Troiano. Do fiscal rules matter? American Economic Journal: Applied Economics, 8(3):1–30, 2016.

Jake Hughey. Data observability in WhatsApp services for social impact. The Agency Fund, 2026. 4 June 2026. https://theagencyfund.substack.c om/p/data-observability-in-whatsapp-servi ces.

IDinsight. Pre-analysis plan: Testing the efectiveness of quality-based WhatsApp messaging on vaccine hesitancy and childhood immunization in a national maternal messaging program (MomConnect), 2023. ClinicalTrials.gov NCT05505760.

Guido W. Imbens and Thomas Lemieux. Regression discontinuity designs: A guide to practice. Journal of Econometrics, 142(2):615–635, 2008.

Zongliang Ji, Andre Carlos Kajdacsy-Balla Amaral, Anna Goldenberg, and Rahul G. Krishnan. ExOS-ITO: Explainable of-policy learning with side in formation for intensive care unit blood test orders. In Proceedings of the Conference on Health, Inference, and Learning, volume 287 of Proceedings of Machine Learning Research, pages 337–368, 2025.

Nan Jiang and Lihong Li. Doubly robust of-policy value evaluation for reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, 2016.

Varun Kanade, H. Brendan McMahan, and Brent Bryan. Sleeping experts and bandits with stochastic action availability and adversarial rewards. In Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics, volume 5 of Proceedings of Machine Learning Research, pages 272–279, 2009.

Edward L. Kaplan and Paul Meier. Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282): 457–481, 1958.

Predrag Klasnja, Eric B. Hekler, Saul Shifman, et al. Microrandomized trials: An experimental design for developing just-in-time adaptive interventions. Health Psychology, 34(S):1220–1228, 2015.

Predrag Klasnja, Shawna Smith, Nicholas J. Seewald, Andy Lee, Kelly Hall, Brook Luers, Eric B. Hekler, and Susan A. Murphy. Eficacy of contextually tailored suggestions for physical activity: A microrandomized optimization trial of HeartSteps. Annals of Behavioral Medicine, 53(6):573–582, 2019.

Robert Kleinberg, Alexandru Niculescu-Mizil, and Yogeshwer Sharma. Regret bounds for sleeping experts and bandits. Machine Learning, 80(2-3): 245–272, 2010.

Matthieu Komorowski, Leo Anthony Celi, Omar Badawi, Anthony C. Gordon, and A. Aldo Faisal. The artificial intelligence clinician learns optimal treatment strategies for sepsis in intensive care. Nature Medicine, 24:1716–1720, 2018.

Florian K¨unzler, Varun Mishra, Jan-Niklas Kramer, David Kotz, Elgar Fleisch, and Tobias Kowatsch. Exploring the state-of-receptivity for mHealth interventions. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3(4), 2019.

Arshika Lalan, Shresth Verma, Paula Rodriguez Diaz, Panayiotis Danassis, Amrita Mahale, Kumar Madhu Sudan, Aparna Hegde, Milind Tambe, and Aparna Taneja. Improving health information access in the world’s largest maternal mobile health program via bandit algorithms. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 2024.

Liliana Laranjo, Adam G. Dunn, Huong Ly Tong, Ahmet Baki Kocaballi, Jessica Chen, Rabia Bashir, Didi Surian, Blanca Gallego, Farah Magrabi, Annie Y. S. Lau, and Enrico Coiera. Conversational agents in healthcare: A systematic review. Journal of the American Medical Informatics Association, 25(9):1248–1258, 2018.

Tor Lattimore and Csaba Szepesv´ari. Bandit Algorithms. Cambridge University Press, 2020.

David S. Lee and Thomas Lemieux. Regression discontinuity designs in economics. Journal of Economic Literature, 48(2):281–355, 2010.

Siew Hwa Lee, Ulugbek B. Nurmatov, Bright I. Nwaru, Mome Mukherjee, Liz Grant, and Claudia Pagliari. Efectiveness of mHealth interventions for maternal, newborn and child health in low- and middle-income countries: Systematic review and meta-analysis. Journal of Global Health, 6(1):010401, 2016.

Amnesty E. LeFevre, P. Dane, C. J. Copley, et al. Unpacking the performance of a mobile health information messaging program for mothers (Mom-Connect) in South Africa: Evidence on program

reach and messaging exposure. BMJ Global Health, 3(Suppl 2):e000583, 2018.

Richard T. Lester, Paul Ritvo, Edward J. Mills, Antony Kariri, Sarah Karanja, Michael H. Chung, William Jack, James Habyarimana, Mohsen Sadatsafavi, Mehdi Najafzadeh, et al. Efects of a mobile phone short message service on antiretroviral treatment adherence in Kenya (WelTel Kenya1): A randomised trial. The Lancet, 376(9755):1838– 1845, 2010.

Lihong Li, Wei Chu, John Langford, and Robert E. Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th International Conference on World Wide Web, pages 661–670, 2010.

Lihong Li, Wei Chu, John Langford, and Xuanhui Wang. Unbiased ofline evaluation of contextualbandit-based news article recommendation algorithms. In Proceedings of the Fourth ACM International Conference on Web Search and Data Mining, 2011.

Peng Liao, Kristjan Greenewald, Predrag Klasnja, and Susan Murphy. Personalized HeartSteps: A reinforcement learning algorithm for optimizing physical activity. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 4(1):1–22, 2020.

S. Lund, M. Hemed, B. B. Nielsen, A. Said, K. Said, M. H. Makungu, and V. Rasch. Mobile phones as a health communication tool to improve skilled attendance at delivery in Zanzibar: A clusterrandomised controlled trial. BJOG: An International Journal of Obstetrics and Gynaecology, 119 (10):1256–1264, 2012.

Aishwarya Mandyam, Matthew J¨orke, William Denton, Barbara E. Engelhardt, and Emma Brunskill. Adaptive interventions with user-defined goals for health behavior change. In Proceedings of the Fifth Conference on Health, Inference, and Learning, volume 248 of Proceedings ofMachine Learning Research, pages 597–618, 2024.

Aishwarya Mandyam, Shengpu Tang, Jiayu Yao, Jenna Wiens, and Barbara E. Engelhardt. CAN-DOR: Counterfactual ANnotated DOubly robust of-policy evaluation. In Proceedings of the Conference on Health, Inference, and Learning, volume

333 of Proceedings of Machine Learning Research, pages 644–687, 2026.

K. Manji, Johanna Hanefeld, Jo Vearey, Helen Walls, and Thea de Gruchy. Using WhatsApp messenger for health systems research: A scoping review of available literature. Health Policy and Planning, 36:774–789, 2021.

Robert Mash, Darcelle Schouw, and Alex Emilio Fischer. Evaluating the implementation of the GREAT4Diabetes WhatsApp chatbot to educate people with type 2 diabetes during the COVID-19 pandemic: Convergent mixed methods study. JMIR Diabetes, 7(2):e37882, 2022.

Aditya Mate, Lovish Madaan, Aparna Taneja, Neha Madhiwalla, Shresth Verma, Gargi Singh, Aparna Hegde, Pradeep Varakantham, and Milind Tambe. Field study in deploying restless multi-armed bandits: Assisting non-profits in improving maternal and child health. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022.

Abhinav Mehrotra, Veljko Pejovic, Jo Vermeulen, Robert Hendley, and Mirco Musolesi. My phone and me: Understanding people’s receptivity to mobile notifications. In Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems, 2016.

Meta for Developers. Per-user marketing template message limits. Developer documentation, 2026a. https://developers.facebook.com/document ation/business-messaging/whatsapp/templa tes/marketing-templates/per-user-limits/, accessed 26 October 2026.

Meta for Developers. Pricing on the WhatsApp business platform. Developer documentation, 2026b. https://developers.facebook.com/document ation/business-messaging/whatsapp/pricing, accessed 26 October 2026.

Madison Milne-Ives, Caroline de Cock, Ernest Lim, Melissa Harper Shehadeh, Nick de Pennington, Guy Mole, Eduardo Normando, and Edward Mein ert. The efectiveness of artificial intelligence conversational agents in health care: Systematic review. Journal of Medical Internet Research, 22(10): e20346, 2020.

Varun Mishra, Florian K¨unzler, Jan-Niklas Kramer, Elgar Fleisch, Tobias Kowatsch, and David Kotz.

Detecting receptivity for mHealth interventions in the natural environment. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 5(2):74, 2021.

Anisha Musti. Usability and usefulness of SMSbased artificial intelligence intervention (Mwana) on breastfeeding outcomes in Lagos, Nigeria: Pilot app development study. JMIR Formative Research, 9:e65157, 2025.

Tom Nadarzynski, Oliver Miles, Aimee Cowie, and Damien Ridge. Acceptability of artificial intelligence (AI)-led chatbot services in healthcare: A mixed-methods study. Digital Health, 5, 2019.

Inbal Nahum-Shani, Shawna N. Smith, Bonnie J. Spring, et al. Just-in-time adaptive interventions (JITAIs) in mobile health: Key components and design principles for ongoing health behavior support. Annals of Behavioral Medicine, 52(6):446– 462, 2018.

Shahriar Noroozizadeh, Pim Welle, Jeremy Weiss, and George H. Chen. The impact of medication non-adherence on adverse outcomes: Evidence from schizophrenia patients via survival analysis. In Proceedings of the Conference on Health, Inference, and Learning, volume 287 of Proceedings of Machine Learning Research, pages 573–609, 2025.

Ominiflow. Nigeria WhatsApp API pricing 2026, 2026. https://ominiflow.com/whatsapp-api -pricing/nigeria.

Gbemisola E. Osanyin, Aduragbemi Banke-Thomas, Esther O. Oluwole, Adesumbo K. Odeseye, and Bosede B. Afolabi. Efects of a mHealth voice messaging intervention on antenatal care utilisation at primary care level in Lagos, Nigeria: A cluster randomised trial. Journal of Public Health in Africa, 13(3):2222, 2022.

Martin Pielot, Karen Church, and Rodrigo de Oliveira. An in-situ study of mobile phone notifications. In Proceedings of the 16th International Conference on Human-Computer Interaction with Mobile Devices and Services, 2014.

Cristian Pop-Eleches, Harsha Thirumurthy, James P. Habyarimana, Joshua Graf Zivin, Markus P. Goldstein, Damien de Walque, Leslie MacKeen, Jessica Haberer, Sylvester Kimaiyo, John E. Sidle,

Duncan Ngare, and David R. Bangsberg. Mobile phone technologies improve adherence to antiretroviral treatment in a resource-limited setting: A randomized controlled trial of text message reminders. AIDS, 25(6):825–834, 2011.

Doina Precup, Richard S. Sutton, and Satinder Singh. Eligibility traces for of-policy policy evaluation. In Proceedings of the 17th International Conference on Machine Learning, pages 759–766, 2000.

Skipper Seabold and Josef Perktold. statsmodels: Econometric and statistical modeling with Python. In Proceedings of the 9th Python in Science Conference, 2010.

Stephanie Felicie Victoria Sondaal et al. Assessing the efect of mHealth interventions in improving maternal and neonatal care in low- and middle-income countries: A systematic review. PLoS ONE, 11(5): e0154664, 2016.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2nd edition, 2018.

Techpoint Africa. WhatsApp will charge businesses per message from October 1, 2026, 2026. https: //techpoint.africa/guide/whatsapp-will-c harge-businesses-per-message/.

Ambuj Tewari and Susan A. Murphy. From ads to interventions: Contextual bandits in mobile health. In James M. Rehg, Susan A. Murphy, and Santosh Kumar, editors, Mobile Health: Sensors, Analytic Methods, and Applications. Springer, 2017.

Philip S. Thomas and Emma Brunskill. Data-eficient of-policy policy evaluation for reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, 2016.

Anna L. Trella, Kelly W. Zhang, Inbal Nahum-Shani, Vivek Shetty, Finale Doshi-Velez, and Susan A. Murphy. Designing reinforcement learn ing algorithms for digital interventions: Preimplementation guidelines. Algorithms, 15(8):255, 2022.

Rajet Vatsa, Wei Chang, Sharon Akinyi, Sarah Little, Catherine Gakii, John Mungai, Cynthia Kahum bura, Anneka Wickramanayake, Sathy Rajasekha

ran, Jessica Cohen, and Margaret McConnell. Empowering patients across the maternal-newborn care continuum: A cluster randomized controlled trial testing a digital health platform in Kenya, 2024. medRxiv 10.1101/2024.06.03.24308340.

Shresth Verma, Arpan Dasgupta, Neha Madhiwalla, Aparna Taneja, and Milind Tambe. Decisions and deployment: The five-year SAHELI project (2020– 2025) on restless multi-armed bandits for improving maternal and child health behaviors, 2026. arXiv:2604.07384.

Kai Wang, Shresth Verma, Aditya Mate, Sanket Shah, Aparna Taneja, Neha Madhiwalla, Aparna Hegde, and Milind Tambe. Scalable decisionfocused learning in restless multi-armed bandits with application to maternal and child health. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 12138–12146, 2023.

Christopher J. C. H. Watkins and Peter Dayan. Qlearning. Machine Learning, 8(3-4):279–292, 1992.

Edwin B. Wilson. Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158):209–212, 1927.

World Health Organization. WHO Guideline: Recommendations on Digital Interventions for Health System Strengthening. World Health Organization, Geneva, 2019.

World Health Organization. Trends in Maternal Mortality 2000 to 2020: Estimates by WHO, UNICEF, UNFPA, World Bank Group and UN-DESA/Population Division. World Health Organization, Geneva, 2023.

Elad Yom-Tov, Guy Feraru, Mark Kozdoba, Shie Mannor, Moshe Tennenholtz, and Irit Hochberg. Encouraging physical activity in patients with diabetes: Intervention using a reinforcement learning system. Journal of Medical Internet Research, 19 (10):e338, 2017.

## Outline of the Appendix

The appendix gives the proofs, the variable definitions, the details of each model and the robustness checks behind the main text. Table 4 lists what each part contains. Observed, modelled and simulated results are labelled as in the main text, and every table and figure caption states its data source.

## Table 4: Contents of the appendix, with the tables and figures in each part.

Contents   
A Proofs of Propositions 1 to $4 ;$ the hindsight   
bound; the scheduling logic (Figure 4)   
B Variables, observed and true window state, trun  
cation check, error codes, discontinuity and   
placebos (Table 5, Table 6, Figure 5, Figure 6,   
Figure 7)   
C Decision models and feature coding, sample   
reconciliation, transport check, calibration, re  
stricted samples, bandit replay (Table 7, Table 8,   
Table 9, Figure 8, Figure 9, Figure 10)   
D Simulator inputs and dynamics, Q-learning de  
tails, strategies and costs, suppressing the ofer,   
validation, stress tests (Table 10, Table 11, Fig  
ure 11, Figure 12, Figure 13)   
E Burden definitions, within-mother and pooled   
models (Table 12)   
F Send time within batched broadcasts, clustering   
(Figure 14)   
G Replication on the second export (Table 13)   
H Software, seeds and tests

## Appendix A. Propositions and Proofs

Throughout, $\mu _ { a } ( x , W )$ is the expected value of action a in context x and window state W, and $\mu ^ { * } ( x , W ) =$ $\operatorname* { m a x } _ { a } \mu _ { a } ( x , W )$ . With the window closed, $\mu _ { F } ( x , 0 ) =$ $0 , \mu _ { T } ( x , 0 ) = p _ { T } ( x ) - c , \mu _ { D } ( x , 0 ) = V _ { D } ( x )$ and sending nothing has value 0.

Proof of Proposition 1. A policy that always sends F has value $P ( W { = } 1 ) ~ P$ (delivered $F , W { = } 1 ) +$ $P ( W { = } 0 ) ~ \cdot ~ 0 ~ \leq ~ P ( W { = } 1 )$ , because P(delivered | $F , W { = } 0 ) = 0$ and in-window messages cost nothing. The window-aware policy earns the same when $W =$ 1. When $W = 0$ it earns max $\{ p _ { T } ( x ) - c , V _ { D } ( x ) , 0 \}$ given x, against 0 for the always-F policy. Taking expectations over x given $W = 0$ and weighting by

$P ( W { = } 0 )$ gives the stated diference, which is never negative. □

Proof of Proposition 2. The regret in round t is $r _ { t } = \mu ^ { * } ( x _ { t } , W _ { t } ) - \mu _ { a _ { t } } ( x _ { t } , W _ { t } ) \geq 0$ . Let $\mathcal { H } _ { t - 1 }$ be the history of earlier contexts, actions and rewards, and $\pi _ { t } ( a \mid x _ { t } , W _ { t } , \mathcal { H } _ { t - 1 } )$ the learner’s policy. Choosing F when $W _ { t } = 0$ contributes $\mu ^ { * } ( x _ { t } , 0 )$ , so

$$
\mathbb { E } [ r _ { t } ] \ge \mathbb { E } \big [ \mathbf { 1 } \{ W _ { t } = 0 \} \pi _ { t } ( F \mid x _ { t } , W _ { t } , \mathcal { H } _ { t - 1 } ) \mu ^ { * } ( x _ { t } , 0 ) \big ] .
$$

A learner that observes W and masks $F$ has $\pi _ { t } ( F \mid$ $x _ { t } , 0 , \mathcal { H } _ { t - 1 } ) = 0$ , so this term vanishes, and because $\mu _ { F } ( x , 0 ) = 0$ is known it never needs to try $F$ to learn it. Now take a learner whose policy $\pi _ { t } ( a \mid x _ { t } , \mathcal { H } _ { t - 1 } )$ does not depend on $W _ { t } .$ , with $( x _ { t } , W _ { t } )$ drawn independently across rounds from a fixed distribution, so that $W _ { t }$ is independent of $\mathcal { H } _ { t - 1 }$ given $x _ { t }$ . Conditional on $x _ { t }$ and $\mathcal { H } _ { t - 1 } ,$ it chooses a with a fixed probability in both window states, so its expected value is $\textstyle \sum _ { a } \pi _ { t } ( a \mid$ $x _ { t } , \mathcal { H } _ { t - 1 } ) \mathbb { E } [ \mu _ { a } ( x _ { t } , W ) \mid x _ { t } ] \leq \operatorname* { m a x } _ { a } \mathbb { E } [ \mu _ { a } ( x _ { t } , \overline { { W } } ) \mid x _ { t } ] ,$ Its expected regret in every round is therefore at least

$$
g = \mathbb { E } _ { x } \Big [ \mathbb { E } [ \mu ^ { * } ( x , W ) \mid x ] - \operatorname* { m a x } _ { a } \mathbb { E } [ \mu _ { a } ( x , W ) \mid x ] \Big ] \geq 0 ,
$$

and at least gn after n rounds, whatever it learns. Let S be the set of contexts with $0 < P ( W { = } 1 \mid x ) < 1$ in which no action is best in both window states. For $x \in S ,$ , every action falls short of $\mu ^ { * } ( x , w )$ in some state w of positive probability, so $\mathbb { E } [ \mu _ { a } ( x , W ) \mid x ] <$ $\mathbb { E } [ \mu ^ { * } ( x , W ) \mid x ]$ for each of the finitely many actions, and the bracket is positive. Hence $P ( S ) > 0$ implies $g > 0$ . If x included h, then $P ( W { = } 1 \mid x ) \in \{ 0 , 1 \}$ , S would be empty and $g = 0 ;$ this is why x excludes h. In the replay, the window-aware learners see $\log ( 1 + h )$ and the window indicator, while the window-blind learner sees neither (Appendix C). □

In our setting S is large: with the window open the best action is $F ,$ which is free and delivered with probability 1 by assumption, and with it closed F is worth nothing.

Proof of Proposition 3. A message sent when the mother writes at time $R \leq H$ is delivered R hours late and is worth $e ^ { - \delta R } ;$ ; if $R > H$ , nothing is delivered. Hence $V _ { D } = \mathbb { E } [ e ^ { - \delta R } \mathbf { 1 } \{ R \leq H \} ]$ . Because $e ^ { - \delta R } \leq 1$ , $V _ { D } \leq P ( R \leq H )$ for any distribution of R. For R exponential with rate λ,

$$
V _ { D } = \int _ { 0 } ^ { H } \lambda e ^ { - \lambda r } e ^ { - \delta r } d r = \frac \lambda { \lambda + \delta } \big ( 1 - e ^ { - ( \lambda + \delta ) H } \big ) .
$$

Dropping $e ^ { - \lambda r } \leq 1$ inside the integral gives $V _ { D } \ \leq$ $( \lambda / \bar { \delta } ) ( 1 - e ^ { - \delta H } ) \leq \lambda \operatorname* { m i n } ( H , 1 / \delta )$ , since $1 - e ^ { - y } \leq$ min(y, 1). Waiting has no template cost, so it beats a template if and only if $V _ { D } \ge p _ { T } ( x ) - c . ~ V _ { D }$ increases with H and λ (a larger rate makes R stochastically smaller, and the integrand is decreasing in R) and decreases with δ. □

![](images/afb15f533c6452b3b37c56b5448d55a2173dfc712668cfa20ebaf332ffb361f9.jpg)  
Figure 4: Window-aware scheduling logic implied by Propositions 1, 3 and 4, for one proactive message. D and $T ^ { + }$ are chained actions: under $D$ the free-form message is sent only if she writes within $H _ { ; }$ and under $T ^ { + }$ the follow-up is sent only if she replies to the scheduled template within 24 hours, so its value is $q ( x )$ times the value of the follow-up net of its cost $c _ { s }$ (Proposition 4). A proposed design: it was not deployed or tested as an intervention in this study.

Remark on estimation. The threshold rule uses $\hat { \lambda } = - \ln ( 1 - \hat { p } _ { R } ) / H$ , where $\hat { p } _ { R }$ is the predicted probability of a return within H, so the exponential model matches $P ( R \leq H )$ . If the return hazard falls over time, as in the mixture fitted in Appendix D, returns within H come earlier than this constant rate implies and the formula understates $V _ { D } .$ . The bound $V _ { D } ~ \le ~ P ( R ~ \le ~ H )$ holds for any hazard, so, with a calibrated $\hat { p } _ { R }$ , the template is preferred whenever $p _ { T } ( x ) - c \geq \hat { p } _ { R } ( x )$ , whatever the shape of the hazard. In the evaluation, the value of the wait rule itself is computed from observed returns, not from this formula.

Proof of Proposition 4. The scheduled template is sent under both $T$ and $T ^ { + }$ , so its value and cost cancel. Under $T ^ { + }$ , the follow-up is sent only after a reply, when the window is open and a free-form message is delivered; its value is $\mathbb { E } [ e ^ { - \delta L } \mid \mathrm { r e p l y } ] - c _ { s }$ with probability $q ( x )$ , and zero otherwise. Because $q ( x )$ is per scheduled send, it equals the template’s delivery probability times the reply rate among delivered templates; the simulator uses these two factors separately. Under $T ,$ the follow-up is a second template with value $p _ { T } ( x ) - c .$ Comparing the two gives the condition. Because $\mathbb { E } [ e ^ { - \delta L } \ | \ \mathrm { r e p l y } ] - c _ { s } \leq 1$ , the value of $T ^ { + }$ is at most $q ( x )$ , so $q ( x ) \geq p _ { T } ( x ) - c$ is necessary. □

Hindsight bound. For each decision, the hindsight policy takes the larger of the template’s expected value and waiting’s realised value. It bounds any policy that knows when each mother will return but not whether a template will be delivered.

## Appendix B. Data and Variables

Variables. $Y _ { i } ~ = ~ 1$ if the final status is “failed”; $D _ { i } ~ = ~ 1$ if “delivered” or “read”; “sent” and “unknown” are unconfirmed. Times are in West Africa Time. Identifiers are salted SHA-256 hashes of phone numbers; names and message text are excluded from analysis tables.

Window definition. The window is computed from every row containing a message from the mother, including 1,398 inbound rows with no bot reply. Computing it from bot rows only changed the state of 30 of 98,607 messages and no result.

Observed and true window state. A live scheduler knows the true state $W _ { i }$ from its own records. The export gives $\tilde { W _ { i } }$ . When an earlier message is observed, histories are contiguous, so $\tilde { W } _ { i } = W _ { i } { }$ . When none is observed, $\tilde { W } _ { i } = 0$ but the window may be open: in the truncation check below, 33 of 551 such messages (6.0%) had an open window, and onboarding and menu flows (the “other content” decisions) almost certainly did. Results that condition on an earlier message being observed (the discontinuity, Figure 1, and Table 8) are free of this error.

Truncation check. For the 131 shared mothers, 67 of 551 messages (12.2%) labelled “no earlier message observed” in the main export went to mothers who had written according to the longer replication history, and 33 had an open window. Results for mothers with an earlier message observed are unafected because histories are contiguous.

Late deliveries. The 7 free-form messages delivered between 24 and 30 hours most likely followed replies the export does not record, such as some button taps.

Error codes. Failed free-form messages are assigned 131047 (re-engagement) and failed templates 131026 (undeliverable). This is an inference from message format, not a recorded code; the operator’s own error summary was computed from the same export, so agreement with it is a consistency check, not verification. Because codes follow format, the resulting split of all 21,456 failures, 66.1% and 33.9%, restates the split of failures by format in Table 2; all medical-ofer failures are free-form and so carry 131047. The 66.1% coincides with, but is distinct from, the share of failures from the two flows in Section 6.2.

Discontinuity. Table 5 gives the jump at each bandwidth. The estimation window is [max(0, 24 − b), 24 + b] for bandwidth b, so for $b \geq 2 4$ the left side is always 0 to 24 hours and only the right side widens. Messages with no earlier message observed have no $h _ { i }$ and are excluded. Placebo cut-ofs at 12, 36, 48, 72 and 120 hours use the same local linear model with a 12-hour bandwidth below 48 hours and 24 hours otherwise (Table 6); an earlier version used mean differences, which can mix a slope with a jump: at 72 hours the mean diference was 5.3 points (CI 0.4 to

10.2) and the local linear estimate is −4.4 (CI −16.2 to 7.4). The diference in discontinuities interacts the jump with format within the 24-hour bandwidth (0 to 48 hours). Figure 7 shows which flows produce the failures.

Table 5: Robustness of the discontinuity to bandwidth. Local linear estimates of the jump in the share of free-form messages that failed at 24 hours after the mother’s last message, main export (observed). Percentage points with 95% intervals clustered by mother; the main text reports the 24-hour bandwidth. Upper limits above 100 arise because the linear probability model’s symmetric intervals are not bounded; no message failed just before 24 hours.
<table><tr><td>Bandwidth (h)</td><td>Jump</td><td>95% CI</td></tr><tr><td>6</td><td>89.8</td><td>75.9 to 103.7</td></tr><tr><td>12</td><td>89.1</td><td>80.2 to 98.0</td></tr><tr><td>18</td><td>91.5</td><td>84.2 to 98.9</td></tr><tr><td>24</td><td>93.0</td><td>87.0 to 99.1</td></tr><tr><td>48</td><td>95.1</td><td>91.3 to 99.0</td></tr><tr><td>72</td><td>93.9</td><td>90.8 to 96.9</td></tr></table>

Table 6: Placebo cut-ofs, main export (observed). Local linear jump in failure at each false cutof, percentage points, 95% intervals clustered by mother; bandwidth 12 hours below 48 hours, 24 hours otherwise. At 12 hours no free-form message failed on either side.
<table><tr><td colspan="3">Free-form</td><td colspan="2">Templates</td></tr><tr><td>Cut-off (h) Jump</td><td></td><td>95% CI Jump</td><td></td><td>95% CI</td></tr><tr><td>12</td><td>0.0</td><td>no failures</td><td>0.0</td><td>0.0 to 0.1</td></tr><tr><td>36</td><td>-4.3</td><td>-11.6 to 3.0</td><td></td><td>1.5 −3.1 to 6.2</td></tr><tr><td>48</td><td></td><td>-14.1 -28.1 to 0.0</td><td></td><td>0.0 −3.2 to 3.1</td></tr><tr><td>72</td><td></td><td>-4.4 −16.2 to 7.4</td><td></td><td>2.5 −1.9 to 7.0</td></tr><tr><td>120</td><td></td><td>4.1 −7.3 to 15.5</td><td></td><td>2.7 −1.9 to 7.2</td></tr></table>

## Appendix C. Decision Models and Ofline Evaluation

Models. Cross-fitting uses two folds with mothers ordered by first appearance (seed 42). The features are an indicator that no earlier message was observed, log(1 + earlier messages), which is zero when there are none, log(1 + h) with h capped at 60 days and set to the cap when no earlier message is observed (so the indicator acts as a separate intercept for that group), weekend and evening (17:00 to 23:59). The template model is trained on the 45,538 bot-initiated templates sent with the window closed or no earlier message observed. Return within 7 days is modelled by logistic regression on the same features, trained on the closed-window decisions, and converted to a rate $\hat { \lambda } = - \ln ( 1 - \hat { p } _ { R } ) / H$ . Cross-fitted AUCs are 0.78 (template delivery) and 0.69 (return). Table 7 shows calibration. Figure 8 shows time to the next conversation; mothers with more earlier conversations returned faster (hazard ratio 1.25 per log unit, 95% CI 1.18 to 1.31; Cox model with a mother-cluster bootstrap).

![](images/afab53c64069e7637ce3f30adc513560b571a4abd6ae6a18853ef338a092a009.jpg)

(b) Failure by window state  
![](images/b12bd3f4ba0fd49bb0f483a25f72287a50558a92d43c9ca976e93044c68e13f1.jpg)  
Figure 5: The medical-professional ofer, main export (observed). (a) Histogram of hours since the mother’s last message (log scale) for the 2,286 ofers with an earlier message observed. (b) Failure rate by window state, with Wilson 95% intervals and the number of ofers. None obs.: no earlier message from the mother in the export, an unobserved history rather than proof that the window was closed.

![](images/797076c86dc7fcf1718cdd167b2f35b4bbe23975a7c79b151a09f923c696844f.jpg)  
Figure 6: Robustness of the jump in failure at 24 hours, main export (observed). (a) Local linear estimates by bandwidth. (b) The true cut-of (24\*) and placebo cut-ofs with the same model, bandwidth 12 hours below 48 hours and 24 hours otherwise. Percentage points with 95% intervals clustered by mother. At 12 hours no free-form message failed on either side, so that estimate has no interval.

![](images/f6fec0767388bcd12d6f995ece3ccf3433c07e4d5d2cac402d9fbf29c496341b.jpg)  
Figure 7: Failed messages by flow, main export (observed): the ten flows with the most failures, labelled with failed and sent messages and the percentage that failed. The error follow-up and the medicalprofessional ofer produce 66.1% of the 21,456 failures.

The medical ofer: samples. The flow contains 4,936 bot messages: the 4,869 interactive ofers anal ysed in Section 6.2, 63 text messages and 4 templates, none of which failed. The 3,933 closed-window decisions are the bot-initiated free-form messages in the flow with 7 days of follow-up: 3,930 interactive offers, of which 9 did not fail, and 3 text messages, all delivered; the logged value of 0.3 per 100 counts all 12. Interactive ofers alone give 0.2 (CI 0.1 to 0.4), and the other values change by at most 0.1 (Table 8). The simulator’s validation targets (Figure 11) use the 1,963 ofers sent directly after the scheduled weekly guide, the chain it models, which is why its failure rates (99.4% overall, 96.9% with an earlier message observed) difer from those of all ofers.

Reply button and transport check. No reply button was ever attached to the guide before the ofer, so the reply-button value is the observed share of offers followed by any message from the mother within 24 hours. Templates in other flows that did carry buttons were answered more often (17.9% within 24 hours, excluding opt-out taps), which is the rate the simulator uses; the ofline value of 11.6 is therefore conservative. The transport check replaces the model with the observed delivery of the scheduled weekly guide template, 90.5% (87.6% confirmed) when an earlier message was observed and 86.9% (78.0% confirmed) when not, weighted by the 36.1% and 63.9% of ofer decisions in each group. Refitting inside the bootstrap widens the pooled template interval (85.5 to 86.5 without refitting; 84.9 to 87.1 with it). Table 8 repeats the evaluation on restricted samples. Figure 2 compares the options for the medical ofer, Figure 9 shows when waiting would beat a template and how this depends on template cost and half-life (grid $c \in [ 0 , 0 . 9 5 ] , \ \ell \ \in \ \{ 6 , 2 4 , 4 8 , 1 6 8 \}$ hours), and Figure 10 shows the bandit replay.

Table 7: Calibration of the cross-fitted template delivery model, main export. Closed-window templates grouped into deciles of predicted delivery (about 4,554 per decile); predicted and observed “not failed” delivery, and observed confirmed delivery, in percent.
<table><tr><td>Decile</td><td>Predicted</td><td>Observed</td><td>Confirmed</td></tr><tr><td>1</td><td>59</td><td>62</td><td>52</td></tr><tr><td>2</td><td>68</td><td>60</td><td>52</td></tr><tr><td>3</td><td>75</td><td>76</td><td>69</td></tr><tr><td>4</td><td>79</td><td>83</td><td>76</td></tr><tr><td>5</td><td>81</td><td>78</td><td>70</td></tr><tr><td>6</td><td>89</td><td>93</td><td>86</td></tr><tr><td>7</td><td>95</td><td>95</td><td>89</td></tr><tr><td>8</td><td>97</td><td>96</td><td>91</td></tr><tr><td>9</td><td>99</td><td>99</td><td>96</td></tr><tr><td>10</td><td>99</td><td>99</td><td>98</td></tr></table>

Bandit replay. Decision points are replayed in time order. Each learner chooses among F, T and $D .$ The reward of $F$ is 1 with the window open and 0 with it closed; of T, a Bernoulli draw with probability ${ \hat { p } } _ { T } ( x )$ minus $c = 0 . 0 5$ , so choosing $T$ with the window open costs at least 0.05; of $D ,$ the observed discounted value of waiting. Regret uses the expected rewards. Each action has a linear model with prior mean zero, prior precision equal to the identity and noise variance 0.25. The window-aware learners see an intercept, the no-earlier-message indicator, log(1 + earlier messages), log(1 + h), weekend, evening and the window indicator; the window-blind learner sees the same without log(1 + h) and the window indicator. Results are means over 30 runs (seeds 0 to 29). Table 9 repeats the replay without the 609 closed-window “other content” decisions, whose window was probably open, and on decisions with an earlier message observed. Both restrictions lower the regret of the window-aware learners, and the ordering is unchanged: masked, then unmasked, then blind, with the logged behaviour far behind.

## Appendix D. Simulator

Inputs. Gap mixture: quick component share 0.46, means 24.9 and 340 days. Template delivery for the guide: 90.5% (earlier message observed) and 86.9% (not); confirmed-only 87.6% and 78.0%. Genuine 24- hour reply rate to delivered button templates 17.9% (11.3% of raw replies were opt-out taps) and 3.4% to delivered plain guides; the reply lag is log-normal with log-hours mean 0.59 and standard deviation 1.99, truncated at 24 hours. Opt-out rate 785 strict opt-outs per 46,394 delivered bot-initiated messages. Evaluation: 8 seeds of 3,000 mothers (seeds 1000 to 1007); stress tests 4 seeds of 2,000, with Q-learning retrained on 100,000 mothers.

Dynamics. A reply is a conversation: it opens the window for 24 hours and restarts the mother’s renewal process, so her next spontaneous message follows a fresh gap. With weekly decisions, a reply in one week rarely leaves the window open at the next. In the persistence stress test, each mother draws her own probability of a quick gap from a Beta distribution with mean 0.46 and concentration $\kappa = 1$ , instead of all mothers sharing 0.46.

Q-learning. The state is the band of hours since her last message (none, 0 to 24, 24 to 72, 72 to 168, over 168) and the number of conversations in the last 28 days (0, 1, 2 or more), 15 states in all. The actions are $F , T ,$ , wait, $T ^ { + }$ and hybrid, with $F$ allowed only when the window is open; there is no option to withhold the ofer in the main runs. The Q-table starts at zero. The weekly reward is the guide delivered plus the ofer’s discounted value, minus 0.05 for every paid message, template or service; the stress tests labelled “cost per paid message” change this 0.05. The main run trains on 150,000 simulated mothers and each stress test on 100,000, enough because the greedy policy stops changing after about 10,000 (Fig ure 13b). The reply-button strategy is left out of the stress tests to save computation; in the main setting it delivers fewer ofers and less reward than the hybrid, and Table 11 includes it. Value in Table 10 is discounted ofers per 100 due; reward is the undiscounted sum over 26 weeks of the weekly reward, so the template rule’s 44.6 is about 23.5 guides plus 23.5 ofers less the message costs.

Suppressing the ofer. Table 11 adds a “none” action and raises the per-message exit probability of the global model in steps. Q-learning (60,000 training mothers; 4 seeds of 2,000) matches the template rule within simulation noise up to four times the observed bound; at eight and sixteen times it suppresses more ofers, delivering 71.9 and 46.5 per 100, and gains at most 0.2 reward units.

Table 8: Closed-window evaluation on restricted samples, main export, per 100 decisions with 95% intervals from 500 mother-cluster bootstrap replicates (delivery models not refitted). Logged values are observed; wait values replay the logs (delivered within 7 days by waiting, assuming waiting does not change return); reply values are a proxy (any message from the mother within 24 hours); template values are modelled (lower: confirmed delivery; upper: not failed). With an earlier message observed, the window state is exact.
<table><tr><td>Sample</td><td>n</td><td></td><td>Logged Template lower Template upper</td><td></td><td>Wait</td><td>Reply</td></tr><tr><td rowspan="2">All closed or none obs. (paper) 12,751 95% CI</td><td rowspan="2"></td><td>5.8</td><td>80.4</td><td>86.0</td><td>20.4</td><td>15.2</td></tr><tr><td>5.2 to 6.5</td><td>79.8 to 81.1</td><td>85.5 to 86.5 18.9 to 22.0 13.9 to 16.5</td><td></td><td></td></tr><tr><td rowspan="2">Earlier message observed 95% CI</td><td rowspan="2">4,225</td><td>0.1</td><td>97.3</td><td>99.0</td><td>28.8</td><td>20.6</td></tr><tr><td>0.0 to 0.1</td><td>97.2 to 97.4</td><td>98.9 to 99.0 26.4 to 31.4 18.4 to 22.7</td><td></td><td></td></tr><tr><td rowspan="2">Offer and error follow-up 95% CI</td><td>12,142</td><td>1.1</td><td>81.1</td><td>86.5</td><td>17.0</td><td>11.6</td></tr><tr><td></td><td>0.8 to 1.4</td><td>80.4 to 81.8</td><td>86.0 to 87.1 15.5 to 18.5 10.4 to 12.9</td><td></td><td></td></tr><tr><td rowspan="2">Offer, interactive only 95% CI</td><td>3,930</td><td>0.2</td><td>81.5</td><td>86.8</td><td>16.5</td><td>11.5</td></tr><tr><td></td><td>0.1 to 0.4</td><td>80.8 to 82.2</td><td>86.3 to 87.4 15.3 to 18.1 10.4 to 12.8</td><td></td><td></td></tr><tr><td rowspan="2">Offer, interactive, earlier msg. 95% CI</td><td>1,421</td><td>0.0</td><td>97.3</td><td>99.0</td><td>28.7</td><td>21.3</td></tr><tr><td></td><td>0.0 to 0.0</td><td>97.2 to 97.4</td><td>98.9 to 99.0 26.2 to 31.6 19.0 to 23.9</td><td></td><td></td></tr></table>

Table 9: Regret per 100 decisions in the bandit replay on restricted samples, main export (modelled rewards; means over 30 runs, seeds 0 to 29, as in the main text).
<table><tr><td>Sample</td><td>n</td><td>Masked</td><td>Unmasked</td><td>Blind</td><td>Logged</td></tr><tr><td>All decisions (main text)</td><td>19,827</td><td>1.4</td><td>1.5</td><td>2.3</td><td>49.7</td></tr><tr><td>Without closed other content</td><td>19,218</td><td>1.0</td><td>1.0</td><td>1.8</td><td>51.4</td></tr><tr><td>Earlier message observed</td><td>11,301</td><td>0.8</td><td>0.9</td><td>2.3</td><td>35.3</td></tr></table>

Results. Figure 11 compares the simulator with the logs, Table 10 and Figure 12 give each strategy’s value and cost, and Figure 13 gives the stress tests and the Q-learning training curve.

## Appendix E. Burden Models

Definitions. The unit is a delivered (not failed) bot-initiated message: 49,256 in the main export, of which 46,394 were confirmed delivered and 2,862 had status “sent” or “unknown”; the simulator’s opt-out rate uses the confirmed count from the raw export as its denominator. Exposure is log(1 + m), where m is the number of such messages, or of templates, in the previous 7 days (24 hours in one specification), excluding the same chain (10 minutes). An increase of ln 2 = 0.69 log units doubles 1 + m, so “per doubling” means doubling one plus the count. Outcomes are an opt-out within 24 hours (a message starting “opt out” or exactly “stop” or “unsubscribe”) and reading within 24 hours. Opt-out models use only flows in which an opt-out occurred. “Recency” is a set of indicators for hours since the mother’s last mes sage (24 or fewer, 1 to 7 days, 7 to 30 days, over 30 days or none observed).

Models. Within-mother models subtract each mother’s mean (mothers with at least 3 messages for opting out, 5 for reading) and add flow and month efects. Because opting out ends some streams, a mother who opts out early has lower later volume, which biases within-mother coeficients downward. The pooled discrete-time model avoids this: it keeps each mother’s messages until her first optout, models whether that opt-out follows within 24 hours by logistic regression on template volume, recency, log(1 + days since her first message), flow and month, and does not demean. Errors are clustered by mother throughout. Table 12 gives every specification.

![](images/e42e989de306f87fb8aa23d9c02289dc624bc1442a9b43eb16d7e4047047414e.jpg)  
Figure 8: Kaplan-Meier estimates of the proportion of conversations not yet followed by another, by conversation number, main export (observed), with pointwise 95% bands; censored at the end of the export. n counts conversations, which are runs of the mother’s own messages split at gaps over 12 hours.

Table 10: Simulated strategies over 26 weeks, main setting (stream-specific opt-out), means over 8 seeds of 3,000 mothers. Ofers delivered and discounted value per 100 ofers due; reward and cost per mother. Cost counts templates and in-window service messages at US\$0.0101 each, as under pricing from October 2026.
<table><tr><td>Strategy</td><td>Offers</td><td>Value</td><td>Reward</td><td>Cost</td></tr><tr><td>Logged</td><td>1.3</td><td>1.3</td><td>22.5</td><td>0.24</td></tr><tr><td>Wait rule</td><td>10.2</td><td>3.8</td><td>23.1</td><td>0.26</td></tr><tr><td>Reply button (T+)</td><td>18.5</td><td>13.8</td><td>25.6</td><td>0.29</td></tr><tr><td>Hybrid</td><td>91.7</td><td>51.2</td><td>34.4</td><td>0.48</td></tr><tr><td>Template rule</td><td>90.3</td><td>90.3</td><td>44.6</td><td>0.47</td></tr><tr><td>Threshold</td><td>90.3</td><td>90.3</td><td>44.6</td><td>0.47</td></tr><tr><td>Q-learning</td><td>90.3</td><td>90.3</td><td>44.6</td><td>0.47</td></tr></table>

Table 11: Simulated reward per mother over 26 weeks when every delivered message ends contact with probability ρ (global opt-out model) and Q-learning may suppress the ofer. $\rho ~ = ~ 1 . 7 \%$ is the observed upper bound. Last column: ofers Q-learning delivers per 100 due. Means over 4 seeds of 2,000 mothers.
<table><tr><td>ρ</td><td>Template</td><td>Suppress</td><td> $T ^ { + }$  Q-learning</td><td>Offers</td></tr><tr><td>1.7%</td><td>30.8</td><td>18.7 20.6</td><td></td><td>30.8 89.0</td></tr><tr><td>3.4%</td><td>22.4</td><td>15.7 16.7</td><td></td><td>22.3 87.2</td></tr><tr><td>6.8%</td><td>13.4</td><td>11.3 11.5</td><td></td><td>13.5 83.6</td></tr><tr><td>13.5%</td><td>7.0</td><td>6.8</td><td>6.6</td><td>7.1 71.9</td></tr><tr><td>27.1%</td><td>3.5</td><td>3.5</td><td>3.4</td><td>3.5 46.5</td></tr></table>

## Appendix F. Send Time

The sample is 17 batched broadcasts (7,794 delivered messages, 2,133 mothers). Windows are morning, afternoon, evening and night (22:00 to 23:59), compared with campaign fixed efects. The night group is small: 266 messages, 193 (73%) from one campaign, so 73 remain when that campaign is dropped. Intervals are clustered by mother; clustering by the 17

(a) Value of waiting minus template  
![](images/76d8f7ae594bed36c49fe14eebed614d9bd55efe593eb420120371fadc8eba4b.jpg)  
(c) Threshold rule: share of decisions that wait

(b) Return rates of closed-window decisions  
![](images/4c7e09b615ed139726123436a4f44396d8941b5d3355bd459a8b5e37595ff390.jpg)  
(d) Threshold rule: gain over template rule

![](images/35371be8329739e92b8ad077964750b9864b52f1b424d0d087f7c1c56448ad1e.jpg)

![](images/3bb28e2ed8445c5280102700ac97ad15b926e0dcf8f3a614b668357cfe093bcc.jpg)  
Figure 9: When waiting beats a template (modelled, main export). (a) Value of waiting up to 7 days minus the value of a template, $V _ { D } - p _ { T } \ ( p _ { T } = 0 . 8 1 , c = 0 )$ , from Proposition ${ 3 ; }$ diverging scale centred at zero with symmetric limits; black curve: break-even line; dashed lines: urgent (6 h) and routine (48 h) half-lives. (b) Estimated return rates of the closed-window decisions. (c) Share of closedwindow decisions for which the threshold rule waits and (d) its gain over the template rule, by template cost c and half-life; waiting becomes common only once c reaches 0.75 to 0.85.

(a) Cumulative regret  
![](images/25eea3dbd3020c33adebd287d5acb1cede743e03b0d5588515bd0444d296b107.jpg)

(b) Final regret (log scale)  
![](images/94a42d59199af5af762272c73a0ac7a14c4e57e33d64e5befb0ac7f46176be9f.jpg)  
Figure 10: Linear Thompson-sampling learners replayed on the 19,827 logged decisions in time order, main export (modelled rewards). (a) Cumulative regret against the window-aware oracle, mean and 95% band over 30 runs (seeds 0 to 29). (b) Regret per 100 decisions, log scale, with the logged behaviour.

(a) Return to the chat  
![](images/50daa1ebc45f69c13c9f137a6e40ae3e54d9116e8bcb9f304a5e9ea87f6a75d5.jpg)

(b) Logs and simulation  
![](images/cc2539ddc7ecc29eab9fecad5ff94a5142773e14da2a853237fcce152e6cda1c.jpg)  
Figure 11: Simulator validation under the logged behaviour. (a) Proportion of mothers who have written again, by days since a conversation: simulated, the fitted two-component mixture, and Kaplan-Meier estimates from the logs (the points the mixture was fitted to). (b) Statistics in the logs and in the simulation, grouped by role: a calibrated input, targets of the gap-mixture fit, and checks not used in fitting. The simulation overstates returns at 30 and 90 days (43.2% against 37.8%, 69.3% against 58.5%). Agreement is conditional on the calibrated model and does not validate its behavioural assumptions.

![](images/056c8bfb0f599b3c9d34234458848952678138a3f39ec15204c4458b9b2485bd.jpg)  
Figure 12: Simulated value against cost per mother over 26 weeks, main setting. Value is discounted ofers delivered per 100 due; cost counts templates and in-window service messages at US\$0.0101 each, as under pricing from October 2026. Means with 95% intervals over 8 seeds, narrower than the markers. The template rule, the threshold rule and Q-learning fall on the same point.

Table 12: Burden models, main export (observed associations), with 95% intervals clustered by mother. Top: within-mother linear models, percentage points per log unit of exposure. Bottom: pooled logistic models, odds ratio per log unit. <sup>a</sup>Template volume, recency and tenure.

<table><tr><td>Specification Estimate (95% CI)</td></tr><tr><td>Opt-out within 24 hours, within mother All messages, 7 days -2.6 (−3.4 to −1.9)</td></tr><tr><td>All messages, 24 hours -2.8 (−3.7 to −1.9)</td></tr><tr><td>Before first opt-out only -1.4 (−2.1 to −0.7)</td></tr><tr><td>Templates, with recency −2.3 (−3.5 to −1.2) Read within 24 hours, within mother</td></tr><tr><td>All messages, 7 days -2.5 (−3.4 to −1.5)</td></tr><tr><td>Templates, with recency −3.2 (−4.4 to −2.0) Opt-out within 24 hours, pooled (odds ratio)</td></tr></table>

campaigns instead gives a narrower interval for the night diference (−26.2 to −19.8), which we do not rely on with so few clusters. Read within 6 hours, night minus morning: −23.0 points (95% CI −30.9 to −15.2); the intervals for reading within 1 hour and reading eventually include zero. Afternoon and evening do not difer from morning on any outcome. Dropping each campaign in turn gives −19.5 to −23.8 points. Prior reading and recent contact are balanced across send windows (joint p = 0.86 and 0.36). Figure 14 shows all estimates.

## Appendix G. Replication

The replication export was analysed with unchanged code. Table 13 sets its results beside the main export. The jump at 24 hours, the share of failures from the same two flows and the ranking of the options for the medical ofer all replicate. Template values and waiting are lower in the replication export because more of its closed-window decisions have no earlier message observed (2,072 of 2,439, 85%, against 8,526 of 12,751, 67%), and that group includes numbers no longer on WhatsApp. With an earlier message observed, the template reaches 95.9 to 98.5 per 100 ofers in the replication export (123 decisions) and 97.3 to 99.0 in the main export. The threshold rule waits on 0.5% of decisions instead of none.

(a) Reward minus template rule (per mother)  
![](images/1c835f8b0baed00988f9fbfd4bbaf4c2be6565cc7bbe666ac55b72c32ff0c62d.jpg)

(b) Q-learning training  
![](images/28c229c955ff2763cad0580466f84d4c44efc886e34758c36639118521bd8680.jpg)  
Figure 13: Stress tests (simulated). (a) Reward per mother of each strategy minus the template rule, in the main setting and 13 single-input stress tests; 4 seeds of 2,000 mothers, Q-learning retrained for each test. One-sided colour scale from zero (white) to the largest loss; values rounded to 0.1, so gaps below 0.05 show as 0.0 (the largest standard error of a reward is 0.19). (b) Reward of Q-learning’s greedy policy by number of training mothers, with the template rule (line and 95% band) and the logged behaviour.

![](images/b8d158f0117f9e1b0fb1eaf1a98b769355bbf89ab1c5069eab770dbfd531c742.jpg)  
Figure 14: Reading by send time within 17 batched broadcasts (7,794 delivered messages), main export. Difference from morning sends, in percentage points, in the share read eventually, within 6 hours and within 1 hour, with campaign fixed efects and 95% intervals clustered by mother. An observed association, not a randomised efect: the order of sending within a batch is undocumented.

## Appendix H. Software, Seeds andTests

Software: Python 3.13, pandas 3.0, NumPy 2.5, SciPy 1.18, statsmodels 0.15, scikit-learn 1.9, matplotlib 3.11. Seeds: 42 (cross-fitting folds, policy bootstrap, Cox bootstrap), 7 (medical-ofer bootstrap), 11 (content-type refit bootstrap and Qlearning training), 0 to 29 (Thompson-sampling runs), 1000 to 1007 (simulation evaluation); for the review checks, 5 (restricted-sample bootstrap), 0 to 9 (restricted replay) and 1000 to 1003 (suppress experiment). One command runs each pipeline end to end; the review checks in Appendices B to G are separate scripts that read its outputs. Unit tests cover the window, delivery and decision functions, the simulator and the result tables, and an automated check compares every number quoted in this paper with the result tables.

Table 13: Main and replication exports side by side. Observed results except the template values (modelled; lower bound confirmed delivery, upper bound not failed) and the simulation gap (simulated). Medical-ofer values are per 100 closed-window ofer decisions $( n = 3 , 9 3 3$ and 752) with 95% intervals from 500 mother-cluster bootstrap replicates that do not refit the models. The upper limit of the replication jump exceeds 100 because the linear probability model’s intervals are not bounded.
<table><tr><td></td><td>Main</td><td>Replication</td></tr><tr><td>Free-form failed, open</td><td>0 of 35,500</td><td>0 of 9,546</td></tr><tr><td>Free-form failed, closed</td><td>98.6%</td><td>97.6%</td></tr><tr><td>closed messages</td><td>6,267</td><td>544</td></tr><tr><td>18 to 30 h, before 24 h</td><td>0 of 114</td><td>0 of 19</td></tr><tr><td>18 to 30 h, after 24 h</td><td>72 of 79</td><td>18 of 18</td></tr><tr><td>Jump at 24 h (points)</td><td>93.0</td><td>100.0</td></tr><tr><td>95%CI</td><td>87.0 to 99.1</td><td>94.5 to 105.6</td></tr><tr><td>Two flows, share of failures</td><td>66.1%</td><td>52.0%</td></tr><tr><td>Offer, logged</td><td>0.3 (0.1 to 0.5)</td><td>0.4 (0.0 to 0.9)</td></tr><tr><td>Offer, template, lower</td><td>81.5 (80.8 to 82.2)</td><td>69.1 (67.5 to 70.9)</td></tr><tr><td>Offer, template, upper</td><td>86.8 (86.3 to 87.4)</td><td>77.6 (76.4 to 79.0)</td></tr><tr><td>Offer, wait 7 days</td><td>16.6 (15.2 to 18.1)</td><td>7.0 (4.9 to 9.7)</td></tr><tr><td>Offer, reply in 24 h</td><td>11.6 (10.5 to 13.0)</td><td>5.3 (3.3 to 7.7)</td></tr><tr><td>Threshold rule waits</td><td>0.0%</td><td>0.5%</td></tr><tr><td>Largest Q-learning gap</td><td>0.05</td><td>0.16</td></tr></table>