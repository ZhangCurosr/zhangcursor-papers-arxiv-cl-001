# RoPE at the End of Its Rope? Theory, Diagnosis,and Mitigation of Long-Context Failures

Yuyang Wu<sup>1</sup>, Yufeng Du<sup>2</sup>, Hao Peng<sup>2</sup>

<sup>1</sup>Inde<sub>p</sub>endent Researcher

<sup>2</sup>Universit<sub>y</sub> of Illinois at Urbana-Cham<sub>p</sub>ai<sub>g</sub>n<sub>,</sub> USA

## Abstract.

L<sub>ong-con</sub>t<sub>ex</sub>t f<sub>a</sub>il<sub>ures o</sub>f R<sub>o</sub>PE<sub>-</sub>b<sub>ase</sub>d l<sub>anguage mo</sub>d<sub>e</sub>l<sub>s can ar</sub>i<sub>se</sub> f<sub>rom</sub> R<sub>o</sub>PE’<sub>s</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>c</sub> t<sub>ra</sub>d<sub>eo</sub>f b<sub>e</sub>t<sub>ween ma</sub>i<sub>n-</sub> t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> <sub>s</sub>t<sub>a</sub>bl<sub>e</sub> t<sub>o</sub>k<sub>en</sub> <sub>pre</sub>f<sub>erences</sub> <sub>an</sub>d di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>hi<sub>ng</sub> <sub>near</sub>b<sub>y</sub> <sub>pos</sub>iti<sub>ons.</sub> D<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>ng</sub> <sub>w</sub>hi<sub>c</sub>h <sub>wea</sub>k<sub>ness</sub> t<sub>o</sub> <sub>a</sub>dd<sub>ress,</sub> <sub>an</sub>d h<sub>ow, requ</sub>i<sub>res a more prec</sub>i<sub>se c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on o</sub>f R<sub>o</sub>PE’<sub>s</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> i<sub>n</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s across con</sub>t<sub>ex</sub>t l<sub>eng</sub>th<sub>s.</sub> W<sub>e a</sub>dd<sub>ress a</sub> k<sub>ey</sub> li<sub>m</sub>it<sub>a</sub>ti<sub>on o</sub>f <sub>pr</sub>i<sub>or</sub> th<sub>eory</sub> b<sub>y a</sub>ll<sub>ow</sub>i<sub>ng unequa</sub>l <sub>query–</sub>k<sub>ey sca</sub>l<sub>es across</sub> R<sub>o</sub>PE f<sub>requenc</sub>i<sub>es,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>a</sub>li<sub>gns we</sub>ll <sub>w</sub>ith <sub>prac</sub>ti<sub>ca</sub>l <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>o</sub>b<sub>serva</sub>ti<sub>ons.</sub> O<sub>ur</sub> th<sub>eory ma</sub>k<sub>es</sub> b<sub>o</sub>th <sub>vu</sub>l<sub>nera</sub>biliti<sub>es measura</sub>bl<sub>e</sub> f<sub>or</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l h<sub>ea</sub>d<sub>s an</sub>d i<sub>npu</sub>t<sub>s, an</sub>d <sub>quan</sub>tifi<sub>es</sub> h<sub>ow</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency componen</sub>t<sub>s suppor</sub>t <sub>pos</sub>iti<sub>ona</sub>l <sub>sens</sub>iti<sub>v</sub>it<sub>y</sub> while potentially disrupting semantic stability. We also derive a theoretical context-length bound beyond which, under specified conditions, a fixed attention-score comparison cannot jointly avoid semantic reversal and positional insensitivity. Guided by our fresh theoretical insights, we introduce RoPE Profiler, a lightweight, plug-and-play diagnostic toolkit that augments existing evaluations with zero additional forward passes by <sub>reus</sub>i<sub>n cac</sub>h<sub>e</sub>d <sub>uer an</sub>d k<sub>e ac</sub>ti<sub>va</sub>ti<sub>ons.</sub> R<sub>eus</sub>i<sub>n ac</sub>ti<sub>va</sub>ti<sub>ons co</sub>ll<sub>ec</sub>t<sub>e</sub>d d<sub>ur</sub>i<sub>n eva</sub>l<sub>ua</sub>ti<sub>on,</sub> th<sub>e</sub> t<sub>oo</sub>lkit i<sub>ncurs</sub> littl<sub>e over</sub>h<sub>ea</sub>d<sub>.</sub> It <sub>supp</sub>l<sub>emen</sub>t<sub>s s</sub>t<sub>an</sub>d<sub>ar</sub>d b<sub>enc</sub>h<sub>mar</sub>k <sub>scores w</sub>ith t<sub>wo</sub> di<sub>agnos</sub>ti<sub>c scores</sub> th<sub>a</sub>t <sub>revea</sub>l <sub>seman</sub>ti<sub>c</sub> <sub>an</sub>d <sub>pos</sub>iti<sub>ona</sub>l <sub>wea</sub>k<sub>nesses an</sub>d h<sub>e</sub>l<sub>p users pr</sub>i<sub>or</sub>iti<sub>ze w</sub>hi<sub>c</sub>h <sub>aspec</sub>t t<sub>o a</sub>dd<sub>ress.</sub> C<sub>ruc</sub>i<sub>a</sub>ll<sub>y, our eva</sub>l<sub>ua</sub>ti<sub>ons across</sub> 49 l<sub>ong-con</sub>t<sub>ex</sub>t t<sub>as</sub>k <sub>se</sub>tti<sub>ngs revea</sub>l <sub>a</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>pa</sub>tt<sub>ern w</sub>h<sub>ere reason</sub>i<sub>ng</sub> t<sub>as</sub>k<sub>s pre</sub>d<sub>om</sub>i<sub>nan</sub>tl<sub>y su</sub>f<sub>er</sub> f<sub>rom se-</sub> <sub>man</sub>ti<sub>c reversa</sub>l<sub>, w</sub>h<sub>ereas re</sub>t<sub>r</sub>i<sub>eva</sub>l t<sub>as</sub>k<sub>s are pr</sub>i<sub>mar</sub>il<sub>y vu</sub>l<sub>nera</sub>bl<sub>e</sub> t<sub>o pos</sub>iti<sub>ona</sub>l i<sub>nsens</sub>iti<sub>v</sub>it<sub>y.</sub> G<sub>u</sub>id<sub>e</sub>d b<sub>y our</sub> th<sub>eory an</sub>d di<sub>agnos</sub>ti<sub>c pro</sub>fil<sub>es,</sub> t<sub>arge</sub>t<sub>e</sub>d hi<sub>g</sub>h<sub>-</sub>f<sub>requency resca</sub>li<sub>ng ac</sub>hi<sub>eves</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e ga</sub>i<sub>ns w</sub>ith<sub>ou</sub>t <sub>a</sub>dditi<sub>ona</sub>l training, improving task accuracy by up to 20 percentage points on Qwen3-8B and 25 percentage points on Llama-3.1-8B-Instruct.

Code: https://github.com/acetocarmine11/rope-profiler

Correspondence: wuyuyang@alumni.pku.edu.cn, haopeng@illinois.edu

## 1. Introduction

Lon<sub>g</sub>-context lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models (LLMs) o<sub>p</sub>en new <sub>p</sub>ossibilities for tacklin<sub>g</sub> com<sub>p</sub>lex, lon<sub>g</sub>-horizon <sub>p</sub>roblems, such as a<sub>g</sub>entic software develo<sub>p</sub>ment and scientific discover<sub>y</sub> (Youn<sub>g</sub>, 2025; Zhao et al., 2026). Rotar<sub>y p</sub>osition embeddin<sub>g</sub> (RoPE; Su et al., 2024) and its variants are widel<sub>y</sub> used in these models (Dube<sub>y</sub> et al., 2024; Yan<sub>g</sub> et al., 2025a; Gemma Team, 2025; Yan<sub>g</sub> et al., 2024; Liu et al., 2026) and have conse<sub>q</sub>uentl<sub>y</sub> become a <sub>p</sub>rominent tar<sub>g</sub>et for im<sub>p</sub>rovin<sub>g</sub> lon<sub>g</sub>-context <sub>p</sub>erformance (Chen et al., 2023b; Pen<sub>g</sub> et al., 2024; Din<sub>g</sub> et al.<sub>,</sub> 2024<sub>;</sub> Barbero et al.<sub>,</sub> 2025<sub>;</sub> Chen et al.<sub>,</sub> 2025<sub>;</sub> Zhan<sub>g</sub> et al.<sub>,</sub> 2024<sub>;</sub> Wan<sub>g</sub> et al.<sub>,</sub> 2026<sub>;</sub> bloc97<sub>,</sub> 2023<sub>;</sub> An et al., 2025; Yan<sub>g</sub> et al., 2025b; Xion<sub>g</sub> et al., 2025; Lu et al., 2025). Recent theoretical and em<sub>p</sub>irical <sub>s</sub>t<sub>u</sub>di<sub>es</sub> hi<sub>g</sub>hli<sub>g</sub>ht <sub>a</sub> t<sub>ens</sub>i<sub>on</sub> b<sub>e</sub>t<sub>ween preserv</sub>i<sub>ng con</sub>t<sub>en</sub>t <sub>pre</sub>f<sub>erences an</sub>d di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>hi<sub>ng</sub> t<sub>o</sub>k<sub>en pos</sub>iti<sub>ons as</sub> context len<sub>g</sub>th <sub>g</sub>rows (Barbero et al., 2025; Du et al., 2026; Urrutia et al., 2026; Liu, 2026; Xu et al., 2024; Jerad et al., 2026). Practical <sub>p</sub>ro<sub>g</sub>ress based on these findin<sub>g</sub>s re<sub>q</sub>uires a more <sub>p</sub>recise characterization of R<sub>o</sub>PE’<sub>s</sub> t<sub>ra</sub>d<sub>eo</sub>f <sub>un</sub>d<sub>er</sub> <sub>assump</sub>ti<sub>ons</sub> th<sub>a</sub>t b<sub>e</sub>tt<sub>er</sub> <sub>re</sub>fl<sub>ec</sub>t l<sub>earne</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>ons,</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith di<sub>agnos</sub>ti<sub>cs</sub> th<sub>a</sub>t <sub>prov</sub>id<sub>e ac</sub>ti<sub>ona</sub>bl<sub>e</sub> i<sub>ns</sub>i<sub>g</sub>ht<sub>s</sub> f<sub>or</sub> t<sub>arge</sub>t<sub>e</sub>d i<sub>n</sub>t<sub>erven</sub>ti<sub>on.</sub> W<sub>e a</sub>d<sub>vance</sub> b<sub>o</sub>th th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>an</sub>d <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l f<sub>ron</sub>t<sub>s.</sub>

O<sub>ur</sub> th<sub>eory</sub> <sub>quan</sub>tifi<sub>es</sub> t<sub>wo</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l f<sub>a</sub>il<sub>ure</sub> <sub>mo</sub>d<sub>es</sub> <sub>o</sub>f R<sub>o</sub>PE<sub>-</sub>b<sub>ase</sub>d <sub>s</sub>i<sub>ng</sub>l<sub>e-</sub>h<sub>ea</sub>d <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> i<sub>n</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> l<sub>ayer</sub> and establishes their <sub>q</sub>uantitative tradeof across ex<sub>p</sub>andin<sub>g</sub> context scales (Fi<sub>g</sub>ure 1a). As illustrated in Fi<sub>gure</sub> 1b<sub>,</sub> R<sub>o</sub>PE <sub>ro</sub>t<sub>a</sub>t<sub>es coor</sub>di<sub>na</sub>t<sub>e pa</sub>i<sub>rs o</sub>f <sub>query an</sub>d k<sub>ey ac</sub>ti<sub>va</sub>ti<sub>ons a</sub>t <sub>geome</sub>t<sub>r</sub>i<sub>ca</sub>ll<sub>y space</sub>d <sub>angu</sub>l<sub>ar</sub> f<sub>requenc</sub>i<sub>es across re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub>k<sub>en</sub> di<sub>s</sub>t<sub>ances.</sub> Th<sub>ese</sub> f<sub>requency c</sub>h<sub>anne</sub>l<sub>s, w</sub>hi<sub>c</sub>h th<sub>e a</sub>tt<sub>en</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sm</sub> <sub>aggrega</sub>t<sub>es,</sub> h<sub>ave</sub> dif<sub>eren</sub>t f<sub>unc</sub>ti<sub>ons: rap</sub>idl<sub>y ro</sub>t<sub>a</sub>ti<sub>ng componen</sub>t<sub>s pro</sub>d<sub>uce</sub> fl<sub>uc</sub>t<sub>ua</sub>ti<sub>ons w</sub>h<sub>ose mean</sub> i<sub>s c</sub>l<sub>ose</sub> to zero and whose variance we can bound anal<sub>y</sub>ticall<sub>y</sub> with a Gaussian a<sub>pp</sub>roximation (Du et al., 2026; Salem & Z<sub>yg</sub>mund, 1947; Aistleitner et al., 2024), whereas slowl<sub>y</sub> rotatin<sub>g</sub> com<sub>p</sub>onents are insensitive to adjacent positions and contribute little change to the attention score (Liu et al., 2024b; Hong et al., 2024). Consequently, these diferent functionalities triggers two fundamental failures: semantic reversal and positional insensitivity (§2). In semantic reversal, changing position alone reverses a query’s preference between ke<sub>y</sub>s, <sub>p</sub>otentiall<sub>y</sub> redirectin<sub>g</sub> attention toward distractors (Bansal et al., 2026; Barbero et al., 2025). In positional insensitivity, a one-token shift changes the attention score too little to distinguish adjacent <sub>pos</sub>iti<sub>ons, w</sub>hi<sub>c</sub>h <sub>can</sub> h<sub>ur</sub>t t<sub>as</sub>k<sub>s</sub> th<sub>a</sub>t <sub>requ</sub>i<sub>re re</sub>t<sub>r</sub>i<sub>ev</sub>i<sub>ng e</sub>l<sub>emen</sub>t<sub>s</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> b<sub>e</sub>f<sub>ore or a</sub>ft<sub>er a</sub> t<sub>arge</sub>t<sub>, as</sub> shown in fre<sub>q</sub>uenc<sub>y</sub>-scalin<sub>g</sub> based len<sub>g</sub>th-extension techni<sub>q</sub>ues (Pen<sub>g</sub> et al., 2024; bloc97, 2023; Wu et al., 2026). As context length grows, Figure 1a illustrates how preserving adjacent positional sensitivity imposes <sub>a</sub> hi<sub>g</sub>h<sub>er m</sub>i<sub>n</sub>i<sub>mum</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency norm s</sub>h<sub>are, ra</sub>i<sub>s</sub>i<sub>ng</sub> th<sub>e ca</sub>lib<sub>ra</sub>t<sub>e</sub>d l<sub>ower</sub> b<sub>oun</sub>d <sub>on reversa</sub>l <sub>r</sub>i<sub>s</sub>k <sub>an</sub>d <sub>s</sub>h<sub>r</sub>i<sub>n</sub>ki<sub>ng</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e reg</sub>i<sub>on.</sub> U<sub>n</sub>d<sub>er</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>ca</sub>lib<sub>ra</sub>ti<sub>on an</sub>d <sub>parame</sub>t<sub>er con</sub>diti<sub>ons,</sub> thi<sub>s</sub> t<sub>ra</sub>d<sub>eo</sub>f <sub>y</sub>i<sub>e</sub>ld<sub>s a</sub> <sub>con</sub>t<sub>ex</sub>t<sub>-</sub>l<sub>eng</sub>th b<sub>oun</sub>d $M _ { \mathrm { m a x } }$ b<sub>eyon</sub>d <sub>w</sub>hi<sub>c</sub>h <sub>a g</sub>i<sub>ven a</sub>tt<sub>en</sub>ti<sub>on-score marg</sub>i<sub>n can no</sub> l<sub>onger avo</sub>id b<sub>o</sub>th f<sub>a</sub>il<sub>ures</sub> within the calibrated windows (§3). Cruciall<sub>y</sub>, our finite-window bounds accommodate arbitrar<sub>y</sub> fixed <sub>q</sub>uer<sub>y</sub> <sub>an</sub>d k<sub>ey vec</sub>t<sub>ors ex</sub>t<sub>rac</sub>t<sub>e</sub>d f<sub>rom</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng unequa</sub>l <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>magn</sub>it<sub>u</sub>d<sub>es across c</sub>h<sub>anne</sub>l<sub>s.</sub> B<sub>y</sub> avoidin<sub>g</sub> restrictive assum<sub>p</sub>tions on token distributions or coordinate re<sub>g</sub>ularit<sub>y</sub> across channels (such as in Xu et al. (2024); Barbero et al. (2025); Du et al. (2026); see also §I.1), our anal<sub>y</sub>sis establishes a realistic th<sub>eore</sub>ti<sub>ca</sub>l f<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> thi<sub>s spec</sub>t<sub>ra</sub>l <sub>ce</sub>ili<sub>ng.</sub>

![](images/928b355ec92d70017176909ebfe3618be5df033007dfec2b28bff32961317606.jpg)  
(a) Semantic-positional tradeof

![](images/2c2d92f4d6c919ba50cc7f1e632c3806d07b2bf58546483e20bd0f07f2fd6912.jpg)  
(b) Rotary frequency dynamics  
Figure 1: The fundamental semantic-positional tradeof and rotary frequency dynamics. (a) The candidate region for semantic stability and adjacent positional sensitivity shrinks with context length. The <sub>orange</sub> <sub>an</sub>d bl<sub>ue</sub> <sub>curves</sub> <sub>represen</sub>t th<sub>e</sub> <sub>seman</sub>ti<sub>c</sub> <sub>upper</sub> li<sub>m</sub>it <sub>an</sub>d <sub>pos</sub>iti<sub>ona</sub>l l<sub>ower</sub> li<sub>m</sub>it <sub>on</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> <sub>norm s</sub>h<sub>are.</sub> Th<sub>e green reg</sub>i<sub>on con</sub>t<sub>a</sub>i<sub>ns can</sub>did<sub>a</sub>t<sub>e va</sub>l<sub>ues sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> b<sub>o</sub>th <sub>requ</sub>i<sub>remen</sub>t<sub>s;</sub> b<sub>eyon</sub>d $M _ { \mathrm { m a x } } ,$ <sub>, no suc</sub>h value remains within the calibrated famil<sub>y</sub>. (b) RoPE rotates coordinate <sub>p</sub>airs of <sub>q</sub>uer<sub>y</sub> and ke<sub>y</sub> vectors b<sub>y</sub> f<sub>requency-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>ang</sub>l<sub>es</sub> <sub>across</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub>k<sub>en</sub> di<sub>s</sub>t<sub>ances.</sub> R<sub>ap</sub>idl<sub>y</sub> <sub>ro</sub>t<sub>a</sub>ti<sub>ng</sub> <sub>componen</sub>t<sub>s,</sub> <sub>w</sub>h<sub>en</sub> <sub>aggrega</sub>t<sub>e</sub>d <sub>across c</sub>h<sub>anne</sub>l<sub>s,</sub> f<sub>orm unpre</sub>di<sub>c</sub>t<sub>a</sub>bl<sub>e</sub> G<sub>auss</sub>i<sub>an no</sub>i<sub>se</sub> th<sub>a</sub>t d<sub>es</sub>t<sub>a</sub>bili<sub>zes</sub> t<sub>o</sub>k<sub>en pre</sub>f<sub>erences.</sub> Sl<sub>ow</sub>l<sub>y ro</sub>t<sub>a</sub>ti<sub>ng</sub> components remain predictable but produce marginal score changes that fail to resolve adjacent positions. C<sub>onsequen</sub>tl<sub>y,</sub> th<sub>e</sub> d<sub>om</sub>i<sub>nance o</sub>f dif<sub>eren</sub>t f<sub>requency</sub> b<sub>an</sub>d<sub>s</sub> t<sub>r</sub>i<sub>ggers</sub> di<sub>s</sub>ti<sub>nc</sub>t f<sub>a</sub>il<sub>ure mo</sub>d<sub>es.</sub> P<sub>rec</sub>i<sub>se</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons</sub> are <sub>g</sub>iven in §3.2 and A<sub>pp</sub>endix A.5.

O<sub>ur</sub> th<sub>eory revea</sub>l<sub>s cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s on m</sub>iti<sub>ga</sub>ti<sub>ng</sub> b<sub>o</sub>th f<sub>a</sub>il<sub>ures s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y.</sub> H<sub>owever, we a</sub>l<sub>so</sub> fi<sub>n</sub>d th<sub>a</sub>t th<sub>ese</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> i<sub>mp</sub>l<sub>y a pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub>d b<sub>as</sub>i<sub>s</sub> f<sub>or</sub> id<sub>en</sub>tif<sub>y</sub>i<sub>ng w</sub>hi<sub>c</sub>h <sub>wea</sub>k<sub>ness</sub> t<sub>o pr</sub>i<sub>or</sub>iti<sub>ze an</sub>d h<sub>ow</sub> t<sub>o a</sub>dd<sub>ress</sub> it f<sub>or a</sub> given task. Building on our theoretical insights, we introduce RoPE Profiler, a lightweight diagnostic tool that seamlessl<sub>y</sub> inte<sub>g</sub>rates into existin<sub>g</sub> benchmarks (§4.1). RoPE Profiler reuses <sub>q</sub>uer<sub>y</sub> and ke<sub>y</sub> activations <sub>cac</sub>h<sub>e</sub>d d<sub>ur</sub>i<sub>ng eva</sub>l<sub>ua</sub>ti<sub>on an</sub>d i<sub>ncurs</sub> littl<sub>e a</sub>dditi<sub>ona</sub>l <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>over</sub>h<sub>ea</sub>d<sub>.</sub> R<sub>o</sub>PE P<sub>ro</sub>fil<sub>er</sub> t<sub>urns</sub> th<sub>e</sub> t<sub>wo</sub> th<sub>eore</sub>ti<sub>ca</sub>l f<sub>a</sub>il<sub>ure cr</sub>it<sub>er</sub>i<sub>a</sub> i<sub>n</sub>t<sub>o prac</sub>ti<sub>ca</sub>l di<sub>agnos</sub>ti<sub>cs, supp</sub>l<sub>emen</sub>ti<sub>ng s</sub>t<sub>an</sub>d<sub>ar</sub>d b<sub>enc</sub>h<sub>mar</sub>k <sub>scores w</sub>ith t<sub>wo</sub> metrics: the Semantic Score measures the stabilit of ke references across ositions while the Positional Score measures sensitivity across adjacent positions. Our theory provides a quantitative basis for two training free interventions: rescaling high-frequency components to adjust their relative strength and freezing their rotations to su<sub>pp</sub>ress <sub>p</sub>osition-induced fluctuations (Qiao et al., 2025; Barbero et al., 2025; Chen et al., 2025). Across 49 task confi<sub>g</sub>urations from nine benchmark sources (described in A<sub>pp</sub>endix E), the evaluated models show broadl<sub>y</sub> similar rankin<sub>g</sub>s of their relative semantic and <sub>p</sub>ositional weaknesses (§4.2). Guided b<sub>y</sub> th<sub>ese</sub> di<sub>agnos</sub>ti<sub>c pro</sub>fil<sub>es, our</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng-</sub>f<sub>ree</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>ons y</sub>i<sub>e</sub>ld b<sub>es</sub>t <sub>o</sub>b<sub>serve</sub>d <sub>accuracy ga</sub>i<sub>ns o</sub>f <sub>up</sub> t<sub>o</sub> 20 percentage points on Qwen3-8B and 25 percentage points on Llama-3.1-8B-Instruct, demonstrating that s ectral dia nosis rovides efective tar eted uidance for lon -context o timization (§4.2).

Althou<sub>g</sub>h our theor<sub>y</sub> ali<sub>g</sub>ns with Du et al. (2026)’s <sub>p</sub>essimistic assessment of RoPE’s intrinsic lon<sub>g</sub>-context li<sub>m</sub>it<sub>a</sub>ti<sub>ons, our prec</sub>i<sub>se spec</sub>t<sub>ra</sub>l <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on revea</sub>l<sub>s concre</sub>t<sub>e oppor</sub>t<sub>un</sub>iti<sub>es</sub> f<sub>or</sub> t<sub>arge</sub>t<sub>e</sub>d i<sub>mprovemen</sub>t<sub>s</sub> in practice. Our findings demonstrate that substantial optimization space remains simply by adjusting f<sub>requency a</sub>ll<sub>oca</sub>ti<sub>ons across</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> h<sub>ea</sub>d<sub>s.</sub> W<sub>e</sub> b<sub>e</sub>li<sub>eve</sub> f<sub>ur</sub>th<sub>er progress</sub> li<sub>es</sub> i<sub>n</sub> i<sub>ncorpora</sub>ti<sub>ng</sub> R<sub>o</sub>PE P<sub>ro</sub>fil<sub>er</sub>’<sub>s</sub> di<sub>agnos</sub>ti<sub>cs</sub> i<sub>n</sub>t<sub>o</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>an</sub>d l<sub>ong-con</sub>t<sub>ex</sub>t <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on.</sub> B<sub>eyon</sub>d <sub>un</sub>if<sub>orm</sub> f<sub>requency</sub> <sub>sca</sub>li<sub>ng,</sub> d<sub>es</sub>i<sub>gn</sub>i<sub>ng a</sub>tt<sub>en</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sms</sub> th<sub>a</sub>t <sub>ass</sub>i<sub>gn comp</sub>l<sub>emen</sub>t<sub>ary ro</sub>l<sub>es</sub> t<sub>o</sub> dif<sub>eren</sub>t h<sub>ea</sub>d<sub>s can</sub> b<sub>e</sub>tt<sub>er</sub> b<sub>a</sub>l<sub>ance</sub> content retrieval and <sub>p</sub>ositional reasonin<sub>g</sub> (Shi et al., 2026; Wan<sub>g</sub> et al., 2026; Xion<sub>g</sub> et al., 2025; Xiao et al., 2025; Tan<sub>g</sub> et al., 2025; Wu et al., 2025a). In the lon<sub>g</sub>er term, full<sub>y</sub> overcomin<sub>g</sub> these intrinsic constraints will re<sub>q</sub>uire rethinkin<sub>g</sub> attention architectures and how <sub>p</sub>ositions are encoded in foundation models (Kimi Team, 2026; Press et al., 2022a; Kazemnejad et al., 2023; Yang et al., 2025b; Hua et al., 2025).

## 2. Two Failure Modes of RoPE: Semantic Reversal and Positional Insensitivity

A good positional embedding (PE) in transformers should achieve two objectives: 1) retain the attention’s ability to distinguish diferent tokens: for a given query, the attention mechanism distinguishes two keys by havin diferent scores, and PEs should not et in the wa of this feature; 2) distinguish diferent positions: var<sub>y</sub>in<sub>g</sub> the score for the same ke<sub>y</sub> across locations informs the model of token <sub>p</sub>ositions (Su et al., 2024; Vaswani et al., 2017; Shaw et al., 2018; Press et al., 2022b). Unmasked attention with <sub>p</sub>osition-inde<sub>p</sub>endent in<sub>p</sub>uts is <sub>p</sub>ermutation-e<sub>q</sub>uivariant (Lee et al., 2019), althou<sub>g</sub>h causal lan<sub>g</sub>ua<sub>g</sub>e models can learn <sub>p</sub>ositional information without ex<sub>p</sub>licit <sub>p</sub>ositional encodin<sub>g</sub>s (Haviv et al., 2022). RoPE encodes relative distance b<sub>y</sub> <sub>ro</sub>t<sub>a</sub>ti<sub>ng</sub> <sub>query</sub> <sub>an</sub>d k<sub>ey</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>pa</sub>i<sub>rs,</sub> <sub>ma</sub>ki<sub>ng</sub> th<sub>e</sub>i<sub>r</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> <sub>score</sub> <sub>pos</sub>iti<sub>on-</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>.</sub>

## 2.1. Background

RoPE as a Sum of Rotated Complex Numbers. Let � and � denote the query and key vectors from an <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> h<sub>ea</sub>d<sub>.</sub> Th<sub>e</sub>i<sub>r sca</sub>l<sub>e</sub>d d<sub>o</sub>t <sub>pro</sub>d<sub>uc</sub>t f<sub>orms</sub> th<sub>e pre-so</sub>ft<sub>max a</sub>tt<sub>en</sub>ti<sub>on score, w</sub>hi<sub>c</sub>h <sub>conver</sub>t<sub>s</sub> i<sub>n</sub>t<sub>o we</sub>i<sub>g</sub>ht<sub>s</sub> for combinin<sub>g</sub> value vectors. RoPE (Su et al., 2024) injects <sub>p</sub>osition information b<sub>y</sub> rotatin<sub>g</sub> coordinate <sub>p</sub>airs <sub>o</sub>f <sub>quer</sub>i<sub>es</sub> <sub>an</sub>d k<sub>eys.</sub> L<sub>e</sub>t $p _ { q }$ <sub>an</sub>d $p _ { k }$ <sub>represen</sub>t th<sub>e</sub> t<sub>o</sub>k<sub>en</sub> <sub>pos</sub>iti<sub>ons,</sub> <sub>an</sub>d l<sub>e</sub>t $m = p _ { q } - p _ { k }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance.</sub> F<sub>or a</sub> h<sub>ea</sub>d <sub>w</sub>ith ℎ <sub>ro</sub>t<sub>ary coor</sub>di<sub>na</sub>t<sub>e pa</sub>i<sub>rs an</sub>d b<sub>ase</sub> $B ,$ <sub>coor</sub>di<sub>na</sub>t<sub>e pa</sub>i<sub>r � ro</sub>t<sub>a</sub>t<sub>es</sub> b<sub>y ang</sub>l<sub>e</sub> $p \omega _ { n }$ <sub>a</sub>t <sub>pos</sub>iti<sub>on</sub> $p ,$ <sub>w</sub>h<sub>ere</sub> $\omega _ { n } = B ^ { - n / h }$ f<sub>or</sub> $n = 0 , \ldots , h - 1$ <sub>.</sub> Whil<sub>e</sub> th<sub>e</sub> b<sub>ase</sub> f<sub>requency</sub> $\omega _ { n }$ d<sub>ecays smoo</sub>thl<sub>y w</sub>ith <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> i<sub>n</sub>d<sub>ex</sub> $n ,$ i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>componen</sub>t<sub>s ex</sub>hibit <sub>s</sub>t<sub>ar</sub>kl<sub>y</sub> di<sub>spara</sub>t<sub>e</sub> b<sub>e</sub>h<sub>av</sub>i<sub>ors across</sub> l<sub>ong con</sub>t<sub>ex</sub>t<sub>s.</sub><sup>1</sup>

Followin<sub>g</sub> the com<sub>p</sub>lex-valued formulation of RoPE in Zhu et al. (2024), let $Q _ { n } = q _ { n , 1 } + i q _ { n , 2 }$ <sub>an</sub>d $K _ { n } = k _ { n , 1 } + i k _ { n , 2 }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e comp</sub>l<sub>ex represen</sub>t<sub>a</sub>ti<sub>ons o</sub>f <sub>eac</sub>h <sub>ro</sub>t<sub>ary pa</sub>i<sub>r, w</sub>ith $i ^ { 2 } = - 1$ <sub>.</sub> With <sub>a</sub>tt<sub>en</sub>ti<sub>on sca</sub>l<sub>e</sub> $c _ { \mathrm { a t t } } ,$ t<sup>h</sup>e rotar<sub>y</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on score s</sub>i<sub>mp</sub>lifi<sub>es</sub> t<sub>o</sub>

$$
S _ { q , k } ( m ) = \mathrm { R e } \sum _ { n = 0 } ^ { h - 1 } z _ { n } e ^ { i m \omega _ { n } } , \qquad z _ { n } = c _ { \mathrm { a t t } } Q _ { n } \overline { { K _ { n } } } ,\tag{1}
$$

where Re takes the real part and the overline denotes complex conjugation. Here, token content determines th<sub>e s</sub>t<sub>a</sub>ti<sub>c amp</sub>lit<sub>u</sub>d<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> ${ \mathfrak { z } } _ { n } ,$ <sub>w</sub>hil<sub>e re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance � mo</sub>d<sub>u</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e</sub> d<sub>ynam</sub>i<sub>c ro</sub>t<sub>a</sub>ti<sub>on p</sub>h<sub>ases</sub> $m \omega _ { n }$ We write $S ( m )$ <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>query</sub> <sub>an</sub>d k<sub>ey</sub> <sub>are</sub> <sub>c</sub>l<sub>ear</sub> f<sub>rom</sub> th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t<sub>.</sub>

The Janus Face of RoPE: Sustaining Content and Localizing Tokens. By applying frequency-dependent <sub>ro</sub>t<sub>a</sub>ti<sub>ons across re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ances,</sub> R<sub>o</sub>PE <sub>ena</sub>bl<sub>es a</sub>tt<sub>en</sub>ti<sub>on</sub> t<sub>o pro</sub>d<sub>uce</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>scores accor</sub>di<sub>ng</sub> t<sub>o</sub> b<sub>o</sub>th t<sub>o</sub>k<sub>en</sub> <sub>con</sub>t<sub>en</sub>t <sub>vec</sub>t<sub>ors an</sub>d <sub>spa</sub>ti<sub>a</sub>l <sub>pos</sub>iti<sub>ons.</sub> E<sub>x</sub>t<sub>ens</sub>i<sub>ve pr</sub>i<sub>or s</sub>t<sub>u</sub>di<sub>es see</sub>k t<sub>o</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> h<sub>ow</sub> R<sub>o</sub>PE <sub>success</sub>f<sub>u</sub>ll<sub>y</sub> <sub>rea</sub>li<sub>zes</sub> th<sub>ese</sub> t<sub>wo core</sub> f<sub>unc</sub>ti<sub>ons.</sub> Th<sub>ese e</sub>f<sub>or</sub>t<sub>s span ear</sub>l<sub>y</sub> d<sub>ecoup</sub>l<sub>e</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> d<sub>es</sub>i<sub>gns, approx</sub>i<sub>ma</sub>t<sub>e</sub> <sub>a</sub>dditi<sub>ve</sub> d<sub>ecompos</sub>iti<sub>ons an</sub>d f<sub>requency-w</sub>i<sub>se</sub> f<sub>unc</sub>ti<sub>ona</sub>l di<sub>v</sub>i<sub>s</sub>i<sub>on</sub> i<sub>n</sub> l<sub>earne</sub>d R<sub>o</sub>PE <sub>represen</sub>t<sub>a</sub>ti<sub>ons, exam</sub>i<sub>na-</sub> ti<sub>ons o</sub>f l<sub>oca</sub>lit <sub>an</sub>d <sub>s mme</sub>t<sub>r un</sub>d<sub>er ro</sub>l<sub>e s</sub>h<sub>u</sub>fli<sub>n , an</sub>d <sub>s m</sub>b<sub>o</sub>li<sub>c-versus- os</sub>iti<sub>ona</sub>l <sub>ermu</sub>t<sub>a</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>ro</sub>b<sub>e</sub> ex<sub>p</sub>ressive ca<sub>p</sub>acit<sub>y</sub> (Kitaev & Klein, 2018; Ke et al., 2021; Chen et al., 2023a; Han & Ji, 2025; Barbero et al., 2025<sub>;</sub> Urrutia et al.<sub>,</sub> 2026<sub>;</sub> Oka et al.<sub>,</sub> 2026a<sub>,</sub>b<sub>;</sub> Chen & Yan<sub>,</sub> 2024<sub>;</sub> Gu et al.<sub>,</sub> 2026<sub>;</sub> Hon<sub>g</sub> et al.<sub>,</sub> 2024<sub>;</sub> Jin et al., 2025). In essence, these investigations explain how RoPE succeeds. However, analyzing its functional <sub>capa</sub>bilit<sub>y prov</sub>id<sub>es on</sub>l<sub>y an</sub> i<sub>ncomp</sub>l<sub>e</sub>t<sub>e p</sub>i<sub>c</sub>t<sub>ure.</sub> Th<sub>e more cr</sub>iti<sub>ca</sub>l <sub>ques</sub>ti<sub>on</sub> i<sub>s</sub> h<sub>ow</sub> thi<sub>s</sub> d<sub>ua</sub>l <sub>respons</sub>ibilit<sub>y</sub> b<sub>ecomes un</sub>b<sub>a</sub>l<sub>ance</sub>d <sub>as</sub> th<sub>e con</sub>t<sub>ex</sub>t <sub>sca</sub>l<sub>e con</sub>ti<sub>nues</sub> t<sub>o expan</sub>d<sub>.</sub>

## 2.2. Formal Definitions of Semantic Reversal and Positional Insensitivity

RoPE gives attention a way to distinguish positions, but what is the cost? Examining this question through the lens of failure modes, Du et al. (2026) formalize four distinct vulnerabilities in RoPE-based attention. In this work, we focus on two core failure modes: semantic reversal and positional insensitivity. These two dimensions <sub>p</sub>rove theoreticall<sub>y</sub> suficient to uncover the fundamental context limits of RoPE in §3, while <sub>rema</sub>i<sub>n</sub>i<sub>ng prac</sub>ti<sub>ca</sub>ll<sub>y</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve</sub> f<sub>or our</sub> di<sub>agnos</sub>ti<sub>c</sub> t<sub>oo</sub>l d<sub>es</sub>i<sub>gn</sub> i<sub>n</sub> $\ S 4$ <sub>.</sub> F<sub>ur</sub>th<sub>ermore, our</sub> th<sub>eore</sub>ti<sub>ca</sub>l f<sub>ramewor</sub>k <sub>na</sub>t<sub>ura</sub>ll<sub>y</sub> <sub>ex</sub>t<sub>en</sub>d<sub>s</sub> t<sub>o</sub> th<sub>e</sub> <sub>o</sub>th<sub>er</sub> f<sub>a</sub>il<sub>ure</sub> <sub>mo</sub>d<sub>es</sub> <sub>ana</sub>l<sub>yze</sub>d i<sub>n</sub> <sub>pr</sub>i<sub>or</sub> lit<sub>era</sub>t<sub>ure,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>we</sub> d<sub>e</sub>t<sub>a</sub>il i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub> R<sub>ea</sub>l<sub>-wor</sub>ld t<sub>as</sub>k<sub>s</sub> d<sub>o no</sub>t <sub>necessar</sub>il<sub>y su</sub>f<sub>er</sub> f<sub>rom</sub> b<sub>o</sub>th <sub>vu</sub>l<sub>nera</sub>biliti<sub>es equa</sub>ll<sub>y, an</sub>d i<sub>n</sub> $\ S 4 . 2$ <sub>we</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub> h<sub>ow</sub> <sub>spec</sub>ifi<sub>c</sub> t<sub>as</sub>k f<sub>am</sub>ili<sub>es</sub> <sub>are</sub> <sub>governe</sub>d b<sub>y</sub> di<sub>s</sub>ti<sub>nc</sub>t f<sub>a</sub>il<sub>ure</sub> <sub>mo</sub>d<sub>es.</sub> B<sub>e</sub>l<sub>ow,</sub> <sub>we</sub> f<sub>orma</sub>li<sub>ze</sub> th<sub>e</sub> <sub>ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l d<sub>e</sub>fi<sub>n</sub>iti<sub>ons</sub> <sub>o</sub>f th<sub>ese</sub> t<sub>wo pr</sub>i<sub>mary</sub> f<sub>a</sub>il<sub>ure mo</sub>d<sub>es.</sub>

Semantic reversal (Diferent Tokens, Same Position). Motivated by Du et al. (2026), we examine th<sub>e</sub> <sub>s</sub>t<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> <sub>pre</sub>f<sub>erences</sub> b<sub>e</sub>t<sub>ween</sub> di<sub>s</sub>ti<sub>nc</sub>t t<sub>o</sub>k<sub>ens</sub> <sub>across</sub> <sub>vary</sub>i<sub>ng</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ances.</sub> Fi<sub>gure</sub> 2<sub>a</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>es an</sub> i<sub>ns</sub>t<sub>ance o</sub>f <sub>seman</sub>ti<sub>c reversa</sub>l<sub>.</sub> F<sub>or a query an</sub>d t<sub>wo</sub> k<sub>eys,</sub> l<sub>e</sub>t $D ( m ) = S _ { + } ( m ) - S _ { - } ( m )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r</sub> <sub>or</sub>d<sub>ere</sub>d <sub>score</sub> <sub>marg</sub>i<sub>n.</sub> With<sub>ou</sub>t l<sub>oss</sub> <sub>o</sub>f <sub>genera</sub>lit<sub>y,</sub> <sub>we</sub> <sub>assume</sub> $D ( 0 ) > 0$ . A semantic reversal occurs at an<sub>y</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance w</sub>h<sub>ere</sub> $D ( m ) < 0$ <sub>.</sub> T<sub>o cap</sub>t<sub>ure overa</sub>ll b<sub>e</sub>h<sub>av</sub>i<sub>or w</sub>ith<sub>ou</sub>t b<sub>e</sub>i<sub>ng</sub> bi<sub>ase</sub>d b<sub>y</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>pos</sub>iti<sub>ons,</sub>

Which word is a country, France or chair? Query: country Key: France Key: chair  
![](images/3aaa7c4d0458556f5fcc834d1c0bca8735794fecfe51e7a477655020cbf5ed7c.jpg)  
(a) Semantic reversal  
(b) Positional insensitivity  
Figure 2: Illustrative task consequences of two RoPE failure modes. (a) For query country, the scores o<sup>f k</sup>eys France an<sup>d</sup> chair reverse or<sup>d</sup>er <sup>b</sup>etween v<sup>i</sup>rtua<sup>l</sup> re<sup>l</sup>at<sup>i</sup>ve <sup>di</sup>stances 0 an<sup>d</sup> $m _ { \star }$ <sub>.</sub> A<sub>s a</sub>tt<sub>en</sub>ti<sub>on</sub> di<sub>rec</sub>t<sub>s</sub> <sup>i</sup>n<sup>f</sup>ormat<sup>i</sup>on rout<sup>i</sup>ng towar<sup>d hi</sup>g<sup>h</sup>er scores, t<sup>hi</sup>s reversa<sup>l f</sup>avors t<sup>h</sup>e <sup>di</sup>stractor chair over t<sup>h</sup>e semant<sup>i</sup>ca<sup>ll</sup>y re<sup>l</sup>evant France at re<sup>l</sup>at<sup>i</sup>ve <sup>di</sup>stance $m _ { \star }$ and can contribute to an incorrect <sub>p</sub>rediction. (b) For <sub>q</sub>uer<sub>y</sub> A and <sup>k</sup>ey B, t<sup>h</sup>e vertica<sup>l b</sup>rac<sup>k</sup>et mar<sup>k</sup>s a margina<sup>l</sup> norma<sup>l</sup>ized score gap <sup>b</sup>etween adjacent virtua<sup>l</sup> re<sup>l</sup>ative distances, $g _ { z } ( m ) < \zeta$ . This leaves the model unable to distinguish between the two adjacent positions.

A … B, how far is A from B? Query: A Key: B Distance: m  
![](images/6cc9ae41256e2c3ff63ad4cc52deed9d9e5ff6cd6a68c709fdf5c01aa6cb16ab.jpg)

$$
m + 1
$$

<sub>we eva</sub>l<sub>ua</sub>t<sub>e</sub> h<sub>ow</sub> f<sub>requen</sub>tl<sub>y seman</sub>ti<sub>c reversa</sub>l<sub>s occur across</sub> th<sub>e en</sub>ti<sub>re con</sub>t<sub>ex</sub>t <sub>w</sub>i<sub>n</sub>d<sub>ow o</sub>f l<sub>eng</sub>th � <sub>as</sub> th<sub>e</sub> <sub>seman</sub>ti<sub>c reversa</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y:</sub>

$$
p _ { \mathrm { r e v } } ( D ; M ) : = \frac 1 M \sum _ { m = 0 } ^ { M - 1 } \mathbf { 1 } _ { \{ D ( m ) < 0 \} } = \mathbb { P } \lbrack D ( \mathbf { m } ) < 0 \rbrack , \qquad \mathbf { m } \sim \mathrm { U n i f } \{ 0 , \dots , M - 1 \} .\tag{2}
$$

Th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>es</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> <sub>pre</sub>f<sub>erence</sub> <sub>across</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>sequence.</sub>

Positional insensitivity (Same Token, Diferent Positions). Motivated by Liu (2026); Sun et al. (2026), <sub>we eva</sub>l<sub>ua</sub>t<sub>e</sub> h<sub>ow sens</sub>iti<sub>ve</sub>l<sub>y an a</sub>tt<sub>en</sub>ti<sub>on score respon</sub>d<sub>s</sub> t<sub>o one-</sub>t<sub>o</sub>k<sub>en c</sub>h<sub>anges</sub> i<sub>n re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance w</sub>hil<sub>e</sub> h<sub>o</sub>ldi<sub>ng</sub> t<sub>o</sub>k<sub>en represen</sub>t<sub>a</sub>ti<sub>ons</sub> fi<sub>xe</sub>d<sub>.</sub> F<sub>or</sub> di<sub>s</sub>t<sub>an</sub>t <sub>or seman</sub>ti<sub>ca</sub>ll<sub>y unre</sub>l<sub>a</sub>t<sub>e</sub>d t<sub>o</sub>k<sub>en pa</sub>i<sub>rs, reso</sub>l<sub>v</sub>i<sub>ng exac</sub>t <sub>s</sub>i<sub>ng</sub>l<sub>e-</sub>t<sub>o</sub>k<sub>en o</sub>f<sub>se</sub>t<sub>s carr</sub>i<sub>es</sub> littl<sub>e opera</sub>ti<sub>ona</sub>l <sub>va</sub>l<sub>ue;</sub> h<sub>owever,</sub> i<sub>n re</sub>t<sub>r</sub>i<sub>eva</sub>l t<sub>as</sub>k<sub>s</sub> th<sub>a</sub>t d<sub>eman</sub>d <sub>prec</sub>i<sub>se re</sub>l<sub>a</sub>ti<sub>ve</sub> order, as illustrated in Figure 2b, distinguishing adjacent positions becomes essential to resolve local sequence structure (Wu et al., 2025b; Wan<sub>g</sub> et al., 2024a).

A<sub>n</sub> i<sub>n</sub>t<sub>u</sub>iti<sub>ve me</sub>t<sub>r</sub>i<sub>c</sub> i<sub>s</sub> t<sub>o measure</sub> th<sub>e raw score c</sub>h<sub>ange</sub> $\left| S ( m + 1 ) - S ( m ) \right|$ across adjacent locations. However, thi<sub>s raw</sub> dif<sub>erence sca</sub>l<sub>es w</sub>ith <sub>represen</sub>t<sub>a</sub>ti<sub>on magn</sub>it<sub>u</sub>d<sub>e,</sub> l<sub>eav</sub>i<sub>ng</sub> th<sub>e raw gap vu</sub>l<sub>nera</sub>bl<sub>e</sub> t<sub>o ar</sub>bit<sub>rary</sub> norm variations across heads (Qi et al., 2026; Jin et al., 2025). To isolate intrinsic <sub>p</sub>ositional sensitivit<sub>y</sub> f<sub>rom overa</sub>ll <sub>vec</sub>t<sub>or magn</sub>it<sub>u</sub>d<sub>e, we norma</sub>li<sub>ze</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>score</sub> dif<sub>erence</sub> b<sub>y</sub> th<sub>e pa</sub>i<sub>r</sub>’<sub>s coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>norm</sub> $\begin{array} { r } { U ( z ) = \| z \| _ { 2 } = ( \sum _ { n } | z _ { n } | ^ { 2 } ) ^ { \bar { 1 / 2 } } } \end{array}$ . The resulting normalized adjacent response is

$$
g _ { z } ( m ) : = \frac { | S ( m + 1 ) - S ( m ) | } { U ( z ) } , \qquad U ( z ) > 0 , \quad 0 \leq m \leq M - 2 .\tag{3}
$$

F<sub>or a c</sub>h<sub>osen response</sub> th<sub>res</sub>h<sub>o</sub>ld $\zeta > 0 ,$ , positional insensitivity occurs when $g _ { z } ( m ) < \zeta$ <sub>,</sub> <sub>quan</sub>tif<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>a</sub>il<sub>ure</sub> t<sub>o preserve</sub> l<sub>oca</sub>l <sub>pos</sub>iti<sub>ona</sub>l <sub>reso</sub>l<sub>u</sub>ti<sub>on.</sub>

## 3. A Theory of Long-Context Failures: The Tradeof Between Semantic Stability and Positional Sensitivity

R<sub>o</sub>PE <sub>re</sub>li<sub>es on a s</sub>h<sub>are</sub>d <sub>se</sub>t <sub>o</sub>f <sub>coor</sub>di<sub>na</sub>t<sub>e ro</sub>t<sub>a</sub>ti<sub>ons</sub> t<sub>o preserve con</sub>t<sub>en</sub>t <sub>pre</sub>f<sub>erences an</sub>d di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h <sub>pos</sub>iti<sub>ons.</sub> I<sub>n</sub> thi<sub>s sec</sub>ti<sub>on, we revea</sub>l <sub>a</sub> t<sub>ra</sub>d<sub>eo</sub>f b<sub>e</sub>t<sub>ween seman</sub>ti<sub>c s</sub>t<sub>a</sub>bilit<sub>y an</sub>d <sub>pos</sub>iti<sub>ona</sub>l <sub>sens</sub>iti<sub>v</sub>it<sub>y</sub> i<sub>n</sub> R<sub>o</sub>PE<sub>,</sub> b<sub>o</sub>th <sub>governe</sub>d by the norm share of high-frequency components. Over long context windows, an insuficient high-frequency contribution leaves attention blind to adjacent positions, whereas amplifying these fast oscillations triggers semantic reversal. Below, we formalize high-frequency score variation under a finite-window framework, and then derive a context-length limit where avoiding both failure modes becomes mathematically impossible.

![](images/f0fba1ded489b894a50da33758e0c42e4ac27e142fbcfeef76da4152f3f074f3.jpg)  
(a) Qwen3-8B distribution

![](images/f2d2d928e37d6ad6195d493f4085af979476d858972bbd6fafa3d6776c1ffb23.jpg)  
(b) Qwen3-8B split selection

![](images/96fdff84543cd656f27e6ebf8f3e185c7935d45a797a8a658f762db48d89b336.jpg)  
(c) Llama-3.1-8B split selection

Figure 3: (a) Under our high-frequency split, the predicted score distribution closely matches the empirical distribution. (b,c) Defining the high-frequency band. The curves show how randomness changes as more <sub>componen</sub>t<sub>s are</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d<sub>.</sub> O<sub>ur</sub> th<sub>eory-gu</sub>id<sub>e</sub>d <sub>sp</sub>lit <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> th<sub>e</sub> l<sub>arges</sub>t b<sub>an</sub>d <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> th<sub>e sp</sub>lit <sub>cr</sub>it<sub>er</sub>i<sub>on w</sub>hil<sub>e</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> hi<sub>g</sub>h <sub>ran</sub>d<sub>omness.</sub> Th<sub>e</sub> t<sub>ren</sub>d<sub>s</sub> <sub>vary</sub> <sub>w</sub>ith <sub>con</sub>t<sub>ex</sub>t l<sub>eng</sub>th<sub>,</sub> <sub>an</sub>d l<sub>arger</sub> � <sub>a</sub>ll<sub>ows</sub> <sub>more</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> com<sub>p</sub>onents, consistent with our theor<sub>y</sub>. (See A<sub>pp</sub>endix D.2.)

## 3.1. High-Frequency Norm Share and Score Variation

T<sub>o ana</sub>l<sub>yze</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> fl<sub>uc</sub>t<sub>ua</sub>ti<sub>ons r</sub>i<sub>gorous</sub>l<sub>y, we</sub> fi<sub>rs</sub>t <sub>spec</sub>if<sub>y an ana</sub>l<sub>y</sub>ti<sub>ca</sub>l <sub>momen</sub>t t<sub>o</sub>l<sub>erance</sub> $\varepsilon \in ( 0 , 1 / 2 )$ th<sub>a</sub>t <sub>governs</sub> th<sub>e prec</sub>i<sub>s</sub>i<sub>on o</sub>f fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow momen</sub>t <sub>con</sub>t<sub>ro</sub>l<sub>.</sub> U<sub>n</sub>d<sub>er</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>geome</sub>t<sub>r</sub>i<sub>c gr</sub>id $\omega _ { n } = \rho ^ { n }$ <sub>w</sub>ith $\rho = B ^ { - 1 / h }$ <sub>,</sub> d<sub>eman</sub>di<sub>ng a</sub> ti<sub>g</sub>ht<sub>er ana</sub>l<sub>y</sub>ti<sub>ca</sub>l t<sub>o</sub>l<sub>erance requ</sub>i<sub>res a</sub> hi<sub>g</sub>h<sub>er</sub> f<sub>requency cu</sub>t<sub>o</sub>f<sub>, y</sub>i<sub>e</sub>ldi<sub>ng</sub> th<sub>e cer</sub>tifi<sub>e</sub>d hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> b<sub>an</sub>d

$$
H = \left\{ n \in \{ 0 , \ldots , h - 1 \} : M \omega _ { n } \geq { \frac { 2 C _ { \mathrm { F } } } { ( 1 - \rho ) \varepsilon } } \right\} ,\tag{4}
$$

<sub>w</sub>h<sub>ere</sub> $C _ { \mathrm { F } } > 0$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>un</sub>i<sub>versa</sub>l F<sub>our</sub>i<sub>er-</sub>f<sub>rame</sub> b<sub>oun</sub>d <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>2<sub>.</sub> Thi<sub>s</sub> <sub>�-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>cr</sub>it<sub>er</sub>i<sub>on</sub> <sub>guaran</sub>t<sub>ees</sub> th<sub>a</sub>t <sub>a</sub>ll <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> i<sub>n</sub> � <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>osc</sub>ill<sub>a</sub>ti<sub>ons</sub> <sub>over</sub> <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> � t<sub>o</sub> <sub>suppress</sub> <sub>cross-</sub>f<sub>requency</sub> i<sub>n</sub>t<sub>er</sub>f<sub>erence.</sub>

Proposition 1 (Analytical moment guarantees under finite windows). For any tolerance $\varepsilon \in ( 0 , 1 / 2 )$ and certified window � where � is nonempty, the high-frequency attention score $\begin{array} { r } { S _ { H } ( m ) : = \Re \sum _ { n \in H } z _ { n } e ^ { i m \omega _ { n } } } \end{array}$ and its coeficient norm $\begin{array} { r } { U _ { H } ( z ) : = ( \sum _ { n \in H } | z _ { n } | ^ { 2 } ) ^ { 1 / 2 } } \end{array}$ over uniformly distributed relative distances $m \sim \operatorname { U n i f } \{ 0 , \dots , M - 1 \}$ satisfy

$$
\left| \mathbb { E } _ { m } \left[ S _ { H } ( m ) \right] \right| \leq \varepsilon U _ { H } ( z ) \qquad a n d \qquad \left| \mathrm { V a r } _ { m } ( S _ { H } ( m ) ) - \frac { 1 } { 2 } U _ { H } ( z ) ^ { 2 } \right| \leq \varepsilon U _ { H } ( z ) ^ { 2 } .\tag{5}
$$

F<sub>u</sub>ll <sub>proo</sub>f<sub>s are</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> $\ S \operatorname { A } . 3 .$ . Followin<sub>g</sub> the statistical behavior observed b<sub>y</sub> Du et al. (2026), which is an a<sub>pp</sub>lication of the lacunar<sub>y</sub> Central Limit Theorem (Salem & Z<sub>yg</sub>mund, 1947; Aistleitner et al., 2024), <sub>sums o</sub>f <sub>rap</sub>idl<sub>y osc</sub>ill<sub>a</sub>ti<sub>ng componen</sub>t<sub>s across re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ances</sub> b<sub>e</sub>h<sub>ave as zero-mean</sub> G<sub>auss</sub>i<sub>an</sub> fl<sub>uc</sub>t<sub>ua</sub>ti<sub>ons,</sub> <sub>y</sub>i<sub>e</sub>ldi<sub>ng</sub> th<sub>e opera</sub>ti<sub>ona</sub>l <sub>score approx</sub>i<sub>ma</sub>ti<sub>on</sub> $S _ { H } ( m ) \approx N ( 0 , U _ { H } ( z ) ^ { 2 } / 2 )$

Error-Bounded Spectral Partitioning. Proposition 1 links the frequency cutof � directly to the analytical <sub>error</sub> t<sub>o</sub>l<sub>erance �, es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng prec</sub>i<sub>se con</sub>t<sub>ro</sub>l <sub>over</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow momen</sub>t<sub>s.</sub> D<sub>eman</sub>di<sub>ng</sub> ti<sub>g</sub>ht<sub>er prec</sub>i<sub>s</sub>i<sub>on</sub> f<sub>orces s</sub>l<sub>ower componen</sub>t<sub>s ou</sub>t <sub>o</sub>f �<sub>, ra</sub>i<sub>s</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>requency cu</sub>t<sub>o</sub>f <sub>accor</sub>di<sub>ng</sub>l<sub>y.</sub> Whil<sub>e pr</sub>i<sub>or ana</sub>l<sub>yses assume</sub> s<sub>p</sub>ecific <sub>q</sub>uer<sub>y</sub> and ke<sub>y</sub> distributions or am<sub>p</sub>litude re<sub>g</sub>ularit<sub>y</sub> to derive bounds (Xu et al., 2024; Barbero et al., 2025; Du et al., 2026), our finite-window moment bounds hold for arbitrar<sub>y</sub> fixed activations extracted from t<sub>ra</sub>i<sub>ne</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s, accommo</sub>d<sub>a</sub>ti<sub>ng genera</sub>l <sub>p</sub>h<sub>ases an</sub>d <sub>non-un</sub>if<sub>orm coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>magn</sub>it<sub>u</sub>d<sub>es.</sub> Fi<sub>gure</sub> 3 ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> th<sub>e emp</sub>i<sub>r</sub>i<sub>ca</sub>l b<sub>e</sub>h<sub>av</sub>i<sub>or o</sub>f <sub>our</sub> th<sub>eore</sub>ti<sub>ca</sub>ll<sub>y pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>par</sub>titi<sub>on</sub>i<sub>ng across</sub> 8k<sub>,</sub> 32k<sub>, an</sub>d 128k <sub>re</sub>l<sub>a</sub>ti<sub>ve-</sub>di<sub>s</sub>t<sub>ance</sub> <sub>w</sub>i<sub>n</sub>d<sub>ows.</sub>

The Governing Norm Share $r _ { H }$ For context len<sub>g</sub>th �<sub>,</sub> we <sub>q</sub>uantif<sub>y</sub> the <sub>p</sub>ro<sub>p</sub>ortion of ener<sub>gy</sub> concentrated i<sub>n</sub> f<sub>as</sub>t <sub>osc</sub>ill<sub>a</sub>ti<sub>ons</sub> th<sub>roug</sub>h th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency norm s</sub>h<sub>are</sub>

$$
r _ { H } ( z ; M ) : = \frac { U _ { H } ( z ) } { U ( z ) } \in [ 0 , 1 ] , \qquad \mathrm { w h e r e } U ( z ) : = \| z \| _ { 2 } = \left( \sum _ { n = 0 } ^ { h - 1 } | z _ { n } | ^ { 2 } \right) ^ { 1 / 2 } .\tag{6}
$$

The <sub>p</sub>arameter $r _ { H } ( z ; M )$ <sub>serves as</sub> th<sub>e cen</sub>t<sub>ra</sub>l <sub>ma</sub>th<sub>ema</sub>ti<sub>ca</sub>l <sub>p</sub>i<sub>vo</sub>t <sub>connec</sub>ti<sub>ng spec</sub>t<sub>ra</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>ons</sub> t<sub>o</sub> d<sub>owns</sub>t<sub>ream</sub> failure modes. S<sub>p</sub>ecificall<sub>y</sub>, the two failure modes defined in §2.2 constrain this ener<sub>gy</sub> share in o<sub>pp</sub>osin<sub>g</sub> di<sub>rec</sub>ti<sub>ons.</sub> F<sub>or an or</sub>d<sub>ere</sub>d <sub>score marg</sub>i<sub>n,</sub> th<sub>e ca</sub>lib<sub>ra</sub>t<sub>e</sub>d l<sub>ower</sub> b<sub>oun</sub>d <sub>on seman</sub>ti<sub>c reversa</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s</sub> <sub>non-</sub>d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n</sub> $r _ { H }$ (A<sub>pp</sub>endix A.5), while retainin<sub>g</sub> adjacent <sub>p</sub>ositional sensitivit<sub>y</sub> for the same mar<sub>g</sub>in i<sub>mposes</sub> <sub>a</sub> <sub>con</sub>t<sub>ex</sub>t<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t l<sub>ower</sub> b<sub>oun</sub>d <sub>on</sub> $r _ { H }$ (A<sub>pp</sub>endix A.4.1). This dual constraint establishes the foundation for the fundamental tradeof derived in §3.2.

## 3.2. The Semantic and Positional Tradeof Limits Context Length

Distinguishing adjacent positions requires attention scores to vary, while preserving token preferences d<sub>eman</sub>d<sub>s</sub> th<sub>a</sub>t th<sub>e</sub>i<sub>r re</sub>l<sub>a</sub>ti<sub>ve or</sub>d<sub>er</sub>i<sub>ng rema</sub>i<sub>ns s</sub>t<sub>a</sub>bl<sub>e.</sub> Hi<sub>g</sub>h<sub>-</sub>f<sub>requency componen</sub>t<sub>s govern</sub> b<sub>o</sub>th <sub>requ</sub>i<sub>remen</sub>t<sub>s.</sub> Suppressing these oscillations stabilizes token orderings at the expense of adjacent positional sensitivity. A<sub>s</sub> th<sub>e con</sub>t<sub>ex</sub>t <sub>w</sub>i<sub>n</sub>d<sub>ow expan</sub>d<sub>s, s</sub>l<sub>ower</sub> f<sub>requenc</sub>i<sub>es en</sub>t<sub>er</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> b<sub>an</sub>d<sub>,</sub> l<sub>eav</sub>i<sub>ng</sub> f<sub>ewer s</sub>t<sub>a</sub>ti<sub>c</sub> <sub>componen</sub>t<sub>s</sub> t<sub>o</sub> <sub>sus</sub>t<sub>a</sub>i<sub>n</sub> l<sub>oca</sub>l dif<sub>erences.</sub> M<sub>a</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> <sub>a</sub> t<sub>arge</sub>t <sub>pos</sub>iti<sub>ona</sub>l <sub>gap</sub> i<sub>mposes</sub> <sub>a</sub> <sub>con</sub>t<sub>ex</sub>t<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t l<sub>ower</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency norm s</sub>h<sub>are, y</sub>i<sub>e</sub>ldi<sub>ng a non-</sub>d<sub>ecreas</sub>i<sub>ng</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>on reversa</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>un</sub>d<sub>er</sub> th<sub>e ca</sub>lib<sub>ra</sub>ti<sub>on con</sub>diti<sub>ons</sub> b<sub>e</sub>l<sub>ow.</sub>

Theorem 1 (Context-length limit on joint reliability (informal)). For afixed ordered score margin and reversal and adjacent-response tolerances � and � satisfying the conditions in Appendix A.5, there exists a context-length bound $M _ { \mathrm { m a x } }$ such that, for every calibrated window,

$$
M > M _ { \mathrm { m a x } } \implies p _ { \mathrm { r e v } } ( M ) > \delta \quad o r \quad \operatorname* { m a x } _ { 0 \leq m \leq M - 2 } g ( m ) < \zeta ,\tag{7}
$$

where $p _ { \mathrm { r e v } } ( M )$ and $g ( m )$ denote the reversal probability and normalized adjacent response of the same margin over relative distance �. Beyond $M _ { \mathrm { m a x } } ,$ this margin cannot satisfy both requirements within the calibrated family.

A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub>5 <sub>g</sub>i<sub>ves</sub> th<sub>e prec</sub>i<sub>se ca</sub>lib<sub>ra</sub>t<sub>e</sub>d<sub>-w</sub>i<sub>n</sub>d<sub>ow con</sub>diti<sub>ons,</sub> th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on o</sub>f $M _ { \mathrm { m a x . } }$ <sub>, an</sub>d th<sub>e</sub> f<sub>orma</sub>l <sub>vers</sub>i<sub>on o</sub>f Th<sub>eorem</sub> 1<sub>.</sub> Th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on</sub> b<sub>a</sub>l<sub>ances</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency s</sub>h<sub>are requ</sub>i<sub>re</sub>d f<sub>or</sub> l<sub>oca</sub>l <sub>sens</sub>iti<sub>v</sub>it<sub>y</sub> <sub>aga</sub>i<sub>ns</sub>t th<sub>e reversa</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> it i<sub>n</sub>d<sub>uces.</sub>

A RoPE-of-War between Content and Position (Diagnose What the Task Needs First). While existing lit<sub>era</sub>t<sub>ure</sub> id<sub>en</sub>tifi<sub>es concep</sub>t<sub>ua</sub>l t<sub>ens</sub>i<sub>ons</sub> i<sub>n ro</sub>t<sub>ary a</sub>tt<sub>en</sub>ti<sub>on regar</sub>di<sub>ng con</sub>t<sub>en</sub>t <sub>se</sub>l<sub>ec</sub>ti<sub>v</sub>it<sub>y an</sub>d <sub>pos</sub>iti<sub>ona</sub>l t<sub>rac</sub>k in<sub>g</sub> (Barbero et al., 2025; Du et al., 2026; Urrutia et al., 2026), we formalize a <sub>q</sub>uantitative tradeof showin<sub>g</sub> th<sub>a</sub>t <sub>seman</sub>ti<sub>c s</sub>t<sub>a</sub>bilit<sub>y an</sub>d <sub>pos</sub>iti<sub>ona</sub>l <sub>sens</sub>iti<sub>v</sub>it<sub>y are mu</sub>t<sub>ua</sub>ll<sub>y cons</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>un</sub>d<sub>er ro</sub>t<sub>ary represen</sub>t<sub>a</sub>ti<sub>ons.</sub> Theorem 1 gives a conditional bound on joint score-level reliability, determined by the frequency grid, the t<sub>wo re</sub>li<sub>a</sub>bilit<sub>y</sub> t<sub>o</sub>l<sub>erances, an</sub>d th<sub>e ca</sub>lib<sub>ra</sub>ti<sub>on error.</sub> Withi<sub>n</sub> thi<sub>s se</sub>tti<sub>ng, s</sub>hifti<sub>ng spec</sub>t<sub>ra</sub>l <sub>we</sub>i<sub>g</sub>ht <sub>c</sub>h<sub>anges</sub> th<sub>e</sub> b<sub>a</sub>l<sub>ance</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>wo requ</sub>i<sub>remen</sub>t<sub>s.</sub> Th<sub>e ana</sub>l<sub>ys</sub>i<sub>s sugges</sub>t<sub>s wea</sub>k<sub>en</sub>i<sub>ng</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> t<sub>o</sub> stabilize key preferences and strengthening them to distinguish adjacent positions, depending on the task. §4 operationalizes this principle by introducing RoPE Profiler to evaluate these competing vulnerabilities <sub>an</sub>d <sub>gu</sub>id<sub>e</sub> t<sub>arge</sub>t<sub>e</sub>d i<sub>n</sub>t<sub>erven</sub>ti<sub>ons.</sub>

## 4. Diagnosis and Targeted Mitigation of Long-Context Failures with RoPE Profiler

Building on our theoretical criteria, we introduce RoPE Profiler to turn spectral failure analysis into actionable diagnostics and targeted model adjustments. Moving beyond prior diagnostic frameworks that primarily identif<sub>y</sub> internal vulnerabilities (A<sub>pp</sub>endix I.4), our tool establishes a brid<sub>g</sub>e between s<sub>p</sub>ectral state <sub>p</sub>rofilin<sub>g</sub> and tar<sub>g</sub>eted trainin<sub>g</sub>-free miti<sub>g</sub>ation. Fi<sub>g</sub>ure 4 outlines the five-sta<sub>g</sub>e <sub>p</sub>i<sub>p</sub>eline. In §4.1, we formalize how cached activations yield the Semantic Score and Positional Score, evaluate relative failure susceptibility across benchmark settin<sub>g</sub>s, and derive theor<sub>y</sub>-motivated interventions. In §4.2, we evaluate these dia<sub>g</sub>nostic <sub>p</sub>rofiles <sub>an</sub>d t<sub>arge</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>u</sub>l<sub>a</sub>ti<sub>ons across</sub> di<sub>verse</sub> t<sub>as</sub>k<sub>s an</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> d<sub>emons</sub>t<sub>ra</sub>ti<sub>ng su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l <sub>accuracy ga</sub>i<sub>ns across</sub> <sub>a compre</sub>h<sub>ens</sub>i<sub>ve</sub> l<sub>ong-con</sub>t<sub>ex</sub>t <sub>su</sub>it<sub>e.</sub>

## 4.1. From Spectral Theory to Practical Profiling and Interventions

D<sub>ur</sub>i<sub>ng</sub> th<sub>e</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d i<sub>n</sub>f<sub>erence</sub> <sub>pass</sub> <sub>o</sub>f b<sub>enc</sub>h<sub>mar</sub>k <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> R<sub>o</sub>PE P<sub>ro</sub>fil<sub>er</sub> <sub>samp</sub>l<sub>es</sub> <sub>query</sub> <sub>an</sub>d k<sub>ey</sub> t<sub>o</sub>k<sub>ens</sub> t<sub>o cac</sub>h<sub>e</sub> th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>t<sub>erna</sub>l <sub>ac</sub>ti<sub>va</sub>ti<sub>ons</sub> $q , k$ across tar<sub>g</sub>et attention heads (selection details and heatma<sub>p</sub>s in A<sub>pp</sub>endix F) (Fi<sub>g</sub>ure 4, Ste<sub>p</sub>s 1 and 2). Holdin<sub>g</sub> these content vectors fixed, we evaluate virtual attention <sub>scores</sub> b<sub>y sweep</sub>i<sub>ng</sub> R<sub>o</sub>PE <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ances across</sub> th<sub>e con</sub>t<sub>ex</sub>t <sub>w</sub>i<sub>n</sub>d<sub>ow,</sub> i<sub>ncurr</sub>i<sub>ng zero a</sub>dditi<sub>ona</sub>l <sub>mo</sub>d<sub>e</sub>l i<sub>n</sub>f<sub>erence passes.</sub>

Theoretical Scores from Cached States (Figure 4, Step 3). Our theoretical criteria translate directly i<sub>n</sub>t<sub>o</sub> t<sub>wo con</sub>ti<sub>nuous me</sub>t<sub>r</sub>i<sub>cs.</sub> F<sub>or</sub> l<sub>oca</sub>l <sub>pos</sub>iti<sub>ona</sub>l <sub>sens</sub>iti<sub>v</sub>it<sub>y, we eva</sub>l<sub>ua</sub>t<sub>e query–</sub>k<sub>ey pa</sub>i<sub>rs</sub> $( q , k )$ <sup>b</sup>y a<sup>v</sup>e<sup>r</sup>ag<sup>in</sup>g the normalized adjacent response $g _ { q , k } ( m ) = | S ( m + 1 ) - S ( m ) | / U ( z )$ <sub>across re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ances</sub> $m ,$ d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> Positional Score as ${ \bar { g } } .$ <sub>.</sub> F<sub>or</sub> <sub>seman</sub>ti<sub>c</sub> <sub>s</sub>t<sub>a</sub>bilit<sub>y,</sub> <sub>we</sub> <sub>cons</sub>t<sub>ruc</sub>t t<sub>r</sub>i<sub>p</sub>l<sub>e</sub>t<sub>s</sub> $( q , k _ { 1 } , k _ { 2 } )$ <sub>w</sub>ith di<sub>s</sub>ti<sub>nc</sub>t k<sub>eys an</sub>d <sub>compu</sub>t<sub>e</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow pro</sub>b<sub>a</sub>bilit<sub>y</sub> $P _ { \mathrm { r e v } }$ th<sub>a</sub>t <sub>pos</sub>iti<sub>on s</sub>hift<sub>s</sub> i<sub>nver</sub>t th<sub>e</sub>i<sub>r re</sub>f<sub>erence ran</sub>ki<sub>ng a</sub>t di<sub>s</sub>t<sub>ance zero, us</sub>i<sub>ng</sub> our moment bounds and Gaussian approximation. The Semantic Score is defined as $1 - \bar { P } _ { \mathrm { r e v } }$ t<sub>o ensure</sub> th<sub>a</sub>t hi h<sub>er va</sub>l<sub>ues are</sub> b<sub>e</sub>tt<sub>er</sub> f<sub>or</sub> b<sub>o</sub>th <sub>me</sub>t<sub>r</sub>i<sub>cs.</sub>

Relative Failure Susceptibility via Normalized Ranks (Figure 4, Step 4). Directly comparing raw scores <sub>across</sub> dif<sub>eren</sub>t t<sub>as</sub>k<sub>s or</sub> f<sub>a</sub>il<sub>ure mo</sub>d<sub>es</sub> i<sub>s</sub> i<sub>ne</sub>f<sub>ec</sub>ti<sub>ve</sub> b<sub>ecause a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e score magn</sub>it<sub>u</sub>d<sub>es</sub> l<sub>ac</sub>k <sub>a s</sub>h<sub>are</sub>d <sub>sca</sub>l<sub>e,</sub> <sub>an</sub>d <sub>meanw</sub>hil<sub>e seman</sub>ti<sub>c an</sub>d <sub>pos</sub>iti<sub>ona</sub>l <sub>me</sub>t<sub>r</sub>i<sub>cs opera</sub>t<sub>e on</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>numer</sub>i<sub>ca</sub>l <sub>ranges.</sub> Th<sub>ere</sub>f<sub>ore, we ran</sub>k <sub>a</sub>ll <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d t<sub>as</sub>k<sub>s separa</sub>t<sub>e</sub>l<sub>y</sub> b<sub>y eac</sub>h <sub>me</sub>t<sub>r</sub>i<sub>c, y</sub>i<sub>e</sub>ldi<sub>ng seman</sub>ti<sub>c ran</sub>k <sub>�� an</sub>d <sub>pos</sub>iti<sub>ona</sub>l <sub>ran</sub>k <sub>��.</sub> W<sub>e use</sub> th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>seman</sub>ti<sub>c</sub> f<sub>a</sub>il<sub>ure</sub> <sub>suscep</sub>tibilit<sub>y</sub> $w _ { S } \in [ 0 , 1 ]$ t<sub>o</sub> <sub>s</sub>h<sub>ow</sub> th<sub>e</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>ran</sub>k <sub>gap:</sub> <sub>a</sub> l<sub>arger</sub> <sub>��</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> <sub>grea</sub>t<sub>er suscep</sub>tibilit<sub>y</sub> t<sub>o seman</sub>ti<sub>c reversa</sub>l <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o pos</sub>iti<sub>ona</sub>l i<sub>nsens</sub>iti<sub>v</sub>it<sub>y w</sub>ithi<sub>n</sub> th<sub>e re</sub>f<sub>erence se</sub>t<sub>.</sub>

![](images/3239bc6bc2cbd139ec8de4c541544992fdfe8405d85d0f68e4778695727de65e.jpg)  
Figure 4: Augmenting accuracy-based evaluation with failure diagnostics. For a given model and the 49 reference task confi<sub>g</sub>urations (described in §E), we use <sub>q</sub>uer<sub>y</sub> and ke<sub>y</sub> activations from standard evaluation t<sub>o</sub> <sub>compu</sub>t<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>seman</sub>ti<sub>c</sub> f<sub>a</sub>il<sub>ure</sub> <sub>suscep</sub>tibilit<sub>y</sub> $\boldsymbol { w _ { S } }$ <sub>.</sub> A hi<sub>g</sub>h<sub>er</sub> $\boldsymbol { w _ { S } }$ <sub>sugges</sub>t<sub>s</sub> <sub>grea</sub>t<sub>er</sub> <sub>suscep</sub>tibilit<sub>y</sub> t<sub>o</sub> <sub>seman</sub>ti<sub>c</sub> <sub>reversa</sub>l <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o pos</sub>iti<sub>ona</sub>l i<sub>nsens</sub>iti<sub>v</sub>it<sub>y w</sub>ithi<sub>n</sub> th<sub>e re</sub>f<sub>erence se</sub>t<sub>.</sub>

Theory-Motivated Interventions (Figure 4, Step 5). The spectral tradeof driven by the high-frequency <sub>norm</sub> <sub>s</sub>h<sub>are</sub> <sub>mo</sub>ti<sub>va</sub>t<sub>es</sub> <sub>a</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng-</sub>f<sub>ree</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> <sub>approac</sub>h <sub>w</sub>ith t<sub>wo</sub> <sub>oppos</sub>i<sub>ng</sub> di<sub>rec</sub>ti<sub>ons</sub> <sub>on</sub> t<sub>as</sub>k<sub>-sens</sub>iti<sub>ve</sub> h<sub>ea</sub>d<sub>s.</sub> G<sub>u</sub>id<sub>e</sub>d b<sub>y</sub> t<sub>as</sub>k di<sub>agnos</sub>ti<sub>c</sub> <sub>pro</sub>fil<sub>es,</sub> <sub>we</sub> <sub>ass</sub>i<sub>gn</sub> th<sub>e</sub> <sub>upper</sub> h<sub>a</sub>lf <sub>o</sub>f <sub>�� ran</sub>ki<sub>ngs</sub> t<sub>o</sub> th<sub>e</sub> <sub>seman</sub>ti<sub>c</sub> di<sub>rec</sub>ti<sub>on,</sub> attenuatin<sub>g</sub> hi<sub>g</sub>h fre<sub>q</sub>uencies or freezin<sub>g</sub> their rotations to su<sub>pp</sub>ress <sub>p</sub>osition-induced fluctuations (Barbero et al., 2025; Chen et al., 2025). The remainin<sub>g</sub> settin<sub>g</sub>s follow the <sub>p</sub>ositional direction, usin<sub>g</sub> � > 1 to am<sub>p</sub>lif<sub>y</sub> adjacent score resolution (Qiao et al., 2025; Chian<sub>g</sub> & Yo<sub>g</sub>atama, 2025; Mikaeili et al., 2026). While � d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> di<sub>rec</sub>ti<sub>on,</sub> th<sub>e spec</sub>ifi<sub>c sca</sub>li<sub>ng sca</sub>l<sub>ar �</sub> i<sub>s se</sub>l<sub>ec</sub>t<sub>e</sub>d f<sub>rom can</sub>did<sub>a</sub>t<sub>e va</sub>l<sub>ues.</sub> Thi<sub>s</sub> strate<sub>gy</sub> de<sub>p</sub>arts from conventional fre<sub>q</sub>uenc<sub>y</sub> mani<sub>p</sub>ulations (Hua et al., 2025; Xion<sub>g</sub> et al., 2025; Li et al., 2026; An et al., 2025) b<sub>y</sub> d<sub>y</sub>namicall<sub>y</sub> assi<sub>g</sub>nin<sub>g</sub> the intervention direction <sub>p</sub>er task from its dia<sub>g</sub>nostic <sub>p</sub>rofile <sub>an</sub>d <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ng mo</sub>difi<sub>ca</sub>ti<sub>ons s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> t<sub>o</sub> t<sub>as</sub>k<sub>-sens</sub>iti<sub>ve</sub> h<sub>ea</sub>d<sub>s.</sub> W<sub>e repor</sub>t th<sub>e</sub> b<sub>es</sub>t <sub>o</sub>b<sub>serve</sub>d <sub>ou</sub>t<sub>come</sub> f<sub>or eac</sub>h <sub>se</sub>tti<sub>ng, w</sub>hil<sub>e</sub> A<sub>ppen</sub>di<sub>x</sub> H d<sub>ocumen</sub>t<sub>s comp</sub>l<sub>e</sub>t<sub>e sweeps across a</sub>ll <sub>can</sub>did<sub>a</sub>t<sub>e con</sub>fi<sub>gura</sub>ti<sub>ons.</sub>

## 4.2. Empirical Validation Across Diverse Long-Context Tasks

Targeted Intervention Gains Reveal Untapped Potential. Guided by diagnostic susceptibility profiles, our targeted training-free interventions achieve performance gains across a majority of evaluated tasks (63.3% of Qwen confi<sub>g</sub>urations and 65.3% of Llama confi<sub>g</sub>urations). As illustrated in Fi<sub>g</sub>ure 5, accurac<sub>y</sub> <sub>g</sub>ains can reach 20% on Qwen3-8B and 25% on Llama-3.1-8B-Instruct, showing the practical value of adjusting the <sub>spec</sub>t<sub>ra</sub>l <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> id<sub>en</sub>tifi<sub>e</sub>d b<sub>y</sub> <sub>our</sub> th<sub>eory.</sub>

![](images/f3703c453a34f940dacfa87d648ac62d8420c14ba64adc67c530a3d0461bbc2b.jpg)  
Figure 5: Accuracy gains from our exploratory intervention search. Bars show the best observed i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> i<sub>n eac</sub>h <sub>se</sub>tti<sub>ng</sub>’<sub>s ass</sub>i<sub>gne</sub>d <sub>seman</sub>ti<sub>c or pos</sub>iti<sub>ona</sub>l di<sub>rec</sub>ti<sub>on m</sub>i<sub>nus</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l b<sub>ase</sub>li<sub>ne on</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>examp</sub>l<sub>es w</sub>ithi<sub>n eac</sub>h <sub>se</sub>tti<sub>ng.</sub> Th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>s exc</sub>l<sub>u</sub>d<sub>e</sub>d f<sub>rom</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e max</sub>i<sub>mum.</sub> E<sub>ac</sub>h <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> 49 task/confi<sub>g</sub>uration settin<sub>g</sub>s (described in A<sub>pp</sub>endix E) are ordered from left to ri<sub>g</sub>ht b<sub>y</sub> increasin<sub>g</sub> semantic f<sub>a</sub>il<sub>ure suscep</sub>tibilit<sub>y,</sub> f<sub>rom</sub> bl<sub>ue</sub> t<sub>o orange.</sub>

RoPE Profiler Reveals Intrinsic Task Properties. Be-<sub>yon</sub>d i<sub>n</sub>t<sub>erven</sub>ti<sub>on ga</sub>i<sub>ns,</sub> Fi<sub>gure</sub> 6 <sub>compares seman</sub>ti<sub>c</sub> f<sub>a</sub>il<sub>-</sub> <sub>ure</sub> <sub>suscep</sub>tibilit<sub>y</sub> <sub>across</sub> th<sub>e</sub> 49 <sub>s</sub>h<sub>are</sub>d <sub>se</sub>tti<sub>ngs</sub> t<sub>o</sub> i<sub>nspec</sub>t h<sub>ow</sub> f<sub>a</sub>il<sub>ure</sub> <sub>pa</sub>tt<sub>erns</sub> <sub>re</sub>l<sub>a</sub>t<sub>e</sub> t<sub>o</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> t<sub>as</sub>k <sub>s</sub>t<sub>ruc</sub>t<sub>ures.</sub> Across the the majority of tasks, the two models exhibit <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>suscep</sub>tibilit<sub>y</sub> <sub>ran</sub>ki<sub>ngs</sub> <sub>w</sub>ith <sub>a</sub> S<sub>pearman</sub> <sub>cor-</sub> <sub>re</sub>l<sub>a</sub>ti<sub>on o</sub>f $\rho = 0 . 6 5 8$ <sub>,</sub> d<sub>emons</sub>t<sub>ra</sub>ti<sub>ng</sub> th<sub>a</sub>t th<sub>e</sub> di<sub>agnos</sub>ti<sub>c</sub> <sub>cap</sub>t<sub>ures</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>c</sub> t<sub>as</sub>k <sub>proper</sub>ti<sub>es.</sub>

Specifically, reasoning tasks cluster in the upper half where semantic reversal dominates, while multi-key retrieval tasks occupy the lower half governed by positional insensitivity. In contrast, adjacent-element retrieval tasks deviates sharply between models, ranking in the top three for Qwen but b<sub>e</sub>t<sub>ween ran</sub>k<sub>s</sub> 34 <sub>an</sub>d 38 f<sub>or</sub> Ll<sub>ama, un</sub>d<sub>erscor</sub>i<sub>ng</sub> th<sub>a</sub>t h<sub>y</sub>b<sub>r</sub>id t<sub>as</sub>k<sub>s coup</sub>li<sub>ng seman</sub>ti<sub>c ma</sub>t<sub>c</sub>hi<sub>ng w</sub>ith l<sub>oca</sub>l <sub>or</sub>d<sub>er</sub> <sub>a</sub>l<sub>so re</sub>fl<sub>ec</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>-spec</sub>ifi<sub>c represen</sub>t<sub>a</sub>ti<sub>on c</sub>h<sub>o</sub>i<sub>ces.</sub> Th<sub>ese</sub> <sub>pa</sub>tt<sub>erns a</sub>li<sub>gn w</sub>ith <sub>prev</sub>i<sub>ous o</sub>b<sub>serva</sub>ti<sub>ons on</sub> dif<sub>eren</sub>t t<sub>as</sub>k settin<sub>g</sub>s (Goldman et al., 2024; Vodrahalli et al., 2024; Kuratov et al.<sub>,</sub> 2024<sub>;</sub> Modarressi et al.<sub>,</sub> 2025<sub>;</sub> Hsieh et al.<sub>,</sub> 2024a).

![](images/204f7aa2475895b2c3a80f8b0e9a2c6d9916919dd5f4aab0b6769b2a3fea2323.jpg)  
Figure 6: Semantic failure susceptibility across tasks. Points compare ranks of semantic f<sub>a</sub>il<sub>ure</sub> <sub>suscep</sub>tibilit<sub>y</sub> $\boldsymbol { w _ { S } }$ in Qwen3-8B and Llama-3.1-8B-Instruct for the 49 task/confi<sub>g</sub>uration <sub>se</sub>tti<sub>ngs</sub> d<sub>escr</sub>ib<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> E<sub>.</sub> M<sub>os</sub>t t<sub>as</sub>k<sub>s</sub> <sub>s</sub>h<sub>ow</sub> <sub>s</sub>i<sub>m</sub>il<sub>ar</sub> <sub>pa</sub>tt<sub>erns</sub> <sub>across</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>ree</sub> adjacent-element retrieval tasks difer.

## 5. Related Work

RoPE and context extension RoPE (Su et al., 2024) is widely used in open LLMs (Dubey et al., 2024; Yang et al., 2025a; Liu et al., 2026). Existin<sub>g</sub> extension methods involve fre<sub>q</sub>uenc<sub>y</sub> rescalin<sub>g</sub>, includin<sub>g</sub> NTK-aware scalin<sub>g</sub> (bloc97, 2023), PI (Chen et al., 2023b), ABF (Xion<sub>g</sub> et al., 2024), YaRN (Pen<sub>g</sub> et al., 2024), Lon<sub>g</sub>RoPE (Din<sub>g</sub> et al., 2024), and Resonance RoPE (Wan<sub>g</sub> et al., 2024b). Others mani<sub>p</sub>ulate fre<sub>q</sub>uenc<sub>y</sub> com<sub>p</sub>onents (Barbero et al., 2025; Hua et al., 2025; Chen et al., 2025; Li et al., 2026; Xion<sub>g</sub> et al., 2025), adjust scalin<sub>g</sub> <sub>p</sub>er head (Wan<sub>g</sub> et al., 2026), or reindex <sub>p</sub>ositions (An et al., 2025; Zhan<sub>g</sub> et al., 2024). These methods are t<sub>as</sub>k<sub>-agnos</sub>ti<sub>c, w</sub>hil<sub>e we</sub> di<sub>agnose</sub> t<sub>as</sub>k<sub>-spec</sub>ifi<sub>c</sub> f<sub>a</sub>il<sub>ures</sub> b<sub>e</sub>f<sub>ore</sub> d<sub>ec</sub>idi<sub>ng</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> di<sub>rec</sub>ti<sub>on.</sub>

Theory of RoPE frequencies Existing RoPE theories derive base-dependent context bounds (Xu et al., 2024; Liu et al., 2024b; Liu, 2026) and roles of low-fre<sub>q</sub>uenc<sub>y</sub> dimensions (Hon<sub>g</sub> et al., 2024; Jin et al., 2025). Prior works observe a tension between semantic and <sub>p</sub>ositional resolution (Barbero et al., 2025; Du et al., 2026; Urrutia et al., 2026). These anal<sub>y</sub>ses assume s<sub>p</sub>ecific distribution <sub>p</sub>ro<sub>p</sub>erties of vector s<sub>p</sub>ace or <sub>amp</sub>lit<sub>u</sub>d<sub>e regu</sub>l<sub>ar</sub>it<sub>y, w</sub>h<sub>ereas our</sub> b<sub>oun</sub>d<sub>s</sub> h<sub>o</sub>ld f<sub>or rea</sub>l<sub>-wor</sub>ld <sub>ac</sub>ti<sub>va</sub>ti<sub>ons w</sub>ith<sub>ou</sub>t <sub>a</sub>dditi<sub>ona</sub>l <sub>assump</sub>ti<sub>ons.</sub>

Internal diagnostics and specialized heads Retrieval heads (Wu et al., 2025a), attention sinks (Xiao et al., 2024), <sub>p</sub>osition-wise features (Don<sub>g</sub> et al., 2024), <sub>p</sub>ositional attention bias (Hsieh et al., 2024b), and other internal si<sub>g</sub>nals (Tan et al., 2026) ex<sub>p</sub>lain lon<sub>g</sub>-context behavior inside the model. Our RoPE Profiler <sub>a</sub>l<sub>so u</sub>tili<sub>zes</sub> th<sub>e cac</sub>h<sub>e</sub>d <sub>ac</sub>ti<sub>va</sub>ti<sub>ons</sub> b<sub>u</sub>t b<sub>o</sub>th id<sub>en</sub>tif<sub>y</sub> th<sub>eore</sub>ti<sub>c vu</sub>l<sub>nera</sub>biliti<sub>es an</sub>d di<sub>agnose w</sub>ith t<sub>as</sub>k<sub>-spec</sub>ifi<sub>c</sub> f<sub>a</sub>il<sub>ure suscep</sub>tibilit<sub>y.</sub> F<sub>or</sub> f<sub>ur</sub>th<sub>er</sub> di<sub>scuss</sub>i<sub>on, see</sub> A<sub>ppen</sub>di<sub>x</sub> I<sub>.</sub>

## 6. Conclusion

I<sub>n</sub> thi<sub>s paper, we</sub> h<sub>ave s</sub>t<sub>u</sub>di<sub>e</sub>d l<sub>ong-con</sub>t<sub>ex</sub>t f<sub>a</sub>il<sub>ures o</sub>f R<sub>o</sub>PE th<sub>roug</sub>h th<sub>e</sub> t<sub>ra</sub>d<sub>eo</sub>f b<sub>e</sub>t<sub>ween seman</sub>ti<sub>c s</sub>t<sub>a</sub>bilit<sub>y</sub> <sub>an</sub>d <sub>pos</sub>iti<sub>ona</sub>l <sub>sens</sub>iti<sub>v</sub>it<sub>y.</sub> Hi<sub>g</sub>h<sub>-</sub>f<sub>requency componen</sub>t<sub>s suppor</sub>t fi<sub>ne pos</sub>iti<sub>ona</sub>l di<sub>s</sub>ti<sub>nc</sub>ti<sub>ons w</sub>hil<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>i<sub>ng</sub> fl<sub>uc</sub>t<sub>ua</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>can</sub> d<sub>es</sub>t<sub>a</sub>bili<sub>ze</sub> t<sub>o</sub>k<sub>en</sub> <sub>pre</sub>f<sub>erences.</sub> O<sub>ur</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>accommo</sub>d<sub>a</sub>t<sub>es</sub> <sub>unequa</sub>l <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>magn</sub>it<sub>u</sub>d<sub>es across</sub> f<sub>requenc</sub>i<sub>es an</sub>d<sub>, un</sub>d<sub>er</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>ca</sub>lib<sub>ra</sub>ti<sub>on con</sub>diti<sub>ons, y</sub>i<sub>e</sub>ld<sub>s a con</sub>t<sub>ex</sub>t length bound on jointly avoiding semantic reversal and positional insensitivity. Building on these insights, <sub>we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d R<sub>o</sub>PE P<sub>ro</sub>fil<sub>er</sub> t<sub>o measure</sub> b<sub>o</sub>th <sub>vu</sub>l<sub>nera</sub>biliti<sub>es</sub> f<sub>rom cac</sub>h<sub>e</sub>d <sub>ac</sub>ti<sub>va</sub>ti<sub>ons an</sub>d <sub>u</sub>id<sub>e</sub> t<sub>ar e</sub>t<sub>e</sub>d frequency interventions. Across the evaluated Qwen and Llama settings, we observed broadly consistent <sub>pa</sub>tt<sub>erns o</sub>f f<sub>a</sub>il<sub>ure suscep</sub>tibilit<sub>y, an</sub>d <sub>our exp</sub>l<sub>ora</sub>t<sub>ory</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on searc</sub>h id<sub>en</sub>tifi<sub>e</sub>d <sub>accuracy ga</sub>i<sub>ns on a</sub> majority of settings without retraining. Together, these findings highlight the value of adapting frequency <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> t<sub>o</sub> th<sub>e nee</sub>d<sub>s o</sub>f <sub>eac</sub>h t<sub>as</sub>k<sub>.</sub> W<sub>e</sub> h<sub>ope</sub> thi<sub>s wor</sub>k i<sub>n</sub>f<sub>orms</sub> th<sub>e</sub> d<sub>es</sub>i<sub>gn o</sub>f l<sub>ong-con</sub>t<sub>ex</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> th<sub>a</sub>t <sub>preserve re</sub>l<sub>evan</sub>t t<sub>o</sub>k<sub>en pre</sub>f<sub>erences w</sub>hil<sub>e reso</sub>l<sub>v</sub>i<sub>ng</sub> th<sub>e pos</sub>iti<sub>ona</sub>l di<sub>s</sub>ti<sub>nc</sub>ti<sub>ons a</sub> t<sub>as</sub>k <sub>requ</sub>i<sub>res.</sub>

## Acknowledgments

W<sub>e</sub> th<sub>an</sub>k <sub>mem</sub>b<sub>ers</sub> <sub>o</sub>f th<sub>e</sub> Alt<sub>a</sub> <sub>group</sub> <sub>a</sub>t th<sub>e</sub> U<sub>n</sub>i<sub>vers</sub>it<sub>y</sub> <sub>o</sub>f Illi<sub>no</sub>i<sub>s</sub> U<sub>r</sub>b<sub>ana-</sub>Ch<sub>ampa</sub>i<sub>gn</sub> f<sub>or</sub> th<sub>e</sub>i<sub>r</sub> h<sub>e</sub>l<sub>p</sub>f<sub>u</sub>l f<sub>ee</sub>db<sub>ac</sub>k<sub>.</sub> This work was su<sub>pp</sub>orted in <sub>p</sub>art b<sub>y</sub> an Amazon AICE award<sub>,</sub> NSF Grant No. CHE2505932<sub>,</sub> and a Ca<sub>p</sub>ital One ASKS award.

## AI Use Statement

We used AI tools to aid or polish writing, adjust manuscript length and layout, create figures, organize the <sub>appen</sub>di<sub>ces, an</sub>d <sub>searc</sub>h f<sub>or re</sub>l<sub>a</sub>t<sub>e</sub>d lit<sub>era</sub>t<sub>ure.</sub> W<sub>e rev</sub>i<sub>ewe</sub>d <sub>a</sub>ll <sub>genera</sub>t<sub>e</sub>d <sub>sen</sub>t<sub>ences an</sub>d fi<sub>gures, an</sub>d <sub>c</sub>h<sub>ec</sub>k<sub>e</sub>d <sub>aga</sub>i<sub>ns</sub>t <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>sources</sub> f<sub>or</sub> th<sub>e va</sub>lidit<sub>y o</sub>f <sub>re</sub>l<sub>a</sub>t<sub>e</sub>d lit<sub>era</sub>t<sub>ure.</sub> W<sub>e</sub> t<sub>a</sub>k<sub>e respons</sub>ibilit<sub>y</sub> f<sub>or</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>con</sub>t<sub>en</sub>t <sub>o</sub>f thi<sub>s wor</sub>k<sub>.</sub>

## References

Ch<sub>r</sub>i<sub>s</sub>t<sub>op</sub>h Ai<sub>s</sub>tl<sub>e</sub>it<sub>ner,</sub> I<sub>s</sub>t<sub>v</sub>á<sub>n</sub> B<sub>er</sub>k<sub>es, an</sub>d R<sub>o</sub>b<sub>er</sub>t Ti<sub>c</sub>h<sub>y.</sub> L<sub>acunary sequences</sub> i<sub>n ana</sub>l<sub>ys</sub>i<sub>s, pro</sub>b<sub>a</sub>bilit<sub>y an</sub>d number theory. In Dijana Kreso, Joël Rivat, and Robert F. Tichy (eds.), Diophantine Problems: Determinism, Randomness and Applications, volume 62 of Panoramas et Synthèses, pp. 1–60. Société Mathématique de France, Par<sup>i</sup>s, France, 2024. URL https://smf.emath.fr/publications/problemes-diophanti ens-determinisme-alea-et-applications.

Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L-Eval: Instituting standardized evaluation for long context language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14388–14411, Bangkok, Thailand, August 2024. Assoc<sup>i</sup>at<sup>i</sup>on <sup>f</sup>or Computat<sup>i</sup>ona<sup>l</sup> L<sup>i</sup>ngu<sup>i</sup>st<sup>i</sup>cs. <sup>d</sup>o<sup>i</sup>: 10.18<sup>6</sup>53/v1/2024.ac<sup>l</sup>-<sup>l</sup>ong.77<sup>6</sup>. URL https: //aclanthology.org/2024.acl-long.776/.

Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan Gong, Yao Luo, Jingjing Xu, and Lingpeng Kong. Wh<sub>y</sub> does the efective context len<sub>g</sub>th of LLMs fall short? In Y. Yue<sub>,</sub> A. Gar<sub>g,</sub> N. Pen<sub>g,</sub> F. Sha<sub>,</sub> and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 54791–54810, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/884baf65392170763b27c9 14087bde01-Paper-Conference.pdf.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, A<sub>o</sub>h<sub>an</sub> Z<sub>eng,</sub> L<sub>e</sub>i H<sub>ou,</sub> Y<sub>ux</sub>i<sub>ao</sub> D<sub>ong,</sub> Ji<sub>e</sub> T<sub>ang, an</sub>d J<sub>uanz</sub>i Li<sub>.</sub> L<sub>ong</sub>B<sub>enc</sub>h<sub>:</sub> A bili<sub>ngua</sub>l<sub>, mu</sub>ltit<sub>as</sub>k b<sub>enc</sub>h<sub>mar</sub>k for long context understanding. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3119–3137<sub>,</sub> B<sub>a</sub>n<sub>g</sub>k<sub>o</sub>k<sub>,</sub> Th<sub>a</sub>il<sub>a</sub>nd<sub>,</sub> A<sub>ugus</sub>t 2024<sub>.</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r C<sub>o</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l Lin<sub>gu</sub>i<sub>s</sub>ti<sub>cs.</sub> d<sub>o</sub>i<sub>:</sub> 10<sub>.</sub>18653/v 1/2024.ac<sup>l</sup>-<sup>l</sup>ong.172. URL https://aclanthology.org/2024.acl-long.172/.

R<sub>ac</sub>hit B<sub>ansa</sub>l<sub>,</sub> A<sub>s</sub>t<sub>on</sub> Zh<sub>ang,</sub> Ri<sub>s</sub>h<sub>a</sub>bh Ti<sub>war</sub>i<sub>,</sub> L<sub>ov</sub>i<sub>s</sub>h M<sub>a</sub>d<sub>aan,</sub> V<sub>en</sub>k<sub>a</sub>t<sub>a</sub> S<sub>a</sub>i S<sub>urya</sub> S<sub>u</sub>b<sub>ramanyam</sub> D<sub>uvvur</sub>i<sub>,</sub> Devvrit Khatri, David Brandfonbrener, David Alvarez-Melis, Prajjwal Bhargava, Mihir Kale, and Samy Jelassi. Let’s (not) just <sub>p</sub>ut thin<sub>g</sub>s in context: Test-time trainin<sub>g</sub> for lon<sub>g</sub>-context LLMs. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, vo<sup>l</sup>ume 202<sup>6</sup>, pp. 112130–112153, 202<sup>6</sup>. URL https://proceedings.iclr.cc/paper\_files/pa per/2026/file/b619cd6dcc986856b8a8da2b08d89396-Paper-Conference.pdf.

F<sub>e</sub>d<sub>er</sub>i<sub>co</sub> B<sub>ar</sub>b<sub>ero,</sub> Al<sub>ex</sub> Vit<sub>v</sub>it<sub>s</sub>k<sub>y</sub>i<sub>,</sub> Ch<sub>r</sub>i<sub>s</sub>t<sub>os</sub> P<sub>er</sub>i<sub>vo</sub>l<sub>aropou</sub>l<sub>os,</sub> R<sub>azvan</sub> P<sub>ascanu, an</sub>d P<sub>e</sub>t<sub>ar</sub> V<sub>e</sub>ličk<sub>ov</sub>ić<sub>.</sub> R<sub>oun</sub>d and round we <sub>g</sub>o! What makes rotar<sub>y</sub> <sub>p</sub>ositional encodin<sub>g</sub>s useful? In Y. Yue<sub>,</sub> A. Gar<sub>g,</sub> N. Pen<sub>g,</sub> F. Sha<sub>,</sub> and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 92408–92438, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/e6d58fc68c0f3c36ae 6e0e64478a69c0-Paper-Conference.pdf.

bloc97. NTK-aware scaled RoPE allows LLaMA models to have extended (8k+) context size without any <sup>fi</sup>ne-tun<sup>i</sup>ng an<sup>d</sup> m<sup>i</sup>n<sup>i</sup>ma<sup>l</sup> perp<sup>l</sup>ex<sup>i</sup>ty <sup>d</sup>egra<sup>d</sup>at<sup>i</sup>on. Re<sup>ddi</sup>t post, r/Loca<sup>l</sup>LLaMA, 2023. URL https: //www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware\_scaled\_rope\_allows\_lla ma\_models\_to\_have/.

Lih<sub>u</sub> Ch<sub>en,</sub> G<sub>a</sub>ël V<sub>aroquaux, an</sub>d F<sub>a</sub>bi<sub>an</sub> M<sub>.</sub> S<sub>uc</sub>h<sub>ane</sub>k<sub>.</sub> Th<sub>e</sub> l<sub>oca</sub>lit<sub>y an</sub>d <sub>symme</sub>t<sub>ry o</sub>f <sub>pos</sub>iti<sub>ona</sub>l <sub>enco</sub>di<sub>ngs.</sub> I<sub>n</sub> Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Associationfor Computational Linguistics: EMNLP 2023, pp. 14313–14331, Singapore, December 2023a. Association for Computational Linguistics.

<sup>d</sup>o<sup>i</sup>: 10.18<sup>6</sup>53/v1/2023.<sup>fi</sup>n<sup>di</sup>ngs-emn<sup>l</sup>p.955. URL https://aclanthology.org/2023.findings-e mnlp.955/.

Shi Chen, Zhengjiang Lin, Yury Polyanskiy, and Philippe Rigollet. Critical attention scaling in long-context transformers. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 86454–86478, 2026. URL https://proceedi ngs.iclr.cc/paper\_files/paper/2026/file/8b08bbf8b420faa6eeb4020720582ec7-Paper -Conference.pdf.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models via <sub>p</sub>ositional inter<sub>p</sub>olation. arXiv <sub>p</sub>re<sub>p</sub>rint arXiv:2306.15595<sub>,</sub> 2023b. URL https://arxiv.org/abs/2306.15595.

Yiti<sub>ng</sub> Ch<sub>en an</sub>d J<sub>unc</sub>hi Y<sub>an.</sub> Wh<sub>a</sub>t <sub>ro</sub>t<sub>ary pos</sub>iti<sub>on em</sub>b<sub>e</sub>ddi<sub>ng can</sub> t<sub>e</sub>ll <sub>us:</sub> Id<sub>en</sub>tif<sub>y</sub>i<sub>ng query an</sub>d k<sub>ey we</sub>i<sub>g</sub>ht<sub>s</sub> corresponding to basic syntactic or high-level semantic information. In Advances in Neural Information Processing Systems, volume 37, pp. 54507–54528. Curran Associates, Inc., 2024. doi: 10.52202/079017-1 727. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/61f425da6e0 a201b8fe1454601abfba5-Paper-Conference.pdf.

Yuhan Chen<sub>,</sub> An<sub>g</sub> Lv<sub>,</sub> Jian Luan<sub>,</sub> Bin Wan<sub>g,</sub> and Wei Liu. HoPE: A novel <sub>p</sub>ositional encodin<sub>g</sub> without lon<sub>g</sub>-term d<sub>ecay</sub> f<sub>or en</sub>h<sub>ance</sub>d <sub>con</sub>t<sub>ex</sub>t <sub>awareness an</sub>d <sub>ex</sub>t<sub>rapo</sub>l<sub>a</sub>ti<sub>on.</sub> I<sub>n</sub> W<sub>anx</sub>i<sub>ang</sub> Ch<sub>e,</sub> J<sub>oyce</sub> N<sub>a</sub>b<sub>en</sub>d<sub>e,</sub> Ek<sub>a</sub>t<sub>er</sub>i<sub>na</sub> Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23044–23056, Vienna, Austria, July 2025. Association for Com<sub>p</sub>utational Lin<sub>g</sub>uistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-lon<sub>g</sub>.11 23. URL https://aclanthology.org/2025.acl-long.1123/.

Ti<sub>ng-</sub>R<sub>u</sub>i Chi<sub>ang an</sub>d D<sub>an</sub>i Y<sub>oga</sub>t<sub>ama.</sub> Th<sub>e ro</sub>t<sub>ary pos</sub>iti<sub>on em</sub>b<sub>e</sub>ddi<sub>ng may cause</sub> di<sub>mens</sub>i<sub>on</sub> i<sub>ne</sub>fi<sub>c</sub>i<sub>ency</sub> i<sub>n</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> h<sub>ea</sub>d<sub>s</sub> f<sub>or</sub> l<sub>ong-</sub>di<sub>s</sub>t<sub>ance re</sub>t<sub>r</sub>i<sub>eva</sub>l<sub>.</sub> I<sub>n</sub> W<sub>anx</sub>i<sub>ang</sub> Ch<sub>e,</sub> J<sub>oyce</sub> N<sub>a</sub>b<sub>en</sub>d<sub>e,</sub> Ek<sub>a</sub>t<sub>er</sub>i<sub>na</sub> Sh<sub>u</sub>t<sub>ova, an</sub>d Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 13552–13562<sub>,</sub> Vi<sub>e</sub>nn<sub>a,</sub> A<sub>us</sub>tri<sub>a,</sub> J<sub>u</sub>l<sub>y</sub> 2025<sub>.</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r C<sub>o</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l Lin<sub>gu</sub>i<sub>s</sub>ti<sub>cs.</sub> d<sub>o</sub>i<sub>:</sub> 10<sub>.</sub>18653/v1/ 2025.<sup>fi</sup>n<sup>di</sup>ngs-ac<sup>l</sup>.<sup>6</sup>97. URL https://aclanthology.org/2025.findings-acl.697/.

Yiran Din<sub>g,</sub> Li L<sub>y</sub>na Zhan<sub>g,</sub> Chen<sub>g</sub>ruidon<sub>g</sub> Zhan<sub>g,</sub> Yuan<sub>y</sub>uan Xu<sub>,</sub> Nin<sub>g</sub> Shan<sub>g,</sub> Jiahan<sub>g</sub> Xu<sub>,</sub> Fan Yan<sub>g,</sub> and Mao Yang. LongRoPE: Extending LLM context window beyond 2 million tokens. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 11091–11104. PMLR, 2024. URL https://proceedings.mlr.press/v235/ding24i.html.

Zican Don<sub>g,</sub> Jun<sub>y</sub>i Li<sub>,</sub> Xin Men<sub>,</sub> Wa<sub>y</sub>ne Xin Zhao<sub>,</sub> Bin<sub>g</sub>nin<sub>g</sub> Wan<sub>g,</sub> Zhen Tian<sub>,</sub> Wei<sub>p</sub>en<sub>g</sub> Chen<sub>,</sub> and Ji-R<sub>ong</sub> W<sub>en.</sub> E<sub>xp</sub>l<sub>or</sub>i<sub>ng con</sub>t<sub>ex</sub>t <sub>w</sub>i<sub>n</sub>d<sub>ow o</sub>f l<sub>arge</sub> l<sub>anguage mo</sub>d<sub>e</sub>l<sub>s v</sub>i<sub>a</sub> d<sub>ecompose</sub>d <sub>pos</sub>iti<sub>ona</sub>l <sub>vec</sub>t<sub>ors.</sub> I<sub>n</sub> A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 10320–10347. Curran Associates, Inc., 2024. doi: 10.52202/079017-0330. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/1403ab1a427050538ec59c7f570aec8b-Paper-Conference.pdf.

Yufen<sub>g</sub> Du<sub>,</sub> Min<sub>y</sub>an<sub>g</sub> Tian<sub>,</sub> Srikanth Ronanki<sub>,</sub> Subendhu Ron<sub>g</sub>ali<sub>,</sub> Sravan Boda<sub>p</sub>ati<sub>,</sub> Aram Galst<sub>y</sub>an<sub>,</sub> Azton W<sub>e</sub>ll<sub>s,</sub> R<sub>oy</sub> S<sub>c</sub>h<sub>war</sub>t<sub>z,</sub> Eli<sub>u</sub> A H<sub>uer</sub>t<sub>a, an</sub>d H<sub>ao</sub> P<sub>eng.</sub> C<sub>on</sub>t<sub>ex</sub>t l<sub>eng</sub>th <sub>a</sub>l<sub>one</sub> h<sub>ur</sub>t<sub>s</sub> ll<sub>m per</sub>f<sub>ormance</sub> d<sub>esp</sub>it<sub>e</sub> per<sup>f</sup>ect retrieva<sup>l</sup>, 2025. URL https://arxiv.org/abs/2510.05381.

Y<sub>u</sub>f<sub>eng</sub> D<sub>u,</sub> Philli<sub>p</sub> H<sub>arr</sub>i<sub>s,</sub> Mi<sub>nyang</sub> Ti<sub>an,</sub> Eli<sub>u</sub> A H<sub>uer</sub>t<sub>a,</sub> S<sub>r</sub>ik<sub>an</sub>th R<sub>onan</sub>ki<sub>,</sub> S<sub>u</sub>b<sub>en</sub>dh<sub>u</sub> R<sub>onga</sub>li<sub>,</sub> A<sub>ram</sub> G<sub>a</sub>l<sub>s</sub>t<sub>yan,</sub> <sub>an</sub>d H<sub>ao</sub> P<sub>eng.</sub> R<sub>o</sub>PE di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>es ne</sub>ith<sub>er pos</sub>iti<sub>ons nor</sub> t<sub>o</sub>k<sub>ens</sub> i<sub>n</sub> l<sub>ong con</sub>t<sub>ex</sub>t<sub>s, prova</sub>bl<sub>y. ar</sub>Xi<sub>v prepr</sub>i<sub>n</sub>t arXiv:2605.15514, 2026. URL https://arxiv.org/abs/2605.15514.

Abhi<sub>manyu</sub> D<sub>u</sub>b<sub>ey,</sub> Abhi<sub>nav</sub> J<sub>au</sub>h<sub>r</sub>i<sub>,</sub> Abhi<sub>nav</sub> P<sub>an</sub>d<sub>ey,</sub> Abhi<sub>s</sub>h<sub>e</sub>k K<sub>a</sub>di<sub>an,</sub> Ah<sub>ma</sub>d Al<sub>-</sub>D<sub>a</sub>hl<sub>e,</sub> Ai<sub>es</sub>h<sub>a</sub> L<sub>e</sub>t<sub>man,</sub> Akhil M<sub>a</sub>th<sub>ur,</sub> Al<sub>an</sub> S<sub>c</sub>h<sub>e</sub>lt<sub>en,</sub> A<sub>my</sub> Y<sub>ang,</sub> A<sub>nge</sub>l<sub>a</sub> F<sub>an,</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub> Th<sub>e</sub> Ll<sub>ama</sub> 3 h<sub>er</sub>d <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> <sub>ar</sub>Xi<sub>v</sub> <sub>prepr</sub>i<sub>n</sub>t arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783v1.

Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. URL https://arxiv. org/abs/2503.19786.

O<sub>mer</sub> G<sub>o</sub>ld<sub>man,</sub> Al<sub>on</sub> J<sub>acov</sub>i<sub>,</sub> A<sub>v</sub>i<sub>v</sub> Sl<sub>o</sub>b<sub>o</sub>dki<sub>n,</sub> A<sub>v</sub>i<sub>ya</sub> M<sub>a</sub>i<sub>mon,</sub> Id<sub>o</sub> D<sub>agan,</sub> <sub>an</sub>d R<sub>eu</sub>t T<sub>sar</sub>f<sub>a</sub>t<sub>y.</sub> I<sub>s</sub> it <sub>rea</sub>ll<sub>y</sub> l<sub>ong</sub> context if all you need is retrieval? towards genuinely dificult long context NLP. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 16576–16586, Miami, Florida, USA<sub>,</sub> 2024. Association for Com<sub>p</sub>utational Lin<sub>g</sub>uistics.

Zihan Gu<sub>,</sub> Ruo<sub>y</sub>u Chen<sub>,</sub> Han Zhan<sub>g,</sub> Hua Zhan<sub>g,</sub> and Yue Hu. Deconstructin<sub>g p</sub>ositional information: From attention logits to training biases. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=D0u0glT060.

Chi H<sub>an an</sub>d H<sub>eng</sub> Ji<sub>.</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sm</sub> b<sub>e</sub>hi<sub>n</sub>d LLM <sub>pos</sub>iti<sub>on genera</sub>li<sub>za</sub>ti<sub>on.</sub> I<sub>n</sub> W<sub>anx</sub>i<sub>ang</sub> Ch<sub>e,</sub> Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 19408–19424, Vienna<sub>,</sub> Austria<sub>,</sub> Jul<sub>y</sub> 2025. Association for Com<sub>p</sub>utational Lin<sub>g</sub>uistics. ISBN 979-8-89176-251-0. doi: 10.18<sup>6</sup>53/v1/2025.ac<sup>l</sup>-<sup>l</sup>ong.953. URL https://aclanthology.org/2025.acl-long.953/.

Adi H<sub>av</sub>i<sub>v,</sub> O<sub>r</sub>i R<sub>am,</sub> Ofi<sub>r</sub> P<sub>ress,</sub> P<sub>e</sub>t<sub>er</sub> I<sub>zsa</sub>k<sub>, an</sub>d O<sub>mer</sub> L<sub>evy.</sub> T<sub>rans</sub>f<sub>ormer</sub> l<sub>anguage mo</sub>d<sub>e</sub>l<sub>s w</sub>ith<sub>ou</sub>t <sub>pos</sub>iti<sub>ona</sub>l encodin<sub>g</sub>s still learn <sub>p</sub>ositional information. In Yoav Goldber<sub>g</sub>, Zornitsa Kozareva, and Yue Zhan<sub>g</sub> (eds.), Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 1382–1390, Abu Dhabi, United A<sub>ra</sub>b E<sub>m</sub>i<sub>ra</sub>t<sub>es,</sub> D<sub>ecem</sub>b<sub>er</sub> 2022<sub>.</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l Li<sub>ngu</sub>i<sub>s</sub>ti<sub>cs.</sub> d<sub>o</sub>i<sub>:</sub> 10<sub>.</sub>18653/<sub>v</sub>1/2022<sub>.</sub>fi<sub>n</sub>di ngs-emn<sup>l</sup>p.99. URL https://aclanthology.org/2022.findings-emnlp.99/.

Xiangyu Hong, Che Jiang, Biqing Qi, Fandong Meng, Mo Yu, Bowen Zhou, and Jie Zhou. On the token distance modeling ability of higher RoPE attention dimension. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 5877–5888, Miami, Florida, USA, 2024. Association for Computational Linguistics. <sup>d</sup>oi: 10.18653/v1/2024.<sup>f</sup>in<sup>d</sup>ings-emn<sup>l</sup>p.338. URL https://aclanthology.org/2024. findings-emnlp.338/.

Chen<sub>g</sub>-Pin<sub>g</sub> Hsieh<sub>,</sub> Simen<sub>g</sub> Sun<sub>,</sub> Samuel Kriman<sub>,</sub> Shantanu Achar<sub>y</sub>a<sub>,</sub> Dima Rekesh<sub>,</sub> Fei Jia<sub>,</sub> Yan<sub>g</sub> Zhan<sub>g,</sub> and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? In Conference on Language Modeling, 2024a. URL https://openreview.net/forum?id=kIoBbc76Sy.

Chen<sub>g</sub>-Yu Hsieh<sub>,</sub> Yun<sub>g</sub>-Sun<sub>g</sub> Chuan<sub>g,</sub> Chun-Lian<sub>g</sub> Li<sub>,</sub> Zifen<sub>g</sub> Wan<sub>g,</sub> Lon<sub>g</sub> T. Le<sub>,</sub> Abhishek Kumar<sub>,</sub> James Glass<sub>,</sub> Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, and Tomas Pfister. Found in the middle: Calibrating <sub>pos</sub>iti<sub>ona</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> bi<sub>as</sub> i<sub>mproves</sub> l<sub>ong con</sub>t<sub>ex</sub>t <sub>u</sub>tili<sub>za</sub>ti<sub>on.</sub> I<sub>n</sub> L<sub>un-</sub>W<sub>e</sub>i K<sub>u,</sub> A<sub>n</sub>d<sub>re</sub> M<sub>ar</sub>ti<sub>ns, an</sub>d Vi<sub>ve</sub>k Srikumar (eds.), Findings of the Associationfor Computational Linguistics: ACL 2024, pp. 14982–14995, B<sub>ang</sub>k<sub>o</sub>k<sub>,</sub> Th<sub>a</sub>il<sub>an</sub>d<sub>,</sub> A<sub>ugus</sub>t 2024b<sub>.</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l Li<sub>ngu</sub>i<sub>s</sub>ti<sub>cs.</sub> d<sub>o</sub>i<sub>:</sub> 10<sub>.</sub>18653/<sub>v</sub>1/2024<sub>.</sub>fi<sub>n</sub> <sup>di</sup>ngs-ac<sup>l</sup>.890. URL https://aclanthology.org/2024.findings-acl.890/.

Ermo Hua, Che Jiang, Xingtai Lv, Kaiyan Zhang, Youbang Sun, Yuchen Fan, Xuekai Zhu, Biqing Qi, Ning Di<sub>ng,</sub> <sub>an</sub>d B<sub>owen</sub> Zh<sub>ou.</sub> F<sub>our</sub>i<sub>er</sub> <sub>pos</sub>iti<sub>on</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ng:</sub> E<sub>n</sub>h<sub>anc</sub>i<sub>ng</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub>’<sub>s</sub> <sub>per</sub>i<sub>o</sub>di<sub>c</sub> <sub>ex</sub>t<sub>ens</sub>i<sub>on</sub> f<sub>or</sub> l<sub>eng</sub>th generalization. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 24932–24949. PMLR, 2025.

Selim Jerad, Anej Svete, Jiaoda Li, and Ryan Cotterell. Disentangling the expressivity of RoPE. In Conference on Language Modeling, 2026. URL https://arxiv.org/abs/2608.11909.

Mingyu Jin, Kai Mei, Wujiang Xu, Mingjie Sun, Ruixiang Tang, Mengnan Du, Zirui Liu, and Yongfeng Zhang. Massive values in self-attention modules are the key to contextual knowledge understanding. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 28063–28096. PMLR, 2025.

Amirhossein Kazemnejad, Inkit Padhi, Karthike an Natesan Ramamurth , Pa el Das, and Siva Redd . The impact of positional encoding on length generalization in transformers. In Advances in Neural Information Processing Systems, 2023.

Guolin Ke, Di He, and Tie-Yan Liu. Rethinking positional encoding in language pre-training. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=09-528y2 Fgf.

K<sup>i</sup>m<sup>i</sup> Team. K<sup>i</sup>m<sup>i</sup> K3: Open <sup>f</sup>ront<sup>i</sup>er <sup>i</sup>nte<sup>lli</sup>gence. arX<sup>i</sup>v prepr<sup>i</sup>nt arX<sup>i</sup>v:2<sup>6</sup>07.24<sup>6</sup>53, 202<sup>6</sup>. URL https: //arxiv.org/abs/2607.24653.

Nikit<sub>a</sub> Kit<sub>aev an</sub>d D<sub>an</sub> Kl<sub>e</sub>i<sub>n.</sub> C<sub>ons</sub>tit<sub>uency pars</sub>i<sub>ng w</sub>ith <sub>a se</sub>lf<sub>-a</sub>tt<sub>en</sub>ti<sub>ve enco</sub>d<sub>er.</sub> I<sub>n</sub> I<sub>ryna</sub> G<sub>urevyc</sub>h <sub>an</sub>d Yusuke Miyao (eds.), Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 2676–2686, Melbourne, Australia, July 2018. Association for Computational Linguistics. <sup>d</sup>oi: 10.18653/v1/P18-1249. URL https://aclanthology.org/P18-1249/.

Y<sub>ur</sub>i K<sub>ura</sub>t<sub>ov,</sub> A<sub>y</sub>d<sub>ar</sub> B<sub>u</sub>l<sub>a</sub>t<sub>ov,</sub> P<sub>e</sub>t<sub>r</sub> A<sub>no</sub>khi<sub>n,</sub> I<sub>van</sub> R<sub>o</sub>dki<sub>n,</sub> D<sub>m</sub>it<sub>ry</sub> S<sub>oro</sub>ki<sub>n,</sub> A<sub>r</sub>t<sub>yom</sub> S<sub>oro</sub>ki<sub>n, an</sub>d Mikh<sub>a</sub>il B<sub>ur</sub>t<sub>sev.</sub> BABILon<sub>g</sub>: Testin<sub>g</sub> the limits of LLMs with lon<sub>g</sub> context reasonin<sub>g</sub>-in-a-ha<sub>y</sub>stack. In A. Globerson<sub>,</sub> L. Macke<sub>y,</sub> D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 106519–106554. Curran Associates, Inc., 2024. doi: 10.52202/079017-3381. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/c0d62e70dbc659c c9bd44cbcf1cb652f-Paper-Datasets\_and\_Benchmarks\_Track.pdf.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. Set Transformer: A framework for attention-based permutation-invariant neural networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 3744–3753. PMLR, 2019. URL https://proceedings.mlr.press/v97/lee19d.html.

M<sub>os</sub>h L<sub>evy,</sub> Al<sub>on</sub> J<sub>aco</sub>b<sub>y, an</sub>d Y<sub>oav</sub> G<sub>o</sub>ldb<sub>erg.</sub> S<sub>ame</sub> t<sub>as</sub>k<sub>, more</sub> t<sub>o</sub>k<sub>ens:</sub> th<sub>e</sub> i<sub>mpac</sub>t <sub>o</sub>f i<sub>npu</sub>t l<sub>eng</sub>th <sub>on</sub> th<sub>e</sub> reasonin<sub>g p</sub>erformance of lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15339–15353, Bangkok, Thailand, August 2024. Association for Computational Linguistics. <sup>d</sup>oi: 10.18653/v1/2024.ac<sup>l</sup>-<sup>l</sup>ong.818. URL https://aclanthology.org/2024.acl-long.818/.

Haoran Li<sub>,</sub> Suchen<sub>g</sub> Ren<sub>,</sub> Alan Yuille<sub>,</sub> and Fen<sub>g</sub> Wan<sub>g</sub>. CoPE: Cli<sub>pp</sub>ed RoPE as a scalable free lunch for lon<sub>g</sub> context LLMs. arXiv preprint arXiv:2602.05258, 2026. URL https://arxiv.org/abs/2602.05258.

Zhan Ling, Kang Liu, Kai Yan, Yifan Yang, Weijian Lin, Ting-Han Fan, Lingfeng Shen, Zhengyin Du, and Ji<sub>ecao</sub> Ch<sub>en.</sub> L<sub>ong</sub>R<sub>eason:</sub> A <sub>syn</sub>th<sub>e</sub>ti<sub>c</sub> l<sub>ong-con</sub>t<sub>ex</sub>t <sub>reason</sub>i<sub>ng</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>v</sub>i<sub>a con</sub>t<sub>ex</sub>t <sub>expans</sub>i<sub>on. ar</sub>Xi<sub>v</sub> preprint arXiv:2501.15089, 2025. URL https://arxiv.org/abs/2501.15089.

Al<sub>exan</sub>d<sub>er</sub> H<sub>.</sub> Li<sub>u,</sub> K<sub>ar</sub>tik Kh<sub>an</sub>d<sub>e</sub>l<sub>wa</sub>l<sub>,</sub> S<sub>an</sub>d<sub>eep</sub> S<sub>u</sub>b<sub>raman</sub>i<sub>an,</sub> Vi<sub>c</sub>t<sub>or</sub> J<sub>ouau</sub>lt<sub>,</sub> Abhi<sub>nav</sub> R<sub>as</sub>t<sub>og</sub>i<sub>,</sub> Ad<sub>r</sub>i<sub>en</sub> S<sub>a</sub>dé<sub>,</sub> Al<sub>an</sub> J<sub>e</sub>f<sub>ares,</sub> Alb<sub>er</sub>t Ji<sub>ang,</sub> Al<sub>exan</sub>d<sub>re</sub> C<sub>a</sub>hill<sub>,</sub> Al<sub>exan</sub>d<sub>re</sub> G<sub>avau</sub>d<sub>an,</sub> Al<sub>exan</sub>d<sub>re</sub> S<sub>a</sub>bl<sub>ayro</sub>ll<sub>es,</sub> A<sub>m</sub>éli<sub>e</sub> Héli<sub>ou,</sub>

Amos You<sub>,</sub> And<sub>y</sub> Ehrenber<sub>g,</sub> And<sub>y</sub> Lo<sub>,</sub> Anton Eliseev<sub>,</sub> Antonia Calvi<sub>,</sub> Avinash Soori<sub>y</sub>arachchi<sub>,</sub> Ba<sub>p</sub>tiste Bout<sub>,</sub> Ba<sub>p</sub>tiste Rozière<sub>,</sub> Baudouin De Monicault<sub>,</sub> Clémence Lanfranchi<sub>,</sub> Corentin Barreau<sub>,</sub> C<sub>yp</sub>rien Courtot<sub>,</sub> D<sub>an</sub>i<sub>e</sub>l<sub>e</sub> G<sub>ra</sub>tt<sub>aro</sub>l<sub>a,</sub> D<sub>ar</sub>i<sub>us</sub> D<sub>a</sub>b<sub>er</sub>t<sub>,</sub> Di<sub>ego</sub> d<sub>e</sub> l<sub>as</sub> C<sub>asas,</sub> Elli<sub>o</sub>t Ch<sub>ane-</sub>S<sub>ane,</sub> F<sub>aru</sub>k Ah<sub>me</sub>d<sub>,</sub> G<sub>a</sub>b<sub>r</sub>i<sub>e</sub>ll<sub>e</sub> B<sub>erra</sub>d<sub>a,</sub> G<sub>a</sub>ët<sub>an</sub> E<sub>crepon</sub>t<sub>,</sub> G<sub>au</sub>thi<sub>er</sub> G<sub>u</sub>i<sub>ne</sub>t<sub>,</sub> G<sub>eorg</sub>ii N<sub>ov</sub>ik<sub>ov,</sub> G<sub>u</sub>ill<sub>aume</sub> K<sub>unsc</sub>h<sub>,</sub> G<sub>u</sub>ill<sub>aume</sub> L<sub>amp</sub>l<sub>e,</sub> G<sub>u</sub>ill<sub>aume</sub> Martin Gunshi Gu ta Jan Ludziejewski Jason Rute Joachim Studnia Jonas Amar José hine Delas J<sub>osse</sub>li<sub>n</sub> S<sub>omerv</sub>ill<sub>e</sub> R<sub>o</sub>b<sub>er</sub>t<sub>s,</sub> K<sub>armes</sub>h Y<sub>a</sub>d<sub>av,</sub> Kh<sub>ya</sub>thi Ch<sub>an</sub>d<sub>u,</sub> K<sub>us</sub>h J<sub>a</sub>i<sub>n,</sub> L<sub>aurence</sub> Ait<sub>c</sub>hi<sub>son,</sub> L<sub>auren</sub>t Fainsin<sub>,</sub> Léonard Blier<sub>,</sub> Lin<sub>g</sub>xiao Zhao<sub>,</sub> Louis Martin<sub>,</sub> Lucile Saulnier<sub>,</sub> Lu<sub>y</sub>u Gao<sub>,</sub> Maarten Bu<sub>y</sub>l<sub>,</sub> Mar<sub>g</sub>aret J<sub>enn</sub>i<sub>ngs,</sub> M<sub>ar</sub>i<sub>e</sub> P<sub>e</sub>ll<sub>a</sub>t<sub>,</sub> M<sub>ar</sub>k P<sub>r</sub>i<sub>ns,</sub> M<sub>a</sub>thi<sub>eu</sub> P<sub>o</sub>i<sub>r</sub>é<sub>e,</sub> M<sub>a</sub>thild<sub>e</sub> G<sub>u</sub>ill<sub>aum</sub>i<sub>n,</sub> M<sub>a</sub>tthi<sub>eu</sub> Di<sub>no</sub>t<sub>,</sub> M<sub>a</sub>tthi<sub>eu</sub> F<sub>u</sub>t<sub>era</sub>l<sub>,</sub> Maxime Darrin, Maximilian Augustin, Mia Chiquier, Michel Schimpf, Nathan Grinsztajn, Neha Gupta, Nikhil R<sub>ag</sub>h<sub>uraman,</sub> Oli<sub>v</sub>i<sub>er</sub> B<sub>ousque</sub>t<sub>,</sub> Oli<sub>v</sub>i<sub>er</sub> D<sub>uc</sub>h<sub>enne,</sub> P<sub>a</sub>t<sub>r</sub>i<sub>c</sub>i<sub>a</sub> W<sub>ang,</sub> P<sub>a</sub>t<sub>r</sub>i<sub>c</sub>k <sub>von</sub> Pl<sub>a</sub>t<sub>en,</sub> P<sub>au</sub>l J<sub>aco</sub>b<sub>,</sub> P<sub>au</sub>l W<sub>am</sub>b<sub>ergue,</sub> P<sub>au</sub>l<sub>a</sub> K<sub>ury</sub>l<sub>ow</sub>i<sub>cz,</sub> P<sub>avan</sub>k<sub>umar</sub> R<sub>e</sub>dd<sub>y</sub> M<sub>u</sub>ddi<sub>re</sub>dd<sub>y,</sub> Phil<sub>om</sub>è<sub>ne</sub> Ch<sub>agn</sub>i<sub>o</sub>t<sub>,</sub> Pi<sub>erre</sub> St<sub>oc</sub>k<sub>,</sub> P<sub>raves</sub>h Agrawal, Quentin Torroba, Romain Sauvestre, Roman Soletskyi, Rupert Menneer, Sagar Vaze, Samuel Barry, Sanchit Gandhi, Siddhant Waghjale, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aith<sub>a</sub>l<sub>,</sub> S<sub>zymon</sub> A<sub>n</sub>t<sub>on</sub>i<sub>a</sub>k<sub>,</sub> T<sub>even</sub> L<sub>e</sub> S<sub>cao,</sub> Thé<sub>o</sub> C<sub>ac</sub>h<sub>e</sub>t<sub>,</sub> Th<sub>eo</sub> Si<sub>mon</sub> S<sub>org,</sub> Thib<sub>au</sub>t L<sub>avr</sub>il<sub>,</sub> Thi<sub>z</sub>i<sub>r</sub>i N<sub>a</sub>it Saada<sub>,</sub> Thomas Chabal<sub>,</sub> Thomas Foubert<sub>,</sub> Thomas Robert<sub>,</sub> Thomas Wan<sub>g,</sub> Tim Lawson<sub>,</sub> Tom Bewle<sub>y,</sub> Tom Ed<sub>war</sub>d<sub>s,</sub> U<sub>mar</sub> J<sub>am</sub>il<sub>,</sub> U<sub>m</sub>b<sub>er</sub>t<sub>o</sub> T<sub>omas</sub>i<sub>n</sub>i<sub>,</sub> V<sub>a</sub>l<sub>er</sub>ii<sub>a</sub> N<sub>emyc</sub>h<sub>n</sub>ik<sub>ova,</sub> V<sub>an</sub> Ph<sub>ung,</sub> Vi<sub>ncen</sub>t M<sub>a</sub>l<sub>a</sub>diè<sub>re,</sub> Vi<sub>rg</sub>il<sub>e</sub> Ri<sub>c</sub>h<sub>ar</sub>d<sub>,</sub> W<sub>ass</sub>i<sub>m</sub> B<sub>ouaz</sub>i<sub>z,</sub> W<sub>en-</sub>Di<sub>ng</sub> Li<sub>,</sub> Willi<sub>am</sub> M<sub>ars</sub>h<sub>a</sub>ll<sub>,</sub> Xi<sub>ng</sub>h<sub>u</sub>i Li<sub>,</sub> Xi<sub>nyu</sub> Y<sub>ang,</sub> Y<sub>ass</sub>i<sub>ne</sub> El O<sub>ua</sub>hidi<sub>,</sub> Yihan Wan<sub>g,</sub> Yunhao Tan<sub>g,</sub> and Zaccharie Ramzi. Ministral 3. arXiv <sub>p</sub>re<sub>p</sub>rint arXiv:2601.08584<sub>,</sub> 2026. URL https://arxiv.org/abs/2601.08584.

F<sub>e</sub>il<sub>ong</sub> Li<sub>u.</sub> R<sub>o</sub>t<sub>ary pos</sub>iti<sub>ona</sub>l <sub>em</sub>b<sub>e</sub>ddi<sub>ngs as p</sub>h<sub>ase mo</sub>d<sub>u</sub>l<sub>a</sub>ti<sub>on:</sub> Th<sub>eore</sub>ti<sub>ca</sub>l b<sub>oun</sub>d<sub>s on</sub> th<sub>e</sub> R<sub>o</sub>PE b<sub>ase</sub> f<sub>or</sub> <sup>l</sup>ong-context trans<sup>f</sup>ormers. arXiv preprint arXiv:2602.10959, 2026. URL https://arxiv.org/abs/26 02.10959.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024a. doi: 10.1162/tacl\_a\_00638. URL https://aclantholo gy.org/2024.tacl-1.9/.

Xiaoran Liu, Hang Yan, Chenxin An, Xipeng Qiu, and Dahua Lin. Scaling laws of RoPE-based extrapolation. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 48160–48185, 2024b. URL https://proceedings.iclr .cc/paper\_files/paper/2024/file/d33b741600f100b31256c70a46f66ec9-Paper-Confere nce.pdf.

Yi Lu<sub>,</sub> Wanxu Zhao<sub>,</sub> Xin Zhou<sub>,</sub> Chenxin An<sub>,</sub> Chen<sub>g</sub>lon<sub>g</sub> Wan<sub>g,</sub> Shuo Li<sub>,</sub> Yumin<sub>g</sub> Yan<sub>g,</sub> Jun Zhao<sub>,</sub> Tao Ji<sub>,</sub> Tao Gui, Qi Zhang, and Xuanjing Huang. Efective length extrapolation via dimension-wise positional embeddings manipulation. In Conference on Language Modeling, 2025. URL https://openreview.net /forum?id=Tahpc3iAnO.

A<sub>ryan</sub> Mik<sub>ae</sub>ili<sub>,</sub> O<sub>r</sub> P<sub>a</sub>t<sub>as</sub>h<sub>n</sub>ik<sub>,</sub> A<sub>n</sub>d<sub>rea</sub> T<sub>ag</sub>li<sub>asacc</sub>hi<sub>,</sub> D<sub>an</sub>i<sub>e</sub>l C<sub>o</sub>h<sub>en-</sub>O<sub>r, an</sub>d Ali M<sub>a</sub>hd<sub>av</sub>i<sub>-</sub>A<sub>m</sub>i<sub>r</sub>i<sub>.</sub> U<sub>n</sub>t<sub>w</sub>i<sub>s</sub>ti<sub>ng</sub> RoPE: Fre<sub>q</sub>uenc<sub>y</sub> control for shared attention in DiTs. arXiv <sub>p</sub>re<sub>p</sub>rint arXiv:2602.05013<sub>,</sub> 2026. URL https://arxiv.org/abs/2602.05013.

Ali Modarressi<sub>,</sub> Hanieh Deilamsaleh<sub>y,</sub> Franck Dernoncourt<sub>,</sub> Trun<sub>g</sub> Bui<sub>,</sub> R<sub>y</sub>an A. Rossi<sub>,</sub> Seun<sub>g</sub>h<sub>y</sub>un Yoon<sub>,</sub> and Hi<sub>nr</sub>i<sub>c</sub>h S<sub>c</sub>hüt<sub>ze.</sub> N<sub>o</sub>LiM<sub>a:</sub> L<sub>ong-con</sub>t<sub>ex</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> b<sub>eyon</sub>d lit<sub>era</sub>l <sub>ma</sub>t<sub>c</sub>hi<sub>ng.</sub> I<sub>n</sub> A<sub>ar</sub>ti Si<sub>ng</sub>h<sub>,</sub> M<sub>aryam</sub> Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of

Machine Learning Research, pp. 44554–44570. PMLR, 13–19 Jul 2025. URL https://proceedings.ml r.press/v267/modarressi25a.html.

H. L. Montgomery and R. C. Vaughan. Hilbert’s inequality. Journal of the London Mathematical Society, s2-8 (1):73–82, May 1974. ISSN 0024-6107. <sup>d</sup>oi: 10.1112/j<sup>l</sup>ms/s2-8.1.73. URL https://doi.org/10.111 2/jlms/s2-8.1.73.

Y<sub>u</sub>i Ok<sub>a,</sub> K<sub>en</sub>t<sub>aro</sub> H<sub>ana</sub>f<sub>usa,</sub> T<sub>a</sub>k<sub>u</sub> H<sub>asegawa,</sub> K<sub>yosu</sub>k<sub>e</sub> Ni<sub>s</sub>hid<sub>a,</sub> <sub>an</sub>d K<sub>un</sub>ik<sub>o</sub> S<sub>a</sub>it<sub>o.</sub> P<sub>ro</sub>bi<sub>ng</sub> <sub>ro</sub>t<sub>ary</sub> <sub>pos</sub>iti<sub>on</sub> embeddin<sub>g</sub>s throu<sub>g</sub>h fre<sub>q</sub>uenc<sub>y</sub> entro<sub>py</sub>. In C. Vondrick<sub>,</sub> B. Hariharan<sub>,</sub> C. Rafel<sub>,</sub> L. Pinto<sub>,</sub> D. Yan<sub>g,</sub> and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 6139–6167, 2026a. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/0aee38a6fe9fffc8b6 58cfb1d872c1d5-Paper-Conference.pdf.

Yui Oka<sub>,</sub> Itsumi Saito<sub>,</sub> K<sub>y</sub>osuke Nishida<sub>,</sub> and Kuniko Saito. Fre<sub>q</sub>uenc<sub>y</sub> bands in RoPE: Base fre<sub>q</sub>uenc<sub>y</sub> and <sub>con</sub>t<sub>ex</sub>t l<sub>eng</sub>th <sub>s</sub>h<sub>ape</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erpo</sub>l<sub>a</sub>ti<sub>on–ex</sub>t<sub>rapo</sub>l<sub>a</sub>ti<sub>on</sub> t<sub>ra</sub>d<sub>e-o</sub>f<sub>.</sub> I<sub>n</sub> C<sub>.</sub> V<sub>on</sub>d<sub>r</sub>i<sub>c</sub>k<sub>,</sub> B<sub>.</sub> H<sub>ar</sub>ih<sub>aran,</sub> C<sub>.</sub> R<sub>a</sub>f<sub>e</sub>l<sub>,</sub> L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 94947–94975, 2026<sup>b</sup>. URL https://proceedings.iclr.cc/paper\_files/paper/2026/fil e/993dbaec0418a6449090f1debbcb8844-Paper-Conference.pdf.

Bowen Peng, Jefrey Quesnelle, Honglu Fan, and Enrico Shippole. YaRN: Eficient context window extension of lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models. In B. Kim, Y. Yue, S. Chaudhuri, K. Fra<sub>g</sub>kiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 31932–31951, 2024. URL https: //proceedings.iclr.cc/paper\_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545c f040-Paper-Conference.pdf.

Ofi<sub>r</sub> P<sub>ress,</sub> N<sub>oa</sub>h S<sub>m</sub>ith<sub>, an</sub>d Mik<sub>e</sub> L<sub>ew</sub>i<sub>s.</sub> T<sub>ra</sub>i<sub>n s</sub>h<sub>or</sub>t<sub>,</sub> t<sub>es</sub>t l<sub>ong:</sub> Att<sub>en</sub>ti<sub>on w</sub>ith li<sub>near</sub> bi<sub>ases ena</sub>bl<sub>es</sub> i<sub>npu</sub>t length extrapolation. In International Conference on Learning Representations, 2022a. URL https: //openreview.net/forum?id=R8sQPpGCv0.

Ofi<sub>r</sub> P<sub>ress,</sub> N<sub>oa</sub>h A<sub>.</sub> S<sub>m</sub>ith<sub>, an</sub>d Mik<sub>e</sub> L<sub>ew</sub>i<sub>s.</sub> T<sub>ra</sub>i<sub>n s</sub>h<sub>or</sub>t<sub>,</sub> t<sub>es</sub>t l<sub>ong:</sub> Att<sub>en</sub>ti<sub>on w</sub>ith li<sub>near</sub> bi<sub>ases ena</sub>bl<sub>es</sub> i<sub>npu</sub>t length extrapolation. In International Conference on Learning Representations, 2022b.

Jianing Qi, Jiawei Liu, Hao Tang, and Zhigang Zhu. Beyond semantics: Rediscovering spatial awareness in vision-language models. In Conference on Language Modeling, 2026. URL https://arxiv.org/abs/25 03.17349.

Ye Qiao, Haocheng Xu, Xiaofan Zhang, and Sitao Huang. Rethinking RoPE scaling in quantized LLM: Theory, outlier<sub>,</sub> and channel-band anal<sub>y</sub>sis with wei<sub>g</sub>ht rescalin<sub>g</sub>. arXiv <sub>p</sub>re<sub>p</sub>rint arXiv:2510.00028<sub>,</sub> 2025. URL https://arxiv.org/abs/2510.00028.

R. Salem and A. Zygmund. On lacunary trigonometric series. Proceedings of the National Academy of Sciences of the United States of America, 33(11):333–338, 1947. doi: 10.1073/pnas.33.11.333. URL https://doi.org/10.1073/pnas.33.11.333.

P<sub>e</sub>t<sub>er</sub> Sh<sub>aw,</sub> J<sub>a</sub>k<sub>o</sub>b U<sub>sz</sub>k<sub>ore</sub>it<sub>, an</sub>d A<sub>s</sub>hi<sub>s</sub>h V<sub>aswan</sub>i<sub>.</sub> S<sub>e</sub>lf<sub>-a</sub>tt<sub>en</sub>ti<sub>on w</sub>ith <sub>re</sub>l<sub>a</sub>ti<sub>ve pos</sub>iti<sub>on represen</sub>t<sub>a</sub>ti<sub>ons.</sub> I<sub>n</sub> Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pp. 464–468, New Orleans, Louisiana, 2018. Association for Com<sub>p</sub>utational Lin<sub>g</sub>uistics.

Runlin Shi, Bojian Yin, and Guoqi Li. Modern transformers are implicit hybrids: From functional diferentiation to principled hybrid architecture design. arXiv preprint arXiv:2609.02986, 2026. URL https://arxiv. org/abs/2609.02986.

Jianlin Su<sub>,</sub> Murtadha Ahmed<sub>,</sub> Yu Lu<sub>,</sub> Shen fen Pan<sub>,</sub> Wen Bo<sub>,</sub> and Yunfen Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, February 2024. ISSN 0925- 2312. <sup>d</sup>oi: 10.1016/j.neucom.2023.127063. URL https://doi.org/10.1016/j.neucom.2023.12 7063.

Bowen Sun, Yujun Cai, Ming-Hsuan Yang, Hang Wu, and Yiwei Wang. PAS: A training-free stabilizer for temporal encoding in video LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14471–14480, 2026. URL https://openaccess.thecvf.com/content/CV PR2026/html/Sun\_PAS\_A\_Training-Free\_Stabilizer\_for\_Temporal\_Encoding\_in\_Video\_ LLMs\_CVPR\_2026\_paper.html.

Rongyuan Tan, Jue Zhang, Zhuozhao Li, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. Contrastive attribution in the wild: An interpretability analysis of LLM failures on realistic benchmarks. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2026. URL https://arxiv.org/abs/2604.17761. To appear.

Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Danning Ke, Shikuan Hong, Yiwu Yao, and Gongyi Wang. RazorAttention: Eficient KV cache compression through retrieval heads. In International Conference on Learning Representations, 2025.

F<sub>e</sub>li<sub>pe</sub> U<sub>rru</sub>ti<sub>a,</sub> J<sub>orge</sub> S<sub>a</sub>l<sub>as,</sub> Al<sub>exan</sub>d<sub>er</sub> K<sub>ozac</sub>hi<sub>ns</sub>ki<sub>y,</sub> C<sub>r</sub>i<sub>s</sub>ti<sub>an</sub> B<sub>uc</sub> C<sub>a</sub>ld<sub>eron,</sub> H<sub>ec</sub>t<sub>or</sub> P<sub>as</sub>t<sub>en, an</sub>d C<sub>r</sub>i<sub>s</sub>t<sub>o</sub>b<sub>a</sub>l Rojas. Decoupling positional and symbolic attention behavior in transformers. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, vo<sup>l</sup>ume 202<sup>6</sup>, pp. <sup>6</sup>4123–<sup>6</sup>4153, 202<sup>6</sup>. URL https://proceedings.iclr.cc/paper\_files/pape r/2026/file/68f1a6933797528dd1ec8d15c6f58f62-Paper-Conference.pdf.

A<sub>s</sub>hi<sub>s</sub>h V<sub>aswan</sub>i<sub>,</sub> N<sub>oam</sub> Sh<sub>azeer,</sub> Niki P<sub>armar,</sub> J<sub>a</sub>k<sub>o</sub>b U<sub>sz</sub>k<sub>ore</sub>it<sub>,</sub> Lli<sub>on</sub> J<sub>ones,</sub> Aid<sub>an</sub> N<sub>.</sub> G<sub>omez,</sub> Ł<sub>u</sub>k<sub>asz</sub> K<sub>a</sub>i<sub>ser,</sub> and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pp. 5998–6008<sub>,</sub> 2017.

Ki<sub>ran</sub> V<sub>o</sub>d<sub>ra</sub>h<sub>a</sub>lli<sub>,</sub> S<sub>an</sub>ti<sub>ago</sub> O<sub>n</sub>t<sub>anon,</sub> Nil<sub>es</sub>h T<sub>r</sub>i<sub>puranen</sub>i<sub>,</sub> K<sub>e</sub>l<sub>v</sub>i<sub>n</sub> X<sub>u,</sub> S<sub>an</sub>il J<sub>a</sub>i<sub>n,</sub> R<sub>a</sub>k<sub>es</sub>h Shi<sub>vanna,</sub> J<sub>e</sub>f<sub>rey</sub> H<sub>u</sub>i<sub>,</sub> Nishanth Dikkala, Mehran Kazemi, Bahare Fatemi, Rohan Anil, Ethan Dyer, Siamak Shakeri, Roopali Vij, Harsh Mehta, Vinay Ramasesh, Quoc Le, Ed Chi, Yifeng Lu, Orhan Firat, Angeliki Lazaridou, Jean-Baptiste L<sub>esp</sub>i<sub>au,</sub> Nith<sub>ya</sub> Att<sub>a</sub>l<sub>ur</sub>i<sub>, an</sub>d K<sub>a</sub>t<sub>e</sub> Ol<sub>szews</sub>k<sub>a.</sub> Mi<sub>c</sub>h<sub>e</sub>l<sub>ange</sub>l<sub>o:</sub> L<sub>ong con</sub>t<sub>ex</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> b<sub>eyon</sub>d h<sub>ays</sub>t<sub>ac</sub>k<sub>s</sub> via latent structure <sub>q</sub>ueries<sub>,</sub> 2024.

Chon<sub>g</sub>hua Wan<sub>g,</sub> Haodon<sub>g</sub> Duan<sub>,</sub> Son<sub>gy</sub>an<sub>g</sub> Zhan<sub>g,</sub> Dahua Lin<sub>,</sub> and Kai Chen. Ada-LEval: Evaluatin<sub>g</sub> l<sub>ong-con</sub>t<sub>ex</sub>t LLM<sub>s w</sub>ith l<sub>eng</sub>th<sub>-a</sub>d<sub>ap</sub>t<sub>a</sub>bl<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub> I<sub>n</sub> K<sub>ev</sub>i<sub>n</sub> D<sub>u</sub>h<sub>,</sub> H<sub>e</sub>l<sub>ena</sub> G<sub>omez, an</sub>d St<sub>even</sub> B<sub>e</sub>th<sub>ar</sub>d (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3712–3724, Mexico City, Mexico, June 2024a. Association for Com<sub>p</sub>utational Lin<sub>g</sub>uistics. doi: 10.18653/v1/2024.naacl-lon<sub>g</sub>.205. URL https://aclanthology.org/2024.naacl-long.205/.

Shaowen Wan <sub>,</sub> Yuke Zhen <sub>,</sub> Tanshen Zhu<sub>,</sub> Shuan Chen<sub>,</sub> Shaofan Liu<sub>,</sub> Suncon Zhen <sub>,</sub> and Jian Li. AdaRoPE: Not all attention heads should rotate and scale equally. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=xNuMtLaxo5.

Su<sub>y</sub>uchen Wan<sub>g,</sub> Ivan Kob<sub>y</sub>zev<sub>,</sub> Pen<sub>g</sub> Lu<sub>,</sub> Mehdi Reza<sub>g</sub>holizadeh<sub>,</sub> and Ban<sub>g</sub> Liu. Resonance RoPE: Im<sub>p</sub>rovin<sub>g</sub> <sub>con</sub>t<sub>ex</sub>t l<sub>eng</sub>th <sub>genera</sub>li<sub>za</sub>ti<sub>on</sub> <sub>o</sub>f l<sub>arge</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> I<sub>n</sub> L<sub>un-</sub>W<sub>e</sub>i K<sub>u,</sub> A<sub>n</sub>d<sub>re</sub> M<sub>ar</sub>ti<sub>ns,</sub> <sub>an</sub>d Vi<sub>ve</sub>k S<sub>r</sub>ik<sub>umar</sub> (eds.), Findings of the Associationfor Computational Linguistics: ACL 2024, pp. 586–598, Bangkok, Thailand,

A<sub>ugus</sub>t 2024b<sub>.</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r C<sub>o</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l Lin<sub>gu</sub>i<sub>s</sub>ti<sub>cs.</sub> d<sub>o</sub>i<sub>:</sub> 10<sub>.</sub>18653/v1/2024<sub>.</sub>findin<sub>gs</sub>-<sub>ac</sub>l<sub>.</sub>32<sub>.</sub> URL https://aclanthology.org/2024.findings-acl.32/.

Davis Wertheimer<sub>,</sub> Aozhon<sub>g</sub> Zhan<sub>g,</sub> Derrick Liu<sub>,</sub> Pen<sub>g</sub>han<sub>g</sub> Yin<sub>,</sub> and Nai<sub>g</sub>an<sub>g</sub> Wan<sub>g</sub>. Fra<sub>y</sub>ed RoPE and lon<sub>g</sub> in<sub>p</sub>uts: A <sub>g</sub>eometric <sub>p</sub>ers<sub>p</sub>ective. In C. Vondrick<sub>,</sub> B. Hariharan<sub>,</sub> C. Rafel<sub>,</sub> L. Pinto<sub>,</sub> D. Yan<sub>g,</sub> and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 156004–156035, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/fc684a13a24ac4735b 1726659ce01049-Paper-Conference.pdf.

Wenhao Wu<sub>,</sub> Yizhon<sub>g</sub> Wan<sub>g,</sub> Guan<sub>g</sub>xuan Xiao<sub>,</sub> Hao Pen<sub>g,</sub> and Yao Fu. Retrieval head mechanisticall<sub>y</sub> ex<sub>p</sub>lains long-context factuality. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 62143–62156, 2025a. URL https://proceedings.iclr .cc/paper\_files/paper/2025/file/9b77f07301b1ef1fe810aae96c12cb7b-Paper-Confere nce.pdf.

Xiaodong Wu, Minhao Wang, Yichen Liu, Xiaoming Shi, He Yan, Xiangju Lu, Junmin Zhu, and Wei Zhang. LIFB<sub>enc</sub>h<sub>:</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>ng</sub> th<sub>e</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng per</sub>f<sub>ormance an</sub>d <sub>s</sub>t<sub>a</sub>bilit<sub>y o</sub>f l<sub>arge</sub> l<sub>anguage mo</sub>d<sub>e</sub>l<sub>s</sub> i<sub>n</sub> l<sub>ong-con</sub>t<sub>ex</sub>t <sub>scenar</sub>i<sub>os.</sub> I<sub>n</sub> W<sub>anx</sub>i<sub>ang</sub> Ch<sub>e,</sub> J<sub>oyce</sub> N<sub>a</sub>b<sub>en</sub>d<sub>e,</sub> Ek<sub>a</sub>t<sub>er</sub>i<sub>na</sub> Sh<sub>u</sub>t<sub>ova, an</sub>d M<sub>o</sub>h<sub>amma</sub>d T<sub>a</sub>h<sub>er</sub> Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16445–16468, Vienna, Austria, July 2025b. Association for Computational L<sup>i</sup>ngu<sup>i</sup>st<sup>i</sup>cs. ISBN 979-8-8917<sup>6</sup>-251-0. <sup>d</sup>o<sup>i</sup>: 10.18<sup>6</sup>53/v1/2025.ac<sup>l</sup>-<sup>l</sup>ong.803. URL https://aclantho logy.org/2025.acl-long.803/.

Xin<sub>y</sub>i Wu<sub>,</sub> Si<sub>y</sub>uan Liu<sub>,</sub> and Ali Jadbabaie. How data sha<sub>p</sub>es RoPE fre<sub>q</sub>uenc<sub>y</sub> usa<sub>g</sub>e: From <sub>p</sub>ositional scale matching to length generalization. In ICML 2026 Workshop on Foundations of Deep Generative Models: Understanding Memorization, Generalization, and Reasoning, 2026. URL https://arxiv.org/abs/26 07.07678. Non-arc<sup>h</sup>iva<sup>l</sup> wor<sup>k</sup>s<sup>h</sup>op paper.

Guan<sub>g</sub>xuan Xiao<sub>,</sub> Yuandon<sub>g</sub> Tian<sub>,</sub> Beidi Chen<sub>,</sub> Son<sub>g</sub> Han<sub>,</sub> and Mike Lewis. Eficient streamin<sub>g</sub> lan<sub>g</sub>ua<sub>g</sub>e <sub>mo</sub>d<sub>e</sub>l<sub>s w</sub>ith <sub>a</sub>tt<sub>en</sub>ti<sub>on s</sub>i<sub>n</sub>k<sub>s.</sub> I<sub>n</sub> B<sub>.</sub> Ki<sub>m,</sub> Y<sub>.</sub> Y<sub>ue,</sub> S<sub>.</sub> Ch<sub>au</sub>dh<sub>ur</sub>i<sub>,</sub> K<sub>.</sub> F<sub>rag</sub>ki<sub>a</sub>d<sub>a</sub>ki<sub>,</sub> M<sub>.</sub> Kh<sub>an, an</sub>d Y<sub>.</sub> S<sub>un</sub> (eds.), International Conference on Learning Representations, volume 2024, pp. 21875–21895, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/5e5fd18f863cbe6d8ae392 a93fd271c9-Paper-Conference.pdf.

Guan<sub>g</sub>xuan Xiao<sub>,</sub> Jiamin<sub>g</sub> Tan<sub>g,</sub> Jin<sub>g</sub>wei Zuo<sub>,</sub> Junxian Guo<sub>,</sub> Shan<sub>g</sub> Yan<sub>g,</sub> Haotian Tan<sub>g,</sub> Yao Fu<sub>,</sub> and Son<sub>g</sub> Han. DuoAttention: Eficient long-context LLM inference with retrieval and streaming heads. In International Conference on Learning Representations, 2025.

Jin<sub>g</sub> Xion<sub>g,</sub> Li<sub>y</sub>an<sub>g</sub> Fan<sub>,</sub> Hui Shen<sub>,</sub> Zunhai Su<sub>,</sub> Min Yan<sub>g,</sub> Lin<sub>gp</sub>en<sub>g</sub> Kon<sub>g,</sub> and N<sub>g</sub>ai Won<sub>g</sub>. DoPE: Denoisin<sub>g</sub> rotary position em<sup>b</sup>e<sup>dd</sup>ing. arXiv preprint arXiv:2511.09146, 2025. URL https://arxiv.org/abs/25 11.09146.

Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Run<sub>g</sub>ta<sub>,</sub> Karthik Abinav Sankararaman<sub>,</sub> Barlas O<sub>g</sub>uz<sub>,</sub> Madian Khabsa<sub>,</sub> Han Fan<sub>g,</sub> Yashar Mehdad<sub>,</sub> Sharan Naran<sub>g,</sub> Kshitiz Malik<sub>,</sub> An<sub>g</sub>ela Fan<sub>,</sub> Shruti Bhosale<sub>,</sub> Ser<sub>g</sub>e<sub>y</sub> Edunov<sub>,</sub> Mike Lewis<sub>,</sub> Sinon<sub>g</sub> Wan<sub>g,</sub> and Hao Ma. Efective long-context scaling of foundation models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4643–4663, Mexico City, Mexico, 2024. Association for Computational Linguistics.

Mingyu Xu, Xin Men, Bingning Wang, Qingyu Zhang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. Base of RoPE bounds context len<sub>g</sub>th. In A. Globerson<sub>,</sub> L. Macke<sub>y,</sub> D. Bel<sub>g</sub>rave<sub>,</sub> A. Fan<sub>,</sub> U. Pa<sub>q</sub>uet<sub>,</sub> J. Tomczak<sub>,</sub> and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 87386–87410. Curran Assoc<sup>i</sup>ates, Inc., 2024. <sup>d</sup>o<sup>i</sup>: 10.52202/079017-2773. URL https://proceedings.neurips. cc/paper\_files/paper/2024/file/9f12dd32d552f3ad9eaa0e9dfec291be-Paper-Confere nce.pdf.

An Yan<sub>g,</sub> Baoson<sub>g</sub> Yan<sub>g,</sub> Beichen Zhan<sub>g,</sub> Bin<sub>y</sub>uan Hui<sub>,</sub> Bo Zhen<sub>g,</sub> Bowen Yu<sub>,</sub> Chen<sub>gy</sub>uan Li<sub>,</sub> Da<sub>y</sub>ihen<sub>g</sub> Liu<sub>,</sub> Fei Huan<sub>g,</sub> Haoran Wei<sub>,</sub> Huan Lin<sub>,</sub> Jian Yan<sub>g,</sub> Jianhon<sub>g</sub> Tu<sub>,</sub> Jianwei Zhan<sub>g,</sub> Jianxin Yan<sub>g,</sub> Jiaxi Yan<sub>g,</sub> Jin<sub>g</sub>ren Zhou<sub>,</sub> Jun<sub>y</sub>an<sub>g</sub> Lin<sub>,</sub> Kai Dan<sub>g,</sub> Kemin<sub>g</sub> Lu<sub>,</sub> Ke<sub>q</sub>in Bao<sub>,</sub> Kexin Yan<sub>g,</sub> Le Yu<sub>,</sub> Mei Li<sub>,</sub> Min<sub>g</sub>fen<sub>g</sub> Xue<sub>,</sub> Pei Zhan<sub>g,</sub> Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024. URL https://arxiv.org/abs/2412.15115v1.

An Yan<sub>g,</sub> Anfen<sub>g</sub> Li<sub>,</sub> Baoson<sub>g</sub> Yan<sub>g,</sub> Beichen Zhan<sub>g,</sub> Bin<sub>y</sub>uan Hui<sub>,</sub> Bo Zhen<sub>g,</sub> Bowen Yu<sub>,</sub> Chan<sub>g</sub> Gao<sub>,</sub> Chen<sub>g</sub>en Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin<sub>,</sub> Jialon<sub>g</sub> Tan<sub>g,</sub> Jian Yan<sub>g,</sub> Jianhon<sub>g</sub> Tu<sub>,</sub> Jianwei Zhan<sub>g,</sub> Jianxin Yan<sub>g,</sub> Jiaxi Yan<sub>g,</sub> Jin<sub>g</sub> Zhou<sub>,</sub> Jin<sub>g</sub>ren Zhou<sub>,</sub> Jun<sub>y</sub>an<sub>g</sub> Lin<sub>,</sub> Kai Dan<sub>g,</sub> Ke<sub>q</sub>in Bao<sub>,</sub> Kexin Yan<sub>g,</sub> Le Yu<sub>,</sub> Lian<sub>g</sub>hao Den<sub>g,</sub> Mei Li<sub>,</sub> Min<sub>g</sub>fen<sub>g</sub> Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li<sub>,</sub> Tian<sub>y</sub>i Tan<sub>g,</sub> Wenbiao Yin<sub>,</sub> Xin<sub>g</sub>zhan<sub>g</sub> Ren<sub>,</sub> Xin<sub>y</sub>u Wan<sub>g,</sub> Xin<sub>y</sub>u Zhan<sub>g,</sub> Xuanchen<sub>g</sub> Ren<sub>,</sub> Yan<sub>g</sub> Fan<sub>,</sub> Yan<sub>g</sub> Su<sub>,</sub> Yichan<sub>g</sub> Zhan<sub>g,</sub> Yin<sub>g</sub>er Zhan<sub>g,</sub> Yu Wan<sub>,</sub> Yu<sub>q</sub>ion<sub>g</sub> Liu<sub>,</sub> Zekun Wan<sub>g,</sub> Ze<sub>y</sub>u Cui<sub>,</sub> Zhenru Zhan<sub>g,</sub> Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a. URL https://arxiv.org/abs/2505.09388.

Bowen Yan<sub>g,</sub> Bharat Venkitesh<sub>,</sub> Dwaraknath Gnaneshwar Talu<sub>p</sub>uru<sub>,</sub> Han<sub>gy</sub>u Lin<sub>,</sub> David Cairuz<sub>,</sub> Phil Blunsom<sub>,</sub> and Ac<sub>y</sub>r Locatelli. Ro<sub>p</sub>e to No<sub>p</sub>e and back a<sub>g</sub>ain: A new h<sub>y</sub>brid attention strate<sub>gy</sub>. In D. Bel<sub>g</sub>rave<sub>,</sub> C. Zhan<sub>g,</sub> H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 64133–64157. Curran Associates, Inc., 2025b. doi: 10.52202/085713-2150. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/5c9ab393551b7a39b4c 02d88fe5e7e69-Paper-Conference.pdf.

Howard Yen<sub>,</sub> Tian<sub>y</sub>u Gao<sub>,</sub> Minmin Hou<sub>,</sub> Ke Din<sub>g,</sub> Daniel Fleischer<sub>,</sub> Peter Izsak<sub>,</sub> Moshe Wasserblat<sub>,</sub> and Dan<sub>q</sub>i Chen. HELMET: How to evaluate lon<sub>g</sub>-context lan<sub>g</sub>ua<sub>g</sub>e models efectivel<sub>y</sub> and thorou<sub>g</sub>hl<sub>y</sub>. In Y. Yue<sub>,</sub> A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 98914–98965, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025 /file/f5332c8273d02729730a9c24dec2135e-Paper-Conference.pdf.

Justin Youn<sub>g</sub>. Efective harnesses for lon<sub>g</sub>-runnin<sub>g</sub> a<sub>g</sub>ents. Anthro<sub>p</sub>ic En<sub>g</sub>ineerin<sub>g,</sub> November 2025. URL https://www.anthropic.com/engineering/effective-harnesses-for-long-running-a gents.

Zhenyu Zhang, Runjin Chen, Shiwei Liu, Zhewei Yao, Olatunji Ruwase, Beidi Chen, Xiaoxia Wu, and Zh<sub>angyang</sub> W<sub>ang.</sub> F<sub>oun</sub>d i<sub>n</sub> th<sub>e</sub> <sub>m</sub>iddl<sub>e:</sub> H<sub>ow</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>use</sub> l<sub>ong</sub> <sub>con</sub>t<sub>ex</sub>t<sub>s</sub> b<sub>e</sub>tt<sub>er</sub> <sub>v</sub>i<sub>a</sub> <sub>p</sub>l<sub>ug-an</sub>d<sub>-p</sub>l<sub>ay</sub> <sub>p</sub>ositional encodin<sub>g</sub>. In A. Globerson<sub>,</sub> L. Macke<sub>y,</sub> D. Bel<sub>g</sub>rave<sub>,</sub> A. Fan<sub>,</sub> U. Pa<sub>q</sub>uet<sub>,</sub> J. Tomczak<sub>,</sub> and C. Zhan<sub>g</sub> (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 60755–60775. Curran Associates, Inc., 2024. <sup>d</sup>o<sup>i</sup>: 10.52202/079017-1943. URL https://proceedings.neurips.cc/paper\_files /paper/2024/file/6ffdbbe354893979367f93e2121e37dd-Paper-Conference.pdf.

Zhison<sub>g</sub> Zhan<sub>g,</sub> Yan Wan<sub>g,</sub> Xintin<sub>g</sub> Huan<sub>g,</sub> Tian<sub>q</sub>in<sub>g</sub> Fan<sub>g,</sub> Hon<sub>g</sub>min<sub>g</sub> Zhan<sub>g,</sub> Chenlon<sub>g</sub> Den<sub>g,</sub> Shuai<sub>y</sub>i Li<sub>, an</sub>d D<sub>ong</sub> Y<sub>u.</sub> Att<sub>en</sub>ti<sub>on en</sub>t<sub>ropy</sub> i<sub>s a</sub> k<sub>ey</sub> f<sub>ac</sub>t<sub>or:</sub> A<sub>n ana</sub>l<sub>ys</sub>i<sub>s o</sub>f <sub>para</sub>ll<sub>e</sub>l <sub>con</sub>t<sub>ex</sub>t <sub>enco</sub>di<sub>ng w</sub>ith f<sub>u</sub>ll<sub>-</sub>

<sub>a</sub>tt<sub>en</sub>ti<sub>on-</sub>b<sub>ase</sub>d <sub>pre-</sub>t<sub>ra</sub>i<sub>ne</sub>d l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> I<sub>n</sub> W<sub>anx</sub>i<sub>ang</sub> Ch<sub>e,</sub> J<sub>oyce</sub> N<sub>a</sub>b<sub>en</sub>d<sub>e,</sub> Ek<sub>a</sub>t<sub>er</sub>i<sub>na</sub> Sh<sub>u</sub>t<sub>ova,</sub> and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9840–9855, Vienna, Austria, July 2025. Association for Com<sub>p</sub>utational Lin<sub>g</sub>uistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-lon<sub>g</sub>.485. URL https://aclanthology.org/2025.acl-long.485/.

Min<sub>g</sub>min<sub>g</sub> Zhao<sub>,</sub> Ji<sub>q</sub>ian Don<sub>g,</sub> Kan<sub>gp</sub>in<sub>g</sub> Xu<sub>,</sub> Zadid Hasan<sub>,</sub> Chen<sub>g</sub>rui Fan<sub>,</sub> Shan Jian<sub>g,</sub> Shuai Mao<sub>,</sub> Yatin<sub>g</sub> Lin<sub>g,</sub> Lin<sub>y</sub>i Zou<sub>,</sub> Tailin Zhou<sub>,</sub> Yun Hin Chan<sub>,</sub> Wenkai Zhan<sub>g,</sub> Zhanhon<sub>g</sub> Zhou<sub>,</sub> Guowei Huan<sub>g,</sub> Hon<sub>g</sub>lian<sub>g</sub> Li, Wenjing Cun, Zhitang Chen, Mingxuan Yuan, and Yanhui Geng. ScienceFlow: A long-horizon agent <sup>f</sup>or ML researc<sup>h</sup>, sc<sup>i</sup>ent<sup>ifi</sup>c <sup>di</sup>scovery an<sup>d b</sup>eyon<sup>d</sup>. arX<sup>i</sup>v prepr<sup>i</sup>nt arX<sup>i</sup>v:2<sup>6</sup>08.14354, 202<sup>6</sup>. URL https: //arxiv.org/abs/2608.14354.

Xi<sub>nyu</sub> Zh<sub>ao,</sub> F<sub>angcong</sub> Yi<sub>n, an</sub>d G<sub>reg</sub> D<sub>urre</sub>tt<sub>.</sub> U<sub>n</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng syn</sub>th<sub>e</sub>ti<sub>c con</sub>t<sub>ex</sub>t <sub>ex</sub>t<sub>ens</sub>i<sub>on v</sub>i<sub>a re</sub>t<sub>r</sub>i<sub>eva</sub>l h<sub>ea</sub>d<sub>s.</sub> In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 77885–77910. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/zhao25ad.html.

Shiyi Zhu, Jing Ye, Wei Jiang, Siqiao Xue, Qi Zhang, Yifan Wu, and Jianguo Li. CoCA: Fusing position <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> <sub>w</sub>ith <sub>co</sub>lli<sub>near</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> i<sub>n</sub> t<sub>rans</sub>f<sub>ormers</sub> f<sub>or</sub> l<sub>ong</sub> <sub>con</sub>t<sub>ex</sub>t <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> <sub>ex</sub>t<sub>en</sub>di<sub>ng.</sub> I<sub>n</sub> Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4247–4262, Bangkok, Thailand, Au<sub>g</sub>ust 2024. Association for Com<sub>p</sub>utational Lin<sub>g</sub>uistics. doi: 10.18653/v1/2024.acl-lon<sub>g</sub>.233. URL https://aclanthology.org/2024.acl-long.233/.

## Appendix

## Contents

A Formal Theory for the Main Results 24   
A<sub>.</sub>1 N<sub>o</sub>t<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e</sub> D<sub>e</sub>f<sub>erre</sub>d A<sub>na</sub>l<sub>ys</sub>i<sub>s .</sub> 24   
A.2 RoPE Score Reduction . 25   
A.3 Finite-Window Hi<sub>g</sub>h-Fre<sub>q</sub>uenc<sub>y</sub> Calibration. 25   
A.4 Positional Re<sub>q</sub>uirements . 26   
A.5 Precise Context-Len<sub>g</sub>th Theorem and Its Ex<sub>p</sub>licit Bound. . 27   
B Extensions to Other Failure Modes 30   
B.1 Near–Far Orderin<sub>g</sub> . 30   
B.2 Score-Barrier Stabilit<sub>y</sub>. 31   
B.3 Full<sub>y</sub>-Hi<sub>g</sub>h Pairwise Reversal 33   
B.4 Inter<sub>p</sub>retation for <sub>p</sub>-RoPE 34   
C Proofs of the Theoretical Results 35   
C.1 Proof of Pro<sub>p</sub>osition 2 . 35   
C.2 Finite-Window Fourier-Frame Bound . 36   
C.3 Proof of Pro<sub>p</sub>osition 1 and Lemma A.1 37   
C<sub>.</sub>4 St<sub>reng</sub>th<sub>ene</sub>d B<sub>an</sub>d<sub>w</sub>i<sub>se</sub> C<sub>a</sub>lib<sub>ra</sub>ti<sub>on .</sub> 39   
C.5 Direct Finite-Sum Intuition 40   
C<sub>.</sub>6 P<sub>os</sub>iti<sub>o</sub>n<sub>a</sub>l R<sub>espo</sub>n<sub>se</sub> B<sub>ou</sub>nd<sub>s a</sub>nd R<sub>e</sub>fin<sub>e</sub>m<sub>e</sub>nt<sub>s .</sub> 40   
C.7 Proof of Theorem 2. 45   
C.8 Proof of Lemma B.1 46   
C.9 Proof of Corollar<sub>y</sub> B.1 . 47   
C.10 Proof of Theorem 1. 48   
C.11 Proof of Theorem 3. . 50   
D Empirical Validation of the Theory 51   
D.1 Local Positional-Res<sub>p</sub>onse Envelo<sub>p</sub>e and Local-Res<sub>p</sub>onse Floor . 51   
D.2 Ex<sub>p</sub>erimental Details for Fi<sub>g</sub>ure 3. . 52   
D.3 Additional Evidence on Hi<sub>g</sub>h-Fre<sub>q</sub>uenc<sub>y</sub> Scores . 54   
E The 49 Reference Task Settings 56   
F Token and Head Selection 58   
F.1 Token Sam<sub>p</sub>lin<sub>g</sub> from the Evaluated In<sub>p</sub>uts. 58   
F.2 S<sub>e</sub>l<sub>ec</sub>ti<sub>ng a</sub> Fi<sub>xe</sub>d S<sub>e</sub>t <sub>o</sub>f Att<sub>en</sub>ti<sub>on</sub> H<sub>ea</sub>d<sub>s.</sub> 58   
G Experimental Details 60   
G.1 Ex<sub>p</sub>eriment 1: Task and Cross-Model Dia<sub>g</sub>nostic Profiles 60   
G.2 Ex<sub>p</sub>eriment 2: Directed Hi<sub>g</sub>h-Fre<sub>q</sub>uenc<sub>y</sub> Intervention 62   
H Results in the Assigned Optimization Direction 65   
H.1 Qwen3-8B: Semantic Direction 65   
H.2 Qwen3-8B: Positional Direction 66   
H.3 Llama-3.1-8B-Instruct: Semantic Direction 68   
H.4 Llama-3.1-8B-Instruct: Positional Direction 69   
I Additional Related Work 71   
I.1 Re<sub>p</sub>resentation Assum<sub>p</sub>tions in RoPE Theor<sub>y</sub>. . 71   
I.2 RoPE Failure Mechanisms . 71   
I.3 Context Extension. 71   
I.4 I<sub>n</sub>t<sub>erna</sub>l R<sub>epresen</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>an</sub>d F<sub>a</sub>il<sub>ure</sub> Di<sub>agnos</sub>i<sub>s</sub> <sub>.</sub> 72   
I.5 B<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l E<sub>va</sub>l<sub>ua</sub>ti<sub>on an</sub>d Att<sub>en</sub>ti<sub>on</sub> C<sub>ompe</sub>titi<sub>on .</sub> 72   
I.6 Lon<sub>g</sub>-Context Failure Modes 72

## A. Formal Theory for the Main Results

Thi<sub>s appen</sub>di<sub>x s</sub>t<sub>a</sub>t<sub>es</sub> th<sub>e score represen</sub>t<sub>a</sub>ti<sub>on,</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow momen</sub>t <sub>guaran</sub>t<sub>ee,</sub> th<sub>e</sub> l<sub>oca</sub>l<sub>-response</sub> <sub>requ</sub>i<sub>remen</sub>t<sub>, an</sub>d th<sub>e prec</sub>i<sub>se ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>-</sub>l<sub>eng</sub>th th<sub>eorem.</sub> Th<sub>e</sub>i<sub>r proo</sub>f<sub>s are co</sub>ll<sub>ec</sub>t<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> $\mathrm { C j }$ com<sub>p</sub>lementar<sub>y</sub> failure modes a<sub>pp</sub>ear in A<sub>pp</sub>endix B.

## A.1. Notation for the Deferred Analysis

W<sub>e recor</sub>d th<sub>e genera</sub>l <sub>no</sub>t<sub>a</sub>ti<sub>on use</sub>d b<sub>y</sub> th<sub>e proo</sub>f<sub>s an</sub>d <sub>re</sub>l<sub>a</sub>t<sub>e</sub> it t<sub>o</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency score</sub> $S _ { H }$ i<sub>n</sub> $\ S 3 . 1$ C<sub>ons</sub>id<sub>er a</sub> f<sub>u</sub>ll<sub>y ro</sub>t<sub>ary</sub> h<sub>ea</sub>d <sub>w</sub>ith $h \geq 1$ <sub>ro</sub>t<sub>ary coor</sub>di<sub>na</sub>t<sub>e pa</sub>i<sub>rs an</sub>d b<sub>ase</sub> $B > 1$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub>

$$
\rho : = B ^ { - 1 / h } , \qquad \omega _ { n } : = \rho ^ { n } , \qquad n = 0 , \ldots , h - 1 , \qquad \mathcal { F } : = \{ 0 , \ldots , h - 1 \} .\tag{8}
$$

F<sub>or</sub> fi<sub>xe</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> $z \in \mathbb { C } ^ { h }$ <sub>an</sub>d <sub>a</sub> f<sub>requency</sub> b<sub>an</sub>d $J \subseteq { \mathcal { F } }$ , write

$$
X _ { J } ^ { z } ( m ) : = \operatorname { R e } \sum _ { n \in J } z _ { n } e ^ { i m \omega _ { n } } , \qquad J \subseteq { \mathcal { F } } .\tag{9}
$$

Th<sub>us</sub> th<sub>e</sub> f<sub>u</sub>ll<sub>y ro</sub>t<sub>ary score</sub> $S ( m )$ i<sub>n</sub> th<sub>e ma</sub>i<sub>n</sub> t<sub>ex</sub>t i<sub>s</sub> $X _ { \mathcal { F } } ^ { z } ( m )$ <sub>, an</sub>d it<sub>s</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>s</sub> $S _ { H } ( m ) =$ $X _ { H } ^ { z } ( m )$ <sub>.</sub> Th<sub>e</sub> <sub>genera</sub>l b<sub>an</sub>d <sub>norm</sub> i<sub>s</sub>

$$
U _ { J } ( z ) : = \left( \sum _ { n \in J } | z _ { n } | ^ { 2 } \right) ^ { 1 / 2 } , \qquad U ( z ) : = U _ { \mathcal { F } } ( z ) = \| z \| _ { 2 } .\tag{10}
$$

F<sub>or</sub> <sub>an</sub> i<sub>n</sub>t<sub>eger</sub> <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> l<sub>eng</sub>th $M \geq 2$ <sub>,</sub> l<sub>e</sub>t

$$
\langle f \rangle _ { M } : = \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } f ( m ) .\tag{11}
$$

Th<sub>e no</sub>t<sub>a</sub>ti<sub>on</sub> ${ \mathrm { V a r } } _ { M } ( f )$ d<sub>eno</sub>t<sub>es</sub> $\left. ( f - \left. f \right. _ { M } ) ^ { 2 } \right. _ { M }$ f<sub>or rea</sub>l<sub>-va</sub>l<sub>ue</sub>d $f .$ E<sub>qu</sub>i<sub>va</sub>l<sub>en</sub>tl<sub>y,</sub> th<sub>ese are</sub> th<sub>e mean an</sub>d <sub>var</sub>i<sub>ance</sub> under m ∼ Unif $\{ 0 , \ldots , M - 1 \}$

Let $C _ { \mathrm { F } } > 0$ b<sub>e</sub> th<sub>e</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> F<sub>our</sub>i<sub>er-</sub>f<sub>rame</sub> <sub>cons</sub>t<sub>an</sub>t i<sub>n</sub> L<sub>emma</sub> C<sub>.</sub>1<sub>.</sub> F<sub>or</sub> <sub>a</sub> fi<sub>xe</sub>d t<sub>o</sub>l<sub>erance</sub> $0 < \varepsilon < 1 / 2 ,$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\Gamma _ { \varepsilon } ( \rho ) : = \frac { 2 C _ { \mathrm { F } } } { ( 1 - \rho ) \varepsilon } , \qquad \mathcal { H } _ { \varepsilon } ( M ) : = \{ n \in \mathscr { F } : M \omega _ { n } \geq \Gamma _ { \varepsilon } ( \rho ) \} , \qquad N _ { \varepsilon } ( M ) : = \mathscr { F } \setminus \mathcal { H } _ { \varepsilon } ( M ) .\tag{12}
$$

A <sub>cer</sub>tifi<sub>e</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ow sa</sub>ti<sub>s</sub>fi<sub>es</sub> $M \geq \Gamma _ { \varepsilon } ( \rho )$ <sub>, so</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency pre</sub>fi<sub>x</sub> $H = \mathcal { H } _ { \varepsilon } ( M )$ <sup>i</sup>s nonempty. Its comp<sup>l</sup>ement i<sub>s</sub> $N = N _ { \varepsilon } ( M )$ <sub>.</sub> Th<sub>e ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>e use</sub>d i<sub>n</sub> th<sub>e ca</sub>lib<sub>ra</sub>ti<sub>on proo</sub>f i<sub>s</sub>

$$
Y _ { H } ^ { z } : = X _ { H } ^ { z } ( { \mathbf { m } } ) = S _ { H } ( { \mathbf { m } } ) , \qquad { \mathbf { m } } \sim { \mathrm { U n i f } } \{ 0 , \dots , M - 1 \} .\tag{13}
$$

For $z \neq 0$ <sub>,</sub> th<sub>e</sub> t<sub>wo norm s</sub>h<sub>ares are</sub>

$$
r _ { H } ( z ; M ) : = \frac { U _ { H } ( z ) } { U ( z ) } , \qquad r _ { N } ( z ; M ) : = \frac { U _ { N } ( z ) } { U ( z ) } , \qquad r _ { H } ( z ; M ) ^ { 2 } + r _ { N } ( z ; M ) ^ { 2 } = 1 .\tag{14}
$$

Th<sub>ese</sub> <sub>co</sub>i<sub>nc</sub>id<sub>e</sub> <sub>w</sub>ith th<sub>e</sub> <sub>ma</sub>i<sub>n-</sub>t<sub>ex</sub>t hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> <sub>norm</sub> <sub>s</sub>h<sub>are</sub> <sub>an</sub>d it<sub>s</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>squares</sub> <sub>are</sub> th<sub>e</sub> corres<sub>p</sub>on<sup>di</sup>n<sub>g</sub> ener<sub>gy</sub> <sup>f</sup>ract<sup>i</sup>ons.

A<sub>pp</sub>l<sub>y</sub> the normalized adjacent res<sub>p</sub>onse from E<sub>q</sub>. (3) to $X _ { \mathcal { F } } ^ { z }$ <sub>.</sub> F<sub>a</sub>il<sub>ure</sub> t<sub>o</sub> <sub>a</sub>tt<sub>a</sub>i<sub>n</sub> <sub>response</sub> l<sub>eve</sub>l $\zeta > 0$ an<sub>y</sub>w<sup>h</sup>ere i<sub>n</sub> th<sub>e w</sub>i<sub>n</sub>d<sub>ow means</sub>

$$
\operatorname* { m a x } _ { 0 \leq m \leq M - 2 } g _ { z } ( m ) = \operatorname* { m a x } _ { 0 \leq m \leq M - 2 } \frac { | X _ { \mathcal { F } } ^ { z } ( m + 1 ) - X _ { \mathcal { F } } ^ { z } ( m ) | } { U ( z ) } < \zeta .\tag{15}
$$

F<sub>or</sub> th<sub>e</sub> <sub>or</sub>d<sub>ere</sub>d <sub>marg</sub>i<sub>n</sub> $D ( m ) = X _ { \mathcal { F } } ^ { d } ( m )$ <sub>w</sub>ith <sub>re</sub>f<sub>erence or</sub>d<sub>er</sub>i<sub>ng</sub> $D ( 0 ) > 0$ <sub>,</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-</sub>b<sub>ase</sub>d <sub>an</sub>d f<sub>unc</sub>ti<sub>on-</sub> b<sub>ase</sub>d <sub>reversa</sub>l <sub>no</sub>t<sub>a</sub>ti<sub>on</sub> <sub>re</sub>f<sub>er</sub> t<sub>o</sub> th<sub>e</sub> <sub>same</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y:</sub>

$$
p _ { \mathtt { r e v } } ( d ; M ) = p _ { \mathtt { r e v } } ( D ; M ) : = \mathbb { P } [ D ( \mathbf { m } ) < 0 ] = \mathbb { P } [ X _ { \mathcal { F } } ^ { d } ( \mathbf { m } ) < 0 ] .\tag{16}
$$

In particular, the main theorem’s �<sub>�</sub> is the normalized adjacent response of this same margin, with denominator $U ( d )$

## A.2. RoPE Score Reduction

The reduction uses RoPE’s relative-rotation identit<sub>y</sub> (Su et al., 2024).

Proposition 2 (RoPE score reduction). For afully rotary RoPE head, as the real query and key blocks range over $\mathbb { R } ^ { 2 h }$ , the class offixed query–key scores and their finite real linear combinations as functions of relative distance is exactly $\{ X _ { \mathcal { F } } ^ { z } : z \in \mathbb { C } ^ { h } \} ;$ hence all subsequentfull- and bandwise analyses reduce to studying $X _ { J } ^ { z }$

Complex-coordinate derivation. Write the two real coordinates of each rotary pair as $Q _ { n } = q _ { n , 1 } + i q _ { n , 2 }$ <sub>an</sub>d $K _ { n } = k _ { n , 1 } + i k _ { n , 2 }$ <sub>,</sub> <sub>an</sub>d l<sub>e</sub>t $R ( \theta )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r p</sub>l<sub>anar ro</sub>t<sub>a</sub>ti<sub>on.</sub> F<sub>or re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance</sub> $m = p _ { q } - p _ { k }$ <sub>an</sub>d <sub>a</sub> <sub>pos</sub>iti<sub>on-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>score sca</sub>l<sub>e</sub> $c _ { \mathrm { a t t } } > 0$

$$
\begin{array} { r l } & { c _ { \mathrm { a t t } } \displaystyle \sum _ { n } \left. R ( p _ { q } \omega _ { n } ) q _ { n } , R ( p _ { k } \omega _ { n } ) k _ { n } \right. = c _ { \mathrm { a t t } } \displaystyle \sum _ { n } q _ { n } ^ { \top } R ( - m \omega _ { n } ) k _ { n } } \\ & { \qquad = \mathrm { R e } \displaystyle \sum _ { n } \underbrace { c _ { \mathrm { a t t } } Q _ { n } \overline { { K _ { n } } } } _ { z _ { n } } e ^ { i m \omega _ { n } } = X _ { \mathcal { F } } ^ { z } ( m ) . } \end{array}
$$

Th<sub>e</sub> <sub>same</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>app</sub>li<sub>es</sub> t<sub>o</sub> <sub>score</sub> <sub>marg</sub>i<sub>ns</sub> b<sub>y</sub> <sub>su</sub>bt<sub>rac</sub>ti<sub>ng</sub> th<sub>e</sub>i<sub>r</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>ors.</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> f<sub>u</sub>ll d<sub>er</sub>i<sub>va</sub>ti<sub>on an</sub>d th<sub>e converse cons</sub>t<sub>ruc</sub>ti<sub>on.</sub>

O<sub>pera</sub>ti<sub>ona</sub>ll<sub>y,</sub> th<sub>e re</sub>d<sub>uc</sub>ti<sub>on</sub> i<sub>so</sub>l<sub>a</sub>t<sub>es a</sub>ll d<sub>epen</sub>d<sub>ence on re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance</sub> i<sub>n a</sub> fi<sub>n</sub>it<sub>e</sub> t<sub>r</sub>i<sub>gonome</sub>t<sub>r</sub>i<sub>c po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>.</sub> It <sub>preserves</sub> th<sub>e spec</sub>t<sub>ra</sub>l <sub>compe</sub>titi<sub>on</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> di<sub>agnos</sub>ti<sub>cs w</sub>hil<sub>e</sub> l<sub>eav</sub>i<sub>ng con</sub>t<sub>en</sub>t<sub>-s</sub>t<sub>a</sub>t<sub>e c</sub>h<sub>anges an</sub>d d<sub>owns</sub>t<sub>ream a</sub>tt<sub>en</sub>ti<sub>on compu</sub>t<sub>a</sub>ti<sub>ons ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e cer</sub>tifi<sub>ca</sub>t<sub>e.</sub>

U<sub>n</sub>d<sub>er</sub> th<sub>e same</sub> fi<sub>xe</sub>d<sub>-con</sub>t<sub>en</sub>t <sub>assump</sub>ti<sub>on,</sub> th<sub>e argumen</sub>t d<sub>oes no</sub>t d<sub>epen</sub>d <sub>on w</sub>h<sub>e</sub>th<sub>er an</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ores</sub> <sub>eac</sub>h <sub>ro</sub>t<sub>ary</sub> <sub>pa</sub>i<sub>r</sub> <sub>con</sub>ti<sub>guous</sub>l<sub>y</sub> <sub>or</sub> i<sub>n</sub> <sub>a</sub> <sub>sp</sub>lit<sub>-</sub>h<sub>a</sub>lf l<sub>ayou</sub>t<sub>;</sub> <sub>on</sub>l<sub>y</sub> th<sub>e</sub> <sub>pa</sub>i<sub>r</sub>i<sub>ng</sub> <sub>o</sub>f <sub>rea</sub>l <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>ma</sub>tt<sub>ers.</sub> A <sub>non-ro</sub>t<sub>ary</sub> t<sub>a</sub>il th<sub>en con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es a pos</sub>iti<sub>on-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>cons</sub>t<sub>an</sub>t<sub>.</sub> It i<sub>s om</sub>itt<sub>e</sub>d f<sub>rom</sub> th<sub>e</sub> f<sub>orma</sub>l f<sub>u</sub>ll<sub>y ro</sub>t<sub>ary</sub> <sub>resu</sub>lt<sub>s, w</sub>hil<sub>e</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>4 di<sub>scusses</sub> it<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>e</sub>f<sub>ec</sub>t<sub>s on cross-</sub>k<sub>ey score marg</sub>i<sub>ns an</sub>d <sub>same-</sub>k<sub>ey pos</sub>iti<sub>ona</sub>l com<sub>p</sub>ar<sup>i</sup>sons.

## A.3. Finite-Window High-Frequency Calibration

Th<sub>e momen</sub>t <sub>conc</sub>l<sub>us</sub>i<sub>on</sub> b<sub>e</sub>l<sub>ow</sub> i<sub>s</sub> P<sub>ropos</sub>iti<sub>on</sub> 1 i<sub>n</sub> th<sub>e no</sub>t<sub>a</sub>ti<sub>on o</sub>f thi<sub>s appen</sub>di<sub>x.</sub> Th<sub>e a</sub>dditi<sub>ona</sub>l <sub>con</sub>diti<sub>ona</sub>l <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t <sub>recor</sub>d<sub>s</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an approx</sub>i<sub>ma</sub>ti<sub>on error use</sub>d b<sub>y</sub> th<sub>e ex</sub>t<sub>ens</sub>i<sub>ons.</sub>

Lemma A.1 (Finite-window moments and conditional Gaussian calibration). Fix $0 < \varepsilon < 1 / 2 ,$ a certified window $M \geq \Gamma _ { \varepsilon } ( \rho )$ , and fixed coeficients � with $U _ { H } ( z ) > 0$ . Use $H = \mathcal { H } _ { \varepsilon } ( M ) f r o m$ Eq. (12) and the uniformly sampled relative distance in Eq. (13). For its random score $Y _ { H } ^ { z } ,$ write

$$
\mu _ { H } : = \big \langle X _ { H } ^ { z } \big \rangle _ { M } , \qquad \sigma _ { H } ^ { 2 } : = \mathsf { V a r } _ { M } ( X _ { H } ^ { z } ) .\tag{17}
$$

Then the following moment bounds hold:

$$
\begin{array} { r } { | \mu _ { H } | \le \varepsilon U _ { H } ( z ) , \qquad \left| \sigma _ { H } ^ { 2 } - \frac 1 2 U _ { H } ( z ) ^ { 2 } \right| \le \varepsilon U _ { H } ( z ) ^ { 2 } . } \end{array}\tag{18}
$$

Because $0 < \varepsilon < 1 / 2 ,$ , Eq. (18) implies $\sigma _ { H } ^ { 2 } \geq ( 1 / 2 - \varepsilon ) U _ { H } ( z ) ^ { 2 } > 0$ whenever $U _ { H } ( z ) > 0$ . The standardization below is therefore well defined. The remaining, non-deterministic source of approximation error is the Gaussian-shape discrepancy of the standardized window law,

$$
\Delta _ { \mathrm { H } , \mathrm { G } } ( z ; M ) : = \operatorname* { s u p } _ { x \in \mathbb { R } } \left. \mathbb { P } \left[ \frac { Y _ { H } ^ { z } - \mu _ { H } } { \sigma _ { H } } \leq x \right] - \Phi _ { \mathrm { s t d } } ( x ) \right. \leq \tau _ { \mathrm { H } , \mathrm { G } } .\tag{19}
$$

Under this condition, the exact form of the Gaussian conclusion is

$$
\operatorname* { s u p } _ { x \in \mathbb { R } } \left. \mathbb { P } [ Y _ { H } ^ { z } \leq x ] - \Phi _ { \mathrm { s t d } } \left( \frac { \sqrt { 2 } x } { U _ { H } ( z ) } \right) \right. \leq \tau _ { \mathrm { H , G } } + c _ { \mathrm { G } } ( \varepsilon ) ,\tag{20}
$$

where the universal moment-replacement term is

$$
c _ { \mathsf { G } } ( \varepsilon ) : = { \frac { \varepsilon } { \sqrt { \pi } } } + { \frac { \log ( ( 1 - 2 \varepsilon ) ^ { - 1 } ) } { 2 { \sqrt { 2 \pi e } } } } = O ( \varepsilon ) \qquad ( \varepsilon \to 0 ) .\tag{21}
$$

We use the convention ${ \cal N } ( \mu , \sigma ^ { 2 } )$ : the target variance is $U _ { H } ( z ) ^ { 2 } / 2 ,$ equivalently the target standard deviation is $U _ { H } ( z ) / { \sqrt { 2 } } .$

## A.4. Positional Requirements

## A.4.1. Local Positional Response

F<sub>or a</sub> h<sub>ea</sub>d <sub>w</sub>h<sub>ose</sub> f<sub>unc</sub>ti<sub>on requ</sub>i<sub>res ne</sub>i<sub>g</sub>hb<sub>or</sub>i<sub>ng re</sub>l<sub>a</sub>ti<sub>ve pos</sub>iti<sub>ons</sub> t<sub>o rema</sub>i<sub>n</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>a</sub>bl<sub>e, mov</sub>i<sub>ng a</sub> k<sub>ey</sub> b<sub>y</sub> one token should induce a non-ne<sub>g</sub>li<sub>g</sub>ible normalized score chan<sub>g</sub>e. In the terminolo<sub>gy</sub> of §3.2, avoidin<sub>g</sub> the window-level failure in Eq. (15) at response level � requires at least one adjacent score change of normalized magnitude �. We now quantify how maintaining this response constrains the certified-high norm share as th<sub>e con</sub>t<sub>ex</sub>t <sub>sca</sub>l<sub>e grows.</sub>

Lemma A.2 (Direct scale-to-share response envelope). For every certified integer $M \geq 2$ and every $a \neq 0 ,$

$$
\begin{array} { r l }  \displaystyle \operatorname* { m a x } _ { 0 \le m \le M - 2 } \frac { \left| X _ { \mathcal { F } } ^ { a } ( m + 1 ) - X _ { \mathcal { F } } ^ { a } ( m ) \right| } { U ( a ) } \le \underbrace { \frac { \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } } { M } } _  \underbrace { n o n \mathrm { - } h i g h \ : b a n d \mathrm { : } } _  s m a l l \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm { - } t \mathrm  -  \end{array}\tag{22}
$$

where $\begin{array} { r } { R _ { \mathcal { F } } : = \left( \sum _ { n \in \mathcal { F } } \left| e ^ { i \omega _ { n } } - 1 \right| ^ { 2 } \right) ^ { 1 / 2 } > 0 } \end{array}$ is the fixed full-grid adjacent-gain factor and is independent of �.

Th<sub>e non-</sub>hi<sub>g</sub>h <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y a</sub> $1 / M$ <sub>enve</sub>l<sub>ope</sub> b<sub>ecause every non-</sub>hi<sub>g</sub>h f<sub>requency ro</sub>t<sub>a</sub>t<sub>es</sub> b<sub>y</sub> l<sub>ess</sub> th<sub>an</sub> $\Gamma _ { \varepsilon } ( \rho ) / M$ <sub>per</sub> t<sub>o</sub>k<sub>en.</sub> Th<sub>e secon</sub>d t<sub>erm</sub> h<sub>as no exp</sub>li<sub>c</sub>it i<sub>nverse-sca</sub>l<sub>e</sub> d<sub>ecay:</sub> it<sub>s</sub> f<sub>u</sub>ll<sub>-gr</sub>id <sub>ga</sub>i<sub>n</sub> f<sub>ac</sub>t<sub>or</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $M ,$ b<sub>u</sub>t th<sub>e cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h <sub>norm s</sub>h<sub>are can s</sub>till <sub>vary w</sub>ith th<sub>e w</sub>i<sub>n</sub>d<sub>ow.</sub>

Corollary A.1 (Stable local-response floor on the certified-high norm share). For every certified integer $M \geq 2 ,$ every $a \neq 0 ,$ and every $\zeta > 0 ,$ if the score $X _ { \mathcal { F } } ^ { a }$ avoids the window-level failure in Eq. (15) at response level $\zeta ,$ then

$$
r _ { H } ( a ; M ) \geq \underline { { r } } _ { \mathrm { p o s } } ( M ; \zeta ) : = \frac { \left[ \zeta - \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } / M \right] _ { + } } { R _ { \mathcal { F } } } .\tag{23}
$$

where $R _ { \mathcal { F } }$ is thefixedfactor in Lemma A.2. If the right-hand side exceeds one, the requested response is impossible.

T<sub>oge</sub>th<sub>er,</sub> L<sub>emma</sub> A<sub>.</sub>2 <sub>an</sub>d C<sub>oro</sub>ll<sub>ary</sub> A<sub>.</sub>1 di<sub>rec</sub>tl<sub>y</sub> li<sub>n</sub>k th<sub>e con</sub>t<sub>ex</sub>t <sub>sca</sub>l<sub>e</sub> � t<sub>o</sub> th<sub>e cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h <sub>norm s</sub>h<sub>are:</sub> maintaining the same adjacent response level $\zeta$ f<sub>orces</sub> $r _ { H } ( a ; M )$ <sub>a</sub>b<sub>ove</sub> $\underline { { r } } _ { \mathrm { p o s } } ( M ; \zeta )$ <sub>.</sub> Thi<sub>s</sub> fl<sub>oor</sub> i<sub>s</sub> th<sub>e res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on</sub> t<sub>o</sub> <sub>cer</sub>tifi<sub>e</sub>d i<sub>n</sub>t<sub>eger w</sub>i<sub>n</sub>d<sub>ows o</sub>f <sub>a con</sub>ti<sub>nuous, non-</sub>d<sub>ecreas</sub>i<sub>ng</sub> f<sub>unc</sub>ti<sub>on o</sub>f <sub>rea</sub>l $M > 0$ <sub>, an</sub>d it i<sub>s s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> <sub>once pos</sub>iti<sub>ve.</sub> Th<sub>us</sub> it <sub>prov</sub>id<sub>es a s</sub>t<sub>a</sub>bl<sub>e non-</sub>d<sub>ecreas</sub>i<sub>ng</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>even</sub> th<sub>oug</sub>h $H ( M )$ c<sup>h</sup>an<sub>g</sub>es on<sup>l</sup><sub>y</sub> at di<sub>scre</sub>t<sub>e</sub> b<sub>rea</sub>k<sub>po</sub>i<sub>n</sub>t<sub>s.</sub> E<sub>n</sub>l<sub>arg</sub>i<sub>ng</sub> $H ( M )$ <sub>a</sub>l<sub>one wou</sub>ld <sub>no</sub>t <sub>g</sub>i<sub>ve</sub> thi<sub>s conc</sub>l<sub>us</sub>i<sub>on, s</sub>i<sub>nce a new</sub>l<sub>y cer</sub>tifi<sub>e</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t ma be zero and the actual share need not jum . The continuit and monotonicit claim concerns the <sub>necessary</sub> fl<sub>oor,</sub> <sub>no</sub>t th<sub>e</sub> <sub>ac</sub>t<sub>ua</sub>l <sub>s</sub>t<sub>epw</sub>i<sub>se</sub> <sub>quan</sub>tit<sub>y</sub> $r _ { H } ( a ; M )$ <sub>or</sub> it<sub>s</sub> i<sub>ncremen</sub>t b<sub>e</sub>t<sub>ween</sub> t<sub>wo w</sub>i<sub>n</sub>d<sub>ows.</sub> N<sub>o pr</sub>i<sub>or</sub> assum<sub>p</sub>t<sup>i</sup>on $U _ { H } ( a ) > 0$ i<sub>s nee</sub>d<sub>e</sub>d<sub>: a pos</sub>iti<sub>ve</sub> fl<sub>oor</sub> it<sub>se</sub>lf f<sub>orces</sub> th<sub>a</sub>t <sub>conc</sub>l<sub>us</sub>i<sub>on.</sub> Sh<sub>arper coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-spec</sub>ifi<sub>c</sub> r<sub>e</sub>fin<sub>e</sub>m<sub>e</sub>nt<sub>s a</sub>r<sub>e g</sub>iv<sub>e</sub>n in A<sub>ppe</sub>ndix C<sub>.</sub>6<sub>.</sub>4<sub>.</sub>

Th<sub>e</sub> <sub>response</sub> fl<sub>oor</sub> <sub>app</sub>li<sub>es</sub> t<sub>o</sub> th<sub>e</sub> <sub>same</sub> <sub>or</sub>d<sub>ere</sub>d <sub>marg</sub>i<sub>n</sub> <sub>use</sub>d i<sub>n</sub> th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t<sub>-</sub>l<sub>eng</sub>th <sub>resu</sub>lt b<sub>e</sub>l<sub>ow.</sub>

## A.5. Precise Context-Length Theorem and Its Explicit Bound

Fi<sub>x</sub> <sub>a</sub> f<sub>u</sub>ll<sub>y</sub> <sub>ro</sub>t<sub>ary</sub> h<sub>ea</sub>d<sub>,</sub> $0 < \varepsilon < 1 / 2$ <sub>, an</sub>d th<sub>e same nonzero or</sub>d<sub>ere</sub>d <sub>marg</sub>i<sub>n coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> � <sub>w</sub>ith $D ( 0 ) > 0$ <sub>across a</sub> fi<sub>n</sub>it<sub>e or</sub>d<sub>ere</sub>d <sub>se</sub>t <sub>o</sub>f <sub>cer</sub>tifi<sub>e</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ows</sub> ${ \mathfrak { M } } _ { \mathrm { c a l } }$ <sub>.</sub> Fi<sub>x</sub> t<sub>o</sub>l<sub>erances</sub> $0 \leq \delta < 1 / 2$ <sub>an</sub>d $0 < \zeta < R _ { \mathcal { F } }$ , to<sub>g</sub>et<sup>h</sup>er <sub>w</sub>ith <sub>a un</sub>if<sub>orm</sub> G<sub>auss</sub>i<sub>an</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>-error</sub> b<sub>oun</sub>d $0 \leq \overline { { \tau } } _ { \mathrm { G } } < 1 / 2$ . For the ordered mar<sub>g</sub>in � in §3.2, set

$$
\mu _ { D } ( M ) : = \big \langle X _ { \mathcal { F } } ^ { d } \big \rangle _ { M } , \qquad \sigma _ { D } ^ { 2 } ( M ) : = \mathrm { V a r } _ { M } ( X _ { \mathcal { F } } ^ { d } ) ,\tag{24}
$$

<sub>an</sub>d <sub>assume</sub> $\sigma _ { D } ( M ) > 0$ <sub>a</sub>t <sub>every</sub> <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> i<sub>n</sub> ${ \mathfrak { M } } _ { \mathrm { c a l } }$ <sub>.</sub> Th<sub>e</sub> <sub>exac</sub>t<sub>-momen</sub>t G<sub>auss</sub>i<sub>an</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>error</sub> i<sub>s</sub>

$$
\Delta _ { \mathrm { r e v } } ( d ; M ) : = \left| p _ { \mathrm { r e v } } ( d ; M ) - \Phi _ { \mathrm { s t d } } \left( - \frac { \mu _ { D } ( M ) } { \sigma _ { D } ( M ) } \right) \right| .\tag{25}
$$

Th<sub>e ca</sub>lib<sub>ra</sub>t<sub>e</sub>d f<sub>am</sub>il<sub>y</sub> i<sub>n</sub> Th<sub>eorem</sub> 1 <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\Delta _ { \mathrm { r e v } } ( d ; M ) \le \overline { { \tau } } _ { \mathtt { G } }$ <sub>un</sub>if<sub>orm</sub>l<sub>y.</sub>

Let

$$
k _ { N } ( M ) : = | N _ { \varepsilon } ( M ) | , \qquad c _ { \varepsilon } : = \sqrt { { \frac { 1 } { 2 } } - \varepsilon } ,\tag{26}
$$

<sub>an</sub>d<sub>,</sub> f<sub>or</sub> $0 \leq r \leq 1$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\begin{array} { l } { { u _ { M } ( r ) : = \varepsilon r + \sqrt { k _ { N } ( M ) } \sqrt { 1 - r ^ { 2 } } , \ } } \\ { { \nu _ { M } ( r ) : = \left[ c _ { \varepsilon } r - \sqrt { k _ { N } ( M ) } \sqrt { 1 - r ^ { 2 } } \right] _ { + } . } } \end{array}\tag{27}
$$

Recall the full-grid adjacent-gain bound $\begin{array} { r } { R _ { \mathcal { F } } = \big ( \sum _ { n \in \mathcal { F } } \big | e ^ { i \omega _ { n } } - 1 \big | ^ { 2 } \big ) ^ { 1 / 2 } > 0 } \end{array}$ <sub>.</sub> Th<sub>e ma</sub>i<sub>n</sub> th<sub>eorem assumes</sub> $0 < \zeta < R _ { \mathcal { F } }$ B<sub>y</sub> Corollar<sub>y</sub> $\mathrm { A . 1 }$ <sub>, re</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng response</sub> l<sub>eve</sub>l $\zeta$ f<sub>or</sub> th<sub>e same marg</sub>i<sub>n</sub> � <sub>requ</sub>i<sub>res</sub>

$$
r _ { H } ( d ; M ) \ge \ell ( M ; \zeta ) : = \frac { [ \zeta - \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } / M ] _ { + } } { R _ { \mathcal { F } } } .\tag{28}
$$

Here $[ x ] _ { + } = \operatorname* { m a x } \{ x , 0 \}$ <sub>.</sub> Th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>range</sub> <sub>o</sub>f $\zeta$ ensures $0 \leq \ell ( M ; \zeta ) < 1$ <sub>, w</sub>ithi<sub>n</sub> th<sub>e</sub> d<sub>oma</sub>i<sub>n o</sub>f th<sub>e reversa</sub>l enve<sup>l</sup>o<sub>p</sub>e.

Th<sub>e</sub> G<sub>auss</sub>i<sub>an reversa</sub>l <sub>enve</sub>l<sub>ope an</sub>d th<sub>e</sub> fl<sub>oor use</sub>d i<sub>n</sub> th<sub>e ma</sub>i<sub>n</sub> t<sub>ex</sub>t <sub>are</sub>

$$
\psi _ { M } ( r ; \overline { { \tau } } _ { \mathrm { G } } ) : = \left\{ \begin{array} { l l } { \left[ \Phi _ { \mathrm { s t d } } ( - u _ { M } ( r ) / \upsilon _ { M } ( r ) ) - \overline { { \tau } } _ { \mathrm { G } } \right] _ { + } , } & { \upsilon _ { M } ( r ) > 0 , } \\ { 0 , } & { \upsilon _ { M } ( r ) = 0 , } \end{array} \right.\tag{29}
$$

$$
\underline { { p } } _ { \mathrm { r e v } } ( M ; \zeta , \overline { { \tau } } _ { \mathrm { G } } ) : = \psi _ { M } ( \ell ( M ; \zeta ) ; \overline { { \tau } } _ { \mathrm { G } } ) .\tag{30}
$$

F<sub>or</sub> fi<sub>xe</sub>d $\zeta$ <sub>an</sub>d $\overline { { \tau } } _ { \mathrm { G } }$ <sub>, we a</sub>bb<sub>rev</sub>i<sub>a</sub>t<sub>e</sub> thi<sub>s</sub> b<sub>oun</sub>d <sub>as</sub> $\underline { { p } } _ { \mathrm { r e v } } ( M )$

F<sub>or</sub> <sub>every</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> <sub>a</sub>t <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> <sub>marg</sub>i<sub>n</sub> <sub>re</sub>t<sub>a</sub>i<sub>ns</sub> <sub>response</sub> l<sub>eve</sub>l $\zeta ,$ th<sub>e</sub> l<sub>ower-</sub>b<sub>oun</sub>d <sub>c</sub>h<sub>a</sub>i<sub>n</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> Th<sub>eo</sub>r<sub>e</sub>m 1 i<sub>s</sub>

$$
p _ { \mathrm { r e v } } ( d ; M ) \ge \psi _ { M } ( r _ { H } ( d ; M ) ; \overline { { \tau } } _ { \mathrm { G } } ) \ge \psi _ { M } ( \ell ( M ; \zeta ) ; \overline { { \tau } } _ { \mathrm { G } } ) = \underline { { p } } _ { \mathrm { r e v } } ( M ; \zeta , \overline { { \tau } } _ { \mathrm { G } } ) .\tag{31}
$$

A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>10 <sub>proves</sub> thi<sub>s c</sub>h<sub>a</sub>i<sub>n an</sub>d it<sub>s mono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y</sub> i<sub>n</sub> �<sub>.</sub>

Wh<sub>en</sub> th<sub>e</sub> <sub>cross</sub>i<sub>ng</sub> <sub>se</sub>t i<sub>s</sub> <sub>nonemp</sub>t<sub>y,</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>exc</sub>l<sub>u</sub>d<sub>e</sub>d <sub>sca</sub>l<sub>e</sub> i<sub>s</sub>

$$
M _ { \dagger } : = \operatorname* { m i n } \left\{ M \in { \mathfrak { M } } _ { \mathrm { c a l } } : \underline { { p } } _ { \mathrm { r e v } } ( M ; \zeta , \overline { { \tau } } _ { \mathrm { G } } ) > \delta \right\} .\tag{32}
$$

For every $M \in \mathfrak { M } _ { \mathrm { c a l } }$ <sub>w</sub>ith $M \geq M _ { \dagger }$ †

$$
p _ { \mathrm { r e v } } ( D ; M ) > \delta \qquad \mathrm { o r } \qquad \operatorname* { m a x } _ { 0 \leq m \leq M - 2 } g _ { d } ( m ) < \zeta .\tag{33}
$$

If th<sub>e cross</sub>i<sub>ng se</sub>t i<sub>s emp</sub>t<sub>y,</sub> th<sub>e</sub> b<sub>oun</sub>d <sub>supp</sub>li<sub>es no exc</sub>l<sub>us</sub>i<sub>on</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>w</sub>ithi<sub>n</sub> th<sub>e c</sub>h<sub>osen ca</sub>lib<sub>ra</sub>t<sub>e</sub>d f<sub>am</sub>il<sub>y.</sub> If th<sub>e</sub> <sub>su</sub>b<sub>-</sub>th<sub>res</sub>h<sub>o</sub>ld <sub>se</sub>t i<sub>s</sub> <sub>nonemp</sub>t<sub>y,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
M _ { \mathrm { m a x } } ^ { \mathrm { c e r t } } : = \operatorname* { m a x } \left\{ M \in \mathfrak { M } _ { \mathrm { c a l } } : \underline { { p } } _ { \mathrm { r e v } } \left( M ; \zeta , \overline { { \tau } } _ { \mathrm { G } } \right) \leq \delta \right\} .\tag{34}
$$

This is the lar<sub>g</sub>est calibrated scale not ruled out b<sub>y</sub> the lower bound. When the crossin<sub>g</sub> set in E<sub>q</sub>. (32) i<sub>s a</sub>l<sub>so nonemp</sub>t<sub>y,</sub> it i<sub>s</sub> th<sub>e gr</sub>id <sub>po</sub>i<sub>n</sub>t i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y prece</sub>di<sub>ng</sub> $M _ { \dagger }$ <sub>.</sub> Th<sub>e ma</sub>i<sub>n</sub> t<sub>ex</sub>t <sub>wr</sub>it<sub>es</sub> $M _ { \mathrm { m a x } } = M _ { \mathrm { m a x } } ^ { \mathrm { c e r t } }$ <sub>an</sub>d <sub>uses</sub> th<sub>e reg</sub>i<sub>me</sub> i<sub>n w</sub>hi<sub>c</sub>h b<sub>o</sub>th th<sub>e cross</sub>i<sub>ng an</sub>d <sub>su</sub>b<sub>-</sub>th<sub>res</sub>h<sub>o</sub>ld <sub>se</sub>t<sub>s are nonemp</sub>t<sub>y.</sub> F<sub>or ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ows,</sub> $M > M _ { \mathrm { m a x } }$ i<sub>s</sub> th<sub>en equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> $M \geq M _ { \dagger }$ <sub>.</sub> All <sub>suc</sub>h <sub>w</sub>i<sub>n</sub>d<sub>ows</sub> f<sub>a</sub>il <sub>a</sub>t l<sub>eas</sub>t <sub>one requ</sub>i<sub>remen</sub>t<sub>, w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> simultaneous-infeasibilit<sub>y</sub> statement in E<sub>q</sub>. (7). A scale below the crossin<sub>g</sub> remains a candidate: the lower bound alone does not establish joint reliability there. This count-based envelope is deliberately conservative. It <sub>uses no coe</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> b<sub>eyon</sub>d th<sub>e cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h <sub>s</sub>h<sub>are;</sub> th<sub>e prac</sub>ti<sub>ca</sub>l <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> G<sub>.</sub>1<sub>.</sub>2 <sub>compu</sub>t<sub>es</sub> th<sub>e exac</sub>t f<sub>u</sub>ll<sub>-w</sub>i<sub>n</sub>d<sub>ow mean an</sub>d <sub>var</sub>i<sub>ance,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng cross-</sub>b<sub>an</sub>d <sub>covar</sub>i<sub>ance.</sub>

Theorem 1 (Precise context-length limit on joint reliability). Fix a fully rotary head with the geometric frequency grid in Eq. (8), a tolerance $0 < \varepsilon < 1 / 2$ , and a nonzero ordered margin $D ( m ) = X _ { \mathcal { F } } ^ { d } ( m )$ with $D ( 0 ) > 0$ Let ${ \mathfrak { M } } _ { \mathrm { c a l } }$ be a finite nonempty set of certified integer windows $M \geq 2 ,$ , with the same coeficients � at every window. Fix $0 \le \delta < 1 / 2 , 0 < \zeta < R _ { \mathcal { F } }$ , and $0 \leq \overline { { \tau } } _ { \mathrm { G } } < 1 / 2$ . Assume, for every $M \in \mathfrak { M } _ { \mathrm { c a l } }$ , that $\sigma _ { D } ( M ) > 0$ and

$$
\Delta _ { \mathrm { r e v } } ( d ; M ) = \left| p _ { \mathrm { r e v } } ( d ; M ) - \Phi _ { \mathrm { s t d } } \bigg ( - \frac { \mu _ { D } ( M ) } { \sigma _ { D } ( M ) } \bigg ) \right| \leq \overline { { \tau } } _ { \mathrm { G } } .
$$

Let $\underline { { p } } _ { \mathrm { r e v } } ( M ; \zeta , \overline { { \tau } } _ { \mathtt { G } } )$ be the explicit bound in Eq. (30). This bound is non-decreasing across ${ \mathfrak { M } } _ { \mathrm { c a l } }$ , and each calibrated window retaining the adjacent response $\begin{array} { r } { \operatorname* { m a x } _ { 0 \leq m \leq M - 2 } g _ { d } ( m ) \geq \zeta } \end{array}$ satisfies

$$
p _ { \mathrm { r e v } } ( d ; M ) \geq \underline { { p } } _ { \mathrm { r e v } } ( M ; \zeta , \overline { { \tau } } _ { \mathrm { G } } ) .
$$

Write $M _ { - } : = \operatorname* { m i n } \mathfrak { M } _ { \mathrm { c a l } }$ and $M _ { + } : = \operatorname* { m a x } { \mathfrak { M } } _ { \mathrm { c a l } }$ . If the reversal tolerance satisfies

$$
\underline { { { p } } } _ { \mathrm { r e v } } ( M _ { - } ) \leq \delta < \underline { { { p } } } _ { \mathrm { r e v } } ( M _ { + } ) ,\tag{35}
$$

define $M _ { \mathrm { m a x } } : = M _ { \mathrm { m a x } } ^ { \mathrm { c e r t } }$ by Eq. (34). Then, for every $M \in \mathfrak { M } _ { \mathrm { c a l } }$ with $M > M _ { \mathrm { m a x } }$

$$
p _ { \mathrm { r e v } } ( d ; M ) > \delta \qquad o r \qquad \operatorname* { m a x } _ { 0 \leq m \leq M - 2 } g _ { d } ( m ) < \zeta .
$$

Thus $M _ { \mathrm { m a x } }$ is an upper bound on joint score-level reliability within the calibrated family. A window at or below $M _ { \mathrm { m a x } }$ remains a candidate; the bound alone does not establish joint reliability there.

An explicit suficient condition for the strict upper inequality in Eq. (35) is $M _ { \mathrm { + } } \geq M _ { \mathrm { s u f f } }$ and $0 \leq \delta < \delta _ { \sf * }$ <sub>★</sub>, where

$$
\delta _ { \star } : = \left[ \Phi _ { \sf s t d } \left( - \frac { \varepsilon } { \sqrt { 1 / 2 - \varepsilon } } \right) - \overline { { \tau } } _ { \tt G } \right] _ { + } ,\tag{36}
$$

$$
\omega _ { \mathrm { m i n } } : = \rho ^ { h - 1 } ,
$$

$$
M _ { \mathrm { s u f f } } : = \operatorname* { m a x } \left\{ 2 , \left\lceil \frac { \Gamma _ { \varepsilon } ( \rho ) } { \omega _ { \mathrm { m i n } } } \right\rceil , \left\lfloor \frac { \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } } { \zeta } \right\rfloor + 1 \right\} .\tag{37}
$$

Indeed, $\underline { { p } } _ { \mathrm { r e v } } ( M ) \leq \delta _ { \star } < 1 / 2$ at every calibrated window, with equality for each such window $M \geq M _ { \mathrm { { s u f f } } }$ . The lower inequality $\underline { { { p } } } _ { \mathrm { r e v } } ( M _ { - } ) \leq \delta$ remains required.

Interpreting the reversal tolerance. The tolerance � specifies the allowed fraction of relative distances at <sub>w</sub>hi<sub>c</sub>h th<sub>e re</sub>f<sub>erence or</sub>d<sub>er</sub>i<sub>ng reverses un</sub>d<sub>er un</sub>if<sub>orm samp</sub>li<sub>ng.</sub> F<sub>or examp</sub>l<sub>e,</sub> $\delta = 0 . 3 8$ <sub>perm</sub>it<sub>s reversa</sub>l<sub>s a</sub>t <sub>up</sub> t<sub>o</sub> 38% <sub>o</sub>f th<sub>e pos</sub>iti<sub>ons</sub> i<sub>n</sub> th<sub>e w</sub>i<sub>n</sub>d<sub>ow.</sub> P<sub>reserv</sub>i<sub>ng re</sub>f<sub>erence or</sub>d<sub>er</sub>i<sub>ngs re</sub>li<sub>a</sub>bl<sub>y ca</sub>ll<sub>s</sub> f<sub>or a muc</sub>h <sub>sma</sub>ll<sub>er</sub> t<sub>o</sub>l<sub>era</sub>t<sub>e</sub>d f<sub>rac</sub>ti<sub>on, so an upper res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on on</sub> � i<sub>s cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith th<sub>e</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d <sub>seman</sub>ti<sub>c-s</sub>t<sub>a</sub>bilit<sub>y requ</sub>i<sub>remen</sub>t<sub>.</sub> The ran<sub>g</sub>e in E<sub>q</sub>. (35) identifies tolerances for which this bound certifies an exclusion threshold. Its use still <sub>requ</sub>i<sub>res</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>ca</sub>lib<sub>ra</sub>ti<sub>on con</sub>diti<sub>ons an</sub>d th<sub>e spec</sub>ifi<sub>e</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> f<sub>am</sub>il<sub>y.</sub>

## B. Extensions to Other Failure Modes

Th<sub>e score represen</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A <sub>a</sub>l<sub>so suppor</sub>t<sub>s near–</sub>f<sub>ar compar</sub>i<sub>sons o</sub>f <sub>one</sub> k<sub>ey, excee</sub>d<sub>ance o</sub>f <sub>a</sub> fi<sub>xe</sub>d <sub>score</sub> b<sub>arr</sub>i<sub>er,</sub> f<sub>u</sub>ll<sub>y-</sub>hi<sub>g</sub>h <sub>pa</sub>i<sub>rw</sub>i<sub>se reversa</sub>l<sub>, an</sub>d <sub>a par</sub>ti<sub>a</sub>l<sub>-</sub>R<sub>o</sub>PE i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>ti<sub>on.</sub> E<sub>ac</sub>h <sub>resu</sub>lt b<sub>e</sub>l<sub>ow re</sub>t<sub>a</sub>i<sub>ns</sub> it<sub>s</sub> <sub>own even</sub>t <sub>an</sub>d <sub>ca</sub>lib<sub>ra</sub>ti<sub>on con</sub>diti<sub>ons.</sub> P<sub>roo</sub>f<sub>s are co</sub>ll<sub>ec</sub>t<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>

## B.1. Near–Far Ordering

## B.1.1. Extended Positional Diagnostic: Near–Far Ordering

Adjacent response measures local resolution, but it does not show whether a head preserves a directional <sub>no</sub>ti<sub>on</sub> <sub>o</sub>f di<sub>s</sub>t<sub>ance</sub> <sub>a</sub>t l<sub>arger</sub> <sub>sca</sub>l<sub>es.</sub> F<sub>or</sub> <sub>a</sub> h<sub>ea</sub>d <sub>or</sub> t<sub>as</sub>k th<sub>a</sub>t <sub>requ</sub>i<sub>res</sub> <sub>mono</sub>t<sub>one</sub> <sub>near</sub> <sub>pre</sub>f<sub>erence,</sub> <sub>we</sub> <sub>or</sub>i<sub>en</sub>t th<sub>e</sub> <sub>score so</sub> th<sub>a</sub>t <sub>a</sub> f<sub>ar</sub>th<sub>er occurrence s</sub>h<sub>ou</sub>ld <sub>rece</sub>i<sub>ve a</sub> l<sub>ower score.</sub> F<sub>or an</sub> i<sub>n</sub>t<sub>eger</sub> $M \geq 1$ <sub>,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y samp</sub>l<sub>e</sub> $m _ { \mathrm { n e a r } }$ f<sub>rom</sub> $\{ M + 1 , \ldots , 2 M \}$ <sub>an</sub>d $m _ { \mathrm { f a r } }$ f<sub>rom</sub> th<sub>e equa</sub>ll<sub>y</sub> l<sub>ong, more</sub> di<sub>s</sub>t<sub>an</sub>t i<sub>n</sub>t<sub>erva</sub>l $\{ 2 M + 1 , \ldots , 3 M \}$ . For a fi<sub>xe</sub>d k<sub>ey</sub> �<sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e or</sub>d<sub>er</sub>i<sub>ng-error pro</sub>b<sub>a</sub>bilit<sub>y</sub>

$$
p _ { \mathrm { o r d } } ( K ; M ) : = \mathbb { P } [ S _ { K } ( m _ { \mathrm { f a r } } ) > S _ { K } ( m _ { \mathrm { n e a r } } ) ] .\tag{38}
$$

A <sub>re</sub>li<sub>a</sub>bl<sub>e near–</sub>f<sub>ar or</sub>d<sub>er requ</sub>i<sub>res</sub> thi<sub>s pro</sub>b<sub>a</sub>bilit<sub>y</sub> t<sub>o</sub> b<sub>e sma</sub>ll<sub>.</sub> Wh<sub>en</sub> it i<sub>s c</sub>l<sub>ose</sub> t<sub>o one</sub> h<sub>a</sub>lf<sub>,</sub> th<sub>e</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d inequality is violated in half of the random comparisons under this audit protocol. We call this near–far order ambiguity; it is a score-level diagnostic of chance-like directional ordering in the two-interval comparison, <sub>no</sub>t <sub>a c</sub>l<sub>a</sub>i<sub>m</sub> th<sub>a</sub>t <sub>a</sub>ll <sub>pos</sub>iti<sub>ona</sub>l i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> h<sub>as</sub> di<sub>sappeare</sub>d<sub>.</sub>

Thi<sub>s</sub> i<sub>s a one-s</sub>id<sub>e</sub>d di<sub>agnos</sub>ti<sub>c: a sma</sub>ll $p _ { \mathrm { { o r d } } }$ i<sub>s necessary</sub> b<sub>u</sub>t <sub>no</sub>t <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t f<sub>or re</sub>li<sub>a</sub>bl<sub>e near pre</sub>f<sub>erence,</sub> b<sub>ecause</sub> <sub>a</sub> <sub>pos</sub>iti<sub>on-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>score</sub> <sub>a</sub>l<sub>so</sub> <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>even</sub>t i<sub>mposs</sub>ibl<sub>e.</sub> Fi<sub>xe</sub>d<sub>-o</sub>f<sub>se</sub>t<sub>,</sub> <sub>per</sub>i<sub>o</sub>di<sub>c,</sub> di<sub>agona</sub>l<sub>,</sub> <sub>an</sub>d <sub>o</sub>th<sub>er non-mono</sub>t<sub>one pa</sub>tt<sub>erns may</sub> lik<sub>ew</sub>i<sub>se rema</sub>i<sub>n</sub> d<sub>e</sub>t<sub>ec</sub>t<sub>a</sub>bl<sub>e.</sub>

## B.1.2. Near–Far Order Ambiguity

W<sub>e</sub> <sub>app</sub>l<sub>y</sub> th<sub>e</sub> t<sub>wo-</sub>i<sub>n</sub>t<sub>erva</sub>l <sub>exper</sub>i<sub>men</sub>t f<sub>rom</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>1<sub>.</sub>1 t<sub>o</sub> <sub>one</sub> fi<sub>xe</sub>d k<sub>ey.</sub> Th<sub>e</sub> th<sub>eorem</sub> b<sub>e</sub>l<sub>ow</sub> <sub>concerns</sub> th<sub>e</sub> <sub>asymp</sub>t<sub>o</sub>ti<sub>c</sub> f<sub>u</sub>ll<sub>y-</sub>hi<sub>g</sub>h <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> <sub>reg</sub>i<sub>me:</sub> <sub>w</sub>h<sub>en</sub> <sub>a</sub>ll <sub>ro</sub>t<sub>a</sub>ti<sub>ng</sub> f<sub>requenc</sub>i<sub>es</sub> <sub>en</sub>t<sub>er</sub> th<sub>e</sub> <sub>cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h b<sub>an</sub>d <sub>an</sub>d th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>ca</sub>lib<sub>ra</sub>ti<sub>on errors van</sub>i<sub>s</sub>h<sub>,</sub> th<sub>e near–</sub>f<sub>ar or</sub>d<sub>er</sub>i<sub>ng error approac</sub>h<sub>es c</sub>h<sub>ance.</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>1<sub>.</sub>3 <sub>g</sub>i<sub>ves</sub> th<sub>e prec</sub>i<sub>se ca</sub>lib<sub>ra</sub>ti<sub>on con</sub>diti<sub>ons.</sub>

Theorem 2 (Near–far ordering approaches chance under full calibration). Fix afully rotary head together with fixed query and key content states that induce a nonzero coeficient vector �. Independently draw �<sub>near</sub> uniformlyfrom $\{ M + 1 , \ldots , 2 M \}$ and $m _ { \mathrm { f a r } }$ uniformlyfrom $\{ 2 M + 1 , \ldots , 3 M \}$ . Suppose the two interval score laws lie in the asymptotic fully-high calibration regime of Appendix B.1.3. Then

$$
p _ { \mathrm { o r d } } ( a ; M ) : = \mathbb { P } [ X _ { \mathcal { F } } ^ { a } ( m _ { \mathrm { f a r } } ) > X _ { \mathcal { F } } ^ { a } ( m _ { \mathrm { n e a r } } ) ] \longrightarrow \frac 1 2 \qquad ( M \to \infty ) .\tag{39}
$$

Th<sub>e s</sub>l<sub>ow</sub>l<sub>y ro</sub>t<sub>a</sub>ti<sub>ng par</sub>t <sub>can</sub> i<sub>n</sub>iti<sub>a</sub>ll<sub>y suppor</sub>t <sub>a s</sub>i<sub>gne</sub>d <sub>mean a</sub>d<sub>van</sub>t<sub>age</sub> f<sub>or</sub> th<sub>e nearer</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>.</sub> A<sub>s</sub> th<sub>ose</sub> f<sub>requenc</sub>i<sub>es en</sub>t<sub>er</sub> th<sub>e cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h b<sub>an</sub>d<sub>,</sub> P<sub>ropos</sub>iti<sub>on</sub> 1 d<sub>r</sub>i<sub>ves</sub> b<sub>o</sub>th i<sub>n</sub>t<sub>erva</sub>l <sub>means</sub> t<sub>owar</sub>d <sub>zero an</sub>d b<sub>o</sub>th <sub>var</sub>i<sub>ances</sub> t<sub>owar</sub>d th<sub>e same</sub> h<sub>a</sub>lf<sub>-energy va</sub>l<sub>ue.</sub> U<sub>n</sub>d<sub>er</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>s</sub>h<sub>ape ca</sub>lib<sub>ra</sub>ti<sub>on,</sub> th<sub>e</sub> t<sub>wo scores</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>approac</sub>h th<sub>e same</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on, so</sub> th<sub>e</sub> f<sub>ar</sub>th<sub>er score excee</sub>d<sub>s</sub> th<sub>e nearer one w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y one</sub> h<sub>a</sub>lf<sub>.</sub> Th<sub>e</sub> <sub>van</sub>i<sub>s</sub>hi<sub>ng s</sub>h<sub>ape error rema</sub>i<sub>ns su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>ve: equa</sub>li<sub>ze</sub>d <sub>momen</sub>t<sub>s a</sub>l<sub>one</sub> d<sub>o no</sub>t <sub>ma</sub>k<sub>e a</sub> fi<sub>xe</sub>d fi<sub>n</sub>it<sub>e</sub> t<sub>r</sub>i<sub>gonome</sub>t<sub>r</sub>i<sub>c</sub> sum Gaussian.

## B.1.3. Two-Interval Calibration Error

F<sub>or</sub> <sub>an</sub> i<sub>n</sub>t<sub>eger</sub> <sub>s</sub>hift $s ,$ d<sub>e</sub>fi<sub>ne</sub>

$$
a _ { n } ^ { [ s ] } : = a _ { n } e ^ { i s \omega _ { n } } , \qquad X _ { \mathcal { F } } ^ { a } ( s + t ) = X _ { \mathcal { F } } ^ { a ^ { [ s ] } } ( t ) , \qquad U ( a ^ { [ s ] } ) = U ( a ) .\tag{40}
$$

Th<sub>e</sub> <sub>near</sub> <sub>an</sub>d f<sub>ar</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s</sub> <sub>are</sub> l<sub>eng</sub>th<sub>-</sub>� <sub>w</sub>i<sub>n</sub>d<sub>ows</sub> <sub>genera</sub>t<sub>e</sub>d b<sub>y</sub> $a ^ { [ M + 1 ] }$ <sub>an</sub>d $a ^ { [ 2 M + 1 ] }$ <sub>,</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> D<sub>e</sub>fi<sub>ne</sub>

$$
\tau _ { \mathrm { n e a r } } ( a ; M ) : = \Delta _ { \mathrm { H , G } } ( a ^ { [ M + 1 ] } ; M ) , \qquad \tau _ { \mathrm { f a r } } ( a ; M ) : = \Delta _ { \mathrm { H , G } } ( a ^ { [ 2 M + 1 ] } ; M ) ,\tag{41}
$$

<sub>an</sub>d

$$
\tau _ { \mathrm { o r d } } ( a ; M , \varepsilon ) : = \tau _ { \mathrm { n e a r } } ( a ; M ) + \tau _ { \mathrm { f a r } } ( a ; M ) + 2 c _ { \mathrm { G } } ( \varepsilon ) .\tag{42}
$$

F<sub>or</sub> fi<sub>xe</sub>d fi<sub>n</sub>it<sub>e</sub> $h ,$ set

$$
\varepsilon _ { M } : = { \frac { 2 C _ { \mathrm { F } } } { ( 1 - \rho ) M \omega _ { h - 1 } } }\tag{43}
$$

f<sub>or a</sub>ll <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge</sub> �<sub>.</sub> Th<sub>e asymp</sub>t<sub>o</sub>ti<sub>c</sub> f<sub>u</sub>ll<sub>y-</sub>hi<sub>g</sub>h <sub>ca</sub>lib<sub>ra</sub>ti<sub>on reg</sub>i<sub>me</sub> i<sub>n</sub> Th<sub>eorem</sub> 2 i<sub>mposes</sub> t<sub>wo requ</sub>i<sub>re-</sub> ments. Both standardized translated laws must satisf<sub>y</sub> E<sub>q</sub>. (19) with $\varepsilon = \varepsilon _ { M }$ <sub>,</sub> <sub>an</sub>d $\tau _ { \mathrm { o r d } } ( a ; M , \varepsilon _ { M } ) \to 0$ <sub>.</sub> Wh<sub>e</sub>n available valid held-out u er bounds on the two sha e errors in E . (41) ma be used in E . (42). This t<sub>o</sub>l<sub>erance c</sub>h<sub>o</sub>i<sub>ce ma</sub>k<sub>es</sub> th<sub>e par</sub>titi<sub>on</sub> f<sub>u</sub>ll<sub>y</sub> hi<sub>g</sub>h b<sub>ecause</sub> $\Gamma _ { \varepsilon _ { M } } ( \rho ) = M \omega _ { h - 1 } , s _ { 0 } \mathcal { H } _ { \varepsilon _ { M } } ( M ) = \mathcal { F }$

## B.2. Score-Barrier Stability

## B.2.1. Single-Key Exceedance Against a Score Barrier

W<sub>e use rare excee</sub>d<sub>ance o</sub>f <sub>a</sub> fi<sub>xe</sub>d b<sub>arr</sub>i<sub>er as a s</sub>i<sub>ng</sub>l<sub>e-</sub>k<sub>ey score-</sub>l<sub>eve</sub>l <sub>cer</sub>tifi<sub>ca</sub>t<sub>e.</sub> It <sub>acqu</sub>i<sub>res a seman</sub>ti<sub>c ran</sub>ki<sub>ng</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>ti<sub>on on</sub>l<sub>y un</sub>d<sub>er</sub> th<sub>e w</sub>i<sub>nner-</sub>b<sub>arr</sub>i<sub>er cons</sub>t<sub>ruc</sub>ti<sub>on</sub> b<sub>e</sub>l<sub>ow an</sub>d it<sub>s a</sub>dditi<sub>ona</sub>l t<sub>a</sub>il <sub>con</sub>diti<sub>ons.</sub>

Fi<sub>x</sub> th<sub>e query,</sub> l<sub>ayer,</sub> h<sub>ea</sub>d<sub>, con</sub>t<sub>en</sub>t <sub>s</sub>t<sub>a</sub>t<sub>es, an</sub>d <sub>one</sub> k<sub>ey</sub> � <sub>w</sub>ith <sub>nonzero</sub> R<sub>o</sub>PE <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>or</sub> �<sub>.</sub> F<sub>or a</sub> <sub>cer</sub>tifi<sub>e</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ow,</sub> d<sub>raw</sub> $\mathbf { m } \sim \mathrm { U n i f } \{ 0 , \dots , M - 1 \}$ <sub>an</sub>d f<sub>or a</sub> fi<sub>xe</sub>d <sub>score</sub> b<sub>arr</sub>i<sub>er</sub> $S \in \mathbb { R } .$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
Y _ { b } : = X _ { \mathcal { F } } ^ { b } ( \mathbf { m } ) , \qquad p _ { \mathrm { e x c } } ( b ; S , M ) : = \mathbb { P } [ Y _ { b } > S ] .\tag{44}
$$

Write $Y _ { N } : = X _ { N } ^ { b } ( { \mathbf { m } } )$ <sub>an</sub>d $Y _ { H } : = X _ { H } ^ { b } ( { \mathbf { m } } )$ <sub>.</sub> G<sub>u</sub>id<sub>e</sub>d b<sub>y</sub> P<sub>ropos</sub>iti<sub>on</sub> 1<sub>, re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e non-</sub>hi<sub>g</sub>h <sub>momen</sub>t<sub>s an</sub>d th<sub>e</sub> <sub>cross-</sub>b<sub>an</sub>d <sub>covar</sub>i<sub>ance,</sub> b<sub>u</sub>t <sub>rep</sub>l<sub>ace</sub> th<sub>e</sub> <sub>cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h <sub>mean</sub> <sub>an</sub>d <sub>var</sub>i<sub>ance</sub> b<sub>y</sub> 0 <sub>an</sub>d $U _ { H } ( b ) ^ { 2 } / 2$ <sub>.</sub> Thi<sub>s</sub> <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an excee</sub>d<sub>ance pre</sub>di<sub>c</sub>t<sub>or</sub>

$$
\begin{array} { c } { \widehat { \sigma } _ { b } ^ { 2 } : = \operatorname { V a r } _ { M } ( Y _ { N } ) + \displaystyle \frac { 1 } { 2 } U _ { H } ( b ) ^ { 2 } + 2 \operatorname { C o v } _ { M } ( Y _ { N } , Y _ { H } ) , } \\ { \widehat { p } _ { \mathrm { e x c } } ( b ; S , M ) : = \displaystyle \Phi _ { \mathrm { s t d } } \left( \frac { \big \langle X _ { N } ^ { b } \big \rangle _ { M } - S } { \widehat { \sigma } _ { b } } \right) , } \end{array}\tag{45}
$$

<sub>w</sub>h<sub>ere</sub> $\mathrm { C o v } _ { M }$ i<sub>s covar</sub>i<sub>ance un</sub>d<sub>er</sub> th<sub>e same un</sub>if<sub>orm w</sub>i<sub>n</sub>d<sub>ow an</sub>d $\widehat { \sigma } _ { b }$ i<sub>s</sub> th<sub>e pos</sub>iti<sub>ve square roo</sub>t<sub>.</sub> L<sub>e</sub>t $\tau _ { \mathrm { e x c } } ( b ; S , M )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>ca</sub>lib<sub>ra</sub>ti<sub>on error</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>2<sub>.</sub>2<sub>.</sub> It <sub>com</sub>bi<sub>nes</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an-s</sub>h<sub>ape</sub> di<sub>screpancy</sub> <sub>o</sub>f th<sub>e</sub> f<sub>u</sub>ll <sub>score</sub> <sub>w</sub>ith th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>b<sub>an</sub>d <sub>momen</sub>t <sub>rep</sub>l<sub>acemen</sub>t <sub>error;</sub> th<sub>e</sub> <sub>resu</sub>lt b<sub>e</sub>l<sub>ow</sub> i<sub>s</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>ve</sub> <sub>w</sub>h<sub>en</sub> thi<sub>s</sub> <sub>measura</sub>bl<sub>e quan</sub>tit<sub>y</sub> i<sub>s sma</sub>ll<sub>.</sub>

Lemma B.1 (Calibrated single-key score exceedance). For every certified �, nonzero $b ,$ and $S \in \mathbb { R }$ in the non-degenerate calibration regime ofAppendix B.2.2,

$$
| p _ { \mathrm { e x c } } ( b ; S , M ) - \widehat { p } _ { \mathrm { e x c } } ( b ; S , M ) | \le \tau _ { \mathrm { e x c } } ( b ; S , M ) .\tag{46}
$$

Let $u _ { \mathrm { { e x c } } } ( b ; S , M , \delta )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e exp</sub>li<sub>c</sub>it<sub>, coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-spec</sub>ifi<sub>c ce</sub>ili<sub>ng g</sub>i<sub>ven</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>2<sub>.</sub>3<sub>.</sub> It<sub>s</sub> f<sub>ormu</sub>l<sub>a</sub> i<sub>s</sub> d<sub>e</sub>f<sub>erre</sub>d b<sub>ecause on</sub>l<sub>y</sub> th<sub>e resu</sub>lti<sub>ng</sub> b<sub>oun</sub>d<sub>, ra</sub>th<sub>er</sub> th<sub>an</sub> th<sub>e qua</sub>d<sub>ra</sub>ti<sub>c</sub> i<sub>nvers</sub>i<sub>on use</sub>d t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub> it<sub>,</sub> i<sub>s nee</sub>d<sub>e</sub>d h<sub>ere.</sub>

Corollary B.1 (Small score-barrier exceedance imposes a certified-high norm-share ceiling). Under the hypotheses of Lemma B.1, assume $U _ { N } ( b ) U _ { H } ( b ) > 0$ and $0 < \delta + \tau _ { \mathrm { e x c } } ( b ; S , M ) < 1 / 2 .$ . Then

$$
p _ { \mathrm { e x c } } ( b ; S , M ) \leq \delta \quad \Longrightarrow \quad r _ { H } ( b ; M ) \leq u _ { \mathrm { e x c } } ( b ; S , M , \delta ) < 1 .\tag{47}
$$

Thi<sub>s</sub> i<sub>s</sub> <sub>exac</sub>tl<sub>y</sub> th<sub>e</sub> <sub>same</sub> $r _ { H } ( b ; M ) = U _ { H } ( b ) / U ( b )$ <sub>cons</sub>t<sub>ra</sub>i<sub>ne</sub>d f<sub>rom</sub> b<sub>e</sub>l<sub>ow</sub> b<sub>y</sub> th<sub>e</sub> l<sub>oca</sub>l<sub>-response</sub> fl<sub>oor w</sub>h<sub>en</sub> it i<sub>s</sub> <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> �<sub>.</sub> H<sub>ence,</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> <sub>an</sub>d i<sub>n</sub>t<sub>er</sub>i<sub>or</sub> h<sub>ypo</sub>th<sub>eses</sub> <sub>a</sub>b<sub>ove,</sub> <sub>a</sub> l<sub>oca</sub>l<sub>-response</sub> fl<sub>oor</sub> <sub>a</sub>b<sub>ove</sub> $u _ { \mathrm { e x c } }$ f<sub>orces</sub> $p _ { \mathrm { e x c } } ( b ; S , M ) > \delta$ <sub>.</sub> Th<sub>e en</sub>d<sub>po</sub>i<sub>n</sub>t <sub>cases are</sub> dif<sub>eren</sub>t<sub>:</sub> ${ \cal U } _ { H } ( b ) = 0$ <sub>g</sub><sup>i</sup>ves $r _ { H } = 0$ di<sub>rec</sub>tl<sub>y, w</sub>h<sub>ereas a</sub>t $U _ { N } ( b ) = 0$ <sub>rare excee</sub>d<sub>ance o</sub>f <sub>a su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> hi<sub>g</sub>h b<sub>arr</sub>i<sub>er nee</sub>d <sub>no</sub>t i<sub>mp</sub>l<sub>y any ce</sub>ili<sub>ng</sub> b<sub>e</sub>l<sub>ow one.</sub>

Winner-barrier interpretation. Let K be a finite key bank at the reference placement $m = 0$ , c<sup>h</sup>oose an<sub>y</sub> <sub>re</sub>f<sub>erence w</sub>i<sub>nner, an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
K _ { + } \in \arg \operatorname* { m a x } _ { K _ { j } \in \mathcal { K } } X _ { \mathcal { F } } ^ { b _ { j } } ( 0 ) , \qquad S _ { \operatorname* { m a x } } : = X _ { \mathcal { F } } ^ { b _ { + } } ( 0 ) , \qquad \alpha _ { + } ( S _ { \operatorname* { m a x } } ; M ) : = \mathbb { P } [ X _ { \mathcal { F } } ^ { b _ { + } } ( \mathbf { m } ) > S _ { \operatorname* { m a x } } ] .\tag{48}
$$

For every compet<sup>i</sup>tor $K _ { j } \neq K _ { + }$ <sub>, even</sub>t i<sub>nc</sub>l<sub>us</sub>i<sub>on g</sub>i<sub>ves</sub>

$$
p _ { \mathrm { r e v } } ( b _ { + } , b _ { j } ; M ) : = \mathbb { P } [ X _ { \mathcal { F } } ^ { b _ { j } } ( \mathbf { m } ) > X _ { \mathcal { F } } ^ { b _ { + } } ( \mathbf { m } ) ] \geq \left[ p _ { \mathrm { e x c } } ( b _ { j } ; S _ { \operatorname* { m a x } } , M ) - \alpha _ { + } ( S _ { \operatorname* { m a x } } ; M ) \right] _ { + } .\tag{49}
$$

C<sub>o</sub>n<sub>seque</sub>ntl<sub>y,</sub> fix $q > 0$ a<sup>nd</sup> suppose $\alpha _ { + } ( S _ { \mathrm { m a x } } ; M ) \le \delta - q$ <sub>.</sub> If <sub>every compe</sub>tit<sub>or</sub> li<sub>es</sub> i<sub>n</sub> th<sub>e ca</sub>lib<sub>ra</sub>t<sub>e</sub>d i<sub>n</sub>t<sub>er</sub>i<sub>or</sub> <sub>reg</sub>i<sub>me a</sub>b<sub>ove an</sub>d it<sub>s</sub> l<sub>oca</sub>l<sub>-response</sub> fl<sub>oor excee</sub>d<sub>s</sub> $u _ { \mathrm { e x c } } ( b _ { j } ; S _ { \mathrm { m a x } } , M , \delta )$ <sub>,</sub> th<sub>en eac</sub>h <sub>compe</sub>tit<sub>or</sub> h<sub>as</sub> th<sub>e marg</sub>i<sub>na</sub>l gua<sup>r</sup>a<sup>nt</sup>ee $p _ { \mathrm { r e v } } ( b _ { + } , b _ { j } ; M ) > q$ <sub>.</sub> Th<sub>us c</sub>h<sub>oos</sub>i<sub>ng</sub> � <sub>as</sub> th<sub>e</sub> l<sub>arges</sub>t <sub>re</sub>f<sub>erence score exposes a compe</sub>tit<sub>or-w</sub>i<sub>se</sub> <sub>marg</sub>i<sub>na</sub>l <sub>reversa</sub>l <sub>r</sub>i<sub>s</sub>k <sub>across</sub> th<sub>e</sub> b<sub>an</sub>k <sub>once a</sub>ll <sub>can</sub>did<sub>a</sub>t<sub>e-spec</sub>ifi<sub>c</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s are</sub> i<sub>ncompa</sub>tibl<sub>e.</sub> It d<sub>oes no</sub>t <sub>asser</sub>t th<sub>a</sub>t <sub>a</sub>ll <sub>compe</sub>tit<sub>ors</sub> <sub>reverse</sub> <sub>a</sub>t th<sub>e</sub> <sub>same</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance,</sub> <sub>nor</sub> d<sub>oes</sub> th<sub>e</sub> <sub>max</sub>i<sub>mum-</sub>b<sub>arr</sub>i<sub>er</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>a</sub>l<sub>one</sub> i<sub>mp</sub>l<sub>y</sub> th<sub>e resu</sub>lt<sub>:</sub> th<sub>e ca</sub>lib<sub>ra</sub>ti<sub>on,</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>cross</sub>i<sub>ng, an</sub>d <sub>w</sub>i<sub>nner-</sub>t<sub>a</sub>il <sub>con</sub>diti<sub>on mus</sub>t <sub>a</sub>ll h<sub>o</sub>ld<sub>.</sub>

Th<sub>e</sub> <sub>score-</sub>b<sub>arr</sub>i<sub>er</sub> <sub>ce</sub>ili<sub>ng</sub> <sub>a</sub>b<sub>ove</sub> <sub>app</sub>li<sub>es</sub> i<sub>n</sub> th<sub>e</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d i<sub>n</sub>t<sub>er</sub>i<sub>or</sub> <sub>reg</sub>i<sub>me.</sub> Th<sub>e</sub> f<sub>u</sub>ll<sub>y-</sub>hi<sub>g</sub>h <sub>en</sub>d<sub>po</sub>i<sub>n</sub>t i<sub>s</sub> <sub>covere</sub>d b<sub>y</sub> th<sub>e or</sub>d<sub>ere</sub>d<sub>-pa</sub>i<sub>r resu</sub>lt i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>3<sub>.</sub>

## B.2.2. Calibration Quantities

For the fixed score in §B.2.1, write $Y _ { N } : = X _ { N } ^ { b } ( { \mathbf m } ) , Y _ { H } : = X _ { H } ^ { b } ( { \mathbf m } )$ <sub>,</sub> <sub>an</sub>d <sub>se</sub>t

$$
\begin{array} { r l } & { \mu _ { b } : = \big \langle X _ { \mathcal { F } } ^ { b } \big \rangle _ { M } , } \\ & { \mu _ { N } : = \big \langle X _ { N } ^ { b } \big \rangle _ { M } , } \\ & { \kappa _ { N H } : = \big \langle ( X _ { N } ^ { b } - \mu _ { N } ) ( X _ { H } ^ { b } - \big \langle X _ { H } ^ { b } \big \rangle _ { M } ) \big \rangle _ { M } , } \\ & { \widehat { \sigma } _ { b } ^ { 2 } : = \sigma _ { N } ^ { 2 } + \displaystyle \frac { 1 } { 2 } U _ { H } ( b ) ^ { 2 } + 2 \kappa _ { N H } . } \end{array}
$$

$$
\begin{array} { r } { \sigma _ { b } ^ { 2 } : = \mathrm { V a r } _ { M } ( X _ { \mathcal { F } } ^ { b } ) , } \\ { \sigma _ { N } ^ { 2 } : = \mathrm { V a r } _ { M } ( X _ { N } ^ { b } ) , } \end{array}\tag{50}
$$

Th<sub>us</sub> $\widehat { \sigma } _ { b } ^ { 2 }$ is exactl<sub>y</sub> the moment-calibrated variance <sub>p</sub>rox<sub>y</sub> in E<sub>q</sub>. (45). The non-de<sub>g</sub>enerate calibration re<sub>g</sub>ime <sub>use</sub>d in L<sub>e</sub>mma B<sub>.</sub>1 i<sub>s</sub>

$$
\sigma _ { b } ^ { 2 } > 0 , \qquad \widehat { \sigma } _ { b } ^ { 2 } > \varepsilon U _ { H } ( b ) ^ { 2 } .\tag{51}
$$

I<sub>n</sub> thi<sub>s</sub> <sub>reg</sub>i<sub>me,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> f<sub>u</sub>ll<sub>-score</sub> G<sub>auss</sub>i<sub>an-s</sub>h<sub>ape</sub> di<sub>screpancy</sub>

$$
\Delta _ { \mathrm { e x c } } ( b ; M ) : = \operatorname* { s u p } _ { x \in \mathbb { R } } \left. \mathbb { P } \left[ \frac { Y _ { b } - \mu _ { b } } { \sigma _ { b } } \leq x \right] - \Phi _ { \mathrm { s t d } } ( x ) \right.\tag{52}
$$

<sub>an</sub>d th<sub>e var</sub>i<sub>ance</sub> fl<sub>oor an</sub>d t<sub>o</sub>t<sub>a</sub>l <sub>ca</sub>lib<sub>ra</sub>ti<sub>on error</sub>

$$
\begin{array} { r l r } {  { s _ { - } ( b ; M ) : = \sqrt { \widehat { \sigma } _ { b } ^ { 2 } - \varepsilon U _ { H } ( b ) ^ { 2 } } , } } \\ & { } & { \tau _ { \mathrm { e x c } } ( b ; S , M ) : = \Delta _ { \mathrm { e x c } } ( b ; M ) + \frac { 1 } { \sqrt { 2 \pi } } [ \frac { \varepsilon U _ { H } ( b ) } { s _ { - } ( b ; M ) } + \frac { | S - \mu _ { N } | \varepsilon U _ { H } ( b ) ^ { 2 } } { s _ { - } ( b ; M ) \widehat { \sigma } _ { b } ( s _ { - } ( b ; M ) + \widehat { \sigma } _ { b } ) } ] . } \end{array}\tag{53}
$$

The first term measures the deviation of the full score law from Gaussianity; the remaining terms account f<sub>or</sub> <sub>rep</sub>l<sub>ac</sub>i<sub>ng</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>b<sub>an</sub>d <sub>momen</sub>t<sub>s</sub> <sub>us</sub>i<sub>ng</sub> P<sub>ropos</sub>iti<sub>on</sub> 1<sub>.</sub>

## B.2.3. Explicit Ceiling Candidate

F<sub>or</sub> th<sub>e</sub> i<sub>n</sub>t<sub>er</sub>i<sub>or case</sub> $U _ { N } ( b ) U _ { H } ( b ) > 0$ <sub>an</sub>d $0 < \delta + \tau _ { \mathrm { e x c } } ( b ; S , M ) < 1 / 2$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\begin{array} { r l } & { \lambda _ { S } : = \frac { S - \mu _ { N } } { U _ { N } ( b ) } , \qquad \qquad \quad \beta _ { N } : = \frac { \sigma _ { N } ^ { 2 } } { U _ { N } ( b ) ^ { 2 } } , } \\ & { \gamma _ { N H } : = \frac { \kappa _ { N H } } { U _ { N } ( b ) U _ { H } ( b ) } , \qquad \quad \qquad \quad \xi _ { \mathrm { e x c } } : = \Phi _ { \mathrm { s t d } } ^ { - 1 } ( 1 - \delta - \tau _ { \mathrm { e x c } } ( b ; S , M ) ) . } \end{array}\tag{54}
$$

Th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>exp</sub>li<sub>c</sub>it <sub>ce</sub>ili<sub>ng</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>s</sub>

$$
\begin{array} { c } { { \overline { { { t } } } _ { \mathrm { e x c } } : = - 2 \gamma _ { N H } + \sqrt { 4 \gamma _ { N H } ^ { 2 } + 2 \left( \displaystyle \frac { \lambda _ { S } ^ { 2 } } { \xi _ { \mathrm { e x c } } ^ { 2 } } - \beta _ { N } \right) } , } } \\ { { u _ { \mathrm { e x c } } ( b ; S , M , \delta ) : = \displaystyle \frac { \overline { { { t } } } _ { \mathrm { e x c } } } { \sqrt { 1 + \overline { { { t } } } _ { \mathrm { e x c } } ^ { 2 } } } . } } \end{array}\tag{55}
$$

It<sub>s ra</sub>di<sub>can</sub>d <sub>nee</sub>d <sub>no</sub>t b<sub>e nonnega</sub>ti<sub>ve</sub> f<sub>or ar</sub>bit<sub>rary</sub> i<sub>npu</sub>t<sub>s.</sub> U<sub>n</sub>d<sub>er</sub> th<sub>e an</sub>t<sub>ece</sub>d<sub>en</sub>t <sub>o</sub>f C<sub>oro</sub>ll<sub>ary</sub> B<sub>.</sub>1<sub>,</sub> th<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d hi<sub>g</sub>h<sub>-</sub>t<sub>o-non-</sub>hi<sub>g</sub>h <sub>norm</sub> <sub>ra</sub>ti<sub>o</sub> i<sub>s</sub> f<sub>eas</sub>ibl<sub>e</sub> f<sub>or</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> <sub>qua</sub>d<sub>ra</sub>ti<sub>c;</sub> thi<sub>s</sub> <sub>guaran</sub>t<sub>ees</sub> th<sub>a</sub>t th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>s</sub> <sub>rea</sub>l <sub>an</sub>d <sub>pos</sub>iti<sub>ve.</sub>

## B.3. Fully-High Pairwise Reversal

F<sub>or an or</sub>d<sub>ere</sub>d k<sub>ey pa</sub>i<sub>r, we ca</sub>ll th<sub>e even</sub>t th<sub>a</sub>t th<sub>e</sub> fi<sub>rs</sub>t <sub>score</sub> f<sub>a</sub>ll<sub>s</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e secon</sub>d <sub>a</sub> di<sub>rec</sub>ti<sub>ona</sub>l <sub>reversa</sub>l<sub>;</sub> it i<sub>s seman</sub>ti<sub>c w</sub>h<sub>en</sub> th<sub>e</sub> fi<sub>rs</sub>t k<sub>ey</sub> i<sub>s</sub> th<sub>e pre</sub>f<sub>erre</sub>d <sub>can</sub>did<sub>a</sub>t<sub>e.</sub>

Theorem 3 (Fully-high pairwise reversal approaches chance). For a fully rotary head, fix a query and an ordered pair of keys with coeficient vectors $a , b \in \mathbb { C } ^ { h }$ such that � $: = a - b \neq 0 .$ . For any $0 < \varepsilon < 1 / 2$ and any certified window satisfying $\mathcal { H } _ { \varepsilon } ( M ) = \mathcal { F } _ { * }$ , draw $\mathbf { m } \sim \mathrm { U n i f } \{ 0 , \dots , M - 1 \}$ . If the Gaussian-shape condition in Eq. (19) holds for $z = d = a - b$ with tolerance $\tau _ { \mathrm { H , G } } ( d ; M )$ , then

$$
p _ { \mathrm { r e v } } ( a , b ; M ) : = \mathbb { P } \big [ X _ { \mathcal { F } } ^ { a } ( \mathbf { m } ) < X _ { \mathcal { F } } ^ { b } ( \mathbf { m } ) \big ] = \mathbb { P } \big [ X _ { \mathcal { F } } ^ { a - b } ( \mathbf { m } ) < 0 \big ] ,
$$

$$
\left| p _ { \mathrm { r e v } } ( a , b ; M ) - \frac { 1 } { 2 } \right| \leq \tau _ { \mathrm { H , G } } ( a - b ; M ) + c _ { \mathrm { G } } ( \varepsilon ) .\tag{56}
$$

Here $c _ { \mathsf { G } } ( \varepsilon )$ is the universal moment-replacement term defined in Eq. (21). For fixed finite ℎ, define $\varepsilon _ { M } : =$ $2 C _ { \mathrm { F } } / ( ( 1 - \rho ) M \omega _ { h - 1 } )$ for all suficiently large �. If the same shape calibration holds for the resulting fully-high partitions and $\tau _ { \mathrm { H , G } } ( a - b ; M ) \to 0 _ { \mathrm { \therefore } }$ , then

$$
\operatorname * { l i m } _ { M  \infty } p _ { \mathrm { r e v } } ( a , b ; M ) = \frac { 1 } { 2 } .\tag{57}
$$

F<sub>u</sub>ll<sub>y-</sub>hi<sub>g</sub>h <sub>mem</sub>b<sub>ers</sub>hi<sub>p con</sub>t<sub>ro</sub>l<sub>s momen</sub>t<sub>s</sub> b<sub>u</sub>t d<sub>oes no</sub>t b<sub>y</sub> it<sub>se</sub>lf <sub>ma</sub>k<sub>e a</sub> fi<sub>xe</sub>d fi<sub>n</sub>it<sub>e</sub> t<sub>r</sub>i<sub>gonome</sub>t<sub>r</sub>i<sub>c sum</sub> G<sub>auss</sub>i<sub>an or guaran</sub>t<sub>ee c</sub>h<sub>ance-</sub>l<sub>eve</sub>l <sub>reversa</sub>l <sub>a</sub>t <sub>a</sub> fi<sub>n</sub>it<sub>e w</sub>i<sub>n</sub>d<sub>ow.</sub> Th<sub>e</sub> th<sub>eorem prov</sub>id<sub>es a ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>error</sub> b<sub>oun</sub>d <sub>an</sub>d <sub>an</sub> <sub>asymp</sub>t<sub>o</sub>ti<sub>c</sub> <sub>conc</sub>l<sub>us</sub>i<sub>on</sub> <sub>on</sub>l<sub>y</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>s</sub>h<sub>ape</sub> <sub>con</sub>diti<sub>on.</sub> Th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> th<sub>eorem</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>com</sub>bi<sub>nes</sub> th<sub>e</sub> l<sub>oca</sub>l<sub>-response</sub> fl<sub>oor w</sub>ith <sub>a conserva</sub>ti<sub>ve</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow reversa</sub>l fl<sub>oor</sub> f<sub>or</sub> th<sub>e same or</sub>d<sub>ere</sub>d <sub>score</sub> mar<sub>g</sub><sup>i</sup>n.

Scope ofthe asymptotic premise. For fixed finite ℎ and fixed nonzero $d ,$ th<sub>e score sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\begin{array} { r } { | X _ { \mathcal { F } } ^ { d } ( m ) | \leq \sum _ { n } | d _ { n } | } \end{array}$ <sub>a</sub>t <sub>ever re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance.</sub> It<sub>s</sub> li<sub>m</sub>iti<sub>n var</sub>i<sub>ance un</sub>d<sub>er</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>-</sub>hi h <sub>momen</sub>t b<sub>oun</sub>d<sub>s</sub> i<sub>s</sub> $U ( d ) ^ { 2 } / 2 > 0$ <sub>.</sub> Th<sub>us</sub> it<sub>s s</sub>t<sub>an</sub>d<sub>ar</sub>di<sub>ze</sub>d <sub>suppor</sub>t <sub>rema</sub>i<sub>ns</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>, w</sub>hil<sub>e a</sub> G<sub>auss</sub>i<sub>an</sub> h<sub>as pos</sub>iti<sub>ve</sub> t<sub>a</sub>il<sub>s</sub> b<sub>eyon</sub>d th<sub>a</sub>t b<sub>oun</sub>d<sub>.</sub> Th<sub>e a</sub>ll<sub>-</sub> th<sub>res</sub>h<sub>o</sub>ld G<sub>auss</sub>i<sub>an-s</sub>h<sub>ape</sub> di<sub>screpancy</sub> th<sub>ere</sub>f<sub>ore canno</sub>t <sub>van</sub>i<sub>s</sub>h <sub>so</sub>l<sub>e</sub>l<sub>y</sub> b<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> � i<sub>n</sub> thi<sub>s</sub> fi<sub>xe</sub>d<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>se</sub>tti<sub>ng.</sub> Th<sub>e</sub> <sub>van</sub>i<sub>s</sub>hi<sub>ng-error</sub> i<sub>mp</sub>li<sub>ca</sub>ti<sub>on</sub> <sub>a</sub>b<sub>ove</sub> i<sub>s</sub> <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>as</sub> <sub>a</sub> <sub>con</sub>diti<sub>ona</sub>l <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t f<sub>rom</sub> th<sub>e</sub> <sub>ear</sub>li<sub>er</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s;</sub> th<sub>e presen</sub>t <sub>paper uses</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow error</sub> b<sub>oun</sub>d <sub>an</sub>d th<sub>e</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow con</sub>t<sub>ex</sub>t<sub>-sca</sub>l<sub>e</sub> th<sub>eorem.</sub> Th<sub>e</sub> <sub>accep</sub>t<sub>e</sub>d <sub>sma</sub>ll fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow</sub> G<sub>auss</sub>i<sub>an approx</sub>i<sub>ma</sub>ti<sub>on</sub> i<sub>s cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith thi<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on.</sub>

## B.4. Interpretation for p-RoPE

Th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> <sub>reversa</sub>l<sub>-cu</sub>t<sub>o</sub>f th<sub>eorem</sub> <sub>an</sub>d th<sub>e</sub> <sub>reversa</sub>l <sub>resu</sub>lt<sub>s</sub> <sub>a</sub>b<sub>ove</sub> <sub>concern</sub> f<sub>u</sub>ll<sub>y</sub> <sub>ro</sub>t<sub>ary</sub> R<sub>o</sub>PE<sub>.</sub> I<sub>n</sub> <sub>a</sub> <sub>par</sub>ti<sub>a</sub>l<sub>-</sub>R<sub>o</sub>PE variant (Barbero et al., 2025), the score of a fixed ke<sub>y</sub> with fixed content states can be written as

$$
S _ { K } ^ { \mathrm { p } } ( m ) = c _ { K } + X _ { \mathcal { R } } ^ { b _ { \mathcal { R } } } ( m ) ,
$$

where R indexes the retained rotatin<sub>g</sub> dimensions and the NoPE com<sub>p</sub>onent $c _ { K }$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance.</sub> Thi<sub>s cons</sub>t<sub>an</sub>t d<sub>oes no</sub>t <sub>cance</sub>l <sub>aga</sub>i<sub>ns</sub>t <sub>a</sub> fi<sub>xe</sub>d <sub>score</sub> b<sub>arr</sub>i<sub>er:</sub>

$$
p _ { \mathrm { e x c } } ^ { \mathrm { p } } ( K ; S , M ) = \mathbb { P } [ X _ { \mathcal { R } } ^ { b _ { \mathcal { R } } } ( \mathbf { m } ) > S - c _ { K } ] .
$$

D<sub>epen</sub>di<sub>ng</sub> <sub>on</sub> it<sub>s</sub> <sub>s</sub>i<sub>gn,</sub> <sub>a</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>s</sub>t<sub>rong</sub> N<sub>o</sub>PE <sub>anc</sub>h<sub>or</sub> <sub>can</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>re</sub>l<sub>ax</sub> <sub>or</sub> ti<sub>g</sub>ht<sub>en</sub> th<sub>e</sub> <sub>score-</sub>b<sub>arr</sub>i<sub>er</sub> <sub>requ</sub>i<sub>remen</sub>t <sub>a</sub>ft<sub>er</sub> th<sub>e ce</sub>ili<sub>ng</sub> i<sub>s reca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>w</sub>ith th<sub>e e</sub>f<sub>ec</sub>ti<sub>ve</sub> b<sub>arr</sub>i<sub>er</sub> $S - c _ { K }$ <sub>.</sub> Lik<sub>ew</sub>i<sub>se, a</sub> f<sub>avora</sub>bl<sub>e</sub> N<sub>o</sub>PE <sub>marg</sub>i<sub>n</sub> <sub>a</sub>dd<sub>s</sub> <sub>a</sub> <sub>nonzero</sub> DC t<sub>erm</sub> t<sub>o</sub> th<sub>e</sub> <sub>cross-</sub>k<sub>ey</sub> <sub>score</sub> dif<sub>erence,</sub> <sub>so</sub> th<sub>e</sub> f<sub>u</sub>ll<sub>y</sub> <sub>ro</sub>t<sub>ary</sub> <sub>one-</sub>h<sub>a</sub>lf <sub>conc</sub>l<sub>us</sub>i<sub>on</sub> <sub>o</sub>f Th<sub>eorem</sub> 3 <sub>nee</sub>d <sub>no</sub>t <sub>app</sub>l<sub>y.</sub> Th<sub>us p-</sub>R<sub>o</sub>PE <sub>may avo</sub>id <sub>a par</sub>ti<sub>cu</sub>l<sub>ar score-</sub>b<sub>arr</sub>i<sub>er or</sub> f<sub>u</sub>ll<sub>y ro</sub>t<sub>ary reversa</sub>l <sub>cu</sub>t<sub>o</sub>f<sub>;</sub> th<sub>e</sub> <sub>magn</sub>it<sub>u</sub>d<sub>e</sub> <sub>o</sub>f it<sub>s</sub> N<sub>o</sub>PE <sub>s</sub>h<sub>are</sub> <sub>a</sub>l<sub>one,</sub> <sub>w</sub>ith<sub>ou</sub>t th<sub>e</sub> <sub>s</sub>i<sub>gn</sub> <sub>o</sub>f th<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>score</sub> <sub>marg</sub>i<sub>n,</sub> i<sub>s</sub> <sub>no</sub>t <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>.</sub>

Th<sub>e</sub> <sub>same</sub> <sub>cons</sub>t<sub>an</sub>t <sub>cance</sub>l<sub>s</sub> <sub>exac</sub>tl<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> <sub>pos</sub>iti<sub>ona</sub>l i<sub>n</sub>t<sub>erva</sub>l <sub>compar</sub>i<sub>son:</sub>

$$
S _ { K } ^ { \mathrm { p } } ( m _ { \mathrm { n e a r } } ) - S _ { K } ^ { \mathrm { p } } ( m _ { \mathrm { f a r } } ) = X _ { \mathcal { R } } ^ { b _ { \mathcal { R } } } ( m _ { \mathrm { n e a r } } ) - X _ { \mathcal { R } } ^ { b _ { \mathcal { R } } } ( m _ { \mathrm { f a r } } ) .
$$

N<sub>o</sub>PE di<sub>mens</sub>i<sub>ons may</sub> b<sub>e v</sub>i<sub>ewe</sub>d <sub>as zero-</sub>f<sub>requency</sub> di<sub>mens</sub>i<sub>ons an</sub>d h<sub>ence never</sub> b<sub>ecome cer</sub>tifi<sub>e</sub>d hi<sub>g</sub>h<sub>,</sub> b<sub>u</sub>t f<sub>or</sub> th<sub>e same reason</sub> th<sub>ey supp</sub>l<sub>y no c</sub>h<sub>ang</sub>i<sub>ng s</sub>i<sub>gna</sub>l <sub>w</sub>ith <sub>w</sub>hi<sub>c</sub>h t<sub>o or</sub>d<sub>er</sub> th<sub>e</sub> t<sub>wo</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s.</sub> If th<sub>e re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>ro</sub>t<sub>a</sub>ti<sub>ng</sub> <sub>componen</sub>t i<sub>s</sub> <sub>nonzero</sub> <sub>an</sub>d <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> <sub>a</sub>ll h<sub>ypo</sub>th<sub>eses</sub> <sub>o</sub>f Th<sub>eorem</sub> 2<sub>,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> f<sub>u</sub>ll <sub>cer</sub>tifi<sub>ca</sub>ti<sub>on</sub> <sub>an</sub>d t<sub>wo-</sub>i<sub>n</sub>t<sub>erva</sub>l <sub>ca</sub>lib<sub>ra</sub>ti<sub>on,</sub> th<sub>e same</sub> b<sub>oun</sub>d <sub>app</sub>li<sub>es</sub> t<sub>o</sub> th<sub>a</sub>t <sub>componen</sub>t<sub>.</sub> If <sub>no ro</sub>t<sub>a</sub>ti<sub>ng componen</sub>t <sub>rema</sub>i<sub>ns,</sub> th<sub>e</sub> t<sub>wo samp</sub>l<sub>e</sub>d <sub>scores are</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>an</sub>d $p _ { \mathrm { o r d } } = 0 ;$ thi<sub>s</sub> i<sub>s pos</sub>iti<sub>ona</sub>l i<sub>n</sub>dif<sub>erence, no</sub>t <sub>a re</sub>li<sub>a</sub>bl<sub>e pre</sub>f<sub>erence</sub> f<sub>or</sub> th<sub>e</sub> <sub>near</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>.</sub> H<sub>ence an unro</sub>t<sub>a</sub>t<sub>e</sub>d <sub>anc</sub>h<sub>or can suppor</sub>t <sub>score-</sub>b<sub>arr</sub>i<sub>er s</sub>t<sub>a</sub>bilit<sub>y,</sub> b<sub>u</sub>t it <sub>canno</sub>t b<sub>y</sub> it<sub>se</sub>lf <sub>prov</sub>id<sub>e</sub> l<sub>ong-</sub>di<sub>s</sub>t<sub>ance pos</sub>iti<sub>ona</sub>l di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on.</sub>

## C. Proofs of the Theoretical Results

## C.1. Proof of Proposition 2

Proposition 2 (RoPE score reduction). For afully rotary RoPE head, as the real query and key blocks range over $\mathbb { R } ^ { 2 h }$ , the class offixed query–key scores and their finite real linear combinations as functions of relative distance is exactly $\{ X _ { \mathcal { F } } ^ { z } : z \in \mathbb { C } ^ { h } \} _ { }$ ; hence all subsequentfull- and bandwise analyses reduce to studying $X _ { J } ^ { z } .$

Proof. The proof has two directions. We first rewrite each rotary plane in complex form and use real linearity; <sub>we</sub> th<sub>en cons</sub>t<sub>ruc</sub>t <sub>rea</sub>l <sub>ro</sub>t<sub>ary</sub> bl<sub>oc</sub>k<sub>s</sub> f<sub>or an ar</sub>bit<sub>rary coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>or.</sub>

Let $q , k _ { + } , k _ { - } \in \mathbb { R } ^ { 2 h }$ denote one <sub>p</sub>re-RoPE <sub>q</sub>uer<sub>y</sub> and two <sub>p</sub>re-RoPE ke<sub>y</sub>s. For each fre<sub>q</sub>uenc<sub>y</sub> <sub>�,</sub> <sub>g</sub>rou<sub>p</sub> the <sub>correspon</sub>di<sub>ng rea</sub>l <sub>ro</sub>t<sub>ary coor</sub>di<sub>na</sub>t<sub>es</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> bl<sub>oc</sub>k<sub>s</sub> $q _ { n } : = ( q _ { n , 1 } , q _ { n , 2 } ) ^ { \top }$ <sub>an</sub>d $( k _ { \pm } ) _ { n } : = ( ( k _ { \pm } ) _ { n , 1 } , ( k _ { \pm } ) _ { n , 2 } ) ^ { \top }$ <sub>,</sub> <sub>an</sub>d <sub>represen</sub>t th<sub>ese</sub> bl<sub>oc</sub>k<sub>s</sub> b<sub>y</sub>

$$
Q _ { n } : = q _ { n , 1 } + i q _ { n , 2 } , \qquad ( K _ { \pm } ) _ { n } : = ( k _ { \pm } ) _ { n , 1 } + i ( k _ { \pm } ) _ { n , 2 } .\tag{58}
$$

Let $p _ { q }$ <sub>an</sub>d $p _ { k }$ b<sub>e</sub> th<sub>e</sub> <sub>query</sub> <sub>an</sub>d k<sub>ey</sub> <sub>pos</sub>iti<sub>ons,</sub> <sub>se</sub>t $m : = p _ { q } - p _ { k }$ <sub>,</sub> <sub>an</sub>d <sub>wr</sub>it<sub>e</sub>

$$
R ( \theta ) : = \left( \begin{array} { c c } { { \cos \theta } } & { { - \sin \theta } } \\ { { \sin \theta } } & { { \cos \theta } } \end{array} \right) .\tag{59}
$$

F<sub>or a pos</sub>iti<sub>on-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>a</sub>tt<sub>en</sub>ti<sub>on-score sca</sub>l<sub>e</sub> $c _ { \mathrm { a t t } } > 0$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
S _ { \pm } ( m ) : = c _ { \mathrm { a t t } } \sum _ { n \in \mathcal { F } } \left. R ( p _ { q } \omega _ { n } ) q _ { n } , R ( p _ { k } \omega _ { n } ) ( k _ { \pm } ) _ { n } \right. .\tag{60}
$$

Th<sub>e ro</sub>t<sub>a</sub>ti<sub>on ma</sub>t<sub>r</sub>i<sub>ces sa</sub>ti<sub>s</sub>f<sub>y</sub>

$$
R ( \alpha ) ^ { \top } R ( \beta ) = R ( \beta - \alpha ) .\tag{61}
$$

C<sub>onsequen</sub>tl<sub>y,</sub> th<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f th<sub>e �</sub>th <sub>ro</sub>t<sub>ary p</sub>l<sub>ane</sub> i<sub>s</sub>

$$
\begin{array} { r l } & { \left. R ( p _ { q } \omega _ { n } ) q _ { n } , R ( p _ { k } \omega _ { n } ) ( k _ { \pm } ) _ { n } \right. = q _ { n } ^ { \top } R \big ( ( p _ { k } - p _ { q } ) \omega _ { n } \big ) ( k _ { \pm } ) _ { n } } \\ & { ~ = q _ { n } ^ { \top } R ( - m \omega _ { n } ) ( k _ { \pm } ) _ { n } . } \end{array}\tag{62}
$$

Th<sub>e</sub> <sub>same</sub> <sub>p</sub>l<sub>ane</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>a</sub>d<sub>m</sub>it<sub>s</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>comp</sub>l<sub>ex</sub> <sub>represen</sub>t<sub>a</sub>ti<sub>on:</sub>

$$
\overline { { Q _ { n } ( K _ { \pm } ) _ { n } } } = q _ { n , 1 } ( k _ { \pm } ) _ { n , 1 } + q _ { n , 2 } ( k _ { \pm } ) _ { n , 2 } + i \big ( q _ { n , 2 } ( k _ { \pm } ) _ { n , 1 } - q _ { n , 1 } ( k _ { \pm } ) _ { n , 2 } \big ) .\tag{63}
$$

T<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> <sub>rea</sub>l <sub>par</sub>t <sub>a</sub>ft<sub>er</sub> <sub>mu</sub>lti<sub>p</sub>li<sub>ca</sub>ti<sub>on</sub> b<sub>y</sub> $e ^ { i m \omega _ { n } }$ <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { r l } & { \mathrm { R e } \Big ( Q _ { n } \overline { { ( K _ { \pm } ) _ { n } } } e ^ { i m \omega _ { n } } \Big ) = \big ( q _ { n , 1 } ( k _ { \pm } ) _ { n , 1 } + q _ { n , 2 } ( k _ { \pm } ) _ { n , 2 } \big ) \cos ( m \omega _ { n } ) } \\ & { \phantom { \frac { 1 } { 1 } } + \big ( q _ { n , 1 } ( k _ { \pm } ) _ { n , 2 } - q _ { n , 2 } ( k _ { \pm } ) _ { n , 1 } \big ) \sin ( m \omega _ { n } ) } \\ & { \phantom { \frac { 1 } { 1 } } = q _ { n } ^ { \top } R ( - m \omega _ { n } ) ( k _ { \pm } ) _ { n } . } \end{array}\tag{64}
$$

Combinin<sub>g</sub> E<sub>q</sub>. (60) and E<sub>q</sub>. (64) <sub>y</sub>ields

$$
S _ { \pm } ( m ) = \mathrm { R e } \sum _ { n \in \mathcal { F } } \left( c _ { \mathrm { a t t } } Q _ { n } \overline { { { \left( K _ { \pm } \right) } _ { n } } } \right) e ^ { i m \omega _ { n } } = X _ { \mathcal { F } } ^ { a _ { \pm } } ( m ) ,\tag{65}
$$

$$
( a _ { + } ) _ { n } : = c _ { \mathrm { a t t } } Q _ { n } \overline { { { ( K _ { + } ) _ { n } } } } , ( a _ { - } ) _ { n } : = c _ { \mathrm { a t t } } Q _ { n } \overline { { { ( K _ { - } ) _ { n } } } } .\tag{66}
$$

Th<sub>us every</sub> fi<sub>xe</sub>d <sub>score</sub> h<sub>as</sub> th<sub>e requ</sub>i<sub>re</sub>d f<sub>orm.</sub> R<sub>ea</sub>l li<sub>near</sub>it<sub>y</sub> i<sub>n</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>or s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>every</sub> fi<sub>n</sub>it<sub>e</sub> <sub>rea</sub>l li<sub>near com</sub>bi<sub>na</sub>ti<sub>on</sub> h<sub>as</sub> th<sub>e same</sub> f<sub>orm.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar, se</sub>tti<sub>ng</sub>

$$
a _ { n } : = ( a _ { + } ) _ { n } , \qquad b _ { n } : = ( a _ { - } ) _ { n } , \qquad d _ { n } : = a _ { n } - b _ { n }\tag{67}
$$

<sub>g</sub><sup>i</sup>ves

$$
D ( m ) : = S _ { + } ( m ) - S _ { - } ( m ) = X _ { \mathcal { F } } ^ { d } ( m ) .\tag{68}
$$

C<sub>o</sub>nv<sub>e</sub>r<sub>se</sub>l<sub>y,</sub> fix an<sub>y</sub> $z \in \mathbb { C } ^ { h }$ . Choosin<sub>g</sub> $Q _ { n } = 1$ <sub>an</sub>d $( K _ { + } ) _ { n } = \overline { { z _ { n } / c _ { \mathrm { a t t } } } }$ <sup>f</sup>or ever<sub>y</sub> � <sub>g</sub><sup>i</sup>ves $c _ { \mathrm { a t t } } Q _ { n } ( K _ { + } ) _ { n } = z _ { n }$ <sub>.</sub> E<sub>ac</sub>h <sub>c</sub>h<sub>osen</sub> <sub>comp</sub>l<sub>ex coor</sub>di<sub>na</sub>t<sub>e correspon</sub>d<sub>s</sub> t<sub>o a rea</sub>l t<sub>wo-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>ro</sub>t<sub>ary</sub> bl<sub>oc</sub>k<sub>, so</sub> thi<sub>s cons</sub>t<sub>ruc</sub>t<sub>s rea</sub>l $q , k _ { + } \in \mathbb { R } ^ { 2 h }$ th<sub>a</sub>t <sub>rea</sub>li<sub>ze</sub> $X _ { \mathcal { F } } ^ { z }$ □

## C.2. Finite-Window Fourier-Frame Bound

For $\alpha \in \mathbb { R } .$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e norma</sub>li<sub>ze</sub>d fi<sub>n</sub>it<sub>e-sum</sub> k<sub>erne</sub>l

$$
K _ { M } ( \alpha ) : = \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } e ^ { i m \alpha } .\tag{69}
$$

F<sub>or</sub> <sub>a</sub> fi<sub>n</sub>it<sub>e</sub> f<sub>requency</sub> <sub>se</sub>t $J \subseteq { \mathcal { F } }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\Lambda _ { J } ^ { 0 } : = \{ 0 \} \cup \{ + \omega _ { n } : n \in J \} \cup \{ - \omega _ { n } : n \in J \} , \qquad \delta _ { J } : = \operatorname* { m i n } _ { \lambda \neq \lambda ^ { \prime } \in \Lambda _ { J } ^ { 0 } } \mathrm { d i s t } ( \lambda - \lambda ^ { \prime } , 2 \pi \mathbb { Z } ) ,\tag{70}
$$

<sub>an</sub>d th<sub>e</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow</sub> G<sub>ram ma</sub>t<sub>r</sub>i<sub>x an</sub>d <sub>s</sub>i<sub>gne</sub>d<sub>-</sub>f<sub>rame</sub> d<sub>e</sub>f<sub>ec</sub>t

$$
\begin{array} { r } { \mathcal { G } _ { J } ( M ) : = \left[ K _ { M } ( \lambda ^ { \prime } - \lambda ) \right] _ { \lambda , \lambda ^ { \prime } \in \Lambda _ { I } ^ { 0 } } , \qquad \mathfrak { d } _ { J } ( M ) : = \| \mathcal { G } _ { J } ( M ) - I \| _ { \mathrm { o p } } . } \end{array}\tag{71}
$$

Lemma C.1 (Finite-window Fourier-frame bound). $I f \delta _ { J } > 0 ,$ , then

$$
{ \mathfrak { d } } _ { J } ( M ) \leq { \frac { C _ { \mathrm { F } } } { M \delta _ { J } } } .\tag{72}
$$

In particular, $M \delta _ { J } \geq 2 C _ { \mathrm { F } } / \varepsilon$ implies $\mathfrak { d } _ { J } ( M ) \le \varepsilon / 2$

Proof. We decompose the of-diagonal kernel into two cosecant forms, apply the separated-point Hilbert i<sub>nequa</sub>lit<sub>y</sub> t<sub>o</sub> <sub>eac</sub>h f<sub>orm,</sub> <sub>an</sub>d th<sub>en</sub> <sub>norma</sub>li<sub>ze</sub> b<sub>y</sub> th<sub>e</sub> <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> l<sub>eng</sub>th<sub>.</sub>

Th<sub>e</sub> fi<sub>n</sub>it<sub>e geome</sub>t<sub>r</sub>i<sub>c-sum</sub> k<sub>erne</sub>l <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\sum _ { m = 0 } ^ { M - 1 } e ^ { i m ( \lambda - \lambda ^ { \prime } ) } = \frac { e ^ { i ( M - 1 / 2 ) ( \lambda - \lambda ^ { \prime } ) } - e ^ { - i ( \lambda - \lambda ^ { \prime } ) / 2 } } { 2 i \sin ( ( \lambda - \lambda ^ { \prime } ) / 2 ) } .\tag{73}
$$

I<sub>n</sub> th<sub>e o</sub>f<sub>-</sub>di<sub>agona</sub>l <sub>qua</sub>d<sub>ra</sub>ti<sub>c</sub> f<sub>orm,</sub> th<sub>e</sub> t<sub>wo numera</sub>t<sub>or</sub> t<sub>erms</sub> d<sub>ecompose</sub> i<sub>n</sub>t<sub>o cosecan</sub>t f<sub>orms w</sub>h<sub>ose coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>ors</sub> dif<sub>er on</sub>l<sub>y</sub> b<sub>y</sub> di<sub>agona</sub>l <sub>un</sub>it<sub>ary mo</sub>d<sub>u</sub>l<sub>a</sub>ti<sub>on.</sub> A<sub>pp</sub>l<sub>y</sub> th<sub>e c</sub>i<sub>rcu</sub>l<sub>ar</sub> Hilb<sub>er</sub>t i<sub>nequa</sub>lit<sub>y o</sub>f M<sub>on</sub>t<sub>gomery</sub> $\&$ Vau<sub>g</sub>han (1974, Theorem 1, E<sub>q</sub>. (1.2)) with $x _ { r } = \lambda _ { r } / ( 2 \pi )$ <sub>an</sub>d <sub>mo</sub>d<sub>u</sub>l<sub>o-one spac</sub>i<sub>ng</sub> $\delta _ { J } / ( 2 \pi )$ <sub>.</sub> It b<sub>oun</sub>d<sub>s eac</sub>h t<sub>erm,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> it<sub>s</sub> f<sub>ac</sub>t<sub>or</sub> $1 / 2 ,$ , <sup>b</sup><sub>y</sub> $\pi / \delta _ { J }$ i<sub>n</sub> <sub>opera</sub>t<sub>or</sub> <sub>norm.</sub> Di<sub>v</sub>idi<sub>ng</sub> b<sub>y</sub> � <sub>an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>g</sub><sup>i</sup>ves

$$
{ \mathfrak { d } } _ { J } ( M ) \leq { \frac { 2 \pi } { M \delta _ { J } } } .\tag{74}
$$

Th<sub>us</sub> th<sub>e</sub> l<sub>emma</sub> h<sub>o</sub>ld<sub>s w</sub>ith th<sub>e un</sub>i<sub>versa</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> $C _ { \mathrm { F } } : = 2 \pi$ <sub>.</sub> Th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>on</sub> f<sub>or</sub> th<sub>e</sub> d<sub>e</sub>f<sub>ec</sub>t b<sub>oun</sub>d f<sub>o</sub>ll<sub>ows</sub> b<sub>y su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on.</sub> □

## C.3. Proof of Proposition 1 and Lemma A.1

Proposition 1 (Analytical moment guarantees under finite windows). For any tolerance $\varepsilon \in ( 0 , 1 / 2 )$ and certified window � where � is nonempty, the high-frequency attention score $\begin{array} { r } { S _ { H } ( m ) : = \Re \sum _ { n \in H } z _ { n } e ^ { i m \omega _ { n } } } \end{array}$ and its coeficient norm $\begin{array} { r } { U _ { H } ( z ) : = ( \sum _ { n \in H } | z _ { n } | ^ { 2 } ) ^ { 1 / 2 } } \end{array}$ over uniformly distributed relative distances $m \sim \operatorname { U n i f } \{ 0 , \dots , M - 1 \}$ satisfy

$$
| \mathbb { E } _ { m } \left[ S _ { H } ( m ) \right] | \le \varepsilon U _ { H } ( z ) \qquad a n d \qquad \left| \mathrm { V a r } _ { m } ( S _ { H } ( m ) ) - \frac { 1 } { 2 } U _ { H } ( z ) ^ { 2 } \right| \le \varepsilon U _ { H } ( z ) ^ { 2 } .\tag{5}
$$

Lemma A.1 (Finite-window moments and conditional Gaussian calibration). Fix $0 < \varepsilon < 1 / 2 ,$ a certified window $M \geq \Gamma _ { \varepsilon } ( \rho )$ , and fixed coeficients � with $U _ { H } ( z ) > 0$ . Use $H = \mathcal { H } _ { \varepsilon } ( M ) f r o m$ Eq. (12) and the uniformly sampled relative distance in Eq. (13). For its random score $Y _ { H } ^ { z } ,$ write

$$
\mu _ { H } : = \big \langle X _ { H } ^ { z } \big \rangle _ { M } , \qquad \sigma _ { H } ^ { 2 } : = \mathsf { V a r } _ { M } ( X _ { H } ^ { z } ) .\tag{17}
$$

Then the following moment bounds hold:

$$
\begin{array} { r } { | \mu _ { H } | \le \varepsilon U _ { H } ( z ) , \qquad \left| \sigma _ { H } ^ { 2 } - \frac 1 2 U _ { H } ( z ) ^ { 2 } \right| \le \varepsilon U _ { H } ( z ) ^ { 2 } . } \end{array}\tag{18}
$$

Because $0 < \varepsilon < 1 / 2 ,$ , Eq. (18) implies $\sigma _ { H } ^ { 2 } \geq ( 1 / 2 - \varepsilon ) U _ { H } ( z ) ^ { 2 } > 0$ whenever $U _ { H } ( z ) > 0 .$ . The standardization below is therefore well defined. The remaining, non-deterministic source of approximation error is the Gaussian-shape discrepancy of the standardized window law,

$$
\Delta _ { \mathrm { H } , \mathrm { G } } ( z ; M ) : = \operatorname* { s u p } _ { x \in \mathbb { R } } \left. \mathbb { P } \left[ \frac { Y _ { H } ^ { z } - \mu _ { H } } { \sigma _ { H } } \leq x \right] - \Phi _ { \mathrm { s t d } } ( x ) \right. \leq \tau _ { \mathrm { H } , \mathrm { G } } .\tag{19}
$$

Under this condition, the exact form of the Gaussian conclusion is

$$
\operatorname* { s u p } _ { x \in \mathbb { R } } \left. \mathbb { P } [ Y _ { H } ^ { z } \leq x ] - \Phi _ { \mathrm { s t d } } \left( \frac { \sqrt { 2 } x } { U _ { H } ( z ) } \right) \right. \leq \tau _ { \mathrm { H , G } } + c _ { \mathrm { G } } ( \varepsilon ) ,\tag{20}
$$

where the universal moment-replacement term is

$$
c _ { \mathsf { G } } ( \varepsilon ) : = { \frac { \varepsilon } { \sqrt { \pi } } } + { \frac { \log ( ( 1 - 2 \varepsilon ) ^ { - 1 } ) } { 2 { \sqrt { 2 \pi e } } } } = O ( \varepsilon ) \qquad ( \varepsilon \to 0 ) .\tag{21}
$$

We use the convention ${ \cal N } ( \mu , \sigma ^ { 2 } )$ : the target variance is $U _ { H } ( z ) ^ { 2 } / 2 ,$ , equivalently the target standard deviation is $U _ { H } ( z ) / \sqrt { 2 }$

Proof. By the exact formulation in Appendix A.3, it is enough to establish the two moment inequalities in E<sub>q</sub>. (18) and then transfer a $\tau _ { \mathrm { H } , \mathsf { G } }$ Gaussian-sha<sub>p</sub>e bound to E<sub>q</sub>. (20). The <sub>p</sub>roof <sub>p</sub>roceeds in four sta<sub>g</sub>es. We fi<sub>rs</sub>t <sub>cer</sub>tif<sub>y separa</sub>ti<sub>on o</sub>f th<sub>e s</sub>i<sub>gne</sub>d f<sub>requenc</sub>i<sub>es an</sub>d <sub>conver</sub>t th<sub>a</sub>t <sub>separa</sub>ti<sub>on</sub> i<sub>n</sub>t<sub>o</sub> bl<sub>oc</sub>k<sub>w</sub>i<sub>se</sub> G<sub>ram-ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>con</sub>t<sub>ro</sub>l<sub>.</sub> W<sub>e</sub> th<sub>en ex</sub>t<sub>rac</sub>t th<sub>e mean an</sub>d <sub>var</sub>i<sub>ance</sub> b<sub>oun</sub>d<sub>s an</sub>d fi<sub>na</sub>ll<sub>y</sub> t<sub>rans</sub>f<sub>er</sub> th<sub>e con</sub>diti<sub>ona</sub>l G<sub>auss</sub>i<sub>an</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e exac</sub>t <sub>momen</sub>t<sub>s</sub> t<sub>o</sub> th<sub>e</sub>i<sub>r ca</sub>lib<sub>ra</sub>t<sub>e</sub>d t<sub>arge</sub>t<sub>s.</sub>

U<sub>se</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e-sum</sub> k<sub>erne</sub>l<sub>,</sub> <sub>s</sub>i<sub>gne</sub>d<sub>-</sub>f<sub>requency</sub> <sub>spac</sub>i<sub>ng,</sub> <sub>an</sub>d G<sub>ram-ma</sub>t<sub>r</sub>i<sub>x</sub> d<sub>e</sub>f<sub>ec</sub>t d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>2<sub>.</sub>

W<sub>e</sub> fi<sub>rs</sub>t <sub>ver</sub>if<sub>y</sub> th<sub>e requ</sub>i<sub>re</sub>d <sub>s</sub>i<sub>gne</sub>d<sub>-</sub>f<sub>requency separa</sub>ti<sub>on.</sub> B<sub>ecause</sub> $M \geq \Gamma _ { \varepsilon } ( \rho )$ <sub>,</sub> th<sub>e</sub> <sub>cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h <sub>se</sub>t i<sub>s</sub> <sub>nonemp</sub>t<sub>y.</sub> Write $H = \{ 0 , \ldots , \nu \}$ <sub>.</sub> T<sub>o</sub> l<sub>ower-</sub>b<sub>oun</sub>d th<sub>e</sub> <sub>s</sub>i<sub>gne</sub>d <sub>wrappe</sub>d <sub>spac</sub>i<sub>ng,</sub> <sub>cons</sub>id<sub>er</sub> th<sub>e</sub> f<sub>our</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>pa</sub>i<sub>r</sub> t<sub>ypes.</sub> Di<sub>s</sub>t<sub>ances</sub> f<sub>rom</sub> th<sub>e s</sub>i<sub>gne</sub>d f<sub>requenc</sub>i<sub>es</sub> t<sub>o zero are a</sub>t l<sub>eas</sub>t $\rho ^ { \nu } ;$ <sub>; same-s</sub>i<sub>gn</sub> di<sub>s</sub>t<sub>ances are a</sub>t l<sub>eas</sub>t $( 1 - \rho ) \rho ^ { \upsilon } .$

<sub>oppos</sub>it<sub>e-s</sub>i<sub>gn</sub> di<sub>s</sub>t<sub>ances</sub> <sub>are</sub> <sub>a</sub>t l<sub>eas</sub>t $2 \rho ^ { \nu } { } _ { ; }$ <sub>;</sub> <sub>an</sub>d th<sub>e</sub> <sub>wrap-aroun</sub>d di<sub>s</sub>t<sub>ance</sub> i<sub>s</sub> l<sub>arger</sub> th<sub>an</sub> $2 \pi - 2$ <sub>.</sub> Th<sub>ese</sub> <sub>cases</sub> <sub>cover every pa</sub>i<sub>r o</sub>f di<sub>s</sub>ti<sub>nc</sub>t <sub>s</sub>i<sub>gne</sub>d f<sub>requenc</sub>i<sub>es.</sub> H<sub>ence</sub>

$$
\delta _ { H } \geq ( 1 - \rho ) \rho ^ { \nu } .\tag{75}
$$

Since $\nu \in H ,$

$$
M \delta _ { H } \geq M ( 1 - \rho ) \rho ^ { \nu } \geq ( 1 - \rho ) \Gamma _ { \varepsilon } ( \rho ) = \frac { 2 C _ { \mathrm { F } } } { \varepsilon } .\tag{76}
$$

Lemma C.1 therefore <sub>g</sub>ives $\mathfrak { d } _ { H } ( M ) \le \varepsilon / 2$

T<sub>o</sub> <sub>rea</sub>d <sub>o</sub>f th<sub>e</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l bl<sub>oc</sub>k<sub>s,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>vec</sub>t<sub>or</sub> $g _ { H }$ <sub>an</sub>d th<sub>e</sub> <sub>ma</sub>t<sub>r</sub>i<sub>ces</sub> $G _ { H } ^ { - }$ <sub>an</sub>d $G _ { H } ^ { + }$ <sup>b</sup><sub>y</sub>

$$
( g _ { H } ) _ { n } : = K _ { M } ( \omega _ { n } ) ,\tag{77}
$$

$$
( G _ { H } ^ { - } ) _ { n , n ^ { \prime } } : = K _ { M } ( \omega _ { n ^ { \prime } } - \omega _ { n } ) , \qquad ( G _ { H } ^ { + } ) _ { n , n ^ { \prime } } : = K _ { M } ( \omega _ { n } + \omega _ { n ^ { \prime } } ) .\tag{78}
$$

Order $\Lambda _ { H } ^ { 0 }$ as $( 0 , + H , - H )$ . Since $K _ { M } ( - \alpha ) = K _ { M } ( \alpha )$ <sub>,</sub> th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>cen</sub>t<sub>ere</sub>d G<sub>ram</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> i<sub>s</sub>

$$
\wp ( M ) - I = \left[ { \frac { 0 } { g _ { H } } } \quad G _ { H } ^ { - } - I \quad { \frac { g _ { H } ^ { * } } { G _ { H } ^ { + } } } \right] .\tag{79}
$$

E<sub>very</sub> di<sub>sp</sub>l<sub>aye</sub>d bl<sub>oc</sub>k i<sub>s</sub> <sub>a</sub> <sub>compress</sub>i<sub>on</sub> <sub>o</sub>f th<sub>e</sub> f<sub>u</sub>ll <sub>cen</sub>t<sub>ere</sub>d <sub>ma</sub>t<sub>r</sub>i<sub>x.</sub> Th<sub>ere</sub>f<sub>ore</sub>

$$
\left\| g _ { H } \right\| _ { 2 } \leq \mathfrak { d } _ { H } ( M ) , \qquad \left\| G _ { H } ^ { - } - I \right\| _ { \mathrm { o p } } \leq \mathfrak { d } _ { H } ( M ) , \qquad \left\| G _ { H } ^ { + } \right\| _ { \mathrm { o p } } \leq \mathfrak { d } _ { H } ( M ) .\tag{80}
$$

W<sub>e now conver</sub>t th<sub>ese</sub> bl<sub>oc</sub>k b<sub>oun</sub>d<sub>s</sub> i<sub>n</sub>t<sub>o momen</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>es.</sub> L<sub>e</sub>t $\begin{array} { r } { Z _ { H } ^ { z } ( m ) : = \sum _ { n \in H } z _ { n } e ^ { i m \omega _ { n } } } \end{array}$ , so $X _ { H } ^ { z } = \operatorname { R e } Z _ { H } ^ { z }$ <sub>.</sub> Th<sub>e</sub> first row of E<sub>q</sub>. (79) <sub>g</sub>ives

$$
\begin{array} { r } { \big \langle X _ { H } ^ { z } \big \rangle _ { M } = \mathrm { R e } ( g _ { H } ^ { \top } z _ { H } ) , \qquad \big | \big \langle X _ { H } ^ { z } \big \rangle _ { M } \big | \le \mathfrak { d } _ { H } ( M ) U _ { H } ( z ) . } \end{array}\tag{81}
$$

Usin<sub>g</sub> $\begin{array} { r } { ( \mathrm { R e } Z ) ^ { 2 } = \frac { 1 } { 2 } \left| Z \right| ^ { 2 } + \frac { 1 } { 2 } \mathrm { R e } ( Z ^ { 2 } ) } \end{array}$

$$
\big \langle ( X _ { H } ^ { z } ) ^ { 2 } \big \rangle _ { M } = \frac { 1 } { 2 } z _ { H } ^ { * } G _ { H } ^ { - } z _ { H } + \frac { 1 } { 2 } \operatorname { R e } \big ( z _ { H } ^ { \top } G _ { H } ^ { + } z _ { H } \big ) .\tag{82}
$$

Th<sub>e</sub> bl<sub>oc</sub>k b<sub>oun</sub>d<sub>s</sub> i<sub>mp</sub>l<sub>y</sub>

$$
\left| \big < ( X _ { H } ^ { z } ) ^ { 2 } \big > _ { M } - \frac { 1 } { 2 } U _ { H } ( z ) ^ { 2 } \right| \leq \mathfrak { d } _ { H } ( M ) U _ { H } ( z ) ^ { 2 } .\tag{83}
$$

Aft<sub>er</sub> <sub>su</sub>bt<sub>rac</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>square</sub>d <sub>mean,</sub>

$$
\left| \operatorname { V a r } _ { M } ( X _ { H } ^ { z } ) - { \frac { 1 } { 2 } } U _ { H } ( z ) ^ { 2 } \right| \leq { \bigl ( } \mathfrak { d } _ { H } { \bigl ( } M { \bigr ) } + \mathfrak { d } _ { H } { \bigl ( } M { \bigr ) } ^ { 2 } { \bigr ) } U _ { H } ( z ) ^ { 2 } .\tag{84}
$$

Fi<sub>na</sub>ll<sub>y,</sub>

$$
\mathfrak { d } _ { H } ( M ) + \mathfrak { d } _ { H } ( M ) ^ { 2 } \leq \frac \varepsilon 2 + \frac { \varepsilon ^ { 2 } } { 4 } \leq \varepsilon .\tag{85}
$$

Substitution into E<sub>q</sub>. (81) and E<sub>q</sub>. (84) <sub>p</sub>roves E<sub>q</sub>. (18).

It <sub>rema</sub>i<sub>ns</sub> t<sub>o prove</sub> th<sub>e con</sub>diti<sub>ona</sub>l G<sub>auss</sub>i<sub>an conc</sub>l<sub>us</sub>i<sub>on.</sub> A<sub>ssume</sub> $U _ { H } ( z ) > 0$ <sub>an</sub>d <sub>se</sub>t $s _ { 0 } : = U _ { H } ( z ) / \sqrt { 2 }$ <sub>.</sub> Th<sub>e</sub> variance bound in E<sub>q</sub>. (18) <sub>g</sub>ives

$$
\sigma _ { H } ^ { 2 } \geq \left( \frac { 1 } { 2 } - \varepsilon \right) U _ { H } ( z ) ^ { 2 } > 0 , \qquad \sqrt { 1 - 2 \varepsilon } \leq \frac { \sigma _ { H } } { s _ { 0 } } \leq \sqrt { 1 + 2 \varepsilon } .\tag{86}
$$

Thus the standardized variable in E<sub>q</sub>. (19) is well defined, and that condition im<sub>p</sub>lies

$$
\operatorname* { s u p } _ { x \in \mathbb { R } } \left. \mathbb { P } [ Y _ { H } ^ { z } \leq x ] - \Phi _ { \mathrm { s t d } } \left( \frac { x - \mu _ { H } } { \sigma _ { H } } \right) \right. \leq \tau _ { \mathrm { H , G } } .\tag{87}
$$

W<sub>e</sub> <sub>nex</sub>t <sub>rep</sub>l<sub>ace</sub> th<sub>e</sub> <sub>exac</sub>t <sub>momen</sub>t<sub>s</sub> b<sub>y</sub> th<sub>e</sub>i<sub>r</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d t<sub>arge</sub>t<sub>s.</sub> Dif<sub>eren</sub>ti<sub>a</sub>ti<sub>ng</sub> $\Phi _ { \mathrm { s t d } } ( ( x - \mu ) / s )$ <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> l<sub>og � s</sub>h<sub>ows</sub> th<sub>a</sub>t it<sub>s a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve</sub> i<sub>s a</sub>t <sub>mos</sub>t $1 / \sqrt { 2 \pi e }$ , uniforml<sub>y</sub> in � and <sub>�</sub>. Hence E<sub>q</sub>. (86) <sub>y</sub>ields

$$
\operatorname* { s u p } _ { x \in \mathbb { R } } \left| \Phi _ { \mathrm { s t d } } \left( \frac { x - \mu _ { H } } { \sigma _ { H } } \right) - \Phi _ { \mathrm { s t d } } \left( \frac { x - \mu _ { H } } { s _ { 0 } } \right) \right| \leq \frac { \log ( ( 1 - 2 \varepsilon ) ^ { - 1 } ) } { 2 \sqrt { 2 \pi e } } .\tag{88}
$$

Si<sub>nce</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d G<sub>auss</sub>i<sub>an</sub> CDF i<sub>s</sub> $1 / { \sqrt { 2 \pi } } -$ Li<sub>p</sub>schitz, the mean bound in E<sub>q</sub>. (18) also <sub>g</sub>ives

$$
\operatorname* { s u p } _ { x \in \mathbb { R } } \left| \Phi _ { \mathrm { s t d } } \left( \frac { x - \mu _ { H } } { s _ { 0 } } \right) - \Phi _ { \mathrm { s t d } } \left( \frac { x } { s _ { 0 } } \right) \right| \leq \frac { | \mu _ { H } | } { s _ { 0 } \sqrt { 2 \pi } } \leq \frac { \varepsilon } { \sqrt { \pi } } .\tag{89}
$$

Combinin<sub>g</sub> E<sub>q</sub>. (87), E<sub>q</sub>. (88), and E<sub>q</sub>. (89) b<sub>y</sub> the trian<sub>g</sub>le ine<sub>q</sub>ualit<sub>y p</sub>roves E<sub>q</sub>. (20). This com<sub>p</sub>letes the <sub>con</sub>diti<sub>ona</sub>l G<sub>auss</sub>i<sub>an</sub> <sub>s</sub>t<sub>ep.</sub> □

## C.4. Strengthened Bandwise Calibration

Th<sub>e ma</sub>i<sub>n</sub> t<sub>ex</sub>t <sub>uses on</sub>l<sub>y</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>ca</sub>lib<sub>ra</sub>ti<sub>on.</sub> F<sub>or comp</sub>l<sub>e</sub>t<sub>eness,</sub> th<sub>e same</sub> f<sub>rame argumen</sub>t <sub>a</sub>l<sub>so con</sub>t<sub>ro</sub>l<sub>s</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve var</sub>i<sub>ance an</sub>d th<sub>e score–</sub>d<sub>er</sub>i<sub>va</sub>ti<sub>ve covar</sub>i<sub>ance on any</sub> b<sub>an</sub>d <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>separa</sub>ti<sub>on con</sub>diti<sub>on.</sub>

Proposition 3 (Strengthened calibration for a separated band). Let $J \subseteq { \mathcal { F } }$ satisfy $M \delta _ { J } \geq 2 C _ { \mathrm { F } } / \varepsilon$ . Define

$$
E _ { 0 } ( J ; z ) : = \sum _ { n \in J } | z _ { n } | ^ { 2 } , \qquad E _ { 1 } ( J ; z ) : = \sum _ { n \in J } \omega _ { n } ^ { 2 } | z _ { n } | ^ { 2 } ,\tag{90}
$$

and

$$
( X _ { J } ^ { z } ) ^ { \prime } ( m ) : = \mathrm { R e } \sum _ { n \in J } i \omega _ { n } z _ { n } e ^ { i m \omega _ { n } } .\tag{91}
$$

Then

$$
\begin{array} { r } { \left| \left. X _ { J } ^ { z } \right. _ { M } \right| \leq \varepsilon \sqrt { E _ { 0 } ( J ; z ) } , } \end{array}\tag{92}
$$

$$
\begin{array} { r } { \left| \mathrm { V a r } _ { M } ( X _ { J } ^ { z } ) - \frac { 1 } { 2 } E _ { 0 } ( J ; z ) \right| \le \varepsilon E _ { 0 } ( J ; z ) , } \end{array}\tag{93}
$$

$$
\begin{array} { r l } & { \left| { \sf V a r } _ { M } ( ( X _ { J } ^ { z } ) ^ { \prime } ) - \frac { 1 } { 2 } E _ { 1 } ( J ; z ) \right| \leq \varepsilon E _ { 1 } ( J ; z ) , } \end{array}\tag{94}
$$

$$
\begin{array} { r l } & { \left. \mathrm { C o v } _ { M } ( X _ { J } ^ { z } , ( X _ { J } ^ { z } ) ^ { \prime } ) \right. \leq \varepsilon \sqrt { E _ { 0 } ( J ; z ) E _ { 1 } ( J ; z ) } . } \end{array}\tag{95}
$$

Proof. Lemma C.1 gives $\mathfrak { d } _ { J } ( M ) \le \varepsilon / 2$ <sub>.</sub> R<sub>epea</sub>ti<sub>ng</sub> th<sub>e</sub> bl<sub>oc</sub>k <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>3 <sub>w</sub>ith � <sub>rep</sub>l<sub>ace</sub>d b<sub>y</sub> � <sub>proves</sub> th<sub>e</sub> fi<sub>rs</sub>t t<sub>wo</sub> i<sub>nequa</sub>liti<sub>es.</sub> F<sub>or</sub> th<sub>e</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve, se</sub>t $w _ { n } = i \omega _ { n } z _ { n }$ <sub>.</sub> Th<sub>e</sub>n $X _ { J } ^ { w } = ( X _ { J } ^ { z } ) ^ { \prime }$ <sub>an</sub>d $\| w _ { J } \| _ { 2 } ^ { 2 } = E _ { 1 } ( J ; z )$ , so th<sub>e</sub> <sub>same</sub> <sub>var</sub>i<sub>ance</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>app</sub>li<sub>es.</sub>

For the covariance, the bilinear version of E<sub>q</sub>. (82) is

$$
\left. X _ { J } ^ { z } X _ { J } ^ { w } \right. _ { M } = { \frac { 1 } { 2 } } \operatorname { R e } \left( z _ { J } ^ { * } G _ { J } ^ { - } w _ { J } + z _ { J } ^ { \top } G _ { J } ^ { + } w _ { J } \right) .\tag{96}
$$

Th<sub>e</sub> id<sub>en</sub>tit<sub>y con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on van</sub>i<sub>s</sub>h<sub>es</sub> b<sub>ecause</sub> R<sub>e</sub> $( z _ { J } ^ { * } w _ { J } ) = 0$ . E<sub>q</sub>. (80) and E<sub>q</sub>. (81), with � re<sub>p</sub>laced b<sub>y</sub> �, th<sub>ere</sub>f<sub>ore g</sub>i<sub>ve</sub>

$$
\left| \mathsf { C o v } _ { M } ( X _ { J } ^ { z } , X _ { J } ^ { w } ) \right| \le \left( \mathfrak { d } _ { J } ( M ) + \mathfrak { d } _ { J } ( M ) ^ { 2 } \right) \| z _ { J } \| _ { 2 } \| w _ { J } \| _ { 2 } .\tag{97}
$$

Fi<sub>na</sub>ll<sub>y,</sub> $\mathfrak { d } _ { J } + \mathfrak { d } _ { J } ^ { 2 } \le \varepsilon / 2 + \varepsilon ^ { 2 } / 4 \le \varepsilon$

## C.5. Direct Finite-Sum Intuition

Th<sub>e</sub> <sub>nex</sub>t <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f th<sub>e</sub> F<sub>our</sub>i<sub>er-</sub>f<sub>rame</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>a</sub>b<sub>ove.</sub> It i<sub>s</sub> <sub>no</sub>t <sub>nee</sub>d<sub>e</sub>d f<sub>or</sub> th<sub>e</sub> <sub>ma</sub>i<sub>n-</sub>t<sub>ex</sub>t <sub>ca</sub>lib<sub>ra</sub>ti<sub>on resu</sub>lt<sub>,</sub> b<sub>u</sub>t it <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency cen</sub>t<sub>er</sub>i<sub>ng mec</sub>h<sub>an</sub>i<sub>sm exp</sub>li<sub>c</sub>it<sub>.</sub>

Proposition 4 (Direct centering on the geometric grid). For a contiguous band $J = \left[ u , \nu \right]$ of size $s = \nu - u + 1$ 9

$$
\left| \left. X _ { J } ^ { z } \right. _ { M } \right| \leq \frac { \pi } { M \rho ^ { \nu } } \sqrt { \frac { 1 - \rho ^ { 2 s } } { 1 - \rho ^ { 2 } } } U _ { J } ( z ) .\tag{98}
$$

Proof. For $0 < \alpha \leq 1$

$$
K _ { M } ( \alpha ) = e ^ { i ( M - 1 ) \alpha / 2 } \frac { \sin ( M \alpha / 2 ) } { M \sin ( \alpha / 2 ) } , \qquad | K _ { M } ( \alpha ) | \le \frac { \pi } { M \alpha } .\tag{99}
$$

Therefore Cauch<sub>y</sub>–Schwarz <sub>g</sub>ives

$$
\left| \left. X _ { J } ^ { z } \right. _ { M } \right| \leq { \frac { \pi } { M } } \left( \sum _ { n = u } ^ { \nu } \rho ^ { - 2 n } \right) ^ { 1 / 2 } U _ { J } ( z )\tag{100}
$$

$$
= \frac { \pi } { M \rho ^ { \nu } } \sqrt { \frac { 1 - \rho ^ { 2 s } } { 1 - \rho ^ { 2 } } U _ { J } ( z ) } .\tag{101}
$$

## C.6. Positional Response Bounds and Refinements

Th<sub>e proo</sub>f<sub>s an</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-spec</sub>ifi<sub>c re</sub>fi<sub>nemen</sub>t<sub>s</sub> i<sub>n</sub> thi<sub>s sec</sub>ti<sub>on use</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng appen</sub>di<sub>x-on</sub>l<sub>y s</sub>h<sub>or</sub>th<sub>an</sub>d<sub>:</sub>

$$
A _ { J } ( z ) : = \sum _ { n \in J } \left| z _ { n } \right. ,\tag{102}
$$

$$
\Delta _ { 1 } X _ { J } ^ { z } ( t ) : = X _ { J } ^ { z } ( t + 1 ) - X _ { J } ^ { z } ( t ) ,\tag{103}
$$

$$
g _ { n } : = \left| e ^ { i \omega _ { n } } - 1 \right| .\tag{104}
$$

## C.6.1. Auxiliary Non-High-Only Response Bound

Lemma C.2 (Non-high-only positional response has linear share cost). For every certified � and every nonzero ${ \mathfrak { z } } ,$

$$
\operatorname* { s u p } _ { t } \left| \Delta _ { 1 } X _ { N } ^ { z } ( t ) \right| \leq \frac { \Gamma _ { \varepsilon } ( \rho ) } { M } A _ { N } ( z ; M ) .\tag{105}
$$

Consequently, for every $\zeta > 0 ,$ , if $\begin{array} { r } { \operatorname* { s u p } _ { t } \left| \Delta _ { 1 } X _ { N } ^ { z } ( t ) \right| / U ( z ) \ge \zeta ; } \end{array}$ , then

$$
r _ { N } ( z ; M ) \geq { \frac { \zeta M } { \Gamma _ { \varepsilon } ( \rho ) { \sqrt { h } } } } .\tag{106}
$$

Proof. For every $n \in N$ <sub>,</sub> th<sub>e</sub> <sub>par</sub>titi<sub>on</sub> <sub>g</sub>i<sub>ves</sub> $\omega _ { n } < \Gamma _ { \varepsilon } ( \rho ) / M$ . Hence

$$
\left| \Delta _ { 1 } X _ { N } ^ { z } ( t ) \right| \le \sum _ { n \in N } g _ { n } \left| z _ { n } \right|\tag{107}
$$

$$
\leq \sum _ { n \in N } \omega _ { n } \left| z _ { n } \right|\tag{108}
$$

$$
\leq \frac { \Gamma _ { \varepsilon } ( \rho ) } { M } A _ { N } ( z ; M ) ,\tag{109}
$$

which <sub>p</sub>roves E<sub>q</sub>. (105). Cauch<sub>y</sub>–Schwarz and $| N | \leq h$ <sub>g</sub><sup>i</sup>ve

$$
A _ { N } ( z ; M ) \leq \sqrt { h } U _ { N } ( z ) = \sqrt { h } U ( z ) r _ { N } ( z ; M ) .\tag{110}
$$

Th<sub>ere</sub>f<sub>ore a norma</sub>li<sub>ze</sub>d <sub>non-</sub>hi<sub>g</sub>h <sub>response o</sub>f <sub>s</sub>i<sub>ze a</sub>t l<sub>eas</sub>t $\zeta$ i<sub>mp</sub>li<sub>es</sub>

$$
\zeta \leq \frac { \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } } { M } r _ { N } ( z ; M ) ,\tag{111}
$$

which <sub>p</sub>roves E<sub>q</sub>. (106).

Th<sub>e ga</sub>i<sub>n mu</sub>lti<sub>p</sub>li<sub>er</sub> i<sub>n</sub> thi<sub>s aux</sub>ili<sub>ary enve</sub>l<sub>ope</sub> i<sub>mproves on</sub> th<sub>e un</sub>i<sub>versa</sub>l <sub>po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se</sub> b<sub>oun</sub>d th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> <sub>cer</sub>tifi<sub>e</sub>d <sub>reg</sub>i<sub>me.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> $M \geq \Gamma _ { \varepsilon } ( \rho )$ i<sub>mp</sub>li<sub>es</sub>

$$
0 < \frac { \Gamma _ { \varepsilon } ( \rho ) } { M } \leq 1 < 2 ,\tag{112}
$$

<sub>w</sub>h<sub>ere</sub> 2 i<sub>s</sub> th<sub>e un</sub>i<sub>versa</sub>l b<sub>oun</sub>d <sub>on</sub> $| e ^ { i \omega } - 1 |$ . For the Qwen grid used in our experiments, $B = 1 0 ^ { 6 }$ <sub>an</sub>d $2 h = 1 2 8$ so $\rho = ( 1 0 ^ { 6 } ) ^ { - 1 / 6 4 } \approx 0 . 8 0 5 8 4 2$ <sub>.</sub> With th<sub>e</sub> fi<sub>xe</sub>d <sub>exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> $\varepsilon = 1 0 ^ { - \bar { 2 } }$ <sub>,</sub> thi<sub>s</sub> <sub>g</sub>i<sub>ves</sub>

$$
\Gamma _ { 1 0 ^ { - 2 } } ( \rho ) \approx 6 4 7 2 . 2 5 , \qquad \frac { \Gamma _ { 1 0 ^ { - 2 } } ( \rho ) } { 3 2 7 6 8 } \approx 0 . 1 9 7 5 .\tag{113}
$$

Th<sub>us</sub> th<sub>e</sub> <sub>exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> i<sub>s</sub> <sub>cer</sub>tifi<sub>e</sub>d<sub>,</sub> <sub>an</sub>d it<sub>s</sub> <sub>non-</sub>hi<sub>g</sub>h <sub>ga</sub>i<sub>n</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>s</sub> <sub>we</sub>ll b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub> t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>l <sub>response</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>.</sub>

## C.6.2. Proof of Lemma A.2

Lemma A.2 (Direct scale-to-share response envelope). For every certified integer $M \geq 2$ and every $a \neq 0 ,$

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { 0 \le m \le M - 2 } \frac { \left| X _ { \mathcal { F } } ^ { a } ( m + 1 ) - X _ { \mathcal { F } } ^ { a } ( m ) \right| } { U ( a ) } \le \underbrace { \frac { \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } } { M } } _ { \begin{array} { c } { n o n - h i g h ~ b a n d : } \\ { s m a l l ~ r o t a t i o n s } \end{array} } _ { \begin{array} { c } { s m a l l ~ r o t a t i o n s } \\ { s m a l l ~ r o t a t i o n } \end{array} } + \underbrace { R _ { \mathcal { F } } r _ { H } ( a ; M ) } _ { \begin{array} { c } { c o r t i f i e d - h i g h ~ b a n d : } \\ { c o e f f i c i e n t ~ n o r m } \end{array} } } & { } \\ { \le \frac { \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } } { M } + R _ { \mathcal { F } } r _ { H } ( a ; M ) . } \end{array}\tag{22}
$$

where $\begin{array} { r } { R _ { \mathcal { F } } : = \left( \sum _ { n \in \mathcal { F } } \left| e ^ { i \omega _ { n } } - 1 \right| ^ { 2 } \right) ^ { 1 / 2 } > 0 } \end{array}$ is the fixed full-grid adjacent-gain factor and is independent of �.

Proof. For every relative distance �, split the score diference over the non-high and certified-high bands. Th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y g</sub>i<sub>ves</sub>

$$
\left| \Delta _ { 1 } X _ { \mathcal { F } } ^ { a } ( t ) \right| \leq \left| \Delta _ { 1 } X _ { N } ^ { a } ( t ) \right| + \left| \Delta _ { 1 } X _ { H } ^ { a } ( t ) \right| .\tag{114}
$$

Lemma C.2 and Cauch<sub>y</sub>–Schwarz im<sub>p</sub>l<sub>y</sub>

$$
\left| \Delta _ { 1 } X _ { N } ^ { a } ( t ) \right| \leq \frac { \Gamma _ { \varepsilon } ( \rho ) } { M } A _ { N } ( a ; M ) \leq \frac { \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } } { M } U _ { N } ( a ) .\tag{115}
$$

F<sub>or</sub> th<sub>e</sub> <sub>cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h b<sub>an</sub>d<sub>,</sub> <sub>ano</sub>th<sub>er</sub> <sub>app</sub>li<sub>ca</sub>ti<sub>on</sub> <sub>o</sub>f C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>warz</sub> <sub>g</sub>i<sub>ves</sub>

$$
\left| \Delta _ { 1 } X _ { H } ^ { a } ( t ) \right| \leq \sum _ { n \in H } g _ { n } \left| a _ { n } \right| \leq \left( \sum _ { n \in H } g _ { n } ^ { 2 } \right) ^ { 1 / 2 } U _ { H } ( a ) \leq R _ { \mathcal { F } } U _ { H } ( a ) .\tag{116}
$$

Substitute these two bounds into E<sub>q</sub>. (114), divide b<sub>y</sub> $U ( a ) > 0 _ { : }$ <sub>, an</sub>d t<sub>a</sub>k<sub>e</sub> th<sub>e max</sub>i<sub>mum over</sub> th<sub>e</sub> i<sub>n</sub>t<sub>eger</sub> window. This <sub>p</sub>roves the first line of E<sub>q</sub>. (22); the second follows from $r _ { N } ( a ; M ) \leq 1$ □

## C.6.3. Proof of Corollary A.1

Corollary A.1 (Stable local-response floor on the certified-high norm share). For every certified integer $M \geq 2 ,$ every $a \neq 0 ,$ , and every $\zeta > 0 ,$ , if the score $X _ { \mathcal { F } } ^ { a }$ avoids the window-level failure in Eq. (15) at response level $\zeta ,$ then

$$
r _ { H } ( a ; M ) \geq \underline { { r } } _ { \mathrm { p o s } } ( M ; \zeta ) : = \frac { \left[ \zeta - \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } / M \right] _ { + } } { R _ { \mathcal { F } } } .\tag{23}
$$

where $R _ { \mathcal { F } }$ is thefixedfactor in Lemma A.2. If the right-hand side exceeds one, the requested response is impossible.

Proof. Combining the response requirement with Lemma A.2 yields

$$
R _ { \mathcal { F } } r _ { H } ( a ; M ) \ge \zeta - \frac { \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } } { M } .\tag{117}
$$

Because $r _ { H } ( a ; M ) \ge 0$ <sub>an</sub>d $R _ { \mathcal { F } } > 0 .$ <sub>,</sub> t<sub>a</sub>k<sub>e</sub> th<sub>e</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>par</sub>t <sub>an</sub>d di<sub>v</sub>id<sub>e</sub> b<sub>y</sub> $R _ { \mathcal { F } }$ to <sub>p</sub>rove E<sub>q</sub>. (23). A lower bound <sub>a</sub>b<sub>ove</sub> <sub>one</sub> <sub>con</sub>t<sub>ra</sub>di<sub>c</sub>t<sub>s</sub> $r _ { H } \leq 1$ □

## C.6.4. Exact-Gain Response Envelope

F<sub>or</sub> thi<sub>s</sub> <sub>appen</sub>di<sub>x</sub> <sub>on</sub>l<sub>y,</sub> <sub>a</sub>bb<sub>rev</sub>i<sub>a</sub>t<sub>e</sub> th<sub>e</sub> $\ S 2$ <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> di<sub>agnos</sub>ti<sub>c</sub> <sub>as</sub>

$$
{ \widehat { \overline { G } } } _ { M } ( a ) : = \operatorname* { m a x } _ { 0 \leq m \leq M - 2 } { \frac { \left| X _ { { \mathcal { F } } } ^ { a } ( m + 1 ) - X _ { { \mathcal { F } } } ^ { a } ( m ) \right| } { U ( a ) } } .\tag{118}
$$

W<sub>e a</sub>l<sub>so</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e con</sub>ti<sub>nuous norma</sub>li<sub>ze</sub>d <sub>enve</sub>l<sub>ope</sub>

$$
\overline { { \mathscr { G } } } ( a ) : = \operatorname* { s u p } _ { t \in \mathbb { R } } \frac { \left| \Delta _ { 1 } X _ { \mathcal { F } } ^ { a } ( t ) \right| } { U ( a ) } .\tag{119}
$$

T<sub>o</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> t<sub>wo</sub> <sub>s</sub>h<sub>arper</sub> <sub>re</sub>l<sub>axa</sub>ti<sub>ons,</sub> <sub>one</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-spec</sub>ifi<sub>c</sub> <sub>an</sub>d <sub>one</sub> d<sub>epen</sub>di<sub>ng</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> b<sub>an</sub>d <sub>norms,</sub> <sub>se</sub>t

$$
R _ { H } ( M ) : = \left( \sum _ { n \in H } g _ { n } ^ { 2 } \right) ^ { 1 / 2 } ,
$$

$$
R _ { N } ( M ) : = \left( \sum _ { n \in N } g _ { n } ^ { 2 } \right) ^ { 1 / 2 } ,
$$

$$
R _ { \mathcal { F } } ^ { 2 } = R _ { H } ( M ) ^ { 2 } + R _ { N } ( M ) ^ { 2 } .\tag{120}
$$

Th<sub>e</sub> f<sub>u</sub>ll<sub>-gr</sub>id f<sub>ac</sub>t<sub>or</sub> $R _ { \mathcal { F } }$ i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> L<sub>emma</sub> A<sub>.</sub>2 <sub>an</sub>d i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f th<sub>e par</sub>titi<sub>on.</sub> F<sub>or</sub> $a \neq 0$ <sub>w</sub>ith $U _ { H } ( a ) > 0$ <sub>a</sub>l<sub>so</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
w _ { N } ( a ; M ) : = \frac { \sum _ { n \in N } g _ { n } | a _ { n } | } { U ( a ) } ,\tag{121}
$$

$$
\kappa _ { H } ( a ; M ) : = \frac { \sum _ { n \in H } g _ { n } \left| a _ { n } \right| } { U _ { H } ( a ) } ,\tag{122}
$$

$$
T ( a ) : = w _ { N } ( a ; M ) + \kappa _ { H } ( a ; M ) r _ { H } ( a ; M ) = \frac { \sum _ { n \in \mathcal { F } } g _ { n } \left| a _ { n } \right| } { U ( a ) } .\tag{123}
$$

Proposition 5 (Band-norm relaxations of the exact-gain envelope). For every certified integer $M \geq 2$ and every positional vector $a \neq 0$ with $U _ { H } ( a ) > 0 _ { : }$ , let $r = r _ { H } ( a ; M )$ and define

$$
F _ { M } ( r ) : = R _ { N } ( M ) \sqrt { 1 - r ^ { 2 } } + R _ { H } ( M ) r .\tag{124}
$$

Then

$$
\widehat { \overline { { G } } } _ { M } ( a ) \leq \overline { { \mathcal { G } } } ( a ) \leq T ( a ) = w _ { N } ( a ; M ) + \kappa _ { H } ( a ; M ) r \leq w _ { N } ( a ; M ) + R _ { H } ( M ) r \leq F _ { M } ( r ) .\tag{125}
$$

Proof. The integer-window maximum ranges over a subset of the continuous relative distances, so $\widehat { \overline { { G } } } _ { M } ( a ) \leq$ $\overline { { \mathcal { G } } } ( a )$ <sub>.</sub> F<sub>or</sub> <sub>every</sub> �<sub>,</sub> th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>g</sub>i<sub>ves</sub>

$$
\frac { \left| \Delta _ { 1 } X _ { \mathcal { F } } ^ { a } ( t ) \right| } { U ( a ) } \leq \frac { \sum _ { n \in \mathcal { F } } g _ { n } \left| a _ { n } \right| } { U ( a ) } = T ( a ) ,\tag{126}
$$

<sub>an</sub>d h<sub>ence</sub> ${ \overline { { G } } } ( a ) \leq T ( a )$ <sub>.</sub> Th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> $T ( a ) = w _ { N } ( a ; M ) + \kappa _ { H } ( a ; M ) r$ is <sub>g</sub>iven b<sub>y</sub> E<sub>q</sub>. (123). Cauch<sub>y</sub>–Schwarz <sub>on</sub> th<sub>e</sub> hi<sub>g</sub>h b<sub>an</sub>d <sub>g</sub>i<sub>ves</sub>

$$
\kappa _ { H } ( a ; M ) \leq \left( \sum _ { n \in H } g _ { n } ^ { 2 } \right) ^ { 1 / 2 } = R _ { H } ( M ) ,\tag{127}
$$

<sub>an</sub>d h<sub>ence</sub>

$$
T ( a ) \leq w _ { N } ( a ; M ) + R _ { H } ( M ) \frac { U _ { H } ( a ) } { U ( a ) } = w _ { N } ( a ; M ) + R _ { H } ( M ) r .\tag{128}
$$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Cauch<sub>y</sub>–Schwarz to the non-hi<sub>g</sub>h <sub>p</sub>art <sub>g</sub>ives

$$
w _ { N } ( a ; M ) \leq R _ { N } ( M ) \frac { U _ { N } ( a ) } { U ( a ) } = R _ { N } ( M ) \sqrt { 1 - r ^ { 2 } } .\tag{129}
$$

Substitution <sub>p</sub>roves E<sub>q</sub>. (125).

Th<sub>e</sub> <sub>m</sub>iddl<sub>e</sub> <sub>quan</sub>tit<sub>y</sub> $T ( a )$ i<sub>s</sub> <sub>a</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-spec</sub>ifi<sub>c,</sub> <sub>p</sub>h<sub>ase-agnos</sub>ti<sub>c</sub> <sub>upper</sub> <sub>cer</sub>tifi<sub>ca</sub>t<sub>e</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d f<sub>rom</sub> th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y.</sub> Th<sub>e</sub> fi<sub>na</sub>l f<sub>unc</sub>ti<sub>on</sub> $F _ { M }$ i<sub>s an upper enve</sub>l<sub>ope</sub> th<sub>a</sub>t <sub>re</sub>t<sub>a</sub>i<sub>ns on</sub>l<sub>y</sub> th<sub>e</sub> t<sub>wo</sub> b<sub>an</sub>d <sub>norms.</sub> N<sub>e</sub>ith<sub>er</sub> i<sub>s a</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>o</sub>f $\mathbb { E } [ \widehat { \overline { { G } } } _ { M } \mid r _ { H } ]$ . In <sub>p</sub>articular<sub>,</sub>

$$
F _ { M } ( 0 ) = R _ { N } ( M ) , \qquad r _ { * } ( M ) : = \frac { R _ { H } ( M ) } { R _ { \mathcal { F } } } ,\tag{130}
$$

<sub>so</sub> th<sub>e enve</sub>l<sub>ope nee</sub>d <sub>no</sub>t <sub>van</sub>i<sub>s</sub>h <sub>a</sub>t <sub>zero cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h <sub>norm s</sub>h<sub>are an</sub>d i<sub>s</sub> i<sub>ncreas</sub>i<sub>ng on</sub>l<sub>y</sub> f<sub>or</sub> $0 \leq r < r _ { * } ( M )$

Corollary C.1 (Sharper coeficient-shape floor). For every certified integer $M \geq 2 ,$ every $a \neq 0$ with $U _ { H } ( a ) > 0 _ { : }$ and every $\zeta > 0 .$ , define

$$
f _ { \mathrm { p o s } } ( a ; M , \zeta ) : = \frac { [ \zeta - w _ { N } ( a ; M ) ] _ { + } } { \kappa _ { H } ( a ; M ) } .\tag{131}
$$

$I f \widehat { \overline { { G } } } _ { M } ( a ) \geq \zeta .$ , then

$$
r _ { H } ( a ; M ) \ge f _ { \mathrm { p o s } } ( a ; M , \zeta ) .\tag{132}
$$

If the right-hand side exceeds one, the requested response is impossible.

Proof. Let $r = r _ { H } ( a ; M )$ . The res<sub>p</sub>onse condition and E<sub>q</sub>. (125) <sub>g</sub>ive

$$
\zeta \leq w _ { N } ( a ; M ) + \kappa _ { H } ( a ; M ) r .\tag{133}
$$

Since $U _ { H } ( a ) > 0$ an<sup>d</sup> ever<sub>y</sub> $g _ { n }$ i<sub>s</sub> <sub>pos</sub>iti<sub>ve,</sub> $\kappa _ { \cal H } ( a ; M ) > 0$ . Rearran<sub>g</sub>in<sub>g</sub> <sub>p</sub>roves E<sub>q</sub>. (132).

Corollary C.2 (Monotone coeficient-specific relaxation). Under the assumptions of Corollary C.1, define

$$
f _ { \mathrm { p o s } } ^ { \mathrm { c } } ( a ; M , \zeta ) : = \frac { [ \zeta - w _ { N } ( a ; M ) ] _ { + } } { R \mathcal { F } } .\tag{134}
$$

$I f \widehat { \overline { { G } } } _ { M } ( a ) \geq \zeta ,$ then

$$
r _ { H } ( a ; M ) \ge f _ { \mathrm { p o s } } ( a ; M , \zeta ) \ge f _ { \mathrm { p o s } } ^ { \mathrm { c } } ( a ; M , \zeta ) \ge \underline { { r } } _ { \mathrm { p o s } } ( M ; \zeta ) .\tag{135}
$$

Forfixed � and $\zeta ,$ this relaxation is non-decreasing in �.

Proof. Cauchy–Schwarz gives $\kappa _ { H } ( a ; M ) \leq R _ { H } ( M ) \leq R _ { \mathcal { F } }$ <sub>,</sub> <sub>an</sub>d h<sub>ence</sub> $f _ { \mathrm { p o s } } ( a ; M , \zeta ) \geq f _ { \mathrm { p o s } } ^ { \mathrm { c } } ( a ; M , \zeta )$ <sub>.</sub> Th<sub>e</sub> l<sub>ea</sub>di<sub>ng</sub> l<sub>ower</sub> b<sub>oun</sub>d i<sub>n</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d hi<sub>erarc</sub>h<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> C<sub>oro</sub>ll<sub>ary</sub> C<sub>.</sub>1<sub>.</sub> M<sub>oreover,</sub> $g _ { n } \leq \omega _ { n } < \Gamma _ { \varepsilon } ( \rho ) / M$ <sup>f</sup>or ever<sub>y</sub> $n \in N$ <sub>,</sub> so Cauch<sub>y</sub>–Schwarz <sub>g</sub>ives

$$
w _ { N } ( a ; M ) \leq \frac { \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } } { M } ,\tag{136}
$$

so $f _ { \mathrm { p o s } } ^ { \mathrm { c } } ( a ; M , \zeta ) \ge \underline { { r } } _ { \mathrm { p o s } } ( M ; \zeta )$ . As � increases<sub>,</sub> $N ( M )$ <sub>s</sub>h<sub>r</sub>i<sub>n</sub>k<sub>s.</sub> Th<sub>ere</sub>f<sub>ore</sub> $w _ { N } ( a ; M )$ i<sub>s non-</sub>i<sub>ncreas</sub>i<sub>ng, w</sub>hil<sub>e</sub> $R _ { \mathcal { F } }$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f th<sub>e</sub> <sub>par</sub>titi<sub>on.</sub> H<sub>ence</sub> $f _ { \mathrm { p o s } } ^ { \mathrm { c } } ( a ; M , \zeta )$ i<sub>s</sub> <sub>non-</sub>d<sub>ecreas</sub>i<sub>ng.</sub> □

## C.6.5. General Separations

For a se<sub>p</sub>aration $\ell > 0 .$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\Delta _ { \ell } X _ { J } ^ { z } ( t ) : = X _ { J } ^ { z } ( t + \ell ) - X _ { J } ^ { z } ( t ) ,\tag{137}
$$

$$
\beta _ { N } ( M , \ell ) : = \mathop { \operatorname* { m a x } } _ { n \in N } \left| e ^ { i \ell \omega _ { n } } - 1 \right| , \qquad R _ { H } ( M , \ell ) : = \left( \sum _ { n \in H } \left| e ^ { i \ell \omega _ { n } } - 1 \right| ^ { 2 } \right) ^ { 1 / 2 } .\tag{138}
$$

Proposition 6 (Response bound at a general separation). For every �,

$$
\left| \Delta _ { \ell } X _ { \mathcal { F } } ^ { z } ( t ) \right| \leq \beta _ { N } ( M , \ell ) A _ { N } ( z ; M ) + R _ { H } ( M , \ell ) U _ { H } ( z ) \leq \frac { \ell \Gamma _ { \varepsilon } ( \rho ) } { M } A _ { N } ( z ; M ) + R _ { H } ( M , \ell ) U _ { H } ( z ) .\tag{139}
$$

Proof. The first inequality follows from the triangle inequality on � and Cauchy–Schwarz on �. For the <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y, use</sub> $\left| e ^ { i \ell \omega _ { n } } - 1 \right| \leq \ell \omega _ { n } < \ell \Gamma _ { \varepsilon } ( \rho ) / M$ on �. □

## C.6.6. Coarse Constant Floor

Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>coarser</sub> f<sub>orm</sub> i<sub>s</sub> <sub>some</sub>ti<sub>mes</sub> <sub>conven</sub>i<sub>en</sub>t <sub>w</sub>h<sub>en</sub> <sub>on</sub>l<sub>y</sub> <sub>un</sub>if<sub>orm</sub> <sub>norm</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>s</sub> <sub>are</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e.</sub>

Corollary C.3 (Constant positional-share floor). Assume $A _ { N } ( a ; M ) \leq A ;$ <sub>∗</sub> and $U ( a ) \leq U _ { * }$ . If

$$
M \geq \operatorname* { m a x } \left\{ \Gamma _ { \varepsilon } ( \rho ) , \frac { 2 \Gamma _ { \varepsilon } ( \rho ) A _ { * } } { \eta } \right\}\tag{140}
$$

and $\left| \Delta _ { 1 } X _ { \mathcal { F } } ^ { a } ( t ) \right| \geq \eta ,$ , then

$$
r _ { H } ( a ; M ) \geq \frac { \eta } { 4 \sqrt { h } U _ { * } } .\tag{141}
$$

Proof. Taking $\ell = 1$ in E<sub>q</sub>. (139), the assumed lower bound on the window len<sub>g</sub>th makes the non-hi<sub>g</sub>h term at most $\eta / 2$ <sub>.</sub> Th<sub>us</sub> $R _ { H } ( M ) U _ { H } ( a ) \geq \eta / 2$ <sub>.</sub> Di<sub>v</sub>id<sub>e</sub> b<sub>y</sub> $U ( a ) \leq U _ { * }$ <sub>∗ an</sub>d <sub>use</sub> $R _ { H } ( M ) \leq 2 \sqrt { h }$ □

## C.7. Proof of Theorem 2

Theorem 2 (Near–far ordering approaches chance under full calibration). Fix afully rotary head together with fixed query and key content states that induce a nonzero coeficient vector �. Independently draw �<sub>near</sub> uniformlyfrom $\{ M + 1 , \ldots , 2 M \}$ and $m _ { \mathrm { f a r } }$ uniformlyfrom $\{ 2 M + 1 , \ldots , 3 M \}$ . Suppose the two interval score laws lie in the asymptotic fully-high calibration regime of Appendix B.1.3. Then

$$
p _ { \mathrm { o r d } } ( a ; M ) : = \mathbb { P } [ X _ { \mathcal { F } } ^ { a } ( m _ { \mathrm { f a r } } ) > X _ { \mathcal { F } } ^ { a } ( m _ { \mathrm { n e a r } } ) ] \longrightarrow \frac 1 2 \qquad ( M \to \infty ) .\tag{39}
$$

Proof. We compare both translated-window laws with a common Gaussian reference law. We first replace th<sub>e near-w</sub>i<sub>n</sub>d<sub>ow</sub> CDF i<sub>n</sub> th<sub>e or</sub>d<sub>er</sub>i<sub>ng pro</sub>b<sub>a</sub>bilit<sub>y,</sub> th<sub>en rep</sub>l<sub>ace</sub> th<sub>e</sub> f<sub>ar-w</sub>i<sub>n</sub>d<sub>ow</sub> CDF<sub>, an</sub>d fi<sub>na</sub>ll<sub>y pass</sub> t<sub>o</sub> th<sub>e</sub> asy<sup>m</sup>p<sup>t</sup>o<sup>ti</sup>c <sup>r</sup>eg<sup>im</sup>e.

Let $Y _ { \mathrm { n e a r } }$ <sub>an</sub>d $Y _ { \mathrm { f a r } }$ be the inde<sub>p</sub>endent interval scores in E<sub>q</sub>. (39), with CDFs $F _ { \mathrm { n e a r } }$ <sub>an</sub>d $F _ { \mathrm { f a r } }$ <sub>.</sub> Si<sub>nce</sub> b<sub>o</sub>th t<sub>rans</sub>l<sub>a</sub>t<sub>e</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ows</sub> h<sub>ave</sub> l<sub>eng</sub>th �<sub>, are</sub> f<sub>u</sub>ll<sub>y</sub> hi<sub>g</sub>h<sub>, an</sub>d <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d G<sub>auss</sub>i<sub>an-s</sub>h<sub>ape ca</sub>lib<sub>ra</sub>ti<sub>on,</sub> P<sub>ropos</sub>iti<sub>on</sub> 1 <sub>compares</sub> th<sub>em w</sub>ith th<sub>e same con</sub>ti<sub>nuous</sub> G<sub>auss</sub>i<sub>an re</sub>f<sub>erence</sub> CDF

$$
G _ { a } ( x ) : = \Phi _ { \mathrm { s t d } } \left( \frac { \sqrt { 2 } x } { U ( a ) } \right) .\tag{142}
$$

S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y,</sub>

$$
\operatorname* { s u p } _ { x } | F _ { \mathrm { n e a r } } ( x ) - G _ { a } ( x ) | \leq \eta _ { \mathrm { n e a r } } : = \tau _ { \mathrm { n e a r } } ( a ; M ) + c _ { \mathrm { G } } ( \varepsilon ) ,
$$

$$
\operatorname* { s u p } _ { x } | F _ { \mathrm { f a r } } ( x ) - G _ { a } ( x ) | \leq \eta _ { \mathrm { f a r } } : = \tau _ { \mathrm { f a r } } ( a ; M ) + c _ { \mathrm { G } } ( \varepsilon ) .\tag{143}
$$

U<sub>s</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>e</sub>ft li<sub>m</sub>it <sub>o</sub>f th<sub>e</sub> <sub>near-</sub>i<sub>n</sub>t<sub>erva</sub>l CDF<sub>,</sub> th<sub>e</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> <sub>even</sub>t i<sub>s</sub>

$$
p _ { \mathrm { o r d } } ( a ; M ) = \int _ { \mathbb { R } } F _ { \mathrm { n e a r } } ( y ^ { - } ) d F _ { \mathrm { f a r } } ( y ) .\tag{144}
$$

C<sub>on</sub>ti<sub>nu</sub>it<sub>y</sub> <sub>o</sub>f $G _ { a }$ and E<sub>q</sub>. (143) <sub>g</sub>ive

$$
\left| p _ { \mathrm { o r d } } ( a ; M ) - \int _ { \mathbb { R } } G _ { a } ( y ) d F _ { \mathrm { f a r } } ( y ) \right| \le \eta _ { \mathrm { n e a r } } .\tag{145}
$$

If � has CDF $G _ { a }$ <sub>an</sub>d i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $Y _ { \mathrm { f a r . } }$ <sub>,</sub> th<sub>en</sub>

$$
\begin{array} { r l } { \displaystyle \int _ { \mathbb { R } } G _ { a } ( y ) d F _ { \mathrm { f a r } } ( y ) = \mathbb { P } [ Z < Y _ { \mathrm { f a r } } ] } \\ { \displaystyle } & { = \int _ { \mathbb { R } } \left( 1 - F _ { \mathrm { f a r } } ( z ) \right) d G _ { a } ( z ) . } \end{array}\tag{146}
$$

The second ine<sub>q</sub>ualit<sub>y</sub> in E<sub>q</sub>. (143) and $\int ( 1 - G _ { a } ) d G _ { a } = 1 / 2$ th<sub>ere</sub>f<sub>ore</sub> i<sub>mp</sub>l<sub>y</sub>

$$
\left| \int _ { \mathbb { R } } G _ { a } ( y ) d F _ { \mathrm { f a r } } ( y ) - \frac { 1 } { 2 } \right| \leq \eta _ { \mathrm { f a r } } .\tag{147}
$$

C<sub>om</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>as</sub>t t<sub>wo</sub> b<sub>oun</sub>d<sub>s</sub> <sub>g</sub>i<sub>ves</sub>

$$
\left| p _ { \mathrm { o r d } } ( a ; M ) - \frac { 1 } { 2 } \right| \leq \tau _ { \mathrm { o r d } } ( a ; M , \varepsilon ) .\tag{148}
$$

Th<sub>e</sub> l<sub>e</sub>ft<sub>-</sub>CDF id<sub>en</sub>tit<sub>y</sub> h<sub>an</sub>dl<sub>es</sub> th<sub>e s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>even</sub>t <sub>w</sub>ith<sub>ou</sub>t <sub>a no-</sub>ti<sub>es assump</sub>ti<sub>on.</sub>

Fi<sub>na</sub>ll<sub>y,</sub> $\varepsilon _ { M } \to 0$ . Takin<sub>g</sub> $\varepsilon = \varepsilon _ { M }$ in E<sub>q</sub>. (148) and usin<sub>g</sub> $\tau _ { \mathrm { o r d } } ( a ; M , \varepsilon _ { M } ) \to 0$ <sub>p</sub>roves E<sub>q</sub>. (39).

## C.8. Proof of Lemma B.1

Lemma B.1 (Calibrated single-key score exceedance). For every certified �, nonzero $b ,$ and $S \in \mathbb { R }$ in the non-degenerate calibration regime of Appendix B.2.2,

$$
\begin{array} { r } { | p _ { \mathrm { e x c } } ( b ; S , M ) - \widehat { p } _ { \mathrm { e x c } } ( b ; S , M ) | \leq \tau _ { \mathrm { e x c } } ( b ; S , M ) . } \end{array}\tag{46}
$$

Proof. We first compare the exact and calibrated moments, then compare the true tail with the exact-moment G<sub>auss</sub>i<sub>an</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> <sub>s</sub>h<sub>ape</sub> di<sub>screpancy,</sub> <sub>an</sub>d fi<sub>na</sub>ll<sub>y</sub> <sub>propaga</sub>t<sub>e</sub> th<sub>e</sub> <sub>momen</sub>t <sub>rep</sub>l<sub>acemen</sub>t<sub>s</sub> throu<sub>g</sub>h the Gaussian CDF.

Let $\mu _ { H } : = \left. X _ { H } ^ { b } \right. _ { M }$ <sub>an</sub>d $\sigma _ { H } ^ { 2 } : = \mathrm { V a r } _ { M } ( X _ { H } ^ { b } )$ <sub>.</sub> Th<sub>e</sub> <sub>e</sub>x<sub>ac</sub>t <sub>su</sub>m $Y _ { b } = Y _ { N } + Y _ { H }$ <sub>g</sub><sup>i</sup>ves

$$
\mu _ { b } = \mu _ { N } + \mu _ { H } , \qquad \sigma _ { b } ^ { 2 } = \sigma _ { N } ^ { 2 } + \sigma _ { H } ^ { 2 } + 2 \kappa _ { N H } .\tag{149}
$$

P<sub>ropos</sub>iti<sub>on</sub> 1<sub>,</sub> <sub>app</sub>li<sub>e</sub>d <sub>w</sub>ith $z = b$ <sub>,</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\begin{array} { r } { | \mu _ { b } - \mu _ { N } | \leq \varepsilon U _ { H } ( b ) , \qquad \big | \sigma _ { b } ^ { 2 } - \widehat { \sigma } _ { b } ^ { 2 } \big | \leq \varepsilon U _ { H } ( b ) ^ { 2 } . } \end{array}\tag{150}
$$

Th<sub>e</sub> <sub>assume</sub>d <sub>var</sub>i<sub>ance</sub> fl<sub>oor</sub> i<sub>mp</sub>li<sub>es</sub> $\sigma _ { b } \ge s _ { - } ( b ; M ) > 0$

B<sub>y</sub> the definition in E<sub>q</sub>. (52) and the identit<sub>y</sub> $1 - \Phi _ { \mathrm { s t d } } ( x ) = \Phi _ { \mathrm { s t d } } ( - x )$ ,

$$
\left| p _ { \mathrm { e x c } } ( b ; S , M ) - \Phi _ { \mathrm { s t d } } \left( \frac { \mu _ { b } - S } { \sigma _ { b } } \right) \right| \leq \Delta _ { \mathrm { e x c } } ( b ; M ) .\tag{151}
$$

A <sub>no-</sub>ti<sub>es</sub> <sub>assump</sub>ti<sub>on</sub> i<sub>s</sub> <sub>unnecessary</sub> b<sub>ecause</sub> $p _ { \mathrm { e x c } } = 1 - \mathbb { P } [ Y _ { b } \leq S ]$ <sub>uses exac</sub>tl<sub>y</sub> th<sub>e c</sub>l<sub>ose</sub>d<sub>-</sub>th<sub>res</sub>h<sub>o</sub>ld CDF a<sub>pp</sub>earin<sub>g</sub> in E<sub>q</sub>. (52).

Th<sub>e</sub> <sub>momen</sub>t b<sub>oun</sub>d<sub>s</sub> <sub>a</sub>l<sub>so</sub> i<sub>mp</sub>l<sub>y</sub>

$$
\begin{array} { r l r } {  {  \frac { \mu _ { b } - S } { \sigma _ { b } } - \frac { \mu _ { N } - S } { \widehat { \sigma } _ { b } }  \leq \frac {  \mu _ { b } - \mu _ { N }  } { \sigma _ { b } } +  S - \mu _ { N }   \frac { 1 } { \sigma _ { b } } - \frac { 1 } { \widehat { \sigma } _ { b } }  } } \\ & { } & { \leq \frac { \varepsilon U _ { H } ( b ) } { s _ { - } ( b ; M ) } + \frac {  S - \mu _ { N }  \varepsilon U _ { H } ( b ) ^ { 2 } } { s _ { - } ( b ; M ) \widehat { \sigma } _ { b } ( s _ { - } ( b ; M ) + \widehat { \sigma } _ { b } ) } . } \end{array}\tag{152}
$$

H<sub>e</sub>r<sub>e</sub> w<sub>e use</sub>d $\left| \sigma _ { b } - \widehat { \sigma } _ { b } \right| = \left| \sigma _ { b } ^ { 2 } - \widehat { \sigma } _ { b } ^ { 2 } \right| / ( \sigma _ { b } + \widehat { \sigma } _ { b } )$ . Since $\Phi _ { \mathrm { { s t d } } }$ i<sub>s</sub> $1 / { \sqrt { 2 \pi } }$ -Li<sub>p</sub>schitz, combinin<sub>g</sub> E<sub>q</sub>. (151) and E<sub>q</sub>. (152) <sub>p</sub>roves E<sub>q</sub>. (46). □

## C.9. Proof of Corollary B.1

Corollary B.1 (Small score-barrier exceedance imposes a certified-high norm-share ceiling). Under the hypotheses of Lemma B.1, assume $U _ { N } ( b ) U _ { H } ( b ) > 0$ and $0 < \delta + \tau _ { \mathrm { e x c } } ( b ; S , M ) < 1 / 2$ . Then

$$
p _ { \mathrm { e x c } } ( b ; S , M ) \leq \delta \quad \Longrightarrow \quad r _ { H } ( b ; M ) \leq u _ { \mathrm { e x c } } ( b ; S , M , \delta ) < 1 .\tag{47}
$$

Proof. The proof converts the probability constraint into a variance bound, rewrites that bound in the share <sub>coor</sub>di<sub>na</sub>t<sub>e,</sub> <sub>an</sub>d th<sub>en</sub> <sub>so</sub>l<sub>ves</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> <sub>qua</sub>d<sub>ra</sub>ti<sub>c</sub> i<sub>nequa</sub>lit<sub>y.</sub>

If $p _ { \mathrm { e x c } } ( b ; S , M ) \leq \delta$ <sub>,</sub> then Lemma B.1 <sub>g</sub>ives

$$
\widehat { p } _ { \mathrm { e x c } } ( b ; S , M ) \leq \delta + \tau _ { \mathrm { e x c } } ( b ; S , M ) < \frac 1 2 .\tag{153}
$$

B<sub>y</sub> monotonicit<sub>y</sub> and s<sub>y</sub>mmetr<sub>y</sub> of the standard Gaussian CDF<sub>,</sub>

$$
S - \mu _ { N } \geq \xi _ { \mathrm { e x c } } \widehat \sigma _ { b } > 0 , \qquad \frac { \widehat \sigma _ { b } ^ { 2 } } { U _ { N } ( b ) ^ { 2 } } \leq \frac { \lambda _ { S } ^ { 2 } } { \xi _ { \mathrm { e x c } } ^ { 2 } } .\tag{154}
$$

In <sub>p</sub>articular<sub>,</sub> $\lambda _ { S } > 0$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub>

$$
t : = \frac { U _ { H } ( b ) } { U _ { N } ( b ) } = \frac { r _ { H } ( b ; M ) } { \sqrt { 1 - r _ { H } ( b ; M ) ^ { 2 } } } > 0 .\tag{155}
$$

Substitutin<sub>g</sub> the normalized descri<sub>p</sub>tors into E<sub>q</sub>. (154) <sub>g</sub>ives the <sub>q</sub>uadratic condition

$$
\frac { 1 } { 2 } t ^ { 2 } + 2 \gamma _ { N H } t + \beta _ { N } \leq \frac { \lambda _ { S } ^ { 2 } } { \xi _ { \mathrm { e x c } } ^ { 2 } } .\tag{156}
$$

Th<sub>e</sub> <sub>ac</sub>t<sub>ua</sub>l $t > 0$ is feasible, so the discriminant in E<sub>q</sub>. (55) is nonne<sub>g</sub>ative and its u<sub>pp</sub>er root $\overline { { t } } _ { \mathrm { e x c } }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $t \leq \bar { t } _ { \mathrm { e x c } } ;$ i<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub> $\bar { t } _ { \mathrm { e x c } } > 0$ . The ma<sub>p</sub> $t \mapsto t / \sqrt { 1 + t ^ { 2 } }$ i<sub>s s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> i<sub>ncreas</sub>i<sub>ng on</sub> $[ 0 , \infty )$ , which <sub>p</sub>roves E<sub>q</sub>. (47) <sub>an</sub>d <sub>a</sub>l<sub>so</sub> <sub>g</sub>i<sub>ves</sub> $u _ { \mathrm { e x c } } < 1$ □

## C.10. Proof of Theorem 1

Theorem 1 (Precise context-length limit on joint reliability). Fix a fully rotary head with the geometric frequency grid in Eq. (8), a tolerance $0 < \varepsilon < 1 / 2 ,$ , and a nonzero ordered margin $D ( m ) = X _ { \mathcal { F } } ^ { d } ( m )$ with $D ( 0 ) > 0$ Let $\mathfrak { M } _ { \mathrm { c a l } }$ be a finite nonempty set of certified integer windows $M \geq 2 ,$ with the same coeficients � at every window. Fix $0 \le \delta < 1 / 2 , 0 < \zeta < R _ { \mathcal { F } }$ , and $0 \leq \overline { { \tau } } _ { \mathrm { G } } < 1 / 2$ . Assume, for every $M \in \mathfrak { M } _ { \mathrm { c a l } }$ , that $\sigma _ { D } ( M ) > 0$ and

$$
\Delta _ { \mathrm { r e v } } ( d ; M ) = \left| p _ { \mathrm { r e v } } ( d ; M ) - \Phi _ { \mathrm { s t d } } \bigg ( - \frac { \mu _ { D } ( M ) } { \sigma _ { D } ( M ) } \bigg ) \right| \leq \overline { { \tau } } _ { \mathrm { G } } .
$$

Let $\underline { { p } } _ { \mathrm { r e v } } ( M ; \zeta , \overline { { \tau } } _ { \mathrm { G } } )$ be the explicit bound in Eq. (30). This bound is non-decreasing across ${ \mathfrak { M } } _ { \mathrm { c a l } }$ , and each calibrated window retaining the adjacent response ma $\begin{array} { r } { \mathtt { X } _ { 0 \le m \le M - 2 } g _ { d } ( m ) \ge \zeta } \end{array}$ satisfies

$$
p _ { \mathrm { r e v } } ( d ; M ) \geq \underline { { p } } _ { \mathrm { r e v } } ( M ; \zeta , \overline { { \tau } } _ { \mathrm { G } } ) .
$$

Write $M _ { - } : = \operatorname* { m i n } \mathfrak { M } _ { \mathrm { c a l } }$ and $M _ { + } : = \operatorname* { m a x } { \mathfrak { M } } _ { \mathrm { c a l } }$ . If the reversal tolerance satisfies

$$
\underline { { { p } } } _ { \mathrm { r e v } } ( M _ { - } ) \leq \delta < \underline { { { p } } } _ { \mathrm { r e v } } ( M _ { + } ) ,\tag{35}
$$

define $M _ { \mathrm { m a x } } : = M _ { \mathrm { m a x } } ^ { \mathrm { c e r t } }$ by Eq. (34). Then, for every $M \in \mathfrak { M } _ { \mathrm { c a l } }$ with $M > M _ { \mathrm { m a x } }$

$$
p _ { \mathrm { r e v } } ( d ; M ) > \delta \qquad o r \qquad \operatorname* { m a x } _ { 0 \leq m \leq M - 2 } g _ { d } ( m ) < \zeta .
$$

Thus $M _ { \mathrm { m a x } }$ is an upper bound on joint score-level reliability within the calibrated family. A window at or below $M _ { \mathrm { m a x } }$ remains a candidate; the bound alone does not establish joint reliability there.

An explicit suficient condition for the strict upper inequality in Eq. (35) is $M _ { \mathrm { + } } \geq M _ { \mathrm { s u f f } }$ and $0 \leq \delta < \delta _ { \star }$ , where

$$
\delta _ { \star } : = \left[ \Phi _ { \mathrm { s t d } } \left( - \frac { \varepsilon } { \sqrt { 1 / 2 - \varepsilon } } \right) - \overline { { \tau } } _ { \mathrm { G } } \right] _ { + } ,\tag{36}
$$

$$
\omega _ { \mathrm { m i n } } : = \rho ^ { h - 1 } ,
$$

$$
M _ { \mathrm { s u f f } } : = \operatorname* { m a x } \left\{ 2 , \left\lceil \frac { \Gamma _ { \varepsilon } ( \rho ) } { \omega _ { \mathrm { m i n } } } \right\rceil , \left\lfloor \frac { \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } } { \zeta } \right\rfloor + 1 \right\} .\tag{37}
$$

Indeed, $\underline { { p } } _ { \mathrm { r e v } } ( M ) \leq \delta _ { \star } < 1 / 2$ at every calibrated window, with equality for each such window $M \geq M _ { \mathrm { s u f f } }$ . The lower inequality $\underline { { { p } } } _ { \mathrm { r e v } } ( M _ { - } ) \leq \delta$ remains required.

Proof. Fix a calibrated window at which the same margin � retains normalized adjacent response �. Corollar<sub>y</sub> A.1 a<sub>pp</sub>lied to � <sub>g</sub>ives

$$
r : = r _ { H } ( d ; M ) \geq \ell ( M ; \zeta ) .\tag{157}
$$

Write $N = N _ { \varepsilon } ( M )$ <sub>an</sub>d $H = \mathcal { H } _ { \varepsilon } ( M )$ . The hi<sub>g</sub>h-band moment bounds in E<sub>q</sub>. (18) im<sub>p</sub>l<sub>y</sub>

$$
\left| \left. X _ { H } ^ { d } \right. _ { M } \right| \leq \varepsilon U _ { H } ( d ) , \qquad { \sqrt { \operatorname { V a r } _ { M } ( X _ { H } ^ { d } ) } } \geq c _ { \varepsilon } U _ { H } ( d ) .\tag{158}
$$

F<sub>or</sub> th<sub>e non-</sub>hi<sub>g</sub>h b<sub>an</sub>d<sub>,</sub> C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>warz g</sub>i<sub>ves</sub>

$$
\left| \left. X _ { N } ^ { d } \right. _ { M } \right| \leq \sum _ { n \in N } \left| d _ { n } \right| \leq \sqrt { k _ { N } ( M ) } U _ { N } ( d ) , \qquad \sqrt { \mathrm { V a r } _ { M } ( X _ { N } ^ { d } ) } \leq \sqrt { k _ { N } ( M ) } U _ { N } ( d ) .\tag{159}
$$

Th<sub>e</sub> <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>ecause</sub> <sub>cen</sub>t<sub>er</sub>i<sub>ng</sub> <sub>canno</sub>t i<sub>ncrease</sub> th<sub>e</sub> <sub>roo</sub>t <sub>mean</sub> <sub>square</sub> <sub>an</sub>d $\begin{array} { r } { \left| X _ { N } ^ { d } ( m ) \right| \le \sum _ { n \in N } | d _ { n } | } \end{array}$ <sub>po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se.</sub>

Since $U _ { H } ( d ) = U ( d ) r$ <sub>an</sub>d $U _ { N } ( d ) = U ( d ) \sqrt { 1 - r ^ { 2 } }$ <sub>,</sub> th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> <sub>an</sub>d <sub>reverse</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>liti<sub>es</sub> i<sub>n</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow</sub> $L _ { 2 }$ <sub>y</sub>i<sub>e</sub>ld

$$
| \mu _ { D } ( M ) | \leq U ( d ) u _ { M } ( r ) , \qquad \sigma _ { D } ( M ) \geq U ( d ) \nu _ { M } ( r ) .\tag{160}
$$

Let $g _ { M } ( s ) : = c _ { \varepsilon } s - \sqrt { k _ { N } ( M ) } \sqrt { 1 - s ^ { 2 } } ;$ this function is non-decreasing on [0, 1] and $\nu _ { M } ( s ) = [ g _ { M } ( s ) ] _ { + } . \mathrm { I f } \nu _ { M } ( r ) = 0 \ :$ th<sub>en</sub> $r \geq \ell ( M ; \zeta )$ i<sub>mp</sub>li<sub>es</sub> $\nu _ { M } ( \ell ( M ; \zeta ) ) = 0$ , so both envelo<sub>p</sub>e values in E<sub>q</sub>. (31) vanish and the desired bound f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $p _ { \mathrm { r e v } } ( d ; M ) \ge 0$ . Otherwise, E<sub>q</sub>. (25) and E<sub>q</sub>. (160) <sub>g</sub>ive

$$
\begin{array} { r } { p _ { \mathrm { r e v } } ( d ; M ) \geq \Phi _ { \mathrm { s t d } } \left( - \frac { \mu _ { D } \left( M \right) } { \sigma _ { D } \left( M \right) } \right) - \overline { { \tau } } _ { \mathtt { G } } } \\ { \geq \Phi _ { \mathrm { s t d } } \left( - \frac { u _ { M } ( r ) } { \nu _ { M } ( r ) } \right) - \overline { { \tau } } _ { \mathtt { G } } . } \end{array}\tag{161}
$$

F<sub>or</sub> th<sub>e</sub> <sub>secon</sub>d li<sub>ne,</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y,</sub> $- \mu _ { D } / \sigma _ { D } \geq - \left| \mu _ { D } \right| / \sigma _ { D } \geq - u _ { M } ( r ) / \nu _ { M } ( r )$ <sub>.</sub> Si<sub>nce</sub> <sub>a</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s</sub> <sub>nonnega</sub>ti<sub>ve,</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>par</sub>t <sub>o</sub>f th<sub>e</sub> l<sub>as</sub>t l<sub>ower</sub> b<sub>oun</sub>d <sub>proves</sub> $p _ { \mathrm { r e v } } ( d ; M ) \geq \psi _ { M } ( r ; \overline { { \tau } } _ { \mathrm { G } } )$

It <sub>rema</sub>i<sub>ns</sub> t<sub>o</sub> <sub>ver</sub>if<sub>y</sub> <sub>mono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y.</sub> O<sub>n</sub> th<sub>e</sub> <sub>reg</sub>i<sub>on</sub> $\nu _ { M } ( r ) > 0$ , se<sup>t</sup> $x ( r ) : = \sqrt { 1 - r ^ { 2 } } / r$ <sub>.</sub> Th<sub>e</sub>n

$$
\frac { u _ { M } ( r ) } { \nu _ { M } ( r ) } = \frac { \varepsilon + \sqrt { k _ { N } ( M ) } x ( r ) } { c _ { \varepsilon } - \sqrt { k _ { N } ( M ) } x ( r ) } .\tag{162}
$$

Th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> i<sub>s</sub> <sub>non-</sub>d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n</sub> $x ,$ <sub>w</sub>h<sub>ereas</sub> $x ( r )$ d<sub>ecreases</sub> <sub>w</sub>ith $r .$ <sub>.</sub> Th<sub>us</sub> $\psi _ { M } ( \boldsymbol { r } ; \overline { { \boldsymbol { \tau } } } _ { \mathrm { G } } )$ i<sub>s non-</sub>d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n</sub> $r \ \mathrm { { o n } }$ th<sub>e pos</sub>iti<sub>ve-</sub> $\boldsymbol { \cdot } \boldsymbol { \nu } _ { M }$ re<sub>g</sub>ion. On the <sub>p</sub>recedin<sub>g</sub> re<sub>g</sub>ion $\upsilon _ { M } = 0$ <sub>,</sub> it i<sub>s</sub> id<sub>en</sub>ti<sub>ca</sub>ll<sub>y zero.</sub> If $k _ { N } ( M ) > 0$ <sub>,</sub> th<sub>e</sub> <sub>ra</sub>ti<sub>o</sub> di<sub>verges a</sub>t th<sub>e</sub> b<sub>oun</sub>d<sub>ary an</sub>d it<sub>s</sub> G<sub>auss</sub>i<sub>an</sub> l<sub>ower</sub> t<sub>a</sub>il t<sub>en</sub>d<sub>s</sub> t<sub>o zero.</sub> If $k _ { N } ( M ) = 0$ <sub>,</sub> th<sub>e</sub> <sub>ra</sub>ti<sub>o</sub> <sub>equa</sub>l<sub>s</sub> $\varepsilon / c _ { \varepsilon }$ <sup>f</sup>or ever<sub>y</sub> $r > 0$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>prescr</sub>ib<sub>e</sub>d <sub>va</sub>l<sub>ue</sub> $\psi _ { M } ( 0 ; \overline { { \tau } } _ { \mathtt { G } } ) = 0$ <sub>a</sub>l<sub>so</sub> <sub>preserves</sub> <sub>mono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y.</sub> P<sub>os</sub>iti<sub>ve-par</sub>t <sub>c</sub>li<sub>pp</sub>i<sub>ng</sub> therefore <sub>p</sub>reserves <sub>g</sub>lobal monotonicit<sub>y</sub>. Combinin<sub>g</sub> this fact with E<sub>q</sub>. (157) <sub>p</sub>roves E<sub>q</sub>. (31).

Fi<sub>na</sub>ll<sub>y,</sub> $\ell ( M ; \zeta )$ i<sub>s non-</sub>d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n</sub> �<sub>, w</sub>hil<sub>e</sub> $k _ { N } ( M )$ i<sub>s non-</sub>i<sub>ncreas</sub>i<sub>ng</sub> b<sub>ecause</sub> th<sub>e cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h <sub>p</sub>refix ex<sub>p</sub>ands with �. The same ratio in E<sub>q</sub>. (162) de<sub>p</sub>ends on these <sub>q</sub>uantities throu<sub>g</sub>h $\sqrt { k _ { N } ( M ) } \sqrt { 1 - \ell ( M ; \zeta ) ^ { 2 } } / \ell ( M ; \zeta )$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>non-</sub>i<sub>ncreas</sub>i<sub>ng</sub> <sub>w</sub>h<sub>enever</sub> $\ell ( M ; \zeta ) ~ > ~ 0$ <sub>.</sub> Wh<sub>e</sub>n $\ell ( M ; \zeta ) ~ = ~ 0$ or $\nu _ { M } ( \ell ( M ; \zeta ) ) = 0$ <sub>,</sub> th<sub>e</sub> fl<sub>oor</sub> i<sub>s zero.</sub> Th<sub>ere</sub>f<sub>ore</sub> $\underline { { p } } _ { \mathrm { r e v } } ( M ; \zeta , \overline { { \tau } } _ { \mathrm { G } } )$ i<sub>s non-</sub>d<sub>ecreas</sub>i<sub>ng across</sub> ${ \mathfrak { M } } _ { \mathrm { c a l } }$ <sub>.</sub> It<sub>s m</sub>i<sub>n</sub>i<sub>mum</sub> <sub>an</sub>d <sub>max</sub>i<sub>mum</sub> <sub>over</sub> thi<sub>s</sub> fi<sub>n</sub>it<sub>e</sub> f<sub>am</sub>il<sub>y</sub> <sub>are</sub> <sub>a</sub>tt<sub>a</sub>i<sub>ne</sub>d <sub>a</sub>t �<sub>− an</sub>d $M _ { + }$ , res<sub>p</sub>ectivel<sub>y</sub>. Thus E<sub>q</sub>. (35) is e<sub>q</sub>uivalent to th<sub>e su</sub>b<sub>-</sub>th<sub>res</sub>h<sub>o</sub>ld <sub>an</sub>d <sub>cross</sub>i<sub>ng se</sub>t<sub>s</sub> b<sub>o</sub>th b<sub>e</sub>i<sub>ng nonemp</sub>t<sub>y.</sub> At <sub>an</sub>d b<sub>eyon</sub>d th<sub>e</sub> fi<sub>rs</sub>t <sub>sca</sub>l<sub>e</sub> $M _ { \dagger }$ in E<sub>q</sub>. (32), the l<sub>ower</sub> b<sub>oun</sub>d <sub>excee</sub>d<sub>s</sub> $\delta ,$ so a<sup>n</sup>y <sup>m</sup>a<sup>r</sup>g<sup>in</sup> <sup>r</sup>e<sup>t</sup>a<sup>inin</sup>g <sup>r</sup>espo<sup>n</sup>se $\zeta$ h<sub>as</sub> $p _ { \mathrm { r e v } } ( d ; M ) > \delta$ <sub>.</sub> Fi<sub>n</sub>it<sub>eness an</sub>d <sub>mono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y</sub> <sub>ma</sub>k<sub>e</sub> th<sub>e</sub> l<sub>arges</sub>t <sub>su</sub>b<sub>-</sub>th<sub>res</sub>h<sub>o</sub>ld <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> th<sub>e</sub> <sub>gr</sub>id <sub>po</sub>i<sub>n</sub>t i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> <sub>prece</sub>di<sub>ng</sub> $M _ { \dagger }$ <sub>an</sub>d <sub>ru</sub>l<sub>e</sub> <sub>ou</sub>t <sub>every</sub> l<sub>arger</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>sca</sub>l<sub>e.</sub> Thi<sub>s proves</sub> $M _ { \mathrm { m a x } } = M _ { \mathrm { m a x } } ^ { \mathrm { c e r t } }$ in E<sub>q</sub>. (34).

T<sub>o prove</sub> th<sub>e exp</sub>li<sub>c</sub>it <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>on, o</sub>b<sub>serve</sub> th<sub>a</sub>t <sub>w</sub>h<sub>enever</sub> $\nu _ { M } ( r ) > 0$ , E<sub>q</sub>. (162) and $x ( r ) \geq 0$ <sub>g</sub><sup>i</sup>ve $u _ { M } ( r ) / \nu _ { M } ( r ) \geq \varepsilon / c _ { \varepsilon }$ . Monotonicit<sub>y</sub> of the Gaussian CDF and <sub>p</sub>ositive-<sub>p</sub>art cli<sub>pp</sub>in<sub>g</sub> im<sub>p</sub>l<sub>y</sub> $\psi _ { M } ( r ; \overline { { \tau } } _ { \mathtt { G } } ) \leq \delta _ { \star }$ <sub>.</sub> Th<sub>e</sub> <sub>same</sub> b<sub>oun</sub>d h<sub>o</sub>ld<sub>s w</sub>h<sub>en</sub> $\begin{array} { r } { \nu _ { M } ( r ) = 0 . } \end{array}$ , since $\psi _ { M }$ th<sub>en van</sub>i<sub>s</sub>h<sub>es.</sub> C<sub>onsequen</sub>tl<sub>y</sub> $\underline { { p } } _ { \mathrm { r e v } } ( M ) \leq \delta _ { \star } < 1 / 2$ . For $M \geq M _ { \mathrm { { s u f f } } } .$ the definition in E<sub>q</sub>. (37) <sub>g</sub>ives $M \omega _ { \mathrm { m i n } } \geq \Gamma _ { \varepsilon } ( \rho )$ <sub>an</sub>d $M > \Gamma _ { \varepsilon } ( \rho ) \sqrt { h } / \zeta$ . Hence $k _ { N } ( M ) = 0$ <sub>an</sub>d $\ell ( M ; \zeta ) > 0$ , so

$$
\frac { u _ { M } ( \ell ( M ; \zeta ) ) } { \nu _ { M } ( \ell ( M ; \zeta ) ) } = \frac { \varepsilon } { c _ { \varepsilon } } , \qquad \underline { { { p } } } _ { \mathrm { r e v } } ( M ) = \delta _ { \star } .
$$

If $M _ { \mathrm { + } } \geq M _ { \mathrm { s u f f } }$ <sub>an</sub>d $0 \leq \delta < \delta _ { \star }$ , this e<sub>q</sub>ualit<sub>y</sub> <sub>p</sub>roves the strict u<sub>pp</sub>er ine<sub>q</sub>ualit<sub>y</sub> in E<sub>q</sub>. (35). Its lower ine<sub>q</sub>ualit<sub>y</sub> <sub>ensures</sub> th<sub>a</sub>t th<sub>e</sub> <sub>su</sub>b<sub>-</sub>th<sub>res</sub>h<sub>o</sub>ld <sub>se</sub>t i<sub>s</sub> <sub>a</sub>l<sub>so</sub> <sub>nonemp</sub>t<sub>y,</sub> <sub>comp</sub>l<sub>e</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub> □

## C.11. Proof of Theorem 3

Theorem 3 (Fully-high pairwise reversal approaches chance). For a fully rotary head, fix a query and an ordered pair of keys with coeficient vectors $a , b \in \mathbb { C } ^ { h }$ such that � $: = a - b \neq 0$ . For any $0 < \varepsilon < 1 / 2$ and any certified window satisfying $\mathcal { H } _ { \varepsilon } ( M ) = \mathcal { F } _ { . }$ , draw m ∼ Unif $\{ 0 , \ldots , M - 1 \}$ . If the Gaussian-shape condition in Eq. (19) holds for $z = d = a - b$ with tolerance $\tau _ { \mathrm { H , G } } ( d ; M )$ , then

$$
\begin{array} { c } { { p _ { \mathrm { r e v } } ( a , b ; M ) : = \mathbb { P } \left[ X _ { \mathcal { F } } ^ { a } ( \mathbf { m } ) < X _ { \mathcal { F } } ^ { b } ( \mathbf { m } ) \right] = \mathbb { P } \left[ X _ { \mathcal { F } } ^ { a - b } ( \mathbf { m } ) < 0 \right] , } } \\ { { \displaystyle \left| p _ { \mathrm { r e v } } ( a , b ; M ) - \frac { 1 } { 2 } \right| \leq \tau _ { \mathrm { H , G } } ( a - b ; M ) + c _ { \mathrm { G } } ( \varepsilon ) . } } \end{array}\tag{56}
$$

Here $c _ { \mathsf { G } } ( \varepsilon )$ is the universal moment-replacement term defined in Eq. (21). For fixed finite ℎ, define $\varepsilon _ { M } : =$ $2 C _ { \mathrm { F } } / ( ( 1 - \rho ) M \omega _ { h - 1 } )$ for all suficiently large �. If the same shape calibration holds for the resulting fully-high partitions and $\tau _ { \mathrm { H , G } } ( a - b ; M ) \to 0 _ { \mathrm { \therefore } }$ , then

$$
\operatorname * { l i m } _ { M  \infty } p _ { \mathrm { r e v } } ( a , b ; M ) = \frac { 1 } { 2 } .\tag{57}
$$

Proof. We first derive the finite-window reversal bound and then obtain the asymptotic limit by letting the <sub>ca</sub>lib<sub>ra</sub>ti<sub>on error van</sub>i<sub>s</sub>h<sub>.</sub>

Fix $0 < \varepsilon < 1 / 2$ <sub>an</sub>d <sub>a w</sub>i<sub>n</sub>d<sub>ow</sub> f<sub>or w</sub>hi<sub>c</sub>h $\mathcal { H } _ { \varepsilon } ( M ) = \mathcal { F }$ . Because $d = a - b \neq 0$

$$
U _ { H } ( d ) = U ( d ) > 0 , \qquad Y _ { H } ^ { d } = X _ { \mathcal { F } } ^ { d } ( { \bf m } ) .\tag{163}
$$

U<sub>n</sub>d<sub>er</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an-s</sub>h<sub>ape</sub> <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> <sub>assume</sub>d i<sub>n</sub> th<sub>e</sub> th<sub>eorem,</sub> P<sub>ropos</sub>iti<sub>on</sub> 1<sub>,</sub> <sub>app</sub>li<sub>e</sub>d <sub>w</sub>ith $z = d$ , g<sup>i</sup>ves

$$
\operatorname* { s u p } _ { x \in \mathbb { R } } \left| \mathbb { P } [ X _ { \mathcal { F } } ^ { d } ( \mathbf { m } ) \leq x ] - \Phi _ { \mathrm { s t d } } \left( \frac { \sqrt { 2 } x } { U ( d ) } \right) \right| \leq \tau _ { \mathrm { H , G } } ( d ; M ) + c _ { \mathsf { G } } ( \varepsilon ) .\tag{164}
$$

Let $x _ { k } = - 1 / k$ <sub>.</sub> Th<sub>e</sub>n $x _ { k } \uparrow 0$ <sub>an</sub>d $\{ X _ { \mathcal { F } } ^ { d } ( \mathbf { m } ) < 0 \} = \bigcup _ { k \geq 1 } \{ X _ { \mathcal { F } } ^ { d } ( \mathbf { m } ) \leq x _ { k } \}$ <sub>.</sub> C<sub>on</sub>ti<sub>nu</sub>it<sub>y</sub> f<sub>rom</sub> b<sub>e</sub>l<sub>ow</sub> <sub>o</sub>f <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>an</sub>d <sub>con</sub>ti<sub>nu</sub>it<sub>y o</sub>f $\Phi _ { \mathrm { { s t d } } }$ <sub>a</sub>ll<sub>ow</sub> $k $ ∞ in E<sub>q</sub>. (164), which <sub>p</sub>roves E<sub>q</sub>. (56). This left-limit ar<sub>g</sub>ument controls th<sub>e s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>reversa</sub>l <sub>even</sub>t <sub>w</sub>ith<sub>ou</sub>t i<sub>mpos</sub>i<sub>ng a no-</sub>ti<sub>es assump</sub>ti<sub>on.</sub>

F<sub>or</sub> th<sub>e asymp</sub>t<sub>o</sub>ti<sub>c c</sub>l<sub>a</sub>i<sub>m,</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> th<sub>eorem g</sub>i<sub>ves</sub>

$$
\Gamma _ { \varepsilon _ { M } } ( \rho ) = M \omega _ { h - 1 } , \qquad \mathcal { H } _ { \varepsilon _ { M } } ( M ) = \mathcal { F } , \qquad \varepsilon _ { M } \longrightarrow 0 .\tag{165}
$$

A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow reversa</sub>l b<sub>oun</sub>d <sub>w</sub>ith $\varepsilon = \varepsilon _ { M }$ <sub>an</sub>d <sub>us</sub>i<sub>ng</sub> $\tau _ { \mathrm { H , G } } ( d ; M ) \to 0$ t<sub>oge</sub>th<sub>er w</sub>ith $c _ { \mathtt { G } } ( \varepsilon _ { M } ) \to 0$ <sub>p</sub>roves E<sub>q</sub>. (57). □

## D. Empirical Validation of the Theory

W<sub>e exam</sub>i<sub>ne</sub> th<sub>e numer</sub>i<sub>ca</sub>l <sub>cons</sub>i<sub>s</sub>t<sub>ency an</sub>d ti<sub>g</sub>ht<sub>ness o</sub>f th<sub>e</sub> l<sub>oca</sub>l<sub>-response</sub> b<sub>oun</sub>d<sub>s on</sub> fi<sub>xe</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>ors,</sub> th<sub>en</sub> d<sub>ocumen</sub>t th<sub>e</sub> f<sub>requency par</sub>titi<sub>on an</sub>d <sub>score-</sub>di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on ev</sub>id<sub>ence use</sub>d i<sub>n</sub> th<sub>e ma</sub>i<sub>n</sub> t<sub>ex</sub>t<sub>.</sub> Th<sub>e</sub> local-response experiment measures the maximum normalized adjacent gap that appears in the theoretical <sub>cer</sub>tifi<sub>ca</sub>t<sub>e.</sub> Th<sub>e</sub> t<sub>as</sub>k<sub>-</sub>l<sub>eve</sub>l P<sub>os</sub>iti<sub>ona</sub>l S<sub>core</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> G<sub>.</sub>1<sub>.</sub>1 <sub>averages</sub> th<sub>ese gaps over re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ances.</sub> All <sub>exper</sub>i<sub>men</sub>t<sub>s</sub> b<sub>e</sub>l<sub>ow</sub> h<sub>o</sub>ld <sub>cac</sub>h<sub>e</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>ors</sub> fi<sub>xe</sub>d <sub>w</sub>hil<sub>e vary</sub>i<sub>ng re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance over</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ows.</sub>

## D.1. Local Positional-Response Envelope and Local-Response Floor

W<sub>e au</sub>dit th<sub>e s</sub>h<sub>arper coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-spec</sub>ifi<sub>c re</sub>fi<sub>nemen</sub>t <sub>o</sub>f th<sub>e s</sub>t<sub>a</sub>bl<sub>e</sub> l<sub>oca</sub>l<sub>-response enve</sub>l<sub>ope, recor</sub>d<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>6<sub>.</sub>4<sub>.</sub> Thi<sub>s exper</sub>i<sub>men</sub>t <sub>uses</sub> th<sub>e</sub> th<sub>eorem-s</sub>id<sub>e c</sub>h<sub>o</sub>i<sub>ce</sub> $C _ { \mathrm { F } } = 2 \pi$ <sub>an</sub>d $\varepsilon = 1 0 ^ { - 2 }$ at $M = 3 2 7 6 8$ . We evaluate Qwen3-8B and Llama-3.1-8B using ten global queries, 200 sampled key tokens, and all 32 first-layer attention heads. Thus each model contributes 64,000 query–head–key coeficient vectors. Qwen3 has the <sub>g</sub>eometric RoPE <sub>g</sub>rid in E<sub>q</sub>. (8); the certified <sub>p</sub>artition has $| H | = 8$ <sub>an</sub>d $| N | = 5 6 ,$ <sub>, so</sub> thi<sub>s mo</sub>d<sub>e</sub>l li<sub>es</sub> di<sub>rec</sub>tl<sub>y</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>e assump</sub>ti<sub>ons o</sub>f th<sub>e ma</sub>i<sub>n</sub> th<sub>eory.</sub> F<sub>or</sub> Ll<sub>ama-</sub>3<sub>.</sub>1<sub>, we se</sub>l<sub>ec</sub>t th<sub>e</sub> l<sub>arges</sub>t <sub>pre</sub>fi<sub>x sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> th<sub>e rea</sub>li<sub>ze</sub>d <sub>s</sub>i<sub>gne</sub>d<sub>-</sub>f<sub>requency separa</sub>ti<sub>on con</sub>diti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> F<sub>our</sub>i<sub>er-</sub>f<sub>rame argumen</sub>t<sub>, w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub> $| H | = 9$ <sub>an</sub>d $| N | = 5 5$ Si<sub>nce</sub> Ll<sub>ama-</sub>3<sub>.</sub>1 <sub>uses sca</sub>l<sub>e</sub>d<sub>, non-geome</sub>t<sub>r</sub>i<sub>c</sub> R<sub>o</sub>PE f<sub>requenc</sub>i<sub>es, we repor</sub>t it<sub>s row as a s</sub>t<sub>ress</sub> t<sub>es</sub>t <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e</sub> <sub>geome</sub>t<sub>r</sub>i<sub>c-gr</sub>id <sub>assump</sub>ti<sub>ons.</sub> Thi<sub>s su</sub>b<sub>sec</sub>ti<sub>on au</sub>dit<sub>s</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>resu</sub>lt i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub>4<sub>.</sub>1<sub>;</sub> it d<sub>oes no</sub>t t<sub>es</sub>t th<sub>e</sub> t<sub>wo-</sub>i<sub>n</sub>t<sub>erva</sub>l <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e near–</sub>f<sub>ar</sub> th<sub>eorem o</sub>f A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>1<sub>.</sub>2<sub>.</sub>

F<sub>or</sub> <sub>eac</sub>h <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>or</sub> $a _ { i } ,$ <sub>,</sub> <sub>we</sub> <sub>use</sub> th<sub>e</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>gap</sub> $\zeta _ { i }$ <sub>an</sub>d it<sub>s</sub> <sub>po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se</sub> <sub>cer</sub>tifi<sub>ca</sub>t<sub>e</sub> $T ( a _ { i } )$ from Pro<sub>p</sub>osition 5. The <sub>p</sub>ro<sub>p</sub>osition and Corollar<sub>y</sub> C.1 re<sub>q</sub>uire<sub>,</sub> res<sub>p</sub>ectivel<sub>y,</sub>

$$
\zeta _ { i } \leq T ( a _ { i } ) , \qquad f _ { \mathrm { p o s } } ( a _ { i } ; M , \zeta _ { i } ) \leq r _ { H } ( a _ { i } ; M ) .\tag{166}
$$

Fi<sub>g</sub>ure 7 re<sub>p</sub>orts all 128<sub>,</sub>000 vectors. Ever<sub>y p</sub>oint satisfies both ine<sub>q</sub>ualities in E<sub>q</sub>. (166). In the left column, th<sub>e me</sub>di<sub>an</sub> ti<sub>g</sub>ht<sub>ness ra</sub>ti<sub>os</sub> $\zeta _ { i } / T ( a _ { i } )$ are 0.829 for Qwen3 and 0.788 for Llama-3.1. The middle column shows <sub>a s</sub>t<sub>rong mono</sub>t<sub>one assoc</sub>i<sub>a</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h <sub>norm s</sub>h<sub>are an</sub>d b<sub>o</sub>th <sub>responses:</sub> th<sub>e</sub> S<sub>pearman</sub> <sub>corre</sub>l<sub>a</sub>ti<sub>ons o</sub>f $r _ { H }$ <sub>w</sub>ith $\left( \zeta _ { i } , T ( a _ { i } ) \right)$ are (0.975, 0.969) for Qwen3 and (0.988, 0.989) for Llama-3.1. This panel d<sub>escr</sub>ib<sub>es</sub> th<sub>e samp</sub>l<sub>e</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>ors,</sub> b<sub>ecause</sub> $T ( a _ { i } )$ <sub>re</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e po</sub>i<sub>n</sub>t<sub>-spec</sub>ifi<sub>c quan</sub>titi<sub>es</sub> $w _ { N } ( a _ { i } ; M )$ <sub>an</sub>d $\kappa _ { H } ( a _ { i } ; M )$

I<sub>n</sub> th<sub>e r</sub>i<sub>g</sub>ht <sub>co</sub>l<sub>umn,</sub> th<sub>e me</sub>di<sub>an ra</sub>ti<sub>os</sub> $f _ { \mathrm { p o s } } ( a _ { i } ; M , \zeta _ { i } ) / r _ { H } ( a _ { i } ; M )$ are 0.782 for Qwen3 and 0.743 for Llama-3.1. Here $\zeta _ { i }$ i<sub>s</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>max</sub>i<sub>mum</sub> <sub>gap</sub> <sub>o</sub>f th<sub>e</sub> <sub>same</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>or</sub> <sub>use</sub>d t<sub>o</sub> <sub>compu</sub>t<sub>e</sub> $f _ { \mathrm { p o s } } .$ <sub>.</sub> Thi<sub>s</sub> i<sub>s</sub> <sub>a</sub> <sub>same-</sub> <sub>samp</sub>l<sub>e</sub> <sub>cons</sub>i<sub>s</sub>t<sub>ency</sub> <sub>an</sub>d ti<sub>g</sub>ht<sub>ness</sub> <sub>c</sub>h<sub>ec</sub>k<sub>.</sub> Th<sub>e</sub> <sub>prox</sub>i<sub>m</sub>it<sub>y</sub> <sub>o</sub>f th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d t<sub>o</sub> th<sub>e</sub> <sub>equa</sub>lit<sub>y</sub> li<sub>ne</sub> <sub>quan</sub>tifi<sub>es</sub> it<sub>s</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l ti<sub>g</sub>ht<sub>ness.</sub>

![](images/6753bc95d582173f5b56d79d099b6d70561894f40c7238e2ae6d29e7da8d7263.jpg)

![](images/3907049a311c24d708a698070668f76b55b6139db55c71102dbc05d51f6193e4.jpg)

![](images/67cfffad6e87e963206b78a20b997fd15998a7f7c75ea4436730b8f80ec7ca98.jpg)

(d)  
![](images/a06ac71fea84cbd24e8015a373c90ad7223731f69662e2f8de8e036fa9646d1c.jpg)

(e)  
![](images/f457811730a2f5b6ce3e447c160427a7be98b423e318f0b9ddb5ba588365e741.jpg)

(f)  
![](images/cf7b09ddb48d7ada89e282dc9e1d2b7fe8e5c9ceec08f857a206caf601a2d031.jpg)  
Fi<sub>gure</sub> 7<sub>:</sub> P<sub>o</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se cons</sub>i<sub>s</sub>t<sub>ency an</sub>d ti<sub>g</sub>ht<sub>ness au</sub>dit <sub>o</sub>f th<sub>e s</sub>h<sub>arper coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-spec</sub>ifi<sub>c</sub> l<sub>oca</sub>l <sub>pos</sub>iti<sub>ona</sub>l<sub>-response</sub> envelo<sub>p</sub>e and local-res<sub>p</sub>onse floor from the a<sub>pp</sub>endix. Panels (a)–(c) form the Qwen3-8B row and <sub>p</sub>anels (d)–(f) form the Llama-3.1-8B row; ever<sub>y p</sub>oint is one first-la<sub>y</sub>er <sub>q</sub>uer<sub>y</sub>–head–ke<sub>y</sub> coeficient vector. Left: th<sub>e o</sub>b<sub>serve</sub>d i<sub>n</sub>t<sub>eger-w</sub>i<sub>n</sub>d<sub>ow max</sub>i<sub>mum gap</sub> $\zeta _ { i }$ <sub>aga</sub>i<sub>ns</sub>t th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-spec</sub>ifi<sub>c upper cer</sub>tifi<sub>ca</sub>t<sub>e</sub> $T ( a _ { i } )$ on lo<sub>g</sub>arithmic axes. Middle: the observed <sub>g</sub>a<sub>p</sub> (blue) and u<sub>pp</sub>er certificate (oran<sub>g</sub>e) a<sub>g</sub>ainst $r _ { H } ( a _ { i } ; M ) _ { : }$ <sub>;</sub> <sub>so</sub>lid curves and shaded re<sub>g</sub>ions are medians and 10–90% ran<sub>g</sub>es over 32 e<sub>q</sub>ual-count $r _ { H }$ bi<sub>ns,</sub> <sub>no</sub>t <sub>an</sub> <sub>��-on</sub>l<sub>y</sub> <sub>pre</sub>di<sub>c</sub>t<sub>or.</sub> Ri<sub>g</sub>ht<sub>:</sub> th<sub>e po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se</sub> l<sub>ower</sub> b<sub>oun</sub>d $f _ { \mathrm { p o s } } ( a _ { i } ; M , \zeta _ { i } )$ <sub>aga</sub>i<sub>ns</sub>t th<sub>e o</sub>b<sub>serve</sub>d <sub>cer</sub>tifi<sub>e</sub>d<sub>-</sub>hi<sub>g</sub>h <sub>norm s</sub>h<sub>are.</sub> Dashed diagonals indicate equality in the left and right columns. The Qwen3 row is an in-assumption <sub>geome</sub>t<sub>r</sub>i<sub>c-gr</sub>id <sub>au</sub>dit<sub>, w</sub>h<sub>ereas</sub> th<sub>e non-geome</sub>t<sub>r</sub>i<sub>c</sub> Ll<sub>ama row</sub> i<sub>s an ou</sub>t<sub>-o</sub>f<sub>-assump</sub>ti<sub>on s</sub>t<sub>ress</sub> t<sub>es</sub>t<sub>.</sub> Th<sub>e r</sub>i<sub>g</sub>ht <sub>co</sub>l<sub>umn uses</sub> th<sub>e same po</sub>i<sub>n</sub>t t<sub>o se</sub>t $\zeta _ { i }$ <sub>an</sub>d th<sub>ere</sub>f<sub>ore measures cons</sub>i<sub>s</sub>t<sub>ency an</sub>d ti<sub>g</sub>ht<sub>ness ra</sub>th<sub>er</sub> th<sub>an ou</sub>t<sub>-o</sub>f<sub>-</sub> <sub>samp</sub>l<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>ve</sub> <sub>accuracy.</sub>

## D.2. Experimental Details for Figure 3

This subsection <sub>g</sub>ives the ex<sub>p</sub>erimental <sub>p</sub>rotocols for §3.1 and Fi<sub>g</sub>ure 3.

## D.2.1. Empirical Choice of the Frequency Split

F<sub>or</sub> <sub>prac</sub>ti<sub>ca</sub>l f<sub>requency</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on,</sub> <sub>we</sub> <sub>use</sub> <sub>an</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>va</sub>l<sub>ue</sub> $C _ { \mathrm { F } } ^ { \mathrm { e m p } }$ f<sub>or</sub> th<sub>e</sub> <sub>cons</sub>t<sub>an</sub>t i<sub>n</sub> th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>sp</sub>lit <sub>cr</sub>it<sub>er</sub>i<sub>on:</sub>

$$
H _ { \mathrm { o p } } ( M ) = \left\{ n : M \omega _ { n } \geq \frac { 2 C _ { \mathrm { F } } ^ { \mathrm { e m p } } } { ( 1 - \rho ) \varepsilon _ { \mathrm { r e f } } } \right\} , \qquad C _ { \mathrm { F } } ^ { \mathrm { e m p } } = 0 . 0 6 , \quad \varepsilon _ { \mathrm { r e f } } = 0 . 1 .\tag{167}
$$

Here $\omega _ { n }$ <sub>are</sub> th<sub>e ac</sub>t<sub>ua</sub>l <sub>mo</sub>d<sub>e</sub>l f<sub>requenc</sub>i<sub>es,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> Ll<sub>ama-</sub>3<sub>.</sub>1’<sub>s na</sub>ti<sub>ve</sub> f<sub>requency sca</sub>li<sub>ng, an</sub>d $\rho = B ^ { - 1 / h }$ i<sub>s</sub> <sub>compu</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>e</sub> b<sub>ase gr</sub>id<sub>.</sub> W<sub>e c</sub>h<sub>oose</sub> $C _ { \mathrm { F } } ^ { \mathrm { e m p } } = 0 . { \overset { \mathrm { e m } } { 0 } } 6$ b<sub>ase</sub>d <sub>on emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub> f<sub>rom</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>an</sub>d <sub>w</sub>i<sub>n</sub>d<sub>ows,</sub> <sub>an</sub>d <sub>use</sub> thi<sub>s</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>across</sub> b<sub>o</sub>th <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>an</sub>d <sub>a</sub>ll th<sub>ree</sub> <sub>con</sub>t<sub>ex</sub>t l<sub>eng</sub>th<sub>s.</sub> Th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d b<sub>an</sub>d<sub>s ac</sub>hi<sub>eve ran</sub>d<sub>omness scores</sub> b<sub>e</sub>t<sub>ween</sub> 0<sub>.</sub>9936 <sub>an</sub>d 0<sub>.</sub>9997 <sub>un</sub>d<sub>er</sub> th<sub>e me</sub>t<sub>r</sub>i<sub>c</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>e</sub>l<sub>ow, s</sub>h<sub>ow</sub>i<sub>ng</sub> <sub>c</sub>l<sub>ose agreemen</sub>t b<sub>e</sub>t<sub>ween measure</sub>d <sub>an</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>no</sub>i<sub>se amp</sub>lit<sub>u</sub>d<sub>es across a</sub>ll <sub>s</sub>i<sub>x se</sub>tti<sub>ngs.</sub>

The <sub>p</sub>roof of Lemma C.1 uses $C _ { \mathrm { F } } = 2 \pi$ <sub>as a conserva</sub>ti<sub>ve su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>cons</sub>t<sub>an</sub>t f<sub>or a un</sub>if<sub>orm</sub> b<sub>oun</sub>d <sub>over ar</sub>bit<sub>rary</sub> fi<sub>xe</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>ors.</sub> Thi<sub>s c</sub>h<sub>o</sub>i<sub>ce es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>guaran</sub>t<sub>ee</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> ${ \mathrm { A . } } 3 ; C _ { \mathrm { F } } ^ { \mathrm { e m p } }$ <sub>a</sub>d<sub>ap</sub>t<sub>s</sub> th<sub>e</sub> <sub>sp</sub>lit <sub>cr</sub>it<sub>er</sub>i<sub>on</sub> t<sub>o</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub> Th<sub>e</sub> <sub>re</sub>f<sub>erence</sub> <sub>va</sub>l<sub>ue</sub> $\varepsilon _ { \mathrm { r e f } }$ <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> th<sub>e</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l th<sub>res</sub>h<sub>o</sub>ld<sub>,</sub> <sub>an</sub>d <sub>we assess</sub> th<sub>e resu</sub>lti<sub>ng sp</sub>lit th<sub>roug</sub>h <sub>measure</sub>d <sub>no</sub>i<sub>se-amp</sub>lit<sub>u</sub>d<sub>e agreemen</sub>t<sub>.</sub> Th<sub>e emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>ru</sub>l<sub>e a</sub>l<sub>so</sub> <sub>app</sub>li<sub>es</sub> t<sub>o</sub> Ll<sub>ama-</sub>3<sub>.</sub>1’<sub>s sca</sub>l<sub>e</sub>d f<sub>requency gr</sub>id<sub>, w</sub>h<sub>ose eva</sub>l<sub>ua</sub>ti<sub>on</sub> li<sub>es ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e geome</sub>t<sub>r</sub>i<sub>c-gr</sub>id <sub>assump</sub>ti<sub>ons</sub> <sub>o</sub>f th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>guaran</sub>t<sub>ee.</sub> Th<sub>e</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on exper</sub>i<sub>men</sub>t<sub>s re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub>i<sub>r or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>parame</sub>t<sub>er</sub> $c _ { \mathrm { i n t } } = 0 . 0 5$ , as documented in A<sub>pp</sub>endix G.2.

For Fi<sub>g</sub>ures 3b and 3c, we evaluate Qwen3-8B (Yan<sub>g</sub> et al., 2025a) and Llama-3.1-8B (Dube<sub>y</sub> et al., 2024) at $M \in \{ 8 1 9 2 , 3 2 7 6 8 , 1 3 1 0 7 2 \}$ <sub>.</sub> F<sub>o</sub>r <sub>eac</sub>h m<sub>o</sub>d<sub>e</sub>l<sub>,</sub> th<sub>e sa</sub>m<sub>e</sub> 40<sub>,</sub>960 fir<sub>s</sub>t-l<sub>aye</sub>r <sub>que</sub>r<sub>y</sub>–<sub>pa</sub>ir–h<sub>ea</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>e</sub>nt <sub>vec</sub>t<sub>ors are use</sub>d <sub>a</sub>t <sub>a</sub>ll th<sub>ree w</sub>i<sub>n</sub>d<sub>ow</sub> l<sub>en</sub> th<sub>s.</sub> W<sub>e</sub> h<sub>o</sub>ld th<sub>ese cac</sub>h<sub>e</sub>d <sub>vec</sub>t<sub>ors</sub> fi<sub>xe</sub>d <sub>an</sub>d <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r</sub> R<sub>o</sub>PE score mar<sub>g</sub><sup>i</sup>ns over � $\in \{ 0 , \ldots , M - 1 \}$ , using the paper’s convention � = query position − key position <sub>an</sub>d <sub>marg</sub>i<sub>n coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> $d _ { n } = Q _ { n } \overline { { ( K _ { a , n } - K _ { b , n } ) } } / \sqrt { d _ { \mathrm { h e a d } } }$ f<sub>or</sub> k<sub>eys � an</sub>d �<sub>.</sub> Th<sub>us</sub> th<sub>e</sub> 128k <sub>curves ex</sub>t<sub>en</sub>d th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve-</sub>di<sub>s</sub>t<sub>ance</sub> <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> <sub>o</sub>f fi<sub>xe</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>ons;</sub> th<sub>ey</sub> <sub>requ</sub>i<sub>re</sub> <sub>no</sub> 128k <sub>mo</sub>d<sub>e</sub>l f<sub>orwar</sub>d <sub>pass.</sub> F<sub>or</sub> <sub>every</sub> <sub>pre</sub>fi<sub>x</sub> $H _ { k } = \{ 0 , \ldots , k - 1 \}$ , <sup>w</sup>e co<sup>m</sup>pu<sup>t</sup>e <sup>th</sup>e agg<sup>r</sup>ega<sup>t</sup>e <sup>v</sup>a<sup>ri</sup>a<sup>n</sup>ce <sup>r</sup>a<sup>ti</sup>o

$$
\mathcal { R } _ { \mathrm { v a r } } ( H _ { k } ; M ) : = \frac { \sum _ { i } \mathsf { V a r } _ { M } ( X _ { H _ { k } } ^ { d _ { i } } ) } { \sum _ { i } U _ { H _ { k } } ( d _ { i } ) ^ { 2 } / 2 } ,\tag{168}
$$

<sub>w</sub>h<sub>ere</sub> � i<sub>n</sub>d<sub>exes</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub>d <sub>query–pa</sub>i<sub>r–</sub>h<sub>ea</sub>d <sub>po</sub>i<sub>n</sub>t<sub>s</sub> <sub>an</sub>d $\mathrm { V a r } _ { M }$ <sub>uses</sub> th<sub>e un</sub>if<sub>orm</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on over</sub> th<sub>e</sub> � i<sub>n</sub>t<sub>eger</sub> di<sub>s</sub>t<sub>ances.</sub> W<sub>e</sub> th<sub>en</sub> <sub>compare</sub> <sub>no</sub>i<sub>se</sub> <sub>amp</sub>lit<sub>u</sub>d<sub>es</sub> th<sub>roug</sub>h

$$
a ( H _ { k } ; M ) = \sqrt { \mathcal { R } _ { \mathrm { v a r } } ( H _ { k } ; M ) } , \qquad R ( H _ { k } ; M ) = \operatorname* { m a x } \{ 0 , 1 - | a ( H _ { k } ; M ) - 1 | \} .\tag{169}
$$

Th<sub>e amp</sub>lit<sub>u</sub>d<sub>e ra</sub>ti<sub>o �</sub> i<sub>s</sub> f<sub>orme</sub>d <sub>a</sub>ft<sub>er aggrega</sub>ti<sub>ng</sub> th<sub>e measure</sub>d <sub>an</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>var</sub>i<sub>ances.</sub> Th<sub>e p</sub>l<sub>o</sub>tt<sub>e</sub>d <sup>r</sup>a<sup>nd</sup>o<sup>mn</sup>ess p<sup>r</sup>o<sup>x</sup>y $R \in [ 0 , 1 ]$ <sub>equa</sub>l<sub>s one a</sub>t <sub>exac</sub>t <sub>amp</sub>lit<sub>u</sub>d<sub>e agreemen</sub>t <sub>an</sub>d d<sub>ecreases w</sub>ith th<sub>e a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve amp</sub>lit<sub>u</sub>d<sub>e error.</sub> F<sub>or examp</sub>l<sub>e, ra</sub>ti<sub>os</sub> $a = 0 . 9$ <sub>an</sub>d $a = 1 . 1$ b<sub>o</sub>th <sub>g</sub>i<sub>ve</sub> $R = 0 . 9$ . Positive cross-fre<sub>q</sub>uenc<sub>y</sub> <sub>covar</sub>i<sub>ance can g</sub>i<sub>ve</sub> $\mathcal { R } _ { \mathrm { v a r } } > 1 ;$ th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on preserves</sub> thi<sub>s</sub> di<sub>screpancy as a</sub> d<sub>ecrease</sub> i<sub>n</sub> �<sub>.</sub> W<sub>e re</sub>t<sub>a</sub>i<sub>n a</sub>ll <sub>resu</sub>lti<sub>ng</sub> di<sub>ps an</sub>d <sub>recover</sub>i<sub>es.</sub> Thi<sub>s me</sub>t<sub>r</sub>i<sub>c measures no</sub>i<sub>se-sca</sub>l<sub>e agreemen</sub>t <sub>an</sub>d <sub>supp</sub>li<sub>es no</sub> t<sub>es</sub>t <sub>o</sub>f G<sub>auss</sub>i<sub>an</sub> <sub>s</sub>h<sub>ape or</sub> t<sub>empora</sub>l i<sub>n</sub>d<sub>epen</sub>d<sub>ence.</sub>

In Fi<sub>g</sub>ures 3b and 3c<sub>,</sub> the <sub>g</sub>ra<sub>y</sub> curves <sub>p</sub>lot $R ( H _ { k } ; M )$ <sub>as</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> � <sub>o</sub>f <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>componen</sub>t<sub>s</sub> i<sub>ncreases.</sub> Oran<sub>g</sub>e circles mark the lar<sub>g</sub>est <sub>p</sub>refixes satisf<sub>y</sub>in<sub>g</sub> E<sub>q</sub>. (167) for 8k, 32k, and 128k windows, from left to <sub>r</sub>i<sub>g</sub>ht<sub>.</sub> Th<sub>e ma</sub>i<sub>n axes magn</sub>if<sub>y</sub> th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>cu</sub>t<sub>o</sub>f<sub>s an</sub>d <sub>near</sub>b<sub>y c</sub>h<sub>anges;</sub> i<sub>nse</sub>t<sub>s s</sub>h<sub>ow a</sub>ll 64 <sub>pre</sub>fi<sub>xes, w</sub>ith <sub>orange</sub> b<sub>oxes</sub> i<sub>n</sub>di<sub>ca</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>en</sub>l<sub>arge</sub>d <sub>reg</sub>i<sub>ons.</sub>

## D.2.2. Score-Distribution Illustration

Figure 3a reuses the Qwen3-8B layer-0 projection bank. For each of its 10 query tokens and 32 query heads, we uniforml<sub>y</sub> sam<sub>p</sub>le four distinct ke<sub>y</sub> tokens from the existin<sub>g</sub> 200-token ke<sub>y</sub> bank (seed 20260923), <sub>g</sub>ivin<sub>g</sub> 1<sub>,</sub>280 <sub>query–</sub>k<sub>ey–</sub>h<sub>ea</sub>d <sub>com</sub>bi<sub>na</sub>ti<sub>ons.</sub> Th<sub>ese are</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on scores;</sub> Fi<sub>gures</sub> 3b <sub>an</sub>d 3<sub>c use score</sub> <sub>marg</sub>i<sub>ns.</sub> F<sub>or</sub> <sub>eac</sub>h fi<sub>xe</sub>d <sub>com</sub>bi<sub>na</sub>ti<sub>on</sub> �<sub>,</sub> <sub>we</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> $S _ { H _ { 4 0 } } ^ { ( i ) } ( m )$ a<sup>t</sup> e<sup>v</sup>e<sup>r</sup>y $m \in \{ 0 , \ldots , 3 2 7 6 7 \}$ <sub>an</sub>d di<sub>v</sub>id<sub>e</sub> b<sub>y</sub> $\sigma _ { i } = U _ { H _ { 4 0 } } ( z ^ { ( i ) } ) / \sqrt { 2 }$ <sub>, w</sub>h<sub>ere</sub> $H _ { k } = \{ 0 , \ldots , k - 1 \}$ <sub>.</sub> Th<sub>e</sub> <sub>gray</sub> b<sub>ars</sub> <sub>average</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> d<sub>ens</sub>iti<sub>es</sub> <sub>w</sub>ith <sub>equa</sub>l <sub>we</sub>i<sub>g</sub>ht<sub>.</sub> N<sub>o emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>cen</sub>t<sub>er</sub>i<sub>ng or var</sub>i<sub>ance</sub> fitti<sub>ng</sub> i<sub>s use</sub>d<sub>.</sub> Th<sub>e orange curve</sub> i<sub>s</sub> $N ( 0 , 1 )$ <sub>un</sub>d<sub>er</sub> thi<sub>s norma</sub>li<sub>za</sub>ti<sub>on.</sub> Thi<sub>s</sub> illustration uses the 32k cutof from E<sub>q</sub>. (167) and the same cached <sub>p</sub>rojections, with no new model forward pass.

## D.3. Additional Evidence on High-Frequency Scores

## D.3.1. The Concentrated Qwen Frequency Behind the Noise-Amplitude Drop

Th<sub>e s</sub>h<sub>arp</sub> d<sub>rop</sub> i<sub>n</sub> th<sub>e</sub> i<sub>nse</sub>t <sub>o</sub>f Fi<sub>gure</sub> 3b <sub>occurs w</sub>h<sub>en</sub> th<sub>e re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>pre</sub>fi<sub>x grows</sub> f<sub>rom</sub> 51 t<sub>o</sub> 52 f<sub>requenc</sub>i<sub>es.</sub> Th<sub>e new</sub>l<sub>y</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d f<sub>requency</sub> h<sub>as zero-</sub>b<sub>ase</sub>d i<sub>n</sub>d<sub>ex</sub> $n = 5 1$ <sub>,</sub> <sub>so</sub> it i<sub>s</sub> th<sub>e</sub> 52<sub>n</sub>d <sub>comp</sub>l<sub>ex</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>pa</sub>i<sub>r,</sub> <sub>correspon</sub>di<sub>ng</sub> t<sub>o rea</sub>l <sub>c</sub>h<sub>anne</sub>l<sub>s</sub> 51 <sub>an</sub>d 115 i<sub>n</sub> th<sub>e sp</sub>lit<sub>-</sub>h<sub>a</sub>lf l<sub>ayou</sub>t<sub>.</sub> O<sub>n</sub> th<sub>e same</sub> 40<sub>,</sub>960 <sub>marg</sub>i<sub>n-coe</sub>fi<sub>c</sub>i<sub>en</sub>t vectors $d _ { i }$ <sub>use</sub>d i<sub>n</sub> th<sub>a</sub>t <sub>pane</sub>l<sub>,</sub> it<sub>s aggrega</sub>t<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>-energy s</sub>h<sub>are</sub> i<sub>s</sub>

$$
\frac { \sum _ { i } | d _ { i , 5 1 } | ^ { 2 } } { \sum _ { i } \sum _ { n } | d _ { i , n } | ^ { 2 } } = 9 2 . 3 5 0 7 \% .\tag{170}
$$

Thi<sub>s</sub> i<sub>s</sub> th<sub>e</sub> f<sub>rac</sub>ti<sub>on o</sub>f th<sub>e summe</sub>d <sub>square</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>norms, w</sub>ith <sub>eac</sub>h <sub>samp</sub>l<sub>e en</sub>t<sub>er</sub>i<sub>ng</sub> th<sub>e sum once.</sub> It i<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>t f<sub>rom</sub> <sub>a</sub> f<sub>rac</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>measure</sub>d <sub>score</sub> <sub>var</sub>i<sub>ance</sub> <sub>or</sub> <sub>a</sub> <sub>norm</sub> <sub>s</sub>h<sub>are</sub> b<sub>e</sub>f<sub>ore</sub> <sub>squar</sub>i<sub>ng.</sub>

At $M = 3 2 7 6 8 _ { \div }$ <sub>,</sub> thi<sub>s</sub> f<sub>requency ro</sub>t<sub>a</sub>t<sub>es</sub> b<sub>y on</sub>l<sub>y</sub> $M \omega _ { 5 1 } = 0 . 5 4 2 2 5$ <sub>ra</sub>di<sub>ans over</sub> th<sub>e w</sub>i<sub>n</sub>d<sub>ow.</sub> It<sub>s own</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow</sub> <sub>var</sub>i<sub>ance</sub> i<sub>s</sub> 4<sub>.</sub>4599% <sub>o</sub>f th<sub>e</sub> h<sub>a</sub>lf<sub>-energy pre</sub>di<sub>c</sub>ti<sub>on.</sub> I<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> it <sub>mu</sub>lti<sub>p</sub>li<sub>es</sub> th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>var</sub>i<sub>ance</sub> b<sub>y</sub> 16.1731<sub>,</sub> while the a<sub>gg</sub>re<sub>g</sub>ate variance ratio ${ \mathcal { R } } _ { \operatorname { v a r } }$ falls from 0.883744 to 0.086922 and the <sub>p</sub>lotted randomness <sub>p</sub>rox<sub>y</sub> � falls from 0.940077 to 0.294826. The slowl<sub>y</sub> var<sub>y</sub>in<sub>g</sub> com<sub>p</sub>onent therefore adds substantial coeficient <sub>energy w</sub>ith<sub>ou</sub>t th<sub>e var</sub>i<sub>ance expec</sub>t<sub>e</sub>d f<sub>rom a</sub> f<sub>as</sub>t <sub>osc</sub>ill<sub>a</sub>ti<sub>on.</sub> Thi<sub>s accoun</sub>t<sub>s</sub> f<sub>or</sub> th<sub>e a</sub>b<sub>rup</sub>t d<sub>rop an</sub>d ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> th<sub>e nee</sub>d t<sub>o</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>magn</sub>it<sub>u</sub>d<sub>e</sub> f<sub>rom</sub> f<sub>requency-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>var</sub>i<sub>a</sub>ti<sub>on.</sub> Th<sub>e componen</sub>t li<sub>es</sub> <sub>ou</sub>t<sub>s</sub>id<sub>e a</sub>ll th<sub>ree opera</sub>ti<sub>ona</sub>l hi<sub>g</sub>h<sub>-</sub>f<sub>requency pre</sub>fi<sub>xes.</sub>

## D.3.2. Representation Assumptions and Gaussian Score Shape

Th<sub>e momen</sub>t <sub>guaran</sub>t<sub>ees</sub> i<sub>n</sub> P<sub>ropos</sub>iti<sub>on</sub> 1 <sub>a</sub>ll<sub>ow ar</sub>bit<sub>rary</sub> fi<sub>xe</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>magn</sub>it<sub>u</sub>d<sub>es an</sub>d <sub>p</sub>h<sub>ases.</sub> R<sub>e</sub>l<sub>evan</sub>t <sub>p</sub>rior anal<sub>y</sub>ses use several diferent assum<sub>p</sub>tions. Xu et al. (2024) derive their ex<sub>p</sub>ected <sub>p</sub>reference bound from i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t id<sub>en</sub>ti<sub>ca</sub>ll<sub>y</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d <sub>query an</sub>d k<sub>ey coor</sub>di<sub>na</sub>t<sub>es w</sub>ith <sub>a common var</sub>i<sub>ance an</sub>d <sub>a s</sub>i<sub>m</sub>il<sub>ar-</sub>k<sub>ey</sub> noise model. Their theorem does not re<sub>q</sub>uire Gaussian coordinate distributions. Du et al. (2026) also fix th<sub>e query an</sub>d k<sub>ey vec</sub>t<sub>ors an</sub>d <sub>samp</sub>l<sub>e re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance.</sub> Th<sub>e</sub>i<sub>r</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency ana</sub>l<sub>ys</sub>i<sub>s g</sub>i<sub>ves</sub> th<sub>e near-zero</sub> <sub>mean</sub> <sub>an</sub>d h<sub>a</sub>lf<sub>-energy</sub> <sub>var</sub>i<sub>ance</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on;</sub> it<sub>s</sub> <sub>norma</sub>l <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>uses</sub> <sub>regu</sub>l<sub>ar</sub> <sub>amp</sub>lit<sub>u</sub>d<sub>es</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>a</sub> d<sub>om</sub>i<sub>nan</sub>t <sub>componen</sub>t <sub>an</sub>d <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence across</sub> f<sub>requenc</sub>i<sub>es.</sub> O<sub>ur</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow momen</sub>t b<sub>oun</sub>d<sub>s</sub> <sub>con</sub>t<sub>ro</sub>l <sub>cross-</sub>f<sub>requency</sub> t<sub>erms</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> <sub>an</sub>d <sub>a</sub>ll<sub>ow</sub> <sub>concen</sub>t<sub>ra</sub>t<sub>e</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>energy.</sub>

Deterministic anal<sub>y</sub>ses also a<sub>pp</sub>ear in Su et al. (2024), Barbero et al. (2025), and Urrutia et al. (2026). In <sub>p</sub>articular, the Gaussian <sub>q</sub>uer<sub>y</sub>/ke<sub>y</sub> model in Pro<sub>p</sub>osition 3.2 of Barbero et al. (2025) is a counterexam<sub>p</sub>le to <sub>un</sub>i<sub>versa</sub>l di<sub>s</sub>t<sub>ance</sub> d<sub>ecay;</sub> th<sub>e</sub>i<sub>r o</sub>th<sub>er cons</sub>t<sub>ruc</sub>ti<sub>ons an</sub>d <sub>seman</sub>ti<sub>c-a</sub>tt<sub>en</sub>ti<sub>on resu</sub>lt<sub>s use</sub> fi<sub>xe</sub>d <sub>vec</sub>t<sub>ors.</sub> Th<sub>ese</sub> <sub>resu</sub>lt<sub>s es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>a</sub>t fi<sub>xe</sub>d<sub>-vec</sub>t<sub>or ana</sub>l<sub>ys</sub>i<sub>s</sub> i<sub>s a</sub>l<sub>rea</sub>d<sub>y par</sub>t <sub>o</sub>f th<sub>e</sub> lit<sub>era</sub>t<sub>ure.</sub> Th<sub>e presen</sub>t <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>s</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow momen</sub>t <sub>con</sub>t<sub>ro</sub>l <sub>an</sub>d th<sub>e resu</sub>lti<sub>ng</sub> f<sub>requency cr</sub>it<sub>er</sub>i<sub>on.</sub> G<sub>auss</sub>i<sub>an score s</sub>h<sub>ape rema</sub>i<sub>ns a separa</sub>t<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on:</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>poo</sub>l<sub>e</sub>d d<sub>ens</sub>it<sub>y suppor</sub>t<sub>s</sub> it <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y, an</sub>d th<sub>e</sub> fi<sub>gure</sub> d<sub>oes no</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>a</sub>t <sub>every</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>query–</sub>k<sub>ey com</sub>bi<sub>na</sub>ti<sub>on</sub> h<sub>as a</sub> G<sub>auss</sub>i<sub>an w</sub>i<sub>n</sub>d<sub>ow</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub>

## E. The 49 Reference Task Settings

Th<sub>e</sub> di<sub>ag</sub>n<sub>os</sub>ti<sub>c</sub> r<sub>e</sub>f<sub>e</sub>r<sub>e</sub>n<sub>ce se</sub>t <sub>co</sub>nt<sub>a</sub>in<sub>s</sub> 49 t<sub>as</sub>k/<sub>co</sub>nfi<sub>gu</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>se</sub>ttin<sub>gs</sub> fr<sub>o</sub>m nin<sub>e</sub> b<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rk <sub>sou</sub>r<sub>ces.</sub> A <sub>se</sub>ttin<sub>g</sub> <sub>com</sub>bi<sub>nes a</sub> t<sub>as</sub>k <sub>w</sub>ith <sub>a</sub> difi<sub>cu</sub>lt<sub>y or</sub> i<sub>npu</sub>t<sub>-</sub>l<sub>eng</sub>th <sub>con</sub>fi<sub>gura</sub>ti<sub>on.</sub> T<sub>a</sub>bl<sub>e</sub> 1 li<sub>s</sub>t<sub>s every se</sub>tti<sub>ng</sub> b<sub>y group</sub>i<sub>ng</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>ons o</sub>f th<sub>e same</sub> t<sub>as</sub>k<sub>.</sub> Th<sub>e se</sub>t <sub>covers</sub> k<sub>ey an</sub>d d<sub>ocumen</sub>t <sub>re</sub>t<sub>r</sub>i<sub>eva</sub>l<sub>, mu</sub>lti<sub>-</sub>h<sub>op ques</sub>ti<sub>on answer</sub>i<sub>ng,</sub> text and list ordering, state tracking, and long-context reasoning. For example, the LIFBench adjacent element retrieval task contributes three len<sub>g</sub>th settin<sub>g</sub>s (3k, 6k, and 13k), while RULER2 multi-ke<sub>y</sub> retrieval contributes four dificult<sub>y</sub> settin<sub>g</sub>s (basic, eas<sub>y</sub>, medium, and hard). These 49 settin<sub>g</sub>s form the rankin<sub>g</sub> <sub>re</sub>f<sub>erence</sub> <sub>se</sub>t i<sub>n</sub> Fi<sub>gure</sub> 4<sub>;</sub> <sub>eac</sub>h <sub>po</sub>i<sub>n</sub>t i<sub>n</sub> Fi<sub>gure</sub> 6 <sub>an</sub>d <sub>eac</sub>h b<sub>ar</sub> i<sub>n</sub> Fi<sub>gure</sub> 5 <sub>correspon</sub>d<sub>s</sub> t<sub>o</sub> <sub>one</sub> <sub>se</sub>tti<sub>ng.</sub>

Qwen3-8B and Llama-3.1-8B-Instruct use the same example IDs within each setting: 2,211 examples per <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d 4<sub>,</sub>422 <sub>examp</sub>l<sub>e–mo</sub>d<sub>e</sub>l <sub>recor</sub>d<sub>s</sub> i<sub>n</sub> t<sub>o</sub>t<sub>a</sub>l<sub>.</sub> W<sub>e</sub> <sub>recor</sub>d <sub>query</sub> <sub>an</sub>d k<sub>ey</sub> <sub>ac</sub>ti<sub>va</sub>ti<sub>ons</sub> d<sub>ur</sub>i<sub>ng</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>ese</sub> <sub>examp</sub>l<sub>es.</sub> Th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> di<sub>agnos</sub>ti<sub>cs</sub> <sub>supp</sub>l<sub>emen</sub>t <sub>eac</sub>h t<sub>as</sub>k’<sub>s</sub> b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l <sub>accuracy</sub> <sub>w</sub>ith i<sub>n</sub>f<sub>orma</sub>ti<sub>on a</sub>b<sub>ou</sub>t <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> f<sub>a</sub>il<sub>ure suscep</sub>tibilit<sub>y.</sub>

Sources and configurations. The 12 MK, MV, and QA settings use the oficial NeMo Skills RULER2 <sub>su</sub>it<sub>e.</sub><sup>2</sup> W<sub>e use</sub> it<sub>s nom</sub>i<sub>na</sub>l 8k <sub>con</sub>fi<sub>gura</sub>ti<sub>on.</sub> E<sub>ac</sub>h f<sub>am</sub>il<sub>y</sub> h<sub>as</sub> b<sub>as</sub>i<sub>c, easy, me</sub>di<sub>um, an</sub>d h<sub>ar</sub>d <sub>var</sub>i<sub>an</sub>t<sub>s, w</sub>ith 100 <sub>examp</sub>l<sub>es per con</sub>fi<sub>gura</sub>ti<sub>on.</sub> Th<sub>e</sub> t<sub>wo</sub> L<sub>ong</sub>B<sub>enc</sub>h t<sub>as</sub>k<sub>s use unmo</sub>difi<sub>e</sub>d <sub>examp</sub>l<sub>es w</sub>h<sub>ose na</sub>ti<sub>ve c</sub>h<sub>a</sub>t i<sub>npu</sub>t<sub>s</sub> fit <sub>w</sub>ithi<sub>n</sub> 16<sub>,</sub>384 t<sub>o</sub>k<sub>ens</sub> f<sub>or</sub> b<sub>o</sub>th <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> L<sub>-</sub>E<sub>va</sub>l T<sub>op</sub>i<sub>c</sub>R<sub>e</sub>t <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> 150 <sub>ques</sub>ti<sub>on recor</sub>d<sub>s</sub> f<sub>rom</sub> 50 conversations, with three topic-retrieval questions per conversation. FLenQA uses the books-padding, <sub>ran</sub>d<sub>om-ev</sub>id<sub>ence con</sub>fi<sub>gura</sub>ti<sub>on a</sub>t <sub>con</sub>t<sub>ex</sub>t <sub>s</sub>i<sub>zes</sub> 500 <sub>an</sub>d 3<sub>,</sub>000<sub>.</sub> N<sub>o</sub>LiM<sub>a uses</sub> b<sub>oo</sub>k 1 <sub>w</sub>ith th<sub>e nee</sub>dl<sub>e a</sub>t 52% de<sub>p</sub>th. The three LIFBench settin<sub>g</sub>s contain 12 exam<sub>p</sub>les each<sub>;</sub> the remainin<sub>g</sub> 33 settin<sub>g</sub>s contain 25 <sub>examp</sub>l<sub>es eac</sub>h<sub>.</sub>

L<sub>eng</sub>th l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> i<sub>n</sub> th<sub>e</sub> t<sub>a</sub>bl<sub>e re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e source</sub> b<sub>enc</sub>h<sub>mar</sub>k’<sub>s con</sub>fi<sub>gura</sub>ti<sub>on names.</sub> F<sub>or</sub> th<sub>e</sub> di<sub>agnos</sub>ti<sub>c ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on,</sub> � i<sub>s</sub> th<sub>e comp</sub>l<sub>e</sub>t<sub>e</sub> i<sub>npu</sub>t l<sub>eng</sub>th <sub>un</sub>d<sub>er</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> t<sub>o</sub>k<sub>en</sub>i<sub>zer,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> it<sub>s promp</sub>t <sub>an</sub>d <sub>c</sub>h<sub>a</sub>t t<sub>emp</sub>l<sub>a</sub>t<sub>e.</sub> Th<sub>e same</sub> l<sub>eng</sub>th l<sub>a</sub>b<sub>e</sub>l <sub>can</sub> th<sub>ere</sub>f<sub>ore y</sub>i<sub>e</sub>ld dif<sub>eren</sub>t � <sub>across mo</sub>d<sub>e</sub>l<sub>s.</sub> Th<sub>e coun</sub>t<sub>s</sub> b<sub>e</sub>l<sub>ow</sub> d<sub>escr</sub>ib<sub>e</sub> th<sub>e</sub> di<sub>agnos</sub>ti<sub>c</sub> <sub>re</sub>f<sub>erence se</sub>t<sub>; ma</sub>t<sub>c</sub>h<sub>e</sub>d <sub>coverage</sub> f<sub>or</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on exper</sub>i<sub>men</sub>t<sub>s</sub> i<sub>s repor</sub>t<sub>e</sub>d <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> H<sub>.</sub>

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> Th<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> 49<sub>-se</sub>tti<sub>ng</sub> di<sub>agnos</sub>ti<sub>c</sub> <sub>re</sub>f<sub>erence</sub> <sub>se</sub>t<sub>.</sub> E<sub>ac</sub>h <sub>con</sub>fi<sub>gura</sub>ti<sub>on</sub> i<sub>n</sub> <sub>a</sub> <sub>row</sub> i<sub>s</sub> <sub>a</sub> <sub>separa</sub>t<sub>e</sub> <sub>se</sub>tti<sub>ng.</sub> � i<sub>s</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>se</sub>tti<sub>ngs an</sub>d � i<sub>s</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>examp</sub>l<sub>es per se</sub>tti<sub>ng an</sub>d <sub>per mo</sub>d<sub>e</sub>l<sub>.</sub> L<sub>eng</sub>th<sub>s are nom</sub>i<sub>na</sub>l b<sub>enc</sub>h<sub>mar</sub>k <sub>con</sub>fi<sub>gura</sub>ti<sub>ons;</sub> th<sub>e</sub> di<sub>agnos</sub>ti<sub>c uses eac</sub>h <sub>mo</sub>d<sub>e</sub>l’<sub>s ac</sub>t<sub>ua</sub>l i<sub>npu</sub>t l<sub>eng</sub>th<sub>.</sub>
<table><tr><td>Task</td><td>What the model must do</td><td>Configurations</td><td>K N</td></tr><tr><td colspan="4">RULER2 (NeMo Skills)</td></tr><tr><td>Multi-key (MK)</td><td>Retrieve by key; harder variants answer Basic, easy, medium, retrieved MMLU questions.</td><td>hard</td><td>4 100</td></tr><tr><td></td><td>Multi-value (MV) Retrieve multiple values or select an ordered question for a key.</td><td>Basic, easy, medium, hard</td><td>4 100</td></tr><tr><td>Question answering</td><td>Retrieve supporting documents and answer HotpotQA questions.</td><td>Basic, easy, medium, 4 100 hard</td><td></td></tr><tr><td colspan="4">LongBench-v1 (Bai et al., 2024)</td></tr><tr><td>Qasper</td><td>Answer questions about an academic paper.</td><td>Native chat ≤ 16k</td><td>1 25</td></tr><tr><td>MuSiQue</td><td>Combine evidence across documents Native chat ≤ 16k for multi-hop answers.</td><td></td><td>1 25</td></tr><tr><td>L-Eval (An et al., 2024)</td><td></td><td></td><td></td></tr><tr><td>TopicRet</td><td>Retrieve the first, second, or third conversation topic.</td><td>Native conversation</td><td>1 150</td></tr><tr><td>Ada-LEval (Wang et al., 2024a) TSort</td><td>Restore the original order of shuffled</td><td>1k</td><td>1 25</td></tr><tr><td></td><td>text fragments.</td><td></td><td></td></tr><tr><td>FLenQA (Levy et al., 2024) PIR</td><td>Infer a room property through a</td><td>500; 3,000</td><td>2 25</td></tr><tr><td>MonoRel</td><td>person-room relation. Infer a transitive conclusion from</td><td>500; 3,000</td><td>25</td></tr><tr><td>Simplified</td><td>ordered relations. Deduce a true/false conclusion from</td><td></td><td>25</td></tr><tr><td>RuleTaker</td><td>rules and facts.</td><td>500; 3,000</td><td>2</td></tr><tr><td>LongReason (Ling et al., 2025) Mixed reasoning</td><td>Solve reading, logical, and</td><td>8k; 16k</td><td>2 25</td></tr><tr><td></td><td>mathematical reasoning questions.</td><td></td><td></td></tr><tr><td>BABILong (Kuratov et al., 2024) qa1: person</td><td>Track a person's current location.</td><td>8k; 16k</td><td>2 25</td></tr><tr><td>location qa2: object</td><td>Track an object's current location.</td><td></td><td>2</td></tr><tr><td>location qa3: previous</td><td>Recover an object's location before a</td><td>8k; 16k</td><td>25</td></tr><tr><td>location qa4: directional</td><td>specified event. Infer a directional relation between</td><td>8k; 16k</td><td>2 25</td></tr><tr><td>relations</td><td>two entities.</td><td>8k; 16k</td><td>2 25</td></tr><tr><td></td><td>qa5: giving events Identify the giver, recipient, or object in 8k; 16k a giving event. Count the objects a person currently</td><td></td><td>2 25</td></tr><tr><td>qa7: counting</td><td>holds.</td><td>8k; 16k</td><td>2 25</td></tr><tr><td>qa8: lists and sets</td><td>List the objects a person currently holds.</td><td>8k; 16k</td><td>2 25</td></tr><tr><td>qa9: negation</td><td>Answer location questions with explicit 8k; 16k negation.</td><td></td><td>2 25</td></tr><tr><td>qa10: indefinite knowledge</td><td>Distinguish known, false, and uncertain locations.</td><td>8k; 16k</td><td>2 25</td></tr><tr><td>LIFBench (Wu et al., 2025b) List offset query</td><td>Return the item immediately before or Official base 3k; 6k;</td><td></td><td>3 12</td></tr><tr><td></td><td>after a target.</td><td>13k</td><td></td></tr><tr><td>NoLiMa (Modarressi et al., 2025) One-hop retrieval Retrieve a fact through one implicit</td><td></td><td>8k; 16k</td><td>2 25</td></tr><tr><td></td><td>semantic association. Two-hop retrieval Retrieve a fact through two implicit</td><td>8k; 16k</td><td>2 25</td></tr><tr><td></td><td>semantic associations.</td><td></td><td></td></tr></table>

## F. Token and Head Selection

## F.1. Token Sampling from the Evaluated Inputs

Th<sub>e curren</sub>t di<sub>agnos</sub>ti<sub>c reuses</sub> th<sub>e recor</sub>d<sub>e</sub>d <sub>query an</sub>d k<sub>ey ac</sub>ti<sub>va</sub>ti<sub>ons</sub> f<sub>or</sub> th<sub>e</sub> 49 <sub>re</sub>f<sub>erence se</sub>tti<sub>ngs</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> E<sub>.</sub> E<sub>ac</sub>h <sub>examp</sub>l<sub>e</sub> i<sub>s</sub> t<sub>o</sub>k<sub>en</sub>i<sub>ze</sub>d <sub>w</sub>ith th<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l’<sub>s na</sub>ti<sub>ve c</sub>h<sub>a</sub>t t<sub>emp</sub>l<sub>a</sub>t<sub>e.</sub> Ch<sub>arac</sub>t<sub>er spans</sub> <sub>supp</sub>li<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> t<sub>as</sub>k <sub>prepara</sub>ti<sub>on</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e ques</sub>ti<sub>on an</sub>d it<sub>s con</sub>t<sub>ex</sub>t<sub>.</sub> Eli<sub>g</sub>ibl<sub>e</sub> t<sub>o</sub>k<sub>en occurrences over</sub>l<sub>ap</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng span, con</sub>t<sub>a</sub>i<sub>n a</sub>t l<sub>eas</sub>t <sub>one</sub> l<sub>e</sub>tt<sub>er or</sub> di<sub>g</sub>it <sub>w</sub>ithi<sub>n</sub> th<sub>a</sub>t <sub>over</sub>l<sub>ap, an</sub>d h<sub>ave nonemp</sub>t<sub>y c</sub>h<sub>arac</sub>t<sub>er</sub> <sub>o</sub>f<sub>se</sub>t<sub>s; spec</sub>i<sub>a</sub>l t<sub>o</sub>k<sub>ens are exc</sub>l<sub>u</sub>d<sub>e</sub>d<sub>.</sub> W<sub>e se</sub>l<sub>ec</sub>t <sub>up</sub> t<sub>o</sub> f<sub>our query pos</sub>iti<sub>ons</sub> f<sub>rom</sub> th<sub>e ques</sub>ti<sub>on an</sub>d <sub>up</sub> t<sub>o</sub> 16 k<sub>ey</sub> <sub>pos</sub>iti<sub>ons</sub> f<sub>rom</sub> th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t<sub>.</sub> E<sub>very</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d k<sub>ey</sub> <sub>prece</sub>d<sub>es</sub> <sub>every</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>query</sub> i<sub>n</sub> th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l i<sub>npu</sub>t<sub>.</sub>

Selection uses a deterministic SHA-256 orderin<sub>g</sub> without re<sub>p</sub>lacement. For the ori<sub>g</sub>inal 12 RULER2 settin<sub>g</sub>s<sub>,</sub> th<sub>e or</sub>d<sub>er</sub>i<sub>ng</sub> i<sub>s</sub> k<sub>eye</sub>d b<sub>y</sub> th<sub>e pro</sub>t<sub>oco</sub>l id<sub>en</sub>tifi<sub>er, examp</sub>l<sub>e</sub> ID<sub>, samp</sub>li<sub>ng ro</sub>l<sub>e, an</sub>d t<sub>o</sub>k<sub>en pos</sub>iti<sub>on.</sub> Th<sub>e rema</sub>i<sub>n</sub>i<sub>ng</sub> 37 <sub>se</sub>tti<sub>ngs a</sub>l<sub>so</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>see</sub>d 20260910<sub>.</sub> S<sub>epara</sub>t<sub>e samp</sub>li<sub>ng ro</sub>l<sub>es se</sub>l<sub>ec</sub>t <sub>quer</sub>i<sub>es,</sub> k<sub>eys, an</sub>d th<sub>e</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng use</sub>d t<sub>o pa</sub>i<sub>r</sub> k<sub>eys.</sub> Th<sub>e save</sub>d t<sub>o</sub>k<sub>en pos</sub>iti<sub>ons rema</sub>i<sub>n</sub> fi<sub>xe</sub>d th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> di<sub>agnos</sub>ti<sub>c ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>ons.</sub>

For the Common-pair branch, each query is combined with every selected key, giving up to 64 query–key pairs per example. For the Common-triplet branch, the selected keys are reordered with the key-pairing role and grouped into eight disjoint pairs; each query is combined with these pairs, giving up to 32 triplets with di<sub>s</sub>tin<sub>c</sub>t k<sub>eys.</sub> All 2<sub>,</sub>211 <sub>e</sub>x<sub>a</sub>m<sub>p</sub>l<sub>es</sub> <sub>pe</sub>r m<sub>o</sub>d<sub>e</sub>l h<sub>a</sub>v<sub>e</sub> 16 <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d k<sub>eys.</sub> Of th<sub>ese,</sub> 2<sub>,</sub>161 h<sub>a</sub>v<sub>e</sub> f<sub>ou</sub>r <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>q</sub>ueries and <sub>y</sub>ield 64 <sub>p</sub>airs and 32 tri<sub>p</sub>lets. The other 50 exam<sub>p</sub>les<sub>,</sub> from the two BABILon<sub>g p</sub>erson-location <sub>se</sub>tti<sub>ngs,</sub> h<sub>ave</sub> th<sub>ree e</sub>li<sub>g</sub>ibl<sub>e quer</sub>i<sub>es an</sub>d <sub>y</sub>i<sub>e</sub>ld 48 <sub>pa</sub>i<sub>rs an</sub>d 24 t<sub>r</sub>i<sub>p</sub>l<sub>e</sub>t<sub>s.</sub> Th<sub>e</sub> C<sub>ommon</sub> b<sub>ranc</sub>h<sub>es use</sub> th<sub>ese</sub> <sub>samp</sub>l<sub>e</sub>d <sub>occurrences w</sub>ith<sub>ou</sub>t t<sub>as</sub>k<sub>-</sub>t<sub>arge</sub>t l<sub>a</sub>b<sub>e</sub>l<sub>s.</sub> Th<sub>e seman</sub>ti<sub>c ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on su</sub>b<sub>sequen</sub>tl<sub>y or</sub>i<sub>en</sub>t<sub>s eac</sub>h t<sub>r</sub>i<sub>p</sub>l<sub>e</sub>t b<sub>y</sub> it<sub>s</sub> <sub>score</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> <sub>a</sub>t <sub>zero</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance,</sub> <sub>as</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> G<sub>.</sub>1<sub>.</sub>1<sub>.</sub>

## F.2. Selecting a Fixed Set of Attention Heads

W<sub>e se</sub>l<sub>ec</sub>t h<sub>ea</sub>d<sub>s w</sub>h<sub>ose</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency norm s</sub>h<sub>are var</sub>i<sub>es across</sub> th<sub>e re</sub>f<sub>erence se</sub>tti<sub>ngs.</sub> F<sub>or eac</sub>h l<sub>ayer–</sub>h<sub>ea</sub>d <sub>un</sub>it ℎ<sub>,</sub> <sub>we</sub> fi<sub>rs</sub>t <sub>average</sub> th<sub>e</sub> C<sub>ommon-pa</sub>i<sub>r</sub> <sub>va</sub>l<sub>ue</sub> $r _ { H }$ <sub>over</sub> <sub>va</sub>lid <sub>pa</sub>i<sub>rs</sub> <sub>w</sub>ithi<sub>n</sub> <sub>eac</sub>h <sub>examp</sub>l<sub>e,</sub> th<sub>en</sub> <sub>average</sub> <sub>over</sub> <sub>examp</sub>l<sub>es w</sub>ithi<sub>n eac</sub>h <sub>se</sub>tti<sub>ng.</sub> L<sub>e</sub>t $\bar { r } _ { t , h }$ b<sub>e</sub> thi<sub>s</sub> m<sub>ea</sub>n f<sub>o</sub>r <sub>se</sub>ttin<sub>g</sub> �<sub>.</sub> With <sub>a</sub>ll 49 <sub>se</sub>ttin<sub>gs</sub> w<sub>e</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>equa</sub>ll<sub>y,</sub> th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>c</sub> i<sub>s</sub> th<sub>e samp</sub>l<sub>e var</sub>i<sub>ance</sub>

$$
\nu _ { h } = { \frac { 1 } { 4 8 } } \sum _ { t = 1 } ^ { 4 9 } \left( \bar { r } _ { t , h } - { \frac { 1 } { 4 9 } } \sum _ { s = 1 } ^ { 4 9 } \bar { r } _ { s , h } \right) ^ { 2 } .\tag{171}
$$

E<sub>ac</sub>h <sub>mo</sub>d<sub>e</sub>l i<sub>s ran</sub>k<sub>e</sub>d <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> b<sub>y</sub> d<sub>ecreas</sub>i<sub>ng</sub> $\nu _ { h }$ . Ties are resolved by increasing layer-major head index; neither model has a tie at the reported selection boundaries. A fraction � retains ⌈��⌉ of the model’s � h<sub>ea</sub>d<sub>s.</sub>

Qwen3-8B has 36 layers and 32 query heads per layer, for 1,152 heads in total; its top 5% and top 10% sets contain 58 and 116 heads. Llama-3.1-8B-Instruct has 32 la<sub>y</sub>ers and 32 <sub>q</sub>uer<sub>y</sub> heads <sub>p</sub>er la<sub>y</sub>er<sub>,</sub> for 1<sub>,</sub>024 heads<sub>;</sub> it<sub>s correspon</sub>di<sub>ng se</sub>t<sub>s con</sub>t<sub>a</sub>i<sub>n</sub> 52 <sub>an</sub>d 103 h<sub>ea</sub>d<sub>s.</sub> Fi<sub>gure</sub> 8 <sub>s</sub>h<sub>ows</sub> th<sub>e se</sub>l<sub>ec</sub>ti<sub>on across</sub> l<sub>ayers an</sub>d h<sub>ea</sub>d<sub>s.</sub> Th<sub>e</sub> t<sub>op-</sub>5% <sub>se</sub>t i<sub>s</sub> fi<sub>xe</sub>d <sub>across</sub> t<sub>as</sub>k<sub>s an</sub>d <sub>supp</sub>li<sub>es</sub> b<sub>o</sub>th th<sub>e</sub> S<sub>eman</sub>ti<sub>c an</sub>d P<sub>os</sub>iti<sub>ona</sub>l S<sub>cores</sub> i<sub>n</sub> th<sub>e ma</sub>i<sub>n</sub> di<sub>agnos</sub>ti<sub>c.</sub> Th<sub>e</sub> <sub>secon</sub>d i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> <sub>s</sub>t<sub>age</sub> <sub>expan</sub>d<sub>s</sub> thi<sub>s</sub> <sub>same</sub> <sub>ran</sub>ki<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> t<sub>op</sub> 10%<sub>,</sub> <sub>as</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> H<sub>.</sub> Thi<sub>s</sub> <sub>ran</sub>ki<sub>ng</sub> d<sub>escr</sub>ib<sub>es var</sub>i<sub>a</sub>ti<sub>on w</sub>ithi<sub>n</sub> th<sub>e c</sub>h<sub>osen re</sub>f<sub>erence se</sub>t<sub>,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> it<sub>s</sub> dif<sub>eren</sub>t i<sub>npu</sub>t<sub>-</sub>l<sub>eng</sub>th <sub>se</sub>tti<sub>ngs.</sub>

![](images/3a08412ca8ad86dfc76bd4253f53bf257fdab2a497ff437b3d53162abb1f2210.jpg)  
Fi<sub>gure</sub> 8<sub>:</sub> H<sub>ea</sub>d <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e</sub> 49 <sub>re</sub>f<sub>erence se</sub>tti<sub>ngs.</sub> E<sub>ac</sub>h <sub>ce</sub>ll <sub>s</sub>h<sub>ows</sub> th<sub>e samp</sub>l<sub>e var</sub>i<sub>ance o</sub>f <sub>a</sub> h<sub>ea</sub>d’<sub>s</sub> <sub>se</sub>tti<sub>ng-</sub>l<sub>eve</sub>l <sub>mean</sub> C<sub>ommon-pa</sub>i<sub>r</sub> $r _ { H } ,$ <sub>, us</sub>i<sub>ng a s</sub>h<sub>are</sub>d <sub>co</sub>l<sub>or sca</sub>l<sub>e across mo</sub>d<sub>e</sub>l<sub>s.</sub> Bl<sub>ac</sub>k <sub>ou</sub>tli<sub>nes</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e</sub> t<sub>op-</sub>5% h<sub>ea</sub>d<sub>s use</sub>d b<sub>y</sub> th<sub>e ma</sub>i<sub>n</sub> di<sub>agnos</sub>ti<sub>c an</sub>d fi<sub>rs</sub>t i<sub>n</sub>t<sub>erven</sub>ti<sub>on s</sub>t<sub>age.</sub> O<sub>range ou</sub>tli<sub>nes</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e a</sub>dditi<sub>ona</sub>l h<sub>ea</sub>d<sub>s</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> t<sub>op-</sub>10% <sub>se</sub>t f<sub>or</sub> th<sub>e</sub> <sub>secon</sub>d <sub>s</sub>t<sub>age.</sub> L<sub>ayer</sub> <sub>an</sub>d <sub>query-</sub>h<sub>ea</sub>d i<sub>n</sub>di<sub>ces</sub> <sub>s</sub>t<sub>ar</sub>t <sub>a</sub>t <sub>one.</sub> All <sub>se</sub>tti<sub>ngs w</sub>ithi<sub>n a mo</sub>d<sub>e</sub>l <sub>use</sub> th<sub>e same se</sub>l<sub>ec</sub>t<sub>e</sub>d h<sub>ea</sub>d<sub>s.</sub>

## G. Experimental Details

We conduct two experiments on Qwen3-8B and Llama-3.1-8B-Instruct: we compare diagnostic profiles across th<sub>e</sub> 49 <sub>s</sub>h<sub>are</sub>d t<sub>as</sub>k <sub>se</sub>tti<sub>ngs,</sub> <sub>an</sub>d <sub>we</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>ons</sub> i<sub>n</sub> <sub>eac</sub>h <sub>se</sub>tti<sub>ng</sub>’<sub>s</sub> <sub>ass</sub>i<sub>gne</sub>d <sub>seman</sub>ti<sub>c</sub> <sub>or pos</sub>iti<sub>ona</sub>l di<sub>rec</sub>ti<sub>on.</sub> A<sub>ppen</sub>di<sub>x</sub> E <sub>spec</sub>ifi<sub>es</sub> th<sub>e</sub> t<sub>as</sub>k d<sub>a</sub>t<sub>a, an</sub>d A<sub>ppen</sub>di<sub>x</sub> F d<sub>escr</sub>ib<sub>es</sub> th<sub>e</sub> fi<sub>xe</sub>d t<sub>o</sub>k<sub>en an</sub>d h<sub>ea</sub>d <sub>se</sub>l<sub>ec</sub>ti<sub>ons.</sub>

## G.1. Experiment 1: Task and Cross-Model Diagnostic Profiles

Each model uses 2<sub>,</sub>211 exam<sub>p</sub>les across the same 49 settin<sub>g</sub>s. The <sub>p</sub>rimar<sub>y</sub> dia<sub>g</sub>nostic uses the fixed to<sub>p</sub>-5% h<sub>ea</sub>d <sub>co</sub>ll<sub>ec</sub>ti<sub>on se</sub>l<sub>ec</sub>t<sub>e</sub>d f<sub>rom cross-</sub>t<sub>as</sub>k <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency norm s</sub>h<sub>are.</sub> All <sub>scores are compu</sub>t<sub>e</sub>d f<sub>rom</sub> <sub>cac</sub>h<sub>e</sub>d <sub>query</sub> <sub>an</sub>d k<sub>ey</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>a</sub>t <sub>eac</sub>h <sub>examp</sub>l<sub>e</sub>’<sub>s</sub> <sub>na</sub>ti<sub>ve</sub> i<sub>npu</sub>t l<sub>eng</sub>th <sub>an</sub>d f<sub>requency</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>on.</sub>

## G.1.1. State Extraction and Score Definitions

F<sub>or eac</sub>h t<sub>as</sub>k <sub>examp</sub>l<sub>e, we use samp</sub>l<sub>e</sub>d <sub>query–</sub>k<sub>ey pa</sub>i<sub>rs an</sub>d <sub>query–</sub>k<sub>ey–</sub>k<sub>ey</sub> t<sub>r</sub>i<sub>p</sub>l<sub>e</sub>t<sub>s,</sub> t<sub>oge</sub>th<sub>er w</sub>ith <sub>a</sub> fi<sub>xe</sub>d <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub> <sub>o</sub>f l<sub>ayer–</sub>h<sub>ea</sub>d <sub>un</sub>it<sub>s.</sub> I<sub>n</sub> <sub>eac</sub>h <sub>ro</sub>t<sub>ary</sub> <sub>p</sub>l<sub>ane,</sub> l<sub>e</sub>t $Q _ { n }$ <sub>an</sub>d $K _ { n }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>comp</sub>l<sub>ex</sub> f<sub>orms</sub> <sub>o</sub>f th<sub>e</sub> <sub>pre-</sub>R<sub>o</sub>PE <sub>query an</sub>d k<sub>ey componen</sub>t<sub>s.</sub> Th<sub>e</sub>i<sub>r score coe</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>s</sub>

$$
z _ { n } = c _ { \mathrm { a t t } } Q _ { n } \overline { { { K _ { n } } } } , \qquad S ( m ) = c _ { 0 } + \mathrm { R e } \sum _ { n } z _ { n } e ^ { i m \omega _ { n } } ,\tag{172}
$$

<sub>w</sub>h<sub>ere</sub> $c _ { \mathrm { a t t } }$ i<sub>s</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on-score</sub> <sub>sca</sub>l<sub>e</sub> <sub>an</sub>d $c _ { 0 }$ is an<sub>y</sub> non-rotar<sub>y</sub> score contribution (zero for full<sub>y</sub> rotar<sub>y</sub> heads). The extraction <sub>p</sub>reserves native <sub>q</sub>uer<sub>y</sub>/ke<sub>y</sub> normalization, rotar<sub>y</sub>-coordinate <sub>p</sub>airin<sub>g</sub>, and the ma<sub>pp</sub>in<sub>g</sub> b<sub>e</sub>tw<sub>ee</sub>n <sub>que</sub>r<sub>y</sub> h<sub>e</sub>ad<sub>s</sub> and k<sub>ey</sub>/val<sub>ue</sub> h<sub>e</sub>ad<sub>s.</sub> F<sub>o</sub>r <sub>p</sub>artiall<sub>y</sub> r<sub>o</sub>tar<sub>y</sub> h<sub>e</sub>ad<sub>s,</sub> th<sub>e</sub> n<sub>o</sub>n-r<sub>o</sub>tar<sub>y</sub> c<sub>o</sub>ntrib<sub>u</sub>ti<sub>o</sub>n i<sub>s</sub> <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>as</sub> <sub>a</sub> <sub>cons</sub>t<sub>an</sub>t i<sub>n</sub> <sub>seman</sub>ti<sub>c</sub> <sub>marg</sub>i<sub>ns;</sub> th<sub>e</sub> <sub>norm</sub> <sub>use</sub>d f<sub>or</sub> <sub>pos</sub>iti<sub>ona</sub>l <sub>response</sub> <sub>re</sub>f<sub>ers</sub> t<sub>o</sub> th<sub>e</sub> <sub>ro</sub>t<sub>ary</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub>

W<sub>e</sub> k<sub>eep</sub> th<sub>ese con</sub>t<sub>en</sub>t <sub>vec</sub>t<sub>ors</sub> fi<sub>xe</sub>d <sub>an</sub>d <sub>vary on</sub>l<sub>y</sub> th<sub>e curren</sub>t l<sub>ayer</sub>’<sub>s</sub> R<sub>o</sub>PE <sub>p</sub>h<sub>ase over</sub> $m = 0 , \ldots , M - 1$ <sub>.</sub> Th<sub>e</sub> <sub>curren</sub>t t<sub>as</sub>k <sub>pro</sub>t<sub>oco</sub>l <sub>se</sub>t<sub>s</sub> � t<sub>o</sub> th<sub>e examp</sub>l<sub>e</sub>’<sub>s</sub> i<sub>npu</sub>t l<sub>eng</sub>th<sub>.</sub> Th<sub>e score a</sub>t <sub>zero re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance supp</sub>li<sub>es</sub> th<sub>e</sub> <sub>re</sub>f<sub>erence compar</sub>i<sub>son.</sub> Th<sub>ese are v</sub>i<sub>r</sub>t<sub>ua</sub>l <sub>score eva</sub>l<sub>ua</sub>ti<sub>ons an</sub>d <sub>requ</sub>i<sub>re no new</sub> f<sub>orwar</sub>d <sub>pass once</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>es</sub> h<sub>ave</sub> b<sub>een recor</sub>d<sub>e</sub>d<sub>.</sub>

Semantic Score. For each triplet, orient the margin (including any non-rotary constant) so that $D ( 0 ) > 0 .$ <sub>w</sub>ith $d = z _ { + } - z _ { - }$ i<sub>n</sub> thi<sub>s</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng,</sub> <sub>an</sub>d <sub>compu</sub>t<sub>e</sub> th<sub>e</sub> f<sub>u</sub>ll fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow</sub> <sub>mean</sub> $\mu _ { D }$ <sub>an</sub>d <sub>var</sub>i<sub>ance</sub> $\sigma _ { D } ^ { 2 }$ <sub>ana</sub>l<sub>y</sub>ti<sub>ca</sub>ll<sub>y.</sub> Fi<sub>n</sub>it<sub>e</sub> <sub>geome</sub>t<sub>r</sub>i<sub>c</sub> <sub>sums</sub> <sub>g</sub>i<sub>ve</sub> th<sub>ese</sub> <sub>momen</sub>t<sub>s,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> <sub>covar</sub>i<sub>ance</sub> b<sub>e</sub>t<sub>ween</sub> f<sub>requency</sub> <sub>componen</sub>t<sub>s;</sub> A<sub>pp</sub>endix G.1.2 <sub>p</sub>rovides the ex<sub>p</sub>ressions. With Φ denotin<sub>g</sub> the standard normal cumulative distribution f<sub>unc</sub>ti<sub>on,</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an es</sub>ti<sub>ma</sub>t<sub>e</sub> i<sub>s</sub>

$$
{ \widehat { p } } _ { \mathrm { r e v } } ( d ; M ) = \Phi \left( - { \frac { \mu _ { D } } { \sigma _ { D } } } \right) , \qquad \sigma _ { D } > 0 .\tag{173}
$$

W<sub>e average</sub> th<sub>ese pro</sub>b<sub>a</sub>biliti<sub>es over va</sub>lid t<sub>r</sub>i<sub>p</sub>l<sub>e</sub>t<sub>s,</sub> th<sub>en over</sub> th<sub>e</sub> fi<sub>xe</sub>d h<sub>ea</sub>d <sub>co</sub>ll<sub>ec</sub>ti<sub>on, an</sub>d fi<sub>na</sub>ll<sub>y over</sub> t<sub>as</sub>k examples with equal example weights. The Semantic Score is one minus this task-level reversal rate. Higher <sub>va</sub>l<sub>ues</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e more s</sub>t<sub>a</sub>bl<sub>e re</sub>f<sub>erence or</sub>d<sub>er</sub>i<sub>ngs.</sub> R<sub>e</sub>f<sub>erence</sub> ti<sub>es are exc</sub>l<sub>u</sub>d<sub>e</sub>d <sub>an</sub>d th<sub>e</sub>i<sub>r coun</sub>t<sub>s are re</sub>t<sub>a</sub>i<sub>ne</sub>d<sub>.</sub> F<sub>or zero var</sub>i<sub>ance, we eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c s</sub>t<sub>r</sub>i<sub>c</sub>t<sub>-nega</sub>ti<sub>ve even</sub>t di<sub>rec</sub>tl<sub>y.</sub>

Th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on uses</sub> th<sub>e accep</sub>t<sub>e</sub>d G<sub>auss</sub>i<sub>an approx</sub>i<sub>ma</sub>ti<sub>on an</sub>d <sub>ana</sub>l<sub>y</sub>ti<sub>c momen</sub>t<sub>s.</sub> It <sub>avo</sub>id<sub>s</sub> <sub>enumera</sub>ti<sub>ng reversa</sub>l <sub>even</sub>t<sub>s a</sub>t <sub>every pos</sub>iti<sub>on.</sub> Th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency s</sub>h<sub>are prov</sub>id<sub>es</sub> th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> $\ S 3 ;$ th<sub>e prac</sub>ti<sub>ca</sub>l <sub>es</sub>ti<sub>ma</sub>t<sub>e uses</sub> th<sub>e</sub> f<sub>u</sub>ll fi<sub>n</sub>it<sub>e-w</sub>i<sub>n</sub>d<sub>ow momen</sub>t<sub>s an</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>vec</sub>t<sub>or.</sub> It<sub>s va</sub>l<sub>ue may vary</sub> <sub>non-mono</sub>t<sub>on</sub>i<sub>ca</sub>ll<sub>y w</sub>ith th<sub>e w</sub>i<sub>n</sub>d<sub>ow even</sub> th<sub>oug</sub>h th<sub>e conserva</sub>ti<sub>ve</sub> th<sub>eore</sub>ti<sub>ca</sub>l l<sub>ower</sub> b<sub>oun</sub>d i<sub>s mono</sub>t<sub>one.</sub> Thi<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on a</sub>ll<sub>ows</sub> th<sub>e</sub> t<sub>oo</sub>l t<sub>o repor</sub>t th<sub>e</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or o</sub>f <sub>samp</sub>l<sub>e</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s w</sub>hil<sub>e</sub> th<sub>e</sub> th<sub>eory exp</sub>l<sub>a</sub>i<sub>ns a</sub> necessary constraint on their joint semantic and positional reliability.

Positional Score. For each query–key pair with $U ( z ) > 0$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
P ( z ; M ) : = \frac { 1 } { M - 1 } \sum _ { m = 0 } ^ { M - 2 } \frac { | S ( m + 1 ) - S ( m ) | } { U ( z ) } .\tag{174}
$$

W<sub>e</sub> fi<sub>rs</sub>t <sub>norma</sub>li<sub>ze eac</sub>h <sub>pa</sub>i<sub>r</sub> b<sub>y</sub> it<sub>s own coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>norm,</sub> th<sub>en average over pa</sub>i<sub>rs,</sub> th<sub>e same</sub> fi<sub>xe</sub>d h<sub>ea</sub>d collection, and task examples. The resulting Positional Score measures average normalized adjacent response. Higher values indicate stronger local response. The score is computed directly from adjacent diferences; <sub>a zero coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>norm</sub> i<sub>s recor</sub>d<sub>e</sub>d <sub>as un</sub>d<sub>e</sub>fi<sub>ne</sub>d<sub>.</sub> Thi<sub>s conven</sub>ti<sub>on preserves</sub> th<sub>e per-pa</sub>i<sub>r sca</sub>l<sub>e</sub> b<sub>e</sub>f<sub>ore</sub> agg<sup>r</sup>ega<sup>ti</sup>o<sup>n</sup>.

Th<sub>e</sub> <sub>response</sub> <sub>con</sub>diti<sub>on</sub> i<sub>n</sub> Th<sub>eorem</sub> 1 <sub>app</sub>li<sub>es</sub> t<sub>o</sub> th<sub>e</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>marg</sub>i<sub>n</sub> $d = a - b $ . Requiring the average adjacent <sub>response o</sub>f th<sub>a</sub>t <sub>same marg</sub>i<sub>n</sub> t<sub>o reac</sub>h $\zeta$ <sub>a</sub>l<sub>so</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>e max</sub>i<sub>mum-response con</sub>diti<sub>on use</sub>d i<sub>n</sub> th<sub>e proo</sub>f<sub>.</sub> Th<sub>e</sub> P<sub>os</sub>iti<sub>ona</sub>l S<sub>core summar</sub>i<sub>zes</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>query–</sub>k<sub>ey curves an</sub>d <sub>supp</sub>li<sub>es a comp</sub>l<sub>emen</sub>t<sub>ary</sub> di<sub>agnos</sub>ti<sub>c o</sub>f l<sub>oca</sub>l <sub>response.</sub> Th<sub>e</sub> th<sub>eorem</sub> id<sub>en</sub>tifi<sub>es a spec</sub>t<sub>ra</sub>l <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>w</sub>h<sub>en</sub> b<sub>o</sub>th <sub>requ</sub>i<sub>remen</sub>t<sub>s are</sub> i<sub>mpose</sub>d <sub>on</sub> th<sub>e</sub> <sub>same marg</sub>i<sub>n.</sub> Th<sub>e</sub> t<sub>as</sub>k<sub>-</sub>l<sub>eve</sub>l <sub>pro</sub>fil<sub>e</sub> it<sub>se</sub>lf <sub>carr</sub>i<sub>es no con</sub>t<sub>ex</sub>t<sub>-</sub>l<sub>eng</sub>th <sub>cer</sub>tifi<sub>ca</sub>t<sub>e.</sub>

## G.1.2. Analytic Finite-Window Moments

For a real an<sub>g</sub>ular fre<sub>q</sub>uenc<sub>y</sub> $\theta ,$ d<sub>e</sub>fi<sub>ne</sub>

$$
K _ { M } ( \theta ) : = \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } e ^ { i m \theta } = \left\{ \frac { 1 - e ^ { i M \theta } } { M ( 1 - e ^ { i \theta } ) } , \quad e ^ { i \theta } \neq 1 , \right.\tag{175}
$$

For $\begin{array} { r } { D ( m ) = c _ { D } + \mathrm { R e } \sum _ { n } d _ { n } e ^ { i m \omega _ { n } } } \end{array}$ <sub>,</sub> l<sub>e</sub>t

$$
\mu _ { D } = c _ { D } + \mathrm { R e } \sum _ { n } d _ { n } K _ { M } ( \omega _ { n } ) ,\tag{176}
$$

$$
\begin{array} { r } { C _ { n \ell } ^ { - } = K _ { M } ( \omega _ { n } - \omega _ { \ell } ) - K _ { M } ( \omega _ { n } ) \overline { { K _ { M } ( \omega _ { \ell } ) } } , } \end{array}\tag{177}
$$

$$
C _ { n \ell } ^ { + } = K _ { M } ( \omega _ { n } + \omega _ { \ell } ) - K _ { M } ( \omega _ { n } ) K _ { M } ( \omega _ { \ell } ) .\tag{178}
$$

E<sub>xpan</sub>di<sub>ng</sub> th<sub>e</sub> <sub>square</sub> <sub>o</sub>f th<sub>e</sub> <sub>rea</sub>l <sub>par</sub>t <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>exac</sub>t <sub>var</sub>i<sub>ance</sub>

$$
\sigma _ { D } ^ { 2 } = \frac { 1 } { 2 } \mathrm { R e } \sum _ { n , \ell } \left( d _ { n } \overline { { { d _ { \ell } } } } C _ { n \ell } ^ { - } + d _ { n } d _ { \ell } C _ { n \ell } ^ { + } \right) .\tag{179}
$$

Here $c _ { D }$ i<sub>s</sub> th<sub>e</sub> dif<sub>erence o</sub>f <sub>any non-ro</sub>t<sub>ary score cons</sub>t<sub>an</sub>t<sub>s an</sub>d i<sub>s zero</sub> f<sub>or</sub> f<sub>u</sub>ll<sub>y ro</sub>t<sub>ary</sub> h<sub>ea</sub>d<sub>s.</sub> It <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> t<sub>o</sub> th<sub>e re</sub>f<sub>erence marg</sub>i<sub>n</sub> $D ( 0 )$ <sub>an</sub>d t<sub>o</sub> $\mu _ { D ; }$ <sub>, an</sub>d l<sub>eaves</sub> th<sub>e var</sub>i<sub>ance unc</sub>h<sub>ange</sub>d<sub>.</sub> Th<sub>ese express</sub>i<sub>ons requ</sub>i<sub>re no</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on on</sub> $d .$ <sub>.</sub> Th<sub>e su</sub>b<sub>sequen</sub>t G<sub>auss</sub>i<sub>an</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s an approx</sub>i<sub>ma</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance</sub> l<sub>aw, w</sub>ith th<sub>e prev</sub>i<sub>ous</sub>l<sub>y es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>ca</sub>lib<sub>ra</sub>ti<sub>on re</sub>t<sub>a</sub>i<sub>ne</sub>d<sub>.</sub>

## G.1.3. Relative Ranks and Failure Susceptibility

F<sub>or a c</sub>h<sub>osen mo</sub>d<sub>e</sub>l <sub>an</sub>d h<sub>ea</sub>d <sub>co</sub>ll<sub>ec</sub>ti<sub>on, ran</sub>k <sub>a</sub> fi<sub>xe</sub>d <sub>re</sub>f<sub>erence se</sub>t <sub>o</sub>f $N > 1$ t<sub>as</sub>k/l<sub>e</sub>n<sub>g</sub>th <sub>se</sub>ttin<sub>gs</sub> b<sub>y</sub> d<sub>ec</sub>r<sub>eas</sub>in<sub>g</sub> S<sub>eman</sub>ti<sub>c</sub> S<sub>core</sub> <sub>an</sub>d d<sub>ecreas</sub>i<sub>ng</sub> P<sub>os</sub>iti<sub>ona</sub>l S<sub>core.</sub> L<sub>e</sub>t $r _ { S }$ <sub>an</sub>d $r _ { P }$ b<sub>e</sub> th<sub>e</sub> <sub>respec</sub>ti<sub>ve</sub> <sub>ran</sub>k<sub>s,</sub> <sub>w</sub>ith <sub>ran</sub>k <sub>one</sub> b<sub>es</sub>t <sub>an</sub>d <sub>equa</sub>l <sub>va</sub>l<sub>ues</sub> <sub>rece</sub>i<sub>v</sub>i<sub>ng</sub> <sub>equa</sub>l <sub>compe</sub>titi<sub>on</sub> <sub>ran</sub>k<sub>s.</sub> W<sub>e</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>seman</sub>ti<sub>c</sub> <sub>an</sub>d <sub>pos</sub>iti<sub>ona</sub>l f<sub>a</sub>il<sub>ure</sub> <sub>suscep</sub>tibiliti<sub>es as</sub>

$$
w _ { S } = \frac { 1 } { 2 } + \frac { r _ { S } - r _ { P } } { 2 ( N - 1 ) } , \qquad w _ { P } = 1 - w _ { S } .\tag{180}
$$

A lar<sub>g</sub>er $\boldsymbol { w _ { S } }$ i<sub>n</sub>di<sub>ca</sub>t<sub>es a re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y wea</sub>k<sub>er seman</sub>ti<sub>c ran</sub>k<sub>, an</sub>d <sub>a</sub> l<sub>arger</sub> $w _ { P }$ i<sub>n</sub>di<sub>ca</sub>t<sub>es a re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y wea</sub>k<sub>er</sub> <sub>pos</sub>iti<sub>ona</sub>l <sub>ran</sub>k<sub>.</sub> E<sub>qua</sub>l <sub>ran</sub>k<sub>s g</sub>i<sub>ve</sub> $\begin{array} { r } { \psi _ { S } = w _ { P } = 1 / 2 } \end{array}$ <sub>.</sub> Th<sub>ese</sub> i<sub>n</sub>di<sub>ces</sub> d<sub>escr</sub>ib<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> f<sub>a</sub>il<sub>ure</sub> <sub>suscep</sub>tibilit<sub>y</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>e c</sub>h<sub>osen compar</sub>i<sub>son se</sub>t<sub>;</sub> th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>ti<sub>on</sub> d<sub>epen</sub>d<sub>s on</sub> th<sub>a</sub>t <sub>se</sub>t<sub>.</sub> R<sub>aw scores an</sub>d b<sub>e</sub>h<sub>av</sub>i<sub>ora</sub>l <sub>success ra</sub>t<sub>es</sub> <sub>rema</sub>i<sub>n v</sub>i<sub>s</sub>ibl<sub>e a</sub>l<sub>ongs</sub>id<sub>e</sub> th<sub>ese</sub> i<sub>n</sub>di<sub>ces.</sub>

## G.1.4. Aggregation and Cross-Model Comparison

S<sub>eman</sub>ti<sub>c reversa</sub>l <sub>pro</sub>b<sub>a</sub>biliti<sub>es are average</sub>d <sub>over va</sub>lid t<sub>r</sub>i<sub>p</sub>l<sub>e</sub>t<sub>s w</sub>ithi<sub>n eac</sub>h <sub>examp</sub>l<sub>e an</sub>d h<sub>ea</sub>d<sub>,</sub> th<sub>en over</sub> th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d h<sub>ea</sub>d<sub>s,</sub> th<sub>en equa</sub>ll<sub>y over examp</sub>l<sub>es.</sub> Th<sub>e</sub> S<sub>eman</sub>ti<sub>c</sub> S<sub>core</sub> i<sub>s one m</sub>i<sub>nus</sub> thi<sub>s average pro</sub>b<sub>a</sub>bilit<sub>y.</sub> F<sub>or</sub> the Positional Score, each pair’s adjacent diferences are divided by its own coeficient norm before averaging <sub>over pa</sub>i<sub>rs,</sub> h<sub>ea</sub>d<sub>s, an</sub>d <sub>examp</sub>l<sub>es.</sub> R<sub>e</sub>f<sub>erence</sub> ti<sub>es sa</sub>ti<sub>s</sub>f<sub>y</sub> $\begin{array} { r } { \left. D ( 0 ) \right. \leq 6 4 \epsilon _ { \mathrm { m a c h } } \sum _ { n } \left. d _ { n } \right. . } \end{array}$ <sub>;</sub> th<sub>e</sub>i<sub>r coun</sub>t<sub>s an</sub>d th<sub>e coun</sub>t<sub>s</sub> <sub>o</sub>f <sub>un</sub>d<sub>e</sub>fi<sub>ne</sub>d <sub>scores</sub> <sub>are</sub> <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d<sub>.</sub>

F<sub>o</sub>r Fi<sub>gu</sub>r<sub>e</sub> 6<sub>, eac</sub>h m<sub>o</sub>d<sub>e</sub>l r<sub>a</sub>nk<sub>s a</sub>ll 49 <sub>se</sub>ttin<sub>gs</sub> b<sub>y</sub> d<sub>ec</sub>r<sub>eas</sub>in<sub>g</sub> $w _ { S } ,$ assi<sub>g</sub>nin<sub>g</sub> avera<sub>g</sub>e ranks to ties. We com<sub>p</sub>ute S<sub>pearman corre</sub>l<sub>a</sub>ti<sub>on as</sub> th<sub>e</sub> P<sub>earson corre</sub>l<sub>a</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>ese</sub> t<sub>wo vec</sub>t<sub>ors o</sub>f <sub>average ran</sub>k<sub>s, w</sub>ith <sub>eac</sub>h <sub>se</sub>ttin<sub>g</sub> w<sub>e</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>equa</sub>ll<sub>y.</sub> Th<sub>e</sub> r<sub>esu</sub>ltin<sub>g co</sub>rr<sub>e</sub>l<sub>a</sub>ti<sub>o</sub>n i<sub>s</sub> 0<sub>.</sub>657983<sub>,</sub> r<sub>epo</sub>rt<sub>e</sub>d <sub>as</sub> 0<sub>.</sub>658 in th<sub>e</sub> m<sub>a</sub>in t<sub>e</sub>xt<sub>.</sub> It includes all 49 settings, including the three adjacent-element retrieval settings highlighted in orange. The plot preserves tied coordinates without jitter. These rankings describe failure susceptibility within the shared <sub>re</sub>f<sub>erence se</sub>t<sub>.</sub>

## G.2. Experiment 2: Directed High-Frequency Intervention

## G.2.1. Assigning the Optimization Direction

Th<sub>e</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> di<sub>rec</sub>ti<sub>on ran</sub>k<sub>s</sub> th<sub>e</sub> 49 <sub>se</sub>tti<sub>ngs</sub> b<sub>y</sub> d<sub>ecreas</sub>i<sub>ng</sub> $w _ { S } ,$ <sub>, ass</sub>i<sub>gn</sub>i<sub>ng</sub> th<sub>e average ran</sub>k t<sub>o</sub> ti<sub>es.</sub> L<sub>e</sub>t $\rho _ { S }$ d<sub>eno</sub>t<sub>e</sub> thi<sub>s ran</sub>k<sub>.</sub> S<sub>e</sub>tti<sub>ngs w</sub>ith $\rho _ { S } / 4 9 \le 1 / 2$ t<sub>es</sub>t <sub>re</sub>d<sub>uce</sub>d hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons;</sub> th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> settings test amplification. This places 24 Qwen settings and 23 Llama settings in the reduction group, k<sub>eep</sub>i<sub>ng</sub> ti<sub>es</sub> t<sub>oge</sub>th<sub>er.</sub> F<sub>or</sub> thi<sub>s re</sub>f<sub>erence se</sub>t<sub>, re</sub>d<sub>uc</sub>ti<sub>on correspon</sub>d<sub>s</sub> t<sub>o</sub> $\boldsymbol { w _ { S } }$ <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y a</sub>b<sub>ove</sub> th<sub>e mo</sub>d<sub>e</sub>l<sub>-spec</sub>ifi<sub>c</sub> median: 50.00% for Qwen and 56.25% for Llama. The value $\boldsymbol { w } _ { S } = 1 / 2$ <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>es equa</sub>l <sub>seman</sub>ti<sub>c</sub> <sub>an</sub>d <sub>pos</sub>iti<sub>ona</sub>l <sub>ran</sub>k<sub>s.</sub>

Th<sub>e re</sub>d<sub>uc</sub>ti<sub>on group op</sub>ti<sub>m</sub>i<sub>zes seman</sub>ti<sub>c s</sub>t<sub>a</sub>bilit<sub>y</sub> b<sub>y a</sub>tt<sub>enua</sub>ti<sub>ng</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency score con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons or</sub> f<sub>reez</sub>i<sub>ng</sub> th<sub>e</sub>i<sub>r ro</sub>t<sub>a</sub>ti<sub>ons.</sub> Th<sub>e amp</sub>lifi<sub>ca</sub>ti<sub>on group op</sub>ti<sub>m</sub>i<sub>zes pos</sub>iti<sub>ona</sub>l <sub>sens</sub>iti<sub>v</sub>it<sub>y</sub> b<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> Th<sub>e ass</sub>i<sub>gne</sub>d di<sub>rec</sub>ti<sub>on rema</sub>i<sub>ns</sub> fi<sub>xe</sub>d th<sub>roug</sub>h b<sub>o</sub>th <sub>s</sub>t<sub>ages o</sub>f th<sub>e searc</sub>h<sub>.</sub>

## G.2.2. Intervention Operators and Frequency Mask

F<sub>o</sub>r <sub>a sca</sub>l<sub>a</sub>r $\alpha ,$ <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d hi<sub>g</sub>h<sub>-</sub>f<sub>requency query an</sub>d k<sub>ey componen</sub>t<sub>s are eac</sub>h <sub>mu</sub>lti<sub>p</sub>li<sub>e</sub>d b<sub>y</sub> $\sqrt { \alpha } ,$ <sub>, w</sub>hi<sub>c</sub>h <sub>mu</sub>lti<sub>p</sub>li<sub>es</sub> th<sub>e</sub>i<sub>r</sub> <sub>score</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t b<sub>y</sub> <sub>�.</sub> L<sub>ow-</sub>f<sub>requency</sub> <sub>componen</sub>t<sub>s</sub> <sub>an</sub>d <sub>unse</sub>l<sub>ec</sub>t<sub>e</sub>d h<sub>ea</sub>d<sub>s</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub>i<sub>r</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>opera</sub>ti<sub>ons.</sub> W<sub>e</sub> <sub>p</sub>r<sub>ese</sub>rv<sub>e</sub> n<sub>a</sub>tiv<sub>e</sub> <sub>que</sub>r<sub>y</sub>/k<sub>ey</sub> n<sub>o</sub>rm<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd th<sub>e</sub> m<sub>app</sub>in<sub>g</sub> b<sub>e</sub>tw<sub>ee</sub>n <sub>que</sub>r<sub>y</sub> h<sub>ea</sub>d<sub>s</sub> <sub>a</sub>nd k<sub>ey</sub>/v<sub>a</sub>l<sub>ue</sub> h<sub>ea</sub>d<sub>s.</sub> Th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on c</sub>h<sub>anges</sub> b<sub>o</sub>th th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>norm an</sub>d it<sub>s</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency s</sub>h<sub>are.</sub> A <sub>separa</sub>t<sub>e no-ro</sub>t<sub>a</sub>ti<sub>on</sub> <sub>con</sub>diti<sub>on</sub> f<sub>reezes</sub> th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> R<sub>o</sub>PE <sub>ro</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>an</sub>d b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> th<sub>e</sub> <sub>seman</sub>ti<sub>c</sub> di<sub>rec</sub>ti<sub>on.</sub>

Th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>f<sub>requency</sub> <sub>mas</sub>k i<sub>s</sub> fi<sub>xe</sub>d f<sub>rom</sub> <sub>promp</sub>t l<sub>eng</sub>th � th<sub>roug</sub>h<sub>ou</sub>t <sub>genera</sub>ti<sub>on:</sub>

$$
H = \{ n : M \omega _ { n } \geq \gamma \} , \gamma = \frac { 2 c _ { \mathrm { i n t } } } { ( 1 - \rho ) \epsilon } , \rho = \theta ^ { - 2 / d _ { \mathrm { r o t } } } , c _ { \mathrm { i n t } } = 0 . 0 5 , \epsilon = 0 . 1 0 .\tag{181}
$$

Here � is the RoPE base<sub>,</sub> $d _ { \mathrm { { r o t } } }$ i<sub>s</sub> th<sub>e ro</sub>t<sub>ary</sub> di<sub>mens</sub>i<sub>on, an</sub>d $\omega _ { n }$ <sub>are</sub> th<sub>e ac</sub>t<sub>ua</sub>l <sub>run</sub>ti<sub>me</sub> f<sub>requenc</sub>i<sub>es,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>na</sub>ti<sub>ve</sub> f<sub>requency</sub> <sub>sca</sub>li<sub>ng.</sub> Th<sub>ese</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> <sub>runs</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub>i<sub>r</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>parame</sub>t<sub>er</sub> $c _ { \mathrm { i n t } } = 0 . 0 5$ th<sub>roug</sub>h<sub>ou</sub>t b<sub>o</sub>th <sub>searc</sub>h <sub>s</sub>t<sub>ages.</sub> Th<sub>e separa</sub>t<sub>e</sub> f<sub>requency-sp</sub>lit <sub>au</sub>dit i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> D<sub>.</sub>2<sub>.</sub>1 <sub>uses</sub> $c _ { \mathrm { o p } } = 0 . 0 6 ;$ it<sub>s</sub> l<sub>a</sub>t<sub>er</sub> <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> l<sub>eaves</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> <sub>mas</sub>k<sub>s</sub> <sub>an</sub>d <sub>repor</sub>t<sub>e</sub>d <sub>ou</sub>t<sub>comes</sub> <sub>unc</sub>h<sub>ange</sub>d<sub>.</sub>

## G.2.3. Two Search Stages

The first stage uses the diagnostic’s top-5% head collection: 58 heads for Qwen and 52 for Llama. Semantic di<sub>rec</sub>ti<sub>on</sub> <sub>se</sub>tti<sub>ngs</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> <sub>no</sub> <sub>ro</sub>t<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d $\alpha \in \{ 0 , 0 . 5 \}$ <sub>;</sub> <sub>pos</sub>iti<sub>ona</sub>l<sub>-</sub>di<sub>rec</sub>ti<sub>on</sub> <sub>se</sub>tti<sub>ngs</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> $\alpha \in \{ 1 . 5 , 2 , 2 . 5 \}$ Th<sub>e unc</sub>h<sub>ange</sub>d $\alpha = 1$ <sub>con</sub>diti<sub>on prov</sub>id<sub>es</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne.</sub> C<sub>an</sub>did<sub>a</sub>t<sub>e an</sub>d b<sub>ase</sub>li<sub>ne ou</sub>t<sub>pu</sub>t<sub>s s</sub>h<sub>are</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l i<sub>npu</sub>t ID<sub>s, gree</sub>d<sub>y</sub> d<sub>eco</sub>di<sub>ng, s</sub>t<sub>opp</sub>i<sub>ng ru</sub>l<sub>es, an</sub>d fi<sub>rs</sub>t<sub>-s</sub>t<sub>age ou</sub>t<sub>pu</sub>t li<sub>m</sub>it<sub>s.</sub> Th<sub>e</sub> $\alpha = 1$ i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on was</sub> <sub>c</sub>h<sub>ec</sub>k<sub>e</sub>d <sub>aga</sub>i<sub>ns</sub>t <sub>na</sub>ti<sub>ve-mo</sub>d<sub>e</sub>l <sub>ou</sub>t<sub>pu</sub>t<sub>s.</sub>

The second stage expands the same fixed head ranking to its top 10%: 116 Qwen heads and 103 Llama heads. It selects the 25 Qwen and 23 Llama settings without an observed first-stage improvement in their assigned di<sub>rec</sub>ti<sub>on.</sub> S<sub>eman</sub>ti<sub>c-</sub>di<sub>rec</sub>ti<sub>on se</sub>tti<sub>ngs eva</sub>l<sub>ua</sub>t<sub>e no ro</sub>t<sub>a</sub>ti<sub>on an</sub>d $\alpha \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 \}$ <sub>; pos</sub>iti<sub>ona</sub>l<sub>-</sub>di<sub>rec</sub>ti<sub>on</sub> <sub>se</sub>tti<sub>ngs eva</sub>l<sub>ua</sub>t<sub>e</sub> $\alpha \in \{ 1 . 2 5 , 1 . 5 , 1 . 7 5 , 2 \}$ <sub>.</sub> Th<sub>e</sub> di<sub>agnos</sub>ti<sub>c</sub> $\boldsymbol { w _ { S } }$ <sub>an</sub>d t<sub>as</sub>k <sub>ran</sub>ki<sub>ngs rema</sub>i<sub>n</sub> fi<sub>xe</sub>d <sub>a</sub>t th<sub>e</sub>i<sub>r</sub> t<sub>op-</sub>5% <sub>va</sub>l<sub>ues.</sub>

## G.2.4. Matched Examples, Scoring, and Output Budgets

We preserve the frozen first-stage comparison sets: 2,171 matched examples for Qwen and 2,074 for Llama. E<sub>ac</sub>h <sub>mo</sub>d<sub>e</sub>l h<sub>as</sub> 42 f<sub>u</sub>ll<sub>y covere</sub>d <sub>se</sub>tti<sub>ngs an</sub>d <sub>seven par</sub>ti<sub>a</sub>ll<sub>y covere</sub>d <sub>se</sub>tti<sub>ngs.</sub> Th<sub>e secon</sub>d <sub>s</sub>t<sub>age comp</sub>l<sub>e</sub>t<sub>e</sub>d 8,440 intervention records. Matching its candidates to saved baseline outputs gives 945 Qwen and 1,071 Ll<sub>a</sub>m<sub>a e</sub>x<sub>a</sub>m<sub>p</sub>l<sub>es;</sub> 16 <sub>a</sub>nd 78 <sub>e</sub>x<sub>a</sub>m<sub>p</sub>l<sub>es</sub> with<sub>ou</sub>t b<sub>ase</sub>lin<sub>e ou</sub>t<sub>pu</sub>t<sub>s a</sub>r<sub>e e</sub>x<sub>c</sub>l<sub>u</sub>d<sub>e</sub>d<sub>.</sub> A mi<sub>ss</sub>in<sub>g o</sub>r <sub>u</sub>nt<sub>es</sub>t<sub>e</sub>d r<sub>esu</sub>lt <sub>rema</sub>i<sub>ns</sub> <sub>unrepor</sub>t<sub>e</sub>d<sub>.</sub>

Withi<sub>n eac</sub>h <sub>s</sub>t<sub>age an</sub>d <sub>se</sub>tti<sub>ng, every</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>s compare</sub>d <sub>w</sub>ith th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l $\alpha = 1$ b<sub>ase</sub>li<sub>ne on</sub> th<sub>e</sub> <sub>same</sub> <sub>examp</sub>l<sub>es.</sub> A<sub>ccuracy</sub> i<sub>s</sub> th<sub>e</sub> f<sub>rac</sub>ti<sub>on</sub> <sub>o</sub>f <sub>examp</sub>l<sub>es</sub> <sub>rece</sub>i<sub>v</sub>i<sub>ng</sub> <sub>an</sub> <sub>o</sub>fi<sub>c</sub>i<sub>a</sub>l t<sub>as</sub>k <sub>score</sub> <sub>o</sub>f <sub>one.</sub> A<sub>ppen</sub>di<sub>x</sub> H <sub>repor</sub>t<sub>s</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d<sub>-</sub>di<sub>rec</sub>ti<sub>on can</sub>did<sub>a</sub>t<sub>e resu</sub>lt<sub>s an</sub>d <sub>ma</sub>t<sub>c</sub>h<sub>e</sub>d <sub>coun</sub>t<sub>s.</sub> Th<sub>e s</sub>t<sub>ages re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub>i<sub>r own compar</sub>i<sub>son</sub> sets: Qwen MV (eas<sub>y</sub>) has 94 exam<sub>p</sub>les in the first sta<sub>g</sub>e and 95 in the second; the other second-sta<sub>g</sub>e settin<sub>g</sub>s <sub>use</sub> th<sub>e</sub> <sub>same</sub> <sub>examp</sub>l<sub>es</sub> <sub>as</sub> th<sub>e</sub>i<sub>r</sub> fi<sub>rs</sub>t<sub>-s</sub>t<sub>age</sub> <sub>compar</sub>i<sub>sons.</sub>

Fi <sub>ure</sub> 5 i<sub>n</sub>t<sub>ersec</sub>t<sub>s</sub> th<sub>e com ar</sub>i<sub>son se</sub>t<sub>s across</sub> b<sub>o</sub>th <sub>s</sub>t<sub>a es, recom u</sub>t<sub>es</sub> th<sub>e</sub>i<sub>r can</sub>did<sub>a</sub>t<sub>es an</sub>d b<sub>ase</sub>li<sub>ne on</sub> thi<sub>s</sub> common set, and dis<sub>p</sub>la<sub>y</sub>s the lar<sub>g</sub>est observed <sub>g</sub>ain in the assi<sub>g</sub>ned direction. Thus Qwen MV (eas<sub>y</sub>) uses 94 <sub>examp</sub>l<sub>es</sub> f<sub>or</sub> b<sub>o</sub>th <sub>s</sub>t<sub>ages</sub> i<sub>n</sub> th<sub>e</sub> fi<sub>gure.</sub> Th<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> i<sub>s exc</sub>l<sub>u</sub>d<sub>e</sub>d f<sub>rom</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e max</sub>i<sub>mum, so se</sub>tti<sub>ngs</sub> f<sub>or</sub> <sub>w</sub>hi<sub>c</sub>h <sub>every can</sub>did<sub>a</sub>t<sub>e re</sub>d<sub>uces accuracy re</sub>t<sub>a</sub>i<sub>n a nega</sub>ti<sub>ve</sub> b<sub>ar.</sub>

Th<sub>e</sub> <sub>secon</sub>d <sub>s</sub>t<sub>age</sub> <sub>reuses</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l i<sub>npu</sub>t ID<sub>s</sub> <sub>an</sub>d b<sub>ase</sub>li<sub>ne</sub> <sub>ou</sub>t<sub>pu</sub>t<sub>s,</sub> <sub>w</sub>ith <sub>s</sub>h<sub>or</sub>t<sub>er</sub> <sub>genera</sub>ti<sub>on</sub> li<sub>m</sub>it<sub>s</sub> f<sub>or</sub> <sub>some</sub> t<sub>as</sub>k<sub>s.</sub> It<sub>s repor</sub>t<sub>e</sub>d <sub>compar</sub>i<sub>sons re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l b<sub>ase</sub>li<sub>ne</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> A<sub>n o</sub>fli<sub>ne con</sub>t<sub>ro</sub>l t<sub>runca</sub>t<sub>es save</sub>d b<sub>ase</sub>li<sub>ne</sub> tokens to the shorter limits before rescoring; this changes three Qwen and 28 Llama baseline correctness l<sub>a</sub>b<sub>e</sub>l<sub>s.</sub> Th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l b<sub>ase</sub>li<sub>ne</sub> i<sub>s use</sub>d i<sub>n a</sub>ll t<sub>a</sub>bl<sub>es an</sub>d th<sub>e ma</sub>i<sub>n</sub> fi<sub>gure.</sub>

Th<sub>e</sub> <sub>searc</sub>h <sub>repor</sub>t<sub>s</sub> th<sub>e</sub> b<sub>es</sub>t <sub>o</sub>b<sub>serve</sub>d <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>on</sub> th<sub>e</sub> <sub>examp</sub>l<sub>es</sub> <sub>use</sub>d t<sub>o</sub> <sub>compare</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> <sub>s</sub>t<sub>reng</sub>th<sub>s.</sub>

S<sub>econ</sub>d<sub>-s</sub>t<sub>age</sub> <sub>se</sub>tti<sub>ngs</sub> <sub>are</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d f<sub>rom</sub> fi<sub>rs</sub>t<sub>-s</sub>t<sub>age</sub> <sub>resu</sub>lt<sub>s.</sub> Th<sub>ese</sub> <sub>ga</sub>i<sub>ns</sub> d<sub>escr</sub>ib<sub>e</sub> th<sub>e</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>ons</sub> i<sub>n</sub> thi<sub>s</sub> <sub>exp</sub>l<sub>ora</sub>t<sub>ory</sub> <sub>searc</sub>h<sub>;</sub> <sub>c</sub>h<sub>oos</sub>i<sub>ng</sub> <sub>an</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on</sub> f<sub>or</sub> d<sub>ep</sub>l<sub>oymen</sub>t <sub>requ</sub>i<sub>res</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>on</sub> h<sub>e</sub>ld<sub>-ou</sub>t <sub>examp</sub>l<sub>es.</sub>

## H. Results in the Assigned Optimization Direction

W<sub>e repor</sub>t <sub>every measure</sub>d <sub>can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>n eac</sub>h <sub>se</sub>tti<sub>ng</sub>’<sub>s ass</sub>i<sub>gne</sub>d <sub>seman</sub>ti<sub>c or pos</sub>iti<sub>ona</sub>l di<sub>rec</sub>ti<sub>on,</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> <sub>pro</sub>t<sub>oco</sub>l i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> G<sub>.</sub>2<sub>.</sub> Th<sub>e</sub> t<sub>a</sub>bl<sub>es</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>a</sub>ll 49 <sub>se</sub>tti<sub>ngs</sub> <sub>per</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>groupe</sub>d b<sub>y</sub> di<sub>rec</sub>ti<sub>on,</sub> <sub>w</sub>ith <sub>separa</sub>t<sub>e</sub> <sub>resu</sub>lt<sub>s</sub> f<sub>or</sub> th<sub>e</sub> t<sub>op-</sub>5% <sub>an</sub>d t<sub>op-</sub>10% h<sub>ea</sub>d <sub>co</sub>ll<sub>ec</sub>ti<sub>ons.</sub>

Observed gains. The first stage improves 24 of 49 Qwen settings and 26 of 49 Llama settings. The second stage adds seven Qwen settings and six Llama settings, giving 31/49 (63.3%) and 32/49 (65.3%) improved settings, respectively. The largest gains are 20 percentage points for Qwen and 25 for Llama. These are th<sub>e</sub> b<sub>es</sub>t <sub>o</sub>b<sub>serve</sub>d <sub>ou</sub>t<sub>comes</sub> f<sub>rom a searc</sub>h <sub>over</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>on s</sub>t<sub>reng</sub>th<sub>s an</sub>d t<sub>wo</sub> h<sub>ea</sub>d <sub>su</sub>b<sub>se</sub>t<sub>s.</sub> Th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>can</sub>did<sub>a</sub>t<sub>es</sub> h<sub>ave no</sub>t b<sub>een va</sub>lid<sub>a</sub>t<sub>e</sub>d <sub>on</sub> h<sub>e</sub>ld<sub>-ou</sub>t <sub>examp</sub>l<sub>es.</sub>

Reading the tables. Each table retains its stage’s matched examples and original baseline. The count �/� <sub>g</sub>i<sub>ves ma</sub>t<sub>c</sub>h<sub>e</sub>d <sub>an</sub>d <sub>p</sub>l<sub>anne</sub>d <sub>examp</sub>l<sub>es.</sub> C<sub>an</sub>did<sub>a</sub>t<sub>e</sub> h<sub>ea</sub>d<sub>ers g</sub>i<sub>ve �, an</sub>d NR d<sub>eno</sub>t<sub>es no ro</sub>t<sub>a</sub>ti<sub>on.</sub> E<sub>ac</sub>h <sub>en</sub>t<sub>ry</sub> <sub>g</sub>ives accurac<sub>y</sub> (%) followed b<sub>y</sub> its chan<sub>g</sub>e from the <sub>p</sub>aired baseline (<sub>p</sub>ercenta<sub>g</sub>e <sub>p</sub>oints). Dashes indicate settin<sub>g</sub>s without a second-sta<sub>g</sub>e evaluation, and † marks a second-sta<sub>g</sub>e <sub>g</sub>eneration limit that difers from th<sub>e save</sub>d b<sub>ase</sub>li<sub>ne.</sub> All <sub>measure</sub>d <sub>ga</sub>i<sub>ns,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng zero an</sub>d <sub>nega</sub>ti<sub>ve va</sub>l<sub>ues, are re</sub>t<sub>a</sub>i<sub>ne</sub>d<sub>.</sub> Fi<sub>gure</sub> 5 <sub>uses</sub> th<sub>e</sub> common sam<sub>p</sub>les across sta<sub>g</sub>es: Qwen MV (eas<sub>y</sub>) uses 94 exam<sub>p</sub>les in the fi<sub>g</sub>ure and 95 in its second-sta<sub>g</sub>e table. All other settings use the same paired cohort in the figure and tables. The adjacent-element retrieval <sub>se</sub>tti<sub>ngs re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub>i<sub>r or</sub>i<sub>g</sub>i<sub>na</sub>l 12 <sub>examp</sub>l<sub>es a</sub>t <sub>eac</sub>h l<sub>eng</sub>th<sub>.</sub>

Th<sub>e</sub> <sub>ou</sub>t<sub>pu</sub>t<sub>-</sub>li<sub>m</sub>it <sub>con</sub>t<sub>ro</sub>l d<sub>escr</sub>ib<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> G<sub>.</sub>2 <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> 8/25 <sub>an</sub>d 7/23 i<sub>mprov</sub>i<sub>ng</sub> <sub>secon</sub>d<sub>-s</sub>t<sub>age</sub> <sub>se</sub>tti<sub>ngs</sub> <sub>w</sub>h<sub>en</sub> th<sub>e save</sub>d b<sub>ase</sub>li<sub>ne ou</sub>t<sub>pu</sub>t<sub>s are</sub> t<sub>runca</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e secon</sub>d<sub>-s</sub>t<sub>age</sub> li<sub>m</sub>it<sub>s</sub> b<sub>e</sub>f<sub>ore rescor</sub>i<sub>ng.</sub> Th<sub>e</sub> t<sub>a</sub>bl<sub>es re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub> <sub>o</sub>ri<sub>g</sub>in<sub>a</sub>l-<sub>ou</sub>t<sub>pu</sub>t <sub>co</sub>m<sub>pa</sub>ri<sub>so</sub>n<sub>s</sub> <sub>o</sub>f 7/25 <sub>a</sub>nd 6/23<sub>.</sub>

## H.1. Qwen3-8B: Semantic Direction

Table 2: Qwen3-8B: semantic direction on the top-5% heads. Entries give accuracy in percent and the change from the <sub>p</sub>aired baseline in <sub>p</sub>arentheses (<sub>p</sub>ercenta<sub>g</sub>e <sub>p</sub>oints).
<table><tr><td>Setting</td><td>ws (%)</td><td>n/N</td><td>Base</td><td>NR</td><td>0</td><td>0.5</td></tr><tr><td>QA (basic)</td><td></td><td>54.2 100/100 78.00</td><td></td><td>70.00 (-8.00)</td><td>76.00 (-2.00)</td><td>75.00 (-3.00)</td></tr><tr><td>QA (easy)</td><td></td><td>54.2 100/100 76.00</td><td></td><td>66.00 (-10.00)</td><td>70.00 (-6.00)</td><td>77.00 (+1.00)</td></tr><tr><td>Multi-hop QA (up to 16k)</td><td>53.1</td><td>25/25</td><td>28.00</td><td>24.00 (-4.00)</td><td>24.00 (-4.00)</td><td>24.00 (-4.00)</td></tr><tr><td>Rule deduction (context size 3,000)</td><td>51.0</td><td></td><td></td><td>25/25 72.00 88.00 (+16.00)</td><td>88.00(+16.00)</td><td>76.00 (+4.00)</td></tr><tr><td>Person location (16k)</td><td>61.5</td><td></td><td>25/25 76.00</td><td>80.00 (+4.00)</td><td>80.00 (+4.00)</td><td>68.00 (-8.00)</td></tr><tr><td>Previous location (8k)</td><td>56.2</td><td>25/25 28.00</td><td></td><td>32.00 (+4.00)</td><td>28.00 (+0.00)</td><td>32.00 (+4.00)</td></tr><tr><td>Previous location (16k)</td><td>60.4</td><td>25/25 32.00</td><td></td><td>32.00 (+0.00)</td><td>40.00 (+8.00)</td><td>36.00(+4.00)</td></tr><tr><td>Directional relations (8k)</td><td>70.8</td><td></td><td>25/25 48.00</td><td>44.00 (-4.00)</td><td>40.00 (-8.00)</td><td>40.00 (-8.00)</td></tr><tr><td>Directional relations (16k)</td><td>72.9</td><td></td><td>25/2548.00</td><td>56.00 (+8.00)</td><td>56.00 (+8.00)</td><td>56.00 (+8.00)</td></tr><tr><td>Giving events (8k)</td><td>62.5</td><td></td><td>25/25 72.00</td><td>80.00 (+8.00)</td><td>84.00(+12.00)</td><td>80.00 (+8.00)</td></tr><tr><td>Giving events (16k)</td><td>80.2</td><td></td><td>25/25 60.00</td><td>64.00 (+4.00)</td><td>64.00 (+4.00)</td><td>60.00 (+0.00)</td></tr><tr><td>Adjacent-element retrieval (3k)</td><td>99.0</td><td></td><td>12/12 50.00</td><td>25.00 (-25.00)</td><td>25.00 (-25.00)</td><td>33.33 (-16.67)</td></tr><tr><td>Adjacent-element retrieval (6k)</td><td>95.8</td><td></td><td>12/12 25.00</td><td>8.33 (-16.67)</td><td>0.00 (-25.00)</td><td>0.00 (-25.00)</td></tr><tr><td>Adjacent-element retrieval (13k)</td><td>97.9</td><td></td><td>12/12 16.67</td><td>8.33 (-8.33)</td><td>8.33 (-8.33)</td><td>8.33 (-8.33)</td></tr><tr><td>Object location (16k)</td><td>74.0</td><td></td><td>25/2540.00</td><td>40.00 (+0.00)</td><td>32.00 (-8.00)</td><td>24.00 (-16.00)</td></tr><tr><td>Object counting (8k)</td><td>68.8</td><td></td><td>25/25 20.00</td><td>24.00 (+4.00)</td><td>24.00 (+4.00)</td><td>24.00 (+4.00)</td></tr><tr><td>Object counting (16k)</td><td>77.1</td><td>25/25</td><td></td><td>4.00 20.00 (+16.00)</td><td>24.00(+20.00)</td><td>16.00 (+12.00)</td></tr></table>

Continued on the next page.

Continued from the preceding page.
<table><tr><td>Setting</td><td>ws (%)</td><td> $n / N$ </td><td>Base</td><td>NR</td><td>0</td><td>0.5</td></tr><tr><td>Object sets (16k)</td><td>56.2</td><td>25/25 36.00</td><td></td><td> $3 6 . 0 0 \left( + 0 . 0 0 \right)$ </td><td> $3 6 . 0 0 \left( + 0 . 0 0 \right)$ </td><td> $3 6 . 0 0 \left( + 0 . 0 0 \right)$ </td></tr><tr><td>Negation (8k)</td><td>54.2</td><td>25/2588.00</td><td></td><td> $8 8 . 0 0 \left( + 0 . 0 0 \right)$ </td><td> $9 2 . 0 0 \left( + 4 . 0 0 \right)$ </td><td> $9 2 . 0 0 \left( + 4 . 0 0 \right)$ </td></tr><tr><td>Negation (16k)</td><td>71.9</td><td>25/2568.00</td><td></td><td> $7 2 . 0 0 \ : ( + 4 . 0 0 )$ </td><td> $7 6 . 0 0 \left( + 8 . 0 0 \right)$ </td><td> $7 6 . 0 0 ( + 8 . 0 0 )$ </td></tr><tr><td>Uncertain knowledge (16k)</td><td>62.5</td><td>25/2544.00</td><td></td><td> $4 8 . 0 0 \left( + 4 . 0 0 \right)$ </td><td> $4 8 . 0 0 \left( + 4 . 0 0 \right)$ </td><td> $4 8 . 0 0 \left( + 4 . 0 0 \right)$ </td></tr><tr><td>One-hop implicit retrieval (16k)</td><td>61.5</td><td>25/25 12.00</td><td></td><td> $4 . 0 0 \left( - 8 . 0 0 \right)$ </td><td> $8 . 0 0 \left( - 4 . 0 0 \right)$ </td><td> $1 2 . 0 0 \left( + 0 . 0 0 \right)$ </td></tr><tr><td>Two-hop implicit retrieval (8k)</td><td>65.6</td><td>25/25</td><td>0.00</td><td> $0 . 0 0 \left( + 0 . 0 0 \right)$ </td><td> $0 . 0 0 \left( + 0 . 0 0 \right)$ </td><td> $0 . 0 0 \left( + 0 . 0 0 \right)$ </td></tr><tr><td>Two-hop implicit retrieval (16k)</td><td>67.7</td><td>25/25</td><td>0.00</td><td> $0 . 0 0 \left( + 0 . 0 0 \right)$ </td><td> $0 . 0 0 \left( + 0 . 0 0 \right)$ </td><td> $0 . 0 0 \left( + 0 . 0 0 \right)$ </td></tr></table>

Table 3: Qwen3-8B: semantic direction on the top-10% heads. Entries give accuracy in percent and the change from the <sub>p</sub>aired baseline in <sub>p</sub>arentheses (<sub>p</sub>ercenta<sub>g</sub>e <sub>p</sub>oints). Dashes mark settin<sub>g</sub>s without a second-sta<sub>g</sub>e evaluation; † marks a <sub>g</sub>eneration limit that difers from the saved baseline.
<table><tr><td>Setting</td><td> $n / N$  Base</td><td>NR</td><td>0.25</td><td>0.5</td><td>0.75</td></tr><tr><td>QA (basic)†</td><td>100/100 78.00</td><td>63.00 (-15.00)</td><td>82.00(+4.00)</td><td>80.00(+2.00)</td><td>78.00 (+0.00)</td></tr><tr><td>QA (easy) Multi-hop QA (up to 16k)</td><td>25/25 28.00</td><td>24.00 (-4.00)</td><td>24.00 (-4.00)</td><td></td><td>24.00 (-4.00) 28.00 (+0.00)</td></tr><tr><td>Rule deduction (context size</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3,000) Person location (16k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Previous location (8k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Previous location (16k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Directional relations (8k)</td><td>25/2548.00</td><td></td><td>44.00 (-4.00)36.00 (-12.00)</td><td></td><td>40.00 (-8.00)52.00 (+4.00)</td></tr><tr><td>Directional relations (16k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Giving events (8k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Giving events (16k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Adjacent-element retrieval (3k)</td><td>12/12 50.00</td><td></td><td>41.67 (-8.33)25.00 (-25.00)</td><td>25.00 (-25.00)</td><td>33.33 (-16.67)</td></tr><tr><td>Adjacent-element retrieval (6k)</td><td></td><td>12/12 25.0025.00 (+0.00)</td><td>16.67 (-8.33)</td><td>16.67 (-8.33)</td><td>16.67 (-8.33)</td></tr><tr><td>Adjacent-element retrieval (13k)</td><td>12/1216.67</td><td>8.33 (-8.33)</td><td>8.33 (-8.33)</td><td>8.33 (-8.33)</td><td>16.67(+0.00)</td></tr><tr><td>Object location (16k)</td><td></td><td></td><td>25/2540.0028.00 (-12.00)28.00 (-12.00)28.00 (-12.00)</td><td></td><td>32.00 (-8.00)</td></tr><tr><td>Object counting (8k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Object counting (16k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Object sets (16k)</td><td>25/2536.00</td><td></td><td></td><td></td><td>32.00 (-4.00)44.00 (+8.00)44.00 (+8.00)44.00 (+8.00)</td></tr><tr><td>Negation (8k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Negation (16k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uncertain knowledge (16k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>One-hop implicit retrieval (16k)</td><td>25/25 12.00</td><td>0.00 (-12.00)</td><td>12.00 (+0.00)</td><td>12.00(+0.00)</td><td>12.00(+0.00)</td></tr><tr><td>Two-hop implicit retrieval (8k)</td><td>25/25 0.00</td><td>0.00 (+0.00)</td><td>0.00 (+0.00)</td><td>0.00 (+0.00)</td><td>0.00 (+0.00)</td></tr><tr><td>Two-hop implicit retrieval (16k)</td><td>25/25 0.00</td><td>0.00 (+0.00)</td><td>0.00 (+0.00)</td><td>0.00 (+0.00)</td><td>0.00 (+0.00)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## H.2. Qwen3-8B: Positional Direction

Table 4: Qwen3-8B: positional direction on the top-5% heads. Entries give accuracy in percent and the chan<sub>g</sub>e from the <sub>p</sub>aired baseline in <sub>p</sub>arentheses (<sub>p</sub>ercenta<sub>g</sub>e <sub>p</sub>oints).
<table><tr><td>Setting</td><td>ws (%)</td><td> $n / N$ </td><td>Base</td><td>1.5</td><td></td><td></td></tr><tr><td>MK (basic)</td><td></td><td>49.0 100/100</td><td>99.00</td><td>100.00(+1.00)</td><td>100.00(+1.00)</td><td>100.00(+1.00)</td></tr><tr><td>MK (easy)</td><td></td><td>16.7 100/100</td><td>96.00</td><td>97.00 (+1.00)</td><td>97.00 (+1.00)</td><td>97.00(+1.00)</td></tr><tr><td>MK (medium)</td><td>22.9</td><td>95/100</td><td>72.63</td><td>72.63(+0.00)</td><td>70.53 (-2.11)</td><td>70.53 (-2.11)</td></tr><tr><td>MK (hard)</td><td>32.3</td><td>95/100</td><td>64.21</td><td>67.37(+3.16)</td><td>68.42(+4.21)</td><td>70.53(+6.32)</td></tr></table>

Continued on the next page.

Continued from the preceding page.
<table><tr><td>Setting</td><td>ws (%)</td><td>n/N</td><td>Base</td><td>1.5</td><td>2</td><td>2.5</td></tr><tr><td>MV (basic)</td><td></td><td>46.9 100/100</td><td>22.00</td><td>15.00 (-7.00)</td><td>19.00 (-3.00)</td><td>20.00 (-2.00)</td></tr><tr><td>MV (easy)</td><td>15.6</td><td>94/100</td><td>43.62</td><td>43.62(+0.00)</td><td>42.55 (-1.06)</td><td>37.23 (-6.38)</td></tr><tr><td>MV (medium)</td><td>35.4</td><td>94/100</td><td>35.11</td><td>38.30 (+3.19)</td><td>37.23 (+2.13)</td><td>36.17(+1.06)</td></tr><tr><td>MV (hard)</td><td>22.9</td><td>94/100</td><td>52.13</td><td>53.19(+1.06)</td><td>50.00 (-2.13)</td><td>48.94 (-3.19)</td></tr><tr><td>QA (medium)</td><td>42.7</td><td>94/100</td><td>69.15</td><td>72.34(+3.19)</td><td>71.28 (+2.13)</td><td>71.28(+2.13)</td></tr><tr><td>QA (hard)</td><td>40.6</td><td>94/100</td><td>78.72</td><td>78.72(+0.00)</td><td>73.40 (-5.32)</td><td>76.60 (-2.13)</td></tr><tr><td>Paper QA (up to 16k)</td><td>12.5</td><td>25/25</td><td>28.00</td><td>20.00 (-8.00)</td><td>20.00 (-8.00)</td><td>20.00 (-8.00)</td></tr><tr><td>Conversation topic retrieval</td><td>21.9</td><td>150/150</td><td>58.00</td><td>64.67(+6.67)</td><td>66.00 (+8.00)</td><td>64.67(+6.67)</td></tr><tr><td>Text sorting (1k)</td><td>12.5</td><td>25/25</td><td>8.00</td><td>8.00 (+0.00)</td><td>8.00 (+0.00)</td><td>8.00(+0.00)</td></tr><tr><td>Room-property reasoning (context size 500)</td><td>16.7</td><td>25/25</td><td>100.00</td><td>100.00 (+0.00)</td><td>100.00 (+0.00)</td><td>100.00 (+0.00)</td></tr><tr><td>Room-property reasoning (context size 3,000)</td><td>50.0</td><td></td><td></td><td>25/25 100.00 100.00 (+0.00)</td><td>100.00(+0.00)100.00(+0.00)</td><td></td></tr><tr><td>Relation chaining (context size 500)</td><td>20.8</td><td>25/25</td><td>96.00</td><td>92.00 (-4.00)</td><td>96.00 (+0.00)</td><td>96.00 (+0.00)</td></tr><tr><td>Relation chaining (context size 3,000)</td><td>41.7</td><td>25/25</td><td>76.00</td><td>72.00 (-4.00)</td><td>68.00 (-8.00)</td><td>64.00 (-12.00)</td></tr><tr><td>Rule deduction (context size 500)</td><td>20.8</td><td>25/25</td><td>84.00</td><td>84.00 (+0.00)</td><td>72.00 (-12.00)</td><td>72.00 (-12.00)</td></tr><tr><td>Mixed long reasoning (8k)</td><td>45.8</td><td>25/25</td><td>84.00</td><td>80.00 (-4.00)</td><td>88.00 (+4.00)</td><td>88.00 (+4.00)</td></tr><tr><td>Mixed long reasoning (16k)</td><td>50.0</td><td>25/25</td><td>72.00</td><td>84.00(+12.00)</td><td>88.00 (+16.00)</td><td>84.00 (+12.00)</td></tr><tr><td>Person location (8k)</td><td>35.4</td><td>25/25</td><td>92.00</td><td>88.00 (-4.00)</td><td>96.00 (+4.00)</td><td>96.00 (+4.00)</td></tr><tr><td>Object location (8k)</td><td>37.5</td><td>25/25</td><td>52.00</td><td>48.00 (-4.00)</td><td>48.00 (-4.00)</td><td>48.00 (-4.00)</td></tr><tr><td>Object sets (8k)</td><td>34.4</td><td>25/25</td><td>52.00</td><td>36.00 (-16.00)</td><td>36.00 (-16.00)</td><td>36.00 (-16.00)</td></tr><tr><td>Uncertain knowledge (8k)</td><td>45.8</td><td>25/25</td><td>64.00</td><td>60.00 (-4.00)</td><td>56.00 (-8.00)</td><td>56.00 (-8.00)</td></tr><tr><td>One-hop implicit retrieval (8k)</td><td>50.0</td><td>25/25</td><td>4.00</td><td>12.00 (+8.00)</td><td>12.00(+8.00)</td><td>16.00 (+12.00)</td></tr></table>

Table 5: Qwen3-8B: positional direction on the top-10% heads. Entries give accuracy in percent and the chan<sub>g</sub>e from the <sub>p</sub>aired baseline in <sub>p</sub>arentheses (<sub>p</sub>ercenta<sub>g</sub>e <sub>p</sub>oints). Dashes mark settin<sub>g</sub>s without a second-sta<sub>g</sub>e evaluation; † marks a <sub>g</sub>eneration limit that difers from the saved baseline.
<table><tr><td>Setting</td><td>n/N</td><td>Base</td><td>1.25</td><td>1.5</td><td>1.75</td><td>2</td></tr><tr><td>MK (basic)</td><td></td><td>一</td><td>—</td><td></td><td>一</td><td></td></tr><tr><td>MK (easy)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MK (medium)†</td><td>95/100</td><td>72.63</td><td>70.53 (-2.11)</td><td>69.47 (-3.16)</td><td>67.37 (-5.26)</td><td>67.37 (-5.26)</td></tr><tr><td>MK (hard) MV (basic)†</td><td>100/100</td><td>22.00</td><td>22.00 (+0.00)</td><td>24.00 (+2.00)</td><td>27.00(+5.00)</td><td>28.00(+6.00)</td></tr><tr><td>MV (easy)</td><td>95/100</td><td>44.21</td><td>40.00 (-4.21)</td><td>37.89 (-6.32)</td><td>33.68 (-10.53)</td><td>28.42 (-15.79)</td></tr><tr><td>MV (medium)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MV (hard)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>QA (medium)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>QA (hard)†</td><td>94/100</td><td>78.72</td><td>79.79(+1.06)</td><td>77.66 (-1.06)</td><td>74.47(-4.26)</td><td>72.34 (-6.38)</td></tr><tr><td>Paper QA (up to 16k)</td><td>25/25</td><td>28.00</td><td>24.00 (-4.00)</td><td>24.00 (-4.00)</td><td>24.00 (-4.00)</td><td>20.00 (-8.00)</td></tr><tr><td>Conversation topic retrieval</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Text sorting (1k)</td><td>25/25</td><td>8.00</td><td>8.00 (+0.00)</td><td>8.00 (+0.00)</td><td>8.00 (+0.00)</td><td>8.00(+0.00)</td></tr><tr><td>Room-property reasoning</td><td>25/25</td><td>100.00</td><td>100.00 (+0.00)</td><td>100.00(+0.00)</td><td>100.00 (+0.00)</td><td>100.00(+0.00)</td></tr><tr><td>(context size 500) Room-property reasoning</td><td></td><td></td><td>25/25100.00100.00 (+0.00)</td><td>100.00 (+0.00)</td><td>100.00(+0.00)</td><td>100.00(+0.00)</td></tr><tr><td>(context size 3,000)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Relation chaining (context size 500)</td><td>25/25</td><td>96.00</td><td>92.00 (-4.00)</td><td>92.00 (-4.00)</td><td>92.00 (-4.00)</td><td>92.00 (-4.00)</td></tr><tr><td>Relation chaining (context size 3,000)</td><td>25/25</td><td>76.00</td><td>72.00 (-4.00)</td><td>72.00 (-4.00)</td><td>68.00 (-8.00)</td><td>60.00 (-16.00)</td></tr></table>

Continued from the preceding page.
<table><tr><td>Setting</td><td>n/N</td><td>Base</td><td>1.25</td><td>1.5</td><td>1.75</td><td>2</td></tr><tr><td>Rule deduction (context size 500)</td><td>25/25</td><td>84.00</td><td>84.00 (+0.00)</td><td>88.00 (+4.00)</td><td>88.00 (+4.00)</td><td>84.00(+0.00)</td></tr><tr><td>Mixed long reasoning (8k)</td><td></td><td></td><td></td><td></td><td></td><td>一</td></tr><tr><td>Mixed long reasoning (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Person location (8k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Object location (8k)</td><td>25/25</td><td>52.00</td><td>56.00 (+4.00)</td><td>60.00 (+8.00)</td><td>56.00(+4.00)</td><td>56.00 (+4.00)</td></tr><tr><td>Object sets (8k)</td><td>25/25</td><td>52.00</td><td>44.00 (-8.00)</td><td>40.00 (-12.00)</td><td>40.00 (-12.00)</td><td>40.00 (-12.00)</td></tr><tr><td>Uncertain knowledge (8k)</td><td>25/25</td><td>64.00</td><td>52.00 (-12.00)</td><td>44.00 (-20.00)</td><td>44.00 (-20.00)</td><td>44.00 (-20.00)</td></tr><tr><td>One-hop implicit retrieval (8k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## H.3. Llama-3.1-8B-Instruct: Semantic Direction

Table 6: Llama-3.1-8B-Instruct: semantic direction on the to<sub>p</sub>-5% heads. Entries <sub>g</sub>ive accurac<sub>y</sub> in <sub>p</sub>ercent and the chan<sub>g</sub>e from the <sub>p</sub>aired baseline in <sub>p</sub>arentheses (<sub>p</sub>ercenta<sub>g</sub>e <sub>p</sub>oints).
<table><tr><td>Setting</td><td>ws (%)</td><td>n/N</td><td>Base</td><td>NR</td><td>0</td><td>0.5</td></tr><tr><td>QA (easy)</td><td>64.6</td><td>80/100</td><td>80.00</td><td>86.25(+6.25)</td><td>87.50(+7.50)</td><td>86.25(+6.25)</td></tr><tr><td>Multi-hop QA (up to 16k)</td><td>67.7</td><td></td><td>25/25 24.00</td><td>20.00 (-4.00)</td><td>16.00 (-8.00)</td><td>16.00 (-8.00)</td></tr><tr><td>Person location (16k)</td><td>77.1</td><td></td><td>25/25 76.00</td><td>76.00 (+0.00)</td><td>84.00 (+8.00)</td><td>84.00 (+8.00)</td></tr><tr><td>Previous location (8k)</td><td>62.5</td><td></td><td>25/25 28.00</td><td>36.00 (+8.00)</td><td>28.00(+0.00)</td><td>32.00(+4.00)</td></tr><tr><td>Previous location (16k)</td><td>80.2</td><td></td><td>25/25 20.00</td><td>28.00 (+8.00)</td><td></td><td>24.00 (+4.00)24.00 (+4.00)</td></tr><tr><td>Directional relations (8k)</td><td>69.8</td><td></td><td>25/25 36.00</td><td>40.00 (+4.00)</td><td>36.00 (+0.00)40.00 (+4.00)</td><td></td></tr><tr><td>Directional relations (16k)</td><td>70.8</td><td>25/25</td><td></td><td>44.0056.00 (+12.00)</td><td></td><td>48.00 (+4.00)48.00 (+4.00)</td></tr><tr><td>Giving events (8k)</td><td>57.3</td><td></td><td>25/25 72.00</td><td>72.00 (+0.00)</td><td>68.00 (-4.00)</td><td>72.00(+0.00)</td></tr><tr><td>Giving events (16k)</td><td>58.3</td><td></td><td>25/25 76.00</td><td>64.00 (-12.00)</td><td>64.00 (-12.00)</td><td>72.00 (-4.00)</td></tr><tr><td>Object location (8k)</td><td>57.3</td><td></td><td>25/25 52.00</td><td>52.00 (+0.00)</td><td>40.00 (-12.00)</td><td>44.00 (-8.00)</td></tr><tr><td>Object location (16k)</td><td>67.7</td><td>25/25</td><td>36.00</td><td>40.00 (+4.00)</td><td>32.00 (-4.00)</td><td>32.00 (-4.00)</td></tr><tr><td>Object counting (8k)</td><td>65.6</td><td>25/25</td><td>4.00</td><td>12.00 (+8.00)</td><td>8.00 (+4.00)</td><td>12.00 (+8.00)</td></tr><tr><td>Object counting (16k)</td><td>80.2</td><td>25/25</td><td>8.00</td><td>12.00 (+4.00)</td><td>16.00 (+8.00)</td><td>12.00(+4.00)</td></tr><tr><td>Object sets (8k)</td><td>57.3</td><td>25/25</td><td>44.00</td><td>44.00 (+0.00)</td><td>52.00 (+8.00)</td><td>)44.00 (+0.00)</td></tr><tr><td>Object sets (16k)</td><td>67.7</td><td></td><td>25/25 40.00</td><td>48.00 (+8.00)</td><td>48.00(+8.00)</td><td>48.00(+8.00)</td></tr><tr><td>Negation (8k)</td><td>58.3</td><td>25/25</td><td>88.00</td><td>88.00 (+0.00)</td><td>84.00 (-4.00)</td><td>84.00 (-4.00)</td></tr><tr><td>Negation (16k)</td><td>76.0</td><td></td><td>25/25 72.00</td><td>64.00 (-8.00)</td><td>72.00 (+0.00)</td><td>72.00 (+0.00)</td></tr><tr><td>Uncertain knowledge (8k)</td><td>71.9</td><td></td><td>25/25 76.00</td><td>80.00 (+4.00)</td><td>72.00 (-4.00)</td><td>72.00 (-4.00)</td></tr><tr><td>Uncertain knowledge (16k)</td><td>80.2</td><td>25/25</td><td>64.00</td><td>68.00 (+4.00)</td><td>56.00 (-8.00)</td><td>68.00 (+4.00)</td></tr><tr><td>One-hop implicit retrieval (8k)</td><td>70.8</td><td></td><td>25/25 36.00</td><td>40.00 (+4.00)</td><td>24.00 (-12.00)</td><td>40.00 (+4.00)</td></tr><tr><td>One-hop implicit retrieval (16k)</td><td>83.3</td><td>25/25</td><td>24.00</td><td>28.00 (+4.00)</td><td>36.00(+12.00)</td><td>28.00(+4.00)</td></tr><tr><td>Two-hop implicit retrieval (8k)</td><td>72.9</td><td>25/25</td><td>4.00</td><td>0.00 (-4.00)</td><td>8.00 (+4.00)</td><td>8.00 (+4.00)</td></tr><tr><td>Two-hop implicit retrieval (16k)</td><td>70.8</td><td>25/25</td><td>0.00</td><td>4.00(+4.00)</td><td>0.00 (+0.00)</td><td>4.00 (+4.00)</td></tr></table>

Table 7: Llama-3.1-8B-Instruct: semantic direction on the to<sub>p</sub>-10% heads. Entries <sub>g</sub>ive accurac<sub>y</sub> in <sub>p</sub>ercent and the chan<sub>g</sub>e from the <sub>p</sub>aired baseline in <sub>p</sub>arentheses (<sub>p</sub>ercenta<sub>g</sub>e <sub>p</sub>oints). Dashes mark settin<sub>g</sub>s without a second-sta<sub>g</sub>e evaluation; † marks a <sub>g</sub>eneration limit that difers from the saved baseline.
<table><tr><td>Setting</td><td> $n / N$ </td><td>Base</td><td>NR</td><td>0.25</td><td>0.5</td><td>0.75</td></tr><tr><td>QA (easy)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Multi-hop QA (up to 16k)</td><td></td><td></td><td>25/25 24.0024.00 (+0.00)</td><td>8.00 (-16.00)</td><td>8.00 (-16.00)</td><td>16.00 (-8.00)</td></tr><tr><td>Person location (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Previous location (8k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Continued on the next page.

Continued from the preceding page.
<table><tr><td>Setting</td><td>n/N</td><td>Base</td><td>NR</td><td>0.25</td><td>0.5</td><td>0.75</td></tr><tr><td>Previous location (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Directional relations (8k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Directional relations (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Giving events (8k)</td><td></td><td></td><td>25/25 72.0072.00 (+0.00)</td><td></td><td>68.00 (-4.00) 72.00 (+0.00)</td><td>72.00(+0.00)</td></tr><tr><td>Giving events (16k)</td><td></td><td>25/2576.00</td><td>68.00 (-8.00)64.00 (-12.00)</td><td></td><td>68.00 (-8.00)</td><td>72.00 (-4.00)</td></tr><tr><td>Object location (8k)</td><td></td><td></td><td>25/2552.00 20.00 (-32.00)</td><td>48.00 (-4.00)40.00 (-12.00)</td><td></td><td>44.00 (-8.00)</td></tr><tr><td>Object location (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Object counting (8k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Object counting (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Object sets (8k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Object sets (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Negation (8k)</td><td></td><td>25/25 88.00</td><td>80.00 (-8.00) 76.00 (-12.00)</td><td></td><td>84.00 (-4.00)</td><td>84.00 (-4.00)</td></tr><tr><td>Negation (16k)</td><td></td><td></td><td>25/2572.0052.00 (-20.00)60.00 (-12.00)</td><td></td><td>64.00 (-8.00)</td><td>68.00 (-4.00)</td></tr><tr><td>Uncertain knowledge (8k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uncertain knowledge (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>One-hop implicit retrieval (8k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>One-hop implicit retrieval (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Two-hop implicit retrieval (8k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Two-hop implicit retrieval (16k)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## H.4. Llama-3.1-8B-Instruct: Positional Direction

Table 8: Llama-3.1-8B-Instruct: <sub>p</sub>ositional direction on the to<sub>p</sub>-5% heads. Entries <sub>g</sub>ive accurac<sub>y</sub> in <sub>p</sub>ercent and the chan<sub>g</sub>e from the <sub>p</sub>aired baseline in <sub>p</sub>arentheses (<sub>p</sub>ercenta<sub>g</sub>e <sub>p</sub>oints).
<table><tr><td>Setting</td><td>Ws (%)</td><td>n/N</td><td>Base</td><td>1.5</td><td>2</td><td>2.5</td></tr><tr><td>MK (basic)</td><td>55.2</td><td>81/100</td><td>98.77</td><td>100.00 (+1.23)</td><td>97.53 (-1.23)</td><td>27.16 (-71.60)</td></tr><tr><td>MK (easy)</td><td>11.5</td><td>100/100</td><td>83.00</td><td>80.00 (-3.00)</td><td>5.00 (-78.00)</td><td>0.00 (-83.00)</td></tr><tr><td>MK (medium)</td><td>8.3</td><td>100/100</td><td>61.00</td><td>61.00 (+0.00)</td><td>24.00 (-37.00)</td><td>1.00 (-60.00)</td></tr><tr><td>MK (hard)</td><td>20.8</td><td></td><td>81/10054.32</td><td>48.15 (-6.17)</td><td>1.23 (-53.09)</td><td>0.00 (-54.32)</td></tr><tr><td>MV (basic)</td><td>54.2</td><td></td><td>100/100 34.00</td><td>20.00 (-14.00)</td><td>7.00 (-27.00)</td><td>0.00 (-34.00)</td></tr><tr><td>MV (easy)</td><td>14.6</td><td></td><td>81/100 27.16</td><td>19.75 (-7.41)</td><td>0.00 (-27.16)</td><td>0.00 (-27.16)</td></tr><tr><td>MV (medium)</td><td>11.5</td><td></td><td>80/100 32.50</td><td>45.00(+12.50)</td><td>3.75 (-28.75)</td><td>0.00 (-32.50)</td></tr><tr><td>MV (hard)</td><td>12.5</td><td></td><td>100/100 24.00</td><td>23.00 (-1.00)</td><td>3.00 (-21.00)</td><td>0.00 (-24.00)</td></tr><tr><td>QA (basic)</td><td>54.2</td><td>80/100</td><td>97.50</td><td>97.50 (+0.00)</td><td>45.00 (-52.50)</td><td>0.00 (-97.50)</td></tr><tr><td>QA (medium)</td><td>56.2</td><td></td><td>80/100 67.50</td><td>66.25 (-1.25)</td><td>36.25 (-31.25)</td><td>2.50 (-65.00)</td></tr><tr><td>QA (hard)</td><td>56.2</td><td></td><td>100/10071.00</td><td>72.00 (+1.00)</td><td>35.00 (-36.00)</td><td>12.00 (-59.00)</td></tr><tr><td>Paper QA (up to 16k)</td><td>31.2</td><td>25/25</td><td>20.00</td><td>20.00 (+0.00)</td><td>20.00 (+0.00)</td><td>0.00 (-20.00)</td></tr><tr><td>Conversation topic retrieval</td><td>17.7</td><td>150/150</td><td>74.00</td><td>80.00 (+6.00)</td><td>35.33 (-38.67)</td><td>0.00 (-74.00)</td></tr><tr><td>Text sorting (1k)</td><td>10.4</td><td>25/25</td><td>0.00</td><td>0.00 (+0.00)</td><td>0.00(+0.00)</td><td>0.00 (+0.00)</td></tr><tr><td>Room-property reasoning (context size 500)</td><td>19.8</td><td>25/25</td><td>64.00</td><td>60.00 (-4.00)</td><td>44.00 (-20.00)</td><td>56.00 (-8.00)</td></tr><tr><td>Room-property reasoning (context size 3,000)</td><td>56.2</td><td>25/25</td><td>44.00</td><td>44.00(+0.00)</td><td>60.00(+16.00)</td><td>48.00(+4.00)</td></tr><tr><td>Relation chaining (context size 500)</td><td>21.9</td><td></td><td>25/25 52.00</td><td>52.00 (+0.00)</td><td>56.00(+4.00)</td><td>56.00(+4.00)</td></tr><tr><td>Relation chaining (context size 3,000)</td><td>45.8</td><td></td><td>25/25 56.00</td><td>52.00 (-4.00)</td><td>48.00 (-8.00)</td><td>48.00 (-8.00)</td></tr><tr><td>Rule deduction (context size 500)</td><td>15.6</td><td>25/25</td><td>60.00</td><td>52.00 (-8.00)</td><td>48.00 (-12.00)</td><td>48.00 (-12.00)</td></tr><tr><td>Rule deduction (context size 3,000)</td><td>42.7</td><td></td><td>25/2548.00</td><td>48.00 (+0.00)</td><td>56.00 (+8.00)</td><td>40.00 (-8.00)</td></tr><tr><td>Mixed long reasoning (8k)</td><td>47.9</td><td></td><td>25/25 76.00</td><td>56.00 (-20.00)</td><td>56.00 (-20.00)</td><td>8.00 (-68.00)</td></tr></table>

Continued on the next page.

Continued from the preceding page.
<table><tr><td>Setting</td><td>ws (%)</td><td> $n / N$  Base</td><td>1.5</td><td></td><td>2 2.5</td></tr><tr><td>Mixed long reasoning (16k)</td><td>39.6</td><td>25/2564.00</td><td>68.00 (+4.00)</td><td>48.00 (-16.00)</td><td>4.00 (-60.00)</td></tr><tr><td>Person location (8k)</td><td>54.2</td><td>25/25 96.00</td><td>92.00 (-4.00)</td><td>44.00 (-52.00)</td><td>0.00 (-96.00)</td></tr><tr><td>Adjacent-element retrieval (3k)</td><td>26.0</td><td>12/12 0.00</td><td>0.00 (+0.00)</td><td>0.00 (+0.00)</td><td>0.00 (+0.00)</td></tr><tr><td>Adjacent-element retrieval (6k)</td><td>37.5</td><td>12/12 0.00</td><td>0.00 (+0.00)</td><td>8.33 (+8.33)</td><td>0.00 (+0.00)</td></tr><tr><td>Adjacent-element retrieval (13k)</td><td>39.6</td><td>12/12 16.67</td><td>8.33 (-8.33)</td><td>8.33 (-8.33)</td><td>0.00 (-16.67)</td></tr></table>

Table 9: Llama-3.1-8B-Instruct: <sub>p</sub>ositional direction on the to<sub>p</sub>-10% heads. Entries <sub>g</sub>ive accurac<sub>y</sub> in <sub>p</sub>ercent and the chan<sub>g</sub>e from the <sub>p</sub>aired baseline in <sub>p</sub>arentheses (<sub>p</sub>ercenta<sub>g</sub>e <sub>p</sub>oints). Dashes mark settin<sub>g</sub>s without a second-sta<sub>g</sub>e evaluation; † marks a <sub>g</sub>eneration limit that difers from the saved baseline.
<table><tr><td>Setting</td><td>n/N Base</td><td>1.25</td><td>1.5</td><td>1.75</td><td>2</td></tr><tr><td>MK (basic)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MK (easy)†</td><td>100/100 83.00</td><td></td><td>79.00 (-4.00) 71.00 (-12.00)</td><td>68.00 (-15.00)</td><td>61.00 (-22.00)</td></tr><tr><td>MK (medium)†</td><td>100/10061.00</td><td>60.00 (-1.00)</td><td>57.00 (-4.00)</td><td>47.00 (-14.00)</td><td>31.00 (-30.00)</td></tr><tr><td>MK (hard)†</td><td>81/100 54.32</td><td>44.44 (-9.88)</td><td>40.74(-13.58)</td><td>23.46 (-30.86)</td><td>11.11 (-43.21)</td></tr><tr><td>MV (basic)†</td><td></td><td>100/10034.0019.00 (-15.00)</td><td>5.00 (-29.00)</td><td>1.00 (-33.00)</td><td>6.00 (-28.00)</td></tr><tr><td>MV (easy)</td><td></td><td>81/10027.1629.63(+2.47)30.86(+3.70)</td><td></td><td>12.35 (-14.81)</td><td>0.00 (-27.16)</td></tr><tr><td>MV (medium)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MV (hard)†</td><td>100/100 24.00</td><td></td><td>19.00 (-5.00) 32.00 (+8.00)</td><td>30.00 (+6.00)</td><td>18.00 (-6.00)</td></tr><tr><td>QA (basic)†</td><td>80/10097.50</td><td></td><td>90.00 (-7.50)86.25 (-11.25)</td><td>66.25 (-31.25)</td><td>13.75 (-83.75)</td></tr><tr><td>QA (medium)†</td><td></td><td>80/10067.5041.25 (-26.25)28.75 (-38.75)</td><td></td><td>38.75 (-28.75)</td><td>23.75 (-43.75)</td></tr><tr><td>QA (hard)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Paper QA (up to 16k)</td><td>25/25 </td><td>20.00 20.00 (+0.00) 20.00 (+0.00)</td><td></td><td>20.00 (+0.00)</td><td>24.00 (+4.00)</td></tr><tr><td>Conversation topic retrieval</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Text sorting (1k)</td><td>25/25 0.00</td><td>0.00(+0.00)</td><td>8.00 (+8.00)</td><td>8.00 (+8.00)</td><td>16.00 (+16.00)</td></tr><tr><td>Room-property reasoning (context size 500)</td><td>25/25</td><td>64.0052.00 (-12.00)48.00 (-16.00)</td><td></td><td>52.00 (-12.00)</td><td>48.00 (-16.00)</td></tr><tr><td>Room-property reasoning (context size 3,000)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Relation chaining (context size 500)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Relation chaining (context size 3,000)</td><td>25/2556.00</td><td>52.00 (-4.00)</td><td>52.00 (-4.00)</td><td>52.00 (-4.00)</td><td>60.00 (+4.00)</td></tr><tr><td>Rule deduction (context size</td><td>25/2560.00</td><td>52.00 (-8.00)</td><td>52.00 (-8.00)</td><td>56.00 (-4.00)</td><td>52.00 (-8.00)</td></tr><tr><td>500) Rule deduction (context size</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3,000) Mixed long reasoning (8k)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mixed long reasoning (16k)</td><td></td><td></td><td>25/25 76.0056.00 (-20.00)48.00 (-28.00)</td><td>40.00 (-36.00)</td><td>36.00 (-40.00)</td></tr><tr><td>Person location (8k)</td><td></td><td></td><td>92.00 (-4.00) 76.00 (-20.00)</td><td>84.00 (-12.00)</td><td></td></tr><tr><td>Adjacent-element retrieval (3k)</td><td>25/25 96.00</td><td>0.00 (+0.00)</td><td>8.33 (+8.33)</td><td>25.00(+25.00)</td><td>32.00 (-64.00)</td></tr><tr><td>Adjacent-element retrieval (6k)</td><td>12/12 0.00</td><td></td><td></td><td></td><td>8.33 (+8.33)</td></tr><tr><td></td><td></td><td></td><td>12/12 16.67 16.67(+0.00)16.67(+0.00)</td><td></td><td></td></tr><tr><td>Adjacent-element retrieval (13k)</td><td></td><td></td><td></td><td>0.00 (-16.67)</td><td>0.00 (-16.67)</td></tr></table>

## I. Additional Related Work

## I.1. Representation Assumptions in RoPE Theory

Existin<sub>g</sub> RoPE theories stud<sub>y</sub> both random re<sub>p</sub>resentations and fixed content vectors. Barbero et al. (2025) <sub>use</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y samp</sub>l<sub>e</sub>d G<sub>auss</sub>i<sub>an quer</sub>i<sub>es an</sub>d k<sub>eys</sub> t<sub>o s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>expec</sub>t<sub>e</sub>d <sub>scores nee</sub>d <sub>no</sub>t d<sub>ecay w</sub>ith di<sub>s</sub>t<sub>ance;</sub> th<sub>e</sub>i<sub>r paper a</sub>l<sub>so prov</sub>id<sub>es</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c cons</sub>t<sub>ruc</sub>ti<sub>ons an</sub>d <sub>a s</sub>i<sub>ng</sub>l<sub>e-</sub>f<sub>requency seman</sub>ti<sub>c-</sub>i<sub>ns</sub>t<sub>a</sub>bilit<sub>y</sub> result. Xu et al. (2024) derive base-de<sub>p</sub>endent context bounds usin<sub>g</sub> i.i.d. <sub>q</sub>uer<sub>y</sub> and ke<sub>y</sub> coordinates with <sub>a common var</sub>i<sub>ance an</sub>d <sub>a s</sub>i<sub>m</sub>il<sub>ar-</sub>k<sub>e mo</sub>d<sub>e</sub>l $k ^ { * } = q + \epsilon$ with zero-mean noise. Du et al. (2026) hold <sub>q</sub>uer<sub>y</sub> <sub>an</sub>d k<sub>ey</sub> <sub>vec</sub>t<sub>ors</sub> fi<sub>xe</sub>d <sub>an</sub>d <sub>samp</sub>l<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>pos</sub>iti<sub>ons.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>norma</sub>l <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>assumes</sub> th<sub>a</sub>t <sub>no</sub> f<sub>requency</sub> <sub>am</sub> lit<sub>u</sub>d<sub>e</sub> d<sub>om</sub>i<sub>na</sub>t<sub>es an</sub>d th<sub>e</sub>i<sub>r con</sub>t<sub>ex</sub>t<sub>-</sub>d<sub>e en</sub>d<sub>en</sub>t f<sub>a</sub>il<sub>ure es</sub>ti<sub>ma</sub>t<sub>es use a</sub>dditi<sub>ona</sub>l <sub>am</sub> lit<sub>u</sub>d<sub>e re u</sub>l<sub>ar</sub>it <sub>.</sub> F<sub>or</sub> th<sub>e</sub>i<sub>r</sub> th<sub>eore</sub>ti<sub>ca</sub>l t<sub>o</sub>k<sub>en-</sub>i<sub>nvers</sub>i<sub>on es</sub>ti<sub>ma</sub>t<sub>e,</sub> th<sub>ey compare a</sub> k<sub>ey w</sub>ith <sub>a</sub> h<sub>ypo</sub>th<sub>e</sub>ti<sub>ca</sub>l <sub>p</sub>h<sub>ase-a</sub>li<sub>gne</sub>d <sub>re</sub>f<sub>erence</sub> <sub>s</sub>h<sub>ar</sub>i<sub>ng</sub> it<sub>s</sub> f<sub>requency amp</sub>lit<sub>u</sub>d<sub>es.</sub> O<sub>ur ana</sub>l<sub>ys</sub>i<sub>s uses</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s o</sub>f <sub>rea</sub>li<sub>ze</sub>d <sub>query an</sub>d k<sub>ey ac</sub>ti<sub>va</sub>ti<sub>ons,</sub> without prescribing a distribution over these activations. The finite-window moment and adjacent-response b<sub>oun</sub>d<sub>s a</sub>ll<sub>ow nonun</sub>if<sub>orm</sub> f<sub>requency amp</sub>lit<sub>u</sub>d<sub>es.</sub> O<sub>ur</sub> di<sub>agnos</sub>ti<sub>c es</sub>ti<sub>ma</sub>t<sub>es reversa</sub>l f<sub>or samp</sub>l<sub>e</sub>d k<sub>ey pa</sub>i<sub>rs</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub>i<sub>r</sub> f<sub>u</sub>ll <sub>score-marg</sub>i<sub>n coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub> Th<sub>e reversa</sub>l <sub>guaran</sub>t<sub>ees re</sub>t<sub>a</sub>i<sub>n an exp</sub>li<sub>c</sub>it G<sub>auss</sub>i<sub>an approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>con</sub>diti<sub>on on</sub> th<sub>e score</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on over pos</sub>iti<sub>ons.</sub> Thi<sub>s</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on suppor</sub>t<sub>s</sub> di<sub>agnos</sub>ti<sub>cs compu</sub>t<sub>e</sub>d di<sub>rec</sub>tl<sub>y</sub> f<sub>rom recor</sub>d<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>ac</sub>ti<sub>va</sub>ti<sub>ons.</sub>

## I.2. RoPE Failure Mechanisms

E<sub>x</sub>i<sub>s</sub>ti<sub>ng ana</sub>l<sub>yses</sub> id<sub>en</sub>tif<sub>y</sub> f<sub>a</sub>il<sub>ures</sub> i<sub>n</sub> b<sub>o</sub>th th<sub>e pos</sub>iti<sub>ona</sub>l <sub>an</sub>d <sub>seman</sub>ti<sub>c</sub> f<sub>unc</sub>ti<sub>ons o</sub>f <sub>ro</sub>t<sub>ary a</sub>tt<sub>en</sub>ti<sub>on.</sub> B<sub>ar</sub>b<sub>ero</sub> et al. (2025) show how rotatin<sub>g</sub> semantic channels can lose their selectivit<sub>y</sub> over lon<sub>g</sub> contexts, with a formal result for a sin<sub>g</sub>le fre<sub>q</sub>uenc<sub>y</sub>, and motivate <sub>p</sub>-RoPE. Du et al. (2026) or<sub>g</sub>anize score-level failures into four types. Position inversion occurs when the same query–key pair scores higher at a distant position than at a nearby one, reversing the locality preference. Position aliasing occurs when that pair receives numerically <sub>equa</sub>l <sub>scores a</sub>t t<sub>wo</sub> dif<sub>eren</sub>t <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ances.</sub> F<sub>or</sub> t<sub>wo</sub> di<sub>s</sub>ti<sub>nc</sub>t k<sub>eys eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>a</sub>t th<sub>e same re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ance,</sub> token inversion reverses the ordering established at zero relative distance, while token aliasing makes their <sub>scores numer</sub>i<sub>ca</sub>ll<sub>y equa</sub>l<sub>.</sub> Th<sub>e a</sub>li<sub>as</sub>i<sub>ng even</sub>t<sub>s exp</sub>li<sub>c</sub>itl<sub>y accoun</sub>t f<sub>or</sub> fi<sub>n</sub>it<sub>e-prec</sub>i<sub>s</sub>i<sub>on compu</sub>t<sub>a</sub>ti<sub>on.</sub> Th<sub>ese</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons separa</sub>t<sub>e</sub> l<sub>oss o</sub>f <sub>pos</sub>iti<sub>ona</sub>l di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on</sub> f<sub>rom</sub> l<sub>oss o</sub>f t<sub>o</sub>k<sub>en</sub> di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on, an</sub>d di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h <sub>a</sub> <sub>reverse</sub>d <sub>pre</sub>f<sub>erence</sub> f<sub>rom a</sub> ti<sub>e.</sub>

In vision–lan<sub>g</sub>ua<sub>g</sub>e models, Qi et al. (2026) <sub>p</sub>robe <sub>p</sub>ositional sensitivit<sub>y</sub> b<sub>y</sub> addin<sub>g</sub> a one-ste<sub>p</sub> RoPE rotation t<sub>o</sub> <sub>v</sub>i<sub>sua</sub>l k<sub>eys</sub> <sub>w</sub>hil<sub>e</sub> h<sub>o</sub>ldi<sub>ng</sub> th<sub>e</sub>i<sub>r</sub> <sub>ex</sub>t<sub>rac</sub>t<sub>e</sub>d <sub>con</sub>t<sub>en</sub>t <sub>vec</sub>t<sub>ors</sub> <sub>an</sub>d th<sub>e</sub> <sub>query</sub> fi<sub>xe</sub>d<sub>.</sub> Th<sub>ey</sub> <sub>repor</sub>t <sub>c</sub>h<sub>anges</sub> i<sub>n</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on we</sub>i<sub>g</sub>ht<sub>s an</sub>d i<sub>n group-average</sub>d <sub>score sens</sub>iti<sub>v</sub>it<sub>y.</sub> Thi<sub>s s</sub>t<sub>u</sub>d<sub>y</sub> hi<sub>g</sub>hli<sub>g</sub>ht<sub>s</sub> l<sub>oca</sub>l <sub>pos</sub>iti<sub>ona</sub>l <sub>reso</sub>l<sub>u</sub>ti<sub>on as</sub> a <sub>p</sub>ractical concern. In §2, our semantic reversal follows the token-inversion event of Du et al. (2026). We define positional insensitivity through the absolute score gap between adjacent positions, normalized by <sub>eac</sub>h <sub>query–</sub>k<sub>ey pa</sub>i<sub>r</sub>’<sub>s own coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>norm.</sub> Thi<sub>s cr</sub>it<sub>er</sub>i<sub>on measures</sub> l<sub>oca</sub>l <sub>response s</sub>t<sub>reng</sub>th i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y o</sub>f it<sub>s</sub> di<sub>rec</sub>ti<sub>on; a pre</sub>f<sub>erence</sub> f<sub>or nearer pos</sub>iti<sub>ons an</sub>d <sub>equa</sub>l <sub>scores a</sub>t <sub>separa</sub>t<sub>e</sub>d <sub>pos</sub>iti<sub>ons are</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>proper</sub>ti<sub>es.</sub>

## I.3. Context Extension

Oth<sub>er wor</sub>k <sub>exp</sub>l<sub>a</sub>i<sub>ns</sub> th<sub>e mec</sub>h<sub>an</sub>i<sub>sms</sub> b<sub>e</sub>hi<sub>n</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>-ex</sub>t<sub>ens</sub>i<sub>on</sub> f<sub>a</sub>il<sub>ures.</sub> H<sub>o</sub>PE <sub>connec</sub>t<sub>s ex</sub>t<sub>rapo</sub>l<sub>a</sub>ti<sub>on</sub> f<sub>a</sub>il<sub>ures</sub> to learned fre<sub>q</sub>uenc<sub>y</sub> com<sub>p</sub>onents whose attention-score <sub>p</sub>atterns chan<sub>g</sub>e be<sub>y</sub>ond the trainin<sub>g</sub> window (Chen et al., 2025). Resonance RoPE identifies a feature <sub>g</sub>a<sub>p</sub> at unseen inte<sub>g</sub>er <sub>p</sub>ositions, includin<sub>g</sub> in fre<sub>q</sub>uenc<sub>y</sub> com<sub>p</sub>onents whose value ran<sub>g</sub>es are covered durin<sub>g</sub> trainin<sub>g</sub> (Wan<sub>g</sub> et al., 2024b). Fra<sub>y</sub>ed RoPE relates l<sub>ong-con</sub>t<sub>ex</sub>t d<sub>egra</sub>d<sub>a</sub>ti<sub>on</sub> t<sub>o</sub> di<sub>spers</sub>i<sub>on an</sub>d <sub>over</sub>l<sub>ap o</sub>f <sub>query–</sub>k<sub>ey c</sub>l<sub>us</sub>t<sub>ers, w</sub>hi<sub>c</sub>h <sub>wea</sub>k<sub>en</sub> th<sub>e a</sub>bilit<sub>y o</sub>f attention-sink tokens to absorb attention (Wertheimer et al., 2026). These accounts connect fre<sub>q</sub>uenc<sub>y</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or an</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>on geome</sub>t<sub>ry</sub> t<sub>o spec</sub>ifi<sub>c</sub> f<sub>a</sub>il<sub>ures, comp</sub>l<sub>emen</sub>ti<sub>ng</sub> th<sub>e even</sub>t<sub>-</sub>b<sub>ase</sub>d <sub>c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on</sub> i<sub>n</sub> A<sub>pp</sub>endix I.2.

Position Interpolation and YaRN adjust RoPE frequencies to extend context (Chen et al., 2023b; Peng et al., 2024). For <sub>p</sub>osition inter<sub>p</sub>olation, Wu et al. (2026) show that its <sub>p</sub>ractical efect de<sub>p</sub>ends on the task: f<sub>requency sca</sub>li<sub>ng can</sub> h<sub>e</sub>l<sub>p</sub> d<sub>epen</sub>d<sub>enc</sub>i<sub>es</sub> th<sub>a</sub>t <sub>s</sub>t<sub>re</sub>t<sub>c</sub>h <sub>w</sub>ith i<sub>npu</sub>t l<sub>eng</sub>th <sub>w</sub>hil<sub>e</sub> i<sub>mpa</sub>i<sub>r</sub>i<sub>ng re</sub>t<sub>r</sub>i<sub>eva</sub>l <sub>a</sub>t fi<sub>xe</sub>d <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>ances an</sub>d <sub>prec</sub>i<sub>se</sub> l<sub>oca</sub>l <sub>a</sub>li<sub>gnmen</sub>t<sub>.</sub> Y<sub>a</sub>RN <sub>a</sub>l<sub>so</sub> id<sub>en</sub>tifi<sub>es</sub> th<sub>e</sub> difi<sub>cu</sub>lt<sub>y o</sub>f di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>hi<sub>ng near</sub>b<sub>y</sub> <sub>p</sub>ositions after uniform fre<sub>q</sub>uenc<sub>y</sub> scalin<sub>g</sub> (Pen<sub>g</sub> et al., 2024).

## I.4. Internal Representations and Failure Diagnosis

Don<sub>g</sub> et al. (2024) decom<sub>p</sub>ose hidden states usin<sub>g p</sub>osition-wise means across se<sub>q</sub>uences, connect <sub>p</sub>ositional <sub>vec</sub>t<sub>or ex</sub>t<sub>rapo</sub>l<sub>a</sub>ti<sub>on</sub> t<sub>o</sub> f<sub>a</sub>il<sub>ures, an</sub>d d<sub>eve</sub>l<sub>op</sub> i<sub>n</sub>t<sub>erven</sub>ti<sub>ons</sub> f<sub>or</sub> N<sub>o</sub>PE <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> Oth<sub>er wor</sub>k di<sub>agnoses pos</sub>iti<sub>ona</sub>l attention bias (Hsieh et al., 2024b), retrieval-head behavior (Wu et al., 2025a), and context-extension mechanisms throu<sub>g</sub>h head ablations and activation <sub>p</sub>atchin<sub>g</sub> (Zhao et al., 2025). Streamin<sub>g</sub>LLM links rollin<sub>g</sub>-cache de<sub>g</sub>radation to attention sinks (Xiao et al., 2024), and contrastive attribution traces out<sub>p</sub>ut errors throu<sub>g</sub>h internal states (Tan et al., 2026). Our local score measurements com<sub>p</sub>lement these anal<sub>y</sub>ses; <sub>re</sub>l<sub>a</sub>ti<sub>ng a measure</sub>d <sub>vu</sub>l<sub>nera</sub>bilit<sub>y</sub> t<sub>o an ou</sub>t<sub>pu</sub>t <sub>error requ</sub>i<sub>res ev</sub>id<sub>ence a</sub>b<sub>ou</sub>t it<sub>s</sub> d<sub>owns</sub>t<sub>ream e</sub>f<sub>ec</sub>t<sub>.</sub>

## I.5. Behavioral Evaluation and Attention Competition

Lon<sub>g</sub>Bench, RULER, and HELMET measure <sub>p</sub>erformance across lon<sub>g</sub>-context tasks (Bai et al., 2024; Hsieh et al., 2024a; Yen et al., 2025). Lost in the Middle and NoLiMa ex<sub>p</sub>ose sensitivit<sub>y</sub> to information <sub>p</sub>lacement and retrieval be<sub>y</sub>ond literal matchin<sub>g</sub> (Liu et al., 2024a; Modarressi et al., 2025). Lon<sub>g</sub>er in<sub>p</sub>uts motivate anal<sub>y</sub>ses of distractor dilution (Bansal et al., 2026) and critical attention scalin<sub>g</sub> under sim<sub>p</sub>lified token <sub>g</sub>eometries (Chen et al., 2026). Parallel context encodin<sub>g</sub> has also been linked to elevated attention entro<sub>py</sub> (Zhan<sub>g</sub> et al., 2025). Our <sub>p</sub>re-softmax dia<sub>g</sub>nostics add two internal measurements alon<sub>g</sub>side task <sub>per</sub>f<sub>ormance;</sub> <sub>so</sub>ft<sub>max</sub> <sub>compe</sub>titi<sub>on</sub> <sub>an</sub>d <sub>su</sub>b<sub>sequen</sub>t l<sub>ayers</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> h<sub>ow</sub> th<sub>ese</sub> <sub>proper</sub>ti<sub>es</sub> i<sub>n</sub>fl<sub>uence</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>pre</sub>di<sub>c</sub>ti<sub>on.</sub>

## I.6. Long-Context Failure Modes

Existin<sub>g</sub> benchmarks and studies (Hsieh et al., 2024a; Bai et al., 2024; Yen et al., 2025; Kuratov et al., 2024; Modarressi et al., 2025; Goldman et al., 2024; Vodrahalli et al., 2024; Lev<sub>y</sub> et al., 2024) identif<sub>y</sub> lon<sub>g</sub>-context failures related to retrieval and reasonin<sub>g</sub>, and that these failure modes can be cou<sub>p</sub>led (Du et al., 2025). O<sub>ur</sub> di<sub>agnos</sub>ti<sub>cs</sub> f<sub>ea</sub>t<sub>ure</sub> t<sub>wo</sub> i<sub>n</sub>t<sub>erna</sub>l <sub>measuremen</sub>t<sub>s an</sub>d <sub>pos</sub>iti<sub>on</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>on a pre</sub>f<sub>erence spec</sub>t<sub>rum w</sub>ith <sub>prese</sub>t t<sub>as</sub>k<sub>s anc</sub>h<sub>ore</sub>d <sub>as com</sub>bi<sub>na</sub>ti<sub>ons o</sub>f b<sub>o</sub>th <sub>mo</sub>d<sub>es.</sub>