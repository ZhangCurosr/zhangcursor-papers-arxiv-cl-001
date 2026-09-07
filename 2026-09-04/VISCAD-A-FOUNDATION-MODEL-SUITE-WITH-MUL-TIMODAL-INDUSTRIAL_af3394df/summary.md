---
title: "VISCAD-A-FOUNDATION-MODEL-SUITE-WITH-MUL-TIMODAL-INDUSTRIAL"
source: https://arxiv.org/pdf/2609.03811v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 23:14:04"
field: "工业CAD智能"
keywords: ["CAD生成", "多模态大模型", "工业AI", "程序生成", "装配设计", "test-time scaling"]
innovations: ["VisCAD-M1: 27B多模态模型统一处理文本/图纸/照片/渲染图生成可执行CAD程序，超越前沿通用模型", "VisCAD-H1: CAD原生装配Harness与DSL无关IR架构，实现跨FreeCAD/Fusion/SolidWorks迁移", "parallel-tts: 模型自身作为重排序器的测试时缩放方法，带来约5%相对提升"]
benchmarks: ["PubCADBench", "RealCADBench"]
---

# 论文速读：VISCAD-A-FOUNDATION-MODEL-SUITE-WITH-MUL-TIMODAL-INDUSTRIAL

## 一句话总结
VisCAD是一个面向工业产品的多模态CAD智能基础模型套件，核心是27B参数模型VisCAD-M1，可将文本、工程图纸、真实产品照片和渲染图统一映射为可执行的FreeCAD Python程序；同时配套VisCAD-H1装配级设计Harness，在PubCADBench和RealCADBench上实现了超越前沿通用模型的零件级生成性能。

## 研究问题与动机
1. 现有专用CAD模型仅支持单一输入模态（如仅文本或仅渲染图），对2D工程图纸和真实工业产品照片覆盖严重不足，泛化能力差。
2. 通用前沿模型（Gemini-3.1-Pro、GPT-5.5等）在多模态覆盖上更广，但在工业CAD特定域上性能不稳定，难以稳定生成高质量可执行CAD程序。
3. 现有数据集（ABC、DeepCAD、Fusion 360 Gallery等）多为CAD原生合成数据，缺乏异构真实工业输入，导致模型在真实场景中表现不佳。
4. 装配级生成并非零件级生成的简单扩展：API错误会级联传播、配合关系缺失不同于单个零件几何误差，通用代码助手工作流缺乏结构化BOM、零件IR门控和全局配合审查等装配必需能力。

## 核心贡献（创新点）
1. **VisCAD-M1（27B多模态CAD生成模型）**：首次用统一接口支持文本、工程图纸、真实产品照片、渲染图四种模态映射到可执行FreeCAD程序，在PubCADBench/RealCADBench平均分0.5540超越最强前沿模型Gemini-3.1-Pro（0.5496）。
2. **VisCAD-H1（CAD原生装配Harness）**：提出结构化四阶段装配工作流（BoM生成→并行零件IR→全局配合审查→CAD实现），引入DSL无关中间表示（IR），可在FreeCAD、Fusion、SolidWorks之间无缝迁移；50实例装配评测得85.0分，显著高于Codex（68.0）和Claude Code（52.0）。
3. **parallel-tts测试时缩放方法**：将VisCAD-M1自身用作重排序器，从k个rollout中选择最优程序，k=16时PubCADBench平均分提升至0.5833，相对SOTA提升约5%。
4. **综合数据策展与多阶段训练管线**：构建从公开CAD程序、工厂自动化产品目录、采购工业资产中抽取数据的全流程，包含逆向意图生成、基于RSI的程序搜索、多模态数据增强和递归自我改进。
5. **DSL无关中间表示（IR）架构**：将装配意图表达为平台无关的结构化IR（BoM、零件局部几何、接口约束、配合关系、全局定位），解耦装配推理与后端语法实现。

## 方法详解
**数据策展管线**：整合三类原始数据源（公开CAD程序库、工厂自动化产品目录图像、采购的STEP/STL模型与工程图纸），经四步类别过滤器保留刚性机械产品（21大类、128子类）。采用正向路径（意图i→程序p→形状s）和反向路径（形状/程序s/p→多样化意图i）双向构建(intent, program, shape)三元组；利用图像编辑模型降维输入复杂度，图像到3D模型生成补全缺失形状，基于递归自我改进（RSI）的程序搜索获取高质量监督信号。

**多阶段训练管线**：①中期训练（Mid-training）：约100万个CadQuery程序被执行导出solid，从多角度投影生成多视图图像作为意图，建立可逆的金标准意图-程序对，教授CAD语法与API映射；②鲁棒SFT（Robust SFT）：约15万张产品图像经清洗标注，教师模型蒸馏出同一意图的多个候选程序，增强对工业多样性和噪声的鲁棒性；③持续高质量SFT（Continual High-Quality SFT）：约1万个精心筛选样本，通过生成-验证-反馈循环恢复细粒度几何保真度。

**VisCAD-H1装配工作流**：四阶段门控流程——①多模态理解与BOM：将产品图像、多视图图纸和文本转化为零件清单、装配顺序及配合提示；②并行零件IR：为每个零件分配平台无关的中间表示（局部几何、接口、约束），在转化为任何具体DSL前完成语法和局部有效性门控；③全局IR审查：读取全部零件IR，检查跨零件几何、接口和定位，输出带位姿的已审查IR；④CAD实现：将审查后的IR翻译为后端建模调用，并行构建零件并装配。

**测试时缩放（parallel-tts）**：对同一输入意图生成k个rollout（程序+导出3D模型的6视图），用VisCAD-M1作为listwise reranker对所有rollout打分并选择最优；实验发现串行细化（sequential-tts）无效，但并行重排序有效，k从8增至16时性能持续提升。

**评估指标**：零件级Profile Average = (执行率 + 实体IoU + 表面IoU + Judge分数/100) / 4；装配级Judge从组件几何质量Q、装配精度F、系统设计D三个维度综合评分（0-100）。

## 实验与结果
**数据集**：PubCADBench（1,100任务，含BenchCAD 200、CADBench 300、正交重建200、P3D-Text 200、P3D-Image 200）；RealCADBench（1,745任务，含Text 568、2D Drawing 236、Real Picture 568、Rendered Image 373）。

**基线模型**：Gemini-3.1-Pro、Claude-Opus-4.8、GPT-5.4、GPT-5.5、Kimi-K3、Doubao-Seed-2.0-pro、Qwen3-VL-8B/32B、Qwen3.8-27B。

**零件级结果**：VisCAD-M1平均Profile Average 0.5540，超越Gemini-3.1-Pro（0.5496）和GPT-5.5（0.5450）；PubCADBench上0.5596 vs 0.5533，RealCADBench上0.5485 vs 0.5459。open-weight基线远低于此水平（Qwen3-VL-32B仅0.3199，Qwen3.8-27B为0.4759）。

**最强结果与提升幅度**：VisCAD-M1 + parallel-tts（k=16）达到0.5833（PubCADBench）和0.5761（RealCADBench），相对此前SOTA提升约5%。在RealCADBench真实产品照片切片上实现最高执行率（0.9877）和实体IoU（0.4456）；在渲染图切片上实体IoU达0.5858，为所有模型最高。

**装配级结果**：50实例研究，VisCAD-H1装配Judge得分为85.0，显著高于Codex（68.0）和Claude Code（52.0），有效样本率均超过98%。

**消融分析**：中期训练在各PubCADBench切片上使IoU指标提升约1个百分点、Judge提升约5个百分点；鲁棒SFT中每输入3个候选程序优于1个；parallel-tts k=8时提升1.3-1.7分，k=16时进一步提升至0.5833。

## 相关工作脉络
1. **DeepCAD/Text2CAD/CAD-Recode**：早期程序原生CAD生成工作，仅支持单一模态输入且限于单零件，VisCAD-M1扩展至四种模态并面向工业真实场景。
2. **BRepNet/BrepGen/CAD-Llama/ParaCAD-RL**：直接生成B-Rep或点云逆向CAD，未聚焦可执行参数化程序的生成，与VisCAD的DSL程序生成路线不同。
3. **OpenECAD/CadVLM/ChatCAD/PICASSO**：扩展了输入模态（草图、工程图），但输出仍限于单零件且装配级能力缺失，VisCAD通过H1实现了从零件到装配的完整链路。
4. **ABC/Fusion 360 Gallery**：CAD原生合成数据集，VisCAD引入RealCADBench（真实工业照片、工程图纸、文本），填补了现实工业场景评测空白。
5. **BenchCAD/CADBench/P3D-Bench**：公开基准的聚合构成了PubCADBench，VisCAD在此复合基准上全面超越各独立基准上的最强模型。
6. **通用编码Harness（Codex/Claude Code）**：VisCAD-H1对比表明，未经领域定制的一般代码助手在复杂装配生成中难以处理结构化BoM和配合关系，Domain-specific harness优势显著。

## 局限性与未来方向
1. 2D工程图纸模态上Gemini-3.1-Pro的Judge分数仍高于VisCAD-M1（79.08 vs 69.13），说明视觉-几何对齐仍有提升空间。
2. 部分切片上执行率已趋近饱和（>0.95），而几何IoU和Judge分数提升较缓慢，细粒度几何保真度仍需改进。
3. 训练数据虽扩展至工业真实图像，但样本量（15万张鲁棒SFT、1万高质量SFT）仍远小于中期训练的100万程序，数据规模不均衡。
4. 目前零件级（M1）与装配级（H1）尚未完全融合为统一模型-Agent闭环，论文展望未来将两者整合。
5. 评估聚焦于机械类刚性产品（21大类128子类），对软体、流体、非刚性场景的泛化未涉及。

## 研究启发与可借鉴点
1. **多阶段训练策略可迁移**：从大规模合成金标准数据（mid-training）→多样化真实数据鲁棒SFT→高质量数据精细SFT的渐进式训练范式，适用于其他垂直领域大模型的训练。
2. **DSL无关中间表示（IR）架构**：将领域推理与后端实现解耦的设计，可推广至其他需要适配多种工具/语言的AI Agent系统。
3. **Test-time Scaling（TTS）中的重排序技巧**：模型自身作为判别器而非生成器使用，以低成本获得显著性能提升，对代码生成、数学推理等序列生成任务具有借鉴价值。
4. **多模态评估指标融合**：结合几何度量（IoU）与基于VLM的Judge评分（免GT），为3D生成任务提供了兼顾客观与主观的完整评估体系。
5. **数据策展中的RSI自改进循环**：利用图像到3D模型生成补全缺失数据、基于程序搜索获取高精度监督信号，为数据稀缺场景下的模型训练提供了可行思路。

## 关键术语表
**VisCAD-M1**：27B参数多模态基础模型，将文本、工程图纸、真实产品照片、渲染图映射为可执行FreeCAD Python程序。
**VisCAD-H1**：CAD原生装配级设计Harness，通过结构化四阶段工作流（BoM→零件IR→配合审查→实现）完成多零件装配生成。
**parallel-tts（Parallel Test-Time Scaling）**：将训练好的模型自身用作重排序器，从k个rollout中选择最优输出的测试时优化方法。
**DSL-agnostic IR（领域特定语言无关中间表示）**：与具体CAD后端解耦的结构化中间表示，包含BOM、零件几何、接口约束、配合关系和全局定位，可翻译至FreeCAD/Fusion/SolidWorks等多种后端。
**RealCADBench**：面向真实工业场景的CAD评测基准，包含文本、2D工程图、真实产品照片、渲染图四类切片，共1,745个任务。
**Profile Average**：零件级综合评估指标，等于（执行率 + 实体IoU + 表面IoU + Judge分数/100）的四分之一均值。
**Solid IoU / Surface IoU**：经PCA对齐后，预测模型与参考模型在体素网格（分辨率96）上的体积重叠率和表面积重叠率。
**Assembly Judge**：装配级评估分数（0-100），综合评分组件几何质量Q、装配精度F和系统设计D三个维度。

## 可复现要素
- **数据集**：PubCADBench（公开基准聚合）和RealCADBench（论文声明使用）；数据策展流程详细描述了来源但具体数据获取方式需自行实现。
- **代码/权重**：论文未明确声明开源状态，需进一步确认。
- **关键超参**：模型规模27B；中期训练~1M程序；鲁棒SFT ~150K产品图像；持续SFT ~10K高质量样本；parallel-tts中k=8和k=16。
- **评估环境**：FreeCAD API作为目标CAD DSL；Kimi-K2.6作为Judge模型；实体/表面IoU体素分辨率R=96，2% padding。
