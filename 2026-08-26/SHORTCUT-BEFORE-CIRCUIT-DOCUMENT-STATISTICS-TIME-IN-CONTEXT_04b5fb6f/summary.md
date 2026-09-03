---
title: "SHORTCUT-BEFORE-CIRCUIT-DOCUMENT-STATISTICS-TIME-IN-CONTEXT"
source: https://arxiv.org/pdf/2608.24460v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:55:31"
---

# 论文速读：SHORTCUT BEFORE CIRCUIT: DOCUMENT STATISTICS TIME IN-CONTEXT CONFLICT RESOLUTION

## 一句话总结
本文构造了一个合成语言任务，使“近期性”与“稀有性”两条规则在训练分布上完全共-extensive，证明当目标函数对两条规则完全 indifferent 时，行为准确率无法区分机制；可复现的仅是模型逃离位置捷径的时机（单调于旧值冗余度），而具体采用哪条规则取决于优化随机性，而非数据统计。

## 研究问题与动机
- **核心问题**：文档常包含冲突事实（旧值被新值覆盖），模型需依赖提示（近期性、重复次数/稀有性、位置等）作答；但自然语料中这些提示高度一致，仅凭准确率无法识别模型实际采用的规则。
- **现有方法不足一**：知识冲突行为研究只能测量部署后模型的偏好倾向，无法归因到数据侧（自然文档不分离候选提示，偏好与相关性混杂）。
- **现有方法不足二**：机制解释工作能定位携带冲突信息的组件，但电路多解性（predictive multiplicity）使得单一 seed 或分析 pipeline 的结论难以稳定复现。
- **动机**：需要一种精确可检验的构造，显式分离共-extensive 规则，给出“何时机制归因可用、何时不可用”的判别准则，而非强行输出唯一解释。

## 核心贡献（创新点）
1. **共-extensive 合成语言构造**：设计生成器使 RECENCY 与 RARITY 在任何训练文档上严格等价（Equation 1），目标函数在该方向上完全 indifferent。与已有工作本质区别：不同于自然语料中提示仅统计相关，此处构造使两条规则在文档层面零分歧，彻底封死观测区分路径。
2. **最小因果干预读读出（Multiplicity Inversion Edit）**：仅翻转槽位内值的频数分布，保持真值、token 总数、答案位置、语句数不变，使 RARITY 预测翻转而 RECENCY 不变。与已有干预方法的本质区别：传统 interchange intervention 改变内部变量，本文在输入侧施加保持任务结构不变的极小扰动，精确隔离单一规则边际效应。
3. **机制归因边界定理的实证刻画**：证明行为饱和（acc ≥ 0.999）下跨 seed 读读出方差可达 0.879（远超 binomial SE 0.025），而逃离时机严格单调于 $R_{old}$。与已有工作的本质区别：既往研究关注“机制是否存在”，本文证明“数据固定的是时机而非身份”，将 interpretability 的焦点从定位转向可归因性的先决条件检验。
4. **双门控评估协议**：结合 copy diagnostic 与准确率区分“位置捷径饱和”与“真实检索电路”，并揭示仅靠电路门控仍不足以保证单次读读出稳定漂移。与已有 phase diagram 研究的本质区别：不因阈值制造虚假相变，用未阈值化的 loss 导数峰值定位转变，并显式报告单 run 内的不稳定性。

## 方法详解
- **语言与任务设计**：文档为 `(entity, attribute, value)` 赋值语句序列后接查询。每个文档含一个查询槽位：旧值 $v_{old}$ 重复 $R_{old}$ 次，新值 $v_{new}$ 仅出现 1 次，查询紧随 $v_{new}$ 后 $\Delta D$ 条语句。标识符来自不相交词表（200/8/512），绑定每文档重采样，流式无限，任务不可从权重求解。
- **共-extensive Alias**：由于 $v_{new}$ 出现 1 次、$v_{old}$ 出现 $R_{old} \geq 3$ 次，对任意文档 $d$ 有 $\arg\max_i \text{pos}(s_i) = \arg\min_v |\{i: \text{val}(s_i)=v\}|$，两条规则在训练集上永远输出相同值，目标函数 indifferent。
- **最小干预编辑（§4.1）**：定义对比 log-odds $m(\bar{x}; v^\star) = \log p(v^\star|x) - \log p(v_{truth}|x)$，读读出 $\Delta = m(x_{edit}; v^\star) - m(x_{base}; v^\star)$。编辑将 $v_{
