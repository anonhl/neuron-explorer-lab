// These labels describe the standalone demo; the live application keeps its real-model descriptions.
Object.assign(en, {
  'lab.intro': 'Static demo: practice residual interventions and independent text comparisons. All outputs, probabilities and internal changes below are simulated in your browser; no model or server is called.',
  'lab.inputHint': 'Add up to 8 texts in Run. Each keeps its own token positions. Click a text label to show its simulated neuron analysis and 3D view.',
  'lab.runSnapshot': 'Settings used for these simulated results',
  'lab.measurement': 'Example measurement at token #{pos}; {count} simulated input tokens. Probabilities and activations are illustrative.',
  'lab.downstreamHint': 'Illustrative neuron changes and residual norms from a toy vector calculation. Candidate scores cover the entire simulated continuation. These are not measurements from Qwen.',
});
Object.assign(zh, {
  'lab.intro': '静态模拟演示：练习残差流干预和多文本对照。下方的回复、概率及内部变化均由浏览器生成，不加载模型、不调用推理服务。',
  'lab.inputHint': '在左侧「运行」添加最多 8 条独立文本，各自选择 token。点击文本标签可切换对应的模拟神经元分析与三维视图。',
  'lab.runSnapshot': '这组模拟结果使用的设置',
  'lab.measurement': '示例测量位置：token #{pos}；输入共 {count} 个模拟 token。概率与激活仅用于演示。',
  'lab.downstreamHint': '这里是简化向量计算生成的示例神经元变化与残差长度。候选分数覆盖完整模拟续写，不是 Qwen 的真实测量结果。',
});
