# Graph

> 如果行，我想代码生成，可惜菜。

行为基于 torch，声明略有差异。建议最低兼容 iOS17，platforms 声明仅为 SwiftPM 正常导入。



MLTensor 看起来像 BNNS 封装，op支持有限，放弃基于其实现。

MPSGraph op 支持更好，适合偷懒。

曾考虑 BNNS + MPSGraph 混合实现，但需要排查具体 op 瓶颈，以及 BNNS 实在难用。

水平有限，力不从心。



## 使用指南

使用场景狭隘，GPU 友好型 op 才能产生收益，以实际运行为准。

个人折腾使用，覆盖有限。

使用参考单元测试。



## 已知问题



### 指针传递

Graph.Tensor.Sub 中的 UnsafeMutablePointer<Graph.Tensor> 在 Release 模式下可能出现内部 graph 与 tensor 变量互换，导致运行时方法调用异常，出现崩溃。init 方法末填充任意不被剔除的代码可缓解。

原因不明，不知是指针传递有误还是编译器bug。



### Noncopyable

subscript 不兼容 ownership，只能妥协。完全重构全 func 形式太丑。

因此 Graph.Tensor 不使用 Noncopyable。



### GPU受限

All in GPU 不是最佳实践，跑不过 CPU-only 也并非不可能。

考虑到此脚手架在个人尝试中只能用于缓解部分因降级 CPU 的推理速度问题，不适用大面积移植，实际加速比有限。因此放弃曾经考虑的 op 运行统计。
