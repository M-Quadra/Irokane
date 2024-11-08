# Graph

> 如果行，我想代码生成，可惜菜。

行为基于 torch，声明略有差异。建议最低兼容 iOS17，platforms 声明仅为 SwiftPM 正常导入。



## 已知问题



### 指针传递

Graph.Tensor.Sub 中的 UnsafeMutablePointer<Graph.Tensor> 在 Release 模式下可能出现内部 graph 与 tensor 变量互换，导致运行时方法调用异常，出现崩溃。init 方法末填充任意不被剔除的代码可缓解。

原因不明，不知是指针传递有误还是编译器bug。



### Noncopyable

subscript 不兼容 ownership，只能妥协。完全重构全 func 形式太丑。

因此 Graph.Tensor 不使用 Noncopyable。

