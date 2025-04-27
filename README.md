# Irokane

MPSGraph DSL，辅助建图。



移植 torch 模型至 CoreML 后，遭遇各类 CPU-only op 与推理加速降级。

一气之下想强上 GPU，瞅了眼 MPSGraph，不顺手。

好累，不想爱了。

> 不行，云老婆就是我所有的爱。

手动拆模型，实现是重写 CPU-only 部分积攒的。精神不佳，太监可能。

就结果而言，大体与我的人生一样失败。

开源献丑。



## 导入说明

仅支持 SwiftPM，建议最低兼容 iOS17。



## 快速开始

- 输入输出

```swift
import Irokane

func forward(x: MLMultiArray) throws -> MLMultiArray {
    let graph = Irokane.Graph()
    
    let x = try x.ik.to(graph: graph)
    let y = x + 1
    
    return try y.tensorData().ik.toMLMultiArray()
}
```



- Functional 支持

```swift
import Irokane
fileprivate typealias F = Irokane.Functional

let y = F.pad(x, pad: (3, 4))
let y = F.pad(x, pad: [(0, 0), (1, 0), (0, 0)])
let y = F.softmax(x, dim: -1)
```

- 切片操作

```swift
// x[..., None] >= y
x[..., .none] >= y

// x[:, :-1]
let y = x[.all, ..<(-1)]
```

- 原地操作

```swift
x[mask] .= y
x[..., 1] .= 2
```

由于操作符重载限制, `=` 赋值使用 `.=` 定义。


## 其他文档

随缘施工。

[Irokane.Graph](./Doc/Graph.md)

[CoreML 踩坑记录](./Doc/CoreML.md)

[MPSGraph 踩坑记录](./Doc/MPSGraph.md)

