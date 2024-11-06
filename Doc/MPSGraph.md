# MPSGraph

MPSGraph也有奇怪行为，疑似计算图断言/优化不到位后遗症。



## scatterNDWithData

mpsGraph.scatterNDWithData 的输出 tensor 需要在 targetTensors 的路径上，否则取中途结果会触发奇怪的断言失败。即使使用 mpsGraph.compile 也不会对实效节点进行提出。[见此](../Tests/IssueTests/Issue+Scatter.swift)

mpsGraph.scatterNDWithData 在模拟器(x86 or arm)支持空tensor更新，但在 iPhone 运行时不支持。运行测试截止 iOS 18.1。[见此](../Tests/RealDeviceTests/RealDeviceTests.swift)



## gatherND

mpsGraph.gatherND 输出也存在上述问题，[见此](../Tests/IssueTests/Issue+Gather.swift)。

