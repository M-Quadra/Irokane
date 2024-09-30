//
//  Wrapper+MLTensor.swift
//  
//
//  Created by m_quadra on 2024/6/26.
//

import CoreML
import MetalPerformanceShadersGraph

#if !SWIFT_PACKAGE || swift(>=6.1)
@available(iOS 18.0, *)
public extension MLTensor {
    var ik: Wrapper<MLTensor> { Wrapper(base: self) }
}
#else
#warning("理解不能, 疑似编译器BUG, 从 beta3 开始报错, 需要在库外重新定义一遍, 不知6.1是否会正常")
#endif

@available(iOS 18.0, *)
public extension Wrapper<MLTensor> {
    
    consuming func toTensor() -> Tensor {
        return Tensor(base: self.base)
    }
    
    consuming func toTensor(at graph: Graph) async throws(Errors) -> Graph.Tensor {
        let data = try await self.base.toTensorData()
        let x = graph.graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        
        graph.feeds[x] = consume data
        return Graph.Tensor(graph: graph, tensor: consume x)
    }
}
