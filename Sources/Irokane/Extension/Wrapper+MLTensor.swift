//
//  Wrapper+MLTensor.swift
//  
//
//  Created by m_quadra on 2024/6/26.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 18.0, *)
public extension MLTensor {
    var ik: Wrapper<MLTensor> { Wrapper(base: self) }
}

@available(iOS 18.0, *)
public extension Wrapper where Base == MLTensor {
    
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
