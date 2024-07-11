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
    
    consuming func toGraph(at graph: MPSGraph) async throws(Errors) -> (dsl: Graph, data: MPSGraphTensorData) {
        let data = try await self.base.toTensorData()
        let ts = graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        return (Graph(tensor: ts, graph: graph), data)
    }
}
