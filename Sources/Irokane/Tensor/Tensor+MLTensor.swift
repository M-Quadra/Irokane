//
//  Tensor+MLTensor.swift
//
//
//  Created by m_quadra on 2024/6/25.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 18.0, *)
extension MLTensor: Tensorable {
    
    func toTensor() -> Tensor {
        return Tensor(base: self)
    }
    
    func toMPS(graph: MPSGraph) async throws(Errors) -> (tensor: MPSGraphTensor, data: MPSGraphTensorData) {
        let data = try await self.toTensorData()
        let ts = graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        return (ts, data)
    }
}
