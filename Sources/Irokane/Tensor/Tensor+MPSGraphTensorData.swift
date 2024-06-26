//
//  Tensor+MPSGraphTensorData.swift
//
//
//  Created by m_quadra on 2024/6/25.
//

import MetalPerformanceShadersGraph

extension MPSGraphTensorData: Tensorable {
    
    consuming func toTensor() -> Tensor {
        return Tensor(base: self)
    }
    
    func toMPS(graph: consuming MPSGraph) async throws -> (tensor: MPSGraphTensor, data: MPSGraphTensorData) {
        let ts = graph.placeholder(shape: self.shape, dataType: self.dataType, name: nil)
        return (ts, self)
    }
}
