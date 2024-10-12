//
//  Tensor+MLMultiArray.swift
//
//
//  Created by m_quadra on 2024/6/25.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 15.4, *)
extension MLMultiArray: Tensorable {
    
    func toTensor() -> Tensor {
        return Tensor(base: self)
    }
    
    public consuming func toMPS(
        graph: consuming MPSGraph
    ) async throws(Errors) -> (tensor: MPSGraphTensor, data: MPSGraphTensorData) {
        let data = try self.toTensorData()
        let ts = graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        return (ts, data)
    }
}
