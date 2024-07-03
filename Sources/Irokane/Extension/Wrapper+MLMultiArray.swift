//
//  Wrapper+MLMultiArray.swift
//
//
//  Created by m_quadra on 2024/6/26.
//

import CoreML
import MetalPerformanceShadersGraph

public extension MLMultiArray {
    var ik: Wrapper<MLMultiArray> { Wrapper(base: self) }
}

public extension Wrapper where Base == MLMultiArray {
    
    consuming func toTensor() -> Tensor {
        return Tensor(base: self.base)
    }
    
    consuming func toGraph(at graph: MPSGraph) throws(Errors) -> (dsl: Graph, data: MPSGraphTensorData) {
        let data = try self.base.toTensorData()
        let ts = graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        return (Graph(tensor: ts, graph: graph), data)
    }
}
