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
    
    consuming func toTensor(at graph: Graph) throws(Errors) -> Graph.Tensor {
        let data = try self.base.toTensorData()
        let x = graph.graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        
        graph.feeds[x] = consume data
        return Graph.Tensor(graph: graph, tensor: consume x)
    }
}
