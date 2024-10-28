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

@available(iOS 15.4, *)
public extension Wrapper<MLMultiArray> {
    
    consuming func to(graph: Graph) throws(Errors) -> Graph.Tensor {
        let data = try self.base.toTensorData()
        let x = graph.graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        
        graph.feeds[x] = consume data
        return Graph.Tensor(graph: graph, tensor: consume x)
    }
}
