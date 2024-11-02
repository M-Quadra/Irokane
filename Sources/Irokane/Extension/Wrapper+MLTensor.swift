//
//  Wrapper+MLTensor.swift
//  
//
//  Created by m_quadra on 2024/6/26.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 18.0, *)
public extension Wrapper<MLTensor> {
    
    consuming func to(graph: Graph) async throws(Errors) -> Graph.Tensor {
        let data = try await self.base.toTensorData()
        let x = graph.mpsGraph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        
        graph.feeds[x] = consume data
        return Graph.Tensor(graph: graph, tensor: consume x)
    }
}
