//
//  Wrapper+Double.swift
//  
//
//  Created by m_quadra on 2024/7/2.
//

import MetalPerformanceShadersGraph

public extension Double {
    var ik: Wrapper<Double> { Wrapper(base: self) }
}

@available(iOS 14.0, *)
public extension Wrapper<Double> {
    
    consuming func to(graph: Graph, dataType: MPSDataType = .float16) -> Graph.Tensor {
        let a = graph.mpsGraph.constant(self.base, dataType: dataType)
        return Graph.Tensor(graph: graph, tensor: consume a)
    }
}
