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

public extension Wrapper where Base == Double {
    
    consuming func toTensor(at graph: Graph, dataType: MPSDataType = .float16) -> Graph.Tensor {
        let a = graph.graph.constant(self.base, dataType: dataType)
        return Graph.Tensor(graph: graph, tensor: consume a)
    }
}
