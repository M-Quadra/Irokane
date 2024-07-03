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
    
    consuming func toGraphDSL(at graph: MPSGraph, dataType: MPSDataType = .float16) -> Graph {
        let ts = graph.constant(self.base, dataType: dataType)
        return Graph(tensor: consume ts, graph: graph)
    }
}
