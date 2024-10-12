//
//  Graph+Creator.swift
//
//
//  Created by m_quadra on 2024/7/1.
//

import MetalPerformanceShadersGraph

@available(iOS 14.0, *)
public extension Graph.Tensor {
    
    static func zerosLike(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = input.graph.graph, x = input.tensor
        let zero = graph.constant(0, dataType: x.dataType)
        
        let y = graph.multiplication(x, zero, name: nil)
        return Graph.Tensor(graph: input.graph, tensor: consume y)
    }
}
