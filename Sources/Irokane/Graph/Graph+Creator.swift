//
//  Graph+Creator.swift
//
//
//  Created by m_quadra on 2024/7/1.
//

import MetalPerformanceShadersGraph

public extension Graph {
    
    static func zerosLike(_ input: borrowing Graph) -> Graph {
        let graph = input.graph, tensor = input.tensor
        let zero = graph.constant(0, dataType: tensor.dataType)
        let ts = graph.multiplication(tensor, zero, name: nil)
        return Graph(tensor: consume ts, graph: graph)
    }
}
