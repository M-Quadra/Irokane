//
//  Irokane.swift
//
//
//  Created by m_quadra on 2024/7/5.
//

public func cumsum(_ input: borrowing Graph, dim: Int) -> Graph {
    let graph = input.graph, x = input.tensor
    
    let y = graph.cumulativeSum(consume x, axis: -1, name: nil)
    return Graph(tensor: consume y, graph: consume graph)
}
