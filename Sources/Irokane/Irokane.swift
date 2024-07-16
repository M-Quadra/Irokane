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

public func sum(_ input: borrowing Graph, dim: Int) -> Graph {
    let graph = input.graph, x = input.tensor
    
    let x0 = graph.reductionSum(with: consume x, axis: dim, name: nil)
    let y = graph.squeeze(consume x0, axis: dim, name: nil)
    return Graph(tensor: consume y, graph: consume graph)
}

public func sqrt(_ input: borrowing Graph) -> Graph {
    let graph = input.graph, x = input.tensor
    
    let y = graph.squareRoot(with: consume x, name: nil)
    return Graph(tensor: consume y, graph: consume graph)
}

/// nan -> a
public func maximum(_ input: borrowing Graph, _ other: Double) -> Graph {
    let graph = input.graph, x = input.tensor
    let a = graph.constant(other, dataType: x.dataType)
    
    let y = graph.maximum(consume x, consume a, name: nil)
    return Graph(tensor: consume y, graph: consume graph)
}

public func log(_ input: borrowing Graph) -> Graph {
    let graph = input.graph, x = input.tensor
    
    let y = graph.logarithm(with: consume x, name: nil)
    return Graph(tensor: consume y, graph: consume graph)
}

public func cat(_ input: Graph, _ other: Graph, dim: Int) -> Graph {
    let graph = input.graph, x = input.tensor
    assert(graph == other.graph)
    
    let y = graph.concatTensor(consume x, with: other.tensor, dimension: dim, name: nil)
    return Graph(tensor: consume y, graph: consume graph)
}
