//
//  Irokane.swift
//
//
//  Created by m_quadra on 2024/7/5.
//

public func cumsum(_ input: borrowing Graph.Tensor, dim: Int) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.cumulativeSum(consume x, axis: -1, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

public func sum(_ input: borrowing Graph.Tensor, dim: Int) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let x0 = graph.reductionSum(with: consume x, axis: dim, name: nil)
    let y = graph.squeeze(consume x0, axis: dim, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

public func sqrt(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.squareRoot(with: consume x, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

/// nan -> a
public func maximum(_ input: borrowing Graph.Tensor, _ other: Double) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    let a = graph.constant(other, dataType: x.dataType)
    
    let y = graph.maximum(consume x, consume a, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

public func log(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.logarithm(with: consume x, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

public func cat(_ input: borrowing Graph.Tensor, _ other: borrowing Graph.Tensor, dim: Int) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    assert(graph == other.graph.graph)
    
    let y = graph.concatTensor(consume x, with: other.tensor, dimension: dim, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

public func exp(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.exponent(with: consume x, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

public func ceil(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.ceil(with: consume x, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}
