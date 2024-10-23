//
//  Irokane.swift
//
//
//  Created by m_quadra on 2024/7/5.
//

import Foundation
import MetalPerformanceShadersGraph

@available(iOS 16.0, *)
public func cumsum(_ input: borrowing Graph.Tensor, dim: Int) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.cumulativeSum(consume x, axis: dim, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 15.4, *)
public func sum(_ input: borrowing Graph.Tensor, dim: Int) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let x0 = graph.reductionSum(with: consume x, axis: dim, name: nil)
    let y = graph.squeeze(consume x0, axis: dim, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}
@available(iOS 15.4, *)
public func sum(_ input: borrowing Graph.Tensor, dims: [Int]) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    let axes = dims as [NSNumber]
    
    let x0 = graph.reductionSum(with: consume x, axes: axes, name: nil)
    let y = graph.squeeze(consume x0, axes: consume axes, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 14.0, *)
public func sqrt(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.squareRoot(with: consume x, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

/// nan -> a
@available(iOS 14.0, *)
public func maximum(_ input: borrowing Graph.Tensor, _ other: Double) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    let a = graph.constant(other, dataType: x.dataType)
    
    let y = graph.maximum(consume x, consume a, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 14.0, *)
public func log(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.logarithm(with: consume x, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 14.0, *)
public func cat(_ input: borrowing Graph.Tensor, _ other: borrowing Graph.Tensor, dim: Int) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    assert(graph == other.graph.graph)
    
    let y = graph.concatTensor(consume x, with: other.tensor, dimension: dim, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 14.0, *)
public func exp(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.exponent(with: consume x, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 14.0, *)
public func ceil(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.ceil(with: consume x, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 15.4, *)
public func arange(_ end: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = end.graph.graph, x = end.tensor
    
    let y = graph.coordinate(alongAxis: 0, withShapeTensor: x, name: nil)
    return Graph.Tensor(graph: end.graph, tensor: consume y)
}

@available(iOS 14.0, *)
public func matmul(_ lhs: borrowing Graph.Tensor, _ rhs: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = lhs.graph.graph, x = lhs.tensor, y = rhs.tensor
    assert(rhs.graph.graph == graph)
    
    let z = graph.matrixMultiplication(primary: consume x, secondary: consume y, name: nil)
    return Graph.Tensor(graph: lhs.graph, tensor: consume z)
}

@available(iOS 15.4, *)
public func randnLike(_ input: borrowing Graph.Tensor) throws(Errors) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    switch x.dataType {
    case .bFloat16, .float16, .float32: break
    default: throw .msg("unsupport dtype")
    }
    guard let op = MPSGraphRandomOpDescriptor(distribution: .normal, dataType: x.dataType)
    else { throw .msg("MPSGraphRandomOpDescriptor") }
    op.samplingMethod = .boxMuller
    
    let shape = graph.shapeOf(consume x, name: nil)
    let y = graph.randomTensor(withShapeTensor: consume shape, descriptor: consume op, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 15.0, *)
public func flip(_ input: borrowing Graph.Tensor, dims: [Int]) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.reverse(consume x, axes: dims as [NSNumber], name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 14.0, *)
public func zerosLike(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
    let graph = input.graph.graph, x = input.tensor
    
    let y = graph.subtraction(x, x, name: nil)
    return Graph.Tensor(graph: input.graph, tensor: consume y)
}

@available(iOS 15.4, *)
public func split(_ input: borrowing Graph.Tensor, splits: consuming [Int], dim: Int) -> [Graph.Tensor] {
    let graph = input.graph.graph, x = input.tensor
    
    let arr = graph.split(consume x, splitSizes: splits as [NSNumber], axis: dim, name: nil)
    return arr.map { Graph.Tensor(graph: input.graph, tensor: $0) }
}
