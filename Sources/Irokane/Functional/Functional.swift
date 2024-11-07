//
//  Functional.swift
//
//
//  Created by m_quadra on 2024/6/24.
//

import Foundation

public struct Functional {
    fileprivate init() {}
}

@available(iOS 14.0, *)
public extension Functional {
    
    static func softmax(_ input: borrowing Graph.Tensor, dim: Int) -> Graph.Tensor {
        let graph = input.graph.mpsGraph, x = input.tensor
        assert(x.dataType != .int32)
        
        let y = graph.softMax(with: x, axis: dim, name: nil)
        return Graph.Tensor(graph: input.graph, tensor: consume y)
    }
    
    static func layerNorm(
        _ input: borrowing Graph.Tensor,
        weight: Graph.Tensor? = nil,
        bias: Graph.Tensor? = nil,
        eps: Float = 1e-5
    ) -> Graph.Tensor {
        let mpsGraph = input.graph.mpsGraph, x = input.tensor
        let mean = mpsGraph.mean(of: x, axes: [-1], name: nil)
        let variance = mpsGraph.variance(of: x, mean: mean, axes: [-1], name: nil)
        
        let y = mpsGraph.normalize(
            consume x,
            mean: consume mean, variance: consume variance,
            gamma: weight?.tensor, beta: bias?.tensor,
            epsilon: eps,
            name: nil
        )
        return input.graph.tensor(consume y)
    }
    
    static func gelu(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = input.graph, mpsGraph = graph.mpsGraph
        let x = input.tensor
        let half = mpsGraph.constant(0.5, dataType: x.dataType)
        let one = mpsGraph.constant(1.0, dataType: x.dataType)
        let sqrt2 = mpsGraph.constant(sqrt(2.0), dataType: x.dataType)
        
        let x0 = mpsGraph.multiplication(consume half, x, name: nil)
        
        let x1 = mpsGraph.division(consume x, consume sqrt2, name: nil)
        let erf = mpsGraph.erf(with: consume x1, name: nil)
        let erf0 = mpsGraph.addition(consume erf, consume one, name: nil)
        
        let y = mpsGraph.multiplication(consume x0, consume erf0, name: nil)
        return graph.tensor(consume y)
    }
}

@available(iOS 15.0, *)
public extension Functional {
    
    static func pad(_ input: borrowing Graph.Tensor, pad: (left: Int, right: Int)) -> Graph.Tensor {
        let graph = input.graph.mpsGraph, x = input.tensor
        guard let rank = x.shape?.count, rank > 0 else {
            assertionFailure("input.shape is nil")
            return Graph.Tensor(graph: input.graph, tensor: consume x)
        }
        var left = [NSNumber](repeating: 0, count: rank)
        left[rank - 1] = pad.left as NSNumber
        var right = [NSNumber](repeating: 0, count: rank)
        right[rank - 1] = pad.right as NSNumber
        
        let y = graph.padTensor(x, with: .zero, leftPadding: consume left, rightPadding: consume right, constantValue: 0, name: nil)
        return Graph.Tensor(graph: input.graph, tensor: consume y)
    }
    static func pad(_ input: borrowing Graph.Tensor, pad: borrowing [(left: Int, right: Int)]) -> Graph.Tensor {
        let graph = input.graph.mpsGraph, x = input.tensor
        guard let rank = x.shape?.count, rank > 0 else {
            assertionFailure("input.shape is nil")
            return Graph.Tensor(graph: input.graph, tensor: consume x)
        }
        guard pad.count <= rank else {
            assertionFailure("pad is invalid")
            return Graph.Tensor(graph: input.graph, tensor: consume x)
        }
        var left = [NSNumber](repeating: 0, count: rank)
        var right = [NSNumber](repeating: 0, count: rank)
        for (i, v) in pad.enumerated() {
            left[rank-1 - i] = v.left as NSNumber
            right[rank-1 - i] = v.right as NSNumber
        }
        
        let y = graph.padTensor(x, with: .zero, leftPadding: consume left, rightPadding: consume right, constantValue: 0, name: nil)
        return Graph.Tensor(graph: input.graph, tensor: consume y)
    }
    
    static func softplus(_ input: borrowing Graph.Tensor, beta: Double = 1, threshold: Double = 20) -> Graph.Tensor {
        let graph = input.graph.mpsGraph, x = input.tensor
        let b = graph.constant(beta, dataType: x.dataType)
        let t = graph.constant(threshold, dataType: x.dataType)
        assert(x.dataType != .int32)
        
        let xb = graph.multiplication(x, b, name: nil)
        let m = graph.lessThanOrEqualTo(xb, t, name: nil)
        
        let m0 = graph.cast(m, to: x.dataType, name: nil)
        let xb0 = graph.multiplication(xb, m0, name: nil)
        let exp = graph.exponent(with: xb0, name: nil)
        let exp0 = graph.addition(exp, m0, name: nil)
        let log = graph.logarithm(with: exp0, name: nil)
        let log0 = graph.division(log, b, name: nil)
        
        let m1 = graph.logicalNOR(m, m, name: nil)
        let x0 = graph.multiplication(
            x, graph.cast(m1, to: x.dataType, name: nil),
            name: nil
        )
        
        let y = graph.addition(log0, x0, name: nil)
        return Graph.Tensor(graph: input.graph, tensor: consume y)
    }
}
