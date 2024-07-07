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

public extension Functional {
    
    static func pad(_ input: Graph, pad: (left: Int, right: Int)) -> Graph {
        let graph = input.graph, x = input.tensor
        guard let rank = x.shape?.count, rank > 0 else {
            assertionFailure("input.shape is nil")
            return input
        }
        var left = [NSNumber](repeating: 0, count: rank)
        left[rank - 1] = pad.left as NSNumber
        var right = [NSNumber](repeating: 0, count: rank)
        right[rank - 1] = pad.right as NSNumber

        let y = graph.padTensor(x, with: .zero, leftPadding: consume left, rightPadding: consume right, constantValue: 0, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    
    static func softmax(_ input: Graph, dim: Int) -> Graph {
        let graph = input.graph, x = input.tensor
        assert(x.dataType != .int32)
        
        let y = graph.softMax(with: x, axis: dim, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    
    static func softplus(_ input: Graph, beta: Int = 1, threshold: Double = 20) -> Graph {
        let graph = input.graph, x = input.tensor
        let b = graph.constant(Double(beta), dataType: x.dataType)
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
        return Graph(tensor: consume y, graph: consume graph)
    }
}
