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
}
