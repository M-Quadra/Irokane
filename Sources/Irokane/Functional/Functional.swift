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
        let graph = input.graph, src = input.tensor
        guard let rank = src.shape?.count, rank > 0 else {
            assertionFailure("input.shape is nil")
            return input
        }
        var left = [NSNumber](repeating: 0, count: rank)
        left[rank - 1] = pad.left as NSNumber
        var right = [NSNumber](repeating: 0, count: rank)
        right[rank - 1] = pad.right as NSNumber

        let dst = graph.padTensor(src, with: .zero, leftPadding: consume left, rightPadding: consume right, constantValue: 0, name: nil)
        return Graph(tensor: consume dst, graph: consume graph)
    }
}
