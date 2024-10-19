//
//  Graph+Operator.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/19.
//

import Testing
@testable import Irokane
import CoreML

@Suite("Graph Operator")
struct GraphOperator {
    
    @available(iOS 15.4, *)
    @Test("x < y")
    func lessThan() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([1, 3, 5]).ik.toTensor(at: graph)
        let y = try MLMultiArray([2, 4, 6]).ik.toTensor(at: graph)
        
        let z = x < y
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toBools()
        #expect(arr == [true, true, true])
    }
}
