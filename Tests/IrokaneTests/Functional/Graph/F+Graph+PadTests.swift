//
//  F+Graph+PadTests.swift
//
//
//  Created by m_quadra on 2024/7/2.
//

import Testing
import Irokane
import CoreML
import MetalPerformanceShadersGraph

fileprivate typealias F = Irokane.Functional

extension FunctionalTests.Graph {
    
    @Test func pad() async throws {
        let graph = MPSGraph()
        
        let (x, xData) = try MLMultiArray(shape: [1, 2], dataType: .float16).ik.toGraph(at: graph)
        let y = F.pad(x, pad: (3, 4))
        
        guard let yData = graph.run(
            feeds: [x.tensor: xData],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 9])
    }
}
