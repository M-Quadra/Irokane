//
//  GraphTests.swift
//
//
//  Created by m_quadra on 2024/7/3.
//

import Testing
@testable import Irokane
import CoreML
import MetalPerformanceShadersGraph

@Suite("Graph")
struct GraphTests {
    
    @available(iOS 18.0, *)
    @Test("x[..., a] = b")
    func setItemSt() async throws {
        let graph = MPSGraph()
        
        let (x, xData) = try await MLTensor(repeating: 0.5, shape: [1, 2]).ik.toGraph(at: graph)
        let y = x.setItem(at: (..., 1), 2)
        
        guard let yData = graph.run(
            feeds: [x.tensor: xData],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 2])
        
        let arr = try yData.toMLMultiArray()
        #expect(arr[0] == 0.5)
        #expect(arr[1] == 2)
    }
    
    @available(iOS 18.0, *)
    @Test("x[..., -1] = a")
    func setItemEd() async throws {
        let graph = MPSGraph()
        
        let (x, xData) = try await MLTensor(repeating: 0.5, shape: [1, 2]).ik.toGraph(at: graph)
        let y = x.setItem(at: (..., -1), 2)
        
        guard let yData = graph.run(
            feeds: [x.tensor: xData],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 2])
        
        let arr = try yData.toMLMultiArray()
        #expect(arr[0] == 0.5)
        #expect(arr[1] == 2)
    }
}
