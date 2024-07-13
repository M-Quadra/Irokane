//
//  Graph+GetItem.swift
//  Irokane
//
//  Created by m_quadra on 2024/7/12.
//

import Testing
@testable import Irokane
import CoreML
import MetalPerformanceShadersGraph

@Suite("Graph GetItem")
struct GraphGetItem {
    
    @Test("x[..., i:]")
    func getItemFrom() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray(0..<3).ik.toGraph(at: graph)
        
        let y = x[..., 1...]
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let arr = try yData.toInt32s()
        #expect(arr == [1, 2])
    }
    
    @Test("x[..., :i]")
    func getItemTo() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray(0..<3).ik.toGraph(at: graph)
        
        let y = x[..., ..<(-1)]
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1])
    }
    
    @Test("x[..., None]")
    func getItemNone() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray(0..<3).ik.toGraph(at: graph)
        
        let y = x[..., nil]
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3, 1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 2])
    }
}
