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
        
        let arr = try yData.toFloat32s()
        #expect(arr == [0.5, 2])
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
        
        let arr = try yData.toFloat32s()
        #expect(arr == [0.5, 2])
    }
    
    @Test("x[..., a:]")
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
    
    @Test("x[..., :a]")
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
    
    @Test("x[..., -1] += a")
    func addItemEd() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray(0..<3).ik.toGraph(at: graph)
        
        let y = x.addItem(at: (..., -1), 1)
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 3])
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
