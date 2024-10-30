//
//  F+GraphTests.swift
//
//
//  Created by m_quadra on 2024/7/2.
//

import Testing
@testable import Irokane
import CoreML
import MetalPerformanceShadersGraph

fileprivate typealias F = Irokane.Functional

@Suite("Graph F.")
struct FunctionalGraphTests {
    
    @available(iOS 16.0, *)
    @Test func pad0() async throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(shape: [1, 2], dataType: .float16).ik.to(graph: graph)
        
        let y = F.pad(x, pad: (3, 4))
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 9])
    }
    @available(iOS 15.4, *)
    @Test func pad1() async throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(1...6).ik.to(graph: graph)
            .reshape([1, 3, 2])
        
        let y = F.pad(x, pad: [(0, 0), (1, 0), (0, 0)])
        
        let yData = try y.tensorData
        #expect(yData.shape == [1, 4, 2])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [
            0, 0,
            1, 2,
            3, 4,
            5, 6
        ])
    }
    
    @available(iOS 15.4, *)
    @Test func softmax() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .cast(to: .float16)
            .reshape([2, 3])
        
        let y = F.softmax(x, dim: -1)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2, 3])
        
        let arr = try yData.toFloat16s()
        #expect(arr == [
            0.09, 0.2448, 0.665,
            0.09, 0.2448, 0.665
        ])
    }
    
    @available(iOS 15.4, *)
    @Test func softplus() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
            .cast(to: .float16) + 0.5
        
        let y = F.softplus(x, beta: 2, threshold: 2)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toFloat16s()
        #expect(arr == [0.6567, 1.5, 2.5])
    }
}
