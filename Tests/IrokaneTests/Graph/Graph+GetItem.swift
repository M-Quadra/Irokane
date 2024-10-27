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
    
    @available(iOS 17.0, *)
    @Test("x[mask]")
    func getByMask() async throws {
        let graph = Graph()
        let x = try MLMultiArray((0..<6)).ik.to(graph: graph)
        let m = try MLMultiArray([
            0, 1, 0,
            1, 0, 1
        ]).ik.to(graph: graph)
        let x0 = x.reshape([2, 3])
        let m0 = m.reshape([2, 3])
        
        let y: Graph.Tensor = x0[m0]
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [1, 3, 5])
    }
    
    @available(iOS 17.0, *)
    @Test("x[mask, :]")
    func getByMaskSlice() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
        let m = try MLMultiArray([0, 2]).ik.to(graph: graph)
        let x0 = x.reshape([2, 3])
        
        let y = x0[m, ...]
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [3, 4, 5])
    }
    
    @available(iOS 15.4, *)
    @Test("x[..., i:]")
    func getItemFrom() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
        
        let y = x[..., 1...]
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let arr = try yData.toInt32s()
        #expect(arr == [1, 2])
    }
    
    @available(iOS 15.4, *)
    @Test("x[..., :i]")
    func getItemTo() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
        
        let y = x[..., ..<(-1)]
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1])
    }
    
    @available(iOS 15.4, *)
    @Test("x[..., None]")
    func getItemNone() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
        
        let y = x[..., .none]
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3, 1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 2])
    }
    
    @available(iOS 15.4, *)
    @Test("x[..., i]")
    func getItemAt() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
        
        let y: Graph.Tensor = x.reshape([3, 2])[..., 0]

        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])

        let arr = try yData.toInt32s()
        #expect(arr == [0, 2, 4])
    }
    
    @available(iOS 15.4, *)
    @Test("x[:, :-1]")
    func getItemSlice() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([1, 3, 2])
        
        let y = x[.all, ..<(-1)]
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 2, 2])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 2, 3])
    }
}
