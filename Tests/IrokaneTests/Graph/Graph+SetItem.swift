//
//  Graph+SetItem.swift
//  Irokane
//
//  Created by m_quadra on 2024/7/16.
//

import Testing
@testable import Irokane
import CoreML
import MetalPerformanceShadersGraph

@Suite("Graph SetItem")
struct GraphSetItem {
    
    @available(iOS 15.4, *)
    @Test("x[mask] = a")
    func byMaskConstant() async throws {
        let graph = Irokane.Graph()
        var x = try MLMultiArray((0..<6)).ik.to(graph: graph)
            .reshape([2, 3])
        let mask = try MLMultiArray([
            0, 1, 0,
            1, 0, 1
        ]).ik.to(graph: graph)
            .reshape([2, 3])
        
        x[mask] .= 6
        
        guard let xData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [x.tensor],
            targetOperations: nil
        )[x.tensor] else { throw Errors.msg("empty result") }
        #expect(xData.shape == [2, 3])
        
        let arr = try xData.toInt32s()
        #expect(arr == [
            0, 6, 2,
            6, 4, 6
        ])
    }
    
    @available(iOS 17.0, *)
    @Test("x[mask] = y")
    func byMaskTensor() async throws {
        let graph = Irokane.Graph()
        
        var x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([2, 3])
        let mask = try MLMultiArray([
            0, 1, 0,
            1, 0, 1,
        ]).ik.to(graph: graph)
            .reshape([2, 3])
        let y = try MLMultiArray([7, 8, 9]).ik.to(graph: graph)
        
        x[mask] .= y
        
        guard let xData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [x.tensor],
            targetOperations: nil
        )[x.tensor] else { throw Errors.msg("empty result") }
        #expect(xData.shape == [2, 3])
        
        let arr = try xData.toInt32s()
        #expect(arr == [
            0, 7, 2,
            8, 4, 9
        ])
    }
    
    @available(iOS 18.0, *)
    @Test("x[..., i] = a")
    func setLast0() async throws {
        let graph = Graph()
        var x = try await MLTensor(repeating: 0.5, shape: [1, 2]).ik.to(graph: graph)
        
        x[..., 1] .= 2
        
        guard let xData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [x.tensor],
            targetOperations: nil
        )[x.tensor] else { throw Errors.msg("empty result") }
        #expect(xData.shape == [1, 2])
        
        let arr = try xData.toFloat32s()
        #expect(arr == [0.5, 2])
    }
    
    @available(iOS 18.0, *)
    @Test("x[..., -i] = a")
    func setLast1() async throws {
        let graph = Graph()
        var x = try await MLTensor(repeating: 0.5, shape: [1, 2]).ik.to(graph: graph)
        
        x[..., -1] .= 2
        
        guard let xData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [x.tensor],
            targetOperations: nil
        )[x.tensor] else { throw Errors.msg("empty result") }
        #expect(xData.shape == [1, 2])
        
        let arr = try xData.toFloat32s()
        #expect(arr == [0.5, 2])
    }
    
    @available(iOS 15.4, *)
    @Test("x[..., -1] += a")
    func addLast() async throws {
        let graph = Graph()
        var x = try MLMultiArray(0..<3).ik.to(graph: graph)
        
        x[..., -1] += 1
        
        guard let xData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [x.tensor],
            targetOperations: nil
        )[x.tensor] else { throw Errors.msg("empty result") }
        #expect(xData.shape == [3])
        
        let arr = try xData.toInt32s()
        #expect(arr == [0, 1, 3])
    }
}
