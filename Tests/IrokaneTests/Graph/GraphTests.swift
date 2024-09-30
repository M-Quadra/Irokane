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
    
    @Test("x >= y")
    func greaterThanOrEqualTensor() async throws {
        let graph = Graph()
        let x = try MLMultiArray([0, 1, 0]).ik.toTensor(at: graph)
        let y = try MLMultiArray([
            1, 0,
            1, 0,
            1, 0
        ]).ik.toTensor(at: graph)
        
        let z = x[..., nil] >= y.reshape([3, 2])
        
        guard let zData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [3, 2])
        
        let arr = try zData.toBools()
        #expect(arr == [
            false, true,
            true,  true,
            false, true,
        ])
    }
    
    @Test("sum(x, dim=-1), 1d")
    func sum1d() async throws {
        let graph = Graph()
        let x = try MLMultiArray([0, 1, 2]).ik.toTensor(at: graph)
        
        let y = Irokane.sum(x, dim: -1)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [3])
    }
    
    @Test("sum(x, dim=-1), 2d")
    func sum2d() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
        let x0 = x.reshape([2, 3])
        
        let y = Irokane.sum(x0, dim: -1)

        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])

        let arr = try yData.toInt32s()
        #expect(arr == [3, 12])
    }
    
    @Test("x.gather(-1, idx)")
    func gather() async throws {
        let graph = Graph()
        let x = try MLMultiArray([
            0, 1,
            2, 3,
            4, 5,
        ]).ik.toTensor(at: graph)
        let i = try MLMultiArray([
            0,
            1,
            0,
        ]).ik.toTensor(at: graph)
        let x0 = x.reshape([3, 2])
        let i0 = i[..., nil]
        
        let y = x0.gather(dim: -1, index: i0)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3, 1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 3, 4])
    }
    
    @Test("x + y")
    func plus() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        let y = try MLMultiArray(1..<4).ik.toTensor(at: graph)
        
        let z = x + y
        
        guard let zData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [3])
        
        let arr = try zData.toInt32s()
        #expect(arr == [1, 3, 5])
    }
    
    @Test("x * y")
    func multiply() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        let y = try MLMultiArray(1..<4).ik.toTensor(at: graph)
        
        let z = x * y
        
        guard let zData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [3])
        
        let arr = try zData.toInt32s()
        #expect(arr == [0, 2, 6])
    }
    
    @Test("x.pow(a)")
    func pow() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        
        let y = x.pow(2)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 4])
    }
    
    @Test func sqrt() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        
        let y = Irokane.sqrt(x*x)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 2])
    }
    
    @Test("cat([x, y], 1)")
    func cat() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        let y = try MLMultiArray(3..<6).ik.toTensor(at: graph)
        let x0 = x.reshape([1, 1, 3])
        let y0 = y.reshape([1, 1, 3])
        
        let z = Irokane.cat(x0, y0, dim: 1)
        
        guard let zData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [1, 2, 3])
        
        let arr = try zData.toInt32s()
        #expect(arr == [
            0, 1, 2,
            3, 4, 5,
        ])
    }
    
    @Test("maximum(x, a)")
    func maximum() async throws {
        let graph = Graph()
        let x = try MLMultiArray([Float.nan, 1]).ik.toTensor(at: graph)
        
        let y = Irokane.maximum(x, 0)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let arr = try yData.toFloat32s()
        #expect(arr == [0, 1])
    }
    
    @Test("x / a")
    func division() async throws {
        let graph = Graph()
        let x = try MLMultiArray([2, 4, 6]).ik.toTensor(at: graph)
        
        let y = x / 2
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { return }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [1, 2, 3])
    }
    
    @Test("exp(x)")
    func exp() async throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([0, 1, 2]).ik.toTensor(at: graph)
            .cast(to: .float32)
        
        let y = Irokane.exp(x)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toFloat32s()
        #expect(arr == [1.0, 2.7182817, 7.3890557])
    }
}
