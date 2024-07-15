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
    
    @Test("x >= y")
    func greaterThanOrEqualTensor() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray([0, 1, 0]).ik.toGraph(at: graph)
        let (y, yData) = try MLMultiArray([
            1, 0,
            1, 0,
            1, 0
        ]).ik.toGraph(at: graph)
        
        let z = x[..., nil] >= y.reshape([3, 2])
        
        guard let zData = graph.run(
            feeds: [
                x.tensor: xData,
                y.tensor: yData,
            ],
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
    
    @Test("sum(x, dim=-1)")
    func sum() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray([0, 1, 2]).ik.toGraph(at: graph)
        
        let y = Irokane.sum(x, dim: -1)
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [3])
    }
    
    @Test("x.gather(-1, idx)")
    func gather() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray([
            0, 1,
            2, 3,
            4, 5,
        ]).ik.toGraph(at: graph)
        let (i, iData) = try MLMultiArray([
            0,
            1,
            0,
        ]).ik.toGraph(at: graph)
        let x0 = x.reshape([3, 2])
        let i0 = i[..., nil]
        
        let y = x0.gather(dim: -1, index: i0)
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
                i.tensor: iData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3, 1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 3, 4])
    }
    
    @Test("x + y")
    func plus() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray(0..<3).ik.toGraph(at: graph)
        let (y, yData) = try MLMultiArray(1..<4).ik.toGraph(at: graph)
        
        let z = x + y
        
        guard let zData = graph.run(
            feeds: [
                x.tensor: xData,
                y.tensor: yData,
            ],
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [3])
        
        let arr = try zData.toInt32s()
        #expect(arr == [1, 3, 5])
    }
    
    @Test("x * y")
    func multiply() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray(0..<3).ik.toGraph(at: graph)
        let (y, yData) = try MLMultiArray(1..<4).ik.toGraph(at: graph)
        
        let z = x * y
        
        guard let zData = graph.run(
            feeds: [
                x.tensor: xData,
                y.tensor: yData,
            ],
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [3])
        
        let arr = try zData.toInt32s()
        #expect(arr == [0, 2, 6])
    }
    
    @Test("x.pow(a)")
    func pow() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray(0..<3).ik.toGraph(at: graph)
        
        let y = x.pow(2)
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 4])
    }
    
    @Test func sqrt() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray(0..<3).ik.toGraph(at: graph)
        
        let y = Irokane.sqrt(x*x)
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 2])
    }
    
    @Test("cat([x, y], 1)")
    func cat() async throws {
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray(0..<3).ik.toGraph(at: graph)
        let (y, yData) = try MLMultiArray(3..<6).ik.toGraph(at: graph)
        let x0 = x.reshape([1, 1, 3])
        let y0 = y.reshape([1, 1, 3])
        
        let z = Irokane.cat(x0, y0, dim: 1)
        
        guard let zData = graph.run(
            feeds: [
                x.tensor: xData,
                y.tensor: yData,
            ],
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
        let graph = MPSGraph()
        let (x, xData) = try MLMultiArray([Float.nan, 1]).ik.toGraph(at: graph)
        
        let y = Irokane.maximum(x, 0)
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let arr = try yData.toFloat32s()
        #expect(arr == [0, 1])
    }
}
