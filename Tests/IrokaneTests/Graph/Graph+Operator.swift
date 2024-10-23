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
    @Test("x + y")
    func plus() throws {
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
    
    @available(iOS 15.4, *)
    @Test("a * x")
    func multiplication0() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
        
        let y = 2 * x
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [6])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 2, 4, 6, 8, 10])
    }
    @available(iOS 15.4, *)
    @Test("x * a")
    func multiplication1() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
        
        let y = x * 2
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [6])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 2, 4, 6, 8, 10])
    }
    
    @available(iOS 15.4, *)
    @Test("x * y")
    func multiply() throws {
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
    
    @available(iOS 15.4, *)
    @Test("x / a")
    func division() throws {
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
    
    @available(iOS 15.4, *)
    @Test("x >= y")
    func greaterThanOrEqualTensor() throws {
        let graph = Irokane.Graph()
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
}
