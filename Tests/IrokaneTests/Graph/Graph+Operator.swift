//
//  Graph+Operator.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/19.
//

import Testing
import Irokane
import CoreML

@Suite("Graph Operator")
struct GraphOperator {
    
    @available(iOS 15.4, *)
    @Test("x + y")
    func plus() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
        let y = try MLMultiArray(1..<4).ik.to(graph: graph)
        
        let z = x + y
        
        let zData = try z.tensorData()
        #expect(zData.shape == [3])
        
        let arr = try zData.ik.toInt32s()
        #expect(arr == [1, 3, 5])
    }
    
    @available(iOS 15.4, *)
    @Test("a * x")
    func mul0() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
        
        let y = 2 * x
        
        let yData = try y.tensorData()
        let arr = try yData.ik.toInt32s()
        #expect(yData.shape == [6])
        #expect(arr == [0, 2, 4, 6, 8, 10])
    }
    @available(iOS 15.4, *)
    @Test("x * a")
    func mul1() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
        
        let y = x * 2
        
        let yData = try y.tensorData()
        let arr = try yData.ik.toInt32s()
        #expect(yData.shape == [6])
        #expect(arr == [0, 2, 4, 6, 8, 10])
    }
    
    @available(iOS 15.4, *)
    @Test("x * y")
    func mul2() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
        let y = try MLMultiArray(1..<4).ik.to(graph: graph)
        
        let z = x * y
        
        let zData = try z.tensorData()
        let arr = try zData.ik.toInt32s()
        #expect(zData.shape == [3])
        #expect(arr == [0, 2, 6])
    }
    
    @available(iOS 15.4, *)
    @Test("x * mask")
    func mul3() throws {
        let graph = Irokane.Graph()
        let x = graph.arange(end: 3)
        let m = try MLMultiArray([
            0, 1, 0
        ]).ik.to(graph: graph)
        
        let y = x * m
        
        let yData = try y.tensorData()
        let arr = try yData.ik.toInt32s()
        #expect(yData.shape == [3])
        #expect(arr == [0, 1, 0])
    }
    
    @available(iOS 15.4, *)
    @Test("mask * x")
    func mul4() throws {
        let graph = Irokane.Graph()
        let x = graph.arange(end: 3)
        let m = try MLMultiArray([
            0, 1, 0
        ]).ik.to(graph: graph)

        let y = m * x

        let yData = try y.tensorData()
        let arr = try yData.ik.toInt32s()
        #expect(yData.shape == [3])
        #expect(arr == [0, 1, 0])
    }
    
    @available(iOS 15.4, *)
    @Test("x / a")
    func division() throws {
        let graph = Graph()
        let x = try MLMultiArray([2, 4, 6]).ik.to(graph: graph)
        
        let y = x / 2
        
        let yData = try y.tensorData()
        #expect(yData.shape == [3])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [1, 2, 3])
    }
    
    @available(iOS 15.4, *)
    @Test("x < y")
    func lessThan() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([1, 3, 5]).ik.to(graph: graph)
        let y = try MLMultiArray([2, 4, 6]).ik.to(graph: graph)
        
        let z = x < y
        
        let zData = try z.tensorData(isFill: false)
        let arr = try zData.ik.toBools()
        #expect(zData.shape == [3])
        #expect(arr == [true, true, true])
    }
    
    @available(iOS 15.4, *)
    @Test("x >= y")
    func greaterThanOrEqualTensor() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([0, 1, 0]).ik.to(graph: graph)
        let y = try MLMultiArray([
            1, 0,
            1, 0,
            1, 0
        ]).ik.to(graph: graph).reshape([3, 2])
        
        let z = x[..., .none] >= y
        
        let zData = try z.tensorData(isFill: false)
        let arr = try zData.ik.toBools()
        #expect(zData.shape == [3, 2])
        #expect(arr == [
            false, true,
            true,  true,
            false, true,
        ])
    }
}
