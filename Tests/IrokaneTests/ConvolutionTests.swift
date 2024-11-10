//
//  ConvolutionTests.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/4.
//

import Testing
import Irokane
import CoreML

struct ConvolutionTests {
    
    @available(iOS 15.4, *)
    @Test("Conv1d(3, 2)")
    func conv1d0() throws {
        let graph = Irokane.Graph()
        let w = try MLMultiArray(0..<6).ik.to(graph: graph)
            .cast(to: .float32)
            .reshape([2, 3, 1])
        let b = try MLMultiArray([Float32](repeating: 0.5, count: 2)).ik.to(graph: graph)
        let conv1d = Conv1d(
            weight: w, bias: b,
            inChannels: 3, outChannels: 2
        )
        #expect(conv1d.inChannels == 3)
        #expect(conv1d.outChannels == 2)
        
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .cast(to: .float32)
            .reshape([1, 3, 2])
        let y = conv1d.forward(x)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1, 2, 2])
        
        let arr = try yData.ik.toFloat32s()
        #expect(arr == [
            10.5, 13.5,
            28.5, 40.5,
        ])
    }
    
    @available(iOS 15.4, *)
    @Test("Conv1d(3, 3)")
    func conv1d1() throws {
        let graph = Irokane.Graph()
        let w = try MLMultiArray(shape: [3, 3, 1], dataType: .float32)
            .ik.to(graph: graph)
        let b = try MLMultiArray(shape: [3], dataType: .float32)
            .ik.to(graph: graph)
        let conv1d = Conv1d(
            weight: w, bias: b,
            inChannels: 3, outChannels: 3
        )
        #expect(conv1d.inChannels == 3)
        #expect(conv1d.outChannels == 3)
        
        let x = try MLMultiArray(shape: [1, 3, 5], dataType: .float32)
            .ik.to(graph: graph)
        let y = conv1d.forward(x)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1, 3, 5])
    }
    
    @available(iOS 15.4, *)
    @Test("Conv1d(5, 5, kernel_size=3, groups=5)")
    func conv1d2() throws {
        let graph = Irokane.Graph()
        let w = graph.arange(end: 15, dtype: .float32)
            .reshape([5, 1, 3])
        let b = graph.zeros(5).cast(to: .float32) + 0.5
        let conv1d = Conv1d(
            weight: w, bias: b,
            inChannels: 5, outChannels: 5, kernelSize: 3, groups: 5
        )
        
        let x = graph.arange(end: 15, dtype: .float32)
            .reshape([1, 5, 3])
        let y = conv1d.forward(x)
        
        let yData = try y.tensorData()
        let arr = try yData.ik.toFloat32s()
        #expect(yData.shape == [1, 5, 1])
        #expect(arr == [5.5, 50.5, 149.5, 302.5, 509.5])
    }
    
    @available(iOS 15.4, *)
    @Test("Conv1d(5, 5, kernel_size=3, groups=5, padding=1)")
    func conv1d3() throws {
        let graph = Irokane.Graph()
        let w = graph.arange(end: 15, dtype: .float32)
            .reshape([5, 1, 3])
        let b = graph.zeros(5).cast(to: .float32) + 0.5
        let conv1d = Conv1d(
            weight: w, bias: b,
            inChannels: 5, outChannels: 5, groups: 5, padding: 1
        )
        #expect(conv1d.inChannels == 5)
        #expect(conv1d.outChannels == 5)
        
        let x = graph.arange(end: 15, dtype: .float32)
            .reshape([1, 5, 3])
        let y = conv1d.forward(x)
        
        let yData = try y.tensorData()
        let arr = try yData.ik.toFloat32s()
        #expect(yData.shape == [1, 5, 3])
        #expect(arr == [
              2.5,   5.5,   2.5,
             32.5,  50.5,  32.5,
             98.5, 149.5,  98.5,
            200.5, 302.5, 200.5,
            338.5, 509.5, 338.5,
        ])
    }
}
