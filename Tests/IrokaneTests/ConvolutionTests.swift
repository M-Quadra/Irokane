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
    @Test func conv1d() throws {
        let graph = Irokane.Graph()
        let w = try MLMultiArray(0..<6).ik.to(graph: graph)
            .cast(to: .float32)
            .reshape([2, 3, 1])
        let b = try MLMultiArray([Float32](repeating: 0.5, count: 2)).ik.to(graph: graph)
        
        let conv1d = Conv1d(weight: w, bias: b)
        
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .cast(to: .float32)
            .reshape([1, 3, 2])
        let y = try conv1d.forward(x)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1, 2, 2])
        
        let arr = try yData.ik.toFloat32s()
        #expect(arr == [
            10.5, 13.5,
            28.5, 40.5,
        ])
    }
}
