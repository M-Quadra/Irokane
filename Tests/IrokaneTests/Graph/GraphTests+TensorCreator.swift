//
//  GraphTests+TensorCreator.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/3.
//

import Testing
import Irokane
import Foundation

struct GraphTestsTensorCreator {
    
    @available(iOS 14.0, *)
    @Test func zeros() throws {
        let graph = Irokane.Graph()
        
        let x = graph.zeros(3)
        
        let xDaya = try x.tensorData()
        let arr = try xDaya.ik.toInt32s()
        #expect(xDaya.shape == [3])
        #expect(arr == [0, 0, 0])
    }
    
    @available(iOS 14.0, *)
    @Test("data -> Graph.Tensor")
    func dataToGraphTensor() throws {
        let graph = Irokane.Graph()
        let src: [Float32] = [1, 2, 3]
        let data = src.withUnsafeBytes { Data($0) }
        
        let x = graph.tensor(data, shape: [3], dataType: .float32)
        
        let xData = try x.tensorData()
        let arr = try xData.ik.toFloat32s()
        #expect(xData.shape == [3])
        #expect(arr == src)
    }
    
    @available(iOS 15.4, *)
    @Test("arange(end=a)")
    func arange0() throws {
        let graph = Irokane.Graph()
        
        let x = graph.arange(end: 3)
        
        let xData = try x.tensorData()
        let arr = try xData.ik.toInt32s()
        #expect(xData.shape == [3])
        #expect(arr == [0, 1, 2])
    }
    
    @available(iOS 15.4, *)
    @Test("arange(end=a, dtype=t)")
    func arange1() throws {
        let graph = Irokane.Graph()
        
        let x = graph.arange(end: 3, dtype: .float32)
        
        let xData = try x.tensorData()
        let arr = try xData.ik.toFloat32s()
        #expect(xData.shape == [3])
        #expect(arr == [0, 1, 2])
    }
}
