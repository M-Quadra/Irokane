//
//  MLMultiArray+Convert.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/1.
//

import Testing
import Irokane
import CoreML

@Suite("MLMultiArray Convert")
struct MLMultiArrayConvert {
    
    @available(iOS 15.4, *)
    @Test("int32 -> Graph.Tensor", arguments: [
        ([0], []),
        ([3], [1, 2, 3]),
    ])
    func int32ToGraphTensor(shape: [NSNumber], values: [Int32]) throws {
        let graph = Irokane.Graph()
        let ipt = try MLMultiArray(values)
        #expect(ipt.dataType == .int32)
        
        let xData = try ipt.ik.to(graph: graph).tensorData()
        #expect(xData.shape == shape)
        #expect(xData.dataType == .int32)
        
        let arr = try xData.ik.toInt32s()
        #expect(arr == values)
    }
    
    @available(iOS 16.0, *)
    @Test("float16 -> Graph.Tensor")
    func float16ToGraphTensor() throws {
        let graph = Irokane.Graph()
        let ipt = try MLMultiArray(shape: [3], dataType: .float16)
        for i in 0..<3 {
            ipt[i] = i as NSNumber
        }
        #expect(ipt.dataType == .float16)
        
        let xData = try ipt.ik.to(graph: graph).tensorData()
        #expect(xData.shape == [3])
        #expect(xData.dataType == .float16)
        
        let arr = try xData.ik.toFloat16s()
        #expect(arr == [0.0, 1.0, 2.0])
    }
    
    @available(iOS 15.4, *)
    @Test("float32 -> Graph.Tensor")
    func float32ToGraphTensor() throws {
        let graph = Irokane.Graph()
        let ipt = try MLMultiArray(shape: [3], dataType: .float32)
        for i in 0..<3 {
            ipt[i] = i as NSNumber
        }
        #expect(ipt.dataType == .float32)
        
        let xData = try ipt.ik.to(graph: graph).tensorData()
        #expect(xData.shape == [3])
        #expect(xData.dataType == .float32)
        
        let arr = try xData.ik.toFloat32s()
        #expect(arr == [0.0, 1.0, 2.0])
    }
    
    @available(iOS 15.4, *)
    @Test("float64 -> Graph.Tensor")
    func float64ToGraphTensor() throws {
        let graph = Irokane.Graph()
        let ipt = try MLMultiArray([1.0, 2.0, 3.0])
        #expect(ipt.dataType == .double)
        
        let xData = try ipt.ik.to(graph: graph).tensorData()
        #expect(xData.shape == [3])
        #expect(xData.dataType == .float32)
        
        let arr = try xData.ik.toFloat32s()
        #expect(arr == [1.0, 2.0, 3.0])
    }
}
