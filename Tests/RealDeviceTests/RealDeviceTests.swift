//
//  RealDeviceTests.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/2.
//

import Testing
import Irokane
import CoreML

@available(iOS 17.0, *)
@Test("x[mask] = y", arguments: Array(0...6))
func byMaskTensor(count: Int) throws {
    let graph = Irokane.Graph()
    
    var x = try MLMultiArray(0..<6).ik.to(graph: graph)
        .reshape([2, 3])
    let maskArr = Array(0..<6).map { $0 < count ? $0+6 : 0 }
        .shuffled()
    let mask = try MLMultiArray(maskArr).ik.to(graph: graph)
        .reshape([2, 3])
    let y = try MLMultiArray(maskArr.filter { $0 != 0 }).ik.to(graph: graph)
    
    x[mask] .= y
    
    let xData = try x.tensorData()
    #expect(xData.shape == [2, 3])
    
    let arr = try xData.ik.toInt32s()
    let expectArr = maskArr.enumerated().map { i, v in
        Int32(v == 0 ? i : v)
    }
    #expect(arr == expectArr)
}
