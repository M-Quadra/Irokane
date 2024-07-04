//
//  Graph+GetSetByMask.swift
//
//
//  Created by m_quadra on 2024/7/4.
//

import Testing
@testable import Irokane
import CoreML
import MetalPerformanceShadersGraph

@Suite("Graph GetSetByMask")
struct GraphGetSetByMask {
    
    @Test("y = x[mask]")
    func getByMask() async throws {
        let xArr = try MLMultiArray(shape: [2, 3], dataType: .int32)
        xArr.withUnsafeMutableBytes { ptr, strides in
            let arr = (0..<6).map { Int32($0) + 1 }
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Int32>.size * xArr.count)
        }
        let maskArr = try MLMultiArray(shape: [2, 3], dataType: .int32)
        maskArr.withUnsafeMutableBytes { ptr, strides in
            let arr: [Int32] = [
                0, 1, 0,
                1, 0, 1
            ]
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Int32>.size * maskArr.count)
        }
        
        let graph = MPSGraph()
        let (x, xData) = try xArr.ik.toGraph(at: graph)
        let (m, mData) = try maskArr.ik.toGraph(at: graph)
        
        let y = x[m]
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
                m.tensor: mData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toMLMultiArray()
        #expect(arr[0] == 2)
        #expect(arr[1] == 4)
        #expect(arr[2] == 6)
    }
    
    @Test("x[mask] = a")
    func setByMaskConstant() async throws {
        let xArr = try MLMultiArray(shape: [2, 3], dataType: .int32)
        xArr.withUnsafeMutableBytes { ptr, strides in
            let arr = (0..<6).map { Int32($0) + 1 }
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Int32>.size * xArr.count)
        }
        let maskArr = try MLMultiArray(shape: [2, 3], dataType: .int32)
        maskArr.withUnsafeMutableBytes { ptr, strides in
            let arr: [Int32] = [
                0, 1, 0,
                1, 0, 1
            ]
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Int32>.size * maskArr.count)
        }
        
        let graph = MPSGraph()
        let (x, xData) = try xArr.ik.toGraph(at: graph)
        let (m, mData) = try maskArr.ik.toGraph(at: graph)
        
        let y = x.setItem(at: m, 7)
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
                m.tensor: mData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2, 3])
        
        let arr = try yData.toMLMultiArray()
        #expect(arr[0] == 1)
        #expect(arr[1] == 7)
        #expect(arr[2] == 3)
        #expect(arr[3] == 7)
        #expect(arr[4] == 5)
        #expect(arr[5] == 7)
    }
    
    @Test("x[mask] = ts")
    func setByMaskTensor() async throws {
        let xArr = try MLMultiArray(shape: [2, 3], dataType: .int32)
        xArr.withUnsafeMutableBytes { ptr, strides in
            let arr = (0..<6).map { Int32($0) }
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Int32>.size * xArr.count)
        }
        let maskArr = try MLMultiArray(shape: [2, 3], dataType: .int32)
        maskArr.withUnsafeMutableBytes { ptr, strides in
            let arr: [Int32] = [
                0, 1, 0,
                1, 0, 1
            ]
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Int32>.size * maskArr.count)
        }
        let updateArr = try MLMultiArray(shape: [3], dataType: .int32)
        updateArr.withUnsafeMutableBytes { ptr, strides in
            let arr: [Int32] = [7, 8, 9]
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Int32>.size * updateArr.count)
        }

        let graph = MPSGraph()
        let (x, xData) = try xArr.ik.toGraph(at: graph)
        let (m, mData) = try maskArr.ik.toGraph(at: graph)
        let (u, uData) = try updateArr.ik.toGraph(at: graph)

        let y = x.setItem(at: m, u)

        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
                m.tensor: mData,
                u.tensor: uData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2, 3])
        
        let arr = try yData.toMLMultiArray().withUnsafeBytes { ptr in
            var arr = [Int32](repeating: -1, count: 6)
            memcpy(&arr, ptr.baseAddress!, MemoryLayout<Int32>.size * 6)
            return arr
        }
        #expect(arr == [
            0, 7, 2,
            8, 4, 9
        ])
    }
}
