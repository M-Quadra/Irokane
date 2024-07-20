//
//  Graph+SetItem.swift
//  Irokane
//
//  Created by m_quadra on 2024/7/16.
//

import Testing
@testable import Irokane
import CoreML
import MetalPerformanceShadersGraph

@Suite("Graph SetItem")
struct GraphSetItem {
    
    @Test("x[mask] = a")
    func byMaskConstant() async throws {
        let graph = Graph()
        let x = try MLMultiArray((0..<6)).ik.toTensor(at: graph)
            .reshape([2, 3])
        let mask = try MLMultiArray([
            0, 1, 0,
            1, 0, 1
        ]).ik.toTensor(at: graph)
            .reshape([2, 3])
        
        x[mask] .= 6
        
        guard let xData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [x.tensor],
            targetOperations: nil
        )[x.tensor] else { throw Errors.msg("empty result") }
        #expect(xData.shape == [2, 3])
        
        let arr = try xData.toInt32s()
        #expect(arr == [
            0, 6, 2,
            6, 4, 6
        ])
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
        
        let graph = Graph()
        let x = try xArr.ik.toTensor(at: graph)
        let m = try maskArr.ik.toTensor(at: graph)
        let u = try updateArr.ik.toTensor(at: graph)
        
        let y = x.setItem(at: m, u)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
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
    
    @available(iOS 18.0, *)
    @Test("x[..., i] = a")
    func setLast0() async throws {
        let graph = Graph()
        let x = try await MLTensor(repeating: 0.5, shape: [1, 2]).ik.toTensor(at: graph)
        
        x[..., 1] .= 2
        
        guard let xData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [x.tensor],
            targetOperations: nil
        )[x.tensor] else { throw Errors.msg("empty result") }
        #expect(xData.shape == [1, 2])
        
        let arr = try xData.toFloat32s()
        #expect(arr == [0.5, 2])
    }
    
    @available(iOS 18.0, *)
    @Test("x[..., -i] = a")
    func setLast1() async throws {
        let graph = Graph()
        let x = try await MLTensor(repeating: 0.5, shape: [1, 2]).ik.toTensor(at: graph)
        
        x[..., -1] .= 2
        
        guard let xData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [x.tensor],
            targetOperations: nil
        )[x.tensor] else { throw Errors.msg("empty result") }
        #expect(xData.shape == [1, 2])
        
        let arr = try xData.toFloat32s()
        #expect(arr == [0.5, 2])
    }
    
    @Test("x[..., -1] += a")
    func addLast() async throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        
        x[..., -1] += 1
        
        guard let xData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [x.tensor],
            targetOperations: nil
        )[x.tensor] else { throw Errors.msg("empty result") }
        #expect(xData.shape == [3])
        
        let arr = try xData.toInt32s()
        #expect(arr == [0, 1, 3])
    }
}
