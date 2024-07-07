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
        
        let arr = try yData.toMLMultiArray()
        #expect(arr[0] == 0.5)
        #expect(arr[1] == 2)
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
        
        let arr = try yData.toMLMultiArray()
        #expect(arr[0] == 0.5)
        #expect(arr[1] == 2)
    }
    
    @Test("x[..., a:]")
    func getItemFrom() async throws {
        let xArr = try MLMultiArray(shape: [3], dataType: .int32)
        xArr.withUnsafeMutableBytes { ptr, strides in
            let arr = (0..<3).map { Int32($0) }
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Int32>.size * xArr.count)
        }
        
        let graph = MPSGraph()
        let (x, xData) = try xArr.ik.toGraph(at: graph)
        
        let y = x[..., 1...]
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let cnt = yData.shape.map { $0.intValue }.reduce(1, *)
        let arr = try yData.toMLMultiArray().withUnsafeBytes { ptr in
            var arr = [Int32](repeating: -1, count: cnt)
            memcpy(&arr, ptr.baseAddress!, MemoryLayout<Int32>.size * cnt)
            return arr
        }
        #expect(arr == [1, 2])
    }
    
    @Test("x[..., :a]")
    func getItemTo() async throws {
        let xArr = try MLMultiArray(shape: [3], dataType: .int32)
        xArr.withUnsafeMutableBytes { ptr, strides in
            let arr = (0..<3).map { Int32($0) }
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Int32>.size * xArr.count)
        }
        
        let graph = MPSGraph()
        let (x, xData) = try xArr.ik.toGraph(at: graph)
        
        let y = x[..., ..<(-1)]
        
        guard let yData = graph.run(
            feeds: [
                x.tensor: xData,
            ],
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let cnt = yData.shape.map { $0.intValue }.reduce(1, *)
        let arr = try yData.toMLMultiArray().withUnsafeBytes { ptr in
            var arr = [Int32](repeating: -1, count: cnt)
            memcpy(&arr, ptr.baseAddress!, MemoryLayout<Int32>.size * cnt)
            return arr
        }
        #expect(arr == [0, 1])
    }
}
