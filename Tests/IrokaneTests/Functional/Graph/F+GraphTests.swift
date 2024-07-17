//
//  F+GraphTests.swift
//
//
//  Created by m_quadra on 2024/7/2.
//

import Testing
@testable import Irokane
import CoreML
import MetalPerformanceShadersGraph

fileprivate typealias F = Irokane.Functional

@Suite("Graph F.")
struct FunctionalGraphTests {
    
    @Test func pad() async throws {
        let graph = Graph()
        
        let x = try MLMultiArray(shape: [1, 2], dataType: .float16).ik.toTensor(at: graph)
        let y = F.pad(x, pad: (3, 4))
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 9])
    }
    
    @Test func softmax() async throws {
        let xArr = try MLMultiArray(shape: [2, 3], dataType: .float16)
        xArr.withUnsafeMutableBytes { ptr, strides in
            let arr = (0..<6).map { Float16($0) }
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Float16>.size * xArr.count)
        }
        
        let graph = Graph()
        let x = try xArr.ik.toTensor(at: graph)
        
        let y = F.softmax(x, dim: -1)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2, 3])
        
        let cnt = yData.shape.map { $0.intValue }.reduce(1, *)
        let arr = try yData.toMLMultiArray().withUnsafeBytes { ptr in
            var arr = [Float16](repeating: -1, count: cnt)
            memcpy(&arr, ptr.baseAddress!, MemoryLayout<Float16>.size * cnt)
            return arr
        }
        #expect(arr == [
            0.09, 0.2448, 0.665,
            0.09, 0.2448, 0.665
        ])
    }
    
    @Test func softplus() async throws {
        let xArr = try MLMultiArray(shape: [3], dataType: .float16)
        xArr.withUnsafeMutableBytes { ptr, strides in
            let arr = (0..<3).map { Float16($0) + 0.5 }
            memcpy(ptr.baseAddress!, arr, MemoryLayout<Float16>.size * xArr.count)
        }

        let graph = Graph()
        let x = try xArr.ik.toTensor(at: graph)

        let y = F.softplus(x, beta: 2, threshold: 2)

        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])

        let cnt = yData.shape.map { $0.intValue }.reduce(1, *)
        let arr = try yData.toMLMultiArray().withUnsafeBytes { ptr in
            var arr = [Float16](repeating: -1, count: cnt)
            memcpy(&arr, ptr.baseAddress!, MemoryLayout<Float16>.size * cnt)
            return arr
        }
        #expect(arr == [0.6567, 1.5, 2.5])
    }
}
