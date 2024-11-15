//
//  FunctionalTests.swift
//
//
//  Created by m_quadra on 2024/7/2.
//

import Testing
import Irokane
import CoreML

fileprivate typealias F = Irokane.Functional

@Suite("F.")
struct FunctionalTests {
    
    @available(iOS 16.0, *)
    @Test("F.pad(x, pad=[3, 4])")
    func pad0() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(shape: [1, 2], dataType: .float16).ik.to(graph: graph)
        
        let y = F.pad(x, pad: (3, 4))
        
        guard let yData = graph.mpsGraph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 9])
    }
    
    @available(iOS 15.4, *)
    @Test("F.pad(x, pad=[0, 0, 1, 0, 0, 0])")
    func pad1() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(1...6).ik.to(graph: graph)
            .reshape([1, 3, 2])
        
        let y = F.pad(x, pad: [(0, 0), (1, 0), (0, 0)])
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1, 4, 2])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [
            0, 0,
            1, 2,
            3, 4,
            5, 6
        ])
    }
    
    @available(iOS 15.4, *)
    @Test func softmax() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .cast(to: .float16)
            .reshape([2, 3])
        
        let y = F.softmax(x, dim: -1)
        
        guard let yData = graph.mpsGraph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2, 3])
        
        let arr = try yData.ik.toFloat16s()
        #expect(arr == [
            0.09, 0.2448, 0.665,
            0.09, 0.2448, 0.665
        ])
    }
    
    @available(iOS 15.4, *)
    @Test func softplus() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
            .cast(to: .float16) + 0.5
        
        let y = F.softplus(x, beta: 2, threshold: 2)
        
        guard let yData = graph.mpsGraph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.ik.toFloat16s()
        #expect(arr == [0.6567, 1.5, 2.5])
    }
    
    @available(iOS 15.4, *)
    @Test("F.layer_norm(x, [x.shape[-1]]")
    func layerNorm0() throws {
        let graph = Irokane.Graph()
        let x = graph.arange(end: 6, dtype: .float32)
            .reshape([2, 3])
        
        let y = F.layerNorm(x)
        
        let yData = try y.tensorData()
        let arr = try yData.ik.toFloat32s()
        #expect(yData.shape == [2, 3])
        #expect(arr == [
            -1.2247356, 0.0, 1.2247356,
            -1.2247356, 0.0, 1.2247356,
        ])
    }
    
    @available(iOS 15.4, *)
    @Test("F.layer_norm(x, [x.shape[-1]], weight=w, bias=b)")
    func layerNorm1() throws {
        let graph = Irokane.Graph()
        let x = graph.arange(end: 6, dtype: .float32)
            .reshape([2, 3])
        let w = graph.zeros(3).cast(to: .float32) + 0.1
        let b = graph.zeros(3).cast(to: .float32) + 0.01
        
        let y = F.layerNorm(x, weight: w, bias: b)
        
        let yData = try y.tensorData()
        let arr = try yData.ik.toFloat32s()
        #expect(yData.shape == [2, 3])
        #expect(arr == [
            -0.11247356, 0.01, 0.13247356,
            -0.11247356, 0.01, 0.13247356,
        ])
    }
    
    @available(iOS 15.4, *)
    @Test("F.gelu(x)")
    func gelu() throws {
        let graph = Irokane.Graph()
        let x = graph.arange(end: 6, dtype: .float32)
            .reshape([1, 2, 3])
        
        let y = F.gelu(x)
        
        let yData = try y.tensorData()
        let arr = try yData.ik.toFloat32s()
        #expect(yData.shape == [1, 2, 3])
        #expect(arr == [
            0.0, 0.8413447, 1.9544997,
            2.9959502, 3.9998732, 4.9999986
        ])
    }
}
