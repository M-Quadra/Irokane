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
    
    @available(iOS 15.4, *)
    @Test("sum(x, dim=-1), 1d")
    func sum1d() throws {
        let graph = Graph()
        let x = try MLMultiArray([0, 1, 2]).ik.to(graph: graph)
        
        let y = Irokane.sum(x, dim: -1)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [3])
    }
    
    @available(iOS 15.4, *)
    @Test("sum(x, dim=-1), 2d")
    func sum2d() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
        let x0 = x.reshape([2, 3])
        
        let y = Irokane.sum(x0, dim: -1)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [2])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [3, 12])
    }
    
    @available(iOS 15.4, *)
    @Test("sum(x, dims=[a, b])")
    func sumDims() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([1, 2, 3])
        
        let y = Irokane.sum(x, dims: [1, 2])
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [15])
    }
    
    @available(iOS 15.4, *)
    @Test("x.gather(-1, idx)")
    func gather() throws {
        let graph = Graph()
        let x = try MLMultiArray([
            0, 1,
            2, 3,
            4, 5,
        ]).ik.to(graph: graph)
        let i = try MLMultiArray([
            0,
            1,
            0,
        ]).ik.to(graph: graph)
        let x0 = x.reshape([3, 2])
        let i0 = i[..., .none]
        
        let y = x0.gather(dim: -1, index: i0)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [3, 1])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [0, 3, 4])
    }
    
    @available(iOS 15.4, *)
    @Test("x.pow(a)")
    func pow() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
        
        let y = x.pow(2)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [3])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [0, 1, 4])
    }
    
    @available(iOS 15.4, *)
    @Test func sqrt() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
        
        let y = Irokane.sqrt(x*x)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [3])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [0, 1, 2])
    }
    
    @available(iOS 15.4, *)
    @Test("cat([x, y], 1)")
    func cat() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
        let y = try MLMultiArray(3..<6).ik.to(graph: graph)
        let x0 = x.reshape([1, 1, 3])
        let y0 = y.reshape([1, 1, 3])
        
        let z = Irokane.cat(x0, y0, dim: 1)
        
        let zData = try z.tensorData()
        #expect(zData.shape == [1, 2, 3])
        
        let arr = try zData.ik.toInt32s()
        #expect(arr == [
            0, 1, 2,
            3, 4, 5,
        ])
    }
    
    @available(iOS 15.4, *)
    @Test("maximum(x, a)")
    func maximum() throws {
        let graph = Graph()
        let x = try MLMultiArray([Float.nan, 1]).ik.to(graph: graph)
        
        let y = Irokane.maximum(x, 0)
        
        guard let yData = graph.mpsGraph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let arr = try yData.ik.toFloat32s()
        #expect(arr == [0, 1])
    }
    
    @available(iOS 15.4, *)
    @Test("exp(x)")
    func exp() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([0, 1, 2]).ik.to(graph: graph)
            .cast(to: .float32)
        
        let y = Irokane.exp(x)
        
        guard let yData = graph.mpsGraph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.ik.toFloat32s()
        #expect(arr == [1.0, 2.7182817, 7.3890557])
    }
    
    @available(iOS 15.4, *)
    @Test("ceil(x)")
    func ceil() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([0.1, 1.5, 2.9]).ik.to(graph: graph)
        
        let y = Irokane.ceil(x)
        
        guard let yData = graph.mpsGraph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.ik.toFloat32s()
        #expect(arr == [1, 2, 3])
    }
    
    @available(iOS 15.4, *)
    @Test("x.max(), 1d")
    func max1d() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<3).ik.to(graph: graph)
        
        let y = x.max()
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [2])
    }
    
    @available(iOS 15.4, *)
    @Test("x.max(), 2d")
    func max2d() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([2, 3])
        
        let y = x.max()
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [5])
    }
    
    @available(iOS 15.4, *)
    @Test("arange(a)", arguments: 0...9)
    func arange0(len: Int) throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([len]).ik.to(graph: graph)
        
        let y = Irokane.arange(x)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [len as NSNumber])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == (0..<len).map { Int32($0) })
    }
    @available(iOS 15.4, *)
    @Test("arange(x.max())")
    func arange1() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([2, 3])
        
        let y = Irokane.arange(x.max())
        
        let yData = try y.tensorData()
        #expect(yData.shape == [5])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [0, 1, 2, 3, 4])
    }
    
    @available(iOS 15.4, *)
    @Test("x.transpose(a, b)")
    func transpose() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([1, 2, 3])
        
        let y = x.transpose(0, 2)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [3, 2, 1])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [
            0, 3,
            1, 4,
            2, 5,
        ])
    }
    
    @available(iOS 15.4, *)
    @Test("matmul(x, y)")
    func matmul() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(1...4).ik.to(graph: graph)
            .cast(to: .float16)
            .reshape([2, 2])
        let y = try MLMultiArray(5...8).ik.to(graph: graph)
            .cast(to: .float16)
            .reshape([2, 2])
        
        let z = Irokane.matmul(x, y)
        
        guard let zData = graph.mpsGraph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [2, 2])
        
        let arr = try zData.toFloat16s()
        #expect(arr == [19, 22, 43, 50])
    }
    
    @available(iOS 15.4, *)
    @Test("squeeze(a)")
    func squeeze() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([2, 1, 3, 1])
        
        let y = x.squeeze(1)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [2, 3, 1])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [0, 1, 2, 3, 4, 5])
    }
    
    @available(iOS 15.4, *)
    @Test("randn_like(x)")
    func randnLike() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([2, 3])
            .cast(to: .float32)
        
        let y = try Irokane.randnLike(x)
        
        guard let yData = graph.mpsGraph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2, 3])
        
        let arr = try yData.ik.toFloat32s()
        #expect(abs(arr.mean) < 0.1)
        #expect(abs(arr.std - 1) < 0.26)
    }
    
    @available(iOS 16.0, *)
    @Test("cumsum(x, a)")
    func cumsum() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
        
        let y = Irokane.cumsum(x, dim: -1)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [6])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [0, 1, 3, 6, 10, 15])
    }
    
    @available(iOS 15.4, *)
    @Test("flip(x, [a])")
    func flip() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([1, 2, 3])
        
        let y = Irokane.flip(x, dims: [1])
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1, 2, 3])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [
            3, 4, 5,
            0, 1, 2,
        ])
    }
    
    @available(iOS 15.4, *)
    @Test("zeros_like(x)")
    func zerosLike() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([1, 2, 3])
        
        let y = Irokane.zerosLike(x)
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1, 2, 3])
        
        let sum = try yData.ik.toInt32s().reduce(0, +)
        #expect(sum == 0)
    }
    
    @available(iOS 15.4, *)
    @Test("split(x, [a, b], c)")
    func split() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([1, 2, 3])
        
        let ys = Irokane.split(x, splits: [1, 1], dim: 1)
        let y0 = ys[0], y1 = ys[1]
        
        let dic = graph.mpsGraph.run(
            feeds: graph.feeds,
            targetTensors: [y0.tensor, y1.tensor],
            targetOperations: nil
        )
        guard let y0Data = dic[y0.tensor],
              let y1Data = dic[y1.tensor]
        else { throw Errors.msg("empty result") }
        #expect(y0Data.shape == [1, 1, 3])
        #expect(y1Data.shape == [1, 1, 3])
        
        let arr0 = try y0Data.ik.toInt32s()
        let arr1 = try y1Data.ik.toInt32s()
        #expect(arr0 == [0, 1, 2])
        #expect(arr1 == [3, 4, 5])
    }
    
    @available(iOS 15.4, *)
    @Test("x.mean(dim=-1)")
    func mean() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([2, 3])
            .cast(to: .float32)
        
        let y = x.mean(dim: -1)
        
        let yData = try y.tensorData()
        let arr = try yData.ik.toFloat32s()
        #expect(yData.shape == [2])
        #expect(arr == [1, 4])
    }
}
