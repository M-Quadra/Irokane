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
    @Test("x >= y")
    func greaterThanOrEqualTensor() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([0, 1, 0]).ik.toTensor(at: graph)
        let y = try MLMultiArray([
            1, 0,
            1, 0,
            1, 0
        ]).ik.toTensor(at: graph)
        
        let z = x[..., nil] >= y.reshape([3, 2])
        
        guard let zData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [3, 2])
        
        let arr = try zData.toBools()
        #expect(arr == [
            false, true,
            true,  true,
            false, true,
        ])
    }
    
    @available(iOS 15.4, *)
    @Test("sum(x, dim=-1), 1d")
    func sum1d() throws {
        let graph = Graph()
        let x = try MLMultiArray([0, 1, 2]).ik.toTensor(at: graph)
        
        let y = Irokane.sum(x, dim: -1)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [3])
    }
    
    @available(iOS 15.4, *)
    @Test("sum(x, dim=-1), 2d")
    func sum2d() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
        let x0 = x.reshape([2, 3])
        
        let y = Irokane.sum(x0, dim: -1)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let arr = try yData.toInt32s()
        #expect(arr == [3, 12])
    }
    
    @available(iOS 15.4, *)
    @Test("sum(x, dims=[a, b])")
    func sumDims() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
            .reshape([1, 2, 3])
        
        let y = Irokane.sum(x, dims: [1, 2])
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1])
        
        let arr = try yData.toInt32s()
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
        ]).ik.toTensor(at: graph)
        let i = try MLMultiArray([
            0,
            1,
            0,
        ]).ik.toTensor(at: graph)
        let x0 = x.reshape([3, 2])
        let i0 = i[..., nil]
        
        let y = x0.gather(dim: -1, index: i0)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3, 1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 3, 4])
    }
    
    @available(iOS 15.4, *)
    @Test("x + y")
    func plus() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        let y = try MLMultiArray(1..<4).ik.toTensor(at: graph)
        
        let z = x + y
        
        guard let zData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [3])
        
        let arr = try zData.toInt32s()
        #expect(arr == [1, 3, 5])
    }
    
    @available(iOS 15.4, *)
    @Test("x * y")
    func multiply() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        let y = try MLMultiArray(1..<4).ik.toTensor(at: graph)
        
        let z = x * y
        
        guard let zData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [3])
        
        let arr = try zData.toInt32s()
        #expect(arr == [0, 2, 6])
    }
    
    @available(iOS 15.4, *)
    @Test("x.pow(a)")
    func pow() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        
        let y = x.pow(2)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 4])
    }
    
    @available(iOS 15.4, *)
    @Test func sqrt() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        
        let y = Irokane.sqrt(x*x)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 2])
    }
    
    @available(iOS 15.4, *)
    @Test("cat([x, y], 1)")
    func cat() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        let y = try MLMultiArray(3..<6).ik.toTensor(at: graph)
        let x0 = x.reshape([1, 1, 3])
        let y0 = y.reshape([1, 1, 3])
        
        let z = Irokane.cat(x0, y0, dim: 1)
        
        guard let zData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [z.tensor],
            targetOperations: nil
        )[z.tensor] else { throw Errors.msg("empty result") }
        #expect(zData.shape == [1, 2, 3])
        
        let arr = try zData.toInt32s()
        #expect(arr == [
            0, 1, 2,
            3, 4, 5,
        ])
    }
    
    @available(iOS 15.4, *)
    @Test("maximum(x, a)")
    func maximum() throws {
        let graph = Graph()
        let x = try MLMultiArray([Float.nan, 1]).ik.toTensor(at: graph)
        
        let y = Irokane.maximum(x, 0)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2])
        
        let arr = try yData.toFloat32s()
        #expect(arr == [0, 1])
    }
    
    @available(iOS 15.4, *)
    @Test("x / a")
    func division() throws {
        let graph = Graph()
        let x = try MLMultiArray([2, 4, 6]).ik.toTensor(at: graph)
        
        let y = x / 2
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { return }
        #expect(yData.shape == [3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [1, 2, 3])
    }
    
    @available(iOS 15.4, *)
    @Test("exp(x)")
    func exp() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([0, 1, 2]).ik.toTensor(at: graph)
            .cast(to: .float32)
        
        let y = Irokane.exp(x)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toFloat32s()
        #expect(arr == [1.0, 2.7182817, 7.3890557])
    }
    
    @available(iOS 15.4, *)
    @Test("ceil(x)")
    func ceil() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([0.1, 1.5, 2.9]).ik.toTensor(at: graph)
        
        let y = Irokane.ceil(x)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toFloat32s()
        #expect(arr == [1, 2, 3])
    }
    
    @available(iOS 15.4, *)
    @Test("x.max(), 1d")
    func max1d() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<3).ik.toTensor(at: graph)
        
        let y = x.max()
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [2])
    }
    
    @available(iOS 15.4, *)
    @Test("x.max(), 2d")
    func max2d() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
            .reshape([2, 3])
        
        let y = x.max()
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [5])
    }
    
    @available(iOS 15.4, *)
    @Test("arange(a)", arguments: 0...9)
    func arange0(len: Int) throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([len]).ik.toTensor(at: graph)
        
        let y = Irokane.arange(x)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [len as NSNumber])
        
        let arr = try yData.toInt32s()
        #expect(arr == (0..<len).map { Int32($0) })
    }
    @available(iOS 15.4, *)
    @Test("arange(x.max())")
    func arange1() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
            .reshape([2, 3])
        
        let y = Irokane.arange(x.max())
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [5])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 2, 3, 4])
    }
    
    @available(iOS 15.4, *)
    @Test("x < y")
    func lessThan() throws {
        let graph = Irokane.Graph()
        let x0 = try MLMultiArray([1, 3, 5]).ik.toTensor(at: graph)
        let x1 = try MLMultiArray([2, 4, 6]).ik.toTensor(at: graph)
        
        let y = x0 < x1
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3])
        
        let arr = try yData.toBools()
        #expect(arr == [true, true, true])
    }
    
    @available(iOS 15.4, *)
    @Test("x.transpose(a, b)")
    func transpose() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
            .reshape([1, 2, 3])
        
        let y = x.transpose(0, 2)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [3, 2, 1])
        
        let arr = try yData.toInt32s()
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
        let x = try MLMultiArray(1...4).ik.toTensor(at: graph)
            .cast(to: .float16)
            .reshape([2, 2])
        let y = try MLMultiArray(5...8).ik.toTensor(at: graph)
            .cast(to: .float16)
            .reshape([2, 2])
        
        let z = Irokane.matmul(x, y)
        
        guard let zData = graph.graph.run(
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
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
            .reshape([2, 1, 3, 1])
        
        let y = x.squeeze(1)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2, 3, 1])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 2, 3, 4, 5])
    }
    
    @available(iOS 15.4, *)
    @Test("randn_like(x)")
    func randnLike() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
            .reshape([2, 3])
            .cast(to: .float32)
        
        let y = try Irokane.randnLike(x)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [2, 3])
        
        let arr = try yData.toFloat32s()
        #expect(abs(arr.mean) < 0.1)
        #expect(abs(arr.std - 1) < 0.26)
    }
    
    @available(iOS 15.4, *)
    @Test("a * x")
    func multiplication0() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
        
        let y = 2 * x
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [6])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 2, 4, 6, 8, 10])
    }
    @available(iOS 15.4, *)
    @Test("x * a")
    func multiplication1() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
        
        let y = x * 2
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [6])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 2, 4, 6, 8, 10])
    }
    
    @available(iOS 16.0, *)
    @Test("cumsum(x, a)")
    func cumsum() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
        
        let y = Irokane.cumsum(x, dim: -1)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [6])
        
        let arr = try yData.toInt32s()
        #expect(arr == [0, 1, 3, 6, 10, 15])
    }
    
    @available(iOS 15.4, *)
    @Test("flip(x, [a])")
    func flip() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
            .reshape([1, 2, 3])
        
        let y = Irokane.flip(x, dims: [1])
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 2, 3])
        
        let arr = try yData.toInt32s()
        #expect(arr == [
            3, 4, 5,
            0, 1, 2,
        ])
    }
    
    @available(iOS 15.4, *)
    @Test("zeros_like(x)")
    func zerosLike() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray(0..<6).ik.toTensor(at: graph)
            .reshape([1, 2, 3])
        
        let y = Irokane.zerosLike(x)
        
        guard let yData = graph.graph.run(
            feeds: graph.feeds,
            targetTensors: [y.tensor],
            targetOperations: nil
        )[y.tensor] else { throw Errors.msg("empty result") }
        #expect(yData.shape == [1, 2, 3])
        
        let sum = try yData.toInt32s().reduce(0, +)
        #expect(sum == 0)
    }
}
