//
//  MPSGraphTensorData+IK.swift
//
//
//  Created by m_quadra on 2024/6/23.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 14.0, *)
public extension Wrapper<MPSGraphTensorData> {
    
    consuming func to(graph: Graph) throws(Errors) -> Graph.Tensor {
        let data = self.base
        let x = graph.mpsGraph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        
        graph.feeds[x] = consume data
        return Graph.Tensor(graph: graph, tensor: consume x)
    }
    
    consuming func toInt32s() throws -> [Int32] {
        let tData = self.base
        if tData.dataType != .int32 { throw Errors.msg("\(tData.dataType)") }
        let cnt = tData.shape.map { $0.intValue }.reduce(1, *)
        
        return try [Int32](unsafeUninitializedCapacity: cnt) { buffer, initializedCount in
            guard let dst = buffer.baseAddress else { throw Errors.msg("buffer.baseAddress") }
            tData.mpsndarray().readBytes(dst, strideBytes: nil)
            initializedCount = cnt
        }
    }
    
    consuming func toFloat16s() throws -> [Float16] {
        let tData = self.base
        if tData.dataType != .float16 { throw Errors.msg("\(tData.dataType)") }
        let cnt = tData.shape.map { $0.intValue }.reduce(1, *)
        
        return try [Float16](unsafeUninitializedCapacity: cnt) { buffer, initializedCount in
            guard let dst = buffer.baseAddress else { throw Errors.msg("buffer.baseAddress") }
            tData.mpsndarray().readBytes(dst, strideBytes: nil)
            initializedCount = cnt
        }
    }
    
    consuming func toFloat32s() throws -> [Float32] {
        let tData = self.base
        if tData.dataType != .float32 { throw Errors.msg("\(tData.dataType)") }
        let cnt = tData.shape.map { $0.intValue }.reduce(1, *)
        
        return try [Float32](unsafeUninitializedCapacity: cnt) { buffer, initializedCount in
            guard let dst = buffer.baseAddress else { throw Errors.msg("buffer.baseAddress") }
            tData.mpsndarray().readBytes(dst, strideBytes: nil)
            initializedCount = cnt
        }
    }
    
    @available(iOS 15.0, *)
    consuming func toBools() throws -> [Bool] {
        let tData = self.base
        if tData.dataType != .bool { throw Errors.msg("\(tData.dataType)") }
        let cnt = tData.shape.map { $0.intValue }.reduce(1, *)
        
        return try [Bool](unsafeUninitializedCapacity: cnt) { buffer, initializedCount in
            guard let dst = buffer.baseAddress else { throw Errors.msg("buffer.baseAddress") }
            tData.mpsndarray().readBytes(dst, strideBytes: nil)
            initializedCount = cnt
        }
    }
}

@available(iOS 14.0, *)
extension MPSGraphTensorData {
    
    @available(iOS 18.0, *)
    func toMLTensor() throws(Errors) -> MLTensor {
        let shape = self.shape.map { $0.intValue }
        let count = shape.reduce(1, { $0 * $1 })
        switch self.dataType {
        case .float16:
            var arr = [Float16](repeating: 0, count: count)
            self.mpsndarray().readBytes(&arr, strideBytes: nil)
            return MLTensor(shape: consume shape, scalars: consume arr)
        case .float32:
            var arr = [Float32](repeating: 0, count: count)
            self.mpsndarray().readBytes(&arr, strideBytes: nil)
            return MLTensor(shape: consume shape, scalars: consume arr)
        case .int32:
            var arr = [Int32](repeating: 0, count: count)
            self.mpsndarray().readBytes(&arr, strideBytes: nil)
            return MLTensor(shape: consume shape, scalars: consume arr)
        default: throw .todo("\(self.dataType)")
        }
    }
    
    func toFloat16s() throws(Errors) -> [Float16] {
        if self.dataType != .float16 { throw .msg("\(self.dataType)") }
        let cnt = self.shape.map { $0.intValue }.reduce(1, *)
        var arr = [Float16](repeating: -.greatestFiniteMagnitude, count: cnt)
        self.mpsndarray().readBytes(&arr, strideBytes: nil)
        return arr
    }
    
    func toFloat32s() throws(Errors) -> [Float32] {
        if self.dataType != .float32 { throw .msg("\(self.dataType)") }
        let cnt = self.shape.map { $0.intValue }.reduce(1, *)
        var arr = [Float32](repeating: -.greatestFiniteMagnitude, count: cnt)
        self.mpsndarray().readBytes(&arr, strideBytes: nil)
        return arr
    }
}
