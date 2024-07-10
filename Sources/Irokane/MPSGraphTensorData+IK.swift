//
//  MPSGraphTensorData+IK.swift
//  
//
//  Created by m_quadra on 2024/6/23.
//

import CoreML
import MetalPerformanceShadersGraph

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
    
    func toMLMultiArray() throws -> MLMultiArray {
        let dtype: MLMultiArrayDataType = switch self.dataType {
        case .float16: .float16
        case .float32: .float32
        case .int32: .int32
        default: throw Errors.todo("\(self.dataType)")
        }
        
        let arr = try MLMultiArray(shape: self.shape, dataType: dtype)
        try arr.withUnsafeMutableBytes { ptr, strides in
            guard let dst = ptr.baseAddress else { throw Errors.msg("ptr.baseAddress") }
            self.mpsndarray().readBytes(consume dst, strideBytes: nil)
        }
        return arr
    }
    
    func toInt32s() throws(Errors) -> [Int32] {
        if self.dataType != .int32 { throw .msg("\(self.dataType)") }
        let cnt = self.shape.map { $0.intValue }.reduce(1, *)
        var arr = [Int32](repeating: Int32.min, count: cnt)
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
