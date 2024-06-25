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
    func toTensor() throws(Errors) -> MLTensor {
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
        default: throw .todo()
        }
    }
    
    func toMLMultiArray() throws -> MLMultiArray {
        let dtype: MLMultiArrayDataType = switch self.dataType {
        case .float16: .float16
        case .float32: .float32
        default: throw Errors.todo()
        }
        
        let arr = try MLMultiArray(shape: self.shape, dataType: dtype)
        try arr.withUnsafeMutableBytes { ptr, strides in
            guard let dst = ptr.baseAddress else { throw Errors.msg("ptr.baseAddress") }
            self.mpsndarray().readBytes(consume dst, strideBytes: nil)
        }
        return arr
    }
}
