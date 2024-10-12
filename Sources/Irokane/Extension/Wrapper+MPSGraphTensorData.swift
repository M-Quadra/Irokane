//
//  Wrapper+MPSGraphTensorData.swift
//  Irokane
//
//  Created by m_quadra on 2024/7/22.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 14.0, *)
public extension MPSGraphTensorData {
    var ik: Wrapper<MPSGraphTensorData> { Wrapper(base: self) }
}

@available(iOS 16.0, *)
public extension Wrapper<MPSGraphTensorData> {
    
    consuming func toMLMultiArray() throws -> MLMultiArray {
        let data = self.base
        let dtype: MLMultiArrayDataType = switch data.dataType {
        case .float16: .float16
        case .float32: .float32
        case .int32: .int32
        default: throw Errors.todo("\(data.dataType)")
        }
        
        let arr = try MLMultiArray(shape: data.shape, dataType: dtype)
        try arr.withUnsafeMutableBytes { ptr, strides in
            guard let dst = ptr.baseAddress else { throw Errors.msg("ptr.baseAddress") }
            data.mpsndarray().readBytes(dst, strideBytes: nil)
        }
        return arr
    }
}
