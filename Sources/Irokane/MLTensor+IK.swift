//
//  MLTensor+IK.swift
//
//
//  Created by m_quadra on 2024/6/23.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 18.0, *)
extension MLTensor {
    
    var mpsDataType: MPSDataType { get throws(Errors) {
        switch true {
        case self.scalarType == Float16.self: .float16
        case self.scalarType == Float32.self: .float32
        default: throw .todo()
        }
    }}
    
    func toTensorData() async throws(Errors) -> MPSGraphTensorData {
        guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { throw .msg("Failed to create MTLDevice") }
        
        switch true {
        case self.scalarType == Float16.self:
            let arr = await self.shapedArray(of: Float16.self)
            let buf: MTLBuffer? = arr.withUnsafeShapedBufferPointer { ptr, shape, strides in
                guard let dst = ptr.baseAddress else { return nil }
                return device.makeBuffer(bytes: consume dst, length: MemoryLayout<Float16>.size * self.scalarCount)
            }
            guard let buf = consume buf else { throw .msg("Failed to create MTLBuffer") }
            return MPSGraphTensorData(consume buf, shape: self.shape as [NSNumber], dataType: .float16)
        case self.scalarType == Float32.self:
            let arr = await self.shapedArray(of: Float32.self)
            let buf: MTLBuffer? = arr.withUnsafeShapedBufferPointer { ptr, shape, strides in
                guard let dst = ptr.baseAddress else { return nil }
                return device.makeBuffer(bytes: consume dst, length: MemoryLayout<Float32>.size * self.scalarCount)
            }
            guard let buf = consume buf else { throw .msg("Failed to create MTLBuffer") }
            return MPSGraphTensorData(consume buf, shape: self.shape as [NSNumber], dataType: .float32)
        default: throw .todo()
        }
    }
}

// MARK: - Public
@available(iOS 18.0, *)
public extension MLTensor {
    
    func toMultiArray() async throws(Errors) -> MLMultiArray {
        switch true {
        case self.scalarType == Float16.self:
            await MLMultiArray(self.shapedArray(of: Float16.self))
        default: throw .todo()
        }
    }
}
