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
        
        let (msize, dtype): (Int, MPSDataType) = switch true {
        case self.scalarType == Float16.self: (MemoryLayout<Float16>.size, .float16)
        case self.scalarType == Float32.self: (MemoryLayout<Float32>.size, .float32)
        case self.scalarType == Int32.self: (MemoryLayout<Int32>.size, .int32)
        case self.scalarType == Bool.self: (MemoryLayout<Bool>.size, .bool)
        default: throw .todo("\(self.scalarType)")
        }
        let len = self.scalarCount * msize
        
        let buf: MTLBuffer? = switch dtype {
        case .float16:
            await self.shapedArray(of: Float16.self)
                .withUnsafeShapedBufferPointer { (ptr, shape, strides) in
                    guard let dst = ptr.baseAddress else { return nil }
                    return device.makeBuffer(bytes: dst, length: len)
                }
        case .float32:
            await self.shapedArray(of: Float32.self)
                .withUnsafeShapedBufferPointer { ptr, shape, strides in
                    guard let dst = ptr.baseAddress else { return nil }
                    return device.makeBuffer(bytes: dst, length: len)
                };
        case .int32:
            await self.shapedArray(of: Int32.self)
                .withUnsafeShapedBufferPointer { ptr, shape, strides in
                    guard let dst = ptr.baseAddress else { return nil }
                    return device.makeBuffer(bytes: dst, length: len)
                }
        case .bool:
            await self.cast(to: Int32.self).shapedArray(of: Int32.self)
                .withUnsafeShapedBufferPointer { ptr, shape, strides in
                    guard let dst = ptr.baseAddress else { return nil }
                    var arr = [Bool](repeating: false, count: self.scalarCount)
                    for i in 0..<self.scalarCount {
                        arr[i] = dst[i] != 0
                    }
                    return device.makeBuffer(bytes: arr, length: len)
                }
        default: throw .todo("\(self.scalarType)")
        }
        guard let buf = consume buf else { throw .msg("Failed to create MTLBuffer") }
        
        return MPSGraphTensorData(consume buf, shape: self.shape as [NSNumber], dataType: dtype)
    }
}

// MARK: - Public
@available(iOS 18.0, *)
public extension MLTensor {
    
    func toMultiArray() async throws(Errors) -> MLMultiArray {
        switch true {
        case self.scalarType == Float16.self:
            await MLMultiArray(self.shapedArray(of: Float16.self))
        case self.scalarType == Float32.self:
            await MLMultiArray(self.shapedArray(of: Float32.self))
        default: throw .todo("\(self.scalarType)")
        }
    }
}
