//
//  MLMultiArray+IK.swift
//  
//
//  Created by m_quadra on 2024/6/23.
//

import CoreML
import MetalPerformanceShadersGraph

extension MLMultiArray {
    
    var mpsDataType: MPSDataType { get throws(Errors) {
        switch self.dataType {
        case .float16: .float16
        case .float32: .float32
        default: throw .todo()
        }
    }}
    
    func toTensorData() async throws(Errors) -> MPSGraphTensorData {
        guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { throw .msg("Failed to create MTLDevice") }
        
        switch self.dataType {
        case .float16:
            let buf: MTLBuffer? = self.withUnsafeBytes { ptr in
                guard let dst = ptr.baseAddress else { return nil }
                return device.makeBuffer(bytes: consume dst, length: MemoryLayout<Float16>.size * self.count)
            }
            guard let buf = consume buf else { throw .msg("Failed to create MTLBuffer") }
            return MPSGraphTensorData(consume buf, shape: self.shape, dataType: .float16)
        case .float32:
            let buf: MTLBuffer? = self.withUnsafeBytes { ptr in
                guard let dst = ptr.baseAddress else { return nil }
                return device.makeBuffer(bytes: consume dst, length: MemoryLayout<Float32>.size * self.count)
            }
            guard let buf = consume buf else { throw .msg("Failed to create MTLBuffer") }
            return MPSGraphTensorData(consume buf, shape: self.shape, dataType: .float32)
        default: throw .todo()
        }
    }
}
