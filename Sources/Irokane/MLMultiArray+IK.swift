//
//  MLMultiArray+IK.swift
//
//
//  Created by m_quadra on 2024/6/23.
//

import CoreML
import MetalPerformanceShadersGraph

extension MLMultiArray {
    
    func toTensorData() throws(Errors) -> MPSGraphTensorData {
        let (msize, dtype): (Int, MPSDataType) = switch self.dataType {
        case .float16: (MemoryLayout<Float16>.size, .float16)
        case .float32: (MemoryLayout<Float32>.size, .float32)
        default: throw .todo("\(self.dataType)")
        }
        let len = self.count * msize
        
        let buf: MTLBuffer? = self.withUnsafeBytes { ptr in
            guard let dst = ptr.baseAddress,
                  let device = MTLCreateSystemDefaultDevice()
            else { return nil }
            return device.makeBuffer(bytes: dst, length: len)
        }
        guard let buf = consume buf else { throw .msg("makeBuffer failed, length:\(len)") }
        
        return MPSGraphTensorData(consume buf, shape: self.shape, dataType: dtype)
    }
}
