//
//  MLMultiArray+IK.swift
//
//
//  Created by m_quadra on 2024/6/23.
//

import CoreML
import MetalPerformanceShadersGraph
import Accelerate

@available(iOS 15.4, *)
extension MLMultiArray {
    
    func toTensorData() throws(Errors) -> MPSGraphTensorData {
        if self.dataType == .float64 {
            return try self.fp64ToTensorData()
        }
        
        let (msize, dtype): (Int, MPSDataType) = switch self.dataType {
        case .float16: (MemoryLayout<Float16>.size, .float16)
        case .float32: (MemoryLayout<Float32>.size, .float32)
        case .int32: (MemoryLayout<Int32>.size, .int32)
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

// MARK: - Private
@available(iOS 15.4, *)
fileprivate extension MLMultiArray {
    
    func fp64ToTensorData() throws(Errors) -> MPSGraphTensorData {
        let cnt = self.count, msize = MemoryLayout<Float32>.size
        assert(self.dataType == .float64)
        
        let len = msize * cnt
        guard let device = MTLCreateSystemDefaultDevice(),
              let buf = device.makeBuffer(length: len)
        else { throw .msg("makeBuffer failed, length:\(len)") }
        
        let dst = buf.contents().bindMemory(to: Float32.self, capacity: cnt)
        let ok = self.withUnsafeBufferPointer(ofType: Float64.self) { ptr in
            guard let src = ptr.baseAddress else { return false }
            vDSP_vdpsp(src, 1, dst, 1, vDSP_Length(cnt))
            return true
        }
        if !ok { throw .msg("ptr.baseAddress") }
        
        return MPSGraphTensorData(consume buf, shape: self.shape, dataType: .float32)
    }
}
