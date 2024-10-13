//
//  Float16s+IK.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/12.
//

import Accelerate

@available(iOS 14.0, *)
extension [Float16] {
    
    func convertFP16ToFP32(_ fp16Array: [Float16]) -> [Float32] {
        let count = fp16Array.count
        var fp32Array = [Float32](repeating: 0, count: count)
        
        fp16Array.withUnsafeBufferPointer { srcPtr in
            fp32Array.withUnsafeMutableBufferPointer { dstPtr in
                var srcBuffer = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!), height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float16>.size)
                var dstBuffer = vImage_Buffer(data: dstPtr.baseAddress, height: 1, width: vImagePixelCount(count), rowBytes: count * MemoryLayout<Float32>.size)
                
                vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0)
            }
        }
        return fp32Array
    }
    
    func toFloat32s() -> [Float32] {
        let cnt = self.count
        let arr = [Float32](unsafeUninitializedCapacity: cnt) { buffer, initializedCount in
            var arr = self
            arr.withUnsafeMutableBufferPointer { ptr in
                var src = vImage_Buffer(data: ptr.baseAddress, height: 1, width: vImagePixelCount(cnt), rowBytes: cnt * MemoryLayout<Float16>.size)
                var dst = vImage_Buffer(data: buffer.baseAddress, height: 1, width: vImagePixelCount(cnt), rowBytes: cnt * MemoryLayout<Float32>.size)
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
            }
            initializedCount = self.count
        }
        return arr
    }
}
