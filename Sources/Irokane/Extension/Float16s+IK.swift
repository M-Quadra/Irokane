//
//  Float16s+IK.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/12.
//

import Accelerate

@available(iOS 14.0, *)
package extension [Float16] {
    func toFloat32s() -> [Float32] {
        let cnt = self.count
        let arr = [Float32](unsafeUninitializedCapacity: cnt) { buffer, initializedCount in
            self.withUnsafeBufferPointer { ptr in
                var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: ptr.baseAddress), height: 1, width: vImagePixelCount(cnt), rowBytes: cnt * MemoryLayout<Float16>.size)
                var dst = vImage_Buffer(data: buffer.baseAddress, height: 1, width: vImagePixelCount(cnt), rowBytes: cnt * MemoryLayout<Float32>.size)
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
            }
            initializedCount = self.count
        }
        return arr
    }
}
