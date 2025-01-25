//
//  Compare.swift
//  Irokane
//
//  Created by m_quadra on 2025/1/25.
//

import Testing

package func compare(_ srcArr: [Float32], _ dstArr: [Float32], tolerance: Float32 = 1e-4) {
    #expect(srcArr.count == dstArr.count)
    for (i, (src, dst)) in zip(srcArr, dstArr).enumerated() {
        #expect(abs(dst - src) < tolerance, "arr[\(i)]: src: \(src) dst: \(dst)")
    }
}
