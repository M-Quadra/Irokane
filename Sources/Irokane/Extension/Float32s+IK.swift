//
//  Float32s+IK.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/12.
//

import Accelerate

infix operator **

package extension [Float32] {
    
    static func ** (lhs: [Float32], rhs: UInt) -> [Float32] {
        var lhs = lhs, arr = [Float32](repeating: 1, count: lhs.count), n = rhs
        while n > 0 {
            if n & 1 == 1 {
                vDSP.multiply(lhs, arr, result: &arr)
            }
            n >>= 1
            vDSP.square(lhs, result: &lhs)
        }
        return arr
    }
    
    static func - (lhs: [Float32], rhs: Float32) -> [Float32] {
        return vDSP.add(-rhs, lhs)
    }
    
    var mean: Float32 { vDSP.mean(self) }
    
    var std: Float32 {
        if #available(iOS 18.0, *) { return vDSP.standardDeviation(self) }
        
        var mean: Float32 = 0
        var stddev: Float32 = 0
        vDSP_normalize(self, 1, nil, 1, &mean, &stddev, vDSP_Length(self.count))
        return stddev
    }
    
    var sum: Float32 { vDSP.sum(self) }
    
    var min: Float32 { vDSP.minimum(self) }
    
    var max: Float32 { vDSP.maximum(self) }
}
