//
//  Float32s+IK.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/12.
//

import Accelerate

infix operator **

extension [Float32] {
    
    static func ** (lhs: [Float32], rhs: UInt) -> [Float32] {
        var lhs = lhs, arr = [Float32](repeating: 1, count: lhs.count), n = rhs
        while n > 0 {
            if n & 1 == 1 {
                vDSP_vmul(lhs, 1, arr, 1, &arr, 1, vDSP_Length(lhs.count))
            }
            n >>= 1
            vDSP_vsq(lhs, 1, &lhs, 1, vDSP_Length(lhs.count))
        }
        return arr
    }
    
    static func - (lhs: [Float32], rhs: Float32) -> [Float32] {
        var lhs = lhs
        vDSP_vsadd(lhs, 1, [-rhs], &lhs, 1, vDSP_Length(lhs.count))
        return lhs
    }
}

extension [Float32] {
    
    var mean: Float32 {
        var mean: Float32 = 0
        vDSP_meanv(self, 1, &mean, vDSP_Length(self.count))
        return mean
    }
    
    var std: Float32 {
        var mean: Float32 = 0
        var stddev: Float32 = 0
        vDSP_normalize(self, 1, nil, 1, &mean, &stddev, vDSP_Length(self.count))
        return stddev
    }
    
    var sum: Float32 {
        var sum: Float32 = 0
        vDSP_sve(self, 1, &sum, vDSP_Length(self.count))
        return sum
    }
}
