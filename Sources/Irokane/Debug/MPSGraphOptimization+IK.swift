//
//  MPSGraphOptimization+IK.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/2.
//

import MetalPerformanceShadersGraph

#if DEBUG
extension MPSGraphOptimization: @retroactive CustomStringConvertible {
    public var description: String { switch self {
    case .level0: "MPSGraphOptimization.level0"
    case .level1: "MPSGraphOptimization.level1"
    @unknown default: "MPSGraphOptimization(rawValue: \(self.rawValue))"
    }}
}
#endif
