//
//  MLMultiArrayDataType+IK.swift
//  
//
//  Created by m_quadra on 2024/7/4.
//

import CoreML

#if DEBUG
extension MLMultiArrayDataType: @retroactive CustomStringConvertible {
    
    public var description: String { switch self {
    case .double: "MLMultiArrayDataType.double"
    case .float64: "MLMultiArrayDataType.float64"
    case .float32: "MLMultiArrayDataType.float32"
    case .float16: "MLMultiArrayDataType.float16"
    case .float: "MLMultiArrayDataType.float"
    case .int32: "MLMultiArrayDataType.int32"

    default: "MLMultiArrayDataType(rawValue: \(rawValue))"
    }}
}
#endif
