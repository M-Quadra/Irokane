//
//  MPSDataType+IK.swift
//
//
//  Created by m_quadra on 2024/6/29.
//

import MetalPerformanceShaders

#if DEBUG
extension MPSDataType: @retroactive CustomStringConvertible {
    
    public var description: String { switch self {
    case .invalid: "MPSDataType.invalid"
        
    case .floatBit: "MPSDataType.floatBit"
    case .float32: "MPSDataType.float32"
    case .float16: "MPSDataType.float16"
        
    case .complexBit: "MPSDataType.complexBit"
    case .complexFloat16: "MPSDataType.complexFloat16"
    case .complexFloat32: "MPSDataType.complexFloat32"
        
    case .signedBit: "MPSDataType.signedBit"
    case .int8: "MPSDataType.int8"
    case .int16: "MPSDataType.int16"
    case .int32: "MPSDataType.int32"
    case .int64: "MPSDataType.int64"
        
    case .uInt8: "MPSDataType.uInt8"
    case .uInt16: "MPSDataType.uInt16"
    case .uInt32: "MPSDataType.uInt32"
    case .uInt64: "MPSDataType.uInt64"
        
    case .alternateEncodingBit: "MPSDataType.alternateEncodingBit"
    case .bool: "MPSDataType.bool"
    case .bFloat16: "MPSDataType.bFloat16"
        
    case .normalizedBit: "MPSDataType.normalizedBit"
    case .unorm1: "MPSDataType.unorm1"
    case .unorm8: "MPSDataType.unorm8"
        
    default: "MPSDataType(rawValue: \(self.rawValue)"
    }}
}
#endif
