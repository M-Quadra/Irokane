//
//  Tensor.swift
//  
//
//  Created by m_quadra on 2024/6/25.
//

import CoreML
import MetalPerformanceShadersGraph

protocol Tensorable {
    
    func toTensor() -> Tensor
    
    @available(iOS 14.0, *)
    func toMPS(graph: MPSGraph) async throws -> (tensor: MPSGraphTensor, data: MPSGraphTensorData)
}

public struct Tensor {
    let base: Tensorable
}

@available(iOS 14.0, *)
public extension Tensor {
    
    var shape: [Int] {
        if #available(iOS 18.0, *) {
            return switch self.base {
            case let ts as MLTensor:
                ts.shape
            case let arr as MLMultiArray:
                arr.shape.map { $0.intValue }
            case let data as MPSGraphTensorData:
                data.shape.map { $0.intValue }
            default: []
            }
        }
        
        return switch self.base {
        case let arr as MLMultiArray:
            arr.shape.map { $0.intValue }
        case let data as MPSGraphTensorData:
            data.shape.map { $0.intValue }
        default: []
        }
    }
}
