//
//  Reshape+Tensor.swift
//  
//
//  Created by m_quadra on 2024/6/25.
//

import Foundation

public extension Tensor {
    
    consuming func reshape(to shape: consuming [Int]) async throws -> Tensor {
        return try await Irokane.reshape(tensor: self, to: shape)
    }
    
    consuming func permute(dims: consuming [Int]) async throws -> Tensor {
        return try await Irokane.permute(tensor: self, dims: dims)
    }
}
