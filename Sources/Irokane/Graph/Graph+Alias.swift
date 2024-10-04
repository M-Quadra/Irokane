//
//  Graph+Alias.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/2.
//

import Foundation

public extension Graph.Tensor {
    
    borrowing func sum(_ dim: Int) -> Graph.Tensor {
        return Irokane.sum(self, dim: dim)
    }
    
    /// int64
    borrowing func long() -> Graph.Tensor {
        return self.cast(to: .int64)
    }
    
    borrowing func view(_ size: [NSNumber]) -> Graph.Tensor {
        return self.reshape(size)
    }
}
