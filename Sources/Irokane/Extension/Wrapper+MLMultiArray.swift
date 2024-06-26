//
//  Wrapper+MLMultiArray.swift
//  
//
//  Created by m_quadra on 2024/6/26.
//

import CoreML

public extension MLMultiArray {
    var ik: Wrapper<MLMultiArray> { Wrapper(base: self) }
}

public extension Wrapper where Base == MLMultiArray {
    
    consuming func toTensor() -> Tensor {
        return Tensor(base: self.base)
    }
}
