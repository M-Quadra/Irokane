//
//  Wrapper+MLTensor.swift
//  
//
//  Created by m_quadra on 2024/6/26.
//

import CoreML

@available(iOS 18.0, *)
public extension MLTensor {
    var ik: Wrapper<MLTensor> { Wrapper(base: self) }
}

@available(iOS 18.0, *)
public extension Wrapper where Base == MLTensor {
    
    func toTensor() -> Tensor {
        return Tensor(base: self.base)
    }
}
