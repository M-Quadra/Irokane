//
//  IrokaneTests.swift
//  Irokane
//
//  Created by m_quadra on 2024/7/11.
//

@testable import Irokane
import CoreML

#if DEBUG
@available(iOS 18.0, *)
public extension MLTensor { // beta3 bug?
    var ik: Wrapper<MLTensor> {
        Wrapper(base: self)
    }
}
#endif
