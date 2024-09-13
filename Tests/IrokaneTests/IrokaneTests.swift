//
//  IrokaneTests.swift
//  Irokane
//
//  Created by m_quadra on 2024/7/11.
//

import Irokane
import CoreML

#if swift(<6.1)
@available(iOS 18.0, *)
extension MLTensor {
    var ik: Wrapper<MLTensor> { Wrapper(base: self) }
}
#endif
