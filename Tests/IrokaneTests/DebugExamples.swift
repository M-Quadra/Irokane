//
//  DebugExamples.swift
//  Irokane
//
//  Created by m_quadra on 2025/4/30.
//

import Testing
import Irokane
import CoreML

@Suite struct DebugTests {
    
    @Test func mlMultiArray() throws {
        let x = try MLMultiArray(shape: [1, 3, 5], dataType: .float32)
        x.ik.debug()
        x[0] = 1
        x.ik.debug()
    }
}
