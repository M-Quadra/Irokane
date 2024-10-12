//
//  SubscriptTests.swift
//
//
//  Created by m_quadra on 2024/6/30.
//

import Testing
@testable import Irokane
import CoreML

struct SubscriptTests {
    
    @available(iOS 16.0, *)
    @Test("..<a MLMultiArray")
    func partialRangeUpToMLMultiArray() async throws {
        let x = try MLMultiArray(shape: [1, 1, 83, 29], dataType: .float16)
            .ik.toTensor()
        let y = try await x.getItem(..., ..<10)
        
        #expect(y.shape == [1, 1, 83, 10])
    }
    
    @available(iOS 18.0, *)
    @Test("..<a MLTensor")
    func partialRangeUpToMLTensor() async throws {
        let x = MLTensor(repeating: 0, shape: [1, 1, 83, 29])
            .ik.toTensor()
        let y = try await x.getItem(..., ..<10)
        
        #expect(y.shape == [1, 1, 83, 10])
    }
}
