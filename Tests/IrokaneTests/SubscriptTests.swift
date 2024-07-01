//
//  SubscriptTests.swift
//  
//
//  Created by m_quadra on 2024/6/30.
//

import Testing
import Irokane
import CoreML

struct SubscriptTests {
    
    @Test("..<a MLMultiArray")
    func partialRangeUpToMLMultiArray() async throws {
        let x = try MLMultiArray(shape: [1, 1, 83, 29], dataType: .float16)
            .ik.toTensor()
        let y = try await x[..., ..<10].value
        
        #expect(y.shape == [1, 1, 83, 10])
    }
    
    @available(iOS 18.0, *)
    @Test("..<a MLTensor")
    func partialRangeUpToMLTensor() async throws {
        let x = MLTensor(repeating: 0, shape: [1, 1, 83, 29])
            .ik.toTensor()
        let y = try await x[..., ..<10].value
        
        #expect(y.shape == [1, 1, 83, 10])
    }
}
