//
//  ReshapeTests.swift
//
//
//  Created by m_quadra on 2024/6/25.
//

import Testing
import Irokane
import CoreML

struct ReshapeTests {
    
    @available(iOS 16.0, *)
    @Test("reshape MLMultiArray")
    func reshapeMLMultiArray() async throws {
        let x = try MLMultiArray(shape: [1, 29, 83], dataType: .float16)
            .ik.toTensor()
        let y = try await x.reshape(to: [1, 1, -1, 83])
        
        #expect(y.shape == [1, 1, 29, 83])
    }
    
    @available(iOS 18.0, *)
    @Test("reshape MLTensor")
    func reshapeMLTensor() async throws {
        let x = MLTensor(zeros: [1, 29, 83], scalarType: Float16.self)
            .ik.toTensor()
        let y = try await x.reshape(to: [1, 1, -1, 83])
        
        #expect(y.shape == [1, 1, 29, 83])
    }
    
    @available(iOS 16.0, *)
    @Test("permute MLMultiArray", arguments: 1..<10)
    func permuteMLMultiArray(len: Int) async throws {
        let shape = (1...len).map { Int.random(in: $0..<10) }
        let dims = Array(0..<len).suffix(len)
        
        let x = try MLMultiArray(shape: shape as [NSNumber], dataType: .float16)
            .ik.toTensor()
        let y = try await x.permute(dims: Array(dims))
        
        #expect(y.shape == dims.map { shape[$0] })
    }
    
    @available(iOS 18.0, *)
    @Test("permute MLTensor", arguments: 1..<10)
    func permuteMLTensor(len: Int) async throws {
        let shape = (1...len).map { Int.random(in: $0..<10) }
        let dims = Array(0..<len).suffix(len)
        
        let ts = MLTensor(zeros: shape, scalarType: Float16.self)
        print("scalarCount:", String(ts.scalarCount, radix: 16, uppercase: true))
        let x = ts.ik.toTensor()
        let y = try await x.permute(dims: Array(dims))
        
        #expect(y.shape == dims.map { shape[$0] })
    }
}
