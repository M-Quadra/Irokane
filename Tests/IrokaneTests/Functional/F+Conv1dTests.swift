//
//  F+Conv1dTests.swift
//  
//
//  Created by m_quadra on 2024/6/24.
//

import Testing
import Irokane
import CoreML

fileprivate typealias F = Irokane.Functional

#if arch(arm64)
fileprivate let isArm64 = true
#else
fileprivate let isArm64 = false
#endif

@Suite("F.conv1d Tests", .enabled(if: isArm64))
struct Conv1dTests {
    
    @available(iOS 16.0, *)
    @Test("kernelSize", arguments: 1...3)
    func test(k: Int) async throws {
        let n: NSNumber = 1, cIn: NSNumber = 2, lIn: NSNumber = 3
        let cOut: NSNumber = 1
        
        let x = try MLMultiArray(shape: [n, cIn, lIn], dataType: .float16)
        let w = try MLMultiArray(shape: [cOut, cIn, k as NSNumber], dataType: .float16)
        for i in 0..<x.count { x[i] = 1 }
        for i in 0..<w.count { w[i] = 1 }
        
        let y = try await F.conv1d(input: x, weight: w)
        #expect(y.shape == [n, cOut, 4-k as NSNumber])
        
        for i in 0..<y.count {
            let dif = abs(y[i].floatValue - Float(k*2))
            #expect(dif <= 1e-4)
        }
    }
    
    @available(iOS 16.0, *)
    @Test("outChannels", arguments: 1...3)
    func test(cOut: Int) async throws {
        let n: NSNumber = 1, cIn: NSNumber = 2, lIn: NSNumber = 3
        let cOut = cOut as NSNumber, k: NSNumber = 1
        
        let x = try MLMultiArray(shape: [n, cIn, lIn], dataType: .float16)
        let w = try MLMultiArray(shape: [cOut, cIn, k], dataType: .float16)
        for i in 0..<x.count { x[i] = 1 }
        for i in 0..<w.count { w[i] = 1 }
        
        let y = try await F.conv1d(input: x, weight: w)
        #expect(y.shape == [n, cOut, 3])
        
        for i in 0..<y.count {
            let dif = abs(y[i].floatValue - k.floatValue*2)
            #expect(dif <= 1e-4)
        }
    }
    
    @available(iOS 16.0, *)
    @Test("bias", arguments: 1..<10, [0.0, 0.5, 1.0])
    func test(cOut: Int, bias: NSNumber) async throws {
        let n: NSNumber = 1, cIn: NSNumber = 2, lIn: NSNumber = 3
        let cOut = cOut as NSNumber
        
        let x = try MLMultiArray(shape: [n, cIn, lIn], dataType: .float16)
        let w = try MLMultiArray(shape: [cOut, cIn, 2], dataType: .float16)
        for i in 0..<x.count { x[i] = 1 }
        for i in 0..<w.count { w[i] = 1 }
        
        let b = try MLMultiArray(shape: [cOut], dataType: .float16)
        for i in 0..<b.count { b[i] = bias }
        
        let y = try await F.conv1d(input: x, weight: w, bias: b)
        #expect(y.shape == [n, cOut, 2])
        
        for i in 0..<y.count {
            let dif = abs(y[i].floatValue - (4+bias.floatValue))
            #expect(dif <= 1e-4)
        }
    }
}
