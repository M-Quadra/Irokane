//
//  RandnTests.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/13.
//

import Testing
import Foundation
@testable import Irokane
import CoreML

fileprivate func jarqueBera(_ arr: [Float]) -> Float {
    // https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test
    let n = arr.count
    let mean = arr.mean
    
    let p0 = arr - mean
    let skew = (p0 ** 3).sum / Float(n) / pow((p0 ** 2).sum/Float(n), 3/2)
    let kurt = (p0 ** 4).sum / Float(n) / pow((p0 ** 2).sum/Float(n), 2)
    return Float(n)/6 * (pow(skew, 2) + pow(kurt-3, 2) / 4)
}

@Suite("MLMultiArray Randn")
struct RandnTests {
    
    @available(iOS 16.0, *)
    @Test("Jarque-Bera test")
    func jarqueBeraTest() throws {
        let shape: [NSNumber] = [2_000]
        let arr = try [
            "BNNS": jarqueBera(MLMultiArray.BNNS.fp32(shape: shape).toFloat32s()),
            "MPS": jarqueBera(MLMultiArray.MPS.fp32(shape: shape).toFloat32s()),
            "BoxMuller": jarqueBera(MLMultiArray.BoxMuller.fp32(shape: shape).toFloat32s()),
        ].sorted { $0.value < $1.value }
        print(arr)
    }
    
    @available(iOS 16.0, *)
    @Test("BNNS")
    func bnns() throws {
        let arrFP32 = try MLMultiArray.BNNS.fp32(shape: [2_000])
            .toFloat32s()
        let tFP32 = jarqueBera(arrFP32)
        #expect(tFP32 < 10)
        
        let arrFP16 = try MLMultiArray.BNNS.fp16(shape: [2_000])
            .toFloat16s()
        let tFP16 = jarqueBera(arrFP16.toFloat32s())
        #expect(tFP16 < 10)
    }
    
    @available(iOS 15.4, *)
    @Test("MPS")
    func mps() throws {
        let arr = try MLMultiArray.MPS.fp32(shape: [2_000])
            .toFloat32s()
        let t = jarqueBera(arr)
        #expect(t < 10)
    }
    
    @available(iOS 15.4, *)
    @Test("Box–Muller")
    func boxMuller() throws {
        let arr = try MLMultiArray.BoxMuller.fp32(shape: [2_000])
            .toFloat32s()
        let t = jarqueBera(arr)
        #expect(t < 10)
    }
}

@Suite("MLMultiArray Randn performance")
struct RandnPerformanceTests {
    
    @available(iOS 16.0, *)
    @Test("BNNS")
    func bnns() throws {
        for i in [100, 1000, 10_000, 100_000] {
            _ = try MLMultiArray.BNNS.fp32(shape: [i as NSNumber])
        }
        
        let st = Date()
        for i in [100, 1000, 10_000, 100_000] {
            _ = try MLMultiArray.BNNS.fp32(shape: [i as NSNumber])
        }
        let sec = Date().timeIntervalSince(st)
        #expect(sec < 0.004)
    }
    
    @available(iOS 15.4, *)
    @Test("MPS")
    func mps() throws {
        for i in [100, 1000, 10_000, 100_000] {
            _ = try MLMultiArray.MPS.fp32(shape: [i as NSNumber])
        }
        
        let st = Date()
        for i in [100, 1000, 10_000, 100_000] {
            _ = try MLMultiArray.MPS.fp32(shape: [i as NSNumber])
        }
        let sec = Date().timeIntervalSince(st)
        #expect(0.01..<0.17 ~= sec)
    }
    
    @Test("Box–Muller")
    func boxMuller() throws {
        for i in [100, 1000, 10_000, 100_000] {
            _ = try MLMultiArray.BoxMuller.fp32(shape: [i as NSNumber])
        }
        
        let st = Date()
        for i in [100, 1000, 10_000, 100_000] {
            _ = try MLMultiArray.BoxMuller.fp32(shape: [i as NSNumber])
        }
        let sec = Date().timeIntervalSince(st)
        #expect(0.1..<0.9 ~= sec)
    }
}
