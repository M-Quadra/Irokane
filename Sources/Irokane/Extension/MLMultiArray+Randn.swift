//
//  MLMultiArray+Randn.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/13.
//

import CoreML
import Accelerate
import MetalPerformanceShadersGraph

public extension MLMultiArray {
    
    @available(iOS 16.0, *)
    static func randnFP16(shape: consuming [NSNumber]) throws -> MLMultiArray {
        return try .BNNS.fp16(shape: shape)
    }
    
    static func randnFP32(shape: consuming [NSNumber]) throws -> MLMultiArray {
        if #available(iOS 16.0, *) {
            return try .BNNS.fp32(shape: shape)
        } else if #available(iOS 15.4, *) {
            return try .MPS.fp32(shape: shape)
        }
        return try .BoxMuller.fp32(shape: shape)
    }
}

@available(iOS 16.0, *)
package extension MLMultiArray { struct BNNS {
    private init() {}
}}
@available(iOS 16.0, *)
package extension MLMultiArray.BNNS {
    static func fp32(shape: consuming [NSNumber], mean: Float32 = 0, std: Float32 = 1) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        let cnt = arr.count
        
        try arr.withUnsafeMutableBufferPointer(ofType: Float32.self) { ptr, strides in
            guard var des = BNNSNDArrayDescriptor(data: ptr, shape: .vector(cnt)),
                  let gen = BNNSCreateRandomGenerator(BNNSRandomGeneratorMethodAES_CTR, nil)
            else { throw Errors.msg("BNNS failed") }
            BNNSRandomFillNormalFloat(gen, &des, mean, std)
            BNNSDestroyRandomGenerator(gen)
        }
        return arr
    }
    
    static func fp16(shape: consuming [NSNumber], mean: Float32 = 0, std: Float32 = 1) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        let cnt = arr.count
        
        _ = [Float32](unsafeUninitializedCapacity: cnt) { buffer, initializedCount in
            guard var des = BNNSNDArrayDescriptor(data: buffer, shape: .vector(cnt)),
                  let gen = BNNSCreateRandomGenerator(BNNSRandomGeneratorMethodAES_CTR, nil)
            else { fatalError() }
            BNNSRandomFillNormalFloat(gen, &des, mean, std)
            BNNSDestroyRandomGenerator(gen)
            initializedCount = cnt
            
            arr.withUnsafeMutableBufferPointer(ofType: Float16.self) { ptr, strides in
                var src = vImage_Buffer(data: buffer.baseAddress, height: 1, width: vImagePixelCount(cnt), rowBytes: cnt * MemoryLayout<Float32>.size)
                var dst = vImage_Buffer(data: ptr.baseAddress, height: 1, width: vImagePixelCount(cnt), rowBytes: cnt * MemoryLayout<Float16>.size)
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
            }
        }
        return arr
    }
}

@available(iOS 15.4, *)
package extension MLMultiArray { struct MPS {
    private init() {}
}}
@available(iOS 15.4, *)
package extension MLMultiArray.MPS {
    static func fp32(shape: [NSNumber], seed: Int = .random(in: 0..<Int.max)) throws -> MLMultiArray {
        guard let op = MPSGraphRandomOpDescriptor(distribution: .normal, dataType: .float32) else { fatalError() }
        op.samplingMethod = .boxMuller
        let graph = MPSGraph()
        
        let y = graph.randomTensor(withShape: shape, descriptor: consume op, seed: seed, name: nil)
        
        guard let yData = graph.run(
            feeds: [:],
            targetTensors: [y],
            targetOperations: nil
        )[consume y] else { throw Errors.msg("graph.run failed") }
        
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        let ok = arr.withUnsafeMutableBytes { ptr, _ in
            guard let dst = ptr.baseAddress else { return false }
            yData.mpsndarray().readBytes(dst, strideBytes: nil)
            return true
        }
        if !ok { throw Errors.msg("readBytes failed") }
        return arr
    }
}

package extension MLMultiArray { struct BoxMuller {
    private init() {}
}}
package extension MLMultiArray.BoxMuller {
    static func fp32(shape: consuming [NSNumber]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        let cnt = arr.count
        
        let mean: Float = 0.0, std: Float = 1.0
        var arr16 = (0..<(cnt/16 + 1) * 16).map { _ in Float32.random(in: 0..<1) }
        for i in stride(from: 0, to: arr16.count, by: 16) {
            for j in i..<(i+8) {
                let u1 = 1 - arr16[j]
                let u2 = arr16[j + 8]
                let radius = sqrt(-2 * log(u1))
                let theta = 2 * Float.pi * u2
                
                arr16[j] = radius * cos(theta) * std + mean
                arr16[j+8] = radius * sin(theta) * std + mean
            }
        }
        
        for i in 0..<cnt {
            arr[i] = arr16[i] as NSNumber
        }
        return arr
    }
}
