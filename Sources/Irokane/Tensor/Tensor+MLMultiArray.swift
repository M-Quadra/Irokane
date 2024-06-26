//
//  Tensor+MLMultiArray.swift
//
//
//  Created by m_quadra on 2024/6/25.
//

import CoreML
import MetalPerformanceShadersGraph

extension MLMultiArray: Tensorable {
    
    func toTensor() -> Tensor {
        return Tensor(base: self)
    }
    
    public consuming func toMPS(
        graph: consuming MPSGraph
    ) async throws(Errors) -> (tensor: MPSGraphTensor, data: MPSGraphTensorData) {
        let data = try await self.toTensorData()
        let ts = graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        return (ts, data)
    }
}

// MARK: - Private
fileprivate extension MLMultiArray {
    
    func toTensorData() async throws(Errors) -> MPSGraphTensorData {
        let (len, dtype): (Int, MPSDataType) = switch self.dataType {
        case .float16: (MemoryLayout<Float16>.size * self.count, .float16)
        case .float32: (MemoryLayout<Float32>.size * self.count, .float32)
        default: throw .todo()
        }
        
        let buf: MTLBuffer? = self.withUnsafeBytes { ptr in
            guard let dst = ptr.baseAddress,
                  let device = MTLCreateSystemDefaultDevice()
            else { return nil }
            return device.makeBuffer(bytes: dst, length: len)
        }
        guard let buf = consume buf else { throw .msg("makeBuffer failed, length:\(len)") }
        
        return MPSGraphTensorData(consume buf, shape: self.shape, dataType: dtype)
    }
}
