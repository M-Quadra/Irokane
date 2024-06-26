//
//  Tensor+MLTensor.swift
//
//
//  Created by m_quadra on 2024/6/25.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 18.0, *)
extension MLTensor: Tensorable {
    
    func toTensor() -> Tensor {
        return Tensor(base: self)
    }
    
    func toMPS(graph: MPSGraph) async throws(Errors) -> (tensor: MPSGraphTensor, data: MPSGraphTensorData) {
        guard let device: MTLDevice = MTLCreateSystemDefaultDevice() else { throw .msg("Failed to create MTLDevice") }
        
        let (msize, dtype): (Int, MPSDataType) = switch true {
        case self.scalarType == Float16.self: (MemoryLayout<Float16>.size, .float16)
        case self.scalarType == Float32.self: (MemoryLayout<Float32>.size, .float32)
        default: throw .todo("\(self.scalarType)")
        }
        let len = self.scalarCount * msize
        
        let buf: MTLBuffer? = switch dtype {
        case .float16: 
            await self.shapedArray(of: Float16.self)
                .withUnsafeShapedBufferPointer { ptr, shape, strides in
                    guard let dst = ptr.baseAddress else { return nil }
                    return device.makeBuffer(bytes: dst, length: len)
                }
        default: throw .todo("\(self.scalarType)")
        }
        guard let buf = consume buf else { throw .msg("Failed to create MTLBuffer") }
        
        let data = MPSGraphTensorData(consume buf, shape: self.shape as [NSNumber], dataType: dtype)
        let ts = graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        return (ts, data)
    }
}
