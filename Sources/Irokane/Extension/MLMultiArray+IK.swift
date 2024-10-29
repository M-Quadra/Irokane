//
//  MLMultiArray+IK.swift
//
//
//  Created by m_quadra on 2024/6/23.
//

import CoreML
import MetalPerformanceShadersGraph
import Accelerate

public extension MLMultiArray {
    var ik: Wrapper<MLMultiArray> { Wrapper(base: self) }
}

@available(iOS 15.4, *)
public extension Wrapper<MLMultiArray> {
    
    consuming func to(graph: Graph) throws(Errors) -> Graph.Tensor {
        let data = try self.base.toTensorData()
        let x = graph.graph.placeholder(shape: data.shape, dataType: data.dataType, name: nil)
        
        graph.feeds[x] = consume data
        return Graph.Tensor(graph: graph, tensor: consume x)
    }
}

@available(iOS 15.4, *)
extension MLMultiArray {
    
    func toTensorData() throws(Errors) -> MPSGraphTensorData {
        if self.dataType == .float64 {
            return try self.fp64ToTensorData()
        }
        
        let (msize, dtype): (Int, MPSDataType) = switch self.dataType {
        case .float16: (MemoryLayout<Float16>.size, .float16)
        case .float32: (MemoryLayout<Float32>.size, .float32)
        case .int32: (MemoryLayout<Int32>.size, .int32)
        default: throw .todo("\(self.dataType)")
        }
        let len = self.count * msize
        
        let buf: MTLBuffer? = self.withUnsafeBytes { ptr in
            guard let dst = ptr.baseAddress,
                  let device = MTLCreateSystemDefaultDevice()
            else { return nil }
            return device.makeBuffer(bytes: dst, length: len)
        }
        guard let buf = consume buf else { throw .msg("makeBuffer failed, length:\(len)") }
        
        return MPSGraphTensorData(consume buf, shape: self.shape, dataType: dtype)
    }
    
    func toFloat32s() throws(Errors) -> [Float32] {
        if self.dataType != .float32 { throw .msg("\(self.dataType)") }
        
        let cnt = self.count
        let dst = [Float32](unsafeUninitializedCapacity: cnt) { buf, cnt in
            cnt = self.withUnsafeBufferPointer(ofType: Float32.self) { ptr in
                guard let src = ptr.baseAddress,
                      let dst = buf.baseAddress
                else { return 0 }
                dst.initialize(from: src, count: ptr.count)
                return ptr.count
            }
        }
        guard dst.count == cnt else { throw .msg("copy failed") }
        return dst
    }
    
    @available(iOS 16.0, *)
    func toFloat16s() throws(Errors) -> [Float16] {
        if self.dataType != .float16 { throw .msg("\(self.dataType)") }
        
        let cnt = self.count
        let dst = [Float16](unsafeUninitializedCapacity: cnt) { buf, cnt in
            cnt = self.withUnsafeBufferPointer(ofType: Float16.self) { ptr in
                guard let src = ptr.baseAddress,
                      let dst = buf.baseAddress
                else { return 0 }
                dst.initialize(from: src, count: ptr.count)
                return ptr.count
            }
        }
        guard dst.count == cnt else { throw .msg("copy failed") }
        return dst
    }
}

// MARK: - Private
@available(iOS 15.4, *)
fileprivate extension MLMultiArray {
    
    func fp64ToTensorData() throws(Errors) -> MPSGraphTensorData {
        let cnt = self.count, msize = MemoryLayout<Float32>.size
        assert(self.dataType == .float64)
        
        let len = msize * cnt
        guard let device = MTLCreateSystemDefaultDevice(),
              let buf = device.makeBuffer(length: len)
        else { throw .msg("makeBuffer failed, length:\(len)") }
        
        let dst = buf.contents().bindMemory(to: Float32.self, capacity: cnt)
        let ok = self.withUnsafeBufferPointer(ofType: Float64.self) { ptr in
            guard let src = ptr.baseAddress else { return false }
            vDSP_vdpsp(src, 1, dst, 1, vDSP_Length(cnt))
            return true
        }
        if !ok { throw .msg("ptr.baseAddress") }
        
        return MPSGraphTensorData(consume buf, shape: self.shape, dataType: .float32)
    }
}
