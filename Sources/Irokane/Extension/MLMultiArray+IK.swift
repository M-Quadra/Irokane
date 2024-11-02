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
        let arr = self.base, dataType = arr.dataType
        let mpsGraph = graph.mpsGraph
        
        switch arr.dataType {
        case .float64:
            let data: Data? = arr.withUnsafeBufferPointer(ofType: Float64.self) { ptr in
                let buf = UnsafeMutableBufferPointer<Float32>.allocate(capacity: arr.count)
                guard let src = ptr.baseAddress,
                      let dst = buf.baseAddress
                else { return nil }
                vDSP_vdpsp(src, 1, dst, 1, vDSP_Length(arr.count))
                
                let cnt = buf.count * MemoryLayout<Float32>.size
                return Data(bytesNoCopy: dst, count: cnt, deallocator: .free)
            }
            guard let data = consume data else { throw .msg("ptr.baseAddress") }
            
            let x = mpsGraph.constant(consume data, shape: arr.shape, dataType: .float32)
            let y = mpsGraph.read(consume x, name: nil)
            return Graph.Tensor(graph: graph, tensor: consume y)
        default: break
        }
        
        let dtype: MPSDataType = switch dataType {
        case .float16: .float16
        case .float32: .float32
        case .int32: .int32
        default: throw .todo("\(dataType)")
        }
        
        let data = arr.withUnsafeBytes { Data($0) }
        let x = mpsGraph.constant(consume data, shape: arr.shape, dataType: dtype)
        let y = mpsGraph.read(consume x, name: nil)
        return Graph.Tensor(graph: graph, tensor: consume y)
    }
}

@available(iOS 15.4, *)
extension MLMultiArray {
    
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
