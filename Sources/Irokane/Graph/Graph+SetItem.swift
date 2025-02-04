//
//  Graph+SetItem.swift
//
//
//  Created by m_quadra on 2024/7/4.
//

import MetalPerformanceShadersGraph

@available(iOS 14.0, *)
enum Subscript {
    case lastIndex(i: Int)
    case mask(m: MPSGraphTensor)
}

@available(iOS 14.0, *)
public extension Graph.Tensor { struct Sub: ~Copyable {
    let base: UnsafeMutablePointer<Graph.Tensor>
    let sub: Subscript
#if !DEBUG // compiler bug?
    init(base: consuming UnsafeMutablePointer<Graph.Tensor>, sub: consuming Subscript) {
        self.base = base
        self.sub = sub
        self.base.pointee.tensor.isKind(of: MPSGraphTensor.self)
    }
#endif
}}

// replace `=`
infix operator .=

@available(iOS 15.0, *)
public extension Graph.Tensor.Sub {
    /*
     the result of subscript can't consume directly.
     eg. x[..., 0] .= 1
     need: let i = x[..., 0]; i .= 1
     
     for using `x[..., 0] .= 1` style, chose `borrowing Graph.Tensor.Sub`
     */
    
    static func .= (lhs: borrowing Graph.Tensor.Sub, rhs: Double) {
        switch lhs.sub {
        case .lastIndex(let i):
            lhs.base.pointee.setLast(index: i, with: rhs)
        case .mask(let m):
            lhs.base.pointee.setBy(mask: m, with: rhs)
        }
    }
    
    @available(iOS 17.0, *)
    static func .= (lhs: borrowing Graph.Tensor.Sub, rhs: Graph.Tensor) {
        switch lhs.sub {
        case .mask(let m):
            lhs.base.pointee.setBy(mask: m, with: rhs.tensor)
        default: assertionFailure("TODO")
        }
    }
    
    static func += (lhs: borrowing Graph.Tensor.Sub, rhs: Double) {
        switch lhs.sub {
        case .lastIndex(let i):
            lhs.base.pointee.addLast(index: i, with: rhs)
        default: assertionFailure("TODO")
        }
    }
}

@available(iOS 14.0, *)
public extension Graph.Tensor {
    
    // x[..., i]
    subscript(_: (UnboundedRange_) -> (), index: Int) -> Graph.Tensor.Sub { mutating get {
        let ptr = withUnsafeMutablePointer(to: &self) { $0 }
        return .init(base: ptr, sub: .lastIndex(i: index))
    }}
    
    // x[mask]
    subscript(mask: Graph.Tensor) -> Graph.Tensor.Sub { mutating get {
        assert(self.graph === mask.graph)
        let ptr = withUnsafeMutablePointer(to: &self) { $0 }
        return .init(base: ptr, sub: .mask(m: mask.tensor))
    }}
}

@available(iOS 15.0, *)
fileprivate extension Graph.Tensor {
    
    /// x[..., i] .= a
    mutating func setLast(index: Int, with a: Double) {
        let graph = self.graph.mpsGraph, x = self.tensor
        guard let len = x.shape?.last?.intValue else {
            assertionFailure("shape error")
            return
        }
        let index = index + (index < 0 ? len : 0)
        guard 0 <= index, index < len else {
            assertionFailure("index error")
            return
        }
        
        let i = graph.constant(Double(index), dataType: .int32)
        let i0 = graph.oneHot(withIndicesTensor: consume i, depth: len, name: nil)
        let i1 = graph.cast(consume i0, to: .bool, name: nil)
        let i2 = graph.logicalNOR(i1, i1, name: nil)
        
        let a = graph.constant(a, dataType: x.dataType)
        let a0 = graph.multiplication(consume i1, consume a, name: nil)
        
        let x0 = graph.multiplication(consume x, consume i2, name: nil)
        let y = graph.addition(consume x0, consume a0, name: nil)
        self.tensor = consume y
        
        // TODO: branch mark with split+concat
        //            var arr = graph.split(src, numSplits: len, axis: -1, name: nil)
        //            arr[index] = graph.multiplication(
        //                arr[index], graph.constant(0, dataType: src.dataType),
        //                name: nil
        //            )
        //            arr[index] = graph.addition(
        //                arr[index], graph.constant(constant, dataType: src.dataType),
        //                name: nil
        //            )
        //            let ts = graph.concatTensors(arr, dimension: -1, name: nil)
    }
    
    /// x[mask] .= a
    mutating func setBy(mask: MPSGraphTensor, with a: Double) {
        let graph = self.graph.mpsGraph, x = self.tensor
        assert(x.shape != nil)
        assert(x.shape == mask.shape)
        
        let m = graph.cast(mask, to: .bool, name: nil)
        let m0 = graph.logicalNOR(m, m, name: nil)
        let m1 = graph.cast(consume m0, to: x.dataType, name: nil)
        
        let a = graph.constant(a, dataType: x.dataType)
        let m2 = graph.cast(consume m, to: x.dataType, name: nil)
        let a0 = graph.multiplication(consume m2, consume a, name: nil)
        
        let x0 = graph.multiplication(consume x, consume m1, name: nil)
        let y = graph.addition(consume x0, consume a0, name: nil)
        self.tensor = consume y
    }
    
    /// x[mask] .= y
    @available(iOS 17.0, *)
    mutating func setBy(mask: MPSGraphTensor, with y: MPSGraphTensor) {
        let mpsGraph = self.graph.mpsGraph, x = self.tensor
        assert(mpsGraph == mask.operation.graph)
        assert(mask.operation.graph == y.operation.graph)
        assert(x.shape != nil)
        assert(x.shape == mask.shape)
        
        let m = mask.dataType == .bool ? mask : mpsGraph.cast(mask, to: .bool, name: nil)
        
#if DEBUG && targetEnvironment(simulator)
        // TODO: check Metal support
        if let path = ProcessInfo.processInfo.environment["XCTestBundlePath"],
           path.hasSuffix("/RealDeviceTests.xctest") {
            let cond = mpsGraph.reductionOr(with: m, axes: nil, name: nil)
            let ys = mpsGraph.if(consume cond, then: {
                let i = mpsGraph.nonZeroIndices(m, name: nil)
                let y = mpsGraph.scatterNDWithData(x, updates: consume y, indices: consume i, batchDimensions: 0, mode: .set, name: nil)
                return [y]
            }, else: {
                return [x]
            }, name: nil)
            self.tensor = ys[0]
        } else {
            let i = mpsGraph.nonZeroIndices(m, name: nil)
            let y = mpsGraph.scatterNDWithData(consume x, updates: y, indices: consume i, batchDimensions: 0, mode: .set, name: nil)
            self.tensor = consume y
        }
#elseif targetEnvironment(simulator)
        let i = mpsGraph.nonZeroIndices(m, name: nil)
        let y = mpsGraph.scatterNDWithData(consume x, updates: y, indices: consume i, batchDimensions: 0, mode: .set, name: nil)
        self.tensor = consume y
#else
        let cond = mpsGraph.reductionOr(with: m, axes: nil, name: nil)
        let ys = mpsGraph.if(consume cond, then: {
            let i = mpsGraph.nonZeroIndices(m, name: nil)
            let y = mpsGraph.scatterNDWithData(x, updates: consume y, indices: consume i, batchDimensions: 0, mode: .set, name: nil)
            return [y]
        }, else: {
            return [x]
        }, name: nil)
        self.tensor = ys[0]
#endif
        self.graph.fillTensors.insert(self.tensor)
    }
    
    /// x[..., i] += a
    mutating func addLast(index: Int, with a: Double) {
        let graph = self.graph.mpsGraph, x = self.tensor
        guard let len = x.shape?.last?.intValue else {
            assertionFailure("shape error")
            return
        }
        let index = index + (index < 0 ? len : 0)
        guard 0 <= index, index < len else {
            assertionFailure("index error")
            return
        }
        
        let i = graph.constant(Double(index), dataType: .int32)
        let i0 = graph.oneHot(withIndicesTensor: consume i, depth: len, name: nil)
        let i1 = graph.cast(consume i0, to: x.dataType, name: nil)
        
        let a = graph.constant(Double(a), dataType: x.dataType)
        let a0 = graph.multiplication(consume i1, consume a, name: nil)
        
        let y = graph.addition(consume x, consume a0, name: nil)
        self.tensor = consume y
    }
}
