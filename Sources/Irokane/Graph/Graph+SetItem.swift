//
//  Graph+SetItem.swift
//
//
//  Created by m_quadra on 2024/7/4.
//

import MetalPerformanceShadersGraph

enum Subscript {
    case lastIndex(i: Int)
    case mask(m: MPSGraphTensor)
}

public extension Graph.Tensor { struct Sub: ~Copyable {
    let base: Graph.Tensor
    let sub: Subscript
}}

// replace `=`
infix operator .=

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
            lhs.base.setLast(index: i, with: rhs)
        case .mask(let m):
            assert(m.operation.graph == lhs.base.graph.graph)
            lhs.base.setBy(mask: m, with: rhs)
        }
    }
    
    static func += (lhs: borrowing Graph.Tensor.Sub, rhs: Double) {
        switch lhs.sub {
        case .lastIndex(let i):
            lhs.base.addLast(index: i, with: rhs)
        default: assertionFailure("TODO")
        }
    }
}

public extension Graph.Tensor {
    
    // x[..., i]
    subscript(_: (UnboundedRange_) -> (), index: Int) -> Graph.Tensor.Sub {
        .init(base: self, sub: .lastIndex(i: index))
    }
    
    // x[mask]
    subscript(mask: Graph.Tensor) -> Graph.Tensor.Sub {
        assert(self.graph === mask.graph)
        return .init(base: self, sub: .mask(m: mask.tensor))
    }
}

fileprivate extension Graph.Tensor {
    
    /// x[..., i] .= a
    borrowing func setLast(index: Int, with a: Double) {
        let graph = self.graph.graph, x = self.tensor
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
    borrowing func setBy(mask: MPSGraphTensor, with a: Double) {
        let graph = self.graph.graph, x = self.tensor
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
    
    /// x[..., i] += a
    borrowing func addLast(index: Int, with a: Double) {
        let graph = self.graph.graph, x = self.tensor
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

public extension Graph.Tensor {
    
    func setItem(at mask: borrowing Graph.Tensor, _ tensor: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = self.graph.graph, x = self.tensor
        assert(graph == mask.graph.graph)
        assert(x.shape != nil)
        assert(x.shape == mask.tensor.shape)
        
        let m = graph.cast(mask.tensor, to: .bool, name: nil)
        let i = graph.nonZeroIndices(m, name: nil)
        
        let y = graph.scatterNDWithData(x, updates: tensor.tensor, indices: i, batchDimensions: 0, mode: .set, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
}
