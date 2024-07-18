//
//  Graph+SetItem.swift
//
//
//  Created by m_quadra on 2024/7/4.
//

enum Subscript {
    case lastIndex(i: Int)
}

public extension Graph.Tensor { struct Sub: ~Copyable {
    let base: Graph.Tensor
    let sub: Subscript
}}

infix operator .=

public extension Graph.Tensor.Sub {
    
    static func .= (lhs: borrowing Graph.Tensor.Sub, rhs: Double) {
        switch lhs.sub {
        case .lastIndex(let i):
            lhs.base.setLast(index: i, with: rhs)
        }
    }
}

public extension Graph.Tensor {
    
    // x[..., i]
    subscript(_: (UnboundedRange_) -> (), index: Int) -> Graph.Tensor.Sub {
        .init(base: self, sub: .lastIndex(i: index))
    }
}

extension Graph.Tensor {
    
    // x[..., i] .= a
    func setLast(index: Int, with a: Double) {
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
        let oneHot = graph.cast(
            graph.oneHot(withIndicesTensor: i, depth: len, name: nil),
            to: .bool, name: nil
        )
        let a = graph.multiplication(
            oneHot, graph.constant(a, dataType: x.dataType),
            name: nil
        )
        
        let ts0 = graph.multiplication(
            x, graph.logicalNOR(oneHot, oneHot, name: nil),
            name: nil
        )
        let y = graph.addition(ts0, a, name: nil)
        self.tensor = y
        
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
}

public extension Graph.Tensor {
        
    func setItem(at mask: borrowing Graph.Tensor, _ constant: Double) -> Graph.Tensor {
        let graph = self.graph.graph, x = self.tensor
        assert(graph == mask.graph.graph)
        assert(x.shape != nil)
        assert(x.shape == mask.tensor.shape)
        
        let m = graph.cast(mask.tensor, to: .bool, name: nil)
        let n = graph.logicalNOR(m, m, name: nil)
        
        let x0 = graph.multiplication(
            x, graph.cast(n, to: x.dataType, name: nil),
            name: nil
        )
        
        let constant = graph.constant(constant, dataType: x.dataType)
        let v = graph.multiplication(
            graph.cast(m, to: x.dataType, name: nil), constant,
            name: nil
        )
        
        let y = graph.addition(x0, v, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
    
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
    
    /// x[..., i] += a
    func addItem(at range: (_: (UnboundedRange_) -> (), index: Int), _ a: Double) -> Graph.Tensor {
        let graph = self.graph.graph, x = self.tensor
        guard let len = x.shape?.last?.intValue else {
            assertionFailure("shape error")
            return Graph.Tensor(graph: self.graph, tensor: x)
        }
        let index = range.index + (range.index < 0 ? len : 0)
        guard 0 <= index, index <= len else {
            assertionFailure("index error")
            return Graph.Tensor(graph: self.graph, tensor: x)
        }
        
        let i = graph.constant(Double(index), dataType: .int32)
        let i0 = graph.oneHot(withIndicesTensor: i, depth: len, name: nil)
        let i1 = graph.cast(i0, to: x.dataType, name: nil)
        
        let a = graph.constant(Double(a), dataType: x.dataType)
        let a0 = graph.multiplication(i1, a, name: nil)
        
        let y = graph.addition(x, a0, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
}
