//
//  Graph+Operator.swift
//
//
//  Created by m_quadra on 2024/7/1.
//

import Foundation

public extension Graph {
    
    subscript(_: (UnboundedRange_) -> (), range: PartialRangeUpTo<Int>) -> Graph {
        let ts = graph.sliceTensor(self.tensor, dimension: -1, start: 0, length: range.upperBound, name: nil)
        return Graph(tensor: consume ts, graph: self.graph)
    }
    
    subscript(_: (UnboundedRange_) -> (), range: Range<Int>) -> Graph {
        let ts = graph.sliceTensor(self.tensor, dimension: -1, start: range.lowerBound, length: range.count, name: nil)
        return Graph(tensor: consume ts, graph: self.graph)
    }
    
    subscript(_: (UnboundedRange_) -> (), range: PartialRangeFrom<Int>) -> Graph {
        assert(self.tensor.shape?.last?.intValue ?? 0 >= range.lowerBound)
        let len = self.tensor.shape?.last ?? 0
        let ts = graph.sliceTensor(self.tensor, dimension: -1, start: range.lowerBound, length: len.intValue - range.lowerBound, name: nil)
        return Graph(tensor: consume ts, graph: self.graph)
    }
    
    subscript(mask: Graph) -> Graph {
        let graph = self.graph, x = self.tensor
        assert(graph == mask.graph)
        assert(x.shape != nil)
        assert(x.shape == mask.tensor.shape)
        
        let cnt = mask.shape.map { $0.intValue }
            .reduce(1, *) as NSNumber
        let x0 = graph.reshape(x, shape: [cnt, 1], name: nil)
        let m0 = graph.reshape(mask.tensor, shape: [cnt, 1], name: nil)
        
        let i = graph.nonZeroIndices(m0, name: nil)
        let y = graph.gatherND(withUpdatesTensor: x0, indicesTensor: i, batchDimensions: 0, name: nil)
        return Graph(tensor: consume y, graph: graph)
    }
    
    static func / (lhs: borrowing Graph, rhs: borrowing Graph) -> Graph {
        assert(lhs.graph == rhs.graph)
        let graph = lhs.graph, lhs = lhs.tensor
        
        var rhs = rhs.tensor
        if lhs.dataType != rhs.dataType {
            rhs = graph.cast(rhs, to: lhs.dataType, name: nil)
        }
        
        let ts = graph.divisionNoNaN(lhs, rhs, name: nil)
        return Graph(tensor: ts, graph: graph)
    }
    
    prefix static func - (tensor: borrowing Graph) -> Graph {
        let ts = tensor.graph.negative(with: tensor.tensor, name: nil)
        return Graph(tensor: consume ts, graph: tensor.graph)
    }
    
    prefix static func ~ (tensor: borrowing Graph) -> Graph {
        let graph = tensor.graph, src = tensor.tensor
        assert(src.dataType == .bool)
        let rhs = graph.constant(0, dataType: src.dataType)
        
        let dst = graph.equal(consume src, consume rhs, name: nil)
        return Graph(tensor: consume dst, graph: consume graph)
    }
    
    static func >= (lhs: borrowing Graph, rhs: Double) -> Graph {
        let graph = lhs.graph, src = lhs.tensor
        let rhs = graph.constant(rhs, dataType: src.dataType)
        
        let dst = graph.greaterThanOrEqualTo(consume src, consume rhs, name: nil)
        return Graph(tensor: consume dst, graph: graph)
    }
    
    static func <= (lhs: borrowing Graph, rhs: Double) -> Graph {
        let graph = lhs.graph, src = lhs.tensor
        let rhs = graph.constant(rhs, dataType: src.dataType)
        
        let dst = graph.lessThanOrEqualTo(consume src, consume rhs, name: nil)
        return Graph(tensor: consume dst, graph: graph)
    }
    
    static func & (lhs: borrowing Graph, rhs: borrowing Graph) -> Graph {
        let graph = lhs.graph, lhs = lhs.tensor
        assert(graph == rhs.graph)
        var rhs = rhs.tensor
        if lhs.dataType != rhs.dataType {
            rhs = graph.cast(rhs, to: lhs.dataType, name: nil)
        }
        
        let ts = graph.logicalAND(consume lhs, consume rhs, name: nil)
        return Graph(tensor: consume ts, graph: consume graph)
    }
}