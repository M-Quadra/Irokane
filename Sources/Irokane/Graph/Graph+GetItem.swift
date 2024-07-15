//
//  Graph+GetItem.swift
//
//
//  Created by m_quadra on 2024/7/5.
//

import Foundation

public enum Mark {}

public extension Graph {
    
    // x[..., ..<i]
    subscript(_: (UnboundedRange_) -> (), range: PartialRangeUpTo<Int>) -> Graph {
        let graph = self.graph, x = self.tensor
        var len = range.upperBound
        if len < 0 {
            len += x.shape?.last?.intValue ?? 0
        }
        
        let y = graph.sliceTensor(self.tensor, dimension: -1, start: 0, length: len, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
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
    
    // x[..., i]
    subscript(_: (UnboundedRange_) -> (), index: Int) -> Graph {
        let graph = self.graph, x = self.tensor
        
        let x0 = graph.sliceTensor(consume x, dimension: -1, start: index, length: 1, name: nil)
        let y = graph.squeeze(consume x0, axis: -1, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
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
    
    subscript(mask: Graph, _: (UnboundedRange_) -> ()) -> Graph {
        let graph = self.graph, x = self.tensor
        assert(graph == mask.graph)
        
        let m = graph.cast(mask.tensor, to: .bool, name: nil)
        let i = graph.nonZeroIndices(m, name: nil)
        
        let y = graph.gatherND(withUpdatesTensor: x, indicesTensor: i, batchDimensions: 0, name: nil)
        return Graph(tensor: consume y, graph: graph)
    }
    
    /// x[..., nil]
    subscript(_: (UnboundedRange_) -> (), mark: Mark?) -> Graph {
        let graph = self.graph, x = self.tensor
        let y = graph.expandDims(x, axis: -1, name: nil)
        return Graph(tensor: consume y, graph: graph)
    }
}
