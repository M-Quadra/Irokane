//
//  Graph+GetItem.swift
//  
//
//  Created by m_quadra on 2024/7/5.
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
    
    subscript(mask: Graph, _: (UnboundedRange_) -> ()) -> Graph {
        let graph = self.graph, x = self.tensor
        assert(graph == mask.graph)
        
        let m = graph.cast(mask.tensor, to: .bool, name: nil)
        let i = graph.nonZeroIndices(m, name: nil)
        
        let y = graph.gatherND(withUpdatesTensor: x, indicesTensor: i, batchDimensions: 0, name: nil)
        return Graph(tensor: consume y, graph: graph)
    }
}
