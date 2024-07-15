//
//  Graph.swift
//
//
//  Created by m_quadra on 2024/6/27.
//

import CoreML
import MetalPerformanceShadersGraph

public struct Graph {
    public let tensor: MPSGraphTensor
    public let graph: MPSGraph
}

public extension Graph {
    
    var shape: [NSNumber] { self.tensor.shape ?? [] }
    
    func cast(to type: consuming MPSDataType) -> Graph {
        let y = self.graph.cast(self.tensor, to: type, name: nil)
        return Graph(tensor: consume y, graph: self.graph)
    }
    
    func reshape(_ shape: consuming [NSNumber]) -> Graph {
        let y = self.graph.reshape(self.tensor, shape: shape, name: nil)
        return Graph(tensor: consume y, graph: self.graph)
    }
    
    func permute(_ dims: consuming [NSNumber]) -> Graph {
        let y = self.graph.transpose(self.tensor, permutation: dims, name: nil)
        return Graph(tensor: consume y, graph: self.graph)
    }
    
    func unsqueeze(_ dim: Int) -> Graph {
        let y = self.graph.expandDims(self.tensor, axis: dim, name: nil)
        return Graph(tensor: consume y, graph: self.graph)
    }
    
    borrowing func gather(dim: Int, index: Graph) -> Graph {
        let graph = self.graph, x = self.tensor
        assert(graph == index.graph)
        
        let y = graph.gatherAlongAxis(dim, updates: consume x, indices: index.tensor, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    
    /// x.pow(a)
    borrowing func pow(_ exponent: Double) -> Graph {
        let graph = self.graph, x = self.tensor
        let a = graph.constant(exponent, dataType: x.dataType)
        
        let y = graph.power(consume x, consume a, name: nil)
        return Graph(tensor: consume y, graph: self.graph)
    }
}
