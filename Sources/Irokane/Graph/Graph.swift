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
        let ts = self.graph.cast(self.tensor, to: type, name: nil)
        return Graph(tensor: consume ts, graph: self.graph)
    }
    
    func reshape(_ shape: consuming [NSNumber]) -> Graph {
        let ts = self.graph.reshape(self.tensor, shape: shape, name: nil)
        return Graph(tensor: consume ts, graph: self.graph)
    }
    
    func permute(_ dims: consuming [NSNumber]) -> Graph {
        let ts = self.graph.transpose(self.tensor, permutation: dims, name: nil)
        return Graph(tensor: consume ts, graph: self.graph)
    }
    
    func unsqueeze(_ dim: Int) -> Graph {
        let ts = self.graph.expandDims(self.tensor, axis: dim, name: nil)
        return Graph(tensor: consume ts, graph: self.graph)
    }
}
