//
//  Graph.swift
//
//
//  Created by m_quadra on 2024/6/27.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 14.0, *)
public class Graph {
    public let mpsGraph: MPSGraph = MPSGraph()
    public internal(set) var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
    var fillTensors: Set<MPSGraphTensor> = []
    
    public init() {}
    
    func tensor(_ tensor: MPSGraphTensor) -> Graph.Tensor {
        return Graph.Tensor(graph: self, tensor: tensor)
    }
}

@available(iOS 14.0, *)
public extension Graph { struct Tensor {
    public let graph: Graph
    public internal(set) var tensor: MPSGraphTensor
    
    init(graph: Graph, tensor: MPSGraphTensor) {
        self.graph = graph
        self.tensor = tensor
    }
}}

@available(iOS 14.0, *)
public extension Graph.Tensor {
    
    var shape: [NSNumber] { self.tensor.shape ?? [] }
    
    @available(iOS 15.0, *)
    borrowing func cast(to type: MPSDataType) -> Graph.Tensor {
        let mpsGraph = self.graph.mpsGraph, x = self.tensor
        if x.dataType == type { return self.graph.tensor(consume x) }
        
        let y = mpsGraph.cast(consume x, to: type, name: nil)
        return self.graph.tensor(consume y)
    }
    
    borrowing func reshape(_ shape: [NSNumber]) -> Graph.Tensor {
        let graph = self.graph.mpsGraph, x = self.tensor
        
        let y = graph.reshape(consume x, shape: shape, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
    
    @available(iOS 16.0, *)
    borrowing func permute(_ dims: [NSNumber]) -> Graph.Tensor {
        let graph = self.graph.mpsGraph, x = self.tensor
        
        let y = graph.transpose(consume x, permutation: dims, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
    
    /// x.pow(a)
    borrowing func pow(_ exponent: Double) -> Graph.Tensor {
        let graph = self.graph.mpsGraph, x = self.tensor
        let a = graph.constant(exponent, dataType: x.dataType)
        
        let y = graph.power(consume x, consume a, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
    
    borrowing func transpose(_ dim0: Int, _ dim1: Int) -> Graph.Tensor {
        let graph = self.graph.mpsGraph, x = self.tensor
        
        let y = graph.transposeTensor(consume x, dimension: dim0, withDimension: dim1, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
    
    borrowing func tensorData(isFill: Bool = true) throws(Errors) -> MPSGraphTensorData {
        let arr = [self.tensor] + (isFill ? self.graph.fillTensors : [])
        guard let tData = self.graph.mpsGraph.run(
            feeds: self.graph.feeds,
            targetTensors: consume arr,
            targetOperations: nil
        )[self.tensor] else { throw Errors.msg("empty result") }
        return tData
    }
}

@available(iOS 15.4, *)
public extension Graph.Tensor {
    
    borrowing func squeeze(_ dim: Int) -> Graph.Tensor {
        let graph = self.graph.mpsGraph, x = self.tensor
        
        let y = graph.squeeze(consume x, axis: dim, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
    
    borrowing func unsqueeze(_ dim: Int) -> Graph.Tensor {
        let graph = self.graph.mpsGraph, x = self.tensor
        
        let y = graph.expandDims(consume x, axis: dim, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
    
    borrowing func gather(dim: Int, index: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = self.graph.mpsGraph, x = self.tensor
        assert(graph == index.graph.mpsGraph)
        
        let y = graph.gatherAlongAxis(dim, updates: consume x, indices: index.tensor, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
    
    borrowing func max() -> Graph.Tensor {
        let graph = self.graph.mpsGraph, x = self.tensor
        
        let x0 = graph.reductionMaximum(with: consume x, axes: nil, name: nil)
        let y = graph.squeeze(consume x0, name: nil)
        return Graph.Tensor(graph: self.graph, tensor: consume y)
    }
    
    borrowing func mean(dim: Int) -> Graph.Tensor {
        let mpsGraph = self.graph.mpsGraph, x = self.tensor
        
        let x0 = mpsGraph.mean(of: consume x, axes: [dim as NSNumber], name: nil)
        let y = mpsGraph.squeeze(consume x0, axis: dim, name: nil)
        return self.graph.tensor(consume y)
    }
}
