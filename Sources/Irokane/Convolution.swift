//
//  Convolution.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/4.
//

import MetalPerformanceShadersGraph

@available(iOS 15.4, *)
public struct Conv1d {
    
    let weight: MPSGraphTensor
    let bias: MPSGraphTensor
    
    init(weight: MPSGraphTensor, bias: MPSGraphTensor) {
        let mpsGraph = weight.operation.graph
        self.weight = mpsGraph.expandDims(consume weight, axis: -1, name: nil)
        self.bias = mpsGraph.expandDims(consume bias, axis: -1, name: nil)
    }
}

@available(iOS 15.4, *)
public extension Conv1d {
    
    init(weight: borrowing Graph.Tensor, bias: borrowing Graph.Tensor) {
        self.init(weight: weight.tensor, bias: bias.tensor)
    }
    
    func forward(_ input: borrowing Graph.Tensor) throws -> Graph.Tensor {
        guard let des = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1,
            dilationRateInX: 1, dilationRateInY: 1,
            groups: 1,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else { throw Errors.msg("MPSGraphConvolution2DOpDescriptor") }
        let graph = input.graph, mpsGraph = graph.mpsGraph
        let x = mpsGraph.expandDims(input.tensor, axis: -1, name: nil)
        
        let x0 = mpsGraph.convolution2D(x, weights: self.weight, descriptor: consume des, name: nil)
        let x1 = mpsGraph.addition(consume x0, self.bias, name: nil)
        let y = mpsGraph.squeeze(consume x1, axis: -1, name: nil)
        return graph.tensor(consume y)
    }
}
