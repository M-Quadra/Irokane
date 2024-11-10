//
//  Convolution.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/4.
//

import MetalPerformanceShadersGraph
fileprivate typealias F = Irokane.Functional

@available(iOS 15.4, *)
public struct Conv1d {
    let weight: MPSGraphTensor
    let bias: MPSGraphTensor
    let groups: Int
    let padding: Int
}

@available(iOS 15.4, *)
public extension Conv1d {
    
    init(
        weight: borrowing Graph.Tensor, // [out, in, kernel]
        bias: borrowing Graph.Tensor, // [out]
        inChannels: Int? = nil,
        outChannels: Int? = nil,
        kernelSize: Int? = nil,
        groups: Int = 1,
        padding: Int = 0
    ) {
        let mpsGraph = weight.graph.mpsGraph, w = weight.tensor, b = bias.tensor
        assert(mpsGraph == bias.graph.mpsGraph)
        assert(w.dataType == b.dataType)
        
        self.weight = mpsGraph.expandDims(consume w, axis: -1, name: nil)
        self.bias = mpsGraph.expandDims(consume b, axis: -1, name: nil)
        self.groups = groups
        self.padding = padding
        
        if let outChannels = outChannels { assert(self.outChannels == outChannels) }
        if let inChannels = inChannels { assert(self.inChannels == inChannels) }
        if let kernelSize = kernelSize { assert(self.kernelSize == kernelSize) }
    }
    
    var outChannels: Int? {
        self.weight.shape?.first?.intValue
    }
    var inChannels: Int? {
        guard let shape = self.weight.shape, shape.count > 1 else { return nil }
        return shape[1].intValue * self.groups
    }
    var kernelSize: Int? {
        guard let shape = self.weight.shape, shape.count > 2 else { return nil }
        return shape[2].intValue
    }
    
    func forward(_ input: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = input.graph, mpsGraph = graph.mpsGraph
        guard let des = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1,
            dilationRateInX: 1, dilationRateInY: 1,
            groups: self.groups,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else { preconditionFailure("MPSGraphConvolution2DOpDescriptor") }
        let xp = if self.padding > 0 {
            F.pad(input, pad: (left: self.padding, right: self.padding)).tensor
        } else { input.tensor }
        let x = mpsGraph.expandDims(xp, axis: -1, name: nil)
        
        let x0 = mpsGraph.convolution2D(x, weights: self.weight, descriptor: consume des, name: nil)
        let x1 = mpsGraph.squeeze(consume x0, axis: -1, name: nil)
        
        assert(x1.dataType == self.bias.dataType)
        let y = mpsGraph.addition(consume x1, self.bias, name: nil)
        return graph.tensor(consume y)
    }
}
