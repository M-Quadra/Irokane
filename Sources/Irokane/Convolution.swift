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
    
    init(
        weight: MPSGraphTensor, // [out, in, kernel]
        bias: MPSGraphTensor, // [out]
        groups: Int,
        padding: Int
    ) {
        let mpsGraph = weight.operation.graph
        self.weight = mpsGraph.expandDims(consume weight, axis: -1, name: nil)
        self.bias = mpsGraph.expandDims(consume bias, axis: -1, name: nil)
        self.groups = groups
        self.padding = padding
    }
}

@available(iOS 15.4, *)
public extension Conv1d {
    
    init(weight: borrowing Graph.Tensor, bias: borrowing Graph.Tensor, groups: Int = 1, padding: Int = 0) {
        self.init(weight: weight.tensor, bias: bias.tensor, groups: groups, padding: padding)
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
    
    func forward(_ input: borrowing Graph.Tensor) throws -> Graph.Tensor {
        guard let des = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1,
            dilationRateInX: 1, dilationRateInY: 1,
            groups: self.groups,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else { throw Errors.msg("MPSGraphConvolution2DOpDescriptor") }
        let graph = input.graph, mpsGraph = graph.mpsGraph
        let xp = if self.padding > 0 {
            F.pad(input, pad: (left: self.padding, right: self.padding)).tensor
        } else { input.tensor }
        let x = mpsGraph.expandDims(xp, axis: -1, name: nil)
        
        let x0 = mpsGraph.convolution2D(x, weights: self.weight, descriptor: consume des, name: nil)
        let x1 = mpsGraph.squeeze(consume x0, axis: -1, name: nil)
        let y = mpsGraph.addition(consume x1, self.bias, name: nil)
        return graph.tensor(consume y)
    }
}
