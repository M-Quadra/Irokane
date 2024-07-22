//
//  F+Conv1d.swift
//
//
//  Created by m_quadra on 2024/6/24.
//

import CoreML
import MetalPerformanceShadersGraph

// MARK: - Public
public extension Functional {
    
    static func conv1d(
        input: MLMultiArray,
        weight: MLMultiArray,
        bias: MLMultiArray? = nil
    ) async throws -> MLMultiArray {
        try self.checkConv1d(input: input, weight: weight, bias: bias)
        let graph = MPSGraph()
        
        let (x, xData) = try await input.toMPS(graph: graph)
        let (w, wData) = try await weight.toMPS(graph: graph)
        var iptDic: [MPSGraphTensor: MPSGraphTensorData] = [
            x: consume xData,
            w: consume wData,
        ]
        
        let x0 = graph.expandDims(consume x, axis: -1, name: nil)
        let w0 = graph.expandDims(consume w, axis: -1, name: nil)
        
        guard let desc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else { throw Errors.msg("MPSGraphConvolution2DOpDescriptor") }
        let conv2d = graph.convolution2D(consume x0, weights: consume w0, descriptor: consume desc, name: nil)
        
        var y = graph.squeeze(conv2d, axis: -1, name: nil)
        if let bias = consume bias {
            let (b, bData) = try await bias.toMPS(graph: graph)
            iptDic[b] = consume bData
            
            let b0 = graph.expandDims(consume b, axes: [0, -1], name: nil)
            y = graph.addition(consume y, consume b0, name: nil)
        }
        
        guard let yData: MPSGraphTensorData = graph.run(
            feeds: iptDic,
            targetTensors: [y],
            targetOperations: nil
        )[consume y] else { throw Errors.msg("graph.run") }
        
        return try yData.ik.toMLMultiArray()
    }
}

// MARK: - Private
fileprivate extension Functional {
    
    static func checkConv1d(
        input: MLMultiArray,
        weight: MLMultiArray,
        bias: MLMultiArray?
    ) throws(Errors) {
        switch true {
        case input.shape.count != 3:
            throw .msg("need [n, cIn, lIn]")
        case weight.shape.count != 3:
            throw .msg("need [cOut, cIn, k]")
        case input.shape[1] != weight.shape[1]:
            throw .msg("cIn must be same")
        case input.shape[2].intValue < weight.shape[2].intValue:
            throw .msg("need lIn >= k")
        default: break
        }
        
        guard let bias = bias else { return }
        switch true {
        case bias.shape.count != 1:
            throw .msg("need (cOut)")
        case bias.shape[0] != weight.shape[0]:
            throw .msg("cOut must be same")
        default: break
        }
    }
}
