//
//  F+Conv2d.swift
//
//
//  Created by m_quadra on 2024/6/23.
//

import CoreML
import MetalPerformanceShadersGraph

public extension Functional {
    
    @available(iOS 18.0, *)
    static func conv2d(
        input: MLTensor,
        weight: MLTensor
    ) async throws(Errors) -> MLTensor {
        let graph = MPSGraph()
        
        let x = try graph.placeholder(shape: input.shape as [NSNumber], dataType: input.mpsDataType, name: nil)
        let w = try graph.placeholder(shape: weight.shape as [NSNumber], dataType: weight.mpsDataType, name: nil)
        guard let desc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingStyle: .TF_SAME,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else { throw .msg("MPSGraphConvolution2DOpDescriptor") }
        let conv2d = graph.convolution2D(x, weights: w, descriptor: consume desc, name: nil)
        
        guard let yData: MPSGraphTensorData = graph.run(
            feeds: [
                consume x: try await input.toTensorData(),
                consume w: try await weight.toTensorData(),
            ],
            targetTensors: [conv2d],
            targetOperations: nil
        )[consume conv2d] else { throw .msg("graph.run") }
        
        return try yData.toMLTensor()
    }
    
    static func conv2d(
        input: MLMultiArray,
        weight: MLMultiArray
    ) async throws -> MLMultiArray {
        let graph = MPSGraph()
        
        let (x, xData) = try await input.toMPS(graph: graph)
        let (w, wData) = try await weight.toMPS(graph: graph)
        
        guard let desc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingStyle: .TF_SAME,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else { throw Errors.msg("MPSGraphConvolution2DOpDescriptor") }
        let conv2d = graph.convolution2D(x, weights: w, descriptor: consume desc, name: nil)
        
        guard let yData: MPSGraphTensorData = graph.run(
            feeds: [
                consume x: consume xData,
                consume w: consume wData,
            ],
            targetTensors: [conv2d],
            targetOperations: nil
        )[consume conv2d] else { throw Errors.msg("graph.run") }
        
        return try yData.ik.toMLMultiArray()
    }
}
