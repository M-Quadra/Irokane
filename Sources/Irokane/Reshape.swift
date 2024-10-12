//
//  Reshape.swift
//  
//
//  Created by m_quadra on 2024/6/25.
//

import CoreML
import MetalPerformanceShadersGraph

@available(iOS 14.0, *)
public func reshape(tensor: Tensor, to shape: consuming [Int]) async throws -> Tensor {
    if #available(iOS 18.0, *),
       let ts = tensor.base as? MLTensor {
        return ts.reshaped(to: shape).toTensor()
    }
    let graph = MPSGraph()
    
    let (x, xData) = try await tensor.base.toMPS(graph: graph)
    let y = graph.reshape(x, shape: shape as [NSNumber], name: nil)
    
    guard let yData: MPSGraphTensorData = graph.run(
        feeds: [consume x: consume xData],
        targetTensors: [y],
        targetOperations: nil
    )[consume y] else { throw Errors.msg("graph.run") }
    return yData.toTensor()
}

@available(iOS 16.0, *)
public func permute(tensor: Tensor, dims: consuming [Int]) async throws -> Tensor {
    if #available(iOS 18.0, *),
       let ts = tensor.base as? MLTensor,
       ts.scalarCount < 0x32A000 {
        return ts.transposed(permutation: dims).toTensor()
    }
    let graph = MPSGraph()
    
    let (x, xData) = try await tensor.base.toMPS(graph: graph)
    let y = graph.transpose(x, permutation: dims as [NSNumber], name: nil)
    
    guard let yData: MPSGraphTensorData = graph.run(
        feeds: [consume x: consume xData],
        targetTensors: [y],
        targetOperations: nil
    )[consume y] else { throw Errors.msg("graph.run") }
    
    return yData.toTensor()
}
