//
//  Tensor+Subscript.swift
//
//
//  Created by m_quadra on 2024/6/30.
//

import CoreML
import MetalPerformanceShadersGraph

public extension Tensor {
    
    subscript(_: (UnboundedRange_) -> (), range: PartialRangeUpTo<Int>) -> Task<Tensor, Error> { Task {
        if #available(iOS 18.0, *),
           let ts = self.base as? MLTensor {
            return ts[..., range].toTensor()
        }
        let graph = MPSGraph()
        
        let (x, xData) = try await self.base.toMPS(graph: graph)
        let y = graph.sliceTensor(x, dimension: -1, start: 0, length: range.upperBound, name: nil)
        
        guard let yData = graph.run(
            feeds: [consume x: consume xData],
            targetTensors: [y],
            targetOperations: nil
        )[consume y] else { throw Errors.msg("graph.run") }
        return yData.toTensor()
    }}
}
