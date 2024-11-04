//
//  Graph+TensorCreator.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/3.
//

import Foundation
import MetalPerformanceShadersGraph

@available(iOS 14.0, *)
public extension Graph {
    
    borrowing func zeros(_ size: Int) -> Graph.Tensor {
        let mpsGraph = self.mpsGraph
        let cnt = size * MemoryLayout<Int32>.size
        
        let x = mpsGraph.constant(Data(repeating: 0, count: cnt), shape: [size as NSNumber], dataType: .int32)
        let y = mpsGraph.read(consume x, name: nil)
        return self.tensor(consume y)
    }
    
    borrowing func tensor(_ data: consuming Data, shape: consuming [Int], dataType: MPSDataType) -> Graph.Tensor {
        let mpsGraph = self.mpsGraph
        
        let x = mpsGraph.constant(data, shape: shape as [NSNumber], dataType: dataType)
        let y = mpsGraph.read(consume x, name: nil)
        return self.tensor(consume y)
    }
}
