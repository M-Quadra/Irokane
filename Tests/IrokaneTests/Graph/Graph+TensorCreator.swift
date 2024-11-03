//
//  Graph+TensorCreator.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/3.
//

import Testing
import Irokane

struct GraphTensorCreator {
    
    @available(iOS 14.0, *)
    @Test func zeros() throws {
        let graph = Irokane.Graph()
        
        let y = graph.zeros(3)
        
        let yDaya = try y.tensorData()
        #expect(yDaya.shape == [3])
        
        let arr = try yDaya.ik.toInt32s()
        #expect(arr == [0, 0, 0])
    }
}
