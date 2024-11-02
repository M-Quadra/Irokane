//
//  Issue+Gather.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/2.
//

import Testing
import Irokane
import CoreML

fileprivate let isRunIssue = true
&& false

struct GatherIssue {
    
    @available(iOS 17.0, *)
    @Test(.enabled(if: isRunIssue))
    func failed0() throws {
        let graph = Graph(), mpsGraph = graph.graph
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([2, 3])
        let m = try MLMultiArray([0, 2]).ik.to(graph: graph)
        
        let y = x[m, ...]
        _ = y
        
        // Assertion failed: (0 && "value has no static type")
        mpsGraph.run(feeds: [:], targetTensors: [m.tensor], targetOperations: nil)
    }
    
    @available(iOS 17.0, *)
    @Test func irokane0() throws {
        let graph = Graph()
        let x = try MLMultiArray(0..<6).ik.to(graph: graph)
            .reshape([2, 3])
        let m = try MLMultiArray([0, 2]).ik.to(graph: graph)
        
        let y = x[m, ...]
        _ = y
        
        let mData = try m.tensorData()
        #expect(mData.shape == [2])
        
        let arr = try mData.ik.toInt32s()
        #expect(arr == [0, 2])
    }
    
    @available(iOS 17.0, *)
    @Test(.enabled(if: isRunIssue))
    func failed1() throws {
        let graph = Irokane.Graph(), mpsGraph = graph.graph
        let x = try MLMultiArray((0..<6)).ik.to(graph: graph)
            .reshape([2, 3])
        let m = try MLMultiArray([
            0, 1, 0,
            1, 0, 1
        ]).ik.to(graph: graph)
            .reshape([2, 3])
        
        let y = x[m]
        _ = y
        
        // Assertion failed: (0 && "value has no static type")
        mpsGraph.run(feeds: [:], targetTensors: [m.tensor], targetOperations: nil)
    }

    @available(iOS 17.0, *)
    @Test func irokane1() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray((0..<6)).ik.to(graph: graph)
            .reshape([2, 3])
        let m = try MLMultiArray([
            0, 1, 0,
            1, 0, 1
        ]).ik.to(graph: graph)
            .reshape([2, 3])
        
        let y = x[m]
        _ = y
        
        let mData = try m.tensorData()
        #expect(mData.shape == [2, 3])
        
        let arr = try mData.ik.toInt32s()
        #expect(arr == [
            0, 1, 0,
            1, 0, 1
        ])
    }
}
