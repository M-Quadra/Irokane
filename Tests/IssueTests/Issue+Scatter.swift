//
//  Issue+Scatter.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/2.
//

import Testing
import Irokane
import CoreML

fileprivate let isRunIssue = true
&& false

struct ScatterIssue {
    
    @available(iOS 17.0, *)
    @Test(.enabled(if: isRunIssue))
    func failed0() throws {
        let graph = Irokane.Graph(), mpsGraph = graph.graph
        var x = try MLMultiArray([1, 2, 3]).ik.to(graph: graph)
        let mask = try MLMultiArray([0, 1, 0]).ik.to(graph: graph)
        let y = try MLMultiArray([4]).ik.to(graph: graph)
        
        x[mask] .= y
        
        // Assertion failed: (0 && "value has no static type")
        mpsGraph.run(feeds: [:], targetTensors: [y.tensor], targetOperations: nil)
    }
    
    @available(iOS 17.0, *)
    @Test func irokane0() throws {
        let graph = Irokane.Graph()
        var x = try MLMultiArray([1, 2, 3]).ik.to(graph: graph)
        let mask = try MLMultiArray([0, 1, 0]).ik.to(graph: graph)
        let y = try MLMultiArray([4]).ik.to(graph: graph)
        
        x[mask] .= y
        
        let yData = try y.tensorData()
        #expect(yData.shape == [1])
        
        let arr = try yData.ik.toInt32s()
        #expect(arr == [4])
    }
    
    @available(iOS 17.0, *)
    @Test(.enabled(if: isRunIssue))
    func failed1() throws {
        let graph = Irokane.Graph(), mpsGraph = graph.graph
        var x = try MLMultiArray([1, 2, 3]).ik.to(graph: graph)
        let mask = try MLMultiArray([0, 1, 0]).ik.to(graph: graph)
        let y = try MLMultiArray([4]).ik.to(graph: graph)
        
        x[mask] .= y
        
        /*
         Assertion failed: (0 && "value has no static type")
         mpsGraph.compile will not delete the dead nodes.
         */
        let exe = mpsGraph.compile(with: nil, feeds: [:], targetTensors: [y.tensor], targetOperations: nil, compilationDescriptor: nil)
        guard let que = MTLCreateSystemDefaultDevice()?.makeCommandQueue(),
              let yData = exe.run(with: que, inputs: [], results: nil, executionDescriptor: nil).first
        else { throw Errors.msg("execute failed") }
        let arr = try yData.ik.toInt32s()
        #expect(arr == [4])
    }
}
