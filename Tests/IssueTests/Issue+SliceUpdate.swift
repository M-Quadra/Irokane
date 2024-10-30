//
//  Issue+SliceUpdate.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/30.
//

import Testing
import CoreML
import Irokane

fileprivate let isRun = true
&& false

/*
 MPS的shape推理存在奇怪问题，想办法介入？
 */

@Suite("slice update")
struct IssueSliceUpdate {
    
    @available(iOS 17.0, *)
    @Test(.enabled(if: isRun))
    func caseFailed_0() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([1, 2, 3])
            .ik.to(graph: graph)
        let mask = try MLMultiArray([0, 1, 0])
            .ik.to(graph: graph)
            .cast(to: .bool)
        
        var y = Irokane.zerosLike(x)
        let x0 = x[mask]
        
        y[mask] .= x0
        
        // Assertion failed: (0 && "value has no static type")
        _ = try x0.tensorData
    }
    
    @available(iOS 17.0, *)
    @Test(.enabled(if: isRun))
    func caseFailed_1() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([1, 2, 3])
            .ik.to(graph: graph)
        let mask = try MLMultiArray([0, 1, 0])
            .ik.to(graph: graph)
            .cast(to: .bool)
        
        var y = Irokane.zerosLike(x)
        
        _ = try x[mask].tensorData
        
        y[mask] .= x[mask]
        
        // Assertion failed: (0 && "value has no static type")
        _ = try x[mask].tensorData
    }
    
    @available(iOS 17.0, *)
    @Test func caseSuccess0() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([1, 2, 3])
            .ik.to(graph: graph)
        let mask = try MLMultiArray([0, 1, 0])
            .ik.to(graph: graph)
            .cast(to: .bool)
        
        var y = Irokane.zerosLike(x)
        let x0 = x[mask]
        _ = try x0.tensorData
        
        y[mask] .= x0
        
        _ = try x0.tensorData
    }
    
    @available(iOS 17.0, *)
    @Test func caseSuccess1() throws {
        let graph = Irokane.Graph()
        let x = try MLMultiArray([1, 2, 3])
            .ik.to(graph: graph)
        let mask = try MLMultiArray([0, 1, 0])
            .ik.to(graph: graph)
            .cast(to: .bool)
        
        var y = Irokane.zerosLike(x)
        let x0 = x[mask]
        _ = try x0.tensorData
        
        y[mask] .= x0
        
        _ = try y.tensorData
    }
}
