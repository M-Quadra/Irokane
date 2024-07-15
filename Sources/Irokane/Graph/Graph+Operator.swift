//
//  Graph+Operator.swift
//
//
//  Created by m_quadra on 2024/7/1.
//

import Foundation

public extension Graph {
    
    static func / (lhs: borrowing Graph, rhs: borrowing Graph) -> Graph {
        assert(lhs.graph == rhs.graph)
        let graph = lhs.graph, lhs = lhs.tensor
        
        var rhs = rhs.tensor
        if lhs.dataType != rhs.dataType {
            rhs = graph.cast(rhs, to: lhs.dataType, name: nil)
        }
        
        let ts = graph.divisionNoNaN(lhs, rhs, name: nil)
        return Graph(tensor: ts, graph: graph)
    }
    
    prefix static func - (tensor: borrowing Graph) -> Graph {
        let ts = tensor.graph.negative(with: tensor.tensor, name: nil)
        return Graph(tensor: consume ts, graph: tensor.graph)
    }
    
    prefix static func ~ (tensor: borrowing Graph) -> Graph {
        let graph = tensor.graph, src = tensor.tensor
        assert(src.dataType == .bool)
        let rhs = graph.constant(0, dataType: src.dataType)
        
        let dst = graph.equal(consume src, consume rhs, name: nil)
        return Graph(tensor: consume dst, graph: consume graph)
    }
    
    static func >= (lhs: borrowing Graph, rhs: Double) -> Graph {
        let graph = lhs.graph, x = lhs.tensor
        let a = graph.constant(rhs, dataType: x.dataType)
        
        let y = graph.greaterThanOrEqualTo(consume x, consume a, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    
    static func >= (lhs: borrowing Graph, rhs: borrowing Graph) -> Graph {
        let graph = lhs.graph, x = lhs.tensor
        assert(graph == rhs.graph)
        
        let y = graph.greaterThanOrEqualTo(x, rhs.tensor, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    
    static func <= (lhs: borrowing Graph, rhs: Double) -> Graph {
        let graph = lhs.graph, src = lhs.tensor
        let rhs = graph.constant(rhs, dataType: src.dataType)
        
        let dst = graph.lessThanOrEqualTo(consume src, consume rhs, name: nil)
        return Graph(tensor: consume dst, graph: graph)
    }
    
    static func & (lhs: borrowing Graph, rhs: borrowing Graph) -> Graph {
        let graph = lhs.graph, lhs = lhs.tensor
        assert(graph == rhs.graph)
        var rhs = rhs.tensor
        if lhs.dataType != rhs.dataType {
            rhs = graph.cast(rhs, to: lhs.dataType, name: nil)
        }
        
        let ts = graph.logicalAND(consume lhs, consume rhs, name: nil)
        return Graph(tensor: consume ts, graph: consume graph)
    }
    
    static func * (lhs: Double, rhs: borrowing Graph) -> Graph {
        let graph = rhs.graph, x = rhs.tensor
        
        let a = graph.constant(lhs, dataType: x.dataType)
        let y = graph.multiplication(consume x, consume a, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    
    static func * (lhs: borrowing Graph, rhs: borrowing Graph) -> Graph {
        let graph = lhs.graph, x = lhs.tensor
        assert(graph == rhs.graph)
        
        let y = graph.multiplication(consume x, rhs.tensor, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    
    static func + (lhs: Double, rhs: borrowing Graph) -> Graph {
        let graph = rhs.graph, x = rhs.tensor

        let a = graph.constant(lhs, dataType: x.dataType)
        let y = graph.addition(consume x, consume a, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    static func + (lhs: borrowing Graph, rhs: Double) -> Graph {
        return rhs + lhs
    }
    
    static func + (lhs: borrowing Graph, rhs: borrowing Graph) -> Graph {
        let graph = lhs.graph, x = lhs.tensor
        assert(graph == rhs.graph)

        let y = graph.addition(consume x, rhs.tensor, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    
    static func - (lhs: borrowing Graph, rhs: borrowing Graph) -> Graph {
        let graph = lhs.graph, x = lhs.tensor
        assert(graph == rhs.graph)

        let y = graph.subtraction(consume x, rhs.tensor, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    static func - (lhs: borrowing Graph, rhs: Int) -> Graph {
        let graph = lhs.graph, x = lhs.tensor
        let a = graph.constant(Double(rhs), dataType: x.dataType)
        
        let y = graph.subtraction(consume x, consume a, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
    static func - (lhs: Int, rhs: borrowing Graph) -> Graph {
        let graph = rhs.graph, x = rhs.tensor
        let a = graph.constant(Double(lhs), dataType: x.dataType)
        
        let y = graph.subtraction(consume a, consume x, name: nil)
        return Graph(tensor: consume y, graph: consume graph)
    }
}
