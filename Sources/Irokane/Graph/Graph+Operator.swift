//
//  Graph+Operator.swift
//
//
//  Created by m_quadra on 2024/7/1.
//

import Foundation

@available(iOS 14.0, *)
public extension Graph.Tensor {
    
    // x / a
    static func / (lhs: borrowing Graph.Tensor, rhs: Double) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        let a = graph.constant(Double(rhs), dataType: x.dataType)
        
        let y = graph.division(consume x, consume a, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume y)
    }
    
    // -x
    prefix static func - (tensor: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = tensor.graph.mpsGraph, x = tensor.tensor
        
        let y = graph.negative(with: x, name: nil)
        return Graph.Tensor(graph: tensor.graph, tensor: consume y)
    }
    
    static func >= (lhs: borrowing Graph.Tensor, rhs: Double) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        let a = graph.constant(rhs, dataType: x.dataType)
        
        let y = graph.greaterThanOrEqualTo(consume x, consume a, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume y)
    }
    
    static func >= (lhs: borrowing Graph.Tensor, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        assert(graph == rhs.graph.mpsGraph)
        
        let y = graph.greaterThanOrEqualTo(x, rhs.tensor, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume y)
    }
    
    static func <= (lhs: borrowing Graph.Tensor, rhs: Double) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        let y = graph.constant(rhs, dataType: x.dataType)
        
        let z = graph.lessThanOrEqualTo(consume x, consume y, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume z)
    }
    
    // a * x
    static func * (lhs: Double, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = rhs.graph.mpsGraph, x = rhs.tensor
        let a = graph.constant(lhs, dataType: x.dataType)
        
        let y = graph.multiplication(consume x, consume a, name: nil)
        return Graph.Tensor(graph: rhs.graph, tensor: consume y)
    }
    // x * a
    static func * (lhs: borrowing Graph.Tensor, rhs: Double) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        let a = graph.constant(rhs, dataType: x.dataType)
        
        let y = graph.multiplication(consume x, consume a, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume y)
    }
    
    // x * y
    static func * (lhs: borrowing Graph.Tensor, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = lhs.graph, mpsGraph = graph.mpsGraph
        let x = lhs.tensor, y = rhs.tensor
        if #available(iOS 15.0, *) {
            assert(x.dataType == y.dataType || x.dataType == .bool || y.dataType == .bool)
        } else {
            assert(x.dataType == y.dataType)
        }
        
        let z = mpsGraph.multiplication(consume x, consume y, name: nil)
        return graph.tensor(consume z)
    }
    
    // a + x
    static func + (lhs: Double, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
        let mpsGraph = rhs.graph.mpsGraph, x = rhs.tensor
        let a = mpsGraph.constant(lhs, dataType: x.dataType)
        
        let y = mpsGraph.addition(consume x, consume a, name: nil)
        return rhs.graph.tensor(consume y)
    }
    static func + (lhs: borrowing Graph.Tensor, rhs: Double) -> Graph.Tensor {
        return rhs + lhs
    }
    
    // x + y
    static func + (lhs: borrowing Graph.Tensor, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        assert(graph == rhs.graph.mpsGraph)

        let y = graph.addition(consume x, rhs.tensor, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume y)
    }
    
    static func - (lhs: borrowing Graph.Tensor, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        assert(graph == rhs.graph.mpsGraph)
        assert(x.dataType == rhs.tensor.dataType)
        
        let y = graph.subtraction(consume x, rhs.tensor, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume y)
    }
    static func - (lhs: borrowing Graph.Tensor, rhs: Int) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        let a = graph.constant(Double(rhs), dataType: x.dataType)
        
        let y = graph.subtraction(consume x, consume a, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume y)
    }
    static func - (lhs: Int, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = rhs.graph.mpsGraph, x = rhs.tensor
        let a = graph.constant(Double(lhs), dataType: x.dataType)
        
        let y = graph.subtraction(consume a, consume x, name: nil)
        return Graph.Tensor(graph: rhs.graph, tensor: consume y)
    }
    
    static func < (lhs: borrowing Graph.Tensor, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
#if DEBUG
        let name: String = "\(#fileID):\(#line)"
#else
        let name: String? = nil
#endif
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        assert(graph == rhs.graph.mpsGraph)
        assert(x.dataType == rhs.tensor.dataType)
        
        let y = graph.lessThan(consume x, rhs.tensor, name: consume name)
        return Graph.Tensor(graph: lhs.graph, tensor: consume y)
    }
}

@available(iOS 15.0, *)
public extension Graph.Tensor {
    
    // x / y
    static func / (lhs: borrowing Graph.Tensor, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        assert(lhs.graph === rhs.graph)
        
        var y = rhs.tensor
        if x.dataType != y.dataType {
            y = graph.cast(y, to: x.dataType, name: nil)
        }
        
        let z = graph.division(consume x, consume y, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume z)
    }
    
    // ~x
    prefix static func ~ (tensor: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = tensor.graph.mpsGraph, x = tensor.tensor
        assert(x.dataType == .bool)
        let a = graph.constant(0, dataType: x.dataType)
        
        let y = graph.equal(consume x, consume a, name: nil)
        return Graph.Tensor(graph: tensor.graph, tensor: consume y)
    }
    
    static func & (lhs: borrowing Graph.Tensor, rhs: borrowing Graph.Tensor) -> Graph.Tensor {
        let graph = lhs.graph.mpsGraph, x = lhs.tensor
        assert(graph == rhs.graph.mpsGraph)
        var y = rhs.tensor
        if x.dataType != y.dataType {
            y = graph.cast(y, to: x.dataType, name: nil)
        }
        
        let z = graph.logicalAND(consume x, consume y, name: nil)
        return Graph.Tensor(graph: lhs.graph, tensor: consume z)
    }
}
