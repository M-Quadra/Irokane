//
//  Graph+SetItem.swift
//
//
//  Created by m_quadra on 2024/7/4.
//

public extension Graph {
    
    func setItem(at range: (_: (UnboundedRange_) -> (), index: Int), _ constant: Double) -> Graph {
        let graph = self.graph, src = self.tensor
        guard let len = src.shape?.last?.intValue else {
            assertionFailure("shape error")
            return self
        }
        let index = range.index + (range.index < 0 ? len : 0)
        guard 0 <= index, index < len else {
            assertionFailure("index error")
            return self
        }
        
        let i = graph.constant(Double(index), dataType: .int32)
        let oneHot = graph.cast(
            graph.oneHot(withIndicesTensor: i, depth: len, name: nil),
            to: .bool, name: nil
        )
        let v = graph.multiplication(
            oneHot, graph.constant(constant, dataType: src.dataType),
            name: nil
        )
        
        let ts0 = graph.multiplication(
            src, graph.logicalNOR(oneHot, oneHot, name: nil),
            name: nil
        )
        let ts1 = graph.addition(ts0, v, name: nil)
        return Graph(tensor: consume ts1, graph: consume graph)
        
        // TODO: branch mark with split+concat
        //            var arr = graph.split(src, numSplits: len, axis: -1, name: nil)
        //            arr[index] = graph.multiplication(
        //                arr[index], graph.constant(0, dataType: src.dataType),
        //                name: nil
        //            )
        //            arr[index] = graph.addition(
        //                arr[index], graph.constant(constant, dataType: src.dataType),
        //                name: nil
        //            )
        //            let ts = graph.concatTensors(arr, dimension: -1, name: nil)
    }
    
    func setItem(at mask: Graph, _ constant: Double) -> Graph {
        let graph = self.graph, x = self.tensor
        assert(graph == mask.graph)
        assert(x.shape != nil)
        assert(x.shape == mask.tensor.shape)
        
        let m = graph.cast(mask.tensor, to: .bool, name: nil)
        let n = graph.logicalNOR(m, m, name: nil)
        
        let x0 = graph.multiplication(
            x, graph.cast(n, to: x.dataType, name: nil),
            name: nil
        )
        
        let constant = graph.constant(constant, dataType: x.dataType)
        let v = graph.multiplication(
            graph.cast(m, to: x.dataType, name: nil), constant,
            name: nil
        )
        
        let y = graph.addition(x0, v, name: nil)
        return Graph(tensor: y, graph: graph)
    }
    
    func setItem(at mask: Graph, _ tensor: Graph) -> Graph {
        let graph = self.graph, x = self.tensor
        assert(graph == mask.graph)
        assert(x.shape != nil)
        assert(x.shape == mask.tensor.shape)
        
        let m = graph.cast(mask.tensor, to: .bool, name: nil)
        let i = graph.nonZeroIndices(m, name: nil)
        
        let y = graph.scatterNDWithData(x, updates: tensor.tensor, indices: i, batchDimensions: 0, mode: .set, name: nil)
        return Graph(tensor: y, graph: graph)
    }
}
