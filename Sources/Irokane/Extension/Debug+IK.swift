//
//  Debug+IK.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/31.
//

import CoreML

#if DEBUG
public extension Wrapper<MLMultiArray> {
    consuming func debug() {
        let arr = self.base
        print("dtype:", arr.dataType)
        print("shape:", arr.shape)
        if arr.count <= 0 { return }
        
        let step = arr.shape.last?.intValue ?? 1
        for i in stride(from: 0, to: arr.count, by: step) {
            var values = [Float32]()
            for j in 0..<step {
                values.append(arr[i + j].floatValue)
            }
            print(consume values)
        }
    }
}

@available(iOS 15.0, *)
public extension Graph.Tensor {
    consuming func debug() {
        let xData = try! self.tensorData
        print("dtype:", xData.dataType)
        print("shape:", xData.shape)
        let step = xData.shape.last?.intValue ?? 1
        
        switch xData.dataType {
        case .float32:
            let arr = try! xData.ik.toFloat32s()
            if arr.isEmpty { return }
            print("mean:", arr.mean, "sum: ", arr.sum)
            
            for i in stride(from: 0, to: arr.count, by: step) {
                let line = (0..<step).map { arr[i+$0] }
                    .map { String(format: "%.4f", $0) }
                    .joined(separator: ", ")
                print(consume line)
            }
        case .int32:
            let arr = try! xData.ik.toInt32s()
            if arr.isEmpty { return }
            
            for i in stride(from: 0, to: arr.count, by: step) {
                let line = (0..<step).map { String(arr[i+$0]) }
                    .joined(separator: ", ")
                print(consume line)
            }
        case .bool:
            let arr = try! xData.ik.toBools()
            if arr.isEmpty { return }
            
            for i in stride(from: 0, to: arr.count, by: step) {
                let line = (0..<step).map { arr[i+$0] ? "True" : "False" }
                    .joined(separator: ", ")
                print(consume line)
            }
        default: print("todo")
        }
    }
}

@available(iOS 14.0, *)
public extension Graph {
    func debug() {
        self.graph.perform(Selector(("dump")))
    }
}
#endif
