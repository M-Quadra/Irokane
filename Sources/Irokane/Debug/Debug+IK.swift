//
//  Debug+IK.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/31.
//

import ObjectiveC

#if DEBUG
@available(iOS 15.0, *)
public extension Graph.Tensor {
    consuming func debug(isFull: Bool = false) {
        let xData = try! self.tensorData()
        print("dtype:", xData.dataType)
        print("shape:", xData.shape)
        let step = xData.shape.last?.intValue ?? 1
        
        switch xData.dataType {
        case .float16:
            let arr = try! xData.ik.toFloat16s().toFloat32s()
            printFloat32s(consume arr, step: step, isFull: isFull)
        case .float32:
            let arr = try! xData.ik.toFloat32s()
            printFloat32s(consume arr, step: step, isFull: isFull)
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
        self.mpsGraph.perform(Selector(("dump")))
    }
}
#endif

fileprivate func joinLine(_ line: [String], isFull: Bool) -> String {
    if isFull || line.count < 6 { return "[\(line.joined(separator: ", "))]" }
    let arr = line.prefix(3) + ["..."] + line.suffix(3)
    return "[\(arr.joined(separator: ", "))]"
}

fileprivate func printFloat32s(_ arr: borrowing [Float32], step: Int, isFull: Bool) {
    if arr.isEmpty { return }
    print(String(format: "mean: %.4f sum: %.4f", arr.mean, arr.sum))
    print(String(format: "min: %.4f max: %.4f", arr.min, arr.max))
    
    var lines = [String]()
    for i in stride(from: 0, to: arr.count, by: step) {
        let line = (0..<step).map { arr[i+$0] }
            .map { String(format: "%.4f", $0) }
        lines.append(joinLine(consume line, isFull: isFull))
    }
    if isFull || lines.count < 6 {
        print(lines.joined(separator: "\n"))
    } else {
        print(lines.prefix(3).joined(separator: "\n"))
        print("\t...")
        print(lines.suffix(3).joined(separator: "\n"))
    }
}
