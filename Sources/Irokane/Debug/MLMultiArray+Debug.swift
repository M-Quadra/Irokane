//
//  MLMultiArray+Debug.swift
//  Irokane
//
//  Created by m_quadra on 2025/4/30.
//

import CoreML
import OSLog

#if DEBUG
public extension Wrapper<MLMultiArray> {
    consuming func debug() {
        Logger().debug("\(self.description)")
    }
}
#endif

private extension Wrapper<MLMultiArray> {
    var description: String {
        let arr = self.base
        var lines = ["dtype: \(arr.dataType) shape: \(arr.shape)"]
        
        let step = arr.shape.last?.intValue ?? 1
        for i in stride(from: 0, to: arr.count, by: step) {
            var values = [Float32]()
            for j in 0..<step {
                values.append(arr[i + j].floatValue)
            }
            lines.append("\(consume values)")
        }
        return lines.joined(separator: "\n")
    }
}
