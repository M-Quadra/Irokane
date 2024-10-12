//
//  Irokane+Alias.swift
//  Irokane
//
//  Created by m_quadra on 2024/10/2.
//

@available(iOS 14.0, *)
public func clampMin(_ input: borrowing Graph.Tensor, _ min: Double) -> Graph.Tensor {
    return Irokane.maximum(input, min)
}

@available(iOS 15.4, *)
public func unsqueeze(_ input: borrowing Graph.Tensor, dim: Int) -> Graph.Tensor {
    return input.unsqueeze(dim)
}
