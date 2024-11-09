//
//  Float32sTests.swift
//  Irokane
//
//  Created by m_quadra on 2024/11/9.
//

import Testing
import Irokane

struct Float32sTests {
    
    @Test func mean() {
        let arr: [Float32] = [0, 1, 2]
        let v = arr.mean
        #expect(v == 1)
    }
    
    @Test func std() {
        let arr: [Float32] = [0, 1, 2]
        let v = arr.std
        #expect(v == 0.8164966)
    }
    
    @Test func sum() {
        let arr: [Float32] = [0, 1, 2]
        let v = arr.sum
        #expect(v == 3)
    }
    
    @Test func min() {
        let arr: [Float32] = [0, 1, 2]
        let v = arr.min
        #expect(v == 0)
    }
    
    @Test func max() {
        let arr: [Float32] = [0, 1, 2]
        let v = arr.max
        #expect(v == 2)
    }

    @Test func power0() {
        let arr: [Float32] = [0, 1, 2]
        let v = arr ** 2
        #expect(v == [0, 1, 4])
    }

    @Test func power1() {
        let arr: [Float32] = [0, 1, 2]
        let v = arr ** 3
        #expect(v == [0, 1, 8])
    }

    @Test func minus() {
        let arr: [Float32] = [0, 1, 2]
        let v = arr - 1
        #expect(v == [-1, 0, 1])
    }
    
    @available(iOS 14.0, *)
    @Test func float16sToFloat32s() {
        let arr: [Float16] = [0, 1, 2]
        let v = arr.toFloat32s()
        #expect(v == [0, 1, 2])
    }
}
