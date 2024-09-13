//
//  Wrapper.swift
//  
//
//  Created by m_quadra on 2024/6/26.
//

#if swift(>=6.1)
public struct Wrapper<Base>: ~Copyable {
    let base: Base
}
#else
public struct Wrapper<Base>: ~Copyable {
    
    public let base: Base
    
    public init(base: Base) {
        self.base = base
    }
}
#endif
