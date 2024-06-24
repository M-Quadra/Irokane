//
//  Errors.swift
//
//
//  Created by m_quadra on 2024/6/23.
//

import Foundation

public enum Errors: Error {
    case msg(_: String = "", file: StaticString = #fileID, line: UInt = #line)
    case todo(_: String = "", file: StaticString = #fileID, line: UInt = #line)
}
