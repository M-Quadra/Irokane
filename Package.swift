// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Irokane",
    platforms: [.iOS(.v14)],
    products: [
        .library(name: "Irokane", targets: ["Irokane"]),
    ],
    targets: [
        .target(name: "Irokane"),
        .target(name: "TestUtil"),
        .testTarget(name: "IrokaneTests", dependencies: ["Irokane", "TestUtil"]),
        .testTarget(name: "IssueTests", dependencies: ["Irokane", "TestUtil"]),
        .testTarget(name: "RealDeviceTests", dependencies: ["Irokane", "TestUtil"]),
    ]
)
