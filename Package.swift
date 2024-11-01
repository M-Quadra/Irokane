// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Irokane",
    platforms: [.iOS(.v13)],
    products: [
        .library(name: "Irokane", targets: ["Irokane"]),
    ],
    targets: [
        .target(name: "Irokane"),
        .testTarget(name: "IrokaneTests", dependencies: ["Irokane"]),
        .testTarget(name: "IssueTests", dependencies: ["Irokane"]),
        .testTarget(name: "RealDeviceTests", dependencies: ["Irokane"]),
    ]
)
