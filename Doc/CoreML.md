# CoreML

非常原始，非常粗糙。

内部细节缺失，[coremltools](https://github.com/apple/coremltools)维护频率低下，算子降级频繁，十分甚至九分的难用。

想找替代品，时间不允许，只好抛砖引玉。

以下内容均以 torch 为前提，虽然 CoreML 一股 FT 味（恼

Neural Engine，本文称NPU，下同。

没有一番高论，全部自我乱写。



## 加速降级

> 某次 WWDC 画饼 Performance 报告将会展示无法加速的原因，结果新版Xcode连报告展示都间歇性抽风。一些 op运行报告在 Xcode 15.4 正常，Xcode 16 各种问题，不知何时才修。最为难堪的是原本在旧 Xcode 正常显示的CPU-only 算子报告，Xcode16 有可能无法显示 CPU-only 算子，以至于问题排查十分痛苦。
>
> 恶意揣测当鸵鸟。

加速降级大抵遵循 NPU、GPU、CPU，依次降级。实际中多为 NPU/GPU 直降 CPU 且大概率升不回来。例如某模型分 ab 2个部分，a部分转换后为 CPU-only，b部分转换后可用 GPU 加速。此时若为偷懒合并ab输入输出，则模型整体直接被降级为 CPU-only。

辣么何时降级捏？[MIL ops 文档](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html)没说，[CoreML 文档](https://developer.apple.com/documentation/coreml/)没提。



- `cumsum`

  无论是否 fixed shape 均会触发降级CPU。

  ```python
  class Model(nn.Module):
      def forward(self, x):
          x0 = torch.cumsum(x, dim=-1)
          return x0 + 1
	```

  不管后续一个一个一个 op 如何简单均会被塞回 CPU（悲

  理论上现有推理优化工作不少针对固定shape，因此 CoreML 有种十分浓烈的管生不管养味。
  
  


- `fill_like`

  Performance 报告显示有可能上GPU，但我个人没尝试出来。不论 shape 是否固定均为 CPU-only。
  
  
  
- `fill`

  CPU-only，且看起来运行时没有任何优化。
  
  
  
- `range_1d`

  对应`torch.arange`，CPU-only op。



## 等效替换

由于`coremltools`尚未成熟，比起外部调整 torch op 结果亦或修改`coremltools`，还是直接动网络更轻松。

截止这句话写完的时间，`coremltools`对`torch.jit.script`支持依旧简陋，实践上推荐以`torch.jit.trace`为主。将原实现保留抽出方法，通过检查是否 trace 区分替换逻辑。

```python
def f0(x: torch.Tensor) -> torch.Tensor:
  if not torch.jit.is_tracing():
    # 原实现
    return x
  
  # 替换后
  return x
```
