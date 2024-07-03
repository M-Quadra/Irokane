# Irokane

被CoreML的龟速推理难到了



对于非固定shape，究竟是每步操作独立，降低单次计算图编译代价快？还是尽可能构建完整计算图编译后再执行快？



iOS18的`MLTensor`加速行为未知，粗略看似乎是CPU？



花时间把2条路都肝一遍？

- Tenor

  类MLTensor即时获得编译结果

- Graph

  MPS计算图构建，先编译再推理



