import tensorrt as trt
import os

# 设置 TensorRT 的日志显示级别
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    # 取消了无用的 fp16_mode 参数，因为精度现在由 ONNX 模型自带属性决定
    print(f"==================================================")
    print(f"🚀 开始转换: {onnx_file_path} -> {engine_file_path}")

    # 1. 创建 Builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 2. 设置显存池使用上限 (最大 2GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    # 创建优化配置文件 (Optimization Profile)
    profile = builder.create_optimization_profile()
    # 告诉 TensorRT 我们期望的 Batch Size 范围 (最小, 最优, 最大)
    if "lprnet" in engine_file_path:
        # LPRNet 允许 1~16 个车牌同时输入
        profile.set_shape('images', (1, 3, 24, 94), (4, 3, 24, 94), (64, 3, 24, 94))
    elif "yolov5" in engine_file_path:
        # YOLOv5 保持单张图输入 [1, 3, 640, 640]
        profile.set_shape('images', (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))

    config.add_optimization_profile(profile)
    if hasattr(trt.BuilderFlag, 'FP16'):
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("⚡ TensorRT 版本不支持显式 FP16 标志，将依赖 ONNX 模型的类型自动推导。")  # 强制解锁全局 FP16 优化！
    # 3. 自动识别类型 🌟
    print("⚡ 依靠强类型(Strongly Typed) ONNX 自动开启 FP16 加速计算！")

    # 4. 解析 ONNX 文件
    if not os.path.exists(onnx_file_path):
        print(f"❌ 找不到 ONNX 文件: {onnx_file_path}")
        return

    print("🔍 正在解析 ONNX 文件...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ ONNX 解析失败！错误信息：")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    print("✅ ONNX 解析成功！")

    # 5. 构建序列化引擎 (耗时操作)
    print("⏳ 正在构建 TensorRT 引擎，显卡正在全力寻找最优算子...")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("❌ 引擎构建失败！")
        return

    # 6. 保存引擎文件
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"🎉 引擎成功保存至: {engine_file_path}")
    print(f"==================================================\n")


if __name__ == '__main__':
    # 转换 YOLOv5 检测模型
    build_engine("yolov5_best.onnx", "yolov5_best.engine")

    # 转换 LPRNet 识别模型
    build_engine("lprnet_best.onnx", "lprnet_best.engine")

    print("✅ 所有模型均已成功转换为 TensorRT 专属格式！")