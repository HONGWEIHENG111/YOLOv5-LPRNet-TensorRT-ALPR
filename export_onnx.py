import torch
from models.experimental import attempt_load
from models.LPRNet import LPRNet, CHARS

def export_yolov5():
    print("正在导出 YOLOv5 (FP16)...")
    weights = './weights/yolov5_best.pt'
    img_size = [640, 640]
    batch_size = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    model = attempt_load(weights, map_location=device)
    model.eval()
    model.half()  # 🌟【新增】将模型的所有参数转换为半精度 (FP16)

    # 2. 设置 export 标志，去除网格解码后处理，对 TensorRT 更友好
    for k, m in model.named_modules():
        if type(m).__name__ == 'Detect':
            m.export = True

    # 3. 创建测试张量，并同样转为 FP16
    dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device).half() # 🌟【新增】 .half()

    # 4. 导出 ONNX
    onnx_path = 'yolov5_best.onnx'
    torch.onnx.export(model, dummy_input, onnx_path,
                      verbose=False, opset_version=17,
                      input_names=['images'],
                      output_names=['output'])
    print(f"YOLOv5 FP16 成功导出至: {onnx_path}")


def export_lprnet():
    print("正在导出 LPRNet (FP16)...")
    weights = './weights/lprnet_best.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. 实例化网络
    model = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    model.half()  # 🌟将模型的所有参数转换为半精度 (FP16)

    # 2. 创建测试张量，并同样转为 FP16
    dummy_input = torch.randn(1, 3, 24, 94).to(device).half() # 🌟【新增】 .half()

    # 3. 导出 ONNX
    onnx_path = 'lprnet_best.onnx'
    torch.onnx.export(model, dummy_input, onnx_path,
                      verbose=False, opset_version=17,
                      input_names=['images'],
                      output_names=['output'],
                      dynamic_axes = {'images': {0: 'batch_size'},  # 🌟【关键】让输入的第0维变成动态
                     'output': {0: 'batch_size'}})
    print(f"LPRNet FP16 成功导出至: {onnx_path}")


if __name__ == '__main__':
    export_yolov5()
    export_lprnet()
    print("全部导出完成！接下来准备转换 TensorRT。")