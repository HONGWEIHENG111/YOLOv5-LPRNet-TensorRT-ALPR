# High-Performance ALPR: YOLOv5 + LPRNet TensorRT Deployment

之前做一个类似的边缘部署任务。只不过是在实习的企业里，相关的代码已经无法获取，但是使用的架构与本模型相似，因此心血来潮用该开源模型复现了一下。数据集用的是ccpd。
基于 YOLOv5 与 LPRNet 的高性能自动车牌识别（ALPR）边缘部署方案。本项目通过 NVIDIA TensorRT 实现了从 PyTorch 动态图到强类型静态计算图的转换，深度优化了显存调度和推理延迟，专为边缘设备和移动端 GPU 设计。

**作者:** Yuchen Ye

## ✨ 核心技术亮点

- **全流程 FP16 半精度加速**: 在 ONNX 导出 (`.half()`) 与 TensorRT 构建阶段强制开启 FP16，大幅降低计算开销，同时维持极高的检测与识别精度。
- **LPRNet 动态 Batch 处理 (Dynamic Shape)**: 打破了传统的一帧一车牌限制。当画面中同时出现多个车牌时，检测模块会将所有车牌直接拼接成 Tensor (`[N, 3, 24, 94]`) 一次性送入 LPRNet，显卡瞬时满载计算，拒绝循环排队排队推理。
- **Zero-Copy 显存级指针绑定**: 移除了多余的 CPU-GPU 数据拷贝，在 `main_trt.py` 中直接将 PyTorch 张量的显存地址 (`data_ptr()`) 绑定至 TensorRT Context，并利用 `execute_async_v3` 挂载到当前 CUDA Stream 实现纯异步推理。
- **内存池硬限制**: 在 Engine 编译期设定了 2GB 的 Workspace 显存上限，完美适配如 RTX 3060 Laptop (6GB VRAM) 等边缘显卡，防止显存溢出 (OOM)。

## 💻 测试硬件环境

- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, 95W TGP)
- **Driver Version**: 595.71
- **CUDA Version**: 13.2
- **OS**: Windows

## 安装 Python 依赖：
```
Bash
pip install -r requirements.txt
注意：请根据您的系统和 CUDA 13.2 环境，前往 PyTorch 官网 安装对应版本的 torch，并确保 tensorrt 库已正确配置。
```
## 🚀 快速开始项目的部署分为三个标准步骤
：ONNX 导出 -> TensorRT 引擎构建 -> 视频流推理。
Step 1: 导出 ONNX 模型将 PyTorch 的 .pt 和 .pth 权重文件转化为通用的 ONNX 格式，并自动完成 FP16 数据类型的降级操作：Bashpython export_onnx.py
成功后将生成 yolov5_best.onnx 和 lprnet_best.onnx。

Step 2: 构建 TensorRT 引擎 (Engine)调用 TensorRT Builder 对 ONNX 模型进行图优化、算子融合，并寻找当前硬件下的最优算子。这一步耗时较长：Bashpython export_tensorrt.py
成功后将生成 yolov5_best.engine 和 lprnet_best.engine。

Step 3: 高性能推理运行 main_trt.py 对图片、视频或实时 RTSP 流进行车牌识别推理：Bash# 测试默认图片文件夹，结果保存在 demo/rec_result_trt
python main_trt.py --source ./demo/images/ --view-img

## 部署结果
运行速度提升约48%，系统吞吐效率提升91%

## 鸣谢 (Acknowledgments)
本项目的检测与识别算法主体基于 [HuKai97/YOLOv5-LPRNet-Licence-Recognition](https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition) 进行二次开发。

本项目主要在其优秀的算法基础上，针对边缘计算场景（Edge Computing）进行了深度的工程化重构与部署优化，完成了从 PyTorch 动态图到 TensorRT 静态图的转换，并引入了动态 Batch 和显存零拷贝技术。在此特别感谢原作者的开源贡献！

This project is built upon the excellent work of [HuKai97/YOLOv5-LPRNet-Licence-Recognition](https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition). We extended the original repository by focusing on high-performance edge deployment, implementing TensorRT conversion, dynamic batching, and zero-copy memory management. Huge thanks to the original author for their open-source contribution!
