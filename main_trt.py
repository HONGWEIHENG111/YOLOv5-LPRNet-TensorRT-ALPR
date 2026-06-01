import argparse
import os
import shutil
import time
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import tensorrt as trt

# 复用你项目原有的工具包
from utils.datasets import *
from utils.utils import *

# 车牌字符字典
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-']


# 🌟 核心封装：支持动态 Batch 的 TensorRT 显存管理
class TRTWrapper:
    def __init__(self, engine_path, device=torch.device('cuda:0')):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.device = device

    def infer(self, input_tensor):
        # 1. 告诉 TRT 当前送进来的真实形状 (比如传入了 3 个车牌，就是 [3, 3, 24, 94])
        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, tuple(input_tensor.shape))

        # 将 PyTorch 张量的显存地址直接绑定给 TRT (零拷贝)
        self.context.set_tensor_address(input_name, input_tensor.contiguous().data_ptr())

        outputs = []
        # 2. 动态分配输出显存
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.context.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
                # 根据 TRT 算好的输出尺寸，在显卡上挖一块对应大小的坑
                out_tensor = torch.empty(tuple(shape), dtype=torch_dtype, device=self.device)
                self.context.set_tensor_address(name, out_tensor.data_ptr())
                outputs.append(out_tensor)

        # 3. 异步触发计算
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)

        return outputs


# 车牌解码逻辑
def decode_lprnet(preds):
    preds = preds.reshape((68, 18))  # [class_num, lpr_max_len]
    prebs_labels = np.argmax(preds, axis=0)

    no_repeat_blank_label = []
    pre_c = prebs_labels[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in prebs_labels:
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    return "".join([CHARS[i] for i in no_repeat_blank_label])


def detect(opt):
    out, source, view_img, save_txt, imgsz = opt.output, opt.source, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # 初始化设备
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # 1. 加载 TensorRT 引擎
    print("🚀 正在加载 YOLOv5 TensorRT 引擎...")
    model_yolo = TRTWrapper(opt.det_engine, device=device)
    print("🚀 正在加载 LPRNet TensorRT 引擎...")
    model_lprnet = TRTWrapper(opt.rec_engine, device=device)

    # 2. 设置数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True
        # 同样关闭摄像头的自适应填充
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        # 强制将图像缩放并填充为标准的 640x640，关闭 stride 自适应，关闭自适应填充
        dataset = LoadImages(source, img_size=imgsz)

    names = ['plate']  # 假设类别为车牌
    colors = [[0, 255, 0]]

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # 我们无视 dataset 吐出来的尺寸不对的 img，直接拿原图 (im0s) 重新做严格的 letterbox
        if webcam:
            img = np.stack([letterbox(x, new_shape=(imgsz, imgsz), auto=False)[0] for x in im0s], 0)
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # B,H,W,C -> B,C,H,W
        else:
            img = letterbox(im0s, new_shape=(imgsz, imgsz), auto=False)[0]
            img = img[:, :, ::-1].transpose((2, 0, 1))  # H,W,C -> C,H,W

        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half()  # TensorRT 必须使用 FP16
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = torch_utils.time_synchronized()

        # [YOLO 推理]
        pred = model_yolo.infer(img)[0]

        # [YOLO NMS 后处理]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]

            if det is not None and len(det):
                # 将预测框还原到原图尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # [重点：循环提取车牌并进行 LPRNet 识别]
                # 🌟【重点修改】：创建一个列表收集当前图里的所有车牌
                lpr_inputs = []
                plate_boxes = []

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)

                    # 防止越界
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(im0.shape[1], x2), min(im0.shape[0], y2)

                    if x2 > x1 and y2 > y1:
                        # 从原图抠出车牌区域
                        crop_img = im0[y1:y2, x1:x2]

                        # [LPRNet 预处理 - 严格对齐原版 PyTorch]
                        img_resized = cv2.resize(crop_img, (94, 24))
                        img_numpy = img_resized.astype(np.float32)
                        img_numpy -= 127.5
                        img_numpy *= 0.0078125
                        img_numpy = img_numpy.transpose((2, 0, 1))  # C,H,W

                        lpr_inputs.append(img_numpy)
                        plate_boxes.append((xyxy, conf, cls))  # 保存对应的框信息

                # 🌟【并行推理】：如果画面里有车牌，一次性送给显卡！
                if len(lpr_inputs) > 0:
                    # 将所有车牌拼接成一个大 Tensor，例如形状变成 [N, 3, 24, 94]
                    lpr_input_tensor = torch.from_numpy(np.array(lpr_inputs)).to(device).half()

                    # 一次性调用 TRT 推理！显卡瞬间满载计算！
                    lpr_outs = model_lprnet.infer(lpr_input_tensor)[0]

                    # 仅仅在这里执行一次同步拷贝，将 N 个结果同时拿回 CPU
                    lpr_outs_np = lpr_outs.cpu().numpy()

                    # 解析结果并画图
                    for idx in range(len(lpr_inputs)):
                        # 逐个解码
                        plate_text = decode_lprnet(lpr_outs_np[idx])

                        # 取出当时存下来的框信息
                        xyxy, conf, cls = plate_boxes[idx]

                        # 绘制带文字的检测框
                        label = f'{plate_text} {conf:.2f}'
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            t2 = torch_utils.time_synchronized()
            print(f'{s}Done. ({(t2 - t1):.3f}s)')

            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        fourcc = 'mp4v'
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    print(f'Done. ({(time.time() - t0):.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 替换原本的 .pt 权重为 .engine 权重
    parser.add_argument('--det-engine', type=str, default='yolov5_best.engine', help='yolov5 engine path')
    parser.add_argument('--rec-engine', type=str, default='lprnet_best.engine', help='lprnet engine path')
    parser.add_argument('--source', type=str, default='./demo/images/', help='source')
    parser.add_argument('--output', type=str, default='demo/rec_result_trt', help='output folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt)