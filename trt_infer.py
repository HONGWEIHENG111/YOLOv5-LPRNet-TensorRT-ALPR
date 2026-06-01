import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTWrapper:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, runtime, engine_path):
        with open(engine_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # 分配主机（CPU）内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 分配设备（GPU）显存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # 记录绑定地址
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        # 1. 将数据平铺并复制到主机内存
        np.copyto(self.inputs[0]['host'], input_data.ravel())

        # 2. 将数据从主机 (CPU) 拷贝到设备 (GPU)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # 3. 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 4. 将结果从设备 (GPU) 拷贝回主机 (CPU)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # 5. 同步流，等待执行完成
        self.stream.synchronize()

        # 返回所有输出结果 (按网络输出层的顺序)
        return [out['host'] for out in self.outputs]

    def __del__(self):
        # 释放显存
        self.context.pop()
        del self.context
        del self.engine
        del self.runtime