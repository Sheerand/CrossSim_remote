#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
import os, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"


from inference_util.inference_net import set_params, inference
from inference_util.keras_parser import get_keras_metadata
from inference_util.CNN_setup import augment_parameters, build_keras_model, model_specific_parameters, \
    get_xy_parallel, get_xy_parallel_parasitics, load_adc_activation_ranges
from inference_util.print_configuration_message import print_configuration_message

import helpers.plot_tools as PT

# ==========================
# ==== 加载配置文件 ====
# ==========================

import inference_config as config

# ===================
# ==== GPU设置 ====
# ===================

#限制TensorFlow GPU内存使用
os.environ["CUDA_VISIBLE_DEVICES"]=str(-1)
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for k in range(len(gpu_devices)):
    tf.config.experimental.set_memory_growth(gpu_devices[k], True)

# 设置 GPU
if config.useGPU:
    import cupy
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu_num)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    cupy.cuda.Device(0).use()

# =====================================
# ==== 导入神经网络模型 ====
# =====================================

# 构建 Keras 模型并准备与 CrossSim 兼容的拓扑元数据
keras_model = build_keras_model(config.model_name,show_model_summary=config.show_model_summary)
layerParams, sizes = get_keras_metadata(keras_model,task=config.task,debug_graph=False)

# 统计总层数和MVM层数
Nlayers = len(layerParams)
config.Nlayers_mvm = np.sum([(layerParams[j]['type'] in ('conv','dense')) for j in range(Nlayers)])

# ===================================================================
# ======= 参数验证和模型特定参数 ========
# ===================================================================
# 一般参数检查
config = augment_parameters(config)
# 某些神经网络特有的参数
config, positiveInputsOnly = model_specific_parameters(config)

# =========================================================
# ==== 加载 ADC 和激活的校准范围 ====
# =========================================================

adc_ranges, dac_ranges = load_adc_activation_ranges(config)

# =======================================
# ======= GPU性能调优 ========
# =======================================

# 卷积：沿 x 和 y 并行计算的滑动窗口数量
xy_pars = get_xy_parallel(config)
# print(xy_pars)

# ================================
# ========= 开始扫描 ==========
# ================================

# 显示所选的模拟设置
print_configuration_message(config)

for q in range(config.Nruns):

    if config.Nruns > 1:
        print('')
        print('===========')
        print(" Run "+str(q+1)+"/"+str(config.Nruns))
        print('===========')

    paramsList, layerParamsCopy = Nlayers*[None], Nlayers*[None]#初始化长度为Nlayers的列表
    j_mvm, j_conv = 0, 0 # MVM 和 conv 层的计数器

    # ===================================================
    # ==== 计算并设置特定于层的参数 ====
    # ===================================================

    for j in range(Nlayers):

        # 对于必须拆分为多个数组的图层，创建多个 params 对象
        if layerParams[j]['type'] in ('conv','dense'):

            # MVM 中使用的总行数
            if layerParams[j]['type'] == 'conv':
                Nrows = layerParams[j]['Kx']*layerParams[j]['Ky']*layerParams[j]['Nic']
            elif layerParams[j]['type'] == 'dense':
                Nrows = sizes[j][2]

            #计算数组矩阵的数量必须划分
            if config.NrowsMax > 0:
                Ncores = (Nrows-1)//config.NrowsMax + 1
            else:
                Ncores = 1

            # 特定于层的 ADC 以及激活分辨率和范围（上面设置）
            adc_range = adc_ranges[j_mvm]
            dac_range = dac_ranges[j_mvm]
            adc_bits_j = config.adc_bits_vec[j_mvm]
            dac_bits_j = config.dac_bits_vec[j_mvm]
            Rp_j = config.Rp # 原则上，每层可以有不同的寄生电阻值

            # 如果启用寄生效应，则修改 x_par 和 y_par 以优化累积和运行时间
            if Rp_j > 0 and layerParams[j]['type'] == 'conv':
                xy_pars[j_conv,:] = get_xy_parallel_parasitics(Nrows,sizes[j][0],sizes[j+1][0],config.model_name)

            if layerParams[j]['type'] == 'conv':
                x_par, y_par = xy_pars[j_conv,:]
                convParams = layerParams[j]

            elif layerParams[j]['type'] == 'dense':
                x_par, y_par = 1, 1
                convParams = None

            params = set_params(task=config.task,
                wtmodel=config.style,
                convParams=convParams,
                alpha_noise=config.alpha_noise,
                balanced_style=config.balanced_style,
                ADC_per_ibit=config.ADC_per_ibit,
                x_par=x_par,
                y_par=y_par,
                weight_bits=config.weight_bits,
                useGPU=config.useGPU,
                proportional_noise=config.proportional_noise,
                alpha_error=config.alpha_error,
                adc_bits=adc_bits_j,
                dac_bits=dac_bits_j,
                adc_range=adc_range,
                dac_range=dac_range,
                error_model=config.error_model,
                noise_model=config.noise_model,
                NrowsMax=config.NrowsMax,
                positiveInputsOnly=positiveInputsOnly[j_mvm],
                input_bitslicing=config.input_bitslicing,
                fast_balanced=config.fast_balanced,
                noRowParasitics=config.noRowParasitics,
                interleaved_posneg=config.interleaved_posneg,
                t_drift=config.t_drift,
                drift_model=config.drift_model,
                Rp=Rp_j,
                digital_offset=config.digital_offset,
                Icol_max=config.Icol_max/config.Icell_max,
                On_off_ratio=config.On_off_ratio,
                adc_range_option=config.adc_range_option,
                proportional_error=config.proportional_error,
                Nslices=config.Nslices,
                digital_bias=config.digital_bias)

            if Ncores == 1:
                paramsList[j] = params
            else:
                paramsList[j] = Ncores*[None]
                for k in range(Ncores):
                    paramsList[j][k] = params.copy()            
            
            j_mvm += 1
            if layerParams[j]['type'] == 'conv':
                j_conv += 1

         # 需要复制一份以防止inference()修改layerParams
        layerParamsCopy[j] = layerParams[j].copy()

    # 运行推理
    accuracy = inference(ntest=config.ntest,
        dataset=config.task,
        paramsList=paramsList,
        sizes=sizes,
        keras_model=keras_model,
        layerParams=layerParamsCopy,
        useGPU=config.useGPU,
        count_interval=config.count_interval,
        weight_percentile=config.weight_percentile,
        randomSampling=config.randomSampling,
        topk=config.topk,
        subtract_pixel_mean=config.subtract_pixel_mean,
        memory_window=config.memory_window,
        model_name=config.model_name,
        fold_batchnorm=config.fold_batchnorm,
        digital_bias=config.digital_bias,
        nstart=config.nstart,
        ntest_batch=config.ntest_batch,
        bias_bits=config.bias_bits,
        time_interval=config.time_interval,
        imagenet_preprocess=config.imagenet_preprocess,
        dataset_normalization=config.dataset_normalization,
        adc_range_option=config.adc_range_option,
        larq=config.larq,
        whetstone=config.whetstone)
    
p = PT.Plot()
