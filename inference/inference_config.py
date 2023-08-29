#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

# ==============================================
# ========== 机器设置 ==================
# ==============================================

# 是否启用 GPU 计算
useGPU = False

#使用哪个 GPU（如果只有一个 GPU，则设置为 True0）
gpu_num = 0

# 具有相同设置但不同随机种子的运行次数（如果适用）
Nruns = 1

# ==============================================
# ======= 数据集和模型设置 ===========
# ==============================================

# task = "imagenet"
# task = "cifar100"
# task = "cifar10"
task = "mnist"

# 根据任务选择神经网络模型
if task == "imagenet":
    # model_name = "Resnet50"
    model_name = "Resnet50-v1.5"
    # model_name = "Resnet50-int4"
    # model_name = "VGG19" ## weights will be downloaded from keras.applications
    # model_name = "InceptionV3"
    # model_name = "MobilenetV2"
    # model_name = "MobilenetV1"
    # model_name = "MobilenetV1-int8"

elif task == "cifar100":
    model_name = "ResNet56_cifar100"

elif task == "cifar10":
    # model_name = "cifar10_cnn_brelu"
    model_name = "ResNet14"
    # model_name = "ResNet20"
    # model_name = "ResNet32"
    # model_name = "ResNet56"

elif task == "mnist":
    model_name = "CNN6"
    # model_name = "CNN6_v2"

# 数据集截断
ntest = 1000 # 推理模拟中的图像数量
ntest_batch = 1000 # 一次在一个连续块中加载多少张图像（对于 ImageNet，应该 <=5000）
nstart = 0 # 起始图像索引

# 随机抽样： 
# 如果为 True，则将从完整数据集中随机选择 ntest 图像
# 如果为 False，图像将按照存储顺序加载
randomSampling = False

# 控制台输出True
# recordFalse 的 top-k 准确度
# count_interval: 每 N 张图像后打印的累积精度 (N = count_interval)
# time_interval：打印处理N张图像之间经过的时间
if task == "imagenet":
    count_interval = 1
    topk = (1,5)
elif task == "cifar10" or task == "cifar100":
    count_interval = 10
    topk = 1
elif task == "mnist":
    count_interval = 100
    topk = 1
time_interval = True

# 显示 Keras 模型摘要
show_model_summary = False

# ==============================================
# ========= Crossbar 配置 =============
# ==============================================

# 权值的分辨率
weight_bits = 8

# 每层中与最大器件电导相对应的权重分布的百分位数
# 一般来说，强烈建议weight_percentile = 100
weight_percentile = 100

# 位片数量
Nslices = 1

# 最大行数
NrowsMax = 1152

# 模拟输入位切片
input_bitslicing = False

# 负数处理（BALANCED 或 OFFSET）
style = "BALANCED"
# style = "OFFSET"

# “BALANCED”的特殊选项*和* Nslices = 1，否则被忽略
# one_side：零值将两个设备映射到最低状态（推荐）
#two_side：零值将两个设备映射到中心状态
balanced_style = "one_sided"

# “OFFSET”的特殊选项，否则被忽略
# 偏移量是否以数字方式计算（True）或使用模拟零点列（False）
digital_offset = False

# 阵列单元寄生电阻
Rp = 0

# 沿行的寄生电压降是否应设置为零
noRowParasitics = True

# 将阳性细胞和阴性细胞交错在一列中；忽略 if OFFSET
interleaved_posneg = False

# 将列底部的电流剪辑到范围 (-Icol_max,+Icol_max)
# 如果 input_bitslicing 为 True，则应用于每个输入位
# 如果是平衡的，则在当前减法之前应用，除非交错
Icol_max = 0
Icell_max = 3.2e-6 # 最大可能的电池电流；仅当 Icol_max > 0 时使用

# 将batchnorm折叠成conv/dense
fold_batchnorm = True

# 以数字方式与阵列方式实现偏置
digital_bias = True

# 偏差权重分辨率
# 0：无量化
# adc：如果 ADC 打开，则跟踪 ADC，否则不进行量化
bias_bits = 0
# bias_bits = "adc"

# ==============================================
# ========= 权重非理想值 ==============
# ==============================================

# 电池电导开/关比：0表示无穷大
On_off_ratio = 1000

###############
#
#   要使用自定义而不是通用设备错误模型，请实施
# 目录中的一个文件中的方法：
#/cross_sim/cross_sim/xbar_simulator/参数/custom_device/
# -- 编程错误：weight_error_device_custom.py
# -- 周期间读取噪声：weight_readnoise_device_custom.py
# -- 电导漂移：weight_drift_device_custom.py
# 有关更多详细信息，请参阅推理手册第 7 章。

### 编程误差
# error_model 可以是 (str): "none"、"alpha"（通用）或自定义设备模型
# 可用的设备型号有：“SONOS”、“PCM_Joshi”、“RRAM_Milo”
# 在weight_error_device_custom.py中定义
error_model = "alpha"
alpha_error = 0.01 # 仅当 error_model 为 alpha 时使用
proportional_error = False # 仅当 error_model 为 alpha 时使用

### 读取噪声
# Noise_model 可以是 (str): "none"、"alpha"（通用）或自定义设备模型
# 可用的设备模型有：“抛物线”（假设）
# 在weight_readnoise_device_custom.py中定义
noise_model = "alpha"
alpha_noise = 0.01 # 仅当 noise_model 为 alpha 时使用
proportional_noise = False # 仅当 noise_model 为 alpha 时使用

### 电导漂移
# 如果 t_drift = 0，则禁用漂移模型
#drift_model 可以是 (str): "none", 或自定义设备模型
# 可用的设备模型有：“SONOS_interpolate”、“PCM_Joshi”
# 在weight_drift_device_custom.py中定义
t_drift = 1 # 编程后的时间（天）
drift_model = 'none'

# ==============================================
# ===== ADC 和激活 (DAC) 设置 ======
# ==============================================

# 分辨率：0表示不量化
adc_bits = 8
dac_bits = 8
# adc_bits = 0
# dac_bits = 0

# 每个输入位后进行数字化（如果 input_bit_slicing 为 False，则忽略）
# 推荐设置：BALANCED -> ADC_per_ibit = False
# OFFSET -> ADC_per_ibit = True
ADC_per_ibit = (False if style == "BALANCED" else True)

# ADC 范围选项
# 校准：为每一层使用保存的 ADC 范围
# 对于非位切片，这给出了最小和最大 ADC 级别，无论 ADC 是否在每个输入位之后应用
# 对于位切片，这给出了 N_i（整数）的值，其中第 i^th 切片的极限 ADC 级别为 MAX / 2^N_i
# max: 将每层的 ADC 范围设置为最大可能值
#粒度：根据#位设置ADC范围以及与FPG对应的固定电平间隔（ADC_per_ibit必须为True）
adc_range_option = "calibrated"

# 如果使用位切片，则用于选择校准范围的百分位数选项（否则将被忽略）
# 典型值为 99.9、99.95、99.99、99.995 和 99.999
pct = 99.995