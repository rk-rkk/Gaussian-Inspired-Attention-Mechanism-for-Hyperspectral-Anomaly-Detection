import numpy as np


def AUC_cal(map, detection):
    M = map.size  # 使用size属性代替shape方法
    mask = map.ravel()  # 使用ravel方法代替reshape方法
    anomaly_map = mask >= 1
    normal_map = mask == 0
    r0 = detection.ravel()
    r_max = r0.max()
    step = 5000
    taus = np.linspace(0, r_max, step)
    anomaly_map_rx = r0 >= taus[:, np.newaxis]  # 使用向量化操作代替循环
    PF = np.sum(anomaly_map_rx & normal_map, axis=1) / np.sum(normal_map)  # 使用布尔索引和内置函数代替条件判断和自定义函数
    PD = np.sum(anomaly_map_rx & anomaly_map, axis=1) / np.sum(anomaly_map)
    area = np.trapz(PD, x=1 - PF)  # 使用trapz函数代替自定义的梯形积分公式
    return area

# import numpy as np
#
#
# def AUC_cal(map, detection):
#     w, h = np.shape(map)
#     M = w * h
#     mask = map
#     mask = np.reshape(mask, [1, M])  # mask表示的是标签
#     anomaly_map = (mask) >= 1
#     normal_map = (mask) == 0
#     r0 = np.reshape(detection, [1, M])
#     r_max = detection.max()
#     step = 5000
#     taus = np.linspace(0, r_max, step)
#     PF = np.zeros([step], dtype=float)
#     PD = np.zeros([step], dtype=float)
#     for index in range(0, step):
#         tau = taus[index]
#         anomaly_map_rx = (r0 >= tau)
#         PF[index] = sum(sum(anomaly_map_rx & normal_map)) * 1.0 / sum(sum(normal_map))
#         PD[index] = sum(sum(anomaly_map_rx & anomaly_map)) * 1.0 / sum(sum(anomaly_map))
#     area = sum((PF[0:step - 1] - PF[1:step]) * (PD[0:step - 1] + PD[1:step])) / 2
#     return area