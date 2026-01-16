"""
时间处理工具模块

本模块提供时间格式转换和时间序列生成功能，主要用于水文模型数据处理。
支持多种时间格式之间的转换，以及时间范围的数组化处理。

主要功能：
- 时间格式转换：支持整数日期、datetime.date、datetime.datetime之间的转换
- 时间序列生成：根据时间范围生成等间隔的时间数组
- 为CAMELS数据集和dPLHBV模型提供标准化的时间处理接口
"""

import datetime as dt
import numpy as np


def t2dt(t, hr=False):
    """
    时间格式转换函数
    
    将各种时间格式转换为标准化的日期或日期时间格式。
    主要用于统一不同来源的时间数据格式，确保后续处理的一致性。
    
    参数:
        t: 输入时间，支持以下格式：
           - int: 整数日期格式 (YYYYMMDD)，如 20200101
           - datetime.date: Python日期对象
           - datetime.datetime: Python日期时间对象
        hr (bool): 是否返回包含小时信息的datetime对象
                  - False: 返回date对象（默认）
                  - True: 返回datetime对象
    
    返回:
        datetime.date 或 datetime.datetime: 转换后的时间对象
        
    异常:
        Exception: 当输入时间格式无法识别时抛出异常
        
    示例:
        >>> t2dt(20200101)  # 返回 datetime.date(2020, 1, 1)
        >>> t2dt(20200101, hr=True)  # 返回 datetime.datetime(2020, 1, 1, 0, 0)
        >>> t2dt(dt.date(2020, 1, 1))  # 返回 datetime.date(2020, 1, 1)
    """
    tOut = None
    
    # 处理整数日期格式 (YYYYMMDD)
    if type(t) is int:
        # 检查是否为有效的8位日期格式 (10000000 < date < 30000000)
        if t < 30000000 and t > 10000000:
            # 将整数转换为字符串，然后解析为日期
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            # 根据hr参数决定返回格式
            tOut = t if hr is False else t.datetime()

    # 处理datetime.date对象
    if type(t) is dt.date:
        # 根据hr参数决定返回格式
        tOut = t if hr is False else t.datetime()

    # 处理datetime.datetime对象
    if type(t) is dt.datetime:
        # 根据hr参数决定返回格式
        tOut = t.date() if hr is False else t

    # 如果所有格式都无法识别，抛出异常
    if tOut is None:
        raise Exception('hydroMLlib.utils.t2dt failed')
    return tOut


def tRange2Array(tRange, *, step=np.timedelta64(1, 'D')):
    """
    时间范围转数组函数
    
    根据给定的时间范围生成等间隔的时间序列数组。
    主要用于生成连续的时间序列，为水文模型提供时间轴数据。
    
    参数:
        tRange (list/tuple): 时间范围，包含两个元素 [开始时间, 结束时间]
                            - 开始时间和结束时间可以是任何t2dt函数支持的格式
                            - 结束时间不包含在生成的数组中（左闭右开区间）
        step (numpy.timedelta64): 时间步长，默认为1天
                                - 默认值: np.timedelta64(1, 'D') 表示1天
                                - 可以修改为其他时间间隔，如小时、分钟等
    
    返回:
        numpy.ndarray: 时间序列数组，包含从开始时间到结束时间的所有时间点
                      - 数组类型为numpy.datetime64
                      - 时间间隔由step参数控制
                      - 不包含结束时间点
    
    示例:
        >>> tRange2Array([20200101, 20200105])  
        # 返回: array(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'], dtype='datetime64[D]')
        >>> tRange2Array([20200101, 20200105], step=np.timedelta64(2, 'D'))
        # 返回: array(['2020-01-01', '2020-01-03'], dtype='datetime64[D]')
        
    注意:
        - 此函数内部调用t2dt函数进行时间格式转换
        - 生成的时间数组用于CAMELS数据集的时间轴处理
        - 时间步长必须是numpy.timedelta64类型
    """
    # 将时间范围的两个端点转换为标准日期格式
    sd = t2dt(tRange[0])  # 开始日期 (start date)
    ed = t2dt(tRange[1])  # 结束日期 (end date)
    
    # 使用numpy.arange生成时间序列数组
    # 从开始日期到结束日期，按指定步长生成
    tArray = np.arange(sd, ed, step)
    return tArray




