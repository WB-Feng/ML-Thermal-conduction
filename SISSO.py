import numpy as np
import pandas as pd
from TorchSisso import SissoModel
from sklearn.feature_selection import mutual_info_regression
import sympy
from sympy import symbols

# 读取包含单个工作表的 Excel 文件
file_path = 'file_name'
data = pd.read_excel(file_path)

# 提取特征和目标变量
target_variable = 'your_target'
X = data.drop(target_variable, axis=1)
y = data[target_variable]

# 定义每个特征对应的维度，增加新特征的维度，根据实际情况更改
dimensionality = [
    'W/(m*K)',
    'W/(m*K)',
    '1',
    'Kg/m^3',
    'Kg/m^3',
    '1/K',
    '1/K',
    'J/(Kg*K)',
    'J/(Kg*K)',
    'm^2/s',
    'm^2/s',
    'K',
    'm',

]

# 计算每个特征与目标变量之间的互信息
mi_scores = mutual_info_regression(X, y)

# 创建一个字典，存储特征名和对应的互信息得分
mi_dict = dict(zip(X.columns, mi_scores))

# 打印每个特征的互信息得分
print("各特征与目标变量的互信息得分：")
for feature, score in mi_dict.items():
    print(f"特征 {feature} 与目标变量的互信息得分: {score}")

# 定义操作符
operators = ['+', '-', '*', '/', 'pow(2)', 'pow(3)','pow(-1)','sqrt','exp','exp(-1)','log',
             'abs', 'round']

# 创建 SissoModel 对象，根据实际情况更改
sm = SissoModel(
    data=data,
    operators=operators,
    n_expansion=5,
    n_term=3,
    k=50,
    initial_screening=["mi", 0.3],
    use_gpu=True,
    dimensionality=dimensionality,
    relational_units=[],
    output_dim=(symbols('W/(m*K)')),
    custom_unary_functions=[],
    custom_binary_functions=[]
)

# 运行 SISSO 算法
try:
    results = sm.fit()
    rmse, equation, r2 = results[:3]
    print(f"均方根误差 (RMSE): {rmse}")
    print(f"最终方程: {equation}")
    print(f"决定系数 (R2): {r2}")
except ValueError as e:
    print(f"解包返回值时出现错误: {e}")
except Exception as e:
    print(f"运行过程中出现其他错误: {e}")
