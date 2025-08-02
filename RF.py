import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

# 设置Matplotlib使用Agg后端（非交互式，避免Tkinter线程问题）
plt.switch_backend('Agg')


class CompositeThermalConductivityPredictor:
    """复合介质材料导热系数预测模型"""

    def __init__(self):
        """初始化模型和数据结构"""
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.df_original = None  # 保存原始数据
        self.train_indices = None  # 训练集索引
        self.test_indices = None  # 测试集索引

    def load_data(self, file_path, target_column="your_target"):
        """
        从Excel文件加载数据

        参数:
            file_path: Excel文件路径
            target_column: 目标变量列名
        """
        try:
            # 读取Excel文件
            self.df_original = pd.read_excel(file_path)

            # 检查目标变量列是否存在
            if target_column not in self.df_original.columns:
                raise ValueError(f"目标变量列 '{target_column}' 不存在于数据中")

            # 提取特征和目标变量
            self.feature_names = [col for col in self.df_original.columns if col != target_column]
            X = self.df_original[self.feature_names].values
            y = self.df_original[target_column].values

            # 划分训练集和测试集，同时保存索引
            self.X_train, self.X_test, self.y_train, self.y_test, self.train_indices, self.test_indices = train_test_split(
                X, y, np.arange(len(y)), test_size=0.2, random_state=42
            )

            print(f"数据加载成功，共{len(self.df_original)}条记录")
            print(f"特征变量: {', '.join(self.feature_names)}")
            print(f"目标变量: {target_column}")
            print(f"训练集: {len(self.y_train)}条记录，测试集: {len(self.y_test)}条记录")

            return True
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return False

    def preprocess_data(self):
        """数据预处理和特征标准化"""
        if self.X_train is None:
            print("请先加载数据")
            return False

        # 特征标准化
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("数据预处理完成")
        return True

    def train_model(self, n_estimators=100, max_depth=None, random_state=42):
        """
        训练随机森林回归模型

        参数:
            n_estimators: 决策树数量
            max_depth: 树的最大深度
            random_state: 随机种子
        """
        if self.X_train is None or self.y_train is None:
            print("请先加载和预处理数据")
            return False

        # 创建随机森林模型
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # 使用所有CPU核心
        )

        # 训练模型
        self.model.fit(self.X_train, self.y_train)

        # 计算特征重要性
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

        print(f"模型训练完成，决策树数量: {n_estimators}")
        return True

    def optimize_hyperparameters(self, param_grid=None):
        """
        使用网格搜索优化超参数

        参数:
            param_grid: 超参数网格
        """
        if self.model is None:
            print("请先初始化模型")
            return False

        # 默认超参数网格
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        # 网格搜索
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=2
        )

        grid_search.fit(self.X_train, self.y_train)

        # 更新模型为最佳参数模型
        self.model = grid_search.best_estimator_
        print(f"超参数优化完成，最佳参数: {grid_search.best_params_}")
        return grid_search.best_params_

    def evaluate_model(self):
        """评估模型性能"""
        if self.model is None:
            print("请先训练模型")
            return False

        # 在训练集和测试集上进行预测
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # 计算评估指标
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)

        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)

        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)

        # 打印评估结果
        print("\n模型评估结果:")
        print("=" * 50)
        print(f"训练集 MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"测试集 MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        print("=" * 50)

        # 可视化预测结果
        self._visualize_predictions(y_test_pred)

        # 可视化特征重要性
        self._visualize_feature_importance()

        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

    def _visualize_predictions(self, y_test_pred):
        """可视化预测结果"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_test_pred, alpha=0.7)
        plt.plot([min(self.y_test), max(self.y_test)],
                 [min(self.y_test), max(self.y_test)],
                 'r--', lw=2)
        plt.xlabel('实际复合导热系数 (W/m·K)')
        plt.ylabel('预测复合导热系数 (W/m·K)')
        plt.title('随机森林模型预测结果')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图像
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig('results/prediction_results.png', dpi=300)
        plt.close()

    def _visualize_feature_importance(self):
        """可视化特征重要性"""
        # 检查系统可用字体
        self._check_and_set_font()

        plt.figure(figsize=(12, 8))

        # 反转特征重要性序列，使重要性最高的特征显示在顶部
        reversed_importance = self.feature_importance.iloc[::-1]

        # 绘制水平条形图，重要性从高到低排列
        ax = reversed_importance.plot(kind='barh')

        # 添加数据标签
        for i, v in enumerate(reversed_importance):
            ax.text(v + 0.002, i, f'{v:.4f}', va='center')

        plt.xlabel('特征重要性')
        plt.ylabel('特征名称')
        plt.title('随机森林模型特征重要性')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图像
        plt.savefig('results/feature_importance.png', dpi=300)
        plt.close()

    def _check_and_set_font(self):
        """检查系统可用字体并设置中文字体"""
        import matplotlib.font_manager as fm

        # 列出所有可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # 常用中文字体列表
        chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC',
                         'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei']

        # 尝试找到系统中存在的中文字体
        found_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                found_font = font
                break

        if found_font:
            plt.rcParams["font.family"] = found_font
            print(f"已设置中文字体: {found_font}")
        else:
            print("未找到合适的中文字体，图表中的中文可能无法正确显示")
            print(f"可用字体列表: {available_fonts[:20]}...")  # 显示前20个字体

    def predict(self, new_data):
        """
        使用训练好的模型进行预测

        参数:
            new_data: 特征数据，格式为DataFrame或numpy数组
        """
        if self.model is None:
            print("请先训练模型")
            return None

        # 处理输入数据
        if isinstance(new_data, pd.DataFrame):
            # 确保列名顺序正确
            if list(new_data.columns) != self.feature_names:
                new_data = new_data[self.feature_names]
            new_data = new_data.values

        # 标准化数据
        new_data_scaled = self.scaler.transform(new_data)

        # 预测
        predictions = self.model.predict(new_data_scaled)

        return predictions

    def save_prediction_to_excel(self, output_path=None):
        """
        保存原始数据、训练集/测试集划分以及预测结果到Excel

        参数:
            output_path: 输出Excel文件路径
        """
        if self.model is None or self.df_original is None:
            print("请先训练模型并加载数据")
            return False

        # 创建预测结果DataFrame
        df_result = self.df_original.copy()

        # 添加数据集划分标记
        df_result['数据集'] = '未知'
        df_result.loc[self.train_indices, '数据集'] = '训练集'
        df_result.loc[self.test_indices, '数据集'] = '测试集'

        # 计算所有样本的预测值
        all_predictions = np.zeros(len(df_result))

        # 训练集预测
        train_predictions = self.model.predict(self.X_train)
        all_predictions[self.train_indices] = train_predictions

        # 测试集预测
        test_predictions = self.model.predict(self.X_test)
        all_predictions[self.test_indices] = test_predictions

        # 添加预测结果列
        df_result['预测复合导热系数'] = all_predictions

        # 添加预测误差列
        df_result['绝对误差'] = np.abs(df_result['复合导热系数'] - df_result['预测复合导热系数'])

        # 保存到Excel
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/prediction_results_{timestamp}.xlsx"

        # 创建结果目录
        result_dir = os.path.dirname(output_path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # 保存Excel文件
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 保存完整结果
            df_result.to_excel(writer, sheet_name='完整数据集', index=False)

            # 保存训练集结果
            df_train = df_result[df_result['数据集'] == '训练集']
            df_train.to_excel(writer, sheet_name='训练集', index=False)

            # 保存测试集结果
            df_test = df_result[df_result['数据集'] == '测试集']
            df_test.to_excel(writer, sheet_name='测试集', index=False)

            # 保存特征重要性
            df_importance = pd.DataFrame({
                '特征名称': self.feature_importance.index,
                '重要性': self.feature_importance.values
            })
            df_importance.to_excel(writer, sheet_name='特征重要性', index=False)

        print(f"预测结果已保存至: {os.path.abspath(output_path)}")
        return True

    def save_model(self, model_path=None):
        """保存模型和标准化器"""
        if self.model is None:
            print("请先训练模型")
            return False

        # 默认路径
        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/random_forest_model_{timestamp}.joblib"

        # 创建模型目录
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 保存模型和标准化器
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, model_path)

        print(f"模型已保存至: {model_path}")
        return True

    @staticmethod
    def load_saved_model(model_path):
        """加载已保存的模型"""
        try:
            # 加载模型和标准化器
            loaded = joblib.load(model_path)

            # 创建新的预测器实例
            predictor = CompositeThermalConductivityPredictor()
            predictor.model = loaded['model']
            predictor.scaler = loaded['scaler']
            predictor.feature_names = loaded['feature_names']

            print(f"模型已从 {model_path} 加载")
            return predictor
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return None


# 示例使用
if __name__ == "__main__":
    # 创建预测器实例
    predictor = CompositeThermalConductivityPredictor()

    # 加载数据
    if predictor.load_data("file_name"):
        # 数据预处理
        predictor.preprocess_data()

        # 训练模型
        predictor.train_model(n_estimators=100)

        # 超参数优化（可选）
        # best_params = predictor.optimize_hyperparameters()

        # 评估模型
        metrics = predictor.evaluate_model()

        # 保存预测结果到Excel
        predictor.save_prediction_to_excel()

        # 保存模型
        predictor.save_model()

        # 示例预测
        # 假设我们有一组新的材料特性数据，自行填写
        new_material_data = pd.DataFrame({
            "基体导热系数": [],
            "增强体导热系数": [],
            "填料占比": [],
            "基体密度": [],
            "增强体密度": [],
            "基体热膨胀系数": [],
            "增强体热膨胀系数": [],
            "基体比热容": [],
            "增强体比热容": [],
            "基体热扩散率": [],
            "增强体热扩散率": [],
            "基体玻璃化转变温度": [],
            "填料尺寸": []
        })

        # 预测新材料的导热系数
        predicted_thermal_conductivity = predictor.predict(new_material_data)
        print(f"\n示例预测: 新复合材料的导热系数预测值为 {predicted_thermal_conductivity[0]:.4f} W/m·K")
