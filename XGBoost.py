# 注释部分为中文，部分位置参数、文件名等需要根据实际情况更改
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  


class CompositeMaterialThermalConductivityPredictor:
    """复合介质材料导热系数预测模型"""

    def __init__(self):
        """初始化模型和相关参数"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.best_params = None
        self.train_indices = None
        self.test_indices = None

    def load_data(self, file_path, sheet_name=0):
        """
        从Excel文件加载数据

        参数:
            file_path: Excel文件路径
            sheet_name: 工作表名称或索引
        """
        try:
            self.data = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"数据加载成功，共{self.data.shape[0]}条记录，{self.data.shape[1]}个特征")
            return self.data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    def preprocess_data(self, target_column="your_target"):
        """
        数据预处理和特征工程

        参数:
            target_column: 目标变量列名
        """
        if not hasattr(self, 'data'):
            print("请先加载数据")
            return False

        # 分离特征和目标变量
        self.X = self.data.drop(target_column, axis=1)
        self.y = self.data[target_column]

        # 处理缺失值
        if self.X.isnull().sum().sum() > 0:
            print(f"检测到缺失值，正在处理...")
            self.X = self.X.fillna(self.X.mean())

        # 特征标准化
        self.X_scaled = self.scaler.fit_transform(self.X)

        # 保存特征名称
        self.feature_names = self.X.columns.tolist()

        print(f"数据预处理完成，特征数量: {len(self.feature_names)}")
        return True

    def split_data(self, test_size=0.2, random_state=42):
        """
        划分训练集和测试集

        参数:
            test_size: 测试集比例
            random_state: 随机种子
        """
        if not hasattr(self, 'X_scaled'):
            print("请先进行数据预处理")
            return False

        # 保存索引用于后续结果匹配
        self.train_indices, self.test_indices, _, _ = train_test_split(
            range(len(self.y)), self.y, test_size=test_size, random_state=random_state
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )

        print(f"数据划分完成，训练集: {self.X_train.shape[0]}条，测试集: {self.X_test.shape[0]}条")
        return True

    def hyperparameter_tuning(self, cv=5):
        """
        超参数优化

        参数:
            cv: 交叉验证折数
        """
        if not hasattr(self, 'X_train'):
            print("请先划分数据集")
            return False

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=cv,
            n_jobs=-1,
            verbose=2
        )

        print("开始超参数优化...")
        grid_search.fit(self.X_train, self.y_train)

        self.best_params = grid_search.best_params_
        print(f"最优参数: {self.best_params}")
        print(f"最优RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

        return self.best_params

    def train_model(self, use_best_params=True):
        """
        训练XGBoost模型

        参数:
            use_best_params: 是否使用最优参数
        """
        if not hasattr(self, 'X_train'):
            print("请先划分数据集")
            return False

        if use_best_params and hasattr(self, 'best_params'):
            print("使用优化后的参数训练模型")
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                **self.best_params
            )
        else:
            print("使用默认参数训练模型")
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8
            )

        # 打印XGBoost版本，便于调试
        xgb_version = xgb.__version__
        print(f"当前使用的XGBoost版本: {xgb_version}")

        # 简化版本检查逻辑
        try:
            # 尝试使用包含early_stopping_rounds的训练方法
            self.model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                early_stopping_rounds=10,
                verbose=1
            )
        except TypeError as e:
            print(f"错误: {e}")
            print("检测到XGBoost版本不支持early_stopping_rounds参数，尝试使用兼容模式...")
            try:
                # 尝试不使用early_stopping_rounds，但保留eval_set
                self.model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_test, self.y_test)],
                    verbose=1
                )
            except Exception as e:
                # 作为最后的手段，完全不使用评估集
                print(f"警告: 进一步降级训练模式，错误: {e}")
                self.model.fit(
                    self.X_train, self.y_train,
                    verbose=1
                )

        # 保存特征重要性
        self.feature_importance = self.model.feature_importances_

        print("模型训练完成")
        return True

    def evaluate_model(self):
        """评估模型性能"""
        if self.model is None:
            print("请先训练模型")
            return False

        # 在训练集和测试集上进行预测
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # 计算评估指标
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)

        print("\n模型评估结果:")
        print(f"训练集 RMSE: {train_rmse:.4f}")
        print(f"测试集 RMSE: {test_rmse:.4f}")
        print(f"训练集 R²: {train_r2:.4f}")
        print(f"测试集 R²: {test_r2:.4f}")

        # 可视化预测结果
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(self.y_train, y_train_pred, alpha=0.7)
        plt.plot([self.y_train.min(), self.y_train.max()],
                 [self.y_train.min(), self.y_train.max()], 'r--')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('训练集预测结果')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.scatter(self.y_test, y_test_pred, alpha=0.7)
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('测试集预测结果')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('prediction_results.png')
        plt.close()

        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            '特征': self.feature_names,
            '重要性': self.feature_importance
        }).sort_values('重要性', ascending=False)

        sns.barplot(x='重要性', y='特征', data=importance_df)

        # 设置横坐标范围为0到0.8
        plt.xlim(0, 0.8)

        plt.title('特征重要性分析')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # 保存预测结果到DataFrame
        self.prediction_results = pd.DataFrame(index=self.data.index)
        self.prediction_results['样本类型'] = ''
        self.prediction_results.loc[self.train_indices, '样本类型'] = '训练集'
        self.prediction_results.loc[self.test_indices, '样本类型'] = '测试集'

        # 添加实际值和预测值
        self.prediction_results['实际复合导热系数'] = self.y
        self.prediction_results.loc[self.train_indices, '预测复合导热系数'] = y_train_pred
        self.prediction_results.loc[self.test_indices, '预测复合导热系数'] = y_test_pred

        # 计算残差
        self.prediction_results['残差'] = (
                self.prediction_results['实际复合导热系数'] -
                self.prediction_results['预测复合导热系数']
        )

        # 计算相对误差(%)
        self.prediction_results['相对误差(%)'] = (
                self.prediction_results['残差'] /
                self.prediction_results['实际复合导热系数'] * 100
        )

        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': importance_df
        }

    def save_prediction_results(self, output_path='prediction_results.xlsx'):
        """
        保存预测结果到Excel文件

        参数:
            output_path: 输出Excel文件路径
        """
        if not hasattr(self, 'prediction_results'):
            print("请先评估模型以生成预测结果")
            return False

        try:
            # 创建保存目录
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 保存结果到Excel
            self.prediction_results.to_excel(output_path, index_label='样本索引')
            print(f"预测结果已保存至: {output_path}")
            return True
        except Exception as e:
            print(f"保存预测结果失败: {e}")
            return False

    def predict(self, new_data):
        """
        对新数据进行预测

        参数:
            new_data: 待预测数据，可以是DataFrame或与训练数据相同格式的数组
        """
        if self.model is None:
            print("请先训练模型")
            return None

        # 确保输入是DataFrame
        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame(new_data, columns=self.feature_names)

        # 数据标准化
        new_data_scaled = self.scaler.transform(new_data)

        # 预测
        predictions = self.model.predict(new_data_scaled)

        return predictions

    def save_model(self, model_dir='models'):
        """
        保存模型和相关对象

        参数:
            model_dir: 模型保存目录
        """
        if self.model is None:
            print("请先训练模型")
            return False

        # 创建保存目录
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存模型
        model_path = os.path.join(model_dir, f"xgboost_model_{timestamp}.joblib")
        joblib.dump(self.model, model_path)

        # 保存特征名称
        features_path = os.path.join(model_dir, f"features_{timestamp}.txt")
        with open(features_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.feature_names))

        # 保存标准化器
        scaler_path = os.path.join(model_dir, f"scaler_{timestamp}.joblib")
        joblib.dump(self.scaler, scaler_path)

        print(f"模型已保存至: {model_path}")
        return True

    @staticmethod
    def load_saved_model(model_path, features_path, scaler_path):
        """
        加载已保存的模型

        参数:
            model_path: 模型文件路径
            features_path: 特征名称文件路径
            scaler_path: 标准化器文件路径
        """
        # 创建新的预测器实例
        predictor = CompositeMaterialThermalConductivityPredictor()

        # 加载模型
        predictor.model = joblib.load(model_path)

        # 加载特征名称
        with open(features_path, 'r', encoding='utf-8') as f:
            predictor.feature_names = [line.strip() for line in f.readlines()]

        # 加载标准化器
        predictor.scaler = joblib.load(scaler_path)

        print(f"模型已从 {model_path} 加载")
        return predictor


def main():
    """主函数，演示模型的使用流程"""
    # 创建预测器实例
    predictor = CompositeMaterialThermalConductivityPredictor()

    # 加载数据 - 请替换为你的Excel文件路径
    data = predictor.load_data("file_name")

    if data is not None:
        # 数据预处理
        predictor.preprocess_data(target_column="your_target")

        # 划分数据集
        predictor.split_data(test_size=0.2)

        # 超参数优化 (可选，可能需要较长时间)
        # predictor.hyperparameter_tuning()

        # 训练模型
        predictor.train_model(use_best_params=False)

        # 评估模型
        evaluation_results = predictor.evaluate_model()

        # 保存预测结果到Excel
        predictor.save_prediction_results("prediction_results.xlsx")

        # 打印特征重要性
        print("\n特征重要性排序:")
        print(evaluation_results['feature_importance'])

        # 示例：对新数据进行预测
        # 假设我们有一组新的材料特性数据，格式与训练数据相同，数据自行填写
        new_material = pd.DataFrame({
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

        predicted_thermal_conductivity = predictor.predict(new_material)
        print(f"\n预测的复合导热系数: {predicted_thermal_conductivity[0]:.4f} W/(m·K)")

        # 保存模型
        predictor.save_model()


if __name__ == "__main__":
    main()
