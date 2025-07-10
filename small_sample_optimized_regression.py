import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, RidgeCV, LassoCV
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from statsmodels.api import OLS, add_constant
import warnings
warnings.filterwarnings('ignore')

# 参数区
CSV = '/data2/liyunhao/VTP_3/mutation/filtered_scores_with_experiment.csv'
TARGET = 'KD'
SAVE_DIR = '/data2/liyunhao/VTP_3/mutation/small_sample_optimized'
os.makedirs(SAVE_DIR, exist_ok=True)

def data_quality_check(df, features, target):
    """数据质量检查 - 小样本版本"""
    print("=== 小样本数据质量检查 ===")
    print(f"数据形状: {df.shape}")
    print(f"样本/特征比: {df.shape[0]/len(features):.2f}")
    
    if df.shape[0]/len(features) < 5:
        print("⚠️  警告：样本数/特征数比例过低，建议进一步减少特征数量")
    
    print(f"缺失值:\n{df[features + [target]].isnull().sum()}")
    
    # 检查异常值 (小样本下更保守)
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[target] < Q1 - 1.5*IQR) | (df[target] > Q3 + 1.5*IQR)]
    print(f"潜在异常值数量: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    # 特征与目标的相关性
    print(f"\n特征与目标变量的相关性 (|r| > 0.3视为强相关):")
    correlations = df[features].corrwith(df[target]).sort_values(key=abs, ascending=False)
    for feat, corr in correlations.items():
        significance = "***" if abs(corr) > 0.5 else "**" if abs(corr) > 0.3 else "*" if abs(corr) > 0.1 else ""
        print(f"{feat}: {corr:.3f} {significance}")
    
    return outliers, correlations

def conservative_feature_engineering(df, correlations, max_features=3):
    """保守的特征工程 - 只保留最相关的特征"""
    print(f"\n=== 保守特征工程 (最多{max_features}个特征) ===")
    
    # 选择相关性最高的特征
    top_features = correlations.abs().head(max_features).index.tolist()
    print(f"选择的核心特征: {top_features}")
    
    # 只进行简单的变换
    X_features = []
    feature_names = []
    
    for feat in top_features:
        # 原始特征 (中心化)
        centered_feat = df[feat] - df[feat].mean()
        X_features.append(centered_feat.values.reshape(-1, 1))
        feature_names.append(f"{feat}_centered")
        
        # 对于最相关的特征，考虑二次项
        if abs(correlations[feat]) > 0.5:
            squared_feat = centered_feat ** 2
            X_features.append(squared_feat.values.reshape(-1, 1))
            feature_names.append(f"{feat}_centered^2")
    
    X_combined = np.hstack(X_features)
    X_df = pd.DataFrame(X_combined, columns=feature_names)
    
    print(f"总特征数: {X_df.shape[1]}")
    return X_df

def smart_feature_selection(X, y, max_final_features=2):
    """智能特征选择 - 适合小样本"""
    print(f"\n=== 智能特征选择 (目标: {max_final_features}个特征) ===")
    
    # 第一步：基于F统计量的单变量选择
    k_best = min(6, X.shape[1])  # 先选择较多特征
    selector = SelectKBest(f_regression, k=k_best)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    print(f"F-test选择的特征: {list(selected_features)}")
    
    # 第二步：使用Ridge进行递归特征消除
    estimator = RidgeCV(alphas=np.logspace(-3, 2, 20), cv=LeaveOneOut())
    rfe = RFE(estimator, n_features_to_select=max_final_features, step=1)
    rfe.fit(X[selected_features], y)
    
    final_features = selected_features[rfe.support_]
    print(f"最终选择的特征: {list(final_features)}")
    
    return final_features

def small_sample_models(X, y):
    """适合小样本的模型集合"""
    models = {}
    loo_scores = {}
    
    print("\n=== 小样本专用模型比较 ===")
    
    # 1. 贝叶斯岭回归 (自动处理不确定性)
    bayesian_model = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True
    )
    bayesian_scores = cross_val_score(bayesian_model, X, y, cv=LeaveOneOut(), scoring='neg_mean_squared_error')
    models['Bayesian_Ridge'] = bayesian_model.fit(X, y)
    loo_scores['Bayesian_Ridge'] = np.sqrt(-bayesian_scores.mean())
    
    # 2. 交叉验证岭回归
    ridge_model = RidgeCV(alphas=np.logspace(-3, 2, 20), cv=LeaveOneOut())
    ridge_scores = cross_val_score(ridge_model, X, y, cv=LeaveOneOut(), scoring='neg_mean_squared_error')
    models['Ridge_CV'] = ridge_model.fit(X, y)
    loo_scores['Ridge_CV'] = np.sqrt(-ridge_scores.mean())
    
    # 3. 交叉验证Lasso回归
    lasso_model = LassoCV(alphas=np.logspace(-3, 1, 20), cv=LeaveOneOut(), max_iter=2000)
    lasso_scores = cross_val_score(lasso_model, X, y, cv=LeaveOneOut(), scoring='neg_mean_squared_error')
    models['Lasso_CV'] = lasso_model.fit(X, y)
    loo_scores['Lasso_CV'] = np.sqrt(-lasso_scores.mean())
    
    # 打印LOO-CV结果
    print("留一法交叉验证 RMSE:")
    for model_name, rmse in loo_scores.items():
        print(f"{model_name}: {rmse:.3f}")
    
    # 选择最佳模型
    best_model_name = min(loo_scores, key=loo_scores.get)
    best_model = models[best_model_name]
    
    print(f"\n最佳模型: {best_model_name} (RMSE: {loo_scores[best_model_name]:.3f})")
    
    return models, loo_scores, best_model, best_model_name

def prediction_intervals(model, X, y, confidence_level=0.95):
    """计算预测区间"""
    print(f"\n=== 预测区间计算 (置信度: {confidence_level*100}%) ===")
    
    if hasattr(model, 'predict') and hasattr(model, 'alpha_'):
        # 对于BayesianRidge，可以获得预测标准差
        if type(model).__name__ == 'BayesianRidge':
            y_pred, y_std = model.predict(X, return_std=True)
            
            # 使用t分布计算区间
            t_val = stats.t.ppf((1 + confidence_level) / 2, len(y) - X.shape[1] - 1)
            lower = y_pred - t_val * y_std
            upper = y_pred + t_val * y_std
            
            print(f"平均预测标准差: {y_std.mean():.3f}")
            print(f"平均预测区间宽度: {(upper - lower).mean():.3f}")
            
            return y_pred, lower, upper, y_std
    
    # 对于其他模型，使用残差估计
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    
    # 假设同方差，计算预测区间
    t_val = stats.t.ppf((1 + confidence_level) / 2, len(y) - X.shape[1] - 1)
    pred_std = np.sqrt(mse * (1 + 1/len(y)))
    
    lower = y_pred - t_val * pred_std
    upper = y_pred + t_val * pred_std
    
    print(f"残差标准误差: {np.sqrt(mse):.3f}")
    print(f"预测区间宽度: {2 * t_val * pred_std:.3f}")
    
    return y_pred, lower, upper, np.full_like(y_pred, pred_std)

def comprehensive_analysis(X, y, model, model_name, feature_names):
    """综合分析"""
    print(f"\n=== 综合分析 ({model_name}) ===")
    
    # 基本性能指标
    y_pred = model.predict(X)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    print(f"模型性能:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # 特征重要性 (如果可获取)
    if hasattr(model, 'coef_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        print(f"\n特征重要性:")
        print(importance)
        importance.to_csv(os.path.join(SAVE_DIR, f'{model_name}_feature_importance.csv'), index=False)
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'y_pred': y_pred
    }

def create_visualization(y_true, y_pred, lower, upper, model_name, save_dir):
    """创建可视化图表"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 预测 vs 实际值 (带预测区间)
    axes[0].scatter(y_true, y_pred, alpha=0.7, s=50)
    axes[0].errorbar(y_true, y_pred, yerr=[y_pred-lower, upper-y_pred], 
                     fmt='none', alpha=0.3, capsize=3)
    
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0].set_xlabel('实际值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title(f'{model_name}: 预测 vs 实际 (带预测区间)')
    
    # 2. 残差图
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.7, s=50)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1].set_xlabel('预测值')
    axes[1].set_ylabel('残差')
    axes[1].set_title('残差分布')
    
    # 3. 残差QQ图
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('残差正态性检验')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_comprehensive_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== 小样本优化回归分析 ===\n")
    
    # 1. 读取数据
    df = pd.read_csv(CSV)
    original_features = ['total_score', 'contact_molecular_surface', 'ddg_norepack',
                        'dsasa', 'hole', 'holes_around_lig', 'packing', 'shape_complementary']
    
    # 2. 数据质量检查
    outliers, correlations = data_quality_check(df, original_features, TARGET)
    
    # 3. 保守特征工程
    X_engineered = conservative_feature_engineering(df, correlations, max_features=3)
    y = df[TARGET].values
    
    # 4. 智能特征选择
    final_features = smart_feature_selection(X_engineered, y, max_final_features=2)
    X_final = X_engineered[final_features]
    
    print(f"\n最终建模数据形状: {X_final.shape}")
    print(f"样本/特征比: {X_final.shape[0]/X_final.shape[1]:.2f}")
    
    # 5. 模型比较与选择
    models, loo_scores, best_model, best_model_name = small_sample_models(X_final, y)
    
    # 6. 预测区间计算
    y_pred, lower, upper, pred_std = prediction_intervals(best_model, X_final, y)
    
    # 7. 综合分析
    results = comprehensive_analysis(X_final, y, best_model, best_model_name, final_features)
    
    # 8. 保存结果
    print(f"\n=== 保存结果到 {SAVE_DIR} ===")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'sample_id': range(len(y)),
        'true_KD': y,
        'predicted_KD': y_pred,
        'prediction_lower': lower,
        'prediction_upper': upper,
        'prediction_std': pred_std,
        'residual': y - y_pred
    })
    results_df.to_csv(os.path.join(SAVE_DIR, 'prediction_results.csv'), index=False)
    
    # 保存模型性能
    performance_df = pd.DataFrame({
        'model': list(loo_scores.keys()),
        'loo_cv_rmse': list(loo_scores.values())
    })
    performance_df.to_csv(os.path.join(SAVE_DIR, 'model_performance.csv'), index=False)
    
    # 保存最终特征
    feature_df = pd.DataFrame({
        'selected_features': final_features,
        'correlation_with_target': [correlations.get(feat.split('_')[0], 0) for feat in final_features]
    })
    feature_df.to_csv(os.path.join(SAVE_DIR, 'final_features.csv'), index=False)
    
    # 9. 创建可视化
    create_visualization(y, y_pred, lower, upper, best_model_name, SAVE_DIR)
    
    # 10. 输出总结
    print(f"\n=== 分析总结 ===")
    print(f"最佳模型: {best_model_name}")
    print(f"留一法交叉验证RMSE: {loo_scores[best_model_name]:.3f}")
    print(f"训练集R²: {results['R2']:.4f}")
    print(f"平均预测区间宽度: {np.mean(upper - lower):.3f}")
    print(f"预测区间覆盖率: {np.mean((y >= lower) & (y <= upper))*100:.1f}%")
    
    # 检查预测可靠性
    coverage = np.mean((y >= lower) & (y <= upper))
    if coverage > 0.8:
        print("✅ 预测区间覆盖率良好，模型预测相对可靠")
    else:
        print("⚠️  预测区间覆盖率偏低，预测不确定性较大")
    
    print(f"\n建议:")
    if X_final.shape[0]/X_final.shape[1] < 10:
        print("- 样本量仍然较少，建议谨慎解释结果")
        print("- 重点关注预测区间而非点预测")
        print("- 如可能，建议收集更多实验数据")
    
    print("- 可以使用此模型进行初步筛选，但需实验验证")

if __name__ == '__main__':
    main()