import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# 参数区
CSV = '/data2/liyunhao/VTP_3/mutation/filtered_scores_with_experiment.csv'
TARGET = 'KD'
SAVE_DIR = '/data2/liyunhao/VTP_3/mutation/final_models_improved'
os.makedirs(SAVE_DIR, exist_ok=True)

# 最佳阶数字典
best_deg = {
    'total_score': 1, 'contact_molecular_surface': 2, 'ddg_norepack': 2,
    'dsasa': 1, 'hole': 1, 'holes_around_lig': 2, 'packing': 2, 'shape_complementary': 1
}

def data_quality_check(df, features, target):
    """数据质量检查"""
    print("=== 数据质量检查 ===")
    print(f"数据形状: {df.shape}")
    print(f"缺失值:\n{df[features + [target]].isnull().sum()}")
    
    # 检查异常值
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[target] < Q1 - 1.5*IQR) | (df[target] > Q3 + 1.5*IQR)]
    print(f"潜在异常值数量: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    # 基本统计信息
    print(f"\n目标变量 {target} 统计信息:")
    print(df[target].describe())
    
    # 特征相关性
    print(f"\n特征与目标变量的相关性:")
    correlations = df[features].corrwith(df[target]).sort_values(key=abs, ascending=False)
    print(correlations)
    
    return outliers

def calculate_vif(X):
    """计算方差膨胀因子"""
    vif_df = pd.DataFrame({
        'feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_df

def handle_multicollinearity(X_features, threshold=10):
    """迭代移除高VIF特征"""
    features_to_keep = X_features.columns.tolist()
    removed_features = []
    
    while True:
        X_subset = X_features[features_to_keep]
        vif_df = calculate_vif(X_subset)
        max_vif = vif_df['VIF'].max()
        
        if max_vif <= threshold:
            break
            
        # 移除VIF最高的特征
        worst_feature = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
        features_to_keep.remove(worst_feature)
        removed_features.append(worst_feature)
        print(f"移除高VIF特征: {worst_feature} (VIF={max_vif:.2f})")
    
    print(f"最终保留特征数: {len(features_to_keep)}")
    return features_to_keep, removed_features

def comprehensive_evaluation(y_true, y_pred):
    """计算多种评估指标"""
    # 避免除零错误
    y_true_nonzero = y_true[y_true != 0]
    y_pred_nonzero = y_pred[y_true != 0]
    
    metrics = {
        'R²': 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100 if len(y_true_nonzero) > 0 else np.nan
    }
    return metrics

def bootstrap_stability(X, y, n_bootstrap=1000):
    """Bootstrap分析模型稳定性"""
    coef_samples = []
    
    print("进行Bootstrap稳定性分析...")
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"完成 {i+1}/{n_bootstrap} 次Bootstrap采样")
            
        # Bootstrap采样
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X.iloc[idx], y[idx]
        
        try:
            # 拟合模型
            model_boot = OLS(y_boot, X_boot).fit()
            coef_samples.append(model_boot.params)
        except:
            continue
    
    coef_df = pd.DataFrame(coef_samples)
    
    # 计算系数的置信区间
    confidence_intervals = {}
    for col in coef_df.columns:
        ci_lower = np.percentile(coef_df[col], 2.5)
        ci_upper = np.percentile(coef_df[col], 97.5)
        confidence_intervals[col] = (ci_lower, ci_upper)
    
    return coef_df, confidence_intervals

def feature_importance_analysis(model, feature_names):
    """分析特征重要性"""
    # 标准化系数分析
    std_coef = np.abs(model.params[1:])  # 排除截距
    importance_df = pd.DataFrame({
        'feature': feature_names[1:],  # 排除const
        'coefficient': model.params[1:],
        'abs_coefficient': std_coef,
        'p_value': model.pvalues[1:],
        'significant': model.pvalues[1:] < 0.05
    })
    
    return importance_df.sort_values('abs_coefficient', ascending=False)

def robust_cross_validation(X, y, cv_folds=10, n_repeats=3):
    """重复交叉验证获得更稳健的结果"""
    rkf = RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats, random_state=42)
    cv_scores = []
    
    print(f"进行 {cv_folds}折 x {n_repeats}次 重复交叉验证...")
    
    for fold, (train_idx, test_idx) in enumerate(rkf.split(X)):
        try:
            model_cv = OLS(y[train_idx], X.iloc[train_idx]).fit()
            y_pred_cv = model_cv.predict(X.iloc[test_idx])
            
            metrics = comprehensive_evaluation(y[test_idx], y_pred_cv)
            cv_scores.append(metrics)
        except:
            continue
    
    # 计算各指标的均值和标准差
    cv_results = {}
    for metric in cv_scores[0].keys():
        values = [score[metric] for score in cv_scores if not np.isnan(score[metric])]
        cv_results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return cv_results

def advanced_residual_analysis(y_true, y_pred, save_prefix):
    """更全面的残差分析"""
    residuals = y_true - y_pred
    standardized_residuals = residuals / np.std(residuals)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 残差直方图
    sns.histplot(residuals, kde=True, ax=axes[0,0])
    axes[0,0].set_title("Residuals Distribution")
    axes[0,0].set_xlabel("Residuals")
    
    # 2. QQ图检验正态性
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title("Q-Q Plot (Normality Test)")
    
    # 3. 残差vs拟合值
    axes[0,2].scatter(y_pred, residuals, alpha=0.6)
    axes[0,2].axhline(y=0, color='red', linestyle='--')
    axes[0,2].set_xlabel("Fitted Values")
    axes[0,2].set_ylabel("Residuals")
    axes[0,2].set_title("Residuals vs Fitted")
    
    # 4. 标准化残差
    axes[1,0].scatter(y_pred, standardized_residuals, alpha=0.6)
    axes[1,0].axhline(y=0, color='red', linestyle='--')
    axes[1,0].axhline(y=2, color='red', linestyle=':', alpha=0.5)
    axes[1,0].axhline(y=-2, color='red', linestyle=':', alpha=0.5)
    axes[1,0].set_xlabel("Fitted Values")
    axes[1,0].set_ylabel("Standardized Residuals")
    axes[1,0].set_title("Standardized Residuals")
    
    # 5. 预测vs真实值
    axes[1,1].scatter(y_true, y_pred, alpha=0.6)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1,1].set_xlabel("True Values")
    axes[1,1].set_ylabel("Predicted Values")
    axes[1,1].set_title("Predicted vs True")
    
    # 6. 残差累积分布
    sorted_residuals = np.sort(np.abs(residuals))
    cumulative_prob = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
    axes[1,2].plot(sorted_residuals, cumulative_prob)
    axes[1,2].set_xlabel("Absolute Residuals")
    axes[1,2].set_ylabel("Cumulative Probability")
    axes[1,2].set_title("Cumulative Distribution of Absolute Residuals")
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_advanced_diagnostics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 进行正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    jarque_bera_stat, jarque_bera_p = stats.jarque_bera(residuals)
    
    print(f"\n=== 残差分析结果 ===")
    print(f"Shapiro-Wilk正态性检验: 统计量={shapiro_stat:.4f}, p值={shapiro_p:.4f}")
    print(f"Jarque-Bera正态性检验: 统计量={jarque_bera_stat:.4f}, p值={jarque_bera_p:.4f}")
    print(f"残差均值: {np.mean(residuals):.6f}")
    print(f"残差标准差: {np.std(residuals):.4f}")
    
    return {
        'shapiro_statistic': shapiro_stat,
        'shapiro_pvalue': shapiro_p,
        'jarque_bera_statistic': jarque_bera_stat,
        'jarque_bera_pvalue': jarque_bera_p,
        'residuals_mean': np.mean(residuals),
        'residuals_std': np.std(residuals)
    }

def compare_models(X, y, X_train_indices=None):
    """比较不同模型的性能"""
    if X_train_indices is not None:
        X_compare = X.iloc[X_train_indices]
        y_compare = y[X_train_indices]
    else:
        X_compare = X
        y_compare = y
    
    models = {}
    results = {}
    
    # OLS
    try:
        models['OLS'] = OLS(y_compare, X_compare).fit()
        y_pred_ols = models['OLS'].predict(X_compare)
        results['OLS'] = comprehensive_evaluation(y_compare, y_pred_ols)
    except:
        results['OLS'] = None
    
    # Ridge回归
    try:
        X_scaled = StandardScaler().fit_transform(X_compare.iloc[:, 1:])  # 排除常数项
        ridge = RidgeCV(cv=5, random_state=42).fit(X_scaled, y_compare)
        y_pred_ridge = ridge.predict(X_scaled)
        models['Ridge'] = ridge
        results['Ridge'] = comprehensive_evaluation(y_compare, y_pred_ridge)
    except:
        results['Ridge'] = None
    
    # Random Forest
    try:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_compare.iloc[:, 1:], y_compare)  # 排除常数项
        y_pred_rf = rf.predict(X_compare.iloc[:, 1:])
        models['RandomForest'] = rf
        results['RandomForest'] = comprehensive_evaluation(y_compare, y_pred_rf)
    except:
        results['RandomForest'] = None
    
    return models, results

def main():
    print("=== 改进的多变量回归分析 ===\n")
    
    # 1. 读取数据
    df = pd.read_csv(CSV)
    feature_names = list(best_deg.keys())
    
    # 2. 数据质量检查
    outliers = data_quality_check(df, feature_names, TARGET)
    
    # 3. 构建多项式特征
    print("\n=== 构建多项式特征 ===")
    poly_blocks = []
    for feat, deg in best_deg.items():
        Xp = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(df[[feat]])
        if deg == 1:
            names = [feat]
        else:
            names = [f'{feat}^{i}' for i in range(1, deg+1)]
        poly_blocks.append(pd.DataFrame(Xp, columns=names))
        print(f"{feat}: 阶数={deg}, 生成特征数={Xp.shape[1]}")
    
    X_full = pd.concat(poly_blocks, axis=1)
    y = df[TARGET].values
    feat_names = X_full.columns
    
    print(f"总特征数: {X_full.shape[1]}")
    
    # 4. ElasticNet特征选择
    print("\n=== ElasticNet特征选择 ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    enet = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, 1], cv=5,
                       max_iter=50000, tol=1e-3, random_state=0).fit(X_scaled, y)
    nonzero = enet.coef_ != 0
    sel_feats = feat_names[nonzero]
    print(f"ElasticNet选择的特征数: {len(sel_feats)}")
    print("选择的特征:", list(sel_feats))
    
    # 保存ElasticNet系数
    enet_coefs = pd.Series(enet.coef_[nonzero], index=sel_feats)
    enet_coefs.to_csv(os.path.join(SAVE_DIR, 'enet_selected_coefs.csv'))
    
    # 5. 多重共线性处理
    print("\n=== 多重共线性处理 ===")
    X_selected = add_constant(X_full[sel_feats])
    kept_features, removed_features = handle_multicollinearity(X_selected, threshold=10)
    X_final = X_selected[kept_features]
    
    # 6. 最终OLS模型
    print("\n=== 最终OLS模型 ===")
    ols_model = OLS(y, X_final).fit()
    print(ols_model.summary())
    
    # 保存模型摘要
    with open(os.path.join(SAVE_DIR, 'final_OLS_summary.txt'), 'w') as f:
        f.write(str(ols_model.summary()))
    
    # 7. VIF计算
    print("\n=== VIF检查 ===")
    vif_df = calculate_vif(X_final)
    print(vif_df)
    vif_df.to_csv(os.path.join(SAVE_DIR, 'final_VIF.csv'), index=False)
    
    # 8. 特征重要性分析
    print("\n=== 特征重要性分析 ===")
    importance_df = feature_importance_analysis(ols_model, X_final.columns)
    print(importance_df)
    importance_df.to_csv(os.path.join(SAVE_DIR, 'feature_importance.csv'), index=False)
    
    # 9. Bootstrap稳定性分析
    print("\n=== Bootstrap稳定性分析 ===")
    coef_bootstrap, confidence_intervals = bootstrap_stability(X_final, y, n_bootstrap=1000)
    
    # 保存Bootstrap结果
    bootstrap_summary = pd.DataFrame({
        'feature': confidence_intervals.keys(),
        'mean_coef': [coef_bootstrap[col].mean() for col in confidence_intervals.keys()],
        'std_coef': [coef_bootstrap[col].std() for col in confidence_intervals.keys()],
        'ci_lower': [ci[0] for ci in confidence_intervals.values()],
        'ci_upper': [ci[1] for ci in confidence_intervals.values()]
    })
    bootstrap_summary.to_csv(os.path.join(SAVE_DIR, 'bootstrap_stability.csv'), index=False)
    print("Bootstrap系数95%置信区间:")
    print(bootstrap_summary)
    
    # 10. 稳健交叉验证
    print("\n=== 稳健交叉验证 ===")
    cv_results = robust_cross_validation(X_final, y, cv_folds=10, n_repeats=3)
    
    print("交叉验证结果:")
    for metric, stats in cv_results.items():
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(范围: {stats['min']:.4f} - {stats['max']:.4f})")
    
    # 保存交叉验证结果
    cv_df = pd.DataFrame(cv_results).T
    cv_df.to_csv(os.path.join(SAVE_DIR, 'cross_validation_results.csv'))
    
    # 11. 模型比较
    print("\n=== 模型比较 ===")
    models, model_results = compare_models(X_final, y)
    
    comparison_df = pd.DataFrame(model_results).T
    print("不同模型的性能比较:")
    print(comparison_df)
    comparison_df.to_csv(os.path.join(SAVE_DIR, 'model_comparison.csv'))
    
    # 12. 残差分析
    print("\n=== 残差分析 ===")
    y_pred_train = ols_model.predict(X_final)
    residual_stats = advanced_residual_analysis(y, y_pred_train, 
                                              os.path.join(SAVE_DIR, 'OLS_final_model'))
    
    # 保存残差分析结果
    pd.Series(residual_stats).to_csv(os.path.join(SAVE_DIR, 'residual_analysis.csv'))
    
    # 13. 保存最终模型预测结果
    results_df = pd.DataFrame({
        'true_values': y,
        'predicted_values': y_pred_train,
        'residuals': y - y_pred_train
    })
    results_df.to_csv(os.path.join(SAVE_DIR, 'final_predictions.csv'), index=False)
    
    print(f"\n=== 分析完成 ===")
    print(f"所有结果已保存到: {SAVE_DIR}")
    print(f"最终模型R²: {ols_model.rsquared:.4f}")
    print(f"调整R²: {ols_model.rsquared_adj:.4f}")
    print(f"最终特征数: {X_final.shape[1]-1}")  # 排除常数项

if __name__ == '__main__':
    main()