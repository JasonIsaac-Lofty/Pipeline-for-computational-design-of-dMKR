# 多变量回归分析代码评估与改进建议

## 代码整体评价

您的代码思路清晰，流程合理，主要包含以下步骤：
1. 基于AIC确定的最佳阶数进行多项式特征工程
2. 使用ElasticNet进行特征选择
3. 构建OLS模型并进行诊断
4. 交叉验证评估和可视化

## 现有代码的优点

1. **特征工程有理论基础**：基于AIC选择多项式阶数是合理的方法
2. **特征选择策略合适**：ElasticNet能有效处理多重共线性和特征选择
3. **模型诊断全面**：包含VIF计算、残差分析和交叉验证
4. **可视化完整**：提供了残差分布和预测vs真实值的图表

## 主要改进建议

### 1. 数据质量检查和预处理

**问题**：缺少对原始数据的质量检查
**建议**：
```python
# 添加数据质量检查
def data_quality_check(df, features, target):
    print("=== 数据质量检查 ===")
    print(f"数据形状: {df.shape}")
    print(f"缺失值:\n{df[features + [target]].isnull().sum()}")
    
    # 检查异常值
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[target] < Q1 - 1.5*IQR) | (df[target] > Q3 + 1.5*IQR)]
    print(f"潜在异常值数量: {len(outliers)}")
    
    # 基本统计信息
    print(f"\n目标变量 {target} 统计信息:")
    print(df[target].describe())
    
    return outliers
```

### 2. 更严格的多重共线性处理

**问题**：多项式特征容易产生严重多重共线性
**建议**：
```python
def handle_multicollinearity(X_features, threshold=10):
    """迭代移除高VIF特征"""
    features_to_keep = X_features.columns.tolist()
    
    while True:
        X_subset = X_features[features_to_keep]
        vif_df = calculate_vif(X_subset)
        max_vif = vif_df['VIF'].max()
        
        if max_vif <= threshold:
            break
            
        # 移除VIF最高的特征
        worst_feature = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
        features_to_keep.remove(worst_feature)
        print(f"移除高VIF特征: {worst_feature} (VIF={max_vif:.2f})")
    
    return features_to_keep
```

### 3. 更全面的模型评估指标

**问题**：只使用R²作为评估指标不够全面
**建议**：
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def comprehensive_evaluation(y_true, y_pred):
    """计算多种评估指标"""
    metrics = {
        'R²': 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics
```

### 4. 模型稳定性分析

**问题**：缺少对模型稳定性的评估
**建议**：
```python
def bootstrap_stability(X, y, n_bootstrap=1000):
    """Bootstrap分析模型稳定性"""
    coef_samples = []
    
    for i in range(n_bootstrap):
        # Bootstrap采样
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X.iloc[idx], y[idx]
        
        # 拟合模型
        model_boot = OLS(y_boot, X_boot).fit()
        coef_samples.append(model_boot.params)
    
    coef_df = pd.DataFrame(coef_samples)
    
    # 计算系数的置信区间
    confidence_intervals = {}
    for col in coef_df.columns:
        ci_lower = np.percentile(coef_df[col], 2.5)
        ci_upper = np.percentile(coef_df[col], 97.5)
        confidence_intervals[col] = (ci_lower, ci_upper)
    
    return coef_df, confidence_intervals
```

### 5. 特征重要性分析

**问题**：缺少特征重要性的定量分析
**建议**：
```python
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
```

### 6. 改进的交叉验证策略

**问题**：简单的5折交叉验证可能不够稳健
**建议**：
```python
def robust_cross_validation(X, y, cv_folds=10, n_repeats=5):
    """重复交叉验证获得更稳健的结果"""
    from sklearn.model_selection import RepeatedKFold
    
    rkf = RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats, random_state=42)
    cv_scores = []
    
    for train_idx, test_idx in rkf.split(X):
        model_cv = OLS(y[train_idx], X.iloc[train_idx]).fit()
        y_pred_cv = model_cv.predict(X.iloc[test_idx])
        
        metrics = comprehensive_evaluation(y[test_idx], y_pred_cv)
        cv_scores.append(metrics)
    
    # 计算各指标的均值和标准差
    cv_results = {}
    for metric in cv_scores[0].keys():
        values = [score[metric] for score in cv_scores]
        cv_results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values)
        }
    
    return cv_results
```

### 7. 残差分析改进

**问题**：残差分析不够深入
**建议**：
```python
def advanced_residual_analysis(y_true, y_pred, X_features, save_prefix):
    """更全面的残差分析"""
    residuals = y_true - y_pred
    standardized_residuals = residuals / np.std(residuals)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 残差直方图
    sns.histplot(residuals, kde=True, ax=axes[0,0])
    axes[0,0].set_title("Residuals Distribution")
    
    # 2. QQ图检验正态性
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title("Q-Q Plot")
    
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
    axes[1,0].set_title("Standardized Residuals")
    
    # 5. 预测vs真实值
    axes[1,1].scatter(y_true, y_pred, alpha=0.6)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1,1].set_xlabel("True Values")
    axes[1,1].set_ylabel("Predicted Values")
    axes[1,1].set_title("Predicted vs True")
    
    # 6. 残差vs特征（选择最重要的几个特征）
    # 这里可以选择系数最大的特征进行分析
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_advanced_diagnostics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 进行正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"Shapiro-Wilk正态性检验: 统计量={shapiro_stat:.4f}, p值={shapiro_p:.4f}")
    
    return {
        'shapiro_statistic': shapiro_stat,
        'shapiro_pvalue': shapiro_p,
        'residuals_mean': np.mean(residuals),
        'residuals_std': np.std(residuals)
    }
```

## 其他建议

### 1. 模型选择优化
考虑使用其他回归方法进行对比：
- Ridge回归
- Random Forest
- XGBoost
- 支持向量回归(SVR)

### 2. 特征工程改进
- 考虑特征之间的交互项
- 使用主成分分析(PCA)降维
- 尝试非线性变换（如对数变换）

### 3. 实验设计改进
- 使用留一法交叉验证(LOOCV)
- 添加外部验证集
- 考虑时间序列分割（如果数据有时间依赖性）

### 4. 结果解释性
- 添加SHAP值分析
- 提供预测置信区间
- 分析模型在不同数据子集上的表现

## 总结

您的代码已经具备了良好的基础架构，主要改进方向是：
1. 加强数据质量控制
2. 更严格的多重共线性处理
3. 更全面的模型评估
4. 增加模型稳定性分析
5. 提高结果的可解释性

这些改进将使您的模型更加稳健可靠，对酶突变体Kd的预测更加准确。