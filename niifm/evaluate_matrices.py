import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def calculate_auroc_aupr(y_true, y_pred):
    """
    计算AUROC和AUPR指标
    
    参数:
    y_true: 真实标签数组, 0表示无边, 1表示有边
    y_pred: 预测概率数组, 表示节点对之间存在边的概率
    
    返回:
    auroc: AUROC值
    aupr: AUPR值
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 获取正样本和负样本的数量
    P = np.sum(y_true == 1)  # 正样本数量
    Q = np.sum(y_true == 0)  # 负样本数量
    L = len(y_true)  # 总样本数量
    
    # 按预测概率从大到小排序的索引
    sorted_indices = np.argsort(y_pred)[::-1]
    
    # 对应排序后的真实标签
    sorted_y_true = y_true[sorted_indices]
    
    # 计算TPR和FPR
    TP = np.cumsum(sorted_y_true)
    FP = np.cumsum(1 - sorted_y_true)
    
    TPR = TP / P if P > 0 else np.zeros_like(TP)
    FPR = FP / Q if Q > 0 else np.zeros_like(FP)
    
    # 计算Precision和Recall
    Precision = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0)
    Recall = TPR  # Recall就是TPR
    
    # 计算AUROC和AUPR
    auroc = auc(FPR, TPR)
    aupr = auc(Recall, Precision)
    
    return auroc, aupr

def evaluate_network_reconstruction(true_adj_matrix, pred_prob_matrix):
    """
    评估网络重构性能
    
    参数:
    true_adj_matrix: 真实邻接矩阵, n x n大小, 其中n是节点数
    pred_prob_matrix: 预测概率矩阵, n x n大小
    
    返回:
    avg_auroc: 所有节点的平均AUROC值
    avg_aupr: 所有节点的平均AUPR值
    """
    n = true_adj_matrix.shape[0]  # 节点数量
    node_aurocs = []
    node_auprs = []
    
    for i in range(n):
        # 不考虑自环
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        y_true = true_adj_matrix[i, mask]
        y_pred = pred_prob_matrix[i, mask]
        
        auroc, aupr = calculate_auroc_aupr(y_true, y_pred)
        node_aurocs.append(auroc)
        node_auprs.append(aupr)
    
    avg_auroc = np.mean(node_aurocs)
    avg_aupr = np.mean(node_auprs)
    
    return avg_auroc, avg_aupr

def plot_roc_pr_curves(y_true, y_pred):
    """
    绘制ROC曲线和PR曲线
    
    参数:
    y_true: 真实标签数组
    y_pred: 预测概率数组
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 获取正样本和负样本的数量
    P = np.sum(y_true == 1)
    Q = np.sum(y_true == 0)
    
    # 按预测概率从大到小排序的索引
    sorted_indices = np.argsort(y_pred)[::-1]
    
    # 对应排序后的真实标签
    sorted_y_true = y_true[sorted_indices]
    
    # 计算TPR和FPR
    TP = np.cumsum(sorted_y_true)
    FP = np.cumsum(1 - sorted_y_true)
    
    TPR = TP / P if P > 0 else np.zeros_like(TP)
    FPR = FP / Q if Q > 0 else np.zeros_like(FP)
    
    # 计算Precision和Recall
    Precision = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0)
    Recall = TPR
    
    # 计算AUROC和AUPR
    auroc = auc(FPR, TPR)
    aupr = auc(Recall, Precision)
    
    # 绘制ROC曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(FPR, TPR, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # 绘制PR曲线
    plt.subplot(1, 2, 2)
    plt.plot(Recall, Precision, label=f'AUPR = {aupr:.4f}')
    random_aupr = P / len(y_true)
    plt.axhline(y=random_aupr, color='k', linestyle='--', label=f'Random: {random_aupr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def load_txt_matrix(file_path):
    """
    从txt文件加载矩阵
    
    参数:
    file_path: txt文件路径
    
    返回:
    matrix: 加载的矩阵作为numpy数组
    """
    try:
        # 尝试直接使用numpy加载
        return np.loadtxt(file_path)
    except Exception as e:
        print(f"使用标准方法加载失败: {e}")
        try:
            # 如果失败，尝试手动解析文件
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 判断文件格式
            if len(lines) > 0:
                # 尝试确定分隔符和矩阵维度
                first_line = lines[0].strip()
                if ',' in first_line:
                    sep = ','
                elif ' ' in first_line.strip():
                    sep = None  # 空白符作为分隔符
                else:
                    sep = '\t'  # 制表符
                
                data = []
                for line in lines:
                    line = line.strip()
                    if line:
                        # 尝试解析每一行
                        try:
                            row = [float(x) for x in line.split(sep) if x.strip()]
                            data.append(row)
                        except ValueError:
                            # 跳过无法解析的行
                            continue
                
                return np.array(data)
            else:
                raise ValueError("文件为空")
        except Exception as e:
            print(f"手动解析也失败: {e}")
            raise

# 主函数
def main():
    # 文件路径
    true_matrix_path = ""
    pred_matrix_path = ""
    
    try:
        # 加载真实矩阵和预测矩阵
        print("正在加载真实矩阵...")
        true_adj = load_txt_matrix(true_matrix_path)
        print(f"真实矩阵加载成功，形状: {true_adj.shape}")
        
        print("正在加载预测矩阵...")
        pred_prob = load_txt_matrix(pred_matrix_path)
        print(f"预测矩阵加载成功，形状: {pred_prob.shape}")
        
        # 检查矩阵尺寸是否匹配
        if true_adj.shape != pred_prob.shape:
            print(f"警告: 矩阵尺寸不匹配! 真实矩阵: {true_adj.shape}, 预测矩阵: {pred_prob.shape}")
            
            # 找出共同的维度
            min_dim = min(true_adj.shape[0], pred_prob.shape[0])
            print(f"使用两个矩阵的前 {min_dim}x{min_dim} 部分进行评估")
            
            true_adj = true_adj[:min_dim, :min_dim]
            pred_prob = pred_prob[:min_dim, :min_dim]
        
        # 评估网络重构性能
        print("正在计算指标...")
        avg_auroc, avg_aupr = evaluate_network_reconstruction(true_adj, pred_prob)
        print(f"每个节点平均 AUROC: {avg_auroc:.4f}")
        print(f"每个节点平均 AUPR: {avg_aupr:.4f}")
        
        # 计算整体性能
        # 不考虑自环和重复边
        n = true_adj.shape[0]
        mask = ~np.eye(n, dtype=bool)
        upper_tri = np.triu_indices(n, 1)
        
        y_true_flat = true_adj[upper_tri]
        y_pred_flat = pred_prob[upper_tri]
        
        print("\n整体评估:")
        overall_auroc, overall_aupr = calculate_auroc_aupr(y_true_flat, y_pred_flat)
        print(f"整体 AUROC: {overall_auroc:.4f}")
        print(f"整体 AUPR: {overall_aupr:.4f}")
        
        # 绘制ROC和PR曲线
        plot_roc_pr_curves(y_true_flat, y_pred_flat)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
