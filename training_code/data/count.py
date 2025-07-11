import csv
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

def count_categories(file_path):
    """统计单个文件中各类别的数量"""
    category_counts = defaultdict(int)
    
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            category = row.get('category', '').strip()
            if category:  # 忽略空值
                category_counts[category] += 1
                
    return dict(category_counts)

def compare_category_distribution(file1, file2):
    """比较两个文件的分类分布"""
    # 统计两个文件
    counts1 = count_categories(file1)
    counts2 = count_categories(file2)
    
    # 获取所有出现过的分类（并集）
    all_categories = sorted(set(counts1.keys()).union(set(counts2.keys())))
    
    # 生成对比报告
    comparison = []
    for cat in all_categories:
        comparison.append({
            'category': cat,
            'file1_count': counts1.get(cat, 0),
            'file2_count': counts2.get(cat, 0),
            'difference': counts1.get(cat, 0) - counts2.get(cat, 0)
        })
    
    # 按差异绝对值排序
    comparison.sort(key=lambda x: abs(x['difference']), reverse=True)
    
    return comparison

def save_results(comparison, output_json, output_csv, plot_path=None):
    """保存统计结果"""
    # 保存JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # 保存CSV
    df = pd.DataFrame(comparison)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # 生成可视化图表
    if plot_path:
        plt.figure(figsize=(12, 6))
        df.set_index('category')[['file1_count', 'file2_count']].plot(kind='bar')
        plt.title('Category Distribution Comparison')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

if __name__ == "__main__":
    # 文件配置
    FILE1 = 'D:/VSCDocument/fakeNews/data/chinese_fake_news/train.csv'  # 第一个CSV文件路径
    FILE2 = 'D:/VSCDocument/fakeNews/data/chinese_fake_news/test.csv'  # 第二个CSV文件路径
    OUTPUT_JSON = 'category_comparison.json'  # 输出JSON文件
    OUTPUT_CSV = 'category_comparison.csv'    # 输出CSV文件
    PLOT_PATH = 'category_distribution.png'   # 可视化图表路径
    
    # 执行比较
    print("开始统计分类分布...")
    result = compare_category_distribution(FILE1, FILE2)
    
    # 保存结果
    save_results(result, OUTPUT_JSON, OUTPUT_CSV, PLOT_PATH)
    
    # 打印摘要
    print("\n分类统计对比（差异最大的前5项）：")
    for item in result[:5]:
        print(f"{item['category']:8} | 文件1: {item['file1_count']:4} | 文件2: {item['file2_count']:4} | 差异: {item['difference']:+d}")
    
    print(f"\n完整结果已保存至：")
    print(f"- JSON格式: {OUTPUT_JSON}")
    print(f"- CSV格式: {OUTPUT_CSV}")
    if PLOT_PATH:
        print(f"- 分布图表: {PLOT_PATH}")