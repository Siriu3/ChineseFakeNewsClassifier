import csv

def get_unique_categories(csv_file, category_column='category'):
    unique_categories = set()
    
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if category_column in row:
                category = row[category_column].strip()
                if category:  # 忽略空值
                    unique_categories.add(category)
    
    return sorted(list(unique_categories))  # 排序后返回列表

# 使用示例
if __name__ == "__main__":
    csv_file_path = 'D:/VSCDocument/fakeNews/data/chinese_fake_news/train.csv'  # 替换为你的CSV文件路径
    unique_categories = get_unique_categories(csv_file_path)
    
    print("Unique categories:")
    print(unique_categories)
    print(f"\nTotal unique categories: {len(unique_categories)}")