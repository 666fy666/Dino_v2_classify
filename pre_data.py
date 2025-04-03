import os
import pandas as pd


def count_images_and_export():
    # 配置参数
    root_folder = 'search_data'  # 根目录路径
    excel_name = 'category_statistics.xlsx'  # 输出Excel文件名
    image_exts = {'.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp', '.tiff'}  # 支持的图片格式

    # 存储统计结果
    statistics = []

    # 遍历根目录下的所有子文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # 跳过非目录文件
        if not os.path.isdir(folder_path):
            continue

        # 解析文件夹名称（格式：前缀-类别代码-类别名称）
        parts = folder_name.split('-')
        if len(parts) < 3:
            print(f"警告：跳过不符合命名规范的文件夹 {folder_name}")
            continue

        # 提取类别信息
        category_code = parts[1]
        category_name = '-'.join(parts[2:])  # 处理名称中包含连字符的情况

        # 统计图片数量
        image_count = 0
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file)
                if ext.lower() in image_exts:
                    image_count += 1

        # 记录统计结果
        statistics.append({
            'category_code': category_code,
            'category_name': category_name,
            'counts': image_count
        })

    # 生成Excel文件
    if statistics:
        df = pd.DataFrame(statistics)
        df.to_excel(excel_name, index=False)
        print(f"成功生成统计文件：{excel_name}")
    else:
        print("未找到有效数据，请检查文件夹结构和命名规范")


if __name__ == "__main__":
    count_images_and_export()