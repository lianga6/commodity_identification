
# 这个是json格式的标记文件转换为yolo格式的标记文件，这个是针对多边形的标记文件，可从这个多边形中提取出矩形的标记文件
import json
import os

def convert_polygon_to_bbox(points, img_width, img_height):
    """
    将多边形的点转换为边界框
    :param points: 多边形的点列表，每个点是一个 [x, y] 列表
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :return: 边界框的坐标 [x_min, y_min, x_max, y_max]
    """
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return [x_min, y_min, x_max, y_max]

def convert_json_to_yolo(json_file_path, output_dir, class_mapping):
    """
    将单个 JSON 文件转换为 YOLO 格式的 TXT 文件
    :param json_file_path: JSON 文件路径
    :param output_dir: 输出目录
    :param class_mapping: 类别名称到类别索引的映射
    """
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    txt_file_name = os.path.splitext(os.path.basename(json_file_path))[0] + '.txt'
    txt_file_path = os.path.join(output_dir, txt_file_name)

    with open(txt_file_path, 'w') as txt_file:
        for shape in data['shapes']:
            class_name = shape['label']
            if class_name not in class_mapping:
                raise ValueError(f"类别 '{class_name}' 未在类别映射中找到。请检查类别映射是否正确。")
            class_id = class_mapping[class_name]

            points = shape['points']
            x_min, y_min, x_max, y_max = convert_polygon_to_bbox(points, img_width, img_height)

            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    input_dir = r"D:\picture\CI3\labels\val_j"
    output_dir = r"D:\picture\CI3\labels\val"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_mapping = {
        "可乐": 0,
        "东鹏特饮": 1,
        "红笔": 2,
        "黑笔": 3,
        "旺旺雪饼": 4,
        # 添加更多类别
    }

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_file_path = os.path.join(input_dir, filename)
            convert_json_to_yolo(json_file_path, output_dir, class_mapping)
            print(f"已转换 {filename} 到 {output_dir}")

if __name__ == "__main__":
    main()