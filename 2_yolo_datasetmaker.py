import os
import random
import shutil
#使得导出的label和图片是对应的
# 获取note_labels文件夹中的所有txt文件名
# label_files = [f for f in os.listdir('note_labels') if f.endswith('.txt')]
#
# # 遍历所有txt文件
# for label_file in label_files:
#     # 去掉后缀，得到图片文件名
#     image_file = label_file[:-4] + '.jpg'
#
#     # 检查dataset_0文件夹中是否存在同名的jpg文件
#     if os.path.exists(os.path.join('dataset_0', image_file)):
#         # 如果存在，将图片复制到note_images文件夹中
#         shutil.copy(os.path.join('dataset_0', image_file), os.path.join('note_images', image_file))

#对数据集进行切分
def copy_files(src_dir, dst_dir, filenames, extension):
    os.makedirs(dst_dir, exist_ok=True)
    missing_files = 0
    for filename in filenames:
        src_path = os.path.join(src_dir, filename + extension)
        dst_path = os.path.join(dst_dir, filename + extension)

        # Check if the file exists before copying
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: File not found for {filename}")
            missing_files += 1

    return missing_files


def split_and_copy_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    # 获取所有图像文件的文件名（不包括文件扩展名）
    image_filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]

    # 随机打乱文件名列表
    random.shuffle(image_filenames)

    # 计算训练集、验证集和测试集的数量
    total_count = len(image_filenames)
    train_count = int(total_count * train_ratio)
    valid_count = int(total_count * valid_ratio)
    test_count = total_count - train_count - valid_count

    # 定义输出文件夹路径
    train_image_dir = os.path.join(output_dir, 'train', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')
    valid_image_dir = os.path.join(output_dir, 'valid', 'images')
    valid_label_dir = os.path.join(output_dir, 'valid', 'labels')
    test_image_dir = os.path.join(output_dir, 'test', 'images')
    test_label_dir = os.path.join(output_dir, 'test', 'labels')

    # 复制图像和标签文件到对应的文件夹
    train_missing_files = copy_files(image_dir, train_image_dir, image_filenames[:train_count], '.jpg')
    train_missing_files += copy_files(label_dir, train_label_dir, image_filenames[:train_count], '.txt')

    valid_missing_files = copy_files(image_dir, valid_image_dir, image_filenames[train_count:train_count + valid_count],
                                     '.jpg')
    valid_missing_files += copy_files(label_dir, valid_label_dir,
                                      image_filenames[train_count:train_count + valid_count], '.txt')

    test_missing_files = copy_files(image_dir, test_image_dir, image_filenames[train_count + valid_count:], '.jpg')
    test_missing_files += copy_files(label_dir, test_label_dir, image_filenames[train_count + valid_count:], '.txt')

    # Print the count of each dataset
    print(f"Train dataset count: {train_count}, Missing files: {train_missing_files}")
    print(f"Validation dataset count: {valid_count}, Missing files: {valid_missing_files}")
    print(f"Test dataset count: {test_count}, Missing files: {test_missing_files}")


# 使用例子
image_dir = 'note_images'
label_dir = 'note_labels'
output_dir = './note_dataset'

split_and_copy_dataset(image_dir, label_dir, output_dir)
