# -*- coding:utf-8 -*-
# 定义要读取的txt文件列表
txt_files = ['1-mer.txt', '2-mer.txt', '3-mer.txt']

# 定义新的txt文件名和路径
output_file = 'k-mer.txt'

# 打开新的txt文件，准备写入数据
with open(output_file, 'w') as output:
    # 获取文件行数最少的txt文件的行数
    min_lines = min([sum(1 for _ in open(file_name, 'r')) for file_name in txt_files])

    # 遍历每行数据
    for line_number in range(min_lines):
        # 拼接每个文件对应行的数据，并使用制表符连接
        line = '\t'.join([open(file_name, 'r').readlines()[line_number].strip() for file_name in txt_files])

        # 将拼接的行写入新的txt文件
        output.write(line + '\n')

# 输出完成的提示
print('拼接完成，结果保存在' + output_file)