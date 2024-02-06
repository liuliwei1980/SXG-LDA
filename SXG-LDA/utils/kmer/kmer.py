from collections import Counter
import pandas as pd
import itertools


def read_fasta_file(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences


def calculate_kmer_feature_encoding(sequence, k_size):
    n_kmers = len(sequence) - k_size + 1
    kmers = [sequence[i:i + k_size] for i in range(n_kmers)]
    kmer_counts = Counter(kmers)
    total_length = len(sequence)
    # 创建包含所有可能k-mer的列表
    all_kmers = ["".join(kmer) for kmer in itertools.product("ACGT", repeat=k_size)]
    # 检查每个k-mer是否在Counter对象中存在，如果不存在则将频率设置为0
    feature_encoding = {kmer: kmer_counts[kmer] / total_length if kmer in kmer_counts else 0 for kmer in all_kmers}
    return feature_encoding

# 从FASTA文件中读取DNA序列
fasta_file = '../../sequences/lncrna_seq.txt'
sequences = read_fasta_file(fasta_file)

# 创建一个空的数据框
df = pd.DataFrame(columns=['k-mer', 'frequency', 'sequences'])

# 对每个DNA序列计算k-mer特征编码
k_size = 3
for sequence in sequences:
    feature_encoding = calculate_kmer_feature_encoding(sequence, k_size)
    df_sequence = pd.DataFrame(feature_encoding.items(), columns=['k-mer', 'frequency'])
    df_sequence['sequences'] = sequence
    df = df.append(df_sequence)

# 去除重复的特征编码数据，保留每个序列的唯一频率数据
df = df.drop_duplicates(subset=['k-mer', 'frequency', 'sequences'])

# 保存每个表格的 'frequency' 列数据按行到文本文件
output_file = '3-mer.txt'

# 打开文件以写入数据
with open(output_file, 'w') as file:
    # 迭代处理每个序列的数据
    for sequence in sequences:
        # 获取当前序列的数据
        sequence_data = df[df['sequences'] == sequence]['frequency']
        # 将数据转换为字符串，并用制表符分隔
        sequence_data_str = '\t'.join(str(val) for val in sequence_data)
        file.write(sequence_data_str + '\n')  # 写入数据到文件，并添加换行符
