### 训练小说大模型

#### 下载训练集
```
curl -Ok https://hf-mirror.com/datasets/wzy816/scifi/resolve/main/data.zip
mkdir data
mv data.zip data/
cd data/
unzip data.zip
```

#### 合并训练集
```
# 合并小说文件到一个txt文件中
def find_txt_files(directory):
    return glob.glob(os.path.join(directory, '**',"*.txt"), recursive=True)

def merge_txt_files(file_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read() + '\n')
 
# 定义               
directory = 'data'
output_file = 'data/scifi.txt'

# 查找txt文件
txt_files = find_txt_files(directory)

# 合并txt文件
merge_txt_files(txt_files, output_file)
```

#### 执行训练
```
python train.py
```