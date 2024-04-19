import os,sys,requests,glob

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

url = 'https://hf-mirror.com/datasets/wzy816/scifi/resolve/main/data.zip'
save_path = 'data/data.zip'

# 判断目录是否存在
if not os.path.exists('data'):
    os.makedirs('data')

download_file(url, save_path)

# 解压缩文件
import zipfile
with zipfile.ZipFile(save_path, 'r') as zip_ref:
    zip_ref.extractall('data')

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