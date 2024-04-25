#### 克隆项目
```
git clone https://github.com/hiyouga/LLaMA-Factory.git
```
#### 安装项目依赖
```
cd LLaMA-Factory
pip install -r requirements.txt
pip install transformers_stream_generator bitsandbytes tiktoken auto-gptq optimum autoawq
pip install --upgrade tensorflow
pip uninstall flash-attn -y
```
#### 运行
```
CUDA_VISIBLE_DEVICES=0 USE_MODELSCOPE_HUB=1 python src/train_web.py
```