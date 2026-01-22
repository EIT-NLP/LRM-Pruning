# Eval
## 环境配置
```bash
cd A_env
conda create -n evalchemy_test python=3.10 -y
conda activate evalchemy_test
pip install --only-binary=:all: pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r evalchemy.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 在 evalchemy 目录下运行
cd ..
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# 上面的命令会更改一些必要包的版本，所以再次运行
cd A_env
pip install -r evalchemy.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

## flash-attn download
# 方法一(很有可能卡住安装失败)
pip insatll ninja
pip install packaging
pip install flash-attn==2.7.2.post1 --no-build-isolation

# 方法二
直接下载：https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl (这是我本地的版本)
pip install　'flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'

pip install ml_collections

# 修改 lm_eval 中的 models
用文件中的 models 对 lm_eval 中的进行覆盖
1. 备份旧的 models
bash
```
# 可以用 `pip show  lm_eval` 获取您当前目录 `lm_eval` 的位置
mv `lm_eval` 的位置/models \
   `lm_eval` 的位置/models.bak
```
2. 覆盖旧的 models
bash
```
# 第一行新的 `models` 位于 `CoT_Baseline/evalchemy/models`
cp -r models \
      `lm_eval` 的位置
```