
from datasets import load_dataset
import os
os.environ["DATASETS_CACHE_VERBOSITY"] = "disable_warning"


# 加载 XSum 数据集
dataset = load_dataset('xsum')

# 保存数据集到本地文件
dataset.save_to_disk('/root/autodl-tmp/xsum')

