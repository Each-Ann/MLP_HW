import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# 类别与数字标签一一对应
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = 10

def load_eurosat(root_dir):
    X, y = [], []
    print("正在加载数据集...")
    for cls in CLASS_NAMES:
        cls_path = os.path.join(root_dir, cls)
        label = CLASS_TO_IDX[cls]
        for img_name in os.listdir(cls_path):
            if img_name.endswith(".jpg"):
                img = Image.open(os.path.join(cls_path, img_name)).convert("RGB")
                img = np.array(img.resize((64,64))) / 255.0
                X.append(img)
                y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"加载完成，总样本数：{len(X)}")
    X = X.reshape(len(X), -1)
    return X, y

# 划分数据集
def get_all_data():
    # 这里路径直接填文件夹名，已经自动匹配
    X, y = load_eurosat("EuroSAT_RGB")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=42, stratify=y_trainval
    )
    print(f"训练集:{X_train.shape}, 验证集:{X_val.shape}, 测试集:{X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    get_all_data()
