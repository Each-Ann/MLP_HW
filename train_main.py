import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from data_loader import get_all_data, CLASS_NAMES
from mlp_model import ThreeLayerMLP
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 【断点续训核心】设置 =====================
SAVE_PATH = "best_model.pkl"
CONTINUE_TRAIN = False
# ===============================================================

X_train, y_train, X_val, y_val, X_test, y_test = get_all_data()

# ===================== 自动加载模型 =====================
if CONTINUE_TRAIN and os.path.exists(SAVE_PATH):
    try:
        with open(SAVE_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ 找到已训练模型，继续训练！")
    except:
        print("⚠️  模型文件损坏，重新开始训练！")
        model = ThreeLayerMLP(h1=256, h2=128, lr=0.001, weight_decay=1e-6)
else:
    print("🆕 未找到模型，从头开始训练！")

    model = ThreeLayerMLP(h1=256, h2=128, lr=0.001, weight_decay=1e-6)

epochs = 200
batch_size = 64
# ====================================================

train_loss_list = []
val_acc_list = []
n_train = X_train.shape[0]

print("\n===== 开始训练（支持断点续训） =====")
for epoch in range(epochs):
    model.update_lr_cosine(epoch, epochs)
    shuffle_idx = np.random.permutation(n_train)
    X_shuffle = X_train[shuffle_idx]
    y_shuffle = y_train[shuffle_idx]
    total_loss = 0.0

    for i in range(0, n_train, batch_size):
        X_batch = X_shuffle[i:i+batch_size]
        y_batch = y_shuffle[i:i+batch_size]
        model.forward(X_batch)
        loss = model.cross_entropy_loss(y_batch)
        model.backward(y_batch)
        total_loss += loss

    avg_loss = total_loss / (n_train // batch_size)
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_loss_list.append(avg_loss)
    val_acc_list.append(val_acc)

    # 每 10 轮保存一次
    if (epoch + 1) % 10 == 0:
        with open(SAVE_PATH, "wb") as f:
            pickle.dump(model, f)
    # ====================================================

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | 训练损失: {avg_loss:.4f} | 验证准确率: {val_acc:.4f} | 学习率: {model.lr:.6f}")

# ===================== 最终测试 =====================
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\n✅ 测试集准确率: {test_acc:.4f}")

# ===================== 画图 =====================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="训练损失", color="red")
plt.title("训练损失曲线")
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_acc_list, label="验证准确率", color="green")
plt.title("验证准确率曲线")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("混淆矩阵")
plt.tight_layout()
plt.show()

# 权重可视化
W1 = model.W1
weight_show = W1[:, :64].T.reshape(-1,32,32,3)  # <-- 适配 32x32
plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    w = (weight_show[i] - weight_show[i].min()) / (weight_show[i].max()-weight_show[i].min() + 1e-8)
    plt.imshow(w)
    plt.axis("off")
plt.suptitle("第一层权重可视化")
plt.show()

# 错例分析
wrong_idx = np.where(y_test != y_test_pred)[0][:8]
X_img = X_test.reshape(-1,32,32,3)  # <-- 适配 32x32
plt.figure(figsize=(14,7))
for i, idx in enumerate(wrong_idx):
    plt.subplot(2,4,i+1)
    plt.imshow(X_img[idx])
    plt.title(f"真:{CLASS_NAMES[y_test[idx]]}\n预:{CLASS_NAMES[y_test_pred[idx]]}")
    plt.axis("off")
plt.suptitle("错例样本分析")
plt.tight_layout()
plt.show()
