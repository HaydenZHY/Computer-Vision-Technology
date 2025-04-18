import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models  # 【新增】用于加载VGG16
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from sklearn.metrics import (
    jaccard_score,  # 用于计算IoU
    f1_score,       # 用于计算F1
    accuracy_score, # 用于计算Accuracy
    recall_score,   # 用于计算Recall
)
from scipy.spatial.distance import directed_hausdorff  # 用于计算HD
from medpy.metric.binary import dc  # 用于计算Dice系数

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform_img=None, transform_label=None): #【新增】split参数【删除】transform参数【新增】transform_img和transform_label两个参数
        self.root = root
        # self.transform = transform 【删除】
        self.split = split #【新增】
        self.transform_img = transform_img #【新增】
        self.transform_label = transform_label #【新增】
        # 图像与标签所在文件夹
        self.img_dir = os.path.join(root, "JPEGImages")
        self.label_dir = os.path.join(root, "SegmentationClass")
        # 【新增】读取图像ID文件
        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        # if transform is None:
        #     self.transform = transforms.Compose([
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                             std=[0.229, 0.224, 0.225])
        #     ])
        # 默认预处理
        if self.transform_img is None:
            self.transform_img = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        # 标签转换：直接转换为long类型；【根据需要可以加入映射忽略区域】
        if self.transform_label is None:
            self.transform_label = transforms.Compose([
                transforms.Resize((256, 256), interpolation=Image.NEAREST),
                transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).long())
            ])
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # img = Image.open(self.img_dir[idx])
        # label = Image.open(self.label_dir[idx])
        img_id = self.ids[idx]
        # 【修改】构造完整路径，注意文件后缀（此处假设图像为.jpg，标签为.png）
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        label_path = os.path.join(self.label_dir, img_id + ".png")
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)
        img = self.transform_img(img)
        label = self.transform_label(label)
        return img, label

# 请不要使用torchvision的VOCSegmentation，独立实现dataset以及dataloader
def get_dataloader(batch_size=8):
    # 注意这里只有train的dataset，在测试时候请实现test的dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    # 独立实现dataset的构建
    dataset = VOCDataset(root="./data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# 【新增】全局函数，用于将标签图像转换为 tensor
def to_tensor_label(img):
    return torch.from_numpy(np.array(img)).long()

# 【新增】分别构造训练集和验证集的 DataLoader
def get_train_dataloader(batch_size=8):
    transform_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    transform_label = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(to_tensor_label)
    ])
    dataset = VOCDataset(root="./VOCdevkit/VOC2012", split="train",
                         transform_img=transform_img, transform_label=transform_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def get_val_dataloader(batch_size=8):
    transform_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    transform_label = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(to_tensor_label)
    ])
    dataset = VOCDataset(root="./VOCdevkit/VOC2012", split="val",
                         transform_img=transform_img, transform_label=transform_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super(SimpleSegmentationModel, self).__init__()
        self.num_classes = num_classes  # 将num_classes保存为成员变量
        
        # 使用预训练的VGG16作为编码器
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except AttributeError:
            vgg = models.vgg16(pretrained=True)  # 旧版本回退
        
        self.encoder = vgg.features
        
        # 解码器部分
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool5 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # 上采样部分
        self.upsample_2x = nn.ConvTranspose2d(
            num_classes, num_classes, 
            kernel_size=4, stride=2, padding=1,
            bias=False
        )  
        self.upsample_8x = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=16, stride=8, padding=4,
            bias=False
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        # 用双线性插值初始化转置卷积
        nn.init.constant_(self.upsample_2x.weight, 0)
        nn.init.constant_(self.upsample_8x.weight, 0)
        
        # 构造双线性插值核
        def bilinear_kernel(in_channels, out_channels, kernel_size):
            factor = (kernel_size + 1) // 2
            center = kernel_size / 2
            og = np.ogrid[:kernel_size, :kernel_size]
            filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
            for i in range(in_channels):
                weight[i, i] = filt
            return torch.from_numpy(weight)
        
        # 使用self.num_classes访问
        self.upsample_2x.weight.data = bilinear_kernel(self.num_classes, self.num_classes, 4)
        self.upsample_8x.weight.data = bilinear_kernel(self.num_classes, self.num_classes, 16)

    def forward(self, x):
        # ================ 编码器部分 ================
        # 获取三个关键层的特征（VGG16的pool3/pool4/pool5）
        pool3 = self.encoder[:17](x)     # [N, 256, 32, 32] (1/8尺寸)
        pool4 = self.encoder[17:24](pool3)  # [N, 512, 16, 16] (1/16尺寸)
        pool5 = self.encoder[24:](pool4)    # [N, 512, 8, 8] (1/32尺寸)
        
        # ================ 解码器部分 ================
        # 第一级：处理pool5特征
        score_pool5 = self.score_pool5(pool5)  # [N, num_classes, 8, 8]
        upscore2 = self.upsample_2x(score_pool5)  # 2倍上采样→[N, num_classes, 16, 16]
        
        # 第二级：融合pool4特征
        score_pool4 = self.score_pool4(pool4)    # [N, num_classes, 16, 16]
        fuse_pool4 = upscore2 + score_pool4      # 特征相加融合
        
        # 第三级：融合pool3特征
        upscore_pool4 = self.upsample_2x(fuse_pool4)  # 2倍上采样→[N, num_classes, 32, 32]
        score_pool3 = self.score_pool3(pool3)         # [N, num_classes, 32, 32]
        fuse_pool3 = upscore_pool4 + score_pool3      # 最终融合
        
        # 最终8倍上采样
        out = self.upsample_8x(fuse_pool3)  # [N, num_classes, 256, 256]
        return {"out": out}  # 保持与原代码相同的输出格式
        # ================ 标准FCN-8s实现结束 ================

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        # 补全训练代码，需要计算loss
        # for images, masks in dataloader:
        #     pass
        model.train()
        epoch_loss = 0.0  # 【新增】累计loss
        num_batches = 0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            # 【修改】提取字典中的输出
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        # 【新增】输出当前epoch平均loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}")
    return model

def decode_segmap(label_mask, nc=21):
    label_colors = np.array([
        (0, 0, 0),       # 0: 背景
        (128, 0, 0),     # 1
        (0, 128, 0),     # 2
        (128, 128, 0),   # 3
        (0, 0, 128),     # 4
        (128, 0, 128),   # 5
        (0, 128, 128),   # 6
        (128, 128, 128), # 7
        (64, 0, 0),      # 8
        (192, 0, 0),     # 9
        (64, 128, 0),    # 10
        (192, 128, 0),   # 11
        (64, 0, 128),    # 12
        (192, 0, 128),   # 13
        (64, 128, 128),  # 14
        (192, 128, 128), # 15
        (0, 64, 0),      # 16
        (128, 64, 0),    # 17
        (0, 192, 0),     # 18
        (128, 192, 0),   # 19
        (0, 64, 128)     # 20
    ])
    # 定义忽略区域颜色，这里将 ignore_index (如 255) 映射为白色
    ignore_color = (255, 255, 255)
    
    r = np.zeros_like(label_mask).astype(np.uint8)
    g = np.zeros_like(label_mask).astype(np.uint8)
    b = np.zeros_like(label_mask).astype(np.uint8)
    
    # 映射正常类别
    for ll in range(0, nc):
        r[label_mask == ll] = label_colors[ll, 0]
        g[label_mask == ll] = label_colors[ll, 1]
        b[label_mask == ll] = label_colors[ll, 2]
    # 对忽略区域（例如标签值为 255）的部分，设置为白色
    r[label_mask == 255] = ignore_color[0]
    g[label_mask == 255] = ignore_color[1]
    b[label_mask == 255] = ignore_color[2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def visualize_results(model, dataloader):
    model.eval()
    # 这里应该补全可视化代码，并且输出<原图，预测图，真实标签图>
    # with torch.no_grad():
        # pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # 获取一批数据
        images, masks = next(iter(dataloader))
        images = images.to(device)
        outputs = model(images)['out']
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
    
    # 显示最多3个样本
    num_samples = min(3, images.shape[0])
    for i in range(num_samples):
        # 原始图像还原（反归一化）
        img = images[i].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        pred_color = decode_segmap(preds[i])
        mask_color = decode_segmap(masks[i])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("init image")
        axes[0].axis("off")
        
        axes[1].imshow(pred_color)
        axes[1].set_title("predict label")
        axes[1].axis("off")
        
        axes[2].imshow(mask_color)
        axes[2].set_title("true label")
        axes[2].axis("off")
        
        # 在当前目录下保存图片
        save_file = f"result_{i}.png"
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        plt.close(fig)  # 关闭图像，释放内存

def calculate_metrics(pred, target, num_classes=21):
    """
    计算多个分割评估指标 mIoU, Dice, HD, Accuracy, Recall, F1
    """
    metrics = {}
    
    # 将输入转换为numpy数组
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    
    # 计算mIoU
    def mIoU():
        return jaccard_score(target, pred, average='macro', 
                           labels=range(num_classes), 
                           zero_division=0)
    
    # 计算Dice系数
    def dice_score():
        return f1_score(target, pred, average='macro',  # Dice系数等价于F1 score
                       labels=range(num_classes),
                       zero_division=0)
    
    # 计算Hausdorff距离
    def hausdorff_distance():
        scores = []
        # 先将 pred 和 target 按 batch 大小 reshape 成 (N, 256, 256)
        pred_reshaped = pred.reshape(-1, 256, 256)
        target_reshaped = target.reshape(-1, 256, 256)
        for p, t in zip(pred_reshaped, target_reshaped):
            for i in range(num_classes):
                pred_mask = (p == i)
                target_mask = (t == i)
                if not np.any(pred_mask) or not np.any(target_mask):
                    scores.append(0)
                    continue
                pred_points = np.array(np.where(pred_mask)).T
                target_points = np.array(np.where(target_mask)).T
                hd = max(directed_hausdorff(pred_points, target_points)[0],
                        directed_hausdorff(target_points, pred_points)[0])
                scores.append(hd)
        return np.mean(scores)

    # 计算Accuracy
    def accuracy():
        return accuracy_score(target, pred)

    # 计算Recall
    def recall():
        return recall_score(target, pred, average='macro',
                          labels=range(num_classes),
                          zero_division=0)

    # 计算F1 score
    def f1():
        return f1_score(target, pred, average='macro',
                       labels=range(num_classes),
                       zero_division=0)
    
    # 计算所有指标
    metrics['mIoU'] = mIoU()
    metrics['Dice'] = dice_score()
    metrics['HD'] = hausdorff_distance()
    metrics['Accuracy'] = accuracy()
    metrics['Recall'] = recall()
    metrics['F1'] = f1()
    
    return metrics

def test_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 初始化所有指标的累加器
    total_metrics = {
        'mIoU': 0,
        'Dice': 0,
        'HD': 0,
        'Accuracy': 0,
        'Recall': 0,
        'F1': 0
    }
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)['out']  # 【修改】提取"out"
            preds = outputs.argmax(dim=1)
            
            batch_metrics = calculate_metrics(preds, masks)

            for metric_name in total_metrics.keys():
                total_metrics[metric_name] += batch_metrics[metric_name]
            
            num_batches += 1
    
    # 计算平均值
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    # 打印所有指标
    print("\n评估指标:")
    print(f"mIoU: {avg_metrics['mIoU']:.4f}")
    print(f"Dice: {avg_metrics['Dice']:.4f}")
    print(f"Hausdorff Distance: {avg_metrics['HD']:.4f}")
    print(f"Accuracy: {avg_metrics['Accuracy']:.4f}")
    print(f"Recall: {avg_metrics['Recall']:.4f}")
    print(f"F1 Score: {avg_metrics['F1']:.4f}")
    
    return avg_metrics

if __name__ == "__main__":
    # dataloader = get_dataloader()
    # 【新增】构造训练集和验证集的dataloader（分别用于训练与测试）
    train_loader = get_train_dataloader(batch_size=8)
    val_loader = get_val_dataloader(batch_size=8)
    model = SimpleSegmentationModel(num_classes=21)
    # 对下面进行调整，不一定需要adam，并分析不同lr对结果的影响
    lr = 3e-4
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # model = train_model(model, dataloader) 
    model = train_model(model, train_loader, criterion, optimizer, num_epochs=200, lr=lr)
    visualize_results(model, val_loader)
    avg_metrics = test_model(model, val_loader)
    