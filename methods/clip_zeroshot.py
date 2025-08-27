# /methods/zeroshot_clip.py

import torch
import torch.nn as nn
import clip
from torchvision.transforms.functional import resize

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY

@ADAPTATION_REGISTRY.register()
class ZeroShotCLIP(TTAMethod):
    """
    一个“伪”TTA方法，仅用于执行CLIP的零样本（Zero-shot）评估。
    该方法不进行任何模型参数的更新。
    """
    def __init__(self, cfg, model, num_classes):
        # 注意：这里的 'model' (base_model) 将不会被使用，因为我们只依赖CLIP。
        super().__init__(cfg, model, num_classes)

        # 1. 加载并冻结CLIP模型
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        
        # 提取CLIP特定的图像预处理参数
        self.clip_input_resolution = self.clip_model.visual.input_resolution
        self.clip_normalize = self.clip_preprocess.transforms[-1]

        # 2. 创建并编码文本提示
        # 您可以根据需要从配置文件加载更丰富的模板
        cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        cifar100_classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 
            'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
            'camel', 'can', 'castle', 'caterpillar', 'cattle', 
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 
            'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 
            'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 
            'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 
            'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 
            'plain', 'plate', 'poppy', 'porcupine', 'possum', 
            'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
            'rose', 'sea', 'seal', 'shark', 'shrew', 
            'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
            'tank', 'telephone', 'television', 'tiger', 'tractor', 
            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 
            'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

        prompts = [f"a photo of a {c}" for c in cifar100_classes]
        text_inputs = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            # 如果使用prompt ensembling，需要对特征进行平均
            # if len(self.cfg.CLIP.PROMPT_TEMPLATE) > 1:
            #     self.text_features = self.text_features.view(len(cifar10_classes), len(self.cfg.CLIP.PROMPT_TEMPLATE), -1).mean(dim=1)
            #     self.text_features /= self.text_features.norm(dim=-1, keepdim=True)


    @torch.no_grad()
    def forward(self, x):
        """
        覆盖基类的 forward 方法，执行零样本分类。
        注意：我们不使用 forward_and_adapt，因为没有适应过程。
        """
        imgs_test = x[0] if isinstance(x, list) else x

        # 预处理图像以匹配CLIP的输入要求
        imgs_resized = resize(imgs_test, size=(self.clip_input_resolution, self.clip_input_resolution))
        imgs_normalized = self.clip_normalize(imgs_resized)

        # 提取图像特征
        image_features = self.clip_model.encode_image(imgs_normalized)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # 计算图像与文本的相似度（logits）
        # 使用 100.0 作为CLIP标准的温度系数
        logits = 100.0 * image_features @ self.text_features.T
        return logits

    def forward_and_adapt(self, x):
        # 因为没有适应过程，直接调用forward即可
        return self.forward(x)

    # 以下方法对于零样本评估是多余的，但为了保持框架兼容性而保留
    def collect_params(self):
        return [], [] # 没有可优化的参数

    ### --- 添加以下方法 --- ###
    def reset(self):
        """
        重写 reset 方法。
        零样本模型的状态是固定的、永远不会改变，因此重置操作没有意义。
        我们用一个空方法覆盖它，以防止基类因没有模型状态而报错。
        """
        pass
    
    ### --- 添加以下方法 --- ###
    def copy_model_and_optimizer(self):
        """
        重写基类的方法以解决AttributeError。
        因为零样本评估没有优化器，所以我们什么都不做，直接返回None。
        """
        return None, None

    def configure_model(self):
        # 冻结的CLIP模型已经在__init__中配置好了
        self.model.eval()
        self.model.requires_grad_(False)