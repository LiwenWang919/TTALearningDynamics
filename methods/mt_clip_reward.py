import logging

import os
import tqdm
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from methods.base import TTAMethod
from models.model import split_up_model
from augmentations.transforms_cotta import get_tta_transforms
from datasets.data_loading import get_source_loader
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import SymmetricCrossEntropy
from utils.misc import ema_update_model

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

import clip
from torchvision.transforms.functional import resize, normalize

logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class MTCLIPREWARD(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
        _, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               preprocess=model.model_preprocess,
                                               data_root_dir=cfg.DATA_DIR,
                                               batch_size=batch_size_src,
                                               ckpt_path=cfg.MODEL.CKPT_PATH,
                                               num_samples=cfg.SOURCE.NUM_SAMPLES,
                                               percentage=cfg.SOURCE.PERCENTAGE,
                                               use_clip=False,
                                               workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
        self.src_loader_iter = iter(self.src_loader)
        self.contrast_mode = cfg.CONTRAST.MODE
        self.temperature = cfg.CONTRAST.TEMPERATURE
        self.base_temperature = self.temperature
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM
        self.lambda_ce_src = cfg.RMT.LAMBDA_CE_SRC
        self.lambda_ce_trg = cfg.RMT.LAMBDA_CE_TRG
        self.lambda_cont = cfg.RMT.LAMBDA_CONT
        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM
        # arguments neeeded for warm up
        self.warmup_steps = cfg.RMT.NUM_SAMPLES_WARM_UP // batch_size_src
        self.final_lr = cfg.OPTIM.LR
        arch_name = cfg.MODEL.ARCH
        ckpt_path = cfg.MODEL.CKPT_PATH

        self.iter = 0

        self.tta_transform = get_tta_transforms(self.img_size)

        # setup loss functions
        self.symmetric_cross_entropy = SymmetricCrossEntropy()

        # Setup EMA model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        # split up the model
        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, self.dataset_name)

        # define the prototype paths
        proto_dir_path = os.path.join(cfg.CKPT_DIR, "prototypes")
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

        # get source prototypes
        if os.path.exists(fname):
            logger.info("Loading class-wise source prototypes...")
            self.prototypes_src = torch.load(fname)
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            features_src = torch.tensor([])
            labels_src = torch.tensor([])
            logger.info("Extracting source prototypes...")
            with torch.no_grad():
                for data in tqdm.tqdm(self.src_loader):
                    x, y = data[0], data[1]
                    tmp_features = self.feature_extractor(x.to(self.device))
                    features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[:2]).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src) > 100000:
                        break

            # create class-wise source prototypes
            self.prototypes_src = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                self.prototypes_src = torch.cat([self.prototypes_src, features_src[mask].mean(dim=0, keepdim=True)], dim=0)

            torch.save(self.prototypes_src, fname)

        self.prototypes_src = self.prototypes_src.to(self.device).unsqueeze(1)
        self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).to(self.device).long()

        # setup projector
        if self.dataset_name == "domainnet126":
            # do not use a projector since the network already clusters the features and reduces the dimensions
            self.projector = nn.Identity()
        else:
            num_channels = self.prototypes_src.shape[-1]
            self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim), nn.ReLU(),
                                           nn.Linear(self.projection_dim, self.projection_dim)).to(self.device)
            self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})

        # warm up the mean-teacher framework
        if self.warmup_steps > 0:
            warmup_ckpt_path = os.path.join(cfg.CKPT_DIR, "warmup")
            if self.dataset_name == "domainnet126":
                source_domain = ckpt_path.split(os.sep)[-1].split('_')[1]
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{source_domain}_{arch_name}_bs{self.src_loader.batch_size}.pth"
            else:
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{arch_name}_bs{self.src_loader.batch_size}.pth"
            ckpt_path = os.path.join(warmup_ckpt_path, ckpt_path)

            if os.path.exists(ckpt_path):
                logger.info("Loading warmup checkpoint...")
                checkpoint = torch.load(ckpt_path)
                self.model.load_state_dict(checkpoint["model"])
                self.model_ema.load_state_dict(checkpoint["model_ema"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info(f"Loaded from {ckpt_path}")
            else:
                os.makedirs(warmup_ckpt_path, exist_ok=True)
                self.warmup()
                torch.save({"model": self.model.state_dict(),
                            "model_ema": self.model_ema.state_dict(),
                            "optimizer": self.optimizer.state_dict()
                            }, ckpt_path)

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema, self.projector]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
        
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
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

        # prompts = [f"a photo of a {c}" for c in cifar10_classes]
        prompts = [f"a photo of a {c}" for c in cifar100_classes]
        text_inputs = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        # --- FIX: Extract CLIP's expected input resolution and normalization stats ---
        self.clip_input_resolution = self.clip_model.visual.input_resolution
        self.clip_normalize = self.clip_preprocess.transforms[-1]

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def warmup(self):
        logger.info(f"Starting warm up...")
        for i in range(self.warmup_steps):
            #linearly increase the learning rate
            for par in self.optimizer.param_groups:
                par["lr"] = self.final_lr * (i+1) / self.warmup_steps

            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src = batch[0].to(self.device)

            # forward the test data and optimize the model
            outputs = self.model(imgs_src)
            outputs_ema = self.model_ema(imgs_src)
            loss = self.symmetric_cross_entropy(outputs, outputs_ema).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model_ema = ema_update_model(
                model_to_update=self.model_ema,
                model_to_merge=self.model,
                momentum=self.m_teacher_momentum,
                device=self.device,
                update_all=True
            )

        logger.info(f"Finished warm up...")
        for par in self.optimizer.param_groups:
            par["lr"] = self.final_lr

    # Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def loss_calculation(self, x):
        imgs_test = x[0]

        # forward original test data
        features_test = self.feature_extractor(imgs_test)
        outputs_test = self.classifier(features_test)

        # forward augmented test data
        features_aug_test = self.feature_extractor(self.tta_transform(imgs_test))
        outputs_aug_test = self.classifier(features_aug_test)

        # CLIP reward
        imgs_resized = resize(imgs_test, size=(self.clip_input_resolution, self.clip_input_resolution))
        imgs_normalized = self.clip_normalize(imgs_resized)
        image_test_features_clip = self.clip_model.encode_image(imgs_normalized)
        image_test_features_clip /= image_test_features_clip.norm(dim=-1, keepdim=True)

        probs_test = F.softmax(outputs_test, dim=1)
        similarity_test = (100.0 * image_test_features_clip @ self.text_features.T).softmax(dim=-1)
        expected_reward_test = (probs_test * similarity_test).sum(dim=1)
        loss_rl_test = -expected_reward_test.mean()

        # forward original test data through the ema model
        outputs_ema = self.model_ema(imgs_test)

        with torch.no_grad():
            # dist[:, i] contains the distance from every source sample to one test sample
            dist = F.cosine_similarity(
                x1=self.prototypes_src.repeat(1, features_test.shape[0], 1),
                x2=features_test.view(1, features_test.shape[0], features_test.shape[1]).repeat(self.prototypes_src.shape[0], 1, 1),
                dim=-1)

            # for every test feature, get the nearest source prototype and derive the label
            _, indices = dist.topk(1, largest=True, dim=0)
            indices = indices.squeeze(0)

        features = torch.cat([self.prototypes_src[indices],
                              features_test.view(features_test.shape[0], 1, features_test.shape[1]),
                              features_aug_test.view(features_test.shape[0], 1, features_test.shape[1])], dim=1)
        loss_contrastive = self.contrastive_loss(features=features, labels=None)

        loss_self_training = (0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) + 0.5 * self.symmetric_cross_entropy(outputs_aug_test, outputs_ema)).mean(0)
        loss = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive + loss_rl_test

        if self.lambda_ce_src > 0:
            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            # train on labeled source data
            imgs_src, labels_src = batch[0], batch[1]
            features_src = self.feature_extractor(imgs_src.to(self.device))
            outputs_src = self.classifier(features_src)
            loss_ce_src = F.cross_entropy(outputs_src, labels_src.to(self.device).long())
            loss += self.lambda_ce_src * loss_ce_src

        # create and return the ensemble prediction
        outputs = outputs_test + outputs_ema
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        self.iter += 1
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.m_teacher_momentum,
            device=self.device,
            update_all=True
        )
        
        # Call the visualization method here for the current batch
        # You might want to do this only periodically (e.g., every N steps) 
        # to avoid generating too many plots.
        #
        # if self.iter % 50 == 0:  # Assuming you have a step counter
        #     self.visualize_features(x, file_path="/media/Storage3/wlw/TTA/TCA/classification/tsne/rmt/"+ str(self.iter/50) + '.png')

        return outputs

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        outputs_test = self.model(imgs_test)
        outputs_ema = self.model_ema(imgs_test)
        return outputs_test + outputs_ema

    def configure_model(self):
        """Configure model"""
        # model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)

    @torch.no_grad()
    def visualize_features(self, x, file_path="/media/Storage3/wlw/TTA/TCA/classification/tsne_visualization.png"):
        """
        使用t-SNE可视化测试数据和源原型的特征。
        此版本将图例放置在绘图区域的外部右侧，并保持高对比度、大尺寸的样式。

        Args:
            x (torch.Tensor): 输入的测试数据。
            file_path (str): 保存可视化图像的路径。
        """
        logger.info("正在生成t-SNE可视化图...")

        # --- 提取特征 ---
        imgs_test = x[0].to(self.device)
        features_test = self.feature_extractor(imgs_test).cpu().numpy()
        
        # --- 为测试数据获取伪标签用于着色 ---
        with torch.no_grad():
            dist = F.cosine_similarity(
                x1=self.prototypes_src.repeat(1, features_test.shape[0], 1),
                x2=torch.from_numpy(features_test).to(self.device).view(1, features_test.shape[0], features_test.shape[1]).repeat(self.prototypes_src.shape[0], 1, 1),
                dim=-1
            )
            _, pseudo_labels_test = dist.topk(1, largest=True, dim=0)
            pseudo_labels_test = pseudo_labels_test.squeeze(0).cpu().numpy()

        # --- 准备源原型用于t-SNE ---
        prototypes_src_np = self.prototypes_src.squeeze(1).cpu().numpy()
        labels_src = np.arange(self.num_classes)

        # --- 合并特征和标签 ---
        all_features = np.concatenate([features_test, prototypes_src_np], axis=0)
        all_labels = np.concatenate([pseudo_labels_test, labels_src], axis=0)

        # --- 应用 t-SNE ---
        logger.info("正在运行 t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(all_features) - 1), random_state=42, n_iter=500, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(all_features)

        # --- 绘制结果 ---
        # 1. 获取对figure和axes的引用，这是精确定位图例的关键
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # 2. 将所有绘图命令从 plt. 改为 ax.
        # 测试特征的散点图 (圆点)
        scatter = ax.scatter(
            tsne_results[:len(features_test), 0], 
            tsne_results[:len(features_test), 1], 
            c=all_labels[:len(features_test)], 
            cmap='tab20',
            marker='o', 
            alpha=0.8,
            s=250,
            label='Test Features'
        )
        
        # 源原型的散点图 (星形)
        ax.scatter(
            tsne_results[len(features_test):, 0], 
            tsne_results[len(features_test):, 1], 
            c=all_labels[len(features_test):], 
            cmap='tab20',
            marker='*', 
            s=1000,
            edgecolor='black', 
            linewidth=1.5,
            label='Source Prototypes'
        )
        
        # 3. 创建图例并将其放置在图的外部右侧
        legend = ax.legend(
            *scatter.legend_elements(num=min(20, self.num_classes)),
            title="Classes",
            loc='center left',            # 将图例的“左侧中点”作为锚点
            bbox_to_anchor=(1.02, 0.5),   # 将锚点放置在主图右侧外部
            fontsize=22,
            markerscale=3.0,
            borderaxespad=0.              # 锚点和图之间的间距
        )
        plt.setp(legend.get_title(), fontsize=24)

        # 4. 关闭坐标轴
        ax.axis('off')
        
        # 5. 保存图像，bbox_inches='tight' 会自动调整画布以包含外部的图例
        plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        logger.info(f"t-SNE 可视化图已保存至 {file_path}")
        
        # 6. 关闭指定的figure
        plt.close(fig)