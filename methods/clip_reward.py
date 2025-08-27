# methods/clip_reward.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
# Import for resizing
from torchvision.transforms.functional import resize, normalize

@ADAPTATION_REGISTRY.register()
class CLIPRewardTTA(TTAMethod):
    """
    Adapts a model by treating CLIP's similarity score as a reward signal
    in a reinforcement learning-like framework.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        
        self.clip_model, self.clip_preprocess = clip.load(self.cfg.MODEL.CLIP_MODEL, device=self.device)
        
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

        self.softmax_entropy = Entropy()
        self.lambda_entropy = self.cfg.CLIP_REWARD.LAMBDA_ENTROPY

        # --- FIX: Extract CLIP's expected input resolution and normalization stats ---
        self.clip_input_resolution = self.clip_model.visual.input_resolution
        self.clip_normalize = self.clip_preprocess.transforms[-1]
        
    @torch.no_grad()
    def loss_calculation(self, x):
        """
        Calculates the adaptation loss for evaluation (no gradients).
        """
        imgs_test = x[0]
        outputs, _ = self.model(imgs_test)

        # --- FIX: Preprocess images for CLIP before encoding ---
        imgs_resized = resize(imgs_test, size=(self.clip_input_resolution, self.clip_input_resolution))
        imgs_normalized = self.clip_normalize(imgs_resized)
        image_features_clip = self.clip_model.encode_image(imgs_normalized)
        image_features_clip /= image_features_clip.norm(dim=-1, keepdim=True)

        probs = F.softmax(outputs, dim=1)
        similarity = (100.0 * image_features_clip @ self.text_features.T).softmax(dim=-1)
        expected_reward = (probs * similarity).sum(dim=1)
        loss_rl = -expected_reward.mean()

        loss_entropy = self.softmax_entropy(outputs).mean()
        total_loss = loss_rl + self.lambda_entropy * loss_entropy
        
        return outputs, total_loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        Forward and adapt model on batch of data.
        """
        imgs_test = x[0]
        
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation_grad(imgs_test)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation_grad(imgs_test)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return outputs

    def loss_calculation_grad(self, imgs_test):
        """
        Calculates the adaptation loss for the main adaptation step (with gradients).
        """
        outputs = self.model(imgs_test)

        # --- FIX: Preprocess images for CLIP before encoding ---
        with torch.no_grad():
            imgs_resized = resize(imgs_test, size=(self.clip_input_resolution, self.clip_input_resolution))
            imgs_normalized = self.clip_normalize(imgs_resized)
        
        # The encode_image call is now outside the no_grad block if you intend to fine-tune CLIP
        # But for this method, CLIP is frozen, so it doesn't matter much.
        # It's cleaner to just preprocess inside no_grad.
        image_features_clip = self.clip_model.encode_image(imgs_normalized)
        image_features_clip /= image_features_clip.norm(dim=-1, keepdim=True)
        
        probs = F.softmax(outputs, dim=1)
        similarity = (100.0 * image_features_clip @ self.text_features.T).softmax(dim=-1)
        expected_reward = (probs * similarity).sum(dim=1)
        loss_rl = -expected_reward.mean()

        loss_entropy = self.softmax_entropy(outputs).mean()
        total_loss = loss_rl + self.lambda_entropy * loss_entropy
        
        return outputs, total_loss

    def collect_params(self):
        """Collects the affine scale + shift parameters from batch norms."""
        params = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias']:
                        params.append(p)
        return params, None # Return None for names if not used

    def configure_model(self):
        """Configures model for adaptation."""
        self.model.train() # Use train mode for BN updates
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None