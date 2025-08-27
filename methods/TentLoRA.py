import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy

# =============================================================================
# 1. 桩模块 (Stubs) - 为了让代码独立运行
# 在您的实际项目中，您会从您的库中导入这些
# =============================================================================

# class ADAPTATION_REGISTRY:
#     """一个假的注册表，用于装饰器。"""
#     _registry = {}
#     @classmethod
#     def register(cls):
#         def decorator(subclass):
#             cls._registry[subclass.__name__] = subclass
#             return subclass
#         return decorator

class Entropy(nn.Module):
    """计算熵的损失函数。"""
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        # x 应该是 logits
        p = torch.softmax(x, dim=1)
        log_p = torch.log_softmax(x, dim=1)
        return -(p * log_p).sum(dim=1)


# =============================================================================
# 2. LoRA 实现和注入逻辑
# =============================================================================

class LoRALayer(nn.Module):
    """A standard LoRA layer, containing low-rank matrices A and B."""
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        """Core computation: output = B * A * x * scaling"""
        # --- FIX: Move LoRA matrices to the same device as the input x ---
        lora_A_on_device = self.lora_A.to(x.device)
        lora_B_on_device = self.lora_B.to(x.device)

        if x.dim() > 2:
            return self.scaling * (lora_B_on_device @ (lora_A_on_device @ x.transpose(-1, -2))).transpose(-1, -2)
        else:
            return self.scaling * (lora_B_on_device @ (lora_A_on_device @ x.unsqueeze(-1))).squeeze(-1)


class InjectedLoRALinear(nn.Module):
    """将 LoRA 层注入到现有的 nn.Linear 层中。"""
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer # 原始的、冻结的线性层
        self.lora = LoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def add_lora_to_model(model, rank, alpha):
    """遍历模型，找到所有的 nn.Linear 层，并用 InjectedLoRALinear 替换它们。"""
    print("--- Injecting LoRA layers ---")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = model.get_submodule(parent_name)
            
            injected_module = InjectedLoRALinear(module, rank, alpha)
            setattr(parent_module, child_name, injected_module)
            print(f"  - Injected LoRA into: {name}")
    print("--- LoRA injection complete ---")


# =============================================================================
# 3. 核心实现: TentLoRA
# =============================================================================

@ADAPTATION_REGISTRY.register()
class TentLoRA(TTAMethod):
    """
    TentLoRA 通过在测试时最小化熵来调整模型。
    它专门更新注入到模型中的 LoRA 层的 B 矩阵。
    """
    def __init__(self, cfg, model, num_classes):
        # 在调用父类构造函数之前，先给模型注入 LoRA 层
        self.lora_rank = 8
        self.lora_alpha = 16
        add_lora_to_model(model, self.lora_rank, self.lora_alpha)
        
        super().__init__(cfg, model, num_classes)

        self.softmax_entropy = Entropy()

    def loss_calculation(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        loss = self.softmax_entropy(outputs).mean(0)
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """前向传播并自适应模型。"""
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
        return outputs

    def collect_params(self):
        """只收集需要训练的参数 (LoRA B 矩阵)。"""
        params = []
        names = []
        for nm, p in self.model.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(nm)
        
        print("\n--- Collecting Trainable Parameters ---")
        print(f"  - Found {len(names)} trainable parameters.")
        if len(names) > 0:
            print(f"  - Example trainable params: {names[:5]}")
        print("-------------------------------------\n")

        return params, names

    def configure_model(self):
        """配置模型：冻结所有，然后只解冻 LoRA B 矩阵。"""
        print("\n--- Configuring Model Gradients ---")
        self.model.eval()
        self.model.requires_grad_(False)
        print("  - All model parameters frozen.")
        
        for m in self.model.modules():
            if isinstance(m, InjectedLoRALinear):
                # 解冻 B 矩阵的参数
                m.lora.lora_B.requires_grad = True
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        print("  - Unfroze all LoRA 'B' matrices.")
        print("-----------------------------------\n")

# =============================================================================
# 4. 使用示例
# =============================================================================

if __name__ == '__main__':
    # --- 1. 创建一个模拟的配置对象 ---
    cfg = SimpleNamespace(
        ADAPTATION=SimpleNamespace(
            METHOD='TentLoRA',
            LORA_RANK=8,
            LORA_ALPHA=16
        ),
        OPTIM=SimpleNamespace(
            lr=1e-4,
            beta=0.9,
            wd=0.0,
            mixed_precision=torch.cuda.is_available() # 只在有 CUDA 时使用
        )
    )

    # --- 2. 创建一个简单的示例模型 ---
    num_classes = 10
    original_model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    print("--- Original Model Structure ---")
    print(original_model)
    print("--------------------------------\n")
    
    # --- 3. 初始化 TentLoRA 适配器 ---
    # 这会修改 `original_model` 对象
    print(">>> Initializing TentLoRA adapter...")
    adapter = TentLoRA(cfg, original_model, num_classes)
    print(">>> TentLoRA adapter initialized.\n")

    print("--- Model Structure After LoRA Injection ---")
    print(adapter.model)
    print("------------------------------------------\n")

    # --- 4. 验证哪些参数是可训练的 ---
    print("--- Verifying Trainable Parameters ---")
    for name, param in adapter.model.named_parameters():
        if param.requires_grad:
            print(f"  - Trainable: {name} (Shape: {param.shape})")
        # else:
        #     print(f"  - Frozen:    {name}")
    print("--------------------------------------\n")

    # --- 5. 创建模拟数据并执行一步自适应 ---
    print(">>> Performing one step of forward_and_adapt...")
    # 模拟一个批次的数据
    dummy_input = torch.randn(4, 128).to(adapter.device)
    # 将输入包装在元组中，以匹配 loss_calculation 的期望格式
    dummy_data_batch = (dummy_input,)

    # 获取 B 矩阵在更新前的值
    b_matrix_before = adapter.model[0].lora.lora_B.clone().detach()

    # 执行自适应
    outputs = adapter.forward_and_adapt(dummy_data_batch)

    # 获取 B 矩阵在更新后的值
    b_matrix_after = adapter.model[0].lora.lora_B.clone().detach()

    # 检查 B 矩阵是否已更新
    assert not torch.equal(b_matrix_before, b_matrix_after), "Error: LoRA 'B' matrix was not updated!"
    # 检查 A 矩阵是否未更新
    # (注意: Adam 优化器可能会因为动量等原因稍微改变冻结的参数，这里我们只检查梯度状态)
    assert not adapter.model[0].lora.lora_A.requires_grad, "Error: LoRA 'A' matrix has gradients!"
    
    print(">>> Step successful. LoRA 'B' matrix was updated.")
    print(f"Output shape: {outputs.shape}")