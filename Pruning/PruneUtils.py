import torch
from sklearn.metrics import balanced_accuracy_score


def compute_grads(model, loss_fn, device, dataloader):
    model.zero_grad()
    n_batches = 0
    loss_value = 0.0
    y_true, y_pred = [], []

    for (x, y) in dataloader:
        x, y = x.to(device), y.to(device)
        n_batches += 1
        y_true.append(y)

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            logits = model(x)
            loss = loss_fn(logits, y)
            y_pred.append(torch.argmax(logits, dim=-1))

        # accumula i gradienti nei .grad
        loss.float().backward()
        loss_value += loss.item()

    # calcola il gradiente medio per ogni parametro
    for params in model.parameters():
        if params.grad is not None:
            params.grad /= n_batches

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    accuracy = balanced_accuracy_score(y_true.cpu(), y_pred.cpu())

    return loss_value / n_batches, accuracy


def importance_score(weights, grads):
    accumul = 0.0

    for weight_group, grad in zip(weights, grads):
        accumul += torch.sum(weight_group * grad)

    return (accumul ** 2).item()


class WeightBias():
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, w_grad: torch.Tensor, b_grad: torch.Tensor,
                 w_mask: torch.Tensor, b_mask: torch.Tensor):
        self.weight = weight
        self.bias = bias
        self.weight_grad = w_grad
        self.bias_grad = b_grad
        self.weight_mask = w_mask
        self.bias_mask = b_mask


class HeadAligned():
    def __init__(self, Q: WeightBias, K: WeightBias, V: WeightBias, proj: WeightBias):
        self.Q = Q
        self.K = K
        self.V = V
        self.proj = proj


def head_alignment(attn_block) -> HeadAligned:
    n_heads = attn_block.num_heads

    if hasattr(attn_block, 'qk_dim'):
        qk_dim = attn_block.qk_dim
        v_dim = attn_block.v_dim
    else:
        qk_dim = attn_block.head_dim
        v_dim = attn_block.head_dim

    emb_in = attn_block.qkv.in_features
    emb_out = attn_block.proj.out_features

    qkv = attn_block.qkv
    proj = attn_block.proj

    total_qk_size = n_heads * qk_dim
    total_v_size = n_heads * v_dim
    split_sections = [total_qk_size, total_qk_size, total_v_size]

    # --- 1. SELEZIONE SORGENTI ---
    if hasattr(qkv, "weight_orig"):
        qkv_w_src = qkv.weight_orig
        qkv_b_src = qkv.bias_orig
    else:
        qkv_w_src = qkv.weight
        qkv_b_src = qkv.bias

    if hasattr(proj, "weight_orig"):
        proj_w_src = proj.weight_orig
        proj_b_src = proj.bias_orig
    else:
        proj_w_src = proj.weight
        proj_b_src = proj.bias

    # --- 2. PESI (WEIGHTS) ---
    q_w, k_w, v_w = torch.split(qkv_w_src, split_sections, dim=0)
    q_weights = q_w.reshape(n_heads, qk_dim, emb_in)
    k_weights = k_w.reshape(n_heads, qk_dim, emb_in)
    v_weights = v_w.reshape(n_heads, v_dim, emb_in)

    # --- 3. BIAS ---
    q_b, k_b, v_b = torch.split(qkv_b_src, split_sections, dim=0)
    q_bias = q_b.reshape(n_heads, qk_dim)
    k_bias = k_b.reshape(n_heads, qk_dim)
    v_bias = v_b.reshape(n_heads, v_dim)

    # --- 4. GRADIENTI (Separati e Sicuri) ---
    q_weights_grad, k_weights_grad, v_weights_grad = None, None, None
    q_bias_grad, k_bias_grad, v_bias_grad = None, None, None
    proj_grad, proj_bias_grad = None, None

    # Gradienti Pesi
    if qkv_w_src.grad is not None:
        q_g, k_g, v_g = torch.split(qkv_w_src.grad, split_sections, dim=0)
        q_weights_grad = q_g.reshape(n_heads, qk_dim, emb_in)
        k_weights_grad = k_g.reshape(n_heads, qk_dim, emb_in)
        v_weights_grad = v_g.reshape(n_heads, v_dim, emb_in)

    # Gradienti Bias (Controllo separato!)
    if qkv_b_src.grad is not None:
        q_bg, k_bg, v_bg = torch.split(qkv_b_src.grad, split_sections, dim=0)
        q_bias_grad = q_bg.reshape(n_heads, qk_dim)
        k_bias_grad = k_bg.reshape(n_heads, qk_dim)
        v_bias_grad = v_bg.reshape(n_heads, v_dim)

    # --- 5. MASCHERE QKV (FUORI dall'if dei gradienti!) ---
    # Inizializziamo a 1 (nessun pruning)
    q_w_mask = torch.ones_like(q_weights)
    k_w_mask = torch.ones_like(k_weights)
    v_w_mask = torch.ones_like(v_weights)

    q_b_mask = torch.ones_like(q_bias)
    k_b_mask = torch.ones_like(k_bias)
    v_b_mask = torch.ones_like(v_bias)

    # Sovrascriviamo se esistono le maschere reali
    if hasattr(qkv, 'weight_mask') and qkv.weight_mask is not None:
        q_m, k_m, v_m = torch.split(qkv.weight_mask, split_sections, dim=0)
        q_w_mask = q_m.reshape(n_heads, qk_dim, emb_in)
        k_w_mask = k_m.reshape(n_heads, qk_dim, emb_in)
        v_w_mask = v_m.reshape(n_heads, v_dim, emb_in)

    if hasattr(qkv, 'bias_mask') and qkv.bias_mask is not None:
        q_bm, k_bm, v_bm = torch.split(qkv.bias_mask, split_sections, dim=0)
        q_b_mask = q_bm.reshape(n_heads, qk_dim)
        k_b_mask = k_bm.reshape(n_heads, qk_dim)
        v_b_mask = v_bm.reshape(n_heads, v_dim)

    # --- 6. PROJ ---
    proj_shape = (emb_out, n_heads, v_dim)
    proj_view = proj_w_src.reshape(proj_shape)

    if proj_w_src.grad is not None:
        proj_grad = proj_w_src.grad.reshape(proj_shape)

    proj_w_mask = torch.ones_like(proj_view)
    if hasattr(proj, 'weight_mask') and proj.weight_mask is not None:
        proj_w_mask = proj.weight_mask.reshape(proj_shape)

    # Bias Proj
    proj_b_grad = proj_b_src.grad  # Può essere None, va bene

    proj_b_mask = torch.ones_like(proj_b_src)
    if hasattr(proj, 'bias_mask') and proj.bias_mask is not None:
        proj_b_mask = proj.bias_mask

    # --- 7. ASSEMBLAGGIO ---
    Q = WeightBias(q_weights, q_bias, q_weights_grad, q_bias_grad, q_w_mask, q_b_mask)
    K = WeightBias(k_weights, k_bias, k_weights_grad, k_bias_grad, k_w_mask, k_b_mask)
    V = WeightBias(v_weights, v_bias, v_weights_grad, v_bias_grad, v_w_mask, v_b_mask)

    proj_obj = WeightBias(proj_view, proj_b_src, proj_grad, proj_b_grad, proj_w_mask, proj_b_mask)

    return HeadAligned(Q, K, V, proj_obj)
