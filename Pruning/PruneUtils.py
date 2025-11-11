import torch

def compute_grads(model, loss_fn, device, dataloader):
    model.zero_grad()
    n_batches = 0
    loss_value = 0.0

    for (x, y) in dataloader:
        x, y = x.to(device), y.to(device)
        n_batches += 1

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            logits = model(x)
            loss = loss_fn(logits, y)
        # accumula i gradienti nei .grad
        loss.float().backward()
        loss_value += loss.item()

    # calcola il gradiente medio per ogni parametro
    for params in model.parameters():
        if params.grad is not None:
            params.grad /= n_batches

    return loss_value / n_batches


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
    qk_dim = attn_block.head_dim
    v_dim = attn_block.head_dim
    emb_in = attn_block.qkv.in_features
    emb_out = attn_block.proj.out_features

    qkv = attn_block.qkv
    proj = attn_block.proj

    qkv_shape = (3, n_heads, qk_dim, emb_in)
    proj_shape = (emb_out, n_heads, v_dim)
    qkv_bias_shape = (3, n_heads, qk_dim)

    # attualmente qkv ha shape (3*H*QK, EMB)
    # deve diventare (3, H, QK, EMB)
    qkv_view = qkv.weight.reshape(qkv_shape)

    # simile per i bias
    qkv_bias_view = qkv.bias.reshape(qkv_bias_shape)

    # proj ha shape (EMB, H*V)
    # diventa (EMB, H, V)
    proj_view = proj.weight.reshape(proj_shape)

    q_weights = qkv_view[0]
    k_weights = qkv_view[1]
    v_weights = qkv_view[2]

    q_bias = qkv_bias_view[0]
    k_bias = qkv_bias_view[1]
    v_bias = qkv_bias_view[2]

    qkv_grads_view = qkv.weight_orig.grad.reshape(qkv_shape)
    q_weights_grad, k_weights_grad, v_weights_grad = qkv_grads_view[0], qkv_grads_view[1], qkv_grads_view[2]
    qkv_weights_mask_view = qkv.weight_mask.reshape(qkv_shape)
    q_weights_mask, k_weights_mask, v_weights_mask = qkv_weights_mask_view[0], qkv_weights_mask_view[1], \
    qkv_weights_mask_view[2]

    qkv_bias_grad_view = qkv.bias_orig.grad.reshape(qkv_bias_shape)
    q_bias_grad, k_bias_grad, v_bias_grad = qkv_bias_grad_view[0], qkv_bias_grad_view[1], qkv_bias_grad_view[2]
    qkv_bias_mask_view = qkv.bias_mask.reshape(qkv_bias_shape)
    q_bias_mask, k_bias_mask, v_bias_mask = qkv_bias_mask_view[0], qkv_bias_mask_view[1], qkv_bias_mask_view[2]

    proj_grad = proj.weight_orig.grad.reshape(proj_shape)
    proj_weight_mask = proj.weight_mask.reshape(proj_shape)
    proj_bias_grad = proj.bias_orig.grad
    proj_bias_mask = proj.bias_mask

    Q = WeightBias(q_weights, q_bias, q_weights_grad, q_bias_grad, q_weights_mask, q_bias_mask)
    K = WeightBias(k_weights, k_bias, k_weights_grad, k_bias_grad, k_weights_mask, k_bias_mask)
    V = WeightBias(v_weights, v_bias, v_weights_grad, v_bias_grad, v_weights_mask, v_bias_mask)
    proj = WeightBias(proj_view, proj.bias, proj_grad, proj_bias_grad, proj_weight_mask, proj_bias_mask)

    return HeadAligned(Q, K, V, proj)
