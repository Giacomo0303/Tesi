import torch
from sklearn.metrics import balanced_accuracy_score


def compute_grads(model, loss_fn, device, dataloader):
    model.zero_grad()
    n_batches = 0
    loss_value = 0.0
    y_true, y_pred = [], []
    scaler = torch.amp.GradScaler()

    for p in model.parameters():
        if p.requires_grad:
            # Creiamo l'attributo .imp sullo stesso device e con la stessa shape del parametro
            p.imp = torch.zeros_like(p.data)

    for (x, y) in dataloader:
        x, y = x.to(device), y.to(device)
        n_batches += 1
        y_true.append(y.cpu())

        #azzero i gradienti
        model.zero_grad()

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            logits = model(x)
            loss = loss_fn(logits, y)
            y_pred.append(torch.argmax(logits, dim=-1).cpu())

        # accumula i gradienti nei .grad
        scaler.scale(loss).backward()
        loss_value += loss.item()

        for p in model.parameters():
            if p.grad is not None:
                # Calcolo g^2 per questo batch e lo aggiungo all'accumulatore
                # .detach() è importante per non mantenere il grafo computazionale e risparmiare memoria
                p.imp += p.grad.data.pow(2).detach()

    for p in model.parameters():
        if hasattr(p, 'imp'):
            # Calcolo la media dei gradienti al quadrato (g^2)
            p.imp /= n_batches

            # Moltiplico per il quadrato del peso (w^2)
            # p.data è il valore del peso (w)
            p.imp *= p.data.pow(2)

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    accuracy = balanced_accuracy_score(y_true.cpu(), y_pred.cpu())

    return loss_value / n_batches, accuracy


def importance_score(parts):

    accumul = 0.0

    for part in parts:
        # Somma tutti i valori scalari nel tensore/slice (riduzione a scalare)
        accumul += torch.sum(part).item()

    return accumul


class WeightImp():
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, w_imp: torch.Tensor, b_imp: torch.Tensor,
                 w_mask: torch.Tensor, b_mask: torch.Tensor):
        self.weight = weight
        self.bias = bias
        self.weight_imp = w_imp
        self.bias_imp = b_imp
        self.weight_mask = w_mask
        self.bias_mask = b_mask


class HeadAligned():
    def __init__(self, Q: WeightImp, K: WeightImp, V: WeightImp, proj: WeightImp):
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
    # Se pruning.identity è attivo, usiamo _orig, altrimenti i pesi standard
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

    # --- 2. QKV: PESI e BIAS ---
    q_w, k_w, v_w = torch.split(qkv_w_src, split_sections, dim=0)
    q_b, k_b, v_b = torch.split(qkv_b_src, split_sections, dim=0)

    # Reshape Pesi e Bias
    q_weights = q_w.reshape(n_heads, qk_dim, emb_in)
    k_weights = k_w.reshape(n_heads, qk_dim, emb_in)
    v_weights = v_w.reshape(n_heads, v_dim, emb_in)

    q_bias = q_b.reshape(n_heads, qk_dim)
    k_bias = k_b.reshape(n_heads, qk_dim)
    v_bias = v_b.reshape(n_heads, v_dim)

    # --- GESTIONE SICURA IMPORTANZE QKV ---
    # Default a None
    q_weights_imp, k_weights_imp, v_weights_imp = None, None, None
    q_bias_imp, k_bias_imp, v_bias_imp = None, None, None

    # Controlliamo se .imp esiste prima di usarlo (FIX PER L'ERRORE)
    if hasattr(qkv_w_src, 'imp') and qkv_w_src.imp is not None:
        q_imp, k_imp, v_imp = torch.split(qkv_w_src.imp, split_sections, dim=0)
        q_weights_imp = q_imp.reshape(n_heads, qk_dim, emb_in)
        k_weights_imp = k_imp.reshape(n_heads, qk_dim, emb_in)
        v_weights_imp = v_imp.reshape(n_heads, v_dim, emb_in)

    if hasattr(qkv_b_src, 'imp') and qkv_b_src.imp is not None:
        q_b_imp, k_b_imp, v_b_imp = torch.split(qkv_b_src.imp, split_sections, dim=0)
        q_bias_imp = q_b_imp.reshape(n_heads, qk_dim)
        k_bias_imp = k_b_imp.reshape(n_heads, qk_dim)
        v_bias_imp = v_b_imp.reshape(n_heads, v_dim)

    # --- MASCHERE QKV ---
    q_w_mask = torch.ones_like(q_weights)
    k_w_mask = torch.ones_like(k_weights)
    v_w_mask = torch.ones_like(v_weights)
    q_b_mask = torch.ones_like(q_bias)
    k_b_mask = torch.ones_like(k_bias)
    v_b_mask = torch.ones_like(v_bias)

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

    # --- 3. PROJ ---
    proj_shape = (emb_out, n_heads, v_dim)
    proj_view = proj_w_src.reshape(proj_shape)

    # Importanze Proj (Default None)
    proj_imp = None
    proj_b_imp = None

    # Controlli sicuri anche qui
    if hasattr(proj_w_src, 'imp') and proj_w_src.imp is not None:
        proj_imp = proj_w_src.imp.reshape(proj_shape)

    if hasattr(proj_b_src, 'imp') and proj_b_src.imp is not None:
        proj_b_imp = proj_b_src.imp

        # Maschere Proj
    proj_w_mask = torch.ones_like(proj_view)
    if hasattr(proj, 'weight_mask') and proj.weight_mask is not None:
        proj_w_mask = proj.weight_mask.reshape(proj_shape)

    proj_b_mask = torch.ones_like(proj_b_src)
    if hasattr(proj, 'bias_mask') and proj.bias_mask is not None:
        proj_b_mask = proj.bias_mask

    # Creazione Oggetti
    Q = WeightImp(q_weights, q_bias, q_weights_imp, q_bias_imp, q_w_mask, q_b_mask)
    K = WeightImp(k_weights, k_bias, k_weights_imp, k_bias_imp, k_w_mask, k_b_mask)
    V = WeightImp(v_weights, v_bias, v_weights_imp, v_bias_imp, v_w_mask, v_b_mask)

    proj_obj = WeightImp(proj_view, proj_b_src, proj_imp, proj_b_imp, proj_w_mask, proj_b_mask)

    return HeadAligned(Q, K, V, proj_obj)
