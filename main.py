import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. 설정 (A100 최적화 세팅)
# =============================================================================
BATCH_SIZE = 64
BLOCK_SIZE = 256       # A100이니까 문맥도 좀 더 길게(128->256) 봐도 됩니다!
MAX_ITERS = 5000       # 일단 5000번만 빠르게 돌려봅시다.
LEARNING_RATE = 3e-4
DEVICE = 'cuda'        # A100이 있으니까 무조건 CUDA
N_EMBD = 1024          # 차원 뻥튀기 (1024)
N_HEAD = 8             # [중요] 1024를 8로 나눠야 딱 떨어짐 (128)
N_LAYER = 8            # A100이니까 층도 좀 더 쌓읍시다 (6->8)
DROPOUT = 0.1          # 0.2는 좀 많으니 0.1로

print(f"🚀 Device: {DEVICE} (A100 Ready)")

# =============================================================================
# 2. 데이터 준비
# =============================================================================
if not os.path.exists('input.txt'):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open('input.txt', 'w') as f: f.write(requests.get(url).text)

with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# =============================================================================
# 3. 안정화된 사원수 비트 리니어
# =============================================================================
def bitnet_quantize(w):
    scale = w.abs().mean().clamp(min=1e-8)
    w_centered = w - w.mean()
    w_bin = torch.round(torch.clamp(w_centered / scale, -1, 1))
    return (w_bin - w).detach() + w, scale

class QuatBitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # 차원 검사 (이젠 1024, 8이라 에러 안 날 겁니다)
        assert in_features % 4 == 0, f"입력 차원({in_features})이 4의 배수가 아님"
        assert out_features % 4 == 0, f"출력 차원({out_features})이 4의 배수가 아님"

        self.in_channels = in_features // 4
        self.out_channels = out_features // 4

        init_std = 0.02
        self.r_weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels) * init_std)
        self.i_weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels) * init_std)
        self.j_weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels) * init_std)
        self.k_weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels) * init_std)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.training_step = 0

    def forward(self, x):
        r_in, i_in, j_in, k_in = torch.chunk(x, 4, dim=-1)

        if self.training and self.training_step < 500:
            r_w, i_w, j_w, k_w = self.r_weight, self.i_weight, self.j_weight, self.k_weight
            scale = 1.0
        else:
            r_w, s1 = bitnet_quantize(self.r_weight)
            i_w, s2 = bitnet_quantize(self.i_weight)
            j_w, s3 = bitnet_quantize(self.j_weight)
            k_w, s4 = bitnet_quantize(self.k_weight)
            scale = (s1 + s2 + s3 + s4) / 4

        if self.training: self.training_step += 1

        r_out = F.linear(r_in, r_w) - F.linear(i_in, i_w) - F.linear(j_in, j_w) - F.linear(k_in, k_w)
        i_out = F.linear(r_in, i_w) + F.linear(i_in, r_w) + F.linear(j_in, k_w) - F.linear(k_in, j_w)
        j_out = F.linear(r_in, j_w) - F.linear(i_in, k_w) + F.linear(j_in, r_w) + F.linear(k_in, i_w)
        k_out = F.linear(r_in, k_w) + F.linear(i_in, j_w) - F.linear(j_in, i_w) + F.linear(k_in, r_w)

        output = torch.cat([r_out, i_out, j_out, k_out], dim=-1)
        if self.bias is not None: output += self.bias
        return output * scale * 0.5

# =============================================================================
# 4. 모델 아키텍처
# =============================================================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = QuatBitLinear(N_EMBD, head_size, bias=False)
        self.query = QuatBitLinear(N_EMBD, head_size, bias=False)
        self.value = QuatBitLinear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = QuatBitLinear(num_heads * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            QuatBitLinear(n_embd, 4 * n_embd),
            nn.GELU(),
            QuatBitLinear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class QuatBitGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =============================================================================
# 5. A100 파워 실행 (수정판)
# =============================================================================
# [중요] A100 성능 잠금 해제 (TF32)
torch.set_float32_matmul_precision('high')

model = QuatBitGPT().to(DEVICE)

# [수정] 컴파일 끄기! (질문자님 코드가 너무 독창적이라 컴파일러가 렉 걸림)
# model = torch.compile(model)

num_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 QuatBitNet Params: {num_params/1e6:.2f}M")
print(f"💡 파라미터 25M 돌파! (아까의 10배 체급입니다)")
print(f"💡 A100 TF32 가속: ON (컴파일 없이 깡성능으로 밀어붙입니다)")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\n🚀 Training Stabilized QuatBitNet on A100...")
start_time = time.time()

for iter in range(MAX_ITERS):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # 100 step마다 로그 출력
    if iter % 100 == 0:
        dt = time.time() - start_time
        # 속도(ms/step) 계산
        ms = dt * 1000 / 100
        print(f"Step {iter}: Loss {loss.item():.4f} | 속도: {ms:.2f}ms/step")
        start_time = time.time()

print("\n📜 생성 결과 확인:")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
try:
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
except Exception as e:
    print(e)


# =============================================================================
# 6. 모델 해부 및 시각화 도구 (자동 실행)
# =============================================================================

def visualize_weight_distribution(model, layer_idx=0):
    # 1. 모델에서 첫 번째 블록의 Attention Projection 레이어를 가져옵니다.
    target_layer = model.blocks[layer_idx].sa.proj

    # 2. 사원수 가중치 (r, i, j, k) 가져오기
    weights = {
        'Real (r)': target_layer.r_weight,
        'Imag (i)': target_layer.i_weight,
        'Imag (j)': target_layer.j_weight,
        'Imag (k)': target_layer.k_weight
    }

    # 3. 그래프 그리기 준비
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Layer {layer_idx} Quaternion BitNet Weight Analysis', fontsize=16)

    axes = axes.flatten()

    for i, (name, w) in enumerate(weights.items()):
        # 양자화된(BitNet) 형태를 흉내내서 보여줌 (-1, 0, 1)
        w_quantized, _ = bitnet_quantize(w)
        w_flat = w_quantized.detach().cpu().flatten().numpy()

        # 히스토그램 그리기
        sns.histplot(w_flat, bins=30, ax=axes[i], color='skyblue', kde=False)
        axes[i].set_title(f"{name} Component Distribution")
        axes[i].set_xlabel("Weight Value (-1, 0, 1)")
        axes[i].set_ylabel("Count")

        # 실제로 -1, 0, 1에 얼마나 몰려있는지 텍스트로 표시
        zero_count = (w_flat == 0).sum()
        total = len(w_flat)
        axes[i].text(0.95, 0.9, f"Sparsity (0): {zero_count/total*100:.1f}%",
                     transform=axes[i].transAxes, ha='right', color='red')

    plt.tight_layout()
    # plt.show() 대신 저장하도록 변경 (서버/Github 환경 고려)
    plt.savefig('weight_distribution.png')
    print("\n📊 그래프가 'weight_distribution.png'로 저장되었습니다!")

# 실행!
print("🎨 모델 해부 시작... (BitNet의 증거를 찾아서)")
visualize_weight_distribution(model)
