import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizer import BPETokenizer

# Hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_embd = 384
n_head = 6
n_layer = 6
num_experts = 8
top_k = 2
dropout = 0.2
vocab_size = 5000
weight_decay = 1e-2

torch.manual_seed(1337)

with open("input.txt", encoding='utf-8') as file:
    text = file.read()

# BPE Tokenizer
bpe = BPETokenizer(vocab_size=vocab_size)

tokens = bpe.tokenize_text(text)
bpe.fit(tokens)
bpe.save('final-tokenizer.json')

encode = lambda t: bpe.encode(t)								#encode: string -> [ints]
decode = lambda s: bpe.decode(s)								#decode: [ints] -> string

# Training Data v/s Testing Data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))										#90% Train, 10% Validation
train_data = data[:n]
val_data = data[n:]

# Loading Data
def get_batch(split):
	# Data Inputs: x, y
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])
	x, y = x.to(device), y.to(device)
	return x, y

@torch.no_grad
def estimate_loss():
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			logits, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

# Head
class Head(nn.Module):
	# Self Attention - Single Head
	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		B,T,C = x.shape
		k = self.key(x)
		q = self.query(x)
		weight = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 	# (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

		weight = weight.masked_fill(self.tril[:T, :T]==0, float('-inf'))
		weight = F.softmax(weight, dim=-1)
		weight = self.dropout(weight)

		v = self.value(x)
		out = weight @ v
		return out

# Multi Head
class MultiHeadAttention(nn.Module):
	# Self Attention - Multi Heading
	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(head_size * num_heads, n_embd)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out))
		return out

# FeedForward -> Expert
class FeedForward(nn.Module):

	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.ReLU(),											# Rectified Linear Unit
			nn.Linear(4 * n_embd, n_embd),
			nn.Dropout(dropout),
		)

	def forward(self, x):
		return self.net(x)

# Top k Router
class NoisyTopkRouter(nn.Module):

	# Top K Gating with Noise
	def __init__(self, n_embd, num_experts, top_k):
		super().__init__()
		self.top_k = top_k

		# logits router layer
		self.topkrouter_linear = nn.Linear(n_embd, num_experts)
		self.noise_linear = nn.Linear(n_embd, num_experts)

	def forward(self, mh_output):
		logits = self.topkrouter_linear(mh_output)
		noise_logits = self.noise_linear(mh_output)						# Linear Noise Logits MultiHeadAttentionO/P -> I/P

		# Gaussian Noise
		noise = torch.randn_like(logits) * F.softplus(noise_logits)
		noisy_logits = logits + noise

		top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
		zeros = torch.full_like(noisy_logits, float('-inf'))
		sparse_logits = zeros.scatter(-1,  indices, top_k_logits)
		router_output = F.softmax(sparse_logits, dim=-1)
		return router_output, indices

# MoE Block
class MOE(nn.Module):
	
	def __init__(self, n_embd, num_experts, top_k, capacity_factor=1.0):
		super().__init__()
		self.router = NoisyTopkRouter(n_embd, num_experts, top_k)
		self.experts = nn.ModuleList([FeedForward(n_embd) for _  in range(num_experts)])
		self.top_k = top_k
		self.capacity_factor = capacity_factor
		self.num_experts = num_experts

	def forward(self, x):
		batch, seq_len, _ = x.shape					# Batching for capacity factor
		gate_output, indices = self.router(x)
		final_output = torch.zeros_like(x)

		# Reshape
		flat_x = x.view(-1, x.size(-1))
		flat_gate_output = gate_output.view(-1, gate_output.size(-1))

		tokens_per_batch = batch * seq_len * self.top_k
		expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)

		updates = torch.zeros_like(flat_x)

		# Data Parallel
		for i, expert in enumerate(self.experts):
			expert_mask = (indices == i).any(dim=-1)
			flat_mask = expert_mask.view(-1)
			selected_indices = torch.nonzero(flat_mask).squeeze(-1)

			limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
			if limited_indices.numel() > 0:
				expert_input = flat_x[limited_indices]
				expert_output = expert(expert_input)

				gating_scores = flat_gate_output[limited_indices, i].unsqueeze(1)
				weighted_output = expert_output * gating_scores

				updates.index_add_(0, limited_indices, weighted_output)

		final_output += updates.view(batch, seq_len, -1)			# Summing up, reshape
		
		return final_output

# Block
class Block(nn.Module):
	
	def __init__(self, n_embd, n_head, num_experts, top_k):
		super().__init__()
		head_size = n_embd // n_head
		self.sa = MultiHeadAttention(n_head, head_size)
		# self.ffwd = FeedForward(n_embd)
		self.moe = MOE(n_embd, num_experts, top_k)
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)

	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		x = x + self.moe(self.ln2(x))
		return x

# GPT Neural Network
class PicoGPTLanguageModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, num_experts=num_experts, top_k=top_k) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd) 						# Final LayerNormalizer
		self.lm_head = nn.Linear(n_embd, vocab_size) 			# Linear Model Heading

		self.apply(self._init_weights)

	# Initialize Weights
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx, targets=None):
		B, T = idx.shape
		tok_embd = self.token_embedding_table(idx) 				# (Batches, Tokens/Time, Channels)
		pos_embd = self.position_embedding_table(torch.arange(T, device=device))
		x = tok_embd + pos_embd
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.lm_head(x) 								# (Batches, Tokens, vocab_size)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)
		return logits, loss

	def generate(self, idx, max_new_tokens, temperature=1.0):
		for _ in range(max_new_tokens):
			idx_contd = idx[:, -block_size:]
			logits, loss = self(idx_contd) 						# predictions
			logits = logits[:, -1, :] / temperature 			# (B, T) -> (B, C)
			probs = F.softmax(logits, dim=-1)					# (B, C)
			idx_next = torch.multinomial(probs, num_samples=1)	# (B, 1)
			idx = torch.cat((idx, idx_next), dim=1)				# (B, T+1)
			if idx_next == vocab_size - 1:	break
		return idx

# Model Init
model = PicoGPTLanguageModel()
m = model.to(device)

# Print Total Parameters
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Adam Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training Loop
for iter in range(max_iters):

	if iter % eval_interval == 0 or iter == max_iters - 1:
		losses = estimate_loss()
		print(f"Step: {iter}, Train Loss: {losses['train']:.4f}, Eval Loss: {losses['val']:.4f}")

	# Data Sampler
	xb, yb = get_batch('train')									# Creates Batches

	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()

# Generating Text
# Generate random text
idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=500, temperature=1.0)[0].tolist()))

# Generate with a prompt
def generate_response(prompt, max_new_tokens=500, temperature=1.0):
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0) # add prompt as a starting point for text generation
    gen_idx = model.generate(idx, max_new_tokens, temperature=temperature)
    response = decode(gen_idx[0].tolist())
    return response

print(generate_response("Hello world, This is a sample text."))
