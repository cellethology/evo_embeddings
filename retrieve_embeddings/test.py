from evo import Evo
import torch

device = 'cuda:0'

evo_model = Evo('evo-1.5-8k-base')
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device)
model.eval()

# monkey patch the unembed function with identity
# this removes the final projection back from the embedding space into tokens
# so the "logits" of the model is now the final layer embedding
# see source for unembed - https://huggingface.co/togethercomputer/evo-1-131k-base/blob/main/model.py#L339

from torch import nn

class CustomEmbedding(nn.Module):
  def unembed(self, u):
    return u

model.unembed = CustomEmbedding()

# end custom code

sequence = 'ACGT'
input_ids = torch.tensor(
    tokenizer.tokenize(sequence),
    dtype=torch.int,
).to(device).unsqueeze(0)

embed, _ = model(input_ids) # (batch, length, embed dim)

print('Embed: ', embed)
print('Shape (batch, length, embed dim): ', embed.shape)

# you can now use embedding for downstream classification tasks
# you probably want to aggregate over position dimension
# e.g. mean value = embed.mean(dim=1) or final token embedding = embed[:, -1, :]
