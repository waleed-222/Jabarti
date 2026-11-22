import torch
import torch.nn as nn 


class InputEmbedding(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        self.max_seq_len= config.max_seq_len
        self.dropout = nn.Dropout(config.dropout) # Layer
        
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )
        
        self.positional_embedding = nn.Embedding(
            config.max_seq_len, config.d_model
        )
        
        
    
    def forward(self, input_ids, position_offset=0):
        
        _, seq_len, = input_ids.shape
        if position_offset +seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {position_offset+seq_len} exceeds max_seq_length {self.max_seq_len}"
            )
            
        positions = torch.arange(
            position_offset, position_offset+seq_len, device=input_ids.device
        )
        tokens= self.token_embedding(input_ids)
        pos= self.positional_embedding(positions)
        
        return self.dropout(tokens + pos)
        
        
