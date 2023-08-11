import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, dim_model, vocab_size):
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dim_model)
    
class PostionalEmbeddings(nn.Module):
    def __init__(self, dim_model: int, seq_len):
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout()
        # Creating a tensor with dim (seq_len, dim_model)
        encodings = torch.zeros(seq_len, dim_model) # This is where all the encodings will be stored

        # Create a tensor with dim (seq_len, 1)
        pos = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # This is the numerator term of the equation

        denominator = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)) 

        encodings[:, 0::2] = torch.sin(pos*denominator)
        encodings[:, 1::2] = torch.cos(pos*denominator)

        # Add a batch Dimension
        encodings = encodings.unsqueeze(1)
        self.register_buffer("encodings", encodings)


    def forward(self, x):
        return x.dropout( x + self.pe[:,:x.shape[1],:].requires_grad(False))

class LayernNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # For Multiplication (Learnable parameter) 
        self.bias = nn.Parameter(torch.zeros(1)) # For Addition (Learnable parameter)

    def forward(self, x):
        # Dimensions of x is (batch, seq_len, hidden_size)
        numerator = x - x.mean(dim = -1, keepdims=True) # calculating the mean of all the sequences
        denominator = math.sqrt(x.std(dim = -1, keepdims =True)**2 + self.eps) # # calculating the standard deviation of all the sequences
        return self.alpha * numerator/denominator + self.bias
    
class FeedForward(nn.Module): 
    # Linear -> Relu -> Linear
    def __init__(self, dim_model, units: int = 2048):
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, units)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(units, dim_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x

class MultiheadAttention(nn.Module):
    '''Same MHA block used in encoder and decoder
    While using in decoder pass inDecoder as True and pass the encoder outputs (ex) which are by default set to None'''

    def __init__(self, h, dim_model, mask = None, inDecoder = False):
        super().__init__()
        self.h = h
        self.mask = mask
        self.inDecoder = inDecoder
        assert dim_model % h == 0, f"dim_model ({dim_model}) is not divisible by h({h})"        
        
        self.dk = dim_model // h
        self.wq = nn.Linear(dim_model, dim_model)
        self.wk = nn.Linear(dim_model, dim_model)
        self.wv = nn.Linear(dim_model, dim_model)
        self.wo = nn.Linear(dim_model, dim_model)

    @staticmethod
    def calculateAttention(self, q, k, v, mask):
        # (batch, h, seq_len, dk) -> (batch, h, dk, seq_len)
        score = (q @ k.transpose(2,3)) / math.sqrt(self.dk)
        # (batch, h, dk, seq_len) -> (batch, h, seq_len, seq_len)
        if self.mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = nn.Softmax(score, dim = -1)
        # (batch, h, seq_len, dk)
        return (score @ v)
    
    def forward(self, x, mask, ex = None):
        '''Only use ex when MHA is used in Decoder'''
        
        if self.inDecoder == True:
            query = self.wq(ex)
            key = self.wk(ex)
            value = self.wv(x)
            
        # (Batch, seq_len, dim_model)
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        
        # (Batch, seq_len, h, dk) -> After transpose -> (batch, h, seq_len, dk)
        query = query.view(query.shape[0], query.shape[1], self.h, self.dk).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.dk).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dk).transpose(1,2)

        attentionScores = MultiheadAttention.calculateAttention(query, key, value, mask) # (batch, h, seq_len, dk)
        # (batch, seq_len, h, dk)
        attentionScores = attentionScores.transpose(1,2).contiguous().view(attentionScores.shape[0], attentionScores.shape[1], self.h*self.dk)

        x = self.wo(attentionScores @ value)
        return x

class Encoder(nn.Module):
    def __init__(self, dim_model, h, vocab_size, seq_len, units: int = 2048):
        super().__init__()
        self.inputEmbed = InputEmbeddings(dim_model, vocab_size)
        self.posEmbed = PostionalEmbeddings(dim_model, seq_len)
        self.MHA = MultiheadAttention(h, dim_model)
        self.norm_1 = LayernNormalization()
        self.norm_2 = LayernNormalization()
        self.feedforward = FeedForward(dim_model, units)
    
    def forward(self, x, mask):
        input = self.inputEmbed(x)
        pos = self.posEmbed(input)
        mha = self.MHA(pos, mask)
        norm_1 = self.norm_1(pos+mha)
        ff = self.feedforward(norm_1)
        norm_2 = self.norm_2(ff+norm_1)

        return norm_2

class Decoder(nn.Module):
    def __init__(self, dim_model, h, vocab_size, seq_len, units: int = 2048):
        super().__init__()
        self.inputEmbed = InputEmbeddings(dim_model, vocab_size)
        self.posEmbed = PostionalEmbeddings(dim_model, seq_len)
        self.MaskedMHA = MultiheadAttention(h, dim_model)
        self.MHA = MultiheadAttention(h, dim_model, inDecoder = True)
        self.norm_1 = LayernNormalization()
        self.norm_2 = LayernNormalization()
        self.norm_3 = LayernNormalization()
        self.feedforward = FeedForward(dim_model, units)
        
    def forward(self, x, src_mask, tgt_mask, ex):
        input = self.inputEmbed(x)
        pos = self.posEmbed(input)
        mmha = self.MaskedMHA(pos, tgt_mask)
        norm_1 = self.norm_1(mmha+pos)
        mha = self.MHA(pos, ex, src_mask)
        norm_2 = self.norm_2(mha+norm_1)
        ff = self.feedforward(norm_2)
        norm_3 = self.norm_3(norm_2+ff)

        return norm_3
    
class finalLayer(nn.Module):
    def __init__(self, dim_model, vocab_size):
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(dim_model, vocab_size)
    def forward(self, x):
        return nn.Softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, finalLayer: finalLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.finalLayer = finalLayer

    def forward(self, inputs, outputs, src_mask, tgt_mask):
        enOut = self.encoder(inputs, src_mask)
        x = self.decoder(outputs, src_mask, tgt_mask, enOut)
        x = self.finalLayer(x)
    
        return x
    
def build_transformer(dim_model, h, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, units: int = 2048):
    
    # Create object for encoder, decoder and finalLayer 
    encoder = Encoder(dim_model, h, src_vocab_size, src_seq_len)
    decoder = Decoder(dim_model, h, tgt_vocab_size, tgt_seq_len)
    LastLayer = finalLayer(dim_model, tgt_vocab_size)

    # Create object for Transformer and pass the encoder, decoder and finalLayer
    transformer = Transformer(encoder, decoder, LastLayer)

    # Initializing parameters which would help in training
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer