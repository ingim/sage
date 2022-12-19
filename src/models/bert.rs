use crate::layers::{
    Dense, Dropout, Embedding, Layer, LayerNorm, Parameter, Relu, Sequential, Softmax,
};
use crate::session::context::Context;
use crate::tensor::Tensor;
use crate::var::Fun;

#[derive(Copy, Clone)]
pub struct BertConfig {
    embed_dim: usize,

    // Embedding
    num_vocabs: usize,
    max_num_tokens: usize,

    // Encoding
    hidden_dim: usize,
    num_layers: usize,
    num_attention_heads: usize,
    layer_norm_eps: f32,
    dropout_prob: f32,

    // classifier
    num_classes: usize,
}

impl BertConfig {
    pub fn simple() -> Self {
        BertConfig {
            embed_dim: 64,
            num_vocabs: 150,
            max_num_tokens: 512,
            hidden_dim: 128,
            num_layers: 6,
            num_attention_heads: 2,
            layer_norm_eps: 0.0,
            dropout_prob: 0.0,
            num_classes: 10,
        }
    }

    pub fn base() -> Self {
        BertConfig {
            embed_dim: 768,
            num_vocabs: 30522,
            max_num_tokens: 512,
            hidden_dim: 3072,
            num_layers: 12,
            num_attention_heads: 12,
            layer_norm_eps: 1e-12,
            dropout_prob: 0.3,
            num_classes: 2,
        }
    }
}

pub struct Bert {
    embedding: BertEmbedding,
    encoder: BertEncoder,
    classifier: Dense,
}

impl Bert {
    pub fn new(config: BertConfig) -> Self {
        Bert {
            embedding: BertEmbedding::new(config),
            encoder: BertEncoder::new(config),
            classifier: Dense::new(config.embed_dim, config.num_classes),
        }
    }

    pub fn pass(&self, token_ids: &Fun, attn_mask: &Fun) -> Fun {
        let embeddings = self.embedding.pass(token_ids);
        let features = self.encoder.pass(embeddings, attn_mask);

        let cls_tokens = features.index(1, 1).squeeze(1);

        let logits = self.classifier.pass(&cls_tokens);

        logits
    }
}

impl Parameter for Bert {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        self.embedding.init(ctx, level);
        self.encoder.init(ctx, level);
        self.classifier.init(ctx, level);
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Fun>) {
        self.embedding.params(p);
        self.encoder.params(p);
        self.classifier.params(p);
    }
}

struct BertEmbedding {
    word_emb: Embedding,
    pos_emb: Embedding,
    norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbedding {
    pub fn new(config: BertConfig) -> Self {
        BertEmbedding {
            word_emb: Embedding::new(config.num_vocabs, config.embed_dim),
            pos_emb: Embedding::new(config.max_num_tokens, config.embed_dim),
            norm: LayerNorm::new(2, config.layer_norm_eps),
            dropout: Dropout::new(config.dropout_prob),
        }
    }
}

impl Parameter for BertEmbedding {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        self.word_emb.init(ctx, level);
        self.pos_emb.init(ctx, level);
        self.norm.init(ctx, level);
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Fun>) {
        self.word_emb.params(p);
        self.pos_emb.params(p);
        self.norm.params(p);
    }
}

impl Layer for BertEmbedding {
    fn pass(&self, x: &Fun) -> Fun {
        // support 2d token ids
        let seq_len = x.extent(1);
        let pos_ids = Fun::new(Tensor::from_iter(x.extent(1), 0..(seq_len as u32)));

        let word_embeddings = self.word_emb.pass(x);
        let pos_embeddings = self.pos_emb.pass(&pos_ids);

        self.norm.pass(&(word_embeddings + pos_embeddings))
    }
}

struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(config: BertConfig) -> Self {
        BertEncoder {
            layers: (0..config.num_layers)
                .into_iter()
                .map(|_| BertLayer::new(config))
                .collect(),
        }
    }

    pub fn pass(&self, x: Fun, attn_mask: &Fun) -> Fun {
        self.layers
            .iter()
            .fold(x, |x, layer| layer.pass(&x, attn_mask))
    }
}

impl Parameter for BertEncoder {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        for layer in self.layers.iter_mut() {
            layer.init(ctx, level);
        }
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Fun>) {
        for layer in self.layers.iter() {
            layer.params(p);
        }
    }
}

struct BertLayer {
    attention: MultiHeadAttention,
    ffn: Sequential,
    norm: LayerNorm,
}

impl BertLayer {
    pub fn new(config: BertConfig) -> Self {
        BertLayer {
            attention: MultiHeadAttention::new(
                config.embed_dim,
                config.num_attention_heads,
                config.dropout_prob,
                config.layer_norm_eps,
            ),
            ffn: Sequential::new()
                .with(Dense::new(config.embed_dim, config.hidden_dim))
                .with(Relu)
                .with(Dense::new(config.hidden_dim, config.embed_dim))
                .with(Dropout::new(config.dropout_prob)),
            norm: LayerNorm::new(2, config.layer_norm_eps),
        }
    }

    pub fn pass(&self, x: &Fun, attn_mask: &Fun) -> Fun {
        let interim_features = self.attention.pass(x, attn_mask);
        let out_features = self.ffn.pass(&interim_features);
        self.norm.pass(&(out_features + interim_features))
    }
}

impl Parameter for BertLayer {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        self.attention.init(ctx, level);
        self.ffn.init(ctx, level);
        self.norm.init(ctx, level);
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Fun>) {
        self.attention.params(p);
        self.ffn.params(p);
        self.norm.params(p);
    }
}

pub struct MultiHeadAttention {
    embed_dim: usize,
    head_dim: usize,
    num_heads: usize,

    key_proj: Dense,
    query_proj: Dense,
    value_proj: Dense,

    attn_softmax: Softmax,
    dense: Dense,
    dropout: Dropout,
    norm: LayerNorm,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout_prob: f32, layer_norm_eps: f32) -> Self {
        MultiHeadAttention {
            embed_dim,
            head_dim: embed_dim / num_heads,
            num_heads,
            key_proj: Dense::new(embed_dim, embed_dim),
            query_proj: Dense::new(embed_dim, embed_dim),
            value_proj: Dense::new(embed_dim, embed_dim),
            attn_softmax: Softmax::new(3),
            dense: Dense::new(embed_dim, embed_dim),
            dropout: Dropout::new(dropout_prob),
            norm: LayerNorm::new(2, layer_norm_eps),
        }
    }

    pub fn pass(&self, x: &Fun, attn_mask: &Fun) -> Fun {
        // (N, L, E) -> (N, L, num_heads * head_dim)

        let query = self.query_proj.pass(x);
        let key = self.key_proj.pass(x);
        let value = self.value_proj.pass(x);

        // (N, L, num_heads * head_dim) -> (N, num_heads, L, head_dim)
        let query = self.separate_heads(query);
        let key = self.separate_heads(key);
        let value = self.separate_heads(value);

        // Calculate the attention scores
        // (N, num_heads, L, head_dim) * (N, num_head, head_dim, L) -> (N, num_head, L, L)
        let attention = query.matmul_batched(key.transpose(-1, -2)) / (self.head_dim as f32).sqrt();

        // Apply softmax to the attention scores
        let attention = self
            .attn_softmax
            .pass(&(attention + self.extend_mask(attn_mask)));

        // Applying attention weights
        // (N, num_heads, L, L) * (N, num_heads, L, head_dim) -> (N, num_heads, L, head_dim)
        let attention_value = attention.matmul_batched(&value);

        // (N, num_heads, L, head_dim) -> (N, L, num_heads * head_dim)
        let attention_value = self.merge_heads(attention_value);

        let y = self.dense.pass(&attention_value);
        self.norm.pass(&(y + x))
    }

    fn separate_heads(&self, features: Fun) -> Fun {
        // (N, L, num_heads * head_dim) -> (N, L, num_heads, head_dim)
        let batch_size = features.extent(0);
        let input_len = features.extent(1);

        let features = features.view([batch_size, input_len, self.num_heads, self.head_dim]);

        // (N, L, num_heads, head_dim) -> (N, num_heads, L, head_dim)
        features.transpose(2, 1)
    }

    fn merge_heads(&self, features: Fun) -> Fun {
        //# (N, num_heads, L, head_dim) -> (N, L, num_heads, head_dim)
        let features = features.transpose(2, 1);

        // # (N, L, num_heads, head_dim) -> (N, L, num_heads * head_dim)
        let batch_size = features.extent(0);
        let input_len = features.extent(1);

        features.view([batch_size, input_len, self.num_heads * self.head_dim])
    }

    fn extend_mask(&self, mask: &Fun) -> Fun {
        //# (N, L) -> (N, 1, 1, L)

        let batch_size = mask.extent(0);
        let input_len = mask.extent(1);

        let extended_mask = mask.view([batch_size, 1, 1, input_len]);

        //# Adding -1e5 makes masked locations zeroed out during softmax
        (-extended_mask + 1.0) * -1e5
    }
}

impl Parameter for MultiHeadAttention {
    fn init(&mut self, ctx: &mut Context, level: usize) {
        self.key_proj.init(ctx, level);
        self.query_proj.init(ctx, level);
        self.value_proj.init(ctx, level);
        self.dense.init(ctx, level);
        self.norm.init(ctx, level);
    }
    fn params<'a>(&'a self, p: &mut Vec<&'a Fun>) {
        self.key_proj.params(p);
        self.query_proj.params(p);
        self.value_proj.params(p);
        self.dense.params(p);
        self.norm.params(p);
    }
}
