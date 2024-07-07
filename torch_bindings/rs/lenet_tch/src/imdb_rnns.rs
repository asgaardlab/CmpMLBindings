use tch::{nn, nn::ModuleT, Tensor, IndexOp};
use tch::nn::{RNN};


const NUM_WORDS: i64 = 10000;
pub const EMBEDDING_VEC_LEN: i64 = 300;
const HIDDEN_SIZE: i64 = 512;


#[derive(Debug)]
enum RNNModelType {
    LSTM(nn::LSTM),
    GRU(nn::GRU),
}


#[derive(Debug)]
pub struct RNNModel {
    embedding: nn::Embedding,
    rnn_model: RNNModelType,
    linear: nn::Linear,
}


impl RNNModel {
    pub fn new(vs: &nn::Path, model: &str) -> RNNModel {
        let embed_vs = vs / "embedding";
        let rnn_model_vs = vs / "rnn_model";
        let linear_vs = vs / "dense";
        let embedding = nn::embedding(embed_vs, NUM_WORDS, EMBEDDING_VEC_LEN, Default::default());
        let rnn_model = match model {
            "lstm" => RNNModelType::LSTM(nn::lstm(&rnn_model_vs, EMBEDDING_VEC_LEN, HIDDEN_SIZE, Default::default())),
            "gru" => RNNModelType::GRU(nn::gru(&rnn_model_vs, EMBEDDING_VEC_LEN, HIDDEN_SIZE, Default::default())),
            _ => panic!("model must be one of ['lstm', 'gru']")
        };
        let linear = nn::linear(linear_vs, HIDDEN_SIZE, 1, Default::default());

        RNNModel {
            embedding,
            rnn_model,
            linear,
        }
    }

    #[allow(dead_code)]
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_t(xs, false)
    }
}

impl nn::ModuleT for RNNModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x_embed = self.embedding.forward_t(xs, train);
        let x_rnn = match self.rnn_model {
            RNNModelType::GRU(ref m) => m.seq(&x_embed).0,
            RNNModelType::LSTM(ref m) => m.seq(&x_embed).0,
        };
        let x_rnn_last_seq = x_rnn.i((.., -1, ..)).dropout(0.5, train);
        let out = self.linear.forward_t(&x_rnn_last_seq, train).sigmoid();
        out
    }
}
