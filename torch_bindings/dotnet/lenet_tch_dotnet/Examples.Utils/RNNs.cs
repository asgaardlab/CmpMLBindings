using System;
using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    class RNNModel : Module
    {
        public static int NUM_WORDS = 10000;
        public static int EMBEDDING_VEC_LEN = 300;
        public static int HIDDEN_SIZE = 512;

        private readonly Module layers;
        private Module embedding;
        private Modules.LSTM lstm = null;
        private Modules.GRU gru = null;
        private Module dropout;
        private Module dense;
        private Module sigmoid;
        private Device _device;
        private string _model_name;

        public RNNModel(string name, string model, Device device = null) : base(name)
        {
            _model_name = model;
            _device = device;
            embedding = Embedding(NUM_WORDS, EMBEDDING_VEC_LEN);
            if (model == "lstm")
                lstm = LSTM(EMBEDDING_VEC_LEN, HIDDEN_SIZE, batchFirst: true);
            else
                gru = GRU(EMBEDDING_VEC_LEN, HIDDEN_SIZE, batchFirst: true);
            // dropout = Dropout(0.5);
            // dense = Linear(HIDDEN_SIZE, 1);
            // sigmoid = Sigmoid();
            
            var modules = new List<(string, Module)>();
            modules.Add(("dropout", Dropout(0.5)));
            modules.Add(("dense", Linear(HIDDEN_SIZE, 1)));
            modules.Add(("sigmoid", Sigmoid()));
            layers = Sequential(modules);
            
            RegisterComponents();
            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            // Console.WriteLine($"input: {input}");
            var x_embed = embedding.forward(input);
            // Console.WriteLine($"x_embed: {x_embed}");
            var h0 = torch.zeros(1, input.shape[0], HIDDEN_SIZE, device: _device);
            Tensor x_rnn;
            if (_model_name == "lstm")
            {
                var c0 = torch.zeros(1, input.shape[0], HIDDEN_SIZE, device: _device);
                (x_rnn, _, _) = lstm.forward(x_embed, (h0, c0));
            }
            else
            {
                (x_rnn, _) = gru.forward(x_embed, h0);
            }
            // Console.WriteLine($"x_rnn: {x_rnn}");
            var x_rnn_last_seq = x_rnn[.., -1, ..];
            // Console.WriteLine($"x_rnn_last_seq: {x_rnn_last_seq}");
            // x_rnn_last_seq = dropout.forward(x_rnn_last_seq);
            // var logits = dense.forward(x_rnn_last_seq);
            // return sigmoid.forward(logits);
            return layers.forward(x_rnn_last_seq);
        }
    }
}