//using System;
//using System.Collections.Generic;
//using System.Linq;
//using static TorchSharp.torch;
//using static TorchSharp.torch.nn;
//
//namespace TorchSharp.Examples
//{
//    class TextCNNModel : Module
//    {
//        private static long[] FILTER_SIZES = { 2, 3, 4, 5 };
//        public static long NUM_FILTERS = 256;
//        public static long NUM_WORDS = 10000;
//        public static long EMBEDDING_VEC_LEN = 300;
//
//        private readonly Module layers;
//        private Module embedding;
//
//        private Module[] convs;
//        private Module[] pools;
//
//        private Module dropout;
//        private Module dense;
//        private Module sigmoid;
//        private Device _device;
//        private string _model_name;
//
//        public TextCNNModel(string name, string model, Device device = null) : base(name)
//        {
//            _model_name = model;
//            _device = device;
//            embedding = Embedding(NUM_WORDS, EMBEDDING_VEC_LEN);
//
//            for (var i = 0; i < FILTER_SIZES.Length; i++) {
//                convs.Append(Conv2d(1, NUM_FILTERS, (FILTER_SIZES[i], EMBEDDING_VEC_LEN)));
//
//                if (FILTER_SIZES[i] == 0) {
//                    modules.Add(($"maxpool2d-{i}a", MaxPool2d(kernelSize: 2, stride: 2)));
//                } else {
//                    modules.Add(($"conv2d-{i}a", );
//                    modules.Add(($"bnrm2d-{i}a", BatchNorm2d(channels[i])));
//                    modules.Add(($"relu-{i}b", ReLU()));
//                    in_channels = channels[i];
//                }
//            }
//
//            dropout = Dropout(0.5);
//            dense = Linear(HIDDEN_SIZE, 1);
//            sigmoid = Sigmoid();
//
//            RegisterComponents();
//            if (device != null && device.type == DeviceType.CUDA)
//                this.to(device);
//        }
//
//        public override Tensor forward(Tensor input)
//        {
//            // Console.WriteLine($"input: {input}");
//            var x_embed = embedding.forward(input);
//            // Console.WriteLine($"x_embed: {x_embed}");
//            var h0 = torch.zeros(1, input.shape[0], HIDDEN_SIZE, device: _device);
//            Tensor x_rnn;
//            if (_model_name == "lstm")
//            {
//                var c0 = torch.zeros(1, input.shape[0], HIDDEN_SIZE, device: _device);
//                (x_rnn, _, _) = lstm.forward(x_embed, (h0, c0));
//            }
//            else
//            {
//                (x_rnn, _) = gru.forward(x_embed, h0);
//            }
//            // Console.WriteLine($"x_rnn: {x_rnn}");
//            var x_rnn_last_seq = x_rnn[.., -1, ..];
//            // Console.WriteLine($"x_rnn_last_seq: {x_rnn_last_seq}");
//            x_rnn_last_seq = dropout.forward(x_rnn_last_seq);
//            var logits = dense.forward(x_rnn_last_seq);
//            return sigmoid.forward(logits);
//        }
//    }
//}