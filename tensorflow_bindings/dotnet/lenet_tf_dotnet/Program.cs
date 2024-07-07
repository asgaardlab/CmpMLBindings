using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using NumSharp;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Engine.DataAdapters;
using Tensorflow.Keras.Utils;
using System.Diagnostics;
using Tensorflow.Keras;
using Tensorflow.Keras.Layers;
using NDArray = Tensorflow.NumPy.NDArray;


namespace Tensorflow.Keras.ArgsDefinition
{
    public class ReLuArgs : LayerArgs
    {
    }
}

namespace Tensorflow.Keras.Layers
{
    public class Relu : Layer
    {
        private ReLuArgs args;

        public Relu(ReLuArgs args)
            : base(args)
        {
            this.args = args;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null) => 
            (Tensors) Binding.tf.nn.relu((Tensor) inputs, this.name);
    }
}


namespace Tensorflow.Keras.Losses
{
    public class BinaryCrossentropy : LossFunctionWrapper, ILossFunc
    {
        // public BinaryCrossentropy(bool from_logits = false, string reduction = null, string name = null)
        //     : base(reduction, name == null ? "binary_categorical_crossentropy" : name)
        // {
        // }

        public override Tensor Apply(Tensor target, Tensor output, bool from_logits = false, int axis = -1)
        {
//            Console.WriteLine($"output: {output}");
            var shape = tf.reduce_prod(tf.shape(output));
//            Console.WriteLine($"shape: {shape}");
            
            var count = tf.cast(shape, TF_DataType.TF_FLOAT);
//            Console.WriteLine($"count: {count}");
            
            output = tf.clip_by_value(output, 1e-6f, 1.0f - 1e-6f);
//            Console.WriteLine($"output: {output}");
            
            var z = target * tf.log(output) + (1 - target) * tf.log(1 - output);
//            Console.WriteLine($"z: {z}");
            
            var result = -1.0f / count * tf.reduce_sum(z);
//            Console.WriteLine($"result: {result}");
            
            return result;
        }
    }
}


namespace tf_dotnet
{
    class Program
    {
//        private static string SOURCE_PATH = Path.Join(Environment.CurrentDirectory, "../../../");
         private static string SOURCE_PATH = Environment.CurrentDirectory;
        private static string IMDb_DATA_PATH = Path.Join(SOURCE_PATH, "../../../data/imdb");
        private static string OUT_PATH = Path.Join(SOURCE_PATH, "../../../out/tensorflow/");
        private static string OUT_LR_ONLY_PATH = Path.Join(SOURCE_PATH, "../../../out/tensorflow_lr_only/");
        private static string SEEDS_PATH = Path.Join(SOURCE_PATH, "../../../random_seeds.txt");

        private static int[] VGG16 = { 64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0 };
        private static int[] TEXTCNN_FILTER_SIZES = { 2, 3, 4, 5 };
        // private static int[] TEXTCNN_FILTER_SIZES = { 2};
        private static int TEXTCNN_FILTER_NUM = 256;
        private static int NUM_WORDS = 10000;
        private static int EMBEDDING_VEC_LEN = 300;
        private static int IMDb_SETENCE_LEN = 300;
        
        internal static void SetRandomSeeds(int seed) {
            tf.set_random_seed(seed);
        }
        
        static double evaluate(Model model, NDArray x, NDArray y, int batch_size = 128)
        {
            Debug.Assert(len(x) == len(y));
            // Debug.Assert(len(x) % batch_size == 0);
            int batches = len(x) / batch_size;
            int correct = 0;

            for (int i = 0; i < batches; i++)
            {
                var start = i * batch_size;
                var end = start + batch_size;
                var x_batch = x[$"{start}:{end}"];
                var y_batch = y[$"{start}:{end}"];
                // Console.WriteLine($"{start}:{end} - {x_batch.shape}, {y_batch.shape}");
                var y_pred_tf = model.predict(x_batch, batch_size: batch_size)[0];
                // Console.WriteLine($"y_pred_tf - {y_pred_tf.shape}");
                NDArray y_pred;
                if (y_pred_tf.shape[1] == 1)
                {
                    y_pred = tf.round(y_pred_tf).numpy().astype(TF_DataType.DtFloatRef).reshape(-1);
                }
                else
                {
                    y_pred = tf.arg_max(y_pred_tf, 1).numpy().astype(TF_DataType.DtUint8Ref);
                }
                for (int j = 0; j < batch_size; j++)
                {
                    // Console.WriteLine($"{y_batch[j]} vs {y_pred[j]} = ...");
                    // Console.WriteLine($"...= {y_batch[j] == y_pred[j]}");
                    if (y_batch[j] == y_pred[j])
                    {
                        correct += 1;
                    }
                }
            }

            var last_batch = len(x) % batch_size;
            if (last_batch != 0)
            {
                var end = len(x);
                var start = len(x) - batch_size;
                var x_batch = x[$"{start}:{end}"];
                var y_batch = y[$"{start}:{end}"];
                var y_pred_tf = model.predict(x_batch, batch_size: batch_size)[0];
                NDArray y_pred;
                if (y_pred_tf.shape[1] == 1)
                {
                    y_pred = tf.round(y_pred_tf).numpy().astype(TF_DataType.DtFloatRef).reshape(-1);
                }
                else
                {
                    y_pred = tf.arg_max(y_pred_tf, 1).numpy().astype(TF_DataType.DtUint8Ref);
                }
                for (int j = batch_size - last_batch; j < batch_size; j++)
                {
                    // Console.WriteLine($"{y_batch[j]} vs {y_pred[j]} = {y_batch[j] == y_pred[j]}");
                    if (y_batch[j] == y_pred[j])
                    {
                        correct += 1;
                    }
                }
            }
            return (double) correct / (double) len(x);
        }

        static NDArray padImage(NDArray x)
        {
            NDArray padding = new int[,] {{0, 0}, {2, 2}, {2, 2}};
            var x_ts = tf.pad(x, padding);
            x_ts = tf.expand_dims(x_ts, axis: 3);
            x_ts = x_ts / 255.0f;
            NDArray x_np = x_ts.ToArray<float>();
            x_np = x_np.reshape((-1, 32, 32, 1));
            return x_np;
        }

        internal static Tuple<NDArray, NDArray, NDArray, NDArray> LoadDataset(string dataset)
        {
            NDArray x_train, y_train, x_test, y_test;
            switch (dataset)
            {
                case "mnist":
                    ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
                    x_train = padImage(x_train);
                    x_test = padImage(x_test);
                    break;
                case "cifar":
                    ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();
                    break;
                case "imdb":
                    var x_train_list = Helper.readTxt2D<Int32>(Path.Combine(IMDb_DATA_PATH, "x_train.txt"));
                    var x_train_flatten = x_train_list.SelectMany(a => a).ToArray();
                    x_train = new NDArray(x_train_flatten, (x_train_list.Count, x_train_list[0].Count));
                    var y_train_list = Helper.readTxt<float>(Path.Combine(IMDb_DATA_PATH, "y_train.txt"));
                    y_train = new NDArray(y_train_list.ToArray(), (y_train_list.Count, 1));
                    
                    var x_test_list = Helper.readTxt2D<Int32>(Path.Combine(IMDb_DATA_PATH, "x_test.txt"));
                    var x_test_flatten = x_test_list.SelectMany(a => a).ToArray();
                    x_test = new NDArray(x_test_flatten, (x_test_list.Count, x_test_list[0].Count));
                    var y_test_list = Helper.readTxt<float>(Path.Combine(IMDb_DATA_PATH, "y_test.txt"));
                    y_test = new NDArray(y_test_list.ToArray(), (y_test_list.Count, 1));
                    break;
                default:
                    throw new InvalidDataException("dataset must be one of ['mnist', 'cifar', 'imdb']");
            }
            return new Tuple<NDArray, NDArray, NDArray, NDArray>(x_train, y_train, x_test, y_test);
        }

        private static Tuple<List<double>, double> ProfEvaluation(Model model, NDArray x, NDArray y)
        {
            var accs = new List<double> { };
            var temp = new List<double> { };
            for (int i = 0; i < 5; i++)
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();
                var acc = evaluate(model, x, y);
                sw.Stop();
                temp.Add(sw.Elapsed.TotalSeconds);
                accs.Add(acc);
                sw.Reset();
            }
            return new Tuple<List<double>, double>(accs, temp.Average());
        }

        static Model createModel(string model_name)
        {
            Model  model;
            if (model_name == "lenet1")
            {
                model = createLeNet1Model((1, 32, 32));
            }
            else if (model_name == "lenet5")
            {
                model = createLeNet5Model((1, 32, 32));
            }
            else if (model_name == "vgg16")
            {
                model = createVGG16Model((3, 32, 32));
            }
            else if (model_name == "textcnn")
            {
                model = createTextCNNModel();
            }
            else
            {
                model = createLSTMModel();
            }
            return model;
        }

        static Functional createLeNet5Model(Tensorflow.Shape img_dim)
        {
            var layers = new Tensorflow.Keras.Layers.LayersApi();

            // input layer
            var inputs = keras.Input(shape: (img_dim.dims[1], img_dim.dims[2], img_dim.dims[0]), name: "img");

            // convolutional layer 1
            var c1 = layers.Conv2D(6, 5, activation: "tanh").Apply(inputs);
            int[] poolSize = {2, 2};
            int[] strides = {2, 2};
            var p1 = layers.max_pooling2d(c1, poolSize, strides);

            // convolutional layer 2
            var c2 = layers.Conv2D(16, 5, activation: "tanh").Apply(p1);
            var p2 = layers.max_pooling2d(c2, poolSize, strides);

            // convolutional layer 3
            var c3 = layers.Conv2D(120, 5, activation: "tanh").Apply(p2);

            // fully connected layer
            var flatten = layers.Flatten().Apply(c3);
            var f1 = layers.Dense(84, activation: "tanh").Apply(flatten);
            var logits = layers.Dense(10, activation: "softmax").Apply(f1);

            // build keras model
            var model = keras.Model(inputs, logits, name: "LeNet5");
            model.summary();

            // compile keras model in tensorflow static graph
             var opt = keras.optimizers.SGD(5e-2f);
//            var opt = keras.optimizers.RMSprop(1e-3f);
            model.compile(
                opt,
                keras.losses.SparseCategoricalCrossentropy(),
                new[] { "acc" });
            return model;
        }

        static Functional createLeNet1Model(Tensorflow.Shape img_dim)
        {
            var layers = new Tensorflow.Keras.Layers.LayersApi();

            // input layer
            var inputs = keras.Input(shape: (img_dim.dims[1], img_dim.dims[2], img_dim.dims[0]), name: "img");

            // convolutional layer 1
            var c1 = layers.Conv2D(4, 5, activation: "tanh").Apply(inputs);
            int[] poolSize = {2, 2};
            int[] strides = {2, 2};
            var p1 = layers.max_pooling2d(c1, poolSize, strides);

            // convolutional layer 2
            var c2 = layers.Conv2D(12, 5, activation: "tanh").Apply(p1);
            var p2 = layers.max_pooling2d(c2, poolSize, strides);

            // fully connected layer
            var flatten = layers.Flatten().Apply(p2);
            var logits = layers.Dense(10, activation: "softmax").Apply(flatten);

            // build keras model
            var model = keras.Model(inputs, logits, name: "LeNet1");
            model.summary();

            // compile keras model in tensorflow static graph
             var opt = keras.optimizers.SGD(5e-2f);
//            var opt = keras.optimizers.RMSprop(1e-3f);
            model.compile(
                opt,
                keras.losses.SparseCategoricalCrossentropy(),
                new[] { "acc" });
            return model;
        }

        static Model createVGG16Model(Tensorflow.Shape img_dim)
        {
            var layers = keras.layers;
            // var layers = new Tensorflow.Keras.Layers.LayersApi();
            var layers_list = new List<ILayer>();

            keras.Sequential();

            // input layer
            // var inputs = keras.Input(shape: (img_dim.dims[1], img_dim.dims[2], img_dim.dims[0]), name: "img");
            // var c1 = layers.Conv2D(6, 5, activation: "relu").Apply(inputs);
            // layers.LeakyReLU()

            // ref: https://github.com/SciSharp/SciSharp-Stack-Examples/blob/master/src/TensorFlowNET.Examples/ImageProcessing/ImageClassificationKeras.cs
            layers_list.add(layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[1], img_dim.dims[2], img_dim.dims[0])));
            for (var i = 0; i < VGG16.Length; i++)
            {
                if (VGG16[i] == 0)
                {
                    layers_list.add(layers.MaxPooling2D(2, 2));
                }
                else
                {
                    layers_list.add(layers.Conv2D(VGG16[i], 3, padding: "same", use_bias: false, kernel_regularizer: null));
                    layers_list.add(layers.BatchNormalization());
                    layers_list.add(new Relu(new ReLuArgs()));
                    // layers_list.add(layers.LeakyReLU(a));
                }
            }

            layers_list.add(layers.Flatten());
            layers_list.add(layers.Dense(4096, activation: keras.activations.Relu));
            layers_list.add(layers.Dropout((float) 0.5));
            layers_list.add(layers.Dense(4096, activation: keras.activations.Relu));
            layers_list.add(layers.Dropout((float) 0.5));
            layers_list.add(layers.Dense(10, activation: keras.activations.Softmax));

            Model model = keras.Sequential(layers_list);
            // var logits = seq.Apply(inputs);

            var opt = keras.optimizers.SGD(5e-2f);
            // var model = keras.Model(inputs, logits, name: "VGG16");
            model.compile(optimizer: opt,
                loss: keras.losses.SparseCategoricalCrossentropy(),
                metrics: new[] { "accuracy" });
            model.summary();

            return model;
        }

        // static Model createLSTMModel()
        // {
        //     var layers = keras.layers;
        //     var layers_list = new List<ILayer>();
        //
        //     keras.Sequential();
        //     layers_list.add(layers.Embedding(NUM_WORDS, EMBEDDING_VEC_LEN, input_length: IMDb_SETENCE_LEN));
        //     layers_list.add(layers.LSTM(512));
        //     layers_list.add(layers.Dropout((float) 0.5));
        //     layers_list.add(layers.Dense(1, activation: keras.activations.Sigmoid));
        //     Model model = keras.Sequential(layers_list);
        //
        //     var opt = keras.optimizers.Adam(8e-5f);
        //     var loss_cls = new Tensorflow.Keras.Losses.BinaryCrossentropy();
        //     model.compile(
        //         optimizer: opt,
        //         loss: loss_cls,
        //         // loss: keras.losses.SparseCategoricalCrossentropy(),
        //         metrics: new[] { "accuracy" }
        //         );
        //     model.summary();
        //     Console.WriteLine($"{layers_list}");
        //     Console.WriteLine($"{model}");
        //
        //     return model;
        // }

        static Model createLSTMModel()
        {
            var layers = new Tensorflow.Keras.Layers.LayersApi();

            // input layer
            var inputs = keras.Input(shape: (IMDb_SETENCE_LEN), name: "text", dtype: TF_DataType.DtInt32Ref);
            Console.WriteLine($"==============================================");
            Console.WriteLine($"inputs: {inputs}");
            var embed = layers.Embedding(NUM_WORDS, EMBEDDING_VEC_LEN, input_length: IMDb_SETENCE_LEN).Apply(inputs);
            Console.WriteLine($"embed: {embed}");
            var lstm = layers.LSTM(512).Apply(embed);
            Console.WriteLine($"lstm: {lstm}");
            // var lstm_dropout = layers.Dropout((float) 0.5).Apply(lstm);
            // Console.WriteLine($"{lstm_dropout}");
            var output = layers.Dense(1, activation: keras.activations.Sigmoid).Apply(lstm);

            // build keras model
            var model = keras.Model(inputs, output, name: "LSTM");

            var opt = keras.optimizers.Adam(8e-5f);
            var loss_cls = new Tensorflow.Keras.Losses.BinaryCrossentropy();
            model.compile(
                optimizer: opt,
                // loss: loss_cls,
                loss: keras.losses.SparseCategoricalCrossentropy(),
                metrics: new[] { "accuracy" }
                );
            model.summary();
            Console.WriteLine($"{model}");

            return model;
        }

        static Model createTextCNNModel()
        {
            var layers = new Tensorflow.Keras.Layers.LayersApi();

            // input layer
            var inputs = keras.Input(shape: (IMDb_SETENCE_LEN), name: "text", dtype: TF_DataType.DtInt32Ref);
            Console.WriteLine($"==============================================");
            Console.WriteLine($"inputs: {inputs}");
            var embed = layers.Embedding(NUM_WORDS, EMBEDDING_VEC_LEN, input_length: IMDb_SETENCE_LEN).Apply(inputs);
            Console.WriteLine($"embed: {embed}");
            var emb_reshaped = tf.expand_dims(embed, -1);
            Console.WriteLine($"emb_reshaped: {emb_reshaped}");

            var conv_list = new List<ILayer>();
            var pool_list = new List<ILayer>();

            var conv_out = new List<Tensor>();
            for (var i = 0; i < TEXTCNN_FILTER_SIZES.Length; i++)
            {
                var c = layers.Conv2D(
                    TEXTCNN_FILTER_NUM, (TEXTCNN_FILTER_SIZES[i], EMBEDDING_VEC_LEN),
                    activation: keras.activations.Relu
                );
                conv_list.add(c);
                var p = layers.MaxPooling2D(
                    (IMDb_SETENCE_LEN - TEXTCNN_FILTER_SIZES[i] + 1, 1),
                    (1, 1)
                );
                pool_list.add(p);
                var o1 = c.Apply(emb_reshaped);
                Console.WriteLine($"o1: {o1}");
                var o2 = p.Apply(o1);
                Console.WriteLine($"o2: {o2}");
                conv_out.Add(o2);
            }

            Console.WriteLine($"conv_out: {conv_out}");
            var concat_out = layers.Concatenate(3).Apply(conv_out);
            // var concat_out = tf.concat(conv_out, 3);
            Console.WriteLine($"concat_out: {concat_out}");

            var concat_out_dropout = layers.Dropout((float) 0.5).Apply(concat_out);
            Console.WriteLine($"concat_out_dropout: {concat_out_dropout}");
            var concat_out_flat = layers.Flatten().Apply(concat_out_dropout);
            Console.WriteLine($"concat_out_flat: {concat_out_flat}");
            // var output = layers.Dense(2).Apply(concat_out_flat);
            var output = layers.Dense(1, activation: keras.activations.Sigmoid).Apply(concat_out_flat);
            // var output = layers.Dense(1).Apply(concat_out_flat);
            Console.WriteLine($"output: {output}");

            // build keras model
            var model = keras.Model(inputs, output, name: "TextCNN");

            var opt = keras.optimizers.Adam(8e-5f);
            // var opt = keras.optimizers.SGD(5e-2f);
            // var opt = keras.optimizers.RMSprop(5e-2f);
            var loss_cls = new Tensorflow.Keras.Losses.BinaryCrossentropy();
            model.compile(
                optimizer: opt,
                loss: loss_cls,
                // loss: keras.losses.SparseCategoricalCrossentropy(),
                // loss: keras.losses.CategoricalCrossentropy(),
                metrics: new[] { "accuracy" }
                );
            model.summary();
            Console.WriteLine($"{model}");

            return model;
        }

        static Tuple<Model, List<double>, List<double>, double, double> run(string model_name, string database_name, int epoches)
        {
            var model = createModel(model_name);

            // prepare dataset
            var (x_train, y_train, x_test, y_test) = LoadDataset(database_name);
            var y_train_flat = y_train.reshape(-1);
            var y_test_flat = y_test.reshape(-1);
            // y_test = y_test_flat;
            // y_train = y_train_flat;
            Console.WriteLine($"{x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}");

            // training
            List<double> train_accs = new List<double>();
            List<double> test_accs = new List<double>();
            var eval_time_total = 0.0;
            Stopwatch sw = new Stopwatch();
            Stopwatch sw_eval = new Stopwatch();
            sw.Start();
            for (int epoch = 0; epoch < epoches; epoch++)
            {
                model.fit(
                    x_train, y_train,
                    batch_size: 128,
                    epochs: 1,
                    verbose: 0,
                    validation_split: 0.0f
                );
                sw_eval.Start();
                var train_acc = evaluate(model, x_train, y_train_flat);
                var test_acc = evaluate(model, x_test, y_test_flat);
                Console.WriteLine($"epoch {epoch} - Train acc {train_acc}, Test acc {test_acc}");
                train_accs.add(train_acc);
                test_accs.add(test_acc);
                sw_eval.Stop();
                eval_time_total += sw_eval.Elapsed.TotalSeconds;
                sw_eval.Reset();
                // model.evaluate(x_train_np, y_train, verbose: 1, return_dict: true);
            }
            sw.Stop();
            var time_elaspsed = sw.Elapsed.TotalSeconds;
            return new Tuple<Model, List<double>, List<double>, double, double>(model, train_accs, test_accs, time_elaspsed, eval_time_total);
        }


        internal static void DeployByModelStates(
            string device, string ld_path, string out_path, string model_name, string dataset, int model_num
            )
        {
            var (x_train, y_train, x_test, y_test) = LoadDataset(dataset);

            var test_acc_same = new List<bool>();
            var test_eval_times = new List<double>();
            var test_accs_res = new List<double>();
            var test_accs_all_stable = new List<bool>();
            var train_acc_same = new List<bool>();
            var train_accs_res = new List<double>();
            var train_eval_times = new List<double>();
            var train_accs_all_stable = new List<bool>();
            y_train = y_train.reshape(-1);
            y_test = y_test.reshape(-1);

            Console.WriteLine($"{x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}");
            for (int i = 0; i < model_num; i++)
            {
                Console.WriteLine($"Deploy Evaluation of {i}th...");
                var model = createModel(model_name);
                model.load_weights(Path.Join(ld_path, $"{dataset}-{model_name}_weights_{i}.h5"));

                var test_acc_gt = readTxt(Path.Join(ld_path, $"testing_errors_{i}.txt"));
                var train_acc_gt = readTxt(Path.Join(ld_path, $"training_errors_{i}.txt"));

                var test_acc0 = evaluate(model, x_test, y_test);
                var (test_accs, eval_test_time) = ProfEvaluation(model, x_test, y_test);
                var test_acc_stable = true;
                foreach (var acc in test_accs)
                {
                    if (!IsAccuracyEqual(test_acc0, acc, len(x_test)))
                    {
                        test_acc_stable = false;
                        break;
                    }
                }
                test_acc_same.Add(IsAccuracyEqual(test_acc0, test_acc_gt[^1], len(x_test)));
                test_eval_times.Add(eval_test_time);
                test_accs_res.Add(test_acc0);
                test_accs_all_stable.Add(test_acc_stable);

                var train_acc0 = evaluate(model, x_train, y_train);
                var (train_accs, eval_train_time) = ProfEvaluation(model, x_train, y_train);
                var train_acc_stable = true;
                foreach (var acc in train_accs)
                {
                    if (!IsAccuracyEqual(train_acc0, acc, len(x_train)))
                    {
                        train_acc_stable = false;
                        break;
                    }
                }
                train_acc_same.Add(IsAccuracyEqual(train_acc0, train_acc_gt[^1], len(x_train)));
                train_eval_times.Add(eval_train_time);
                train_accs_res.Add(train_acc0);
                train_accs_all_stable.Add(train_acc_stable);
            }

            StreamWriter f1 = new StreamWriter(Path.Join(out_path, $"deploy_eval_{device}_states.txt"));
            f1.WriteLine("Test Eval Average Time:");
            f1.WriteLine(test_eval_times.Average().ToString("G17"));
            f1.WriteLine("Test Eval Time:");
            foreach(double x in test_eval_times)
                f1.WriteLine(x.ToString("G17"));
            f1.WriteLine("Test Accuracy0 Same as Original:");
            for (int i = 0; i < test_acc_same.Count; i++)
                f1.WriteLine($"{test_acc_same[i].ToString()}, {test_accs_res[i].ToString("G17")}");
            f1.WriteLine("Test Accuracy Stable:");
            for (int i = 0; i < test_accs_all_stable.Count; i++)
                f1.WriteLine($"{test_accs_all_stable[i].ToString()}");

            f1.WriteLine("Train Eval Average Time:");
            f1.WriteLine(train_eval_times.Average().ToString("G17"));
            f1.WriteLine("Train Eval Time:");
            foreach(double x in train_eval_times)
                f1.WriteLine(x.ToString("G17"));
            f1.WriteLine("Train Accuracy0 Same as Original:");
            for (int i = 0; i < train_acc_same.Count; i++)
                f1.WriteLine($"{train_acc_same[i].ToString()}, {train_accs_res[i].ToString("G17")}");
            f1.WriteLine("Train Accuracy Stable:");
            for (int i = 0; i < train_accs_all_stable.Count; i++)
                f1.WriteLine($"{train_accs_all_stable[i].ToString()}");
            f1.Close();
        }

        static List<double> readTxt(string path)
        {
            StreamReader dataStream = new StreamReader(path);
            string datasample;
            var data = new List<double> {};
            while ((datasample = dataStream.ReadLine()) != null)
            {
                data.Add(double.Parse(datasample));
            }

            return data;
        }

        static bool IsAccuracyEqual(double acc1, double acc2, int totalNum)
        {
            return (int) Math.Round(acc1 * totalNum) == (int) Math.Round(acc2 * totalNum);
        }

        static void test_embeddning()
        {

            var inputs = keras.Input(shape: (5), name: "text", dtype: TF_DataType.DtInt32Ref);
            // var inputs = keras.Input(shape: (5, 64), name: "text", dtype: TF_DataType.DtFloatRef);
            Console.WriteLine($"inputs: {inputs}");
            var embed = keras.layers.Embedding(1000, 64, input_length: 5).Apply(inputs);
            Console.WriteLine($"embed: {embed}");
            var embed_flat = keras.layers.Flatten().Apply(embed);
            // var embed_flat = keras.layers.Flatten().Apply(inputs);
            Console.WriteLine($"embed_flat: {embed_flat}");
            var output = keras.layers.Dense(10).Apply(embed_flat);
            Console.WriteLine($"output: {output}");

            // build keras model
            var model = keras.Model(inputs, output, name: "test");

            var input_array = np.random.randint(1000, size: (100, 5)).ravel().ToArray<int>();
            var x_train = new NDArray(input_array, (100, 5));
            // var input_array = np.random.randint(1000, size: (100, 5, 64)).astype(np.float32).ravel().ToArray<float>();
            // var x_train = new NDArray(input_array, (100, 5, 64));
            var labels = np.random.randint(10, size: (100, 1)).ravel().ToArray<int>();
            var y_train = new NDArray(labels, (100));

            var opt = keras.optimizers.SGD(5e-2f);
            model.compile(
                optimizer: opt,
                loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                metrics: new[] { "accuracy" }
            );
            model.summary();
            Console.WriteLine($"{model}");
            model.fit(
                x_train, 
                y_train, 
                10,
                10,
                1,
                0.0f
                );
        }

        static void test_bce()
        {
            var inputs = keras.Input(shape: (5, 64), name: "text", dtype: TF_DataType.DtFloatRef);
            Console.WriteLine($"inputs: {inputs}");
            var embed_flat = keras.layers.Flatten().Apply(inputs);
            Console.WriteLine($"embed_flat: {embed_flat}");
            var output = keras.layers.Dense(1, activation: keras.activations.Sigmoid).Apply(embed_flat);
            Console.WriteLine($"output: {output}");

            // build keras model
            var model = keras.Model(inputs, output, name: "test");
            
            var input_array = np.random.randint(1000, size: (100, 5, 64)).astype(np.float32).ravel().ToArray<float>();
            var x_train = new NDArray(input_array, (100, 5, 64));
            var labels = np.random.randint(1, size: (100, 1)).ravel().ToArray<int>();
            var y_train = new NDArray(labels, (100));
            
            var opt = keras.optimizers.SGD(5e-2f);
            var loss_cls = new Tensorflow.Keras.Losses.BinaryCrossentropy();
            model.compile(
                optimizer: opt,
                loss: loss_cls,
                metrics: new[] { "accuracy" }
            );
            model.summary();
            Console.WriteLine($"{model}");
            model.fit(
                x_train, 
                y_train, 
                10,
                10,
                1,
                0.0f
                );
        }
        
        static void Main(string[] args)
        {
            // test_embeddning();
            // test_bce();
            Debug.Assert(args.Length >= 4, "require five parameters: mode device model epochs run_num [gpu_device_index]");
            
            var mode = args[0];
            Debug.Assert(mode == "train" || mode == "deploy", "mode must be one of ['train', 'deploy']");
            var device_name = args[1];
            Debug.Assert(device_name == "cpu" || device_name == "gpu", "model must be one of ['cpu', 'gpu']");
            var model_name = args[2];
            Debug.Assert(
                model_name == "lenet1" || model_name == "lenet5" || model_name == "resnet20" || model_name == "textcnn" || model_name == "vgg16" || model_name == "lstm",
                "model must be one of ['lenet1', 'lenet5', 'resnet20', 'textcnn', 'vgg16', 'lstm']"
                );
            string dataset;
            if (model_name == "lenet1" || model_name == "lenet5")
            {
                dataset = "mnist";
            }
            else if (model_name == "vgg16" || model_name == "resnet20")
            {
                dataset = "cifar";
            }
            else
            {
                dataset = "imdb";
            }
            var epochs = Int32.Parse(args[3]);
            var run_num = Int32.Parse(args[4]);
            
            var gpu_device_index = args.Length > 5 ? Int32.Parse(args[5]) : 0;
            if (device_name == "gpu")
            {
                Debug.Assert(tf.config.list_physical_devices().Length > 1, "CUDA must be available");
                var device_obj = tf.config.list_physical_devices()[gpu_device_index];
                tf.config.experimental.set_memory_growth(device_obj, true);
            }
            
            var sgd_lr_only = model_name == "lenet1" || model_name == "lenet5" || model_name == "vgg16" || model_name == "resnet20";
            Console.WriteLine($"mode: {mode}, device: {device_name}:{gpu_device_index}, epochs: {epochs}, run_num: {run_num}, sgd_lr_only: {sgd_lr_only}");
            
            var out_path = sgd_lr_only ? Path.Join(OUT_LR_ONLY_PATH, $"{model_name}/dotnet") : Path.Join(OUT_PATH, $"{model_name}/dotnet");
            if (!Directory.Exists(out_path))
                Directory.CreateDirectory(out_path);
            switch (mode) {
                // Training the model
                case "train":
                    var SEEDS = readTxtInt(SEEDS_PATH);
                    for (int i = 0; i < run_num; i++)
                    {
                        var seed = SEEDS[i];
                        SetRandomSeeds(seed);
                        Console.WriteLine($"seed: {seed}");
                        var (model, training_errors, testing_errors, total_time, eval_time) =
                            run(model_name, dataset, epochs);
                        System.IO.File.WriteAllLines(Path.Join(out_path, $"training_errors_{i}.txt"), training_errors.Select(o => o.ToString("G17")));
                        System.IO.File.WriteAllLines(Path.Join(out_path, $"testing_errors_{i}.txt"), testing_errors.Select(o => o.ToString("G17")));
                        StreamWriter f1 = new StreamWriter(Path.Join(out_path, $"time_cost_{device_name}_{i}.txt"));
                        f1.WriteLine(total_time.ToString("G17"));
                        f1.WriteLine(eval_time.ToString("G17"));
                        f1.WriteLine((total_time - eval_time).ToString("G17"));
                        f1.WriteLine(seed.ToString());
                        model.save(Path.Join(out_path, $"{dataset}-{model_name}_{i}"));
                        model.save_weights(Path.Join(out_path, $"{dataset}-{model_name}_weights_{i}"));
                        model.save_weights(Path.Join(out_path, $"{dataset}-{model_name}_weights_{i}.h5"));
                        f1.Close();
                    }
                    break;
                case "deploy":
                    var py_out_path = sgd_lr_only ? Path.Join(OUT_LR_ONLY_PATH, $"{model_name}/py/") : Path.Join(OUT_PATH, $"{model_name}/py/");
                    DeployByModelStates(device_name, py_out_path, out_path, model_name, dataset, run_num);
                    break;
                default:
                    throw new InvalidArgumentError("mode must be 'train', 'prof', or 'deploy'");
            }
        }
        
        static List<int> readTxtInt(string path)
        {
            StreamReader dataStream = new StreamReader(path);   
            string datasample;
            var data = new List<int> {};
            while ((datasample = dataStream.ReadLine()) != null)
            {
                data.Add(int.Parse(datasample));
            }

            return data;
        }
    }
}
