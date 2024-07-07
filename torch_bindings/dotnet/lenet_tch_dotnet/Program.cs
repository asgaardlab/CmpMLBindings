﻿// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using TorchSharp;
using TorchText;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Test;


namespace TorchSharp.Examples
{
    public class Run
    {
        private static string SOURCE_PATH = Environment.CurrentDirectory;
        private static string MNIST_DATA_PATH = Path.Join(SOURCE_PATH, "../../../data/MNIST/raw");
        private static string CIFAR10_DATA_PATH = Path.Join(SOURCE_PATH, "../../../data/cifar-10-batches-bin");
        private static string IMDb_DATA_PATH = Path.Join(SOURCE_PATH, "../../../data/imdb");
        private static string OUT_PATH = Path.Join(SOURCE_PATH, "../../../out/pytorch/");
        private static string OUT_LR_ONLY_PATH = Path.Join(SOURCE_PATH, "../../../out/pytorch_lr_only/");
        private static string SEEDS_PATH = Path.Join(SOURCE_PATH, "../../../random_seeds.txt");
        // private static int _trainBatchSize = 128;
        // private static int _testBatchSize = 128;

        class Reproduce : Module
        {
            private readonly Module layers;
            public Reproduce(string name) : base(name)
            {
                var modules = new List<(string, Module)>();
                modules.Add(("fc_linear_1", Linear(100, 7)));
                modules.Add(("fc_dropout_1", Dropout(0.5)));
                layers = Sequential(modules);
                RegisterComponents();
            }

            public override void Eval()
            // public override void eval()
            {
                base.Eval();
                layers.Eval();
                // base.eval();
                // layers.eval();
            }

            public override Tensor forward(Tensor input)
            {
                return layers.forward(input);
            }
        }

        class Reproduce2 : Module
        {
            private readonly Module layers;
            private Module dropout;
            private Module dense;
            public Reproduce2(string name) : base(name)
            {
                dense = Linear(100, 7);
                dropout = Dropout(0.5);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                return dropout.forward(dense.forward(input));
            }
        }

        internal static void SetRandomSeeds(Int64 seed) {
            torch.random.manual_seed(seed);
        }

        internal static void IssueReproduce()
        {
            torch.random.manual_seed(0);
            var x = torch.arange(0, 100, 1, torch.float32, torch.CPU).reshape(1, 100);

            var model = new Reproduce("reproduce");
            model.Eval();
//             model.eval();
            Console.WriteLine("Reproducing 1...");
            Tensor y;
            for (var run = 0; run < 3; run++)
            {
                Console.WriteLine($"{run}th running...");
                y = model.forward(x);
                for (var i = 0; i < 7; i++)
                {
                    Console.Write($"{y.ReadCpuSingle(i)} ");
                }
                Console.WriteLine("");
            }

            Console.WriteLine("Reproducing 2...");
            var model2 = new Reproduce2("reproduce");
            model2.Eval();
//             model2.eval();
            for (var run = 0; run < 3; run++)
            {
                Console.WriteLine($"{run}th running...");
                y = model2.forward(x);
                for (var i = 0; i < 7; i++)
                {
                    Console.Write($"{y.ReadCpuSingle(i)} ");
                }
                Console.WriteLine("");
            }
        }

        internal static void Main(string[] args)
        {
            // var lstm = new LSTMModel("lstm", torch.CPU);
            // lstm.save("./lstm.dat");
            // throw new Exception("exit");

//            var lenet1 = new LeNet1Model("lenet1", torch.CPU);
//            lenet1.save("./lenet1.dat");
//            throw new Exception("exit");

            Debug.Assert(args.Length >= 4, "require five parameters: mode device model epochs run_num [gpu_device_index]");
            var mode = args[0];
            Debug.Assert(mode == "train" || mode == "deploy" || mode == "model", "mode must be one of ['train', 'deploy', 'model']");
            var device_name = args[1];
            Debug.Assert(device_name == "cpu" || device_name == "gpu", "model must be one of ['cpu', 'gpu']");
            var model_name = args[2];
            Debug.Assert(
                model_name == "lenet5" || model_name == "lenet1" || model_name == "vgg16" || model_name == "resnet20" || model_name == "textcnn" || model_name == "lstm" || model_name == "gru",
                "model must be one of ['lenet1', 'lenet5', 'resnet20', 'vgg16', 'lstm', 'gru']"
                );
            string dataset;
            if (model_name == "lenet5" || model_name == "lenet1")
            {
                dataset = "mnist";
            }
            else if (model_name == "vgg16" || model_name == "resnet20")
            {
                dataset = "cifar";
            }
            else if (model_name == "lstm" || model_name == "gru" || model_name == "textcnn")
            {
                dataset = "imdb";
            }
            else
            {
                throw new InvalidDataException("model_name must be one of ['lenet1', 'lenet5', 'resnet20', 'vgg16', 'textcnn', 'lstm', 'gru']");
            }

            var epochs = Int32.Parse(args[3]);
            var run_num = Int32.Parse(args[4]);
            var gpu_device_index = args.Length > 5 ? Int32.Parse(args[5]) : -1;
            Debug.Assert(torch.cuda.is_available(), "CUDA must be available");
            var sgd_lr_only = (args.Length > 6 && args[6] == "sgd_lr_only") ? true : false;

            Device device;
            if (device_name == "cpu")
            {
                device = torch.CPU;
            }
            else
            {
                device = new torch.Device(DeviceType.CUDA, gpu_device_index);
            }
            Console.WriteLine($"{mode}ing {dataset} x {model_name} on {device.type.ToString()}:{device.index}, epochs: {epochs}, run_num: {run_num}, sgd_lr_only? {sgd_lr_only}");
            if (mode == "train")
            {
                Train(device, model_name, dataset, epochs, run_num, sgd_lr_only);
            }
            else if (mode == "deploy")
            {
                DeployByModelStates(device, model_name, dataset, run_num, sgd_lr_only);
            }
            else
            {
                var out_path = sgd_lr_only ? Path.Join(OUT_LR_ONLY_PATH, $"{model_name}/dotnet") : Path.Join(OUT_PATH, $"{model_name}/dotnet");
                var (model, _, _) = CreateModel(model_name, 10, device, sgd_lr_only);
                foreach( KeyValuePair<string, Tensor> kvPair in model.state_dict() )
                    Console.WriteLine($"{kvPair.Key}, {kvPair.Value}");
                model.save(Path.Join(out_path, $"{model_name}_weights.dat"));
                Console.WriteLine((long) model.state_dict().Count);
            }
        }


        internal static Tuple<Module, optim.Optimizer, Loss> CreateModel(string model_name, int numClasses, Device device, bool sgd_lr_only)
        {
            Module model;
            optim.Optimizer opt;
            Loss loss_func;

            if (model_name == "lenet5")
            {
                model = new LeNet5(model_name, device);
                if (sgd_lr_only)
                    opt = torch.optim.SGD(model.parameters(), 5e-2);
                else
                    opt = torch.optim.SGD(model.parameters(), 5e-2, 9e-1);
                loss_func = cross_entropy_loss();
            }
            else if (model_name == "lenet1")
            {
                model = new LeNet1(model_name, device);
                if (sgd_lr_only)
                    opt = torch.optim.SGD(model.parameters(), 5e-2);
                else
                    opt = torch.optim.SGD(model.parameters(), 5e-2, 9e-1);
                loss_func = cross_entropy_loss();
            }
            else if (model_name == "vgg16")
            {
                 model = new VGG(model_name, numClasses, device);
                // model = new VGG_fixed(model_name, numClasses, device);
                if (sgd_lr_only)
                    opt = torch.optim.SGD(model.parameters(), 5e-2);
                else
                    opt = torch.optim.SGD(model.parameters(), 1e-1, 9e-1, weight_decay: 1e-4);
                loss_func = cross_entropy_loss();
            }
            else if (model_name == "lstm")
            {
                model = new RNNModel(model_name, model_name, device);
                opt = torch.optim.Adam(model.parameters(), 8e-5);
                loss_func = binary_cross_entropy_loss();
            }
            else if (model_name == "gru")
            {
                model = new RNNModel(model_name, model_name, device);
                opt = torch.optim.Adam(model.parameters(), 3e-4);
                loss_func = binary_cross_entropy_loss();
            }
            else
            {
                throw new InvalidDataException("model_name must be one of ['lenet1', 'lenet5', 'resnet20', 'vgg16', 'lstm', 'gru']");
            }

            return new Tuple<Module, optim.Optimizer, Loss>(model, opt, loss_func);
        }

        internal static Tuple<Reader, Reader> LoadDataset(string dataset, Device device, int batch_size)
        {
            Reader train, test;
            switch (dataset)
            {
                case "mnist":
                    train = new MNISTReader(MNIST_DATA_PATH, "train", batch_size, device: device, shuffle: false);
                    test = new MNISTReader(MNIST_DATA_PATH, "t10k", batch_size, device: device, shuffle: false);
                    break;
                case "cifar":
                    train = new CIFARReader(CIFAR10_DATA_PATH, false, batch_size, device: device, shuffle: false);
                    test = new CIFARReader(CIFAR10_DATA_PATH, true, batch_size, device: device, shuffle: false);
                    break;
                case "imdb":
                    train = new IMDbReader(IMDb_DATA_PATH, false, batch_size, device);
                    test = new IMDbReader(IMDb_DATA_PATH, true, batch_size, device);
                    break;
                default:
                    throw new InvalidDataException("dataset must be one of ['mnist', 'cifar']");
            }
            return new Tuple<Reader, Reader>(train, test);
        }

        internal static int GetBatchSize(string dataset)
        {
            if (dataset == "imdb")
                return 256;
            return 128;
        }
        
        internal static void Train(Device device, string model_name, string dataset, int epochs, int run_num, bool sgd_lr_only)
        {
            var SEEDS = readTxtI64(SEEDS_PATH);
            for (var run = 0; run < run_num; run++)
            {
                var seed = SEEDS[run];
                SetRandomSeeds(seed);
                Console.WriteLine($"seed: {seed}");
                var out_path = sgd_lr_only ? Path.Join(OUT_LR_ONLY_PATH, $"{model_name}/dotnet") : Path.Join(OUT_PATH, $"{model_name}/dotnet");
                if (!Directory.Exists(out_path))
                    Directory.CreateDirectory(out_path);

                var (model, opt, loss_func) = CreateModel(model_name, 10, device, sgd_lr_only);
                var (train, test) = LoadDataset(dataset, device, GetBatchSize(dataset));
                Console.WriteLine($"train_x shape: {train.data.Count} x {train.data[0]}");
                Console.WriteLine($"train_y shape: {train.labels.Count} x {train.labels[0]}");
                
                var (train_accs, test_accs, total_time, eval_time) = 
                    Fit(model, opt, loss_func, train, test, epochs);
                
                StreamWriter f1 = new StreamWriter(Path.Join(out_path, $"testing_errors_{run}.txt"));
                foreach(double x in test_accs)
                    f1.WriteLine(x.ToString("G17"));
                f1.Close();
                StreamWriter f2 = new StreamWriter(Path.Join(out_path, $"training_errors_{run}.txt"));
                foreach(double x in train_accs)
                    f2.WriteLine(x.ToString("G17"));
                f2.Close();
                
                StreamWriter f3 = new StreamWriter(Path.Join(out_path, $"time_cost_{device.type.ToString()}_{run}.txt"));
                f3.WriteLine(total_time.ToString("G17"));
                f3.WriteLine(eval_time.ToString("G17"));
                f3.WriteLine((total_time - eval_time).ToString("G17"));
                f3.WriteLine(seed.ToString());
                f3.Close();
                
                Console.WriteLine("Saving model...");
                model.save(Path.Join(out_path, dataset + $"_model_{run}.bin"));
            }
        }
        
        internal static void DeployByModelStates(Device device, string model_name, string dataset, int model_num, bool sgd_lr_only)
        {
            var test_acc_same = new List<bool>();
            var test_eval_times = new List<double>();
            var test_accs_res = new List<double>();
            var test_accs_all_stable = new List<bool>();
            var train_acc_same = new List<bool>();
            var train_accs_res = new List<double>();
            var train_eval_times = new List<double>();
            var train_accs_all_stable = new List<bool>();
            var (train, test) = LoadDataset(dataset, device, GetBatchSize(dataset));
            
            var out_path = sgd_lr_only ? Path.Join(OUT_LR_ONLY_PATH, $"{model_name}/dotnet") : Path.Join(OUT_PATH, $"{model_name}/dotnet");
            var py_out_path = sgd_lr_only ? Path.Join(OUT_LR_ONLY_PATH, $"{model_name}/py") : Path.Join(OUT_PATH, $"{model_name}/py");
            
            for (var i = 0; i < model_num; i++)
            {
                Console.WriteLine($"Deploy Evaluation of {i}th...");
//                SetRandomSeeds();
                var (model, opt, loss_func) = CreateModel(model_name, 10, device, sgd_lr_only);
                model.to(torch.CPU);
                // model.save("model_weights.dat");
                // Console.WriteLine((long) model.state_dict().Count);
                // foreach(KeyValuePair<string, Tensor> entry in model.state_dict())
                //     Console.WriteLine($"{entry.Key}, {entry.Value}");

                var model_path = Path.Join(py_out_path, $"{dataset}-{model_name}_{i}_dotnet.dat"); 
                model.load(model_path);
                model.to(device);
                Console.WriteLine($"loaded {model_path}");

                var test_acc_gt = readTxt(Path.Join(py_out_path, $"testing_errors_{i}.txt"));
                var train_acc_gt = readTxt(Path.Join(py_out_path, $"training_errors_{i}.txt"));
                // warm up
                var test_acc0 = Evaluation(model, test.Data(), test.Size);
                var (test_accs, eval_test_time) = ProfEvaluation(model, test.Data(), test.Size);
                var test_acc_stable = true;
                foreach (var acc in test_accs)
                {
                    if (!IsAccuracyEqual(test_acc0, acc, test.Size))
                        test_acc_stable = false;
                    // Debug.Assert(IsAccuracyEqual(test_acc0, acc, test.Size), $"{test_acc0} != {acc}");
                }
                test_acc_same.Add(IsAccuracyEqual(test_acc0, test_acc_gt[^1], test.Size));
                test_eval_times.Add(eval_test_time);
                test_accs_res.Add(test_acc0);
                test_accs_all_stable.Add(test_acc_stable);
                
                // warm up
                var train_acc0 = Evaluation(model, train.Data(), train.Size);
                var (train_accs, eval_train_time) = ProfEvaluation(model, train.Data(), train.Size);
                var train_acc_stable = true;
                foreach (var acc in train_accs)
                {
                    if (!IsAccuracyEqual(train_acc0, acc, train.Size))
                        train_acc_stable = false;
                    // Debug.Assert(IsAccuracyEqual(train_acc0, acc, train.Size), $"{train_acc0} != {acc}");
                }
                train_acc_same.Add(IsAccuracyEqual(train_acc0, train_acc_gt[^1], train.Size));
                train_eval_times.Add(eval_train_time);
                train_accs_res.Add(train_acc0);
                model.Dispose();
                train_accs_all_stable.Add(train_acc_stable);
            }
            StreamWriter f1 = new StreamWriter(Path.Join(out_path, $"deploy_eval_{device.type.ToString()}_states.txt"));
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
            // foreach(bool x in test_acc_same)
            //     f1.WriteLine(x.ToString());
            
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
            // foreach(bool x in train_acc_same)
            //     f1.WriteLine(x.ToString());
            f1.Close();
        }
        
        internal static Tuple<List<double>, List<double>, double, double>
            Fit(Module model, optim.Optimizer optimizer, Loss loss_func, Reader train, Reader test, int epochs)
        {
            // var loss_func = cross_entropy_loss();
            // var loss_func = binary_cross_entropy_loss();
            List<Double> test_accs = new List<Double>();
            List<Double> train_accs = new List<Double>();
            
            var eval_time_total = 0.0;
            Stopwatch sw = new Stopwatch();
            Stopwatch sw_eval = new Stopwatch();
            sw.Start();
            for (var epoch = 1; epoch <= epochs; epoch++) {
                model.Train();
//                 model.train();
                foreach (var (data, target) in train.Data()) {
                    optimizer.zero_grad();
                    var prediction = model.forward(data);
                    var output = loss_func(prediction, target);
                    output.backward();
                    optimizer.step();
                    GC.Collect();
                }
                
                sw_eval.Start();
                var test_accuracy = Evaluation(model, test.Data(), test.Size);
                var train_accuracy = Evaluation(model, train.Data(), train.Size);
                Console.WriteLine($"epoch {epoch} - Train acc {(train_accuracy):P2}, Test acc {(test_accuracy):P2}");
                Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");
                test_accs.Add(test_accuracy);
                train_accs.Add(train_accuracy);
                sw_eval.Stop();
                eval_time_total += sw_eval.Elapsed.TotalSeconds;
                sw_eval.Reset();
            }
            sw.Stop();
            var time_elaspsed = sw.Elapsed.TotalSeconds;
            Console.WriteLine($"Elapsed time: {time_elaspsed:F1} s.");
            
            return new Tuple<List<double>, List<double>, double, double>(
                train_accs, test_accs, time_elaspsed, eval_time_total 
                );
        }

        private static double Evaluation(
            Module model,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            long size)
        {
            model.Eval();
//             model.eval();
            int correct = 0;
            foreach (var (data, target) in dataLoader) {
                var prediction = model.forward(data);
                Tensor pred = prediction.shape[1] == 1 ? prediction.round() : prediction.argmax(1);
                correct += pred.eq(target).sum().ToInt32();
                pred.Dispose();
                GC.Collect();
            }
            // Console.WriteLine($"Size: {size}, Total: {size}");
            var accuracy = (double) correct / size;
            // Console.WriteLine($"\rEvaluation: Average loss {(testLoss / size):F4} | Accuracy {(accuracy):P2}");
            return accuracy;
        }

        private static Tuple<List<double>, double> ProfEvaluation(Module model,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            long size)
        {
            var accs = new List<double> { };
            var temp = new List<double> { };
            for (int i = 0; i < 5; i++)
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();
                var acc = Evaluation(model, dataLoader, size);
                sw.Stop();
                temp.Add(sw.Elapsed.TotalSeconds);
                accs.Add(acc);
                sw.Reset();
            }
            return new Tuple<List<double>, double>(accs, temp.Average());
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
        
        static List<Int64> readTxtI64(string path)
        {
            StreamReader dataStream = new StreamReader(path);   
            string datasample;
            var data = new List<Int64> {};
            while ((datasample = dataStream.ReadLine()) != null)
            {
                data.Add(Int64.Parse(datasample));
            }

            return data;
        }
        
        static List<List<int>> readTxt2D(string path)
        {
            StreamReader dataStream = new StreamReader(path);   
            string datasample;
            var data = new List<List<int>> {};
            while ((datasample = dataStream.ReadLine()) != null)
            {
                var line_num = new List<int>();
                string[] numbers = datasample.Split(" ");
                foreach (var num in numbers)
                    if (!string.IsNullOrEmpty(num))
                        line_num.Add(int.Parse(num));
                data.Add(line_num);
            }
            return data;
        }

        static bool IsAccuracyEqual(double acc1, double acc2, int totalNum)
        {
            return (int) Math.Round(acc1 * totalNum) == (int) Math.Round(acc2 * totalNum);
        }
    }
}
