using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static TorchSharp.torch;


namespace TorchSharp.Examples
{
    public sealed class IMDbReader : Reader
    {
        public IMDbReader(string path, bool test, int batch_size = 256, Device device = null)
        {
            List<List<int>> x;
            List<float> y;
            if (test)
            {
                x = Helper.readTxt2D<int>(Path.Combine(path, "x_test.txt"));
                y = Helper.readTxt<float>(Path.Combine(path, "y_test.txt"));
            }
            else
            {
                x = Helper.readTxt2D<int>(Path.Combine(path, "x_train.txt"));
                y = Helper.readTxt<float>(Path.Combine(path, "y_train.txt"));
            }
            Console.WriteLine($"x: {x.Count} x {x[0].Count}");
            Console.WriteLine($"y: {y.Count}");
            
            int seq_length = x[0].Count;
            for (var i = 0; i < x.Count; i+=batch_size) {
                var take = Math.Min(batch_size, Math.Max(0, x.Count - i));
                if (take < 1) break;
                int end = i + take;
                // Console.WriteLine($"========= i: {i}, {take}, {end}");

                var x_batch_flatten = x.GetRange(i, take).SelectMany(x => x).ToList();
                // IList<int> Ix_batch_flatten = x_batch_flatten.ToList<int>();
                var y_batch = y.GetRange(i, take);
                var dataTensor = torch.tensor(x_batch_flatten, take, seq_length, torch.int32, device);
                var lablTensor = torch.tensor(y_batch, take, 1, torch.float32, device);
                
                // Console.WriteLine($"x_batch_flatten: {x_batch_flatten.Count}, {dataTensor.shape[0]} x {dataTensor.shape[1]}");
                // Console.WriteLine($"y_batch: {y_batch.Count}, {lablTensor.shape[0]}");
                
                data.Add(dataTensor);
                labels.Add(lablTensor);
            }
            
            // Console.WriteLine($"data: {data.Count}, {data[0].shape[0]} x {data[0].shape[1]}");
            // Console.WriteLine($"labels: {labels.Count}, {labels[0].shape[0]} x {labels[0].shape[1]}");
            
            Size = y.Count;
        }
    }
}
