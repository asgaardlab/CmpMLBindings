// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Data reader utility for datasets that follow the MNIST data set's layout:
    ///
    /// A number of single-channel (grayscale) images are laid out in a flat file with four 32-bit integers at the head.
    /// The format is documented at the bottom of the page at: http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public sealed class CIFARReader : Reader
    {
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="path">Path to the folder containing the image files.</param>
        /// <param name="test">True if this is a test set, otherwise false.</param>
        /// <param name="batch_size">The batch size</param>
        /// <param name="shuffle">Randomly shuffle the images.</param>
        /// <param name="device">The device, i.e. CPU or GPU to place the output tensors on.</param>
        /// <param name="transforms">A list of image transformations, helpful for data augmentation.</param>
        public CIFARReader(string path, bool test, int batch_size = 32, bool shuffle = false, Device device = null, torchvision.ITransform transform = null)
        {
            int count = 0;
            if (test) {
                count = ReadSingleFile(Path.Combine(path, "test_batch.bin"), batch_size, shuffle, device, transform);
            } else {
                count += ReadSingleFile(Path.Combine(path, "data_batch_1.bin"), batch_size, shuffle, device, transform);
                count += ReadSingleFile(Path.Combine(path, "data_batch_2.bin"), batch_size, shuffle, device, transform);
                count += ReadSingleFile(Path.Combine(path, "data_batch_3.bin"), batch_size, shuffle, device, transform);
                count += ReadSingleFile(Path.Combine(path, "data_batch_4.bin"), batch_size, shuffle, device, transform);
                count += ReadSingleFile(Path.Combine(path, "data_batch_5.bin"), batch_size, shuffle, device, transform);
            }

            Size = count;
        }

        private int ReadSingleFile(string path, int batch_size, bool shuffle, Device device, torchvision.ITransform transform)
        {
            const int height = 32;
            const int width = 32;
            const int channels = 3;
            const int count = 10000;

            byte[] dataBytes = File.ReadAllBytes(path);

            if (dataBytes.Length != (1 + channels * height * width) * count)
                throw new InvalidDataException($"Not a proper CIFAR10 file: {path}");

            // Set up the indices array.
            Random rnd = new Random();
            var indices = !shuffle ?
                Enumerable.Range(0, count).ToArray() :
                Enumerable.Range(0, count).OrderBy(c => rnd.Next()).ToArray();

            var imgSize = channels * height * width;

            // Go through the data and create tensors
            for (var i = 0; i < count;) {

                var take = Math.Min(batch_size, Math.Max(0, count - i));

                if (take < 1) break;

                var dataTensor = torch.zeros(new long[] { take, imgSize }, device: device);
                var lablTensor = torch.zeros(new long[] { take }, torch.int64, device: device);

                // Take
                for (var j = 0; j < take; j++) {
                    var idx = indices[i++];
                    var lblStart = idx * (1 + imgSize);
                    var imgStart = lblStart + 1;

                    lablTensor[j] = torch.tensor(dataBytes[lblStart], torch.int64);

                    var floats = dataBytes[imgStart..(imgStart + imgSize)].Select(b => (float)b/255.0).ToArray();
                    using (var inputTensor = torch.tensor(floats))
                        dataTensor.index_put_(inputTensor, TensorIndex.Single(j));
                }

                var batch = dataTensor.reshape(take, channels, height, width);

                if (transform != null) {
                    // Carefully dispose the original
                    using(var batch_copy = batch)
                        batch = transform.forward(batch);
                }
                
                data.Add(batch);
                dataTensor.Dispose();
                labels.Add(lablTensor);
            }

            return count;
        }

        // public int Size { get {
        //         return _size;
        //     } }
        // private int _size = 0;
    }
}
