// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp.torchvision;
using static TorchSharp.torch;

namespace TorchSharp.Examples
{
    public class Reader : IDisposable
    {
        public int BatchSize { get; protected set; }
        public int Size { get; set; }

        public List<Tensor> data = new List<Tensor>();
        public List<Tensor> labels = new List<Tensor>();


        public IEnumerable<(Tensor, Tensor)> Data()
        {
            for (var i = 0; i < data.Count; i++) {
                yield return (data[i], labels[i]);
            }
        }
        
        public void Dispose()
        {
            data.ForEach(d => d.Dispose());
            labels.ForEach(d => d.Dispose());
        }
    }
}
