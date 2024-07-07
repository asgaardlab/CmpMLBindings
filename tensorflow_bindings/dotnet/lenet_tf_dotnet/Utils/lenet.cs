// using System;
// using System.IO;
// using System.Collections.Generic;
// using System.Linq;
// using static Tensorflow.Binding;
// using static Tensorflow.KerasApi;
// using Tensorflow;
// using NumSharp;
// using Tensorflow.Keras.ArgsDefinition;
// using Tensorflow.Keras.Engine;
// using Tensorflow.Keras.Engine.DataAdapters;
// using Tensorflow.Keras.Utils;
// using System.Diagnostics;
// using NDArray = Tensorflow.NumPy.NDArray;
//
//
// namespace tf_dotnet
// {
//     static Functional LeNet5()
//     {
//         var layers = new Tensorflow.Keras.Layers.LayersApi();
//         
//         // input layer
//         var inputs = keras.Input(shape: (32, 32, 1), name: "img");
//         
//         // convolutional layer 1
//         var c1 = layers.Conv2D(6, 5, activation: "tanh").Apply(inputs);
//         int[] poolSize = {2, 2};
//         int[] strides = {2, 2};
//         var p1 = layers.max_pooling2d(c1, poolSize, strides);
//
//         // convolutional layer 2
//         var c2 = layers.Conv2D(16, 5, activation: "tanh").Apply(p1);
//         var p2 = layers.max_pooling2d(c2, poolSize, strides);
//         
//         // convolutional layer 3
//         var c3 = layers.Conv2D(120, 5, activation: "tanh").Apply(p2);
//         
//         // fully connected layer
//         var flatten = layers.Flatten().Apply(c3);
//         var f1 = layers.Dense(84, activation: "tanh").Apply(flatten);
//         var logits = layers.Dense(10).Apply(f1);
//         
//         // build keras model
//         var model = keras.Model(inputs, logits, name: "LeNet5");
//         model.summary();
//         
//         // compile keras model in tensorflow static graph
//         var opt = keras.optimizers.SGD(5e-2f);
// //            var opt = keras.optimizers.RMSprop(1e-3f);
//         model.compile(
//             opt,
//             keras.losses.SparseCategoricalCrossentropy(from_logits: true),
//             new[] { "acc" });
//         return model;
//     }
// }