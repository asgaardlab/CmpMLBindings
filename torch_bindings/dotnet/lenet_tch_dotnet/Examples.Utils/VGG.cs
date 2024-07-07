using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Modified version of VGG to classify CIFAR10 32x32 images.
    /// </summary>
    /// <remarks>
    /// With an unaugmented CIFAR-10 data set, the author of this saw training converge
    /// at roughly 85% accuracy on the test set, after 50 epochs using VGG-16.
    /// </remarks>
    class VGG : Module
    {
        // The code here is is loosely based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
        // Licence and copypright notice at: https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE

        private readonly Dictionary<string, long[]> _channels = new Dictionary<string, long[]>() {
            { "vgg11", new long[] { 64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
            { "vgg13", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
            { "vgg16", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0 } },
            { "vgg19", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512, 0, 512, 512, 512, 512, 0 } }
        };

        private readonly Module layers;

        public VGG(string name, int numClasses, Device device = null) : base(name)
        {
            var modules = new List<(string, Module)>();

            var channels = _channels[name];

            long in_channels = 3;

            for (var i = 0; i < channels.Length; i++) {

                if (channels[i] == 0) {
                    modules.Add(($"maxpool2d-{i}a", MaxPool2d(kernelSize: 2, stride: 2)));
                } else {
                    modules.Add(($"conv2d-{i}a", Conv2d(in_channels, channels[i], kernelSize: 3, padding: 1, bias: false)));
                    modules.Add(($"bnrm2d-{i}a", BatchNorm2d(channels[i])));
                    modules.Add(($"relu-{i}b", ReLU()));
                    in_channels = channels[i];
                }
            }
            // modules.Add(("avgpool2d", AvgPool2d(kernel_size: 1, stride: 1)));
            modules.Add(("flatten", Flatten()));
            modules.Add(("fc_linear_1", Linear(512, 4096)));
            modules.Add(("fc_relu_1", ReLU()));
            modules.Add(("fc_dropout_1", Dropout(0.5)));
            modules.Add(("fc_linear_2", Linear(4096, 4096)));
            modules.Add(("fc_relu_2", ReLU()));
            modules.Add(("fc_dropout_2", Dropout(0.5)));
            modules.Add(("fc_out", Linear(4096, numClasses)));

            layers = Sequential(modules);

            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            return layers.forward(input);
        }
    }
    
    class VGG_fixed : Module
    {
        // The code here is is loosely based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
        // Licence and copypright notice at: https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE

        private readonly Dictionary<string, long[]> _channels = new Dictionary<string, long[]>() {
            { "vgg11", new long[] { 64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
            { "vgg13", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
            { "vgg16", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0 } },
            { "vgg19", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512, 0, 512, 512, 512, 512, 0 } }
        };

        private readonly Module layers;
        private Module flatten;
        private Module fc_linear_1;
        private Module fc_linear_2;
        private Module fc_out;
        private Module relu_m;
        private Module dropout_m;

        public VGG_fixed(string name, int numClasses, Device device = null) : base(name)
        {
            var modules = new List<(string, Module)>();

            var channels = _channels[name];

            long in_channels = 3;

            for (var i = 0; i < channels.Length; i++) {

                if (channels[i] == 0) {
                    modules.Add(($"maxpool2d-{i}a", MaxPool2d(kernelSize: 2, stride: 2)));
                } else {
                    modules.Add(($"conv2d-{i}a", Conv2d(in_channels, channels[i], kernelSize: 3, padding: 1, bias: false)));
                    modules.Add(($"bnrm2d-{i}a", BatchNorm2d(channels[i])));
                    modules.Add(($"relu-{i}b", ReLU()));
                    in_channels = channels[i];
                }
            }
            layers = Sequential(modules);
            
            flatten = Flatten();
            fc_linear_1 = Linear(512, 4096);
            relu_m = ReLU();
            dropout_m = Dropout(0.5);
            fc_linear_2 = Linear(4096, 4096);
            fc_out = Linear(4096, numClasses);

            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            var o1 = layers.forward(input);
            var f1 = flatten.forward(o1);
            var fc1_out = dropout_m.forward(relu_m.forward(fc_linear_1.forward(f1)));
            var fc2_out = dropout_m.forward(relu_m.forward(fc_linear_2.forward(fc1_out)));
            return fc_out.forward(fc2_out);
        }
    }
}