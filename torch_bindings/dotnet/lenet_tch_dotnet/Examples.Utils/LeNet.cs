using System;
using System.IO;
using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    public class LeNet5 : Module
    {
        private Module conv1 = Conv2d(1, 6, 5, padding:2);
        private Module conv2 = Conv2d(6, 16, 5);
        private Module conv3 = Conv2d(16, 120, 5);
        private Module fc1 = Linear(120, 84);
        private Module fc2 = Linear(84, 10);

        // These don't have any parameters, so the only reason to instantiate
        // them is performance, since they will be used over and over.
        private Module pool1 = MaxPool2d(2, 2);
        private Module pool2 = MaxPool2d(2, 2);
        // private Module pool1 = AvgPool2d(2, 2);
        // private Module pool2 = AvgPool2d(2, 2);

        private Module tanh1 = Tanh();
        private Module tanh2 = Tanh();
        private Module tanh3 = Tanh();

        private Module flatten = Flatten();
        private Module logsm = LogSoftmax(1);

        public LeNet5(string name, torch.Device device = null) : base(name)
        {
            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            var l11 = conv1.forward(input);
            var l12 = tanh1.forward(l11);
            var l13 = pool1.forward(l12);

            var l21 = conv2.forward(l13);
            var l22 = tanh2.forward(l21);
            var l23 = pool2.forward(l22);

            var l31 = conv3.forward(l23);
            var l32 = tanh3.forward(l31);

            var x = flatten.forward(l32);

            var l41 = fc1.forward(x);
            var l42 = tanh3.forward(l41);

            var l51 = fc2.forward(l42);
            return logsm.forward(l51);
        }
    }

    public class LeNet1 : Module
    {
        private Module conv1 = Conv2d(1, 4, 5, padding:2);
        private Module conv2 = Conv2d(4, 12, 5);
        private Module fc1 = Linear(300, 10);

        // These don't have any parameters, so the only reason to instantiate
        // them is performance, since they will be used over and over.
        private Module pool1 = MaxPool2d(2, 2);
        private Module pool2 = MaxPool2d(2, 2);
        // private Module pool1 = AvgPool2d(2, 2);
        // private Module pool2 = AvgPool2d(2, 2);

        private Module tanh1 = Tanh();
        private Module tanh2 = Tanh();

        private Module flatten = Flatten();
        private Module logsm = LogSoftmax(1);

        public LeNet1(string name, torch.Device device = null) : base(name)
        {
            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            var l11 = conv1.forward(input);
            var l12 = tanh1.forward(l11);
            var l13 = pool1.forward(l12);

            var l21 = conv2.forward(l13);
            var l22 = tanh2.forward(l21);
            var l23 = pool2.forward(l22);

            var x = flatten.forward(l23);

            var l31 = fc1.forward(x);
            return logsm.forward(l31);
        }
    }
}