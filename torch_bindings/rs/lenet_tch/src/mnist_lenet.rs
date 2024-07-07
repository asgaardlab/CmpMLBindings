use tch::{nn, nn::ModuleT, Tensor};


#[derive(Debug)]
pub struct LeNet5 {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl LeNet5 {
    pub fn new(vs: &nn::Path) -> LeNet5 {
        let mut config = nn::ConvConfig::default();
        config.padding = 2;
        let conv1 = nn::conv2d(vs, 1, 6, 5, config);
        let conv2 = nn::conv2d(vs, 6, 16, 5, Default::default());
        let conv3 = nn::conv2d(vs, 16, 120, 5, Default::default());
        let fc1 = nn::linear(vs, 120, 84, Default::default());
        let fc2 = nn::linear(vs, 84, 10, Default::default());
        LeNet5 {
            conv1,
            conv2,
            conv3,
            fc1,
            fc2,
        }
    }

    #[allow(dead_code)]
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_t(xs, false)
    }
}

impl nn::ModuleT for LeNet5 {
    #![allow(unused_variables)]
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .tanh()
            // .avg_pool2d_default(2)
            // .avg_pool2d(&[2, 2], &[2, 2], &[0, 0], false, true, Option::None)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .tanh()
            // .avg_pool2d_default(2)
            // .avg_pool2d(&[2, 2], &[2, 2], &[0, 0], false, true, Option::None)
            .max_pool2d_default(2)
            .apply(&self.conv3)
            .tanh()
            .flatten(1, -1)
            .apply(&self.fc1)
            .tanh()
            .apply(&self.fc2)
    }
}

#[derive(Debug)]
pub struct LeNet1 {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
}

impl LeNet1 {
    pub fn new(vs: &nn::Path) -> LeNet1 {
        let mut config = nn::ConvConfig::default();
        config.padding = 2;
        let conv1 = nn::conv2d(vs, 1, 4, 5, config);
        let conv2 = nn::conv2d(vs, 4, 12, 5, Default::default());
        let fc1 = nn::linear(vs, 300, 10, Default::default());
        LeNet1 {
            conv1,
            conv2,
            fc1,
        }
    }

    #[allow(dead_code)]
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_t(xs, false)
    }

}

impl nn::ModuleT for LeNet1 {
    #![allow(unused_variables)]
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .tanh()
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .tanh()
            .max_pool2d_default(2)
            .flatten(1, -1)
            .apply(&self.fc1)
    }
}
