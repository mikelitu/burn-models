use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        ReLU, BatchNorm, BatchNormConfig
    },
    tensor::{backend::Backend, Tensor}, 
};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    batchnorm: BatchNorm<B, 2>,
    activation: ReLU,
}

#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    channels: [usize; 2],
    kernel: [usize; 2],
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self) -> ConvBlock<B> {
        ConvBlock {
            conv: Conv2dConfig::new(self.channels, self.kernel)
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(),
            batchnorm: BatchNormConfig::new(self.channels[1]).init(),
            activation: ReLU::new(),
        }
    }

    pub fn init_with<B: Backend>(&self, record: ConvBlockRecord<B>) -> ConvBlock<B> {
        ConvBlock {
            conv: Conv2dConfig::new(self.channels, self.kernel).init_with(record.conv),
            batchnorm: BatchNormConfig::new(self.channels[1]).init(),
            activation: ReLU::new(),
        }
    }
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.batchnorm.forward(x);
        self.activation.forward(x)
    }
}