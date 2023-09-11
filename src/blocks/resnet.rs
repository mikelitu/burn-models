use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Linear, LinearConfig
    }
};

use crate::blocks::residual::{ResBlock, ResBlockConfig};

#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    conv: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    res_blocks: Vec<ResBlock<B>>,
    reducing_conv: Vec<Conv2d<B>>,
    classification: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ResNetConfig {
    num_classes: usize,
    #[config(default="[7, 7]")]
    init_kernel: [usize; 2],
    #[config(default="[3, 32]")]
    in_channels: [usize; 2],
    blocks: Vec<usize>,
    #[config(default="[128, 128]")]
    input_size: [usize; 2],
}

impl ResNetConfig {

    pub fn init<B: Backend>(&self, layers: Vec<usize>) -> ResNet<B> {
        let reduced_shape: [usize; 2] = self.input_size.map(|x| x / (2_usize.pow(self.blocks.len() as u32)) - 1);
        println!("The reduces shape of the tensor is {:?}", reduced_shape);
        let linear_input = layers[layers.len()-1] * reduced_shape[0] * reduced_shape[1];
        let mut res_blocks = Vec::new();
        let mut reducing_conv = Vec::new();
        let llen = layers.len() - 1;

        for (idx, (l, b)) in layers.clone().into_iter().zip(self.blocks.clone()).enumerate() {
            res_blocks.push(
                ResBlockConfig::new(l, [3, 3]).init(b)
            );
            if idx < llen {
                reducing_conv.push(
                    Conv2dConfig::new([l, layers[idx+1]], [3, 3])
                    .with_stride([2, 2])
                    .init()
                )
            }
            
        }

        ResNet { 
            conv: Conv2dConfig::new(self.in_channels, self.init_kernel).init(),
            pool: AdaptiveAvgPool2dConfig::new([64, 64]).init(),
            res_blocks,
            reducing_conv,
            classification: LinearConfig::new(linear_input, self.num_classes).init()}
    }

    pub fn init_with<B:Backend>(&self, record: ResNetRecord<B>, layers: Vec<usize>) -> ResNet<B> {
        let reduced_shape: [usize; 2] = self.input_size.map(|x| x / (2_usize.pow(self.blocks.len() as u32)) - 1);
        
        let linear_input = layers[layers.len() - 1] * reduced_shape[0] * reduced_shape[1];
        let mut res_blocks = Vec::new();
        let mut reducing_conv = Vec::new();
        for (idx, (l, b)) in layers.clone().into_iter().zip(self.blocks.clone()).enumerate() {
            res_blocks.push(
                ResBlockConfig::new(l, [3, 3]).init_with(b, record.res_blocks[idx].clone())
            );
            if idx < layers.len() {
                reducing_conv.push(
                    Conv2dConfig::new([l, layers[idx+1]], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(burn::nn::PaddingConfig2d::Same)
                    .init_with(record.reducing_conv[idx].clone())
                )
            }
            
        }
        
        ResNet { 
            conv: Conv2dConfig::new(self.in_channels, self.init_kernel).init_with(record.conv),
            pool: AdaptiveAvgPool2dConfig::new([128, 128]).init(),
            res_blocks,
            reducing_conv,
            classification: LinearConfig::new(linear_input, self.num_classes).init_with(record.classification)
        }
    }
}


impl<B: Backend> ResNet<B> {
    pub fn forward(&self, mut input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _channels, _width, _height] = input.dims();
        input = self.conv.forward(input);
        input = self.pool.forward(input);

        let iter = self.res_blocks.clone().into_iter().zip(self.reducing_conv.clone());
        for (b, r) in iter {
            input = r.forward(b.forward(input));
        }
        let input = input.reshape([batch_size, 256 * 7 * 7]);
        self.classification.forward(input)
    }
}

pub fn resnet18<B: Backend>(num_classes: usize)  -> ResNet<B> {
    ResNetConfig::new(num_classes, vec![2, 2, 2, 2]).init(vec![32, 64, 128, 256])
}

pub fn resnet50<B: Backend>(num_classes: usize) -> ResNet<B> {
    ResNetConfig::new(num_classes, vec![3, 4, 6, 3]).init(vec![32, 64, 128, 256])
}