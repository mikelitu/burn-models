use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor}
};

use crate::blocks::conv::{ConvBlock, ConvBlockConfig};

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    blocks: Vec<ConvBlock<B>>,
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    channel: usize,
    kernel: [usize; 2],
}

impl ResBlockConfig {
    pub fn init<B: Backend>(&self, num_blocks: usize) -> ResBlock<B> {
        let mut blocks = Vec::new();
        for _n in 1..num_blocks {
            blocks.push(
                ConvBlockConfig::new([self.channel, self.channel], self.kernel)
                .init()
            )
        }
        ResBlock { blocks }
    }

    pub fn init_with<B: Backend>(&self, num_blocks: usize, record: ResBlockRecord<B>) -> ResBlock<B> {
        let mut blocks = Vec::new();
        for n in 1..num_blocks {
            blocks.push(
                ConvBlockConfig::new([self.channel, self.channel], self.kernel)
                .init_with(record.blocks[n].clone())
            )
        }
        ResBlock { blocks }
    }
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        
        let input = x.clone();
        for block in &self.blocks {
            x = block.forward(x);
        }
        input + x
    }
}