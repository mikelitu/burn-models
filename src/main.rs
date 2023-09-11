use burn::backend::{WgpuBackend, wgpu::AutoGraphicsApi};
use burn::tensor::Tensor;
use burn_demo::blocks::resnet::{resnet18, resnet50};
use std::time::Instant;

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    let num_classes = 100;
    let tensor1 = Tensor::<MyBackend, 4>::ones([6, 3, 256, 256]);
    let start = Instant::now();
    let model50 = resnet50::<MyBackend>(num_classes);
    {
        let _out50 = model50.forward(tensor1.clone());
    }
    let elapsed = start.elapsed();
    println!("Time for the creation and inference of the ResNet50: {:.2?}", elapsed);

    let start = Instant::now();
    let model18 = resnet18::<MyBackend>(num_classes);
    {
        let _out18 = model18.forward(tensor1.clone());
    }
    let elapsed = start.elapsed();
    println!("Time for the creation and inference of the ResNet18: {:.2?}", elapsed);
}