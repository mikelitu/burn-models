use burn::backend::{WgpuBackend, wgpu::AutoGraphicsApi};
use burn::tensor::Tensor;
use burn_demo::blocks::resnet::{resnet18, resnet50};

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;

    let tensor1 = Tensor::<MyBackend, 4>::ones([6, 3, 256, 256]);
    let my_model = resnet50::<MyBackend>(100);
    let out = my_model.forward(tensor1.clone());
    println!("The shape of the input tensor is: {:?} | The shape of the output tensor is: {:?}", tensor1.dims(), out.dims());
}