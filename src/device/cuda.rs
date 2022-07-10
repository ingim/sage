use rustacuda::context::{Context, ContextFlags};


thread_local!(static CUDA_INIT:bool = rustacuda::init(rustacuda::CudaFlags::empty()).is_ok());

pub struct Device {
    cuda_device: rustacuda::device::Device,
}

impl Device {
    pub fn list() -> Vec<Device> {
        let devices = rustacuda::device::Device::devices().unwrap();

        devices.map(|d| Device {
            cuda_device: d.unwrap()
        }).collect()
    }

    pub fn new() {}
}


// goes to the tensor..
pub struct Memory {}

// device::Cuda::list();
// device::Cuda::new(0);

// device::OpenCl::list();
// device::OpenCl::new();

// let mut ctx = device.create_context();

// Tensor::new().to(&mut ctx);
//
