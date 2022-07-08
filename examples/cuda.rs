#[macro_use]
extern crate rustacuda;

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use std::error::Error;
use std::ffi::CString;
use std::fs;

use std::process::Command;


fn compile_cuda(code: &str) -> CString {
    // step 1. save file

    // step 2. call nvcc

    let filename = "asd123";
    fs::create_dir_all("./.cache/kernels/").unwrap();

    let input_path = format!("./.cache/kernels/{filename}.cu");
    let output_path = format!("./.cache/kernels/{filename}.ptx");


    // step 3. load string
    fs::write(&input_path, code).expect("Unable to write file");

    println!("start building...");

    let output = Command::new("nvcc")
        .arg(&input_path)
        .arg(format!("-o={output_path}"))
        .arg("-c")
        .arg("--ptx")
        //.arg("-cudart=shared")
        //.arg("-gencode")
        //.arg("arch=compute_61,code=sm_61")
        .output()
        .expect("failed to execute process");


    let aa = fs::read_to_string(output_path).expect("unable to read file!");
    CString::new(aa).unwrap()
    //CString::new("asdasd").unwrap()
}


fn main() -> Result<(), Box<dyn Error>> {
    let src = compile_cuda(r#"

extern "C"  __global__ void sum(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + y[i];
    }
}
"#);

    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // Create a context associated to this device
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module = Module::load_from_string(&src)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create buffers for data
    let mut in_x = DeviceBuffer::from_slice(&[1.0f32; 10])?;
    let mut in_y = DeviceBuffer::from_slice(&[2.0f32; 10])?;
    let mut out_1 = DeviceBuffer::from_slice(&[0.0f32; 10])?;
    let mut out_2 = DeviceBuffer::from_slice(&[0.0f32; 10])?;

    // This kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
    unsafe {
        // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
        let result = launch!(module.sum<<<1, 1, 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_1.as_device_ptr(),
            out_1.len()
        ));
        result?;

        // Launch the kernel again using the `function` form:
        let function_name = CString::new("sum")?;
        let sum = module.get_function(&function_name)?;
        // Launch with 1x1x1 (1) blocks of 10x1x1 (10) threads, to show that you can use tuples to
        // configure grid and block size.
        let result = launch!(sum<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_2.as_device_ptr(),
            out_2.len()
        ));
        result?;
    }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;

    // Copy the results back to host memory
    let mut out_host = [0.0f32; 20];
    out_1.copy_to(&mut out_host[0..10])?;
    out_2.copy_to(&mut out_host[10..20])?;

    for x in out_host.iter() {
        assert_eq!(3.0 as u32, *x as u32);
    }

    println!("Launched kernel successfully.");
    Ok(())
}