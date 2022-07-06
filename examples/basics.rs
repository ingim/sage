use sage::session::context::Context;
use sage::tensor::Tensor;
use sage::var::Var;

fn main() {

    // Async operations

    // let device_list = Device::list()
    // Select GPU
    // device = device_list[0];

    // let mut ctx = Context::new(device);



    // Coordination between multiple processors

    // tensor.move(ctx)

    // Context specifies the processor (e.g., GPU) that executes the program.
    let mut ctx = Context::with_device(0);

    // Tensors are n-dimension array
    let x_data = Tensor::new([
        [0.5173, -0.9896, -0.7773],
        [0.1546, -0.7499, 0.2420],
        [-1.6632, 1.0712, -0.2654],
    ]).to_device(&mut ctx);

    // Variables hold (un)evaluated tensors.
    let x = Var::new(x_data);

    let y = Var::new(Tensor::new([
        [0.5173, -0.9896, -0.7773],
        [0.1546, -0.7499, 0.2420],
        [-1.6632, 1.0712, -0.2654],
    ]).to_device(&mut ctx));
}