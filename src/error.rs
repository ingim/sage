use crate::session::device::DeviceError;
use crate::session::memory::MemoryError;
use crate::shape::ShapeError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("memory error")]
    Memory(MemoryError),
    #[error("device error")]
    Device(DeviceError),
    #[error("shape error")]
    Shape(ShapeError),
}
