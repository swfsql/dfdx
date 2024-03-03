//! Layers that mostly wraps the functionality of a [tensor_ops].

#[allow(unused_imports)]
use crate::tensor_ops;

pub mod abs;
pub mod cos;
pub mod dropout;
pub mod exp;
pub mod gelu;
pub mod leaky_relu;
pub mod ln;
pub mod log_softmax;
#[cfg(feature = "nightly")]
pub mod pool_2d_avg;
#[cfg(feature = "nightly")]
pub mod pool_2d_max;
#[cfg(feature = "nightly")]
pub mod pool_2d_min;
pub mod prelu;
pub mod prelu1d;
pub mod relu;
pub mod sigmoid;
pub mod sin;
pub mod softmax;
pub mod sqrt;
pub mod square;
pub mod tanh;

pub use abs::Abs;
pub use cos::Cos;
pub use dropout::{Dropout, DropoutOneIn};
pub use exp::Exp;
pub use gelu::{AccurateGeLU, FastGeLU};
pub use leaky_relu::LeakyReLU;
pub use ln::Ln;
pub use log_softmax::LogSoftmax;
#[cfg(feature = "nightly")]
pub use pool_2d_avg::{AvgPool2D, AvgPool2DConst};
#[cfg(feature = "nightly")]
pub use pool_2d_max::{MaxPool2D, MaxPool2DConst};
pub use pool_2d_min::{MinPool2D, MinPool2DConst};
#[cfg(feature = "nightly")]
pub use prelu::{PReLU, PReLUConfig};
pub use prelu1d::{PReLU1D, PReLU1DConfig};
pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use sin::Sin;
pub use softmax::Softmax;
pub use sqrt::Sqrt;
pub use square::Square;
pub use tanh::Tanh;
