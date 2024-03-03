use crate::prelude::*;

/// Calls on [std::ops::Neg], which for tensors is [crate::tensor_ops::negate].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Neg;

impl<Input: std::ops::Neg> Module<Input> for Neg {
    type Output = <Input as std::ops::Neg>::Output;

    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        Ok(x.neg())
    }
}
