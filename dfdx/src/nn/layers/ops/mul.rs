use crate::prelude::*;

/// Calls on [crate::tensor_ops::TryMul], which for tensors is [crate::tensor_ops::mul()].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Mul;
impl<Lhs, Rhs> Module<(Lhs, Rhs)> for Mul
where
    Lhs: TryMul<Rhs>,
{
    type Output = <Lhs as TryMul<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        x.0.try_mul(x.1)
    }
}
