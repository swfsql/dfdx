use crate::prelude::*;

/// Calls on [crate::tensor_ops::TryDiv], which for tensors is [crate::tensor_ops::div()].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Div;
impl<Lhs, Rhs> Module<(Lhs, Rhs)> for Div
where
    Lhs: TryDiv<Rhs>,
{
    type Output = <Lhs as TryDiv<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        x.0.try_div(x.1)
    }
}
