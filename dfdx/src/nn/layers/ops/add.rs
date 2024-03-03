use crate::prelude::*;

/// Calls on [crate::tensor_ops::TryAdd], which for tensors is [crate::tensor_ops::add()].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Add;
impl<Lhs, Rhs> Module<(Lhs, Rhs)> for Add
where
    Lhs: TryAdd<Rhs>,
{
    type Output = <Lhs as TryAdd<Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        x.0.try_add(x.1)
    }
}
