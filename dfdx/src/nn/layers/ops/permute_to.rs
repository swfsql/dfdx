use crate::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

/// Calls on [crate::tensor_ops::PermuteTo].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct PermuteTo<Dst: Shape, Ax: Axes + Debug>(PhantomData<(Dst, Ax)>);

impl<Dst: Shape, Ax: Axes + Debug, Input> Module<Input> for PermuteTo<Dst, Ax>
where
    Input: dfdx_core::tensor_ops::PermuteTo,
    <Input as HasShape>::Shape: PermuteShapeTo<Dst, Ax>,
{
    type Output = <Input as HasShape>::WithShape<Dst>;

    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        x.try_permute()
    }
}
