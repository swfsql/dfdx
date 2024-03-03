use crate::prelude::*;
use std::{fmt::Debug, marker::PhantomData};

/// Calls on [crate::tensor_ops::RealizeTo].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct RealizeTo<Dst: Shape>(PhantomData<Dst>);

impl<Dst: Shape<Concrete = <<Input as HasShape>::Shape as Shape>::Concrete>, Input> Module<Input>
    for RealizeTo<Dst>
where
    Input: dfdx_core::tensor_ops::RealizeTo + HasShape,
{
    type Output = Result<<Input as HasShape>::WithShape<Dst>, Input>;

    fn try_forward(&self, x: Input) -> Result<Self::Output, Error> {
        Ok(x.try_realize::<Dst>())
    }
}
