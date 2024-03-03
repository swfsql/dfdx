use crate::prelude::*;

/// Calls on [crate::tensor_ops::ChooseFrom].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Choose<S: Shape, Dev: Storage<bool>> {
    #[cfg_attr(feature = "safetensors", serialize)]
    pub choose: Tensor<S, bool, Dev>,
}

impl<S: Shape, Elem: Dtype, Dev: Device<Elem>> ::dfdx::nn_traits::ResetParams<Elem, Dev>
    for Choose<S, Dev>
{
    fn try_reset_params(&mut self) -> Result<(), ::dfdx::tensor::Error> {
        Ok(())
    }
}

impl<S: Shape, Elem: Dtype, Dev: Device<Elem>> ::dfdx::nn_traits::UpdateParams<Elem, Dev>
    for Choose<S, Dev>
{
    fn try_update_params<_Model, Optim: ::dfdx::nn_traits::Optimizer<_Model, Elem, Dev>>(
        &mut self,
        _optimizer: &mut Optim,
        _gradients: &::dfdx::tensor::Gradients<Elem, Dev>,
        _missing_tensors: &mut Vec<::dfdx::tensor::UniqueId>,
    ) -> Result<(), ::dfdx::tensor::Error> {
        Ok(())
    }
}

impl<S: Shape, Elem: Dtype, Dev: Device<Elem>> ::dfdx::nn_traits::ZeroGrads<Elem, Dev>
    for Choose<S, Dev>
{
    fn try_zero_grads(
        &self,
        _grads: &mut ::dfdx::prelude::Gradients<Elem, Dev>,
    ) -> Result<(), ::dfdx::tensor::Error> {
        Ok(())
    }
}

impl<S: Shape, Lhs, Rhs, Dev: Storage<bool>> Module<(Lhs, Rhs)> for Choose<S, Dev>
where
    Tensor<S, bool, Dev>: ChooseFrom<Lhs, Rhs>,
{
    type Output = <Tensor<S, bool, Dev> as ChooseFrom<Lhs, Rhs>>::Output;

    fn try_forward(&self, x: (Lhs, Rhs)) -> Result<Self::Output, Error> {
        self.choose.clone().try_choose(x.0, x.1)
    }
}
