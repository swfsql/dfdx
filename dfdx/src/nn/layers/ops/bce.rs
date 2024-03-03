use crate::prelude::*;

/// Calls [crate::tensor_ops::bce_with_logits].
#[derive(Default, Debug, Clone, Copy, crate::nn::CustomModule)]
pub struct Bce;
type Logits<S, E, D, T> = Tensor<S, E, D, T>;
type Probs<S, E, D, T> = Tensor<S, E, D, T>;

impl<S: Shape, E: Dtype, D: Device<E>, LTape: Tape<E, D>, RTape: Tape<E, D>>
    Module<(Logits<S, E, D, LTape>, Probs<S, E, D, RTape>)> for Bce
where
    LTape: Merge<RTape>,
{
    type Output = Logits<S, E, D, LTape>;

    fn try_forward(
        &self,
        x: (Logits<S, E, D, LTape>, Probs<S, E, D, RTape>),
    ) -> Result<Self::Output, Error> {
        x.0.try_bce_with_logits(x.1)
    }
}
