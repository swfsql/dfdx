use crate::prelude::*;

#[derive(Clone, Copy, Debug, Default, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Upscale2D<
    OutHeight: Dim,
    OutWidth: Dim = OutHeight,
    Method: UpscaleMethod = NearestNeighbor,
> {
    #[cfg_attr(feature = "safetensors", serialize)]
    pub out_width: OutWidth,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub out_height: OutHeight,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub method: Method,
}

pub type Upscale2DConst<const OH: usize, const OW: usize = OH, M = NearestNeighbor> =
    Upscale2D<Const<OH>, Const<OW>, M>;

impl<Oh: Dim, Ow: Dim, M: UpscaleMethod, E: Dtype, D: Device<E>> BuildOnDevice<E, D>
    for Upscale2D<Oh, Ow, M>
{
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<H: Dim, W: Dim, M: UpscaleMethod, Img: GenericUpscale2D<M>> Module<Img>
    for Upscale2D<H, W, M>
{
    type Output = Img::Output<H, W>;
    fn try_forward(&self, x: Img) -> Result<Self::Output, Error> {
        x.generic_upscale2d_like(self.method, self.out_height, self.out_width)
    }
}

#[derive(Clone, Copy, Debug, Default, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct Upscale2DBy<H: Dim, W: Dim = H, Method: UpscaleMethod = NearestNeighbor> {
    #[cfg_attr(feature = "safetensors", serialize)]
    pub height_factor: H,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub width_factor: W,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub method: Method,
}

pub type Upscale2DByConst<const HF: usize, const WF: usize = HF, M = NearestNeighbor> =
    Upscale2DBy<Const<HF>, Const<WF>, M>;

impl<H: Dim, W: Dim, M: UpscaleMethod, E: Dtype, D: Device<E>> BuildOnDevice<E, D>
    for Upscale2DBy<H, W, M>
{
    type Built = Self;
    fn try_build_on_device(&self, _device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(*self)
    }
}

impl<HF: Dim, WF: Dim, C: Dim, H, W, M: UpscaleMethod, E: Dtype, D, T: Tape<E, D>>
    Module<Tensor<(C, H, W), E, D, T>> for Upscale2DBy<HF, WF, M>
where
    H: Dim + std::ops::Mul<HF>,
    H::Output: Dim,
    W: Dim + std::ops::Mul<WF>,
    W::Output: Dim,
    D: Device<E> + Upscale2DKernel<E, M>,
{
    type Output = Tensor<(C, H::Output, W::Output), E, D, T>;
    fn try_forward(&self, x: Tensor<(C, H, W), E, D, T>) -> Result<Self::Output, Error> {
        let (_c, h, w) = *x.shape();
        let h = h * self.height_factor;
        let w = w * self.width_factor;
        x.generic_upscale2d_like(self.method, h, w)
    }
}

impl<HF: Dim, WF: Dim, B: Dim, C: Dim, H, W, M: UpscaleMethod, E: Dtype, D, T>
    Module<Tensor<(B, C, H, W), E, D, T>> for Upscale2DBy<HF, WF, M>
where
    H: Dim + std::ops::Mul<HF>,
    H::Output: Dim,
    W: Dim + std::ops::Mul<WF>,
    W::Output: Dim,
    D: Device<E> + Upscale2DKernel<E, M>,
    T: 'static + Tape<E, D>,
{
    type Output = Tensor<(B, C, H::Output, W::Output), E, D, T>;
    fn try_forward(&self, x: Tensor<(B, C, H, W), E, D, T>) -> Result<Self::Output, Error> {
        let (_b, _c, h, w) = *x.shape();
        let h = h * self.height_factor;
        let w = w * self.width_factor;
        x.generic_upscale2d_like(self.method, h, w)
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_upscale2d() {
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank3<3, 4, 4>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank3<3, 8, 8>, _, _> = Upscale2D::<Const<8>>::default().forward(x.clone());
        let _: Tensor<Rank3<3, 8, 12>, _, _> =
            Upscale2D::<Const<8>, Const<12>>::default().forward(x.clone());
        let _: Tensor<Rank3<3, 9, 9>, _, _> =
            Upscale2D::<Const<9>, Const<9>, NearestNeighbor>::default().forward(x);
    }

    #[test]
    fn test_upscale2dby() {
        use crate::prelude::Bilinear;
        let dev: TestDevice = Default::default();
        let x: Tensor<Rank3<3, 4, 4>, TestDtype, _> = dev.zeros();
        let _: Tensor<Rank3<3, 8, 8>, _, _> = Upscale2DBy::<Const<2>>::default().forward(x.clone());
        let _: Tensor<Rank3<3, 8, 12>, _, _> =
            Upscale2DBy::<Const<2>, Const<3>>::default().forward(x.clone());
        let _: Tensor<Rank3<3, 12, 12>, _, _> =
            Upscale2DBy::<Const<3>, Const<3>, Bilinear>::default().forward(x.clone());
    }
}
