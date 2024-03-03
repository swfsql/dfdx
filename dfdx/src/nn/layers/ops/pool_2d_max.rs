use crate::prelude::*;

// TODO: check if that information is saved on safetensor serialization
//
/// Max pool with 2d kernel that operates on images (3d) and batches of images (4d).
/// Each patch reduces to the maximum value in that patch.
///
/// Generics:
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
/// - `DILATION` How dilated the kernel should be. Defaults to `1`.
#[derive(Clone, Debug, Default, ResetParams, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct MaxPool2D<
    KernelSize: Dim,
    Stride: Dim = Const<1>,
    Padding: Dim = Const<0>,
    Dilation: Dim = Const<1>,
> {
    #[cfg_attr(feature = "safetensors", serialize)]
    pub kernel_size: KernelSize,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub stride: Stride,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub padding: Padding,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub dilation: Dilation,
}

pub type MaxPool2DConst<
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
    const DILATION: usize = 1,
> = MaxPool2D<Const<KERNEL_SIZE>, Const<STRIDE>, Const<PADDING>, Const<DILATION>>;

impl<K: Dim, S: Dim, P: Dim, D: Dim, Elem: Dtype, Dev: Device<Elem>> BuildOnDevice<Elem, Dev>
    for MaxPool2D<K, S, P, D>
{
    type Built = Self;
    fn try_build_on_device(&self, _device: &Dev) -> Result<Self::Built, crate::tensor::Error> {
        Ok(self.clone())
    }
}

impl<K: Dim, S: Dim, P: Dim, L: Dim, Img: TryPool2D<K, S, P, L>> Module<Img>
    for MaxPool2D<K, S, P, L>
{
    type Output = Img::Pooled;
    fn try_forward(&self, x: Img) -> Result<Self::Output, Error> {
        x.try_pool2d(
            crate::tensor_ops::Pool2DKind::Max,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
