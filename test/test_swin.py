import torch
from swiftfmri.models.swin_transformer import SwinTransformer


def test_swin():
    model = SwinTransformer()

    x = torch.randn(4, 3, 224, 224)  # b, c, h, w, d, t

    z = model(x)
    loss = z.sum()
    loss.backward()
    assert not torch.isnan(loss) 
