"""Keep SAM's CPU-offloaded video on CPU until its detector requests a frame."""
import torch


class _CPUFrames(torch.Tensor):
    """CPU storage adapter; selected frames are ordinary, transferable tensors."""

    @staticmethod
    def __new__(cls, images):
        return images.as_subclass(cls)

    def to(self, *args, **kwargs):
        # The initial-batch copier must not transfer the entire video.
        return self

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.cpu()
        elif isinstance(index, (list, tuple)):
            index = type(index)(i.cpu() if isinstance(i, torch.Tensor) else i for i in index)
        return self.as_subclass(torch.Tensor)[index]


def keep_video_on_cpu(model):
    """Adapt this model instance without changing decoding, tensors, or inference.

    SAM's initial-batch copier otherwise moves the entire CPU video to CUDA.
    Its detector keeps its Tensor/advanced-indexing branch, preserving the exact
    selected-frame layout before transfer. Other batch fields are copied normally.
    Call once after constructing the predictor, before starting a session.
    """
    original = model._construct_initial_input_batch

    def construct(inference_state, images):
        if isinstance(images, torch.Tensor) and images.device.type == "cpu":
            images = _CPUFrames(images)
        return original(inference_state, images)

    model._construct_initial_input_batch = construct
