from albumentations import Compose, LongestMaxSize, DualTransform
import albumentations.augmentations.functional as F
import cv2


class HCompose(Compose):
    def __init__(self, transforms, *args, additional_targets=None, no_nearest_for_masks=True, **kwargs):
        if additional_targets is None:
            additional_targets = {
                'target_image': 'image',
                'object_mask': 'mask'
            }
        self.additional_targets = additional_targets
        super().__init__(transforms, *args, additional_targets=additional_targets, **kwargs)
        if no_nearest_for_masks:
            for t in transforms:
                if isinstance(t, DualTransform):
                    t._additional_targets['object_mask'] = 'image'

