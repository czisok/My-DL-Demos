"""
DIN with Dice Activation - PyTorch 版
对应原 TF1 din/model_dice.py：FCN 中间层用 Dice 激活替代 sigmoid。
继承 DINModel，仅将 use_dice=True。
"""

from model import DINModel


class Model(DINModel):
    def __init__(
        self,
        user_count: int,
        item_count: int,
        cate_count: int,
        cate_list,
        predict_batch_size: int,
        predict_ads_num: int,
        device=None,
    ):
        super().__init__(
            user_count=user_count,
            item_count=item_count,
            cate_count=cate_count,
            cate_list=cate_list,
            predict_batch_size=predict_batch_size,
            predict_ads_num=predict_ads_num,
            use_dice=True,
            device=device,
        )
