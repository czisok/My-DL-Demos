"""
DIN (Deep Interest Network) with Dice activation - TF2 版
对应原 TF1 版 din/model_dice.py：FCN 中间层用 Dice 激活替代 sigmoid。
内部直接复用 model.DINModel(use_dice=True)，保证建模逻辑与原 TF1 完全一致。
"""

from model import DINModel


class Model(DINModel):
    """
    与原 TF1 model_dice.py 的 Model 类同名。
    继承 DINModel，仅将 use_dice 置为 True 以启用 Dice 激活替代 sigmoid。
    """

    def __init__(self, user_count, item_count, cate_count, cate_list,
                 predict_batch_size, predict_ads_num):
        super().__init__(
            user_count=user_count,
            item_count=item_count,
            cate_count=cate_count,
            cate_list=cate_list,
            predict_batch_size=predict_batch_size,
            predict_ads_num=predict_ads_num,
            use_dice=True,
            name='DIN_Dice',
        )
