import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties, FontManager

"""
# 1.中文显示问题
## Linux中可以用：fc-list :lang=zh 查看所有中文字体，选择一个字体，例如：Source Han Serif CN Heavy
## Mac中常用中文：["PingFang", "Heiti", "STHeiti", "Arial Unicode", "Songti", "STSong", "Hiragino Sans GB"]
# 2. plt中文设置
## 2.1 直接设置字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"] # 字体可调换
plt.rcParams["axes.unicode_minus"] = False # 解决负号"-"显示方框问题

## 2.2 遇到各种原因就是不能使用系统字体，只能用otf字体，用下面方法
# 字体路径: sudo wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip
otf_path = "/usr/share/fonts/SourceHanSerifCN-Heavy.ttf"
# 加载字体并注册
fm = FontManager()
fm.addfont(otf_path)
# 提取字体名并设为全局字体
fp = FontProperties(fname=otf_path)
font_name = fp.get_name()
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False
"""

plt.rcParams["font.family"] = ["Heiti TC"] # 字体可调换
plt.rcParams["axes.unicode_minus"] = False # 解决负号"-"显示方框问题

def show_image(img_path, title="img", figsize=None):
    """
    给定图片路径，展示图片
    """
    # 1. 读取图片（替换为你的图片路径）
    img = mpimg.imread(img_path)
    if figsize is not None:
        plt.figure(figsize=figsize)
    # 2. 显示图片
    plt.imshow(img)
    plt.axis("off")  # 关闭坐标轴
    plt.title(title)
    plt.show()


def show_image_by_array(img, title="img", figsize=None):
    """
    给定array数据的图片，展示图片
    img: [H, W, C]
    """
    if figsize is not None:
        plt.figure(figsize=figsize)
    # 2. 显示图片
    plt.imshow(img)
    plt.axis("off")  # 关闭坐标轴
    plt.title(title)
    plt.show()


def show_multi_image(imgs, h, w, titles=[], figsize=None):
    """
    给定多张array类型的图片，即list(array)，展示图片
    """
    if figsize is None:
        figsize = (15, 3)
    if titles is None or len(titles) != len(imgs):
        titles = ["img_%d" % i for i in range(len(imgs))]
    fig, axes = plt.subplots(
        h, w, figsize=figsize
    )  # 创建一个h行w列的图形，大小为15x3英寸

    for ax, img, title in zip(axes, imgs, titles):
        # 将图片从张量格式转换为matplotlib可以显示的格式（即去除非法值，并调整通道顺序）
        # img = img.numpy().transpose((1, 2, 0))  # 将通道从CHW转换为HWC
        ax.imshow(img)  # 在子图上显示图片
        ax.axis("off")  # 不显示坐标轴
        ax.set_title(title)
    plt.tight_layout()  # 自动调整间距，防止重叠
    plt.show()
