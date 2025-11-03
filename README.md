# My First Project

vscode
vscode-copilot
github
deepseek
qq email

This repo is based on robosuite, bddl, LIBERO and robomimic.

1. v0.0.0: 建立初步模型，以及分阶段训练逻辑

2. v0.0.1: 修改模型，试验模糊量状态空间假设，尝试了一些初步修改（主要在highlevel模型上）：

- 20251101201547：修改了模型输入，原本只有图像模态输入的单链pipe，robot0_eef_pos并不输入
- 20251102003057：修改了模型输入，原本只有图像模态输入的单链pipe，robot0_eef_pos并不输入，并在decoder之前拼接transformer特征后输入其中
- 20251102010811：修改了模型输入，原本只有图像模态输入的单链pipe，robot0_eef_pos并不输入，在transformer和decoder之间加入了一个mlp，拼接transformer特征后输入其中
- 20251102115409：相比于上一次学习，增加了特定epoch位置的学习率衰减，应对loss曲线均匀震荡的收敛问题。

> 上述训练问题在于缺少训练经验，对训练过程中的loss曲线趋势没有认识（修改前没有先复现一遍原模型），都只训练了200-300个epoch就结束，不知道模型复杂度与loss平滑性的经验规律如何。
> 同时也缺少对应复杂模型训练的解决技巧。
