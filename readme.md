# PROJECT: MOSAIC

> 没有使用Python3.9和Python3.10的新语法，低版本的大家可以放心使用

方便地给你的微信聊天记录长截图打码

足够鲁棒（支持负数索引，不要求倍率整除）与易用（全方位支持链式调用）

```python
(
    WeChatScreenShot
    .imread("testcases/2.jpg")
    .mosaic_title()
    .shift_and_mosaic_icons()
    .mosaic_hit_message(x1=300, y1=1810)
    .imwrite("testcase/2.jpg")
    .show()
)
```

写成一行即

```python
WeChatScreenShot.imread("testcases/2.jpg").mosaic_title().shift_and_mosaic_icons().mosaic_hit_message(x1=300, y1=1810).imwrite("testcase/2.jpg").show()
```

> 由于打字到一半电脑死机了，整个markdown保存到一半，所以就全部丢失了
> 本来写了好多，现在不想写了，所以这个readme看起来可能比较残缺，见谅

---

### advanced feature

#### `shift` 函数

- 可以变换rgb通道为别的排列
- 这使得不用特别大尺寸的马赛克也能抹掉头像特征成为可能

#### 混合多种插值方式的降采样

- 创建了一个预设，用少许最邻近采样来抹掉一些特征
- 这使得不用特别大尺寸的马赛克也能抹掉头像特征成为可能

### future feature

- 其实可以扩展 `methods` 参数，以后可以结合升降采样
- 甚至可以实现插入高斯模糊预处理的步骤
- 尚未封装给己方引用别人消息的用户名打码的功能
- 尚未实现对多种字体大小和屏幕尺寸的适配（将来可能考虑用OCR结合取中位数/平均数的方法来实现鲁棒的适配）
- 是否可以将大部分功能都自动化？真正实现**一键**打码
- 能否单独模糊每个头像？准备用 `OpenCV` 或者 `scikit-image` 中的高阶函数实现
