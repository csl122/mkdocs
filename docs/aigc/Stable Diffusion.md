# Stable Diffusion

tags:: Generative, AIGC, Tool, Diffusion
alias::  SD, StableDiffusion

- Paper: [[2112.10752] High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- Code: [GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model](https://github.com/CompVis/stable-diffusion)
	- [GitHub - Stability-AI/stablediffusion: High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/Stability-AI/stablediffusion)
	- ### ⬇️Latent Diffusion Models
	  collapsed:: true
		- [Latent Diffusion Models](https://nn.labml.ai/diffusion/stable_diffusion/latent_diffusion.html)
		- 首先是整个大类的定义, 需要UNet, AutoEncoder, CLIPTextEmbedder这几个模型; latent scaling factor用来scale encoder encode的encodings, 再输入到UNet里面, n_steps表示我们实际要进行的diffusion steps, 两个linear是方差beta的schedule起始和结束的位置. Beta会在初始化中定义好, alpha也会定义好为之后做准备
		- ![image.png](../assets/image_1683550000918_0.png)
		- Text conditioning 是用CLIP模型得到prompts string list以后得到的. Encode函数会把图片给到autoencoder去得到一个distribution, 然后我们从中sample一个, 但是最后还要乘上一个scaling  factor. Decode函数会把反向diffusion sample到的z除以之前scale的参数给解码得到图片. 整个模型的forward就是调用UNet来预测该时间点前所使用的epsilon
	- ### ⬇️Autoencoder for Stable Diffusion
	  collapsed:: true
		- [Autoencoder for Stable Diffusion](https://nn.labml.ai/diffusion/stable_diffusion/model/autoencoder.html)
		- This implements the auto-encoder model used to map between image space and latent space.
		- 这里的autoencoder有两种 embedding space. 一个是embedding space, 一个是quantised embedding space. 这里是因为SD使用了VQ-reg, 用vector quantisation来实现正则, 体现在代码中就使用一个11conv来变换了维度, (真正的VQ应该还会quantise函数调用, 从codebook里找到最接近的) 同时这个类其实是AutoencoderKL, 是一个VAE, 因此还从中采样得到一个posterior. 各自有各自的channels数量, 同时他们会有一半的channels是表示mean, 一半表示vairance
		- ![image.png](../assets/image_1683551207842_0.png)
		- Encode 和 Decode过程包括首先使用encoder得到latent vector z, 再对z quantise到 quantised embedding space作为moments, 然后利用这个mean和variance来sample; decode则是得到quantised的z 对他进行反向映射回embedding space, 然后decode到图片
		- ![image.png](../assets/image_1683551659246_0.png)
		- Encoder module 负责把给到的图片一点点缩小, 提取到最需要的特征作为latent z, 尺寸变小的过程中, channels数会越来越多, channel的增长是由channel_multipliers控制的
		- ![image.png](../assets/image_1683551797292_0.png)
		- Encode过程中, 分辨率的变化取决于有多少个channel_multipliers, 即要变化多少次channel, 每次都会减半分辨率. 首先使用一个3311的conv2d不改变分辨率但是把channel数先变成第一个需要的channels数量. 对于每一个分辨率, 都会有数个定义好数量的resnet_block用来提取特征, 这个resnet list会首先把当前channel变成下一轮需要的channel数量, 然后后面几个resnet就保持channels不变了. 每个分辨率都会有这么一组resnet, 然后在最后一次downsample到一半的分辨率. 最后还会有mid blocks, 不改变分辨率和通道, 利用了两个resnet中间夹了一个attention. 最后normalise了一下然后用11卷积map到了需要的mean+variance embedding channel
		- ![image.png](../assets/image_1683551826779_0.png)
		- Encoder的前向过程包括先把图片变成初始channel, 然后对于多个分辨率, 用多个resnet处理以后downsample, 最后经过mid处理 attention最后conv到mean+variance 的 embeeding space输出
		- ![image.png](../assets/image_1683552635663_0.png)
		- Decoder 模块由于没有使用convTranspose2D, 所以整个过程就是反转一下encode的过程
		- ![image.png](../assets/image_1683552937749_0.png)
		- ![image.png](../assets/image_1683552948898_0.png)
		- Decoder的前向过程就是, 先把z变换成最后的那个channel数, 然后经过mid blocks attention然后再提升分辨率的过程中, 经过一系列的resnet以及upsample, 最后norm一下再map到图片的3channel
		- ![image.png](../assets/image_1683552966463_0.png)
		- Gaussian Distribution是得到quantised embedding 以后利用其mean和variance来从中sample样本用的
		- ![image.png](../assets/image_1683553206179_0.png)
		- Attention block 实现的是把沿着channel维度的像素们作为sequence, 一个feature map是一个sequence, 一共有像素个词, 每个像素词的embedding长度都是channels数.  torch.einsum做的也是最后生成ixj的attention matrix, 最后沿着j维做横向的softmax得到加权百分比, 乘上各自的v最后得到att完了的全局att的v. 当然最后还有额外的11conv 和 residual connection.
		- ![image.png](../assets/image_1683553376688_0.png)
		- Upsampling 和Downsampling的实现比较简单. Up是先直接使用interpolate nearest来2倍放大, 再用不改变分辩率的conv调整. Down的话在下面和右边填充了0(因为要pad1而不是两边都pad2才能实现刚好/2), 然后用3320conv变小
		- ![image.png](../assets/image_1683555043724_0.png)
		- ResNet Block的实现也是简单的, 定义好in和out Channels后, norm, conv(in, out), norm, conv(out, out), 以及shortcut就行了(shortcut需要考虑到inout不同的情况就要11conv调一下). 这里注释一下比较神奇的是act在conv前面
		- ![image.png](../assets/image_1683555396439_0.png)
	- ### ⬇️U-Net for Stable Diffusion
	  collapsed:: true
		- [U-Net for Stable Diffusion](https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html)
		- 这个U-Net用到了attention, 是个spatialTransformer, 其构造和DDPM里面用到的很像, 可以去参考那边的也. 多的两个参数是tf
		- ![image.png](../assets/image_1683559407638_0.png)
		- 下面这个部分定义了一下U-Net有多少个分辨率level, 造了一个timeembedding的MLP, 然后初始化了input_blocks作为左半边的ModuleList. TimestepEmbedSequential包裹住了第一个用来对齐channel用的conv2d, 原因是这个类可以自动识别里面是什么类型的function, 来自动分配所需要的feature map, 或是fm以及time embedding
		- ![image.png](../assets/image_1683561530524_0.png)
		- 下面构造了U-Net的左半边, 一共有level个分辨率层, 每个分辨率层有n_res_block个resnet_block, 其中如果这里是需要用attention的话, 就会在ResBlock后面append上一个attention block, 这里是SpatialTransformer. 然后在最后一层进行downsample
		- ![image.png](../assets/image_1683561517004_0.png)
		- 中间层由res, ST, res组成
		- ![image.png](../assets/image_1683562177686_0.png)
		- U-Net的右半边就是左半边反一下
		- ![image.png](../assets/image_1683562448108_0.png)
		- Time embedding和DDPM里的差不多
		- ![image.png](../assets/image_1683562491476_0.png)
		- 前向过程就是左半边input_blocks里面的module一个个处理x, 然后并且记录下来, 用作右边加上去的skip connection. 注意input和output_blocks里的每一个module都会输入到t_emb, 也就是说里面的resblock都会用到t_emb
		- ![image.png](../assets/image_1683562507508_0.png)
		- Sequential block for modules with different inputs 可以接收x, t, cond三种, 但是选择性地apply进其中的module中作为输入, 解决了不同signature的对齐输入问题
		- ![image.png](../assets/image_1683562972761_0.png)
		- UP和Down sample和autoencoder类似
		- ![image.png](../assets/image_1683563418703_0.png)
		- ResNet Block 也大同小异, 定义了in和out layers, 然后是skip_connection
		- ![image.png](../assets/image_1683563631674_0.png)
		- forward过程就是先in, 加time embedding, 是一个channel一个值, 广播到这个channel所有像素, 然后out, 最后skip
		- ![image.png](../assets/image_1683563750022_0.png)
	- ### ⬇️Transformer for Stable Diffusion
	  collapsed:: true
		- [Transformer for Stable Diffusion U-Net](https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html)
		- Channels就是每个词的feature dim, 也作为模型的dimension, 多头注意力因此有n_heads, n_layers define the number of transformer layers, d_cond定义了conditional embedding的大小. 其中又定义了1x1的卷积用来处理进入的tensor, 以及堆砌了n_layers层的transformer block, 最后再加上一个线性变换, 只不过也用conv来进行了
		- ![image.png](../assets/image_1683565314494_0.png)
		- Forward部分, spatialTransformer 只利用到了前一步resblock处理完的x和condition, 先对x norm 再线性变换, 再把形状变换成transformer能够处理的形式[batch, sequence, embedding], 然后输入到transformer_blocks中经过多层transformer处理, 最后变回原来的模样, 并且加一个residual
		- ![image.png](../assets/image_1683567792282_0.png)
		- Transformer layer的basic block用的是cross attention, 但是在没有cond的情况下, 就会变成self-attention. 这个basic block由两个attention层组成, 第一个固定进行self-attention, 第二个则是会考虑到cond, 做的是cross-attention, 最后再有一个FFN. 和最普通的transformer非常类似, 类似的是decoder部分, 也是一个selfattention, cross attention, ffn, 然后每一层做完都会有add & norm
		- ![image.png](../assets/image_1683568807657_0.png){:height 583, :width 704}
		- CrossAttention的定义主要还是包含了和传统transformer一样的部分, 有三个FC来map到qkv. 其中需要注意的是attention的dimension是head*d_head, 也就是model dimension
		- ![image.png](../assets/image_1683569737535_0.png)
		- CrossAttention的forward过程就是先把x和cond变成qkv, 然后进行常规的attention操作, 这里把attention的具体过程定义到了一个另外的函数里面, 包括normal_attention 和另一个flash attention(会更快一点)
		- ![image.png](../assets/image_1683570003257_0.png)
		- FlashAttention 会比较快一点
		- ![image.png](../assets/image_1683570030325_0.png)
		- Normal Attention, 由于之前的qkv的fc map到的是n_heads x d_head大小的向量, 因此这里要把每个head的给分开来, 从 [batch_size, seq, d_attn] ->[batch_size, seq_len, n_heads, d_head], 然后生成的attention matrix也要是b个h个seq*seq. softmax后乘v得到最终的seq个h个v, 这里还是d_attn, 最后有一个mlp map回d_model来使得可以放到下一个attention里面
		- ![image.png](../assets/image_1683570050941_0.png){:height 506, :width 704}
		- FFN, 对每个sequence里的词作同样的上下变换
		- ![image.png](../assets/image_1683570065620_0.png)
	- ### ⬇️Base class for sampling algorithms
	  collapsed:: true
		- [Sampling algorithms for stable diffusion](https://nn.labml.ai/diffusion/stable_diffusion/sampler/index.html)
		- 下面是基础Stable Diffusion sampler的构造模式
		- ![image.png](../assets/image_1683499500426_0.png)
		- 首先有一个函数用于获取模型估计的epsilon, 输入的是x, t, c, CFG scale, 和uncond的cond例如None. x和t都会duplicate两份, 传入模型中获取con的估计的epsilon和uncon估计的epsilon, 然后由CFG公式得到最后的epsilon, 即指向x0的方向.
		- ![image.png](../assets/image_1683499820490_0.png)
		- Sampling 和 painting的loop 定义是下图这样, 具体在[DDIM中有解释过](logseq://graph/main?block-id=6457a8f0-5565-4699-8b98-28f4627d02e9)
		- ![image.png](../assets/image_1683500002278_0.png)
		-
	- ### ⬇️Stable Diffusion Code的全局认知
	  collapsed:: true
		- Stable Diffusion在各个元件的实现上有一些细节上的差异
		- 需要UNet, AutoEncoder, CLIPTextEmbedder这几个模型;
		- latent scaling factor用来scale encoder encode的encodings, 再输入到UNet里面
		- Autoencoder 得到的latent z也是多channel的, 且设定上z有dim_z, 实际上会给到2*dim_z用来分别容纳mean和variance; 而且这个z还不是最终给到的latent embedding, 还要经过quantisation到quantised embedding space, 并且还要根据这个quantised embedding space 的mean和variance再sample 一个sample, 这个才是用来diffusion的东西
		  collapsed:: true
			- ![image.png](../assets/image_1683586867150_0.png)
		- U-Net也是一个大改过的模型. 与DDPM相似的地方在于, 两个模型的结构写法都非常相似, 都是左半边, 中间, 右半边(几乎是左半边的反转版). 每个半边的循环都是resolution level为主要循环, 每个循环里有定义次数的resblock, 如果启用了attention的话就会在每个resblock后面紧跟着一个attention模块, 只不过两个版本的attention模块有点不一样. DDPM的attention是完全的selfattention, 每个像素通道作为一个词, 词向量长度就是channel数, 像素们作self-attention. 而SD中, attention化身为了spatialTransformer, 类似于传统的transformer decoder, 作为一整个模块出现, 而不是单单只是做一次attention就结束了, 包含了一次self-attention, 一次cross-attention(与cond做), 以及一个FFN, 基本上和transformer decoder一模一样
		  collapsed:: true
			- ![image.png](../assets/image_1683586811075_0.png)
		- Autoencoder只是用来转换成潜在空间, U-Net是用来预测epsilon, 真正sampling地时候是可以使用不同的sampler的, sampler会从训练好的模型(LatentDiffusion model, 包含了定义好的总diffusion步数, beta, alpha等参数, 以及三个主要模型unet, autoencoder, clip)处得到定义好的总diffusion步数, beta, alpha等参数, 再用各个sampler的算法如DDPM, DDIM等来进行sample
	- ### ⬇️Text2Img
	  collapsed:: true
		- [Generate images using stable diffusion with a prompt](https://nn.labml.ai/diffusion/stable_diffusion/scripts/text_to_image.html)
		- 读入模型, 和sampler. 用了utils里面的load_model函数, 这个函数里面加载了所有需要的sub model比如说autoencoder的encoder和decoder类, clip和unet, 然后实例化了一个latentDiffusion model其中包含了这些部分, 加载的statedict也是包含了这些子模型的参数的
		- ![image.png](../assets/image_1683635995838_0.png)
		- call的时候需要给到 文件路径, 一次生成多少张图片(batch_size), prompt, 以及图片宽高数据和CFG scale. 有多少batch就copy多少个prompt来重复生成, CFG scale也在这里处理到, 大于1会有强guidance, 等于1的时候就是单单cond, uncon这里是“”. 用sampler sample就得到了latent space的去噪结果. 这里注意有一个很神奇的地方就是f是Image to latent space resolution reduction, 也就是说我们提前知道了img到latent会有八倍缩放, 所以我们用来sample地noise就是长宽除以f的大小, autoencoder会负责把1/8的latent恢复到原始的尺寸, 是自适应的, 由于是全卷积, autoencoder和unet都是自适应的
		- ![image.png](../assets/image_1683636012870_0.png)
		- 主函数
		- ![image.png](../assets/image_1683636059927_0.png)
	- ### ⬇️Img2Img
	  collapsed:: true
		- [Generate images using stable diffusion with a prompt from a given image](https://nn.labml.ai/diffusion/stable_diffusion/scripts/image_to_image.html)
		- 与上面类似的初始化
		- ![image.png](../assets/image_1683636252100_0.png)
		- call中增加了original image和strength, 也就是一个0-1的比例, 要从diffusion的第几步开始denoise 加噪的原图, 与之前的区别在于, 要用q_sample获取一个初始的noised original image, 然后把它放到paint里面sample
		- ![image.png](../assets/image_1683636546684_0.png)
		- 主函数
		- ![image.png](../assets/image_1683636577050_0.png)
	- ### ⬇️Impaint
	  collapsed:: true
		- [In-paint images using stable diffusion with a prompt](https://nn.labml.ai/diffusion/stable_diffusion/scripts/in_paint.html)
		- 在i2i的基础上多一个mask传入paint函数
		- ![image.png](../assets/image_1683636699427_0.png)
	- ### ⬇️Utils
	  collapsed:: true
		- ```def load_model(path: Path = None) -> LatentDiffusion:``` 实例化几个模型, 载入weights
		  		- ![image.png](../assets/image_1683636887407_0.png)
		  		- ```python
		  def load_img(path: str):
		      """
		      ### Load an image
		  
		      This loads an image from a file and returns a PyTorch tensor.
		  
		      :param path: is the path of the image
		      """
		      # Open Image
		      image = Image.open(path).convert("RGB")
		      # Get image size
		      w, h = image.size
		      # Resize to a multiple of 32
		      w = w - w % 32
		      h = h - h % 32
		      image = image.resize((w, h), resample=PIL.Image.LANCZOS)
		      # Convert to numpy and map to `[-1, 1]` for `[0, 255]`
		      image = np.array(image).astype(np.float32) * (2. / 255.0) - 1
		      # Transpose to shape `[batch_size, channels, height, width]`
		      image = image[None].transpose(0, 3, 1, 2)
		      # Convert to torch
		      return torch.from_numpy(image)
		  ```
		- ```python
		  def save_images(images: torch.Tensor, dest_path: str, prefix: str = '', img_format: str = 'jpeg'):
		      """
		      ### Save a images
		  
		      :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
		      :param dest_path: is the folder to save images in
		      :param prefix: is the prefix to add to file names
		      :param img_format: is the image format
		      """
		  
		      # Create the destination folder
		      os.makedirs(dest_path, exist_ok=True)
		  
		      # Map images to `[0, 1]` space and clip
		      images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
		      # Transpose to `[batch_size, height, width, channels]` and convert to numpy
		      images = images.cpu().permute(0, 2, 3, 1).numpy()
		  
		      # Save images
		      for i, img in enumerate(images):
		          img = Image.fromarray((255. * img).astype(np.uint8))
		          img.save(os.path.join(dest_path, f"{prefix}{i:05}.{img_format}"), format=img_format)
		  ```
- Explained:
	- [Stable Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models | ML Coding Series - YouTube](https://www.youtube.com/watch?v=f6PtJKdey8E)
	- [Stable Diffusion Clearly Explained! | by Steins | Medium](https://medium.com/@steinsfu/stable-diffusion-clearly-explained-ed008044e07e)
- Info
  collapsed:: true
	- 一开始被称为Latent Diffusion Models. use an auto-encoder to map between image space and latent space. The diffusion model works on the latent space, which makes it a lot easier to train.
	- 自编码器AutoEncoder和文字编码器CLIP都是预训练好的, 且是配套的, 其中的U-Net with attention需要现场训练
	- ![image.png](../assets/image_1683122530780_0.png)⬇️原图展开可见
	  collapsed:: true
		- ![image.png](../assets/image_1682197752988_0.png)
		-
	- 和之前diffusion models 的不同在于, diffusion过程在latent space中进行, 而之前DDPM是在pixel space中进行的. 减少了计算资源的需求. 先将x encode到latent space, 所有的diffusion 前向和后向过程都是在latent space中进行的
		- 问题: 编码和解码是用什么模型做的, 是不是一个预训练的模型
		- 答案: 是的, VAE是针对特定数据集训练的. 特定模型需要特定的VAE
	- ZT和一些conditioning的东西, 例如说text, images这些编码后的向量一起作为denoising的输入, 输入到Denoising U-Net中.
	- 这里的Denoising U-Net是专门设计的带有attention的类transformer的U-Net. ZT各阶段应该会作为Q, 一方面被U-Net up down, 一方面与conditioning 进行cross attention 操作, 来获取condition的信息, 对输出加以限制
	- 问题: How img2img diffusion works
		- 答案: 例如50步的inference steps, Image/Noise Strength Parameter 会控制加百分之多少的noise到原图中, 例如0.8就是加40步的noise到原图中, 然后inference剩余的40步
		- 注意, 1仅仅表示加了50步noise, 并不会完全不保留原图的信息
		- Reference: [How img2img Diffusion Works · Chris McCormick](https://mccormickml.com/2022/12/06/how-img2img-works/)
	-
- WebUI
	- [GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
	- [Home · AUTOMATIC1111/stable-diffusion-webui Wiki · GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) #GitHub #安装 #SD #WebUI
	- [GitHub - Mikubill/sd-webui-controlnet: WebUI extension for ControlNet](https://github.com/Mikubill/sd-webui-controlnet)
	  id:: 64526dd6-022b-42cd-bdb3-3844bc23ae3e
	- Prompt
	  collapsed:: true
		- ![image.png](../assets/image_1683305697387_0.png)
		- 正向提示语：
		  collapsed:: true
			- 万能画质要求
			  (masterpiece, best quality),
		- 反向提示语：
		  collapsed:: true
			- 避免糟糕人像的
			  ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy,disfigured, poorly drawn face, mutation, mutated, (extra_limb),(ugly), (poorly drawn hands fingers), messy drawing, morbid,mutilated, tranny, trans, trannsexual, [out of frame], (bad proportions),(poorly drawn body), (poorly drawn legs), worst quality, low quality,normal quality, text, censored, gown, latex, pencil,
			  
			  避免生成水印和文字内容
			  lowres, bad anatomy, bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worst quality, low quality,normal quality, jpeg artifacts, signature, watermark, username, blurry,
			  
			  通用
			  lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,
			  
			  避免变形的手和多余的手
			  extra fingers,fused fingers,too many fingers,mutated hands,malformed limbs,extra limbs,missing arms,poorly drawn hands,
	- Sampler
	  collapsed:: true
		- ![image.png](../assets/image_1683305937523_0.png)
- 教程
	- [Home · AUTOMATIC1111/stable-diffusion-webui Wiki · GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)
	- 【SD教程·Stable Diffusion本地部署教学 官方WebUI-哔哩哔哩】 [SD教程·Stable Diffusion本地部署教学 官方WebUI_哔哩哔哩_bilibili](https://b23.tv/onRyxAZ)
	- https://mp.weixin.qq.com/s/8czNX-pXyOeFDFhs2fo7HA
	- [（上）【一文掌握Al绘图】Stable Diffusion 绘图喂饭教程](https://ki6j1b0d92h.feishu.cn/wiki/wikcnPdHiLM91LbcfPnyaFUjjOe)
	- [（下)【一文掌握A绘图】Stable Diffusion 绘图喂饭教程](https://ki6j1b0d92h.feishu.cn/wiki/wikcnA89UF8N78ZlgKudMq0hhyb)
	- [Home · AUTOMATIC1111/stable-diffusion-webui Wiki · GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)
- Training
	- [Train a diffusion model](https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training)
	- [ControlNet](https://huggingface.co/docs/diffusers/main/en/training/controlnet)
	- [Create a dataset for training](https://huggingface.co/docs/diffusers/main/en/training/create_dataset)
	-
- 模型
  collapsed:: true
	- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-image&sort=downloads)
	- [AI绘画模型博物馆](https://aimodel.subrecovery.top/)
	- [SD - WebUI 资源站](https://www.123114514.xyz/models)
	- [Civitai | Stable Diffusion models, embeddings, LoRAs and more](https://civitai.com/)
	- model
		- [CompVis/stable-diffusion-v-1-4-original · Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
		- [runwayml/stable-diffusion-v1-5 · Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5)
		- [stabilityai/stable-diffusion-2 · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2)
		- [model](https://qg0r-my.sharepoint.com/:f:/g/personal/autumnmoon_qg0r_onmicrosoft_com/Euj8bkOwhX1BlRbmNI9KLasBGzZ2Vf0Bu8Ab1smjRYSf1g?e=sMzegC)
		- [coreml (Core ML Models)](https://huggingface.co/coreml) Apple优化的模型
	- VAE
		- [stabilityai/sd-vae-ft-ema-original · Hugging Face](https://huggingface.co/stabilityai/sd-vae-ft-ema-original)
	- embedding
		- [GitHub - autumn-moon-py/aimodel-embeddings: 一个收集embeddings的仓库](https://github.com/autumn-moon-py/aimodel-embeddings)
	- lora
		- [lora](https://qg0r-my.sharepoint.com/:f:/g/personal/autumnmoon_qg0r_onmicrosoft_com/Eqg6D0C3KIJAt62eHHai9tEB8th1m8SLRGk0Fx7EkB_kqw?e=AOTlct)
	- controlnet
		- [GitHub - Mikubill/sd-webui-controlnet: WebUI extension for ControlNet](https://github.com/Mikubill/sd-webui-controlnet)
		- [GitHub - lllyasviel/ControlNet: Let us control diffusion models!](https://github.com/lllyasviel/ControlNet)
		- [lllyasviel/ControlNet-v1-1 at main](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main)
		- [TencentARC/T2I-Adapter at main](https://huggingface.co/TencentARC/T2I-Adapter/tree/main/models)
	- Upscale ([[ESRGAN]])
		- [Model Database - Upscale Wiki](https://upscale.wiki/wiki/Model_Database#Universal_Models)
	- Tiled 低显存生成高分辨率
		- [GitHub - pkuliyi2015/multidiffusion-upscaler-for-automatic1111: Tiled Diffusion and VAE optimize, licensed under CC BY-NC-SA 4.0](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)
- 其他工具和资源
	- [GitHub - Akegarasu/stable-diffusion-inspector: read pnginfo in stable diffusion generated images / inspect models](https://github.com/Akegarasu/stable-diffusion-inspector) #StableDiffusion #Tool #GitHub #Prompt 读取png的信息
		- [Stable Diffusion 法术解析](https://spell.novelai.dev/)
	- [GitHub - pharmapsychotic/clip-interrogator: Image to prompt with BLIP and CLIP](https://github.com/pharmapsychotic/clip-interrogator) #StableDiffusion #Tool #GitHub #Prompt
		- [methexis-inc/img2prompt – Run with an API on Replicate](https://replicate.com/methexis-inc/img2prompt)
	- [Stable Diffusion prompt Generator - promptoMANIA](https://promptomania.com/stable-diffusion-prompt-builder/)
	- https://github.com/camenduru/controlnet-colab
	- https://github.com/camenduru/stable-diffusion-webui-colab
	- [GitHub - godly-devotion/MochiDiffusion: Run Stable Diffusion on Mac natively](https://github.com/godly-devotion/MochiDiffusion) #GitHub #SD #Mac #app/创作软件
	- https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/fast-kohya-trainer.ipynb#scrollTo=gZ1WxKjp_1-L
-
