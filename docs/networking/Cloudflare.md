# Cloudflare

- 之前给手机里下载了一个1.1.1.1, 是可以用来建立安全链接, 配合telegram bot可以获得用不完的流量
- Cloudflare有一个zero trust服务可以用来tunnel, 可以实现特定端口应用甚至整个网络环境的内网穿透
	- [Via the command line · Cloudflare Zero Trust docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/local/)
	- [使用cloudflare tunnel打洞，随时随地访问内网服务 - yunyuyuan blog](https://blog.yunyuyuan.net/articles/5896)
- 类似于vercel和github page, cloudflare也提供托管网页的服务
	- [Cloudflare Pages](https://pages.cloudflare.com/)
- 关于[[SSL]] 的理解, cloudflare里面有edge和origin两个certificate, 相当于cf代替我们去访问origin. 我们访问cloudflare的时候用的PKI分发的公钥验证cf身份, 而cf访问origin时同样也用到了我们自己创建的私钥和公钥对, cf用公钥验证origin签名信息建立https链接
	- ![image.png](../assets/image_1687703288030_0.png)
- [[CSR]] Certificate Signing Request
	- [什么是CSR，CSR文件的作用和生成](https://www.sslchaoshi.com/help/docs/article_54)
	- 生成私钥的同时, 生成一个CSR发送给CA, 生成证书公钥文件, 是发给用户的用来验证我们自己服务器的证书
