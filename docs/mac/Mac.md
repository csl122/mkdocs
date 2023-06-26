-
- ## locale
	- 语言环境的选项, 告诉程序我的语言环境. #locale #linux #Terminal
	- LC_ALL 会覆盖所有子选项, C是ascii, 排序会按照ascii来
	- ```bash
	  locale
	  LANG="en_GB.UTF-8"
	  LC_COLLATE="en_GB.UTF-8"
	  LC_CTYPE="en_GB.UTF-8"
	  LC_MESSAGES="en_GB.UTF-8"
	  LC_MONETARY="en_GB.UTF-8"
	  LC_NUMERIC="en_GB.UTF-8"
	  LC_TIME="en_GB.UTF-8"
	  LC_ALL="en_GB.UTF-8"
	  ```
	- ```bash
	  .zshrc中export, 避免默认值造成影响
	  export LANG=en_GB.UTF-8
	  export LC_ALL=en_GB.UTF-8
	  
	  特定情况下, 重新自定义环境变量, 让特定程序使用特定语言
	  alias gic='LC_ALL=zh_CN.UTF-8 git'
	  echo "alias git='LANG=zh_CN git'" >> ~/.zshrc
	  
	  /etc/ssh/ssh_config 文件最后
	  	Host *
	      	SendEnv LANG LC_*
	  会将本机的locale 信息发送到服务器同步, 出问题要注释掉
	  ```
	- ![3201687274244_.pic.jpg](../assets/3201687274244_.pic_1687293977119_0.jpg)