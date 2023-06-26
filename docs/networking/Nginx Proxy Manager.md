# Nginx Proxy Manager

- #安装
	- ```bash
	  注意要使用最新版本的docker-compose, 不然不适配
	  首先把下面的文件变成 docker-compose.yml
	  version: '3.8'
	  services:
	    app:
	      image: 'jc21/nginx-proxy-manager:latest'
	      restart: always
	      ports:
	        - '80:80'
	        - '81:81'
	        - '443:443'
	      volumes:
	        - ./data:/data
	        - ./letsencrypt:/etc/letsencrypt
	  
	  然后up它 
	  docker-compose up -d
	  ```
- #manual
	- [127.0.0.1](http://127.0.0.1:81) 为默认访问点
	- [https://nginx.shiliangchen.xyz](https://nginx.shiliangchen.xyz/)可以把自己反代一下直接访问
	- ![image.png](../assets/image_1680473888087_0.png){:height 436, :width 382}
	- ![image.png](../assets/image_1680473899779_0.png){:height 324, :width 412}
	- proxy_buffering off;
	-
