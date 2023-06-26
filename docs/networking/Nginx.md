# Nginx

- ## #安装
	- ```bash
	  sudo apt-get update
	  sudo apt-get upgrade
	  sudo apt-get install nginx
	  
	  
	  ```
- 成功配置了反向代理镜像网站, nginx manager 的data/nginx下可以配置
  collapsed:: true
	- ```bash
	  # ------------------------------------------------------------
	  # gh.csl122.com
	  # ------------------------------------------------------------
	  
	  
	  server
	      {
	          listen 80;
	          listen 443 ssl;
	          ssl on;
	          # Custom SSL
	          ssl_certificate /data/custom_ssl/npm-3/fullchain.pem;
	          ssl_certificate_key /data/custom_ssl/npm-3/privkey.pem;
	          ssl_session_cache shared:SSL:10m;
	          ssl_session_timeout  10m;
	              proxy_ssl_server_name on;
	              proxy_ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
	          server_name gh.csl122.com;
	          add_header Strict-Transport-Security "max-age=31536000";
	          
	          if ( $scheme = http ){
	              return 301 https://$server_name$request_uri;
	          }
	          
	          if ($http_user_agent ~* (baiduspider|360spider|haosouspider|googlebot|soso|bing|sogou|yahoo|sohu-search|yodao|YoudaoBot|robozilla|msnbot|MJ12bot|NHN|Twiceler)) {
	          return  403;
	          }
	    
	          location / {
	          sub_filter github.com gh.csl122.com;
	          sub_filter_once off;
	          proxy_set_header X-Real-IP $remote_addr;
	          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
	          proxy_set_header Referer https://github.com;
	          proxy_set_header Host github.com;
	          proxy_pass https://github.com;
	          proxy_set_header Accept-Encoding "";
	          }
	  
	  }
	  
	  server {
	          listen 443 ssl; 
	      server_name gh.csl122.com;
	          return 301 https://gh.csl122.com$request_uri;
	  
	      ssl_certificate /data/custom_ssl/npm-3/fullchain.pem;
	      ssl_certificate_key /data/custom_ssl/npm-3/privkey.pem;
	  }
	  
	  
	  
	  server {
	      if ($host = gh.csl122.com) {
	          return 301 https://$host$request_uri;
	      } # managed by Certbot
	  
	  
	          listen 80;
	          listen [::]:80; 
	      server_name gh.csl122.com;
	          return 404; # managed by Certbot
	  }
	  
	  ```
- 想要所有子域名都原样跳转到对应页面
  collapsed:: true
	- ```bash
	  # ------------------------------------------------------------
	  # *.korin.eu.org
	  # ------------------------------------------------------------
	  
	  
	  server {
	    listen 80;
	    listen [::]:80;
	  
	    listen 443 ssl http2;
	    listen [::]:443 ssl http2;
	  
	  
	    server_name "~^(?!cat)(?<subdomain>.+)\.korin\.eu\.org$";
	    return 301 $scheme://$subdomain.csl122.com$request_uri;
	  
	    # Custom SSL
	    ssl_certificate /data/custom_ssl/npm-1/fullchain.pem;
	    ssl_certificate_key /data/custom_ssl/npm-1/privkey.pem;
	  
	  
	  }
	  ```
- 配置
	- `/etc/nginx/nginx.conf`中配置
		- gzip 注释掉
		- include需要的server配置
		- ```bash
		  user www-data;
		  worker_processes auto;
		  pid /run/nginx.pid;
		  include /etc/nginx/modules-enabled/*.conf;
		  
		  events {
		  	worker_connections 768;
		  	# multi_accept on;
		  }
		  
		  http {
		  
		  	##
		  	# Basic Settings
		  	##
		  
		  	sendfile on;
		  	tcp_nopush on;
		  	tcp_nodelay on;
		  	keepalive_timeout 65;
		  	types_hash_max_size 2048;
		  	# server_tokens off;
		  
		  	# server_names_hash_bucket_size 64;
		  	# server_name_in_redirect off;
		  
		  	include /etc/nginx/mime.types;
		  	default_type application/octet-stream;
		  
		  	##
		  	# SSL Settings
		  	##
		  
		  	ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3; # Dropping SSLv3, ref: POODLE
		  	ssl_prefer_server_ciphers on;
		  
		  	##
		  	# Logging Settings
		  	##
		  
		  	access_log /var/log/nginx/access.log;
		  	error_log /var/log/nginx/error.log;
		  
		  	##
		  	# Gzip Settings
		  	##
		  
		  	# gzip on;
		  
		  	# gzip_vary on;
		  	# gzip_proxied any;
		  	# gzip_comp_level 6;
		  	# gzip_buffers 16 8k;
		  	# gzip_http_version 1.1;
		  	# gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
		  
		  	##
		  	# Virtual Host Configs
		  	##
		  
		  	include /etc/nginx/conf.d/*.conf;
		  	include /etc/nginx/sites-enabled/*;
		  }
		  
		  
		  #mail {
		  #	# See sample authentication script at:
		  #	# [Using a PHP Script on an Apache Server as the IMAP Auth Backend | NGINX](http://wiki.nginx.org/ImapAuthenticateWithApachePhpScript)
		  # 
		  #	# auth_http localhost/auth.php;
		  #	# pop3_capabilities "TOP" "USER";
		  #	# imap_capabilities "IMAP4rev1" "UIDPLUS";
		  # 
		  #	server {
		  #		listen     localhost:110;
		  #		protocol   pop3;
		  #		proxy      on;
		  #	}
		  # 
		  #	server {
		  #		listen     localhost:143;
		  #		protocol   imap;
		  #		proxy      on;
		  #	}
		  #}
		  
		  ```
- 注意事项
	- `/etc/nginx/sites-enabled/`中存着default
		- 80端口聆听http请求, 443端口聆听ssl, 加ssl指令enable ssl加密
		- default_server 告诉nginx这个server block作为默认的server, 没有在别的地方指定的域名都会被导向这个server
		- 证书需要提前放到服务器中, 并且在下面指定位置
		- ```bash
		  server {
		  	listen 80 default_server;
		  	listen [::]:80 default_server;
		  
		  	# SSL configuration
		  	#
		  	listen 443 ssl default_server;
		  	listen [::]:443 ssl default_server;
		  	ssl_certificate /home/ubuntu/shiliangchen.xyz.pem;
		  	ssl_certificate_key /home/ubuntu/shiliangchen.xyz.key;
		  	#
		  	# Note: You should disable gzip for SSL traffic.
		  	# See: [#773332 - Default nginx.conf leaves sites vulnerable to BREACH - Debian Bug report logs](https://bugs.debian.org/773332)
		  	#
		  	# Read up on ssl_ciphers to ensure a secure configuration.
		  	# See: [#765782 - nginx: The sample TLS config should recommend a better cipher list - Debian Bug report logs](https://bugs.debian.org/765782)
		  	#
		  	# Self signed certs generated by the ssl-cert package
		  	# Don't use them in a production server!
		  	#
		  	# include snippets/snakeoil.conf;
		  
		  	root /var/www/html;
		  
		  	# Add index.php to the list if you are using PHP
		  	index index.html index.htm index.nginx-debian.html;
		  
		  	server_name shiliangchen.xyz;
		  
		  	location / {
		  		# First attempt to serve request as file, then
		  		# as directory, then fall back to displaying a 404.
		  		try_files $uri $uri/ =404;
		  	}
		  ```
	- `/etc/nginx/conf.d/chat3000.conf`是自定义的server block文件
		- 可以指定多个server, 例如第一个是chatgpt的, 第二个是mongo db的
		- ```bash
		  server {
		      listen 80;
		      listen [::]:80;
		      listen 443 ssl;
		      listen [::]:443 ssl;
		  
		      ssl_certificate /home/ubuntu/shiliangchen.xyz.pem;
		      ssl_certificate_key /home/ubuntu/shiliangchen.xyz.key;
		  
		      server_name shiliangchen.xyz www.shiliangchen.xyz;
		  
		      location / {
		          proxy_pass http://localhost:3000;
		          # proxy_set_header Host $host;
		          # proxy_set_header X-Real-IP $remote_addr;
		          proxy_buffering off;
		      }
		  }
		  
		  server {
		      listen 80;
		      listen [::]:80;
		      listen 443 ssl;
		      listen [::]:443 ssl;
		  
		      ssl_certificate /home/ubuntu/shiliangchen.xyz.pem;
		      ssl_certificate_key /home/ubuntu/shiliangchen.xyz.key;
		  
		      server_name mongo.shiliangchen.xyz;
		  
		      location / {
		          proxy_pass http://localhost:8081;
		          # proxy_set_header Host $host;
		          # proxy_set_header X-Real-IP $remote_addr;
		          proxy_buffering off;
		      }
		  }
		  ```
