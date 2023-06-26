# Tmux

tags::  命令行, unix, Mac, Tool

- （1）它允许在单个窗口中，同时访问多个会话。这对于同时运行多个命令行程序很有用。
  （2） 它可以让新窗口"接入"已经存在的会话。
  （3）它允许每个会话有多个连接窗口，因此可以多人实时共享会话。
  （4）它还支持窗口任意的垂直和水平拆分。
- Manual #manual
	- [Tmux 使用教程 - 阮一峰的网络日志](https://www.ruanyifeng.com/blog/2019/10/tmux.html)
	- 需要设置mouse on 来开启滚动
		- echo "set -g mouse on" >> ~/.tmux.conf
		- set -g mouse on
		- [how do i scroll](https://superuser.com/questions/209437/how-do-i-scroll-in-tmux)
			- https://superuser.com/questions/209437/how-do-i-scroll-in-tmux
	- 禁止自启动
		- touch ~/.no_auto_tmux
	- ```
	  tmux new -s <session-name>
	  tmux attach -t 0
	  tmux kill-session -t 0
	  tmux switch -t 0
	  tmux detach
	  ```
- References
	- [GitHub - tmux/tmux: tmux source code](https://github.com/tmux/tmux)
	- [Tmux Cheat Sheet & Quick Reference](https://tmuxcheatsheet.com)
	-
-
-
