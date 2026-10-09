# iKataGo 上游下载说明

发布包只包含我们编写的下载、部署和管理脚本，不包含 iKataGo Server 二进制、上游 FRP 配置或平台令牌。

部署时，用户的机器直接通过 HTTPS 从上游下载：
https://ikatago-resources.oss-cn-beijing.aliyuncs.com/all/linux-work.zip

项目：https://github.com/kinfkong/ikatago-server

下载器校验固定 SHA256，再读取必要文件；不执行上游安装脚本。上游替换下载内容时会停止安装，需审查后更新校验值。下载来源和哈希记录在部署目录 config/ikatago-provenance.json。

平台令牌仅在安装时从上游公开启动文件读取，不是用户账号。iKataGo 远程发现、穿透及令牌由上游维护。使用需遵守上游适用条款。直接下载避免本发布包附带上游程序副本，不代表获得任意使用、修改或再分发许可。
