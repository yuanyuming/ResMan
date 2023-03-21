# Tricks

## Python Venv

### 1.创建

```
python -m venv 虚拟环境名称
```

### 2.激活

```
虚拟环境名称\scripts\activate.ps1
```

`setting.json`中添加

```json
"python.terminal.activateEnvInCurrentTerminal": true
```

完成开启工作区自动加载虚拟环境

### 3.迁移

更改 `pyenv.cfg`文件中的路径

如果安装的包有环境依赖,则需要重新安装

```
python -m pip install --force-reinstall pip
```

目前发现路径依赖的是 `pip`
python3.8 -m venv --upgrade <path_to_dir>
