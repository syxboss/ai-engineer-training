# 第 6 周 - DSL语言设计与执行引擎

本项目包含一系列用于学习 DSL语言设计与执行引擎 的 Jupyter notebook 文件。

## 开始使用

以下说明将帮助您在本地机器上设置和运行此项目，以便进行开发和学习。

### 环境要求

*   Python 3.11 或更高版本
*   已安装 [uv](https://github.com/astral-sh/uv)。`uv` 是一个极速的 Python 包安装工具。

### 安装步骤

1.  **安装依赖:**
    在项目根目录中打开终端，然后运行：
    ```bash
	cd week06
	pip install uv
    uv sync --locked
    ```
    这将自动安装依赖并在当前目录下创建一个名为 `.venv` 的虚拟环境目录。

2.  **激活虚拟环境:**
    *   在 macOS 和 Linux 上:
        ```bash
        source .venv/bin/activate
        ```
    *   在 Windows 上:
        ```bash
        .venv\Scripts\activate
        ```

## 设置项目专属的 Jupyter 内核

为了确保您的 notebook 使用本项目定义的特定 Python 环境和依赖项，您可以将其注册为自定义的 Jupyter 内核。

1.  **激活虚拟环境:**
    首先，请确保您已经激活了项目的虚拟环境。
    ```bash
    source .venv/bin/activate
    ```

2.  **注册内核:**
    运行以下命令，将当前环境注册为一个新的 Jupyter 内核：
    ```bash
    python -m ipykernel install --user --name=week06 --display-name="AI工程化(week06)"
    ```

	运行下面的命令查看当前的 kernel 列表：
	```bash
	jupyter kernelspec list
	```
	应该能看到类似下面的输出:
	```bash
	Available kernels:
	week06     /Users/your_username/Library/Jupyter/kernels/week06
	python3    /usr/local/share/jupyter/kernels/python3
	```
	如果看到 `week06` 在列表中，则说明注册成功。
	

## 运行 JupyterLab

安装完成后，您可以运行 JupyterLab。

1.  **启动 JupyterLab:**
    在您的终端中（确保虚拟环境仍处于激活状态），运行：
    ```bash
    jupyter lab
    ```
    这将启动 Jupyter 服务，并在您的默认网络浏览器中打开一个新标签页。

2.  **打开 Notebook 文件:**
    在浏览器标签页中，单击任何 `.ipynb` 文件以打开并运行它。

3.  **选择内核:**
    可以在 **Kernel > Change kernel** 菜单中看到并选择 **"AI工程化(week06)"**。这可以确保您的 notebook 在正确的项目环境中运行。
