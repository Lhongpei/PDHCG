import os
import sys
from setuptools import setup, find_packages
import subprocess
from setuptools.command.install import install

def check_julia_installed():
    """Check if Julia is installed and return its version."""
    try:
        julia_version = subprocess.check_output(["julia", "--version"], stderr=subprocess.STDOUT).decode().strip()
        print(f"✅ Julia detected: {julia_version}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Julia is not detected. Please install Julia first: https://julialang.org/downloads/")
        return False

class CustomInstallCommand(install):
    """Custom install command that checks Julia installation."""
    def run(self):
        if not check_julia_installed():
            sys.exit("❌ Installation failed: Julia is not installed. Please install Julia first.")
        super().run()
        print(logo)
        print("🚀 PDHCG has been successfully installed. If you have any problem, please contact ishongpeili@gmail.com.")

logo = r"""
    ██████╗ ██████╗ ██╗  ██╗ ██████╗ ██████╗ 
    ██╔══██╗██╔══██╗██║  ██║██╔════╝██╔════╝ 
    ██████╔╝██║  ██║███████║██║     ██║  ███╗
    ██╔═══╝ ██║  ██║██╔══██║██║     ██║   ██║
    ██║     ██████╔╝██║  ██║╚██████╗╚██████╔╝
    ╚═╝     ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ 
        An optimizer for Large Convex Quadratic Programming            
    """

setup(
    name='PDHCG', 
    version='0.0.1', 
    keywords='Optimizer',
    license='MIT', 
    author='hongpeili',
    author_email='ishongpeili@gmail.com',
    packages=find_packages(), 
    description = 'A python wrapper for PDHCG.jl',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    include_package_data = True,
    entry_points={
        'console_scripts': [
            'PDHCG=run:main' 
        ],
    },
    # cmdclass={
    #     'install': CustomInstallCommand,  # 绑定自定义安装流程
    # },
    install_requires=[
        'juliacall', 'numpy', 'scipy'
    ],
)