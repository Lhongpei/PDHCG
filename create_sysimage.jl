# compile.jl
using PackageCompiler
using PDHCG
using Pkg
create_sysimage(
    ["PDHCG"],
    sysimage_path="PDHCG_sysimage.so",
    precompile_execution_file="precompile.jl"
)