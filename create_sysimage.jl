# compile.jl
using PackageCompiler
using PDHCG
using Pkg
Pkg.develop(path="PDHCG")
# precompile.jl
open("precompile.jl", "w") do f
    write(f, """
    using PDHCG
    qp = PDHCG.generateProblem("randomqp")
    PDHCG.pdhcgSolve(qp)
    """)
end


create_sysimage(
    ["PDHCG"],
    sysimage_path="PDHCG_sysimage.so",
    precompile_execution_file="precompile.jl"
)