osgm_band_update_path = joinpath(KERNEL_DIR, "osgm_band_update_kernel.ptx")
osgm_band_update_func_name = "osgm_band_update"

osgm_band_update_module = CuModule(read(osgm_band_update_path))
osgm_band_update_kernel = CuFunction(osgm_band_update_module, osgm_band_update_func_name)

function osgm_band_update(
    prev_grad::CuArray{Float64, 1},
    grad::CuArray{Float64, 1},
    adagrad_state::CuArray{Float64, 2},
    osgm_state::CuArray{Float64, 2},
    x::CuArray{Float64, 1},
    n::Int64,
    norm_squared::Float64,
    eps::Float64,
    lr::Float64,
    bandwidth::Int64
)
    NumBlock =  ceil(Int64, n / ThreadPerBlock)
    CUDA.@sync begin
        CUDA.cudacall(
            osgm_band_update_kernel,
            (CuPtr(Float64), CuPtr(Float64), CuPtr(Float64), CuPtr(Float64), CuPtr(Float64), Int64, Float64, Float64, Int64),
            prev_grad, grad, adagrad_state, osgm_state, x, n, norm_squared, eps, lr, bandwidth;
            blocks = NumBlock,
            threads = ThreadPerBlock
        )
    end
end