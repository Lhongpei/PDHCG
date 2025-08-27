module PDHCG_CLI

using PDHCG, ArgParse, Printf

"""
    julia_main()

PackageCompiler
"""
function julia_main()::Cint
    s = ArgParseSettings(
        description = "PDHCG solver (GPU-ready)",
        commands_are_required = false,
        version = "1.0"
    )

    @add_arg_table! s begin
        "filename"
            help = "QPS / MPS problem file"
            required = false     
            default = ""
        "--gpu", "-g"
            help = "use GPU"
            arg_type = Bool
            default = true
        "--warm-up"
            help = "run warm-up"
            arg_type = Bool
            default = true
        "--verbose", "-v"
            help = "verbosity (0-3)"
            arg_type = Int
            default = 2
        "--time-limit"
            help = "time limit (s)"
            arg_type = Float64
            default = 3600.0
        "--rel-tol"
            help = "relative tolerance"
            arg_type = Float64
            default = 1e-6
        "--iter-limit"
            help = "max iterations"
            arg_type = Int
            default = typemax(Int32)
        "--ruiz-iters"
            help = "Ruiz rescaling iterations"
            arg_type = Int
            default = 10
        "--l2-rescale"
            help = "use L2 rescaling"
            action = :store_true
        "--pock-alpha"
            help = "Pock-Chambolle α"
            arg_type = Float64
            default = 1.0
        "--restart-thresh"
            help = "artificial restart threshold"
            arg_type = Float64
            default = 0.2
        "--suff-red"
            help = "sufficient reduction"
            arg_type = Float64
            default = 0.2
        "--nece-red"
            help = "necessary reduction"
            arg_type = Float64
            default = 0.8
        "--primal-smooth"
            help = "primal weight smoothing"
            arg_type = Float64
            default = 0.2
        "--save"
            help = "save result"
            action = :store_true
        "--saved-name"
            help = "output file name"
            arg_type = String
            default = ""
        "--output-dir"
            help = "output directory"
            arg_type = String
            default = ""
    end

    args = parse_args(s)

    isempty(args["filename"]) && return 0

    result = PDHCG.pdhcgSolveFile(
        args["filename"];
        gpu_flag                      = args["gpu"],
        warm_up_flag                  = args["warm-up"],
        verbose_level                 = args["verbose"],
        time_limit                    = args["time-limit"],
        relat_error_tolerance         = args["rel-tol"],
        iteration_limit               = args["iter-limit"],
        ruiz_rescaling_iters          = args["ruiz-iters"],
        l2_norm_rescaling_flag        = args["l2-rescale"],
        pock_chambolle_alpha          = args["pock-alpha"],
        artificial_restart_threshold  = args["restart-thresh"],
        sufficient_reduction          = args["suff-red"],
        necessary_reduction           = args["nece-red"],
        primal_weight_update_smoothing= args["primal-smooth"],
        save_flag                     = args["save"],
        saved_name                    = isempty(args["saved-name"]) ? nothing : args["saved-name"],
        output_dir                    = isempty(args["output-dir"]) ? nothing : args["output-dir"],
    )

    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    julia_main()
end

end # module PDHCG_CLI