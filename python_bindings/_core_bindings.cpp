/*
Copyright 2025 Haihao Lu
Copyright 2026 Hongpei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cbf_parser.h"
#include "cone_utils.h"
#include "mps_parser.h"
#include "pdhcg.h"
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

extern "C"
{
    extern volatile sig_atomic_t g_pdhcg_cancel_request;
}

void sigint_handler(int signum)
{
    (void)signum;
    g_pdhcg_cancel_request = 1;
}

// keepalive for numpy arrays
struct MatrixKeepalive
{
    // keep every owner to prolong lifetime
    std::vector<py::object> owners;
    // temporary storage for index downcast
    std::vector<int32_t> tmp_rowptr, tmp_colind;
    std::vector<int32_t> tmp_row, tmp_col;
};

// view of matrix with keepalive
struct PyMatrixView
{
    matrix_desc_t desc{};
    MatrixKeepalive keep;
};

// get contiguous double numpy array
static py::array get_array_f64_c_contig(py::object obj, const char *name)
{
    // nullptr if obj is None
    if (!obj || obj.is_none())
    {
        throw std::invalid_argument(std::string(name) + " is None.");
    }
    // cast to numpy array
    py::array arr = py::cast<py::array>(obj);
    // must have at least 1 dim
    if (arr.ndim() <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be array.");
    }
    // make contiguous double array
    py::array_t<double, py::array::c_style | py::array::forcecast> out(arr);
    return py::reinterpret_borrow<py::array>(out);
}

// get double pointer to contiguous 1D numpy array
static const double *get_arr_ptr_f64_or_null(py::object obj, const char *name, MatrixKeepalive &keep)
{
    // nullptr if obj is None
    if (!obj || obj.is_none())
    {
        return nullptr;
    }
    // cast to numpy array
    py::array arr = py::cast<py::array>(obj);
    // must have at least 1 dim
    if (arr.ndim() != 1)
    {
        throw std::invalid_argument(std::string(name) + " must be 1D.");
    }
    // make contiguous double array
    py::array_t<double, py::array::c_style | py::array::forcecast> out(arr);
    // keep alive the array owning the memory
    keep.owners.push_back(out);
    // return pointer
    return out.data();
}

// get int32 pointer to contiguous numpy array
static const int32_t *
get_index_ptr_i32(py::object obj, const char *name, MatrixKeepalive &keep, std::vector<int32_t> &tmp_vec)
{
    // nullptr if obj is None
    if (!obj || obj.is_none())
    {
        throw std::invalid_argument(std::string(name) + " is None.");
    }
    // cast to numpy array
    py::array arr = py::cast<py::array>(obj);
    // must have at least 1 dim
    if (arr.ndim() != 1)
    {
        throw std::invalid_argument(std::string(name) + " must be 1D.");
    }
    // make int32 array
    const auto dt = py::dtype(arr.dtype());
    constexpr int64_t I32_MAX = std::numeric_limits<int32_t>::max();
    // contiguous int32 array
    if (dt.equal(py::dtype::of<int32_t>()))
    {
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> out(arr);
        keep.owners.push_back(out);
        return out.data();
    }
    // int64 -> int32 with range check
    if (dt.equal(py::dtype::of<int64_t>()))
    {
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> a(arr);
        const int64_t *p = a.data();
        const py::ssize_t n = a.size();
        tmp_vec.resize(static_cast<size_t>(n));
        for (py::ssize_t i = 0; i < n; ++i)
        {
            int64_t v = p[i];
            if (v < 0 || v > I32_MAX)
            {
                throw std::overflow_error(std::string(name) +
                                          " has value out of int32 range; "
                                          "backend currently supports only 32-bit indices.");
            }
            tmp_vec[static_cast<size_t>(i)] = static_cast<int32_t>(v);
        }
        return tmp_vec.data();
    }
    // unsupported dtype
    throw std::invalid_argument(std::string(name) + " must be int32 or int64.");
}

// helper function to convert norm string to enum
static norm_type_t parse_norm_string(const std::string &s)
{
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "l2")
    {
        return NORM_TYPE_L2;
    }
    else if (lower == "linf")
    {
        return NORM_TYPE_L_INF;
    }
    else
    {
        throw std::invalid_argument("Unknown norm type: " + s + ". Use 'l2' or 'linf'.");
    }
}

// ensure 1D array or None with expected length
static void ensure_len_or_null(py::object obj, const char *name, int expect_len)
{
    // nullptr if obj is None
    if (!obj || obj.is_none())
    {
        return;
    }
    // cast to numpy array
    py::array arr = py::cast<py::array>(obj);
    // must have at least 1 dim
    if (arr.ndim() != 1)
    {
        throw std::invalid_argument(std::string(name) + " must be 1D.");
    }
    // check length
    if ((int)arr.size() != expect_len)
    {
        throw std::invalid_argument(std::string(name) + " length mismatch: expect " + std::to_string(expect_len) +
                                    ", got " + std::to_string((int)arr.size()));
    }
}

// convert termination reason to string
static const char *status_to_str(termination_reason_t r)
{
    switch (r)
    {
        case TERMINATION_REASON_OPTIMAL:
            return "OPTIMAL";
        case TERMINATION_REASON_PRIMAL_INFEASIBLE:
            return "PRIMAL_INFEASIBLE";
        case TERMINATION_REASON_DUAL_INFEASIBLE:
            return "DUAL_INFEASIBLE";
        case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED:
            return "INFEASIBLE_OR_UNBOUNDED";
        case TERMINATION_REASON_TIME_LIMIT:
            return "TIME_LIMIT";
        case TERMINATION_REASON_ITERATION_LIMIT:
            return "ITERATION_LIMIT";
        case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
            return "FEAS_POLISH_SUCCESS";
        case TERMINATION_REASON_UNSPECIFIED:
            return "UNSPECIFIED";
        default:
            return "UNKNOWN";
    }
}

// convert termination reason to int code
static int status_to_code(termination_reason_t r)
{
    switch (r)
    {
        case TERMINATION_REASON_OPTIMAL:
            return 0;
        case TERMINATION_REASON_PRIMAL_INFEASIBLE:
            return 1;
        case TERMINATION_REASON_DUAL_INFEASIBLE:
            return 2;
        case TERMINATION_REASON_TIME_LIMIT:
            return 3;
        case TERMINATION_REASON_ITERATION_LIMIT:
            return 4;
        case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED:
            return 5;
        case TERMINATION_REASON_UNSPECIFIED:
        default:
            return -1;
    }
}

// get default parameters as Python dict
static py::dict get_default_params_py()
{
    pdhg_parameters_t p;
    set_default_parameters(&p);
    py::dict d;

    // verbosity
    d["verbose"] = p.verbose;
    d["termination_evaluation_frequency"] = p.termination_evaluation_frequency;

    // tolerances
    d["eps_optimal_relative"] = p.termination_criteria.eps_optimal_relative;
    d["eps_feasible_relative"] = p.termination_criteria.eps_feasible_relative;

    // limits
    d["time_sec_limit"] = p.termination_criteria.time_sec_limit;
    d["iteration_limit"] = p.termination_criteria.iteration_limit;

    // rescaling
    d["curtis_reid_iterations"] = p.curtis_reid_iterations;
    d["l_inf_ruiz_iterations"] = p.l_inf_ruiz_iterations;
    d["has_pock_chambolle_alpha"] = p.has_pock_chambolle_alpha;
    d["pock_chambolle_alpha"] = p.pock_chambolle_alpha;
    d["bound_objective_rescaling"] = p.bound_objective_rescaling;
    d["use_cone_preserving_scaling"] = p.use_cone_preserving_scaling;

    // restart
    d["artificial_restart_threshold"] = p.restart_params.artificial_restart_threshold;
    d["sufficient_reduction_for_restart"] = p.restart_params.sufficient_reduction_for_restart;
    d["necessary_reduction_for_restart"] = p.restart_params.necessary_reduction_for_restart;
    d["k_p"] = p.restart_params.k_p;

    // reflection
    d["reflection_coefficient"] = p.reflection_coefficient;

    // feasiblity polishing
    d["feasibility_polishing"] = p.feasibility_polishing;
    d["eps_feas_polish_relative"] = p.termination_criteria.eps_feas_polish_relative;

    // presolve
    d["presolve"] = p.presolve;

    // Termination criteria norm
    d["optimality_norm"] = (p.optimality_norm == NORM_TYPE_L_INF) ? "linf" : "l2";

    // power method for singular value estimation
    d["sv_max_iter"] = p.sv_max_iter;
    d["sv_tol"] = p.sv_tol;

    // inner solver parameters
    d["inner_iter_limit"] = p.inner_solver_parameters.iteration_limit;
    d["inner_init_tol"] = p.inner_solver_parameters.initial_tolerance;
    d["inner_min_tol"] = p.inner_solver_parameters.min_tolerance;
    d["diag_jacobi_precond"] = p.diag_jacobi_precond;

    return d;
}

// parse parameters from Python dict
static void parse_params_from_python(py::object params_obj, pdhg_parameters_t *p)
{
    if (!params_obj || params_obj.is_none())
        return;
    py::dict d = params_obj.cast<py::dict>();

    auto getf = [&](const char *k, double &tgt)
    {
        if (d.contains(k))
            tgt = py::cast<double>(d[k]);
    };
    auto geti = [&](const char *k, int &tgt)
    {
        if (d.contains(k))
            tgt = py::cast<int>(d[k]);
    };
    auto getb = [&](const char *k, bool &tgt)
    {
        if (d.contains(k))
            tgt = py::cast<bool>(d[k]);
    };
    auto get_norm = [&](const char *k, norm_type_t &tgt)
    {
        if (d.contains(k))
        {
            py::object val = d[k];
            if (py::isinstance<py::str>(val))
            {
                std::string sval = py::cast<std::string>(val);
                tgt = parse_norm_string(sval);
            }
            else
            {
                throw std::invalid_argument("optimality_norm must be a string ('l2'/'linf')");
            }
        }
    };

    // verbosity
    geti("verbose", p->verbose);
    geti("termination_evaluation_frequency", p->termination_evaluation_frequency);

    // tolerances
    getf("eps_optimal_relative", p->termination_criteria.eps_optimal_relative);
    getf("eps_feasible_relative", p->termination_criteria.eps_feasible_relative);

    // limits
    getf("time_sec_limit", p->termination_criteria.time_sec_limit);
    geti("iteration_limit", p->termination_criteria.iteration_limit);

    // rescaling
    geti("curtis_reid_iterations", p->curtis_reid_iterations);
    geti("l_inf_ruiz_iterations", p->l_inf_ruiz_iterations);
    getb("has_pock_chambolle_alpha", p->has_pock_chambolle_alpha);
    getf("pock_chambolle_alpha", p->pock_chambolle_alpha);
    getb("bound_objective_rescaling", p->bound_objective_rescaling);
    getb("use_cone_preserving_scaling", p->use_cone_preserving_scaling);

    // restart
    getf("artificial_restart_threshold", p->restart_params.artificial_restart_threshold);
    getf("sufficient_reduction_for_restart", p->restart_params.sufficient_reduction_for_restart);
    getf("necessary_reduction_for_restart", p->restart_params.necessary_reduction_for_restart);
    getf("k_p", p->restart_params.k_p);

    // reflection
    getf("reflection_coefficient", p->reflection_coefficient);

    // Feasibility Polishing
    getb("feasibility_polishing", p->feasibility_polishing);
    getf("eps_feas_polish_relative", p->termination_criteria.eps_feas_polish_relative);

    // Termination criteria norm
    get_norm("optimality_norm", p->optimality_norm);

    // power method for singular value estimation
    geti("sv_max_iter", p->sv_max_iter);
    getf("sv_tol", p->sv_tol);

    // inner solver parameters
    geti("inner_iter_limit", p->inner_solver_parameters.iteration_limit);
    getf("inner_init_tol", p->inner_solver_parameters.initial_tolerance);
    getf("inner_min_tol", p->inner_solver_parameters.min_tolerance);
    getb("diag_jacobi_precond", p->diag_jacobi_precond);

    // presolve
    getb("presolve", p->presolve);
}

// view of matrix from Python
static PyMatrixView get_matrix_from_python(py::object A, double zero_tol)
{
    // initialize output
    PyMatrixView out;
    auto &desc = out.desc;
    desc.zero_tolerance = zero_tol;
    // get shape
    if (!py::hasattr(A, "shape"))
    {
        throw std::invalid_argument("matrix A must be numpy.ndarray or "
                                    "scipy.sparse matrix (no .shape attr)");
    }
    auto shape = A.attr("shape").cast<py::tuple>();
    if (shape.size() != 2)
    {
        throw std::invalid_argument("matrix A must be 2D");
    }
    desc.m = shape[0].cast<int>();
    desc.n = shape[1].cast<int>();

    // numpy ndarray as dense matrix
    if (py::isinstance<py::array>(A))
    {
        py::array d = get_array_f64_c_contig(A, "dense matrix (float64)"); // get contiguous data array
        auto req = d.request();
        if (req.ndim != 2)
        {
            throw std::invalid_argument("dense matrix must be 2D");
        }
        desc.m = static_cast<int>(req.shape[0]);
        desc.n = static_cast<int>(req.shape[1]);
        desc.fmt = matrix_dense;
        desc.data.dense.A = static_cast<const double *>(req.ptr);
        out.keep.owners.push_back(d); // keep alive
        return out;
    }

    // SciPy sparse
    std::string fmt = "unknown";
    if (py::hasattr(A, "format"))
        fmt = py::str(A.attr("format"));
    // CSR
    if (fmt == "csr")
    {
        py::object rp = A.attr("indptr");
        py::object ci = A.attr("indices");
        py::object vv = A.attr("data");
        py::array v64 = get_array_f64_c_contig(vv, "csr.data(float64)"); // get contiguous data array
        desc.fmt = matrix_csr;
        desc.data.csr.nnz = static_cast<int>(v64.size());
        desc.data.csr.row_ptr = get_index_ptr_i32(rp, "csr.indptr", out.keep, out.keep.tmp_rowptr);
        desc.data.csr.col_ind = get_index_ptr_i32(ci, "csr.indices", out.keep, out.keep.tmp_colind);
        desc.data.csr.vals = static_cast<const double *>(v64.request().ptr);
        out.keep.owners.push_back(v64); // keep alive
        return out;
    }
    // CSC
    if (fmt == "csc")
    {
        py::object cp = A.attr("indptr");
        py::object ri = A.attr("indices");
        py::object vv = A.attr("data");
        py::array v64 = get_array_f64_c_contig(vv, "csc.data(float64)"); // get contiguous data array
        desc.fmt = matrix_csc;
        desc.data.csc.nnz = static_cast<int>(v64.size());
        desc.data.csc.col_ptr = get_index_ptr_i32(cp, "csc.indptr", out.keep, out.keep.tmp_rowptr);
        desc.data.csc.row_ind = get_index_ptr_i32(ri, "csc.indices", out.keep, out.keep.tmp_colind);
        desc.data.csc.vals = static_cast<const double *>(v64.request().ptr);
        out.keep.owners.push_back(v64); // keep alive
        return out;
    }
    // COO
    if (fmt == "coo")
    {
        py::object rr = A.attr("row");
        py::object cc = A.attr("col");
        py::object vv = A.attr("data");
        py::array v64 = get_array_f64_c_contig(vv, "coo.data(float64)"); // get contiguous data array
        desc.fmt = matrix_coo;
        desc.data.coo.nnz = static_cast<int>(v64.size());
        desc.data.coo.row_ind = get_index_ptr_i32(rr, "coo.row", out.keep, out.keep.tmp_row);
        desc.data.coo.col_ind = get_index_ptr_i32(cc, "coo.col", out.keep, out.keep.tmp_col);
        desc.data.coo.vals = static_cast<const double *>(v64.request().ptr);
        out.keep.owners.push_back(v64); // keep alive
        return out;
    }

    // unsupported format
    throw std::invalid_argument("Unsupported matrix A: expected numpy.ndarray or "
                                "scipy.sparse (csr/csc/coo)");
}
static py::dict
cone_blocks_to_columnar(const cone_blocks_t *blocks, int ambient_dimension, const std::vector<int> *starts = nullptr)
{
    int count = blocks ? blocks->num_cones : 0;
    if (starts && (int)starts->size() != count)
        throw std::logic_error("compact cone starts have the wrong length");

    py::array_t<int32_t> types({count});
    py::array_t<int32_t> start_indices({count});
    py::array_t<int32_t> v_dims({count});
    py::array_t<double> power_alphas({count});
    int32_t *type_data = types.mutable_data();
    int32_t *start_data = start_indices.mutable_data();
    int32_t *v_dim_data = v_dims.mutable_data();
    double *alpha_data = power_alphas.mutable_data();
    for (int cone = 0; cone < count; ++cone)
    {
        type_data[cone] = static_cast<int32_t>(blocks->type[cone]);
        start_data[cone] = starts ? (*starts)[cone] : blocks->start_idx[cone];
        v_dim_data[cone] = blocks->v_dim[cone];
        alpha_data[cone] = blocks->type[cone] == CONE_POWER && blocks->power_alpha ? blocks->power_alpha[cone] : 0.0;
    }

    py::dict result;
    result["types"] = types;
    result["starts"] = start_indices;
    result["v_dims"] = v_dims;
    result["power_alphas"] = power_alphas;
    if (blocks && blocks->is_fixed)
    {
        if (blocks->fixed_mask_size != ambient_dimension)
            throw std::logic_error("cone fixed mask has the wrong ambient dimension");
        py::array_t<uint8_t> fixed_mask({ambient_dimension});
        uint8_t *fixed_data = fixed_mask.mutable_data();
        for (int index = 0; index < ambient_dimension; ++index)
            fixed_data[index] = blocks->is_fixed[index] ? 1 : 0;
        result["fixed_mask"] = fixed_mask;
    }
    else
    {
        result["fixed_mask"] = py::none();
    }
    return result;
}

struct ParsedConeSpecs
{
    std::vector<cone_spec_t> specs;
    std::vector<py::object> owners;
};

static bool has_columnar_cone_fields(const py::object &cones)
{
    return py::hasattr(cones, "types") && py::hasattr(cones, "starts") && py::hasattr(cones, "v_dims") &&
        py::hasattr(cones, "power_alphas");
}

static py::object cone_field(const py::object &cones, const char *name, bool required = true)
{
    if (py::hasattr(cones, name))
        return cones.attr(name);
    if (required)
        throw std::invalid_argument(std::string("columnar cone metadata requires '") + name + "'");
    return py::none();
}

static ParsedConeSpecs parse_columnar_cone_specs(py::object cones, bool affine, int ambient_dimension)
{
    using IntArray = py::array_t<int32_t, py::array::c_style>;
    using DoubleArray = py::array_t<double, py::array::c_style>;

    IntArray types(cone_field(cones, "types"));
    IntArray starts(cone_field(cones, "starts"));
    IntArray v_dims(cone_field(cones, "v_dims"));
    DoubleArray power_alphas(cone_field(cones, "power_alphas"));
    if (types.ndim() != 1 || starts.ndim() != 1 || v_dims.ndim() != 1 || power_alphas.ndim() != 1)
        throw std::invalid_argument("columnar cone fields must be one-dimensional arrays");
    py::ssize_t count = starts.size();
    if (types.size() != count || v_dims.size() != count || power_alphas.size() != count)
        throw std::invalid_argument("columnar cone fields must have the same length");
    if (count > std::numeric_limits<int>::max())
        throw std::invalid_argument("too many cone blocks");

    ParsedConeSpecs out;
    out.specs.resize((size_t)count);

    const uint8_t *fixed_data = nullptr;
    py::object fixed_field = cone_field(cones, "fixed_mask", false);
    if (!fixed_field.is_none())
    {
        if (affine)
            throw std::invalid_argument("affine cones do not support fixed slots");
        py::array_t<uint8_t, py::array::c_style> fixed_mask(fixed_field);
        if (fixed_mask.ndim() != 1 || fixed_mask.size() != ambient_dimension)
            throw std::invalid_argument("fixed_mask length must equal the variable dimension");
        fixed_data = fixed_mask.data();
        out.owners.push_back(std::move(fixed_mask));
    }

    const char *kind = affine ? "affine cone" : "cone";
    const int32_t *type_data = types.data();
    const int32_t *start_data = starts.data();
    const int32_t *v_dim_data = v_dims.data();
    const double *alpha_data = power_alphas.data();
    for (py::ssize_t cone = 0; cone < count; ++cone)
    {
        int type_code = type_data[cone];
        if (type_code < CONE_ROTATED_SOC || type_code > CONE_POWER)
            throw std::invalid_argument(std::string(kind) + " has an invalid type code");
        cone_spec_t &spec = out.specs[(size_t)cone];
        spec.type = static_cast<cone_type_t>(type_code);
        spec.start_idx = start_data[cone];
        spec.v_dim = v_dim_data[cone];
        spec.power_alpha = alpha_data[cone];
        if (spec.v_dim <= 0)
            throw std::invalid_argument(std::string(kind) + " v_dim must be positive");
        if ((spec.type == CONE_EXPONENTIAL || spec.type == CONE_POWER) && spec.v_dim != 1)
            throw std::invalid_argument(std::string(kind) + " EXP and POWER blocks require v_dim == 1");
        if (spec.type == CONE_POWER &&
            !(spec.power_alpha > 0.0 && spec.power_alpha < 1.0 && std::isfinite(spec.power_alpha)))
            throw std::invalid_argument(std::string(kind) + " power alpha must be in (0,1)");
        int length = cone_length(spec.type, spec.v_dim);
        if (spec.start_idx < 0 || length <= 0 || (long long)spec.start_idx + length > (long long)ambient_dimension)
            throw std::invalid_argument(std::string(kind) + " range exceeds the ambient dimension");
        spec.is_fixed = fixed_data ? reinterpret_cast<const char *>(fixed_data + spec.start_idx) : nullptr;
    }
    return out;
}

static ParsedConeSpecs parse_cone_specs(py::object cones, bool affine, int ambient_dimension)
{
    ParsedConeSpecs out;
    if (cones.is_none())
        return out;
    if (!has_columnar_cone_fields(cones))
        throw std::invalid_argument("cones must be a pdhcg.ConeSpec");
    return parse_columnar_cone_specs(cones, affine, ambient_dimension);
}

static py::dict solve_once(py::object Q,
                           py::object R,
                           py::object A,
                           py::object objective_vector,
                           py::object objective_constant,
                           py::object variable_lower_bound,
                           py::object variable_upper_bound,
                           py::object constraint_lower_bound,
                           py::object constraint_upper_bound,
                           double zero_tolerance = 0.0,
                           py::object params = py::none(),
                           py::object primal_start = py::none(),
                           py::object dual_start = py::none(),
                           py::object D = py::none(),
                           py::object cones = py::none(),
                           py::object affine_F = py::none(),
                           py::object affine_g = py::none(),
                           py::object affine_cones = py::none())
{
    static std::once_flag cuda_init_flag;
    std::call_once(cuda_init_flag, []() { cudaFree(0); });

    PyMatrixView view_a, view_q, view_r, view_f;
    if (!A.is_none())
    {
        view_a = get_matrix_from_python(A, zero_tolerance);
    }

    if (!Q.is_none())
    {
        view_q = get_matrix_from_python(Q, zero_tolerance);
    }

    if (!R.is_none())
    {
        view_r = get_matrix_from_python(R, zero_tolerance);
    }

    if (!affine_F.is_none())
    {
        view_f = get_matrix_from_python(affine_F, zero_tolerance);
    }

    int n = 0;
    int m = 0;

    if (view_a.desc.n > 0)
        n = view_a.desc.n;
    else if (view_q.desc.n > 0)
        n = view_q.desc.n;
    else if (view_r.desc.n > 0)
        n = view_r.desc.n;
    else if (view_f.desc.n > 0)
        n = view_f.desc.n;

    if (view_a.desc.m > 0)
        m = view_a.desc.m;

    if (!affine_F.is_none() && view_f.desc.n != n)
        throw std::invalid_argument("affine_F column count must match the number of variables");
    if (affine_F.is_none() && (!affine_g.is_none() || !affine_cones.is_none()))
        throw std::invalid_argument("affine_F is required when affine_g or affine_cones is provided");

    view_a.keep.owners.insert(view_a.keep.owners.end(), view_q.keep.owners.begin(), view_q.keep.owners.end());
    view_a.keep.owners.insert(view_a.keep.owners.end(), view_r.keep.owners.begin(), view_r.keep.owners.end());

    ensure_len_or_null(objective_vector, "objective_vector", n);
    ensure_len_or_null(variable_lower_bound, "variable_lower_bound", n);
    ensure_len_or_null(variable_upper_bound, "variable_upper_bound", n);
    ensure_len_or_null(constraint_lower_bound, "constraint_lower_bound", m);
    ensure_len_or_null(constraint_upper_bound, "constraint_upper_bound", m);

    const double *c_ptr = get_arr_ptr_f64_or_null(objective_vector, "objective_vector", view_a.keep);
    const double *lb_ptr = get_arr_ptr_f64_or_null(variable_lower_bound, "variable_lower_bound", view_a.keep);
    const double *ub_ptr = get_arr_ptr_f64_or_null(variable_upper_bound, "variable_upper_bound", view_a.keep);
    const double *l_ptr = get_arr_ptr_f64_or_null(constraint_lower_bound, "constraint_lower_bound", view_a.keep);
    const double *u_ptr = get_arr_ptr_f64_or_null(constraint_upper_bound, "constraint_upper_bound", view_a.keep);

    double c0_local = 0.0;
    double *c0_ptr = nullptr;
    if (objective_constant && !objective_constant.is_none())
    {
        c0_local = py::cast<double>(objective_constant);
        c0_ptr = &c0_local;
    }

    const matrix_desc_t *q_desc_ptr = Q.is_none() ? nullptr : &view_q.desc;
    const matrix_desc_t *r_desc_ptr = R.is_none() ? nullptr : &view_r.desc;
    const matrix_desc_t *a_desc_ptr = A.is_none() ? nullptr : &view_a.desc;
    PyMatrixView combined_constraint_view;
    if (!affine_F.is_none())
    {
        py::object combined_constraints;
        if (A.is_none())
        {
            combined_constraints = affine_F;
        }
        else if (py::hasattr(A, "format") || py::hasattr(affine_F, "format"))
        {
            py::list matrices;
            matrices.append(A);
            matrices.append(affine_F);
            combined_constraints =
                py::module_::import("scipy.sparse").attr("vstack")(matrices, py::arg("format") = "csr");
        }
        else
        {
            combined_constraints =
                py::module_::import("numpy").attr("concatenate")(py::make_tuple(A, affine_F), py::arg("axis") = 0);
        }
        combined_constraint_view = get_matrix_from_python(combined_constraints, zero_tolerance);
        a_desc_ptr = &combined_constraint_view.desc;
    }

    PyMatrixView view_d;
    std::vector<int32_t> d_diag_rp, d_diag_ci;
    std::vector<double> d_diag_vv;
    const matrix_desc_t *d_desc_ptr = nullptr;
    if (D && !D.is_none())
    {
        if (R.is_none())
        {
            throw std::invalid_argument("D was provided but R is None; D is only meaningful with a low-rank R.");
        }
        int rank = view_r.desc.m;
        bool is_1d = false;
        if (py::isinstance<py::array>(D))
        {
            py::array d_arr = py::cast<py::array>(D);
            if (d_arr.ndim() == 1)
                is_1d = true;
        }
        if (is_1d)
        {
            /* Build a CSR diag(d) directly. */
            py::array_t<double, py::array::c_style | py::array::forcecast> d64(py::cast<py::array>(D));
            if ((int)d64.size() != rank)
            {
                throw std::invalid_argument("D (diag) length " + std::to_string((int)d64.size()) + " must equal rank " +
                                            std::to_string(rank));
            }
            const double *p = d64.data();
            d_diag_rp.reserve(rank + 1);
            d_diag_ci.reserve(rank);
            d_diag_vv.reserve(rank);
            int nz = 0;
            d_diag_rp.push_back(0);
            for (int i = 0; i < rank; ++i)
            {
                if (p[i] != 0.0)
                {
                    d_diag_ci.push_back(i);
                    d_diag_vv.push_back(p[i]);
                    ++nz;
                }
                d_diag_rp.push_back(nz);
            }
            view_d.desc.m = rank;
            view_d.desc.n = rank;
            view_d.desc.fmt = matrix_csr;
            view_d.desc.zero_tolerance = 0.0;
            view_d.desc.data.csr.nnz = (int)d_diag_vv.size();
            view_d.desc.data.csr.row_ptr = d_diag_rp.data();
            view_d.desc.data.csr.col_ind = d_diag_ci.data();
            view_d.desc.data.csr.vals = d_diag_vv.data();
            d_desc_ptr = &view_d.desc;
        }
        else
        {
            view_d = get_matrix_from_python(D, 0.0);
            if (view_d.desc.m != rank || view_d.desc.n != rank)
            {
                throw std::invalid_argument("D shape (" + std::to_string(view_d.desc.m) + ", " +
                                            std::to_string(view_d.desc.n) + ") must be (" + std::to_string(rank) +
                                            ", " + std::to_string(rank) + ")");
            }
            view_a.keep.owners.insert(view_a.keep.owners.end(), view_d.keep.owners.begin(), view_d.keep.owners.end());
            d_desc_ptr = &view_d.desc;
        }
    }

    int num_affine_rows = affine_F.is_none() ? 0 : view_f.desc.m;
    ParsedConeSpecs parsed_cones = parse_cone_specs(cones, false, n);
    ParsedConeSpecs parsed_affine_cones = parse_cone_specs(affine_cones, true, num_affine_rows);
    std::vector<cone_spec_t> &cones_vec = parsed_cones.specs;
    std::vector<cone_spec_t> &affine_cones_vec = parsed_affine_cones.specs;
    const double *affine_g_ptr = nullptr;
    if (!affine_F.is_none())
    {
        if (affine_cones_vec.empty())
            throw std::invalid_argument("affine_cones must describe every row of affine_F");
        ensure_len_or_null(affine_g, "affine_g", num_affine_rows);
        affine_g_ptr = get_arr_ptr_f64_or_null(affine_g, "affine_g", view_f.keep);
    }

    int total_constraint_rows = m + num_affine_rows;
    std::vector<double> combined_lower;
    std::vector<double> combined_upper;
    std::vector<double> combined_affine_offset;
    const double *combined_lower_ptr = l_ptr;
    const double *combined_upper_ptr = u_ptr;
    const double *combined_affine_offset_ptr = nullptr;
    if (num_affine_rows > 0)
    {
        combined_lower.assign((size_t)total_constraint_rows, -std::numeric_limits<double>::infinity());
        combined_upper.assign((size_t)total_constraint_rows, std::numeric_limits<double>::infinity());
        if (l_ptr)
            std::copy(l_ptr, l_ptr + m, combined_lower.begin());
        if (u_ptr)
            std::copy(u_ptr, u_ptr + m, combined_upper.begin());
        combined_lower_ptr = combined_lower.data();
        combined_upper_ptr = combined_upper.data();

        for (cone_spec_t &spec : affine_cones_vec)
            spec.start_idx += m;
        if (affine_g_ptr)
        {
            combined_affine_offset.assign((size_t)total_constraint_rows, 0.0);
            std::copy(affine_g_ptr, affine_g_ptr + num_affine_rows, combined_affine_offset.begin() + m);
            combined_affine_offset_ptr = combined_affine_offset.data();
        }
    }

    qp_problem_t *prob = create_qp_problem(c_ptr,
                                           q_desc_ptr,
                                           r_desc_ptr,
                                           d_desc_ptr,
                                           a_desc_ptr,
                                           combined_lower_ptr,
                                           combined_upper_ptr,
                                           lb_ptr,
                                           ub_ptr,
                                           c0_ptr,
                                           (int)cones_vec.size(),
                                           cones_vec.empty() ? nullptr : cones_vec.data(),
                                           (int)affine_cones_vec.size(),
                                           affine_cones_vec.empty() ? nullptr : affine_cones_vec.data(),
                                           combined_affine_offset_ptr);
    if (!prob)
    {
        throw std::runtime_error("create_qp_problem failed.");
    }
    // set warm start values if provided
    if ((primal_start && !primal_start.is_none()) || (dual_start && !dual_start.is_none()))
    {
        // validate dimensions and get pointers
        ensure_len_or_null(primal_start, "primal_start", n);
        ensure_len_or_null(dual_start, "dual_start", m + num_affine_rows);
        const double *primal_ptr = get_arr_ptr_f64_or_null(primal_start, "primal_start", view_a.keep);
        const double *dual_ptr = get_arr_ptr_f64_or_null(dual_start, "dual_start", view_a.keep);

        set_start_values(prob, primal_ptr, dual_ptr);
    }

    // parse PDHG params
    pdhg_parameters_t local_params;
    set_default_parameters(&local_params);
    parse_params_from_python(params, &local_params);
    // solve (release GIL during compute)
    pdhcg_result_t *res = nullptr;
    g_pdhcg_cancel_request = 0;
    void (*old_sigint_handler)(int) = std::signal(SIGINT, sigint_handler);

    {
        py::gil_scoped_release release;
        res = solve_qp_problem(prob, &local_params);
    }

    std::signal(SIGINT, old_sigint_handler);

    // Note: A user interrupt will only terminate the optimization process, without killing the Python instance.
    // if (g_pdhcg_cancel_request) {
    //     PyErr_SetInterrupt();
    //     if (PyErr_CheckSignals() != 0) {
    //         qp_problem_free(prob);
    //         if (res) pdhcg_result_free(res);
    //         throw py::error_already_set();
    //     }
    // }

    qp_problem_free(prob);
    if (!res)
    {
        throw std::runtime_error("solve_qp_problem returned NULL.");
    }

    // parse result
    const int n_out = res->num_variables;
    const int m_out = res->num_constraints;
    py::array_t<double> x({n_out});
    py::array_t<double> y({m_out});
    {
        auto xb = x.request(), yb = y.request();
        std::memcpy(xb.ptr, res->primal_solution, sizeof(double) * n_out);
        std::memcpy(yb.ptr, res->dual_solution, sizeof(double) * m_out);
    }
    // build info dict
    py::dict info;
    // solution
    info["X"] = x;
    info["Pi"] = y;
    // objectives and gaps
    info["PrimalObj"] = res->primal_objective_value;
    info["DualObj"] = res->dual_objective_value;
    info["ObjectiveGap"] = res->objective_gap;
    info["RelativeObjectiveGap"] = res->relative_objective_gap;
    // stats
    info["Status"] = py::str(status_to_str(res->termination_reason));
    info["StatusCode"] = status_to_code(res->termination_reason);
    info["Iterations"] = res->total_count;
    info["RescalingTimeSec"] = res->rescaling_time_sec;
    info["RuntimeSec"] = res->cumulative_time_sec;
    // residuals
    info["RelativePrimalResidual"] = res->relative_primal_residual;
    info["RelativeDualResidual"] = res->relative_dual_residual;
    // rays
    info["MaxPrimalRayInfeas"] = res->max_primal_ray_infeasibility;
    info["MaxDualRayInfeas"] = res->max_dual_ray_infeasibility;
    info["PrimalRayLinObj"] = res->primal_ray_linear_objective;
    info["DualRayObj"] = res->dual_ray_objective;

    // free result
    pdhcg_result_free(res);

    return info;
}

/* Convert a CsrComponent + shape (m, n, nnz) into a Python dict suitable for
   passing to scipy.sparse.csr_matrix(...). Returns None if the matrix is empty. */
static py::object csr_to_py(const CsrComponent *csr, int rows, int cols, int nnz)
{
    if (!csr || nnz <= 0 || !csr->row_ptr)
        return py::none();
    py::array_t<int32_t> indptr({rows + 1});
    py::array_t<int32_t> indices({nnz});
    py::array_t<double> vals({nnz});
    std::memcpy(indptr.request().ptr, csr->row_ptr, sizeof(int) * (rows + 1));
    std::memcpy(indices.request().ptr, csr->col_ind, sizeof(int) * nnz);
    std::memcpy(vals.request().ptr, csr->val, sizeof(double) * nnz);
    py::dict d;
    d["indptr"] = indptr;
    d["indices"] = indices;
    d["data"] = vals;
    d["shape"] = py::make_tuple(rows, cols);
    return d;
}

static py::object csr_selected_rows_to_py(const CsrComponent *csr, const std::vector<int> &selected_rows, int cols)
{
    if (!csr || !csr->row_ptr || selected_rows.empty())
        return py::none();
    int nnz = 0;
    for (int row : selected_rows)
        nnz += csr->row_ptr[row + 1] - csr->row_ptr[row];
    py::array_t<int32_t> indptr({(int)selected_rows.size() + 1});
    py::array_t<int32_t> indices({nnz});
    py::array_t<double> vals({nnz});
    int32_t *indptr_data = indptr.mutable_data();
    int32_t *indices_data = indices.mutable_data();
    double *values_data = vals.mutable_data();
    indptr_data[0] = 0;
    int cursor = 0;
    for (size_t out_row = 0; out_row < selected_rows.size(); ++out_row)
    {
        int row = selected_rows[out_row];
        int begin = csr->row_ptr[row];
        int count = csr->row_ptr[row + 1] - begin;
        if (count > 0)
        {
            std::memcpy(indices_data + cursor, csr->col_ind + begin, sizeof(int) * count);
            std::memcpy(values_data + cursor, csr->val + begin, sizeof(double) * count);
        }
        cursor += count;
        indptr_data[out_row + 1] = cursor;
    }
    py::dict d;
    d["indptr"] = indptr;
    d["indices"] = indices;
    d["data"] = vals;
    d["shape"] = py::make_tuple((int)selected_rows.size(), cols);
    return d;
}

/* Read an MPS or CBF problem file. Dispatches on file extension (.cbf/.cbf.gz -> CBF,
   otherwise MPS). Affine cone rows are returned separately as affine_F, affine_g,
   and affine_cones. Sparse matrices use {indptr, indices, data, shape} payloads. */
static py::dict read_problem_file_py(const std::string &path)
{
    qp_problem_t *prob = nullptr;
    size_t n = path.size();
    bool is_cbf = false;
    size_t stem_end = n;
    if (n > 3 && path.compare(n - 3, 3, ".gz") == 0)
        stem_end = n - 3;
    if (stem_end >= 4 && path.compare(stem_end - 4, 4, ".cbf") == 0)
        is_cbf = true;

    prob = is_cbf ? read_cbf_file(path.c_str()) : read_mps_file(path.c_str());
    if (!prob)
        throw std::runtime_error("failed to read problem file: " + path);

    /* QCQP files: transform quadratic constraints to SOCP cones so the extracted
       problem is directly solvable via solve_once. Default to rotated SOC form. */
    if (prob->num_quadratic_constraints > 0)
    {
        qp_problem_t *lifted = qcqp_to_socp_qp(prob, CONE_ROTATED_SOC);
        qp_problem_free(prob);
        if (!lifted)
            throw std::runtime_error("QCQP -> SOCP transform failed for: " + path);
        prob = lifted;
    }

    py::dict out;
    int n_var = prob->num_variables;
    int m_con = prob->num_constraints;
    std::vector<char> is_cone_row((size_t)m_con, 0);
    std::vector<int> affine_rows;
    for (int cone = 0; cone < prob->affine_cones.num_cones; ++cone)
    {
        int length = cone_block_length(&prob->affine_cones, cone);
        int start = prob->affine_cones.start_idx[cone];
        for (int slot = 0; slot < length; ++slot)
        {
            is_cone_row[start + slot] = 1;
            affine_rows.push_back(start + slot);
        }
    }
    std::vector<int> scalar_rows;
    for (int row = 0; row < m_con; ++row)
        if (!is_cone_row[row])
            scalar_rows.push_back(row);
    int m_scalar = (int)scalar_rows.size();
    int m_affine = (int)affine_rows.size();

    py::array_t<double> c({n_var});
    std::memcpy(c.request().ptr, prob->objective_vector, sizeof(double) * n_var);
    out["c"] = c;
    out["obj_const"] = prob->objective_constant;

    out["Q"] = csr_to_py(prob->objective_sparse_matrix, n_var, n_var, prob->objective_sparse_matrix_num_nonzeros);
    out["A"] = csr_selected_rows_to_py(prob->constraint_matrix, scalar_rows, n_var);

    py::array_t<double> constr_lb({m_scalar});
    py::array_t<double> constr_ub({m_scalar});
    py::array_t<double> var_lb({n_var});
    py::array_t<double> var_ub({n_var});
    if (m_scalar > 0)
    {
        double *lower = constr_lb.mutable_data();
        double *upper = constr_ub.mutable_data();
        for (int i = 0; i < m_scalar; ++i)
        {
            int row = scalar_rows[i];
            double constant = prob->affine_cone_offset[row];
            lower[i] = prob->constraint_lower_bound[row] - constant;
            upper[i] = prob->constraint_upper_bound[row] - constant;
        }
    }
    std::memcpy(var_lb.request().ptr, prob->variable_lower_bound, sizeof(double) * n_var);
    std::memcpy(var_ub.request().ptr, prob->variable_upper_bound, sizeof(double) * n_var);
    out["constr_lb"] = constr_lb;
    out["constr_ub"] = constr_ub;
    out["var_lb"] = var_lb;
    out["var_ub"] = var_ub;

    if (prob->cones.num_cones > 0)
        out["cones"] = cone_blocks_to_columnar(&prob->cones, n_var);
    else
        out["cones"] = py::none();

    if (m_affine > 0)
    {
        out["affine_F"] = csr_selected_rows_to_py(prob->constraint_matrix, affine_rows, n_var);
        py::array_t<double> affine_g({m_affine});
        double *affine_g_data = affine_g.mutable_data();
        for (int row = 0; row < m_affine; ++row)
            affine_g_data[row] = prob->affine_cone_offset[affine_rows[row]];
        out["affine_g"] = affine_g;
        int compact_start = 0;
        std::vector<int> compact_starts;
        compact_starts.reserve((size_t)prob->affine_cones.num_cones);
        for (int i = 0; i < prob->affine_cones.num_cones; ++i)
        {
            compact_starts.push_back(compact_start);
            compact_start += cone_block_length(&prob->affine_cones, i);
        }
        out["affine_cones"] = cone_blocks_to_columnar(&prob->affine_cones, m_affine, &compact_starts);
    }
    else
    {
        out["affine_F"] = py::none();
        out["affine_g"] = py::none();
        out["affine_cones"] = py::none();
    }

    if (prob->primal_start)
    {
        py::array_t<double> ps({n_var});
        std::memcpy(ps.request().ptr, prob->primal_start, sizeof(double) * n_var);
        out["primal_start"] = ps;
    }

    qp_problem_free(prob);
    return out;
}

PYBIND11_MODULE(_pdhcg_core, m)
{
    m.doc() = "pdhcg core bindings (auto-detect dense/CSR/CSC/COO; initialize "
              "default params here)";

    m.def("get_default_params", &get_default_params_py, "Return default PDHG parameters as a dict");

    m.def("read_problem_file",
          &read_problem_file_py,
          py::arg("path"),
          "Read an MPS or CBF file (.mps/.mps.gz/.cbf/.cbf.gz) and return a dict with "
          "c, obj_const, Q, A, constr_lb, constr_ub, var_lb, var_ub, cones, affine_F, "
          "affine_g, affine_cones, and primal_start.");

    m.def("solve_once",
          &solve_once,
          py::arg("Q"),
          py::arg("R"),
          py::arg("A"),
          py::arg("objective_vector"),
          py::arg("objective_constant") = py::none(),
          py::arg("variable_lower_bound") = py::none(),
          py::arg("variable_upper_bound") = py::none(),
          py::arg("constraint_lower_bound") = py::none(),
          py::arg("constraint_upper_bound") = py::none(),
          py::arg("zero_tolerance") = 0.0,
          py::arg("params") = py::none(),
          py::arg("primal_start") = py::none(),
          py::arg("dual_start") = py::none(),
          py::arg("D") = py::none(),
          py::arg("cones") = py::none(),
          py::arg("affine_F") = py::none(),
          py::arg("affine_g") = py::none(),
          py::arg("affine_cones") = py::none());
}
