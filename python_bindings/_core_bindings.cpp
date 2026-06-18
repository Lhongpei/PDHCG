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

#include "pdhcg.h"
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
    d["l_inf_ruiz_iterations"] = p.l_inf_ruiz_iterations;
    d["has_pock_chambolle_alpha"] = p.has_pock_chambolle_alpha;
    d["pock_chambolle_alpha"] = p.pock_chambolle_alpha;
    d["bound_objective_rescaling"] = p.bound_objective_rescaling;

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
    geti("l_inf_ruiz_iterations", p->l_inf_ruiz_iterations);
    getb("has_pock_chambolle_alpha", p->has_pock_chambolle_alpha);
    getf("pock_chambolle_alpha", p->pock_chambolle_alpha);
    getb("bound_objective_rescaling", p->bound_objective_rescaling);

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
// solve function
/* Parse a Python list of dicts into a vector of cone_spec_t plus per-cone is_fixed byte
   storage (kept alive in cones_keep_alive until create_qp_problem returns).  Each entry:
       {"type": "soc" | "rsoc" | "exp", "start_idx": int, "v_dim": int (omitted for exp),
        "is_fixed": optional bytes/list of length v_dim+2 (or 3 for exp)} */
static std::vector<cone_spec_t> parse_cone_specs(py::object cones, std::vector<std::vector<char>> &keep_alive)
{
    std::vector<cone_spec_t> out;
    if (cones.is_none())
        return out;
    py::list lst = py::cast<py::list>(cones);
    out.reserve(lst.size());
    keep_alive.reserve(lst.size());
    for (auto &handle : lst)
    {
        py::dict d = py::cast<py::dict>(py::reinterpret_borrow<py::object>(handle));
        std::string ty = py::cast<std::string>(d["type"]);
        cone_spec_t cs{};
        if (ty == "soc")
            cs.type = CONE_STANDARD_SOC;
        else if (ty == "rsoc")
            cs.type = CONE_ROTATED_SOC;
        else if (ty == "exp")
            cs.type = CONE_EXPONENTIAL;
        else
            throw std::invalid_argument("cone type must be 'soc', 'rsoc', or 'exp'; got '" + ty + "'");
        cs.start_idx = py::cast<int>(d["start_idx"]);
        cs.v_dim = (cs.type == CONE_EXPONENTIAL) ? 1 : py::cast<int>(d["v_dim"]);
        cs.is_fixed = nullptr;
        if (d.contains("is_fixed") && !d["is_fixed"].is_none())
        {
            py::sequence seq = py::cast<py::sequence>(d["is_fixed"]);
            int slot_count = (cs.type == CONE_EXPONENTIAL) ? 3 : (cs.v_dim + 2);
            if ((int)seq.size() != slot_count)
                throw std::invalid_argument("is_fixed length must equal cone slot count (" +
                                            std::to_string(slot_count) + ")");
            std::vector<char> buf(slot_count);
            for (int j = 0; j < slot_count; ++j)
                buf[j] = py::cast<bool>(seq[j]) ? 1 : 0;
            keep_alive.push_back(std::move(buf));
            cs.is_fixed = keep_alive.back().data();
        }
        out.push_back(cs);
    }
    return out;
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
                           py::object cones = py::none())
{
    static std::once_flag cuda_init_flag;
    std::call_once(cuda_init_flag, []() { cudaFree(0); });

    PyMatrixView view_a, view_q, view_r;
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

    int n = 0;
    int m = 0;

    if (view_a.desc.n > 0)
        n = view_a.desc.n;
    else if (view_q.desc.n > 0)
        n = view_q.desc.n;
    else if (view_r.desc.n > 0)
        n = view_r.desc.n;

    if (view_a.desc.m > 0)
        m = view_a.desc.m;

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

    std::vector<std::vector<char>> cones_keep;
    std::vector<cone_spec_t> cones_vec = parse_cone_specs(cones, cones_keep);

    qp_problem_t *prob = create_qp_problem(c_ptr,
                                           q_desc_ptr,
                                           r_desc_ptr,
                                           d_desc_ptr,
                                           a_desc_ptr,
                                           l_ptr,
                                           u_ptr,
                                           lb_ptr,
                                           ub_ptr,
                                           c0_ptr,
                                           (int)cones_vec.size(),
                                           cones_vec.empty() ? nullptr : cones_vec.data());
    if (!prob)
    {
        throw std::runtime_error("create_qp_problem failed.");
    }

    // set warm start values if provided
    if ((primal_start && !primal_start.is_none()) || (dual_start && !dual_start.is_none()))
    {
        // validate dimensions and get pointers
        ensure_len_or_null(primal_start, "primal_start", n);
        ensure_len_or_null(dual_start, "dual_start", m);
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

PYBIND11_MODULE(_pdhcg_core, m)
{
    m.doc() = "pdhcg core bindings (auto-detect dense/CSR/CSC/COO; initialize "
              "default params here)";

    m.def("get_default_params", &get_default_params_py, "Return default PDHG parameters as a dict");

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
          py::arg("cones") = py::none());
}
