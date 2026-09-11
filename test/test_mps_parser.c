#include "mps_parser.h"
#include "pdhcg.h"

#include <math.h>
#include <stdio.h>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition);                            \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static int write_model(const char *path, const char *model)
{
    FILE *file = fopen(path, "w");
    if (!file)
        return 0;
    int write_ok = fputs(model, file) >= 0;
    return fclose(file) == 0 && write_ok;
}

static int check_close(double actual, double expected)
{
    return fabs(actual - expected) <= 1e-12;
}

static int test_missing_file(void)
{
    qp_problem_t *problem = read_mps_file("test_mps_parser_file_that_does_not_exist.mps");
    CHECK(problem == NULL);
    return 0;
}

static int test_set_name_less_sections(void)
{
    static const char model[] = "NAME          NAMELESS\n"
                                "OBJSENS MAX\n"
                                "ROWS\n"
                                " N  COST\n"
                                " L  R1\n"
                                " G  R2\n"
                                " E  R3\n"
                                "COLUMNS\n"
                                "    X1  COST  1.5  R1  1\n"
                                "        R2    2    R3  3\n"
                                "    X2  R1    4    R2  5\n"
                                "RHS\n"
                                "    R1  5  R2  2\n"
                                "    R3  8  COST  -1.5\n"
                                "RANGES\n"
                                "    R3  6\n"
                                "BOUNDS\n"
                                " UP X1 4\n"
                                " LO X2 0.5\n"
                                " MI X1\n"
                                "ENDATA\n";
    const char *path = "test_mps_parser_nameless_tmp.mps";
    CHECK(write_model(path, model));
    qp_problem_t *problem = read_mps_file(path);
    remove(path);
    CHECK(problem != NULL);

    CHECK(problem->num_variables == 2);
    CHECK(problem->num_constraints == 3);
    CHECK(check_close(problem->objective_vector[0], -1.5));
    CHECK(check_close(problem->objective_vector[1], 0.0));
    CHECK(check_close(problem->objective_constant, -1.5));
    CHECK(isinf(problem->variable_lower_bound[0]) && problem->variable_lower_bound[0] < 0.0);
    CHECK(check_close(problem->variable_upper_bound[0], 4.0));
    CHECK(check_close(problem->variable_lower_bound[1], 0.5));
    CHECK(isinf(problem->variable_upper_bound[1]) && problem->variable_upper_bound[1] > 0.0);

    CHECK(isinf(problem->constraint_lower_bound[0]) && problem->constraint_lower_bound[0] < 0.0);
    CHECK(check_close(problem->constraint_upper_bound[0], 5.0));
    CHECK(check_close(problem->constraint_lower_bound[1], 2.0));
    CHECK(isinf(problem->constraint_upper_bound[1]) && problem->constraint_upper_bound[1] > 0.0);
    CHECK(check_close(problem->constraint_lower_bound[2], 8.0));
    CHECK(check_close(problem->constraint_upper_bound[2], 14.0));

    CHECK(problem->constraint_matrix_num_nonzeros == 5);
    CHECK(problem->constraint_matrix->row_ptr[0] == 0);
    CHECK(problem->constraint_matrix->row_ptr[1] == 2);
    CHECK(problem->constraint_matrix->row_ptr[2] == 4);
    CHECK(problem->constraint_matrix->row_ptr[3] == 5);
    CHECK(problem->constraint_matrix->col_ind[0] == 0 && check_close(problem->constraint_matrix->val[0], 1.0));
    CHECK(problem->constraint_matrix->col_ind[1] == 1 && check_close(problem->constraint_matrix->val[1], 4.0));
    CHECK(problem->constraint_matrix->col_ind[2] == 0 && check_close(problem->constraint_matrix->val[2], 2.0));
    CHECK(problem->constraint_matrix->col_ind[3] == 1 && check_close(problem->constraint_matrix->val[3], 5.0));
    CHECK(problem->constraint_matrix->col_ind[4] == 0 && check_close(problem->constraint_matrix->val[4], 3.0));

    qp_problem_free(problem);
    return 0;
}

static int test_integer_bound_defaults(void)
{
    static const char model[] = "NAME          INTBOUNDS\n"
                                "ROWS\n"
                                " N  OBJ\n"
                                " L  R1\n"
                                "COLUMNS\n"
                                "    C0       R1  1\n"
                                "    MARK0000 'MARKER' 'INTORG'\n"
                                "    I1       R1  1\n"
                                "    I2       R1  1\n"
                                "    I3       R1  1\n"
                                "    I4       R1  1\n"
                                "    I5       R1  1\n"
                                "    MARK0001 'MARKER' 'INTEND'\n"
                                "    C6       R1  1\n"
                                "    C7       R1  1\n"
                                "    C8       R1  1\n"
                                "BOUNDS\n"
                                " UI BND I2 5\n"
                                " LI BND I3 2\n"
                                " MI BND I4\n"
                                " UP BND I5 -3\n"
                                " UP BND C6 -3\n"
                                " LO BND C7 -10\n"
                                " UP BND C7 -3\n"
                                " UP BND C8 1e20\n"
                                "ENDATA\n";
    const char *path = "test_mps_parser_integer_tmp.mps";
    CHECK(write_model(path, model));
    qp_problem_t *problem = read_mps_file(path);
    remove(path);
    CHECK(problem != NULL);
    CHECK(problem->num_variables == 9);

    CHECK(check_close(problem->variable_lower_bound[0], 0.0) && isinf(problem->variable_upper_bound[0]));
    CHECK(check_close(problem->variable_lower_bound[1], 0.0) && check_close(problem->variable_upper_bound[1], 1.0));
    CHECK(check_close(problem->variable_lower_bound[2], 0.0) && check_close(problem->variable_upper_bound[2], 5.0));
    CHECK(check_close(problem->variable_lower_bound[3], 2.0) && isinf(problem->variable_upper_bound[3]));
    CHECK(isinf(problem->variable_lower_bound[4]) && problem->variable_lower_bound[4] < 0.0);
    CHECK(isinf(problem->variable_upper_bound[4]) && problem->variable_upper_bound[4] > 0.0);
    CHECK(isinf(problem->variable_lower_bound[5]) && check_close(problem->variable_upper_bound[5], -3.0));
    CHECK(isinf(problem->variable_lower_bound[6]) && check_close(problem->variable_upper_bound[6], -3.0));
    CHECK(check_close(problem->variable_lower_bound[7], -10.0) && check_close(problem->variable_upper_bound[7], -3.0));
    CHECK(check_close(problem->variable_lower_bound[8], 0.0) && check_close(problem->variable_upper_bound[8], 1e20));

    qp_problem_free(problem);
    return 0;
}

static int test_no_objective_row(void)
{
    static const char model[] = "NAME          FEASIBILITY\n"
                                "ROWS\n"
                                " L  R1\n"
                                " G  R2\n"
                                "COLUMNS\n"
                                "    X1  R1  1  R2  2\n"
                                "RHS\n"
                                "    RHS1 R1 3 R2 4\n"
                                "ENDATA\n";
    const char *path = "test_mps_parser_no_objective_tmp.mps";
    CHECK(write_model(path, model));
    qp_problem_t *problem = read_mps_file(path);
    remove(path);
    CHECK(problem != NULL);
    CHECK(problem->num_variables == 1);
    CHECK(problem->num_constraints == 2);
    CHECK(check_close(problem->objective_vector[0], 0.0));
    CHECK(check_close(problem->constraint_upper_bound[0], 3.0));
    CHECK(check_close(problem->constraint_lower_bound[1], 4.0));
    qp_problem_free(problem);
    return 0;
}

static int test_multiline_objective_sense(void)
{
    static const char model[] = "NAME          MAXMODEL\n"
                                "OBJSENSE\n"
                                " MAX\n"
                                "ROWS\n"
                                " N  OBJ\n"
                                "COLUMNS\n"
                                "    X  OBJ  2\n"
                                "ENDATA\n";
    const char *path = "test_mps_parser_objsense_tmp.mps";
    CHECK(write_model(path, model));
    qp_problem_t *problem = read_mps_file(path);
    remove(path);
    CHECK(problem != NULL);
    CHECK(problem->num_variables == 1);
    CHECK(check_close(problem->objective_vector[0], -2.0));
    qp_problem_free(problem);
    return 0;
}

static int test_optional_and_unsupported_sections(void)
{
    static const char quadratic_model[] = "NAME          QUADRATIC\n"
                                          "ROWS\n"
                                          " N  OBJ\n"
                                          "COLUMNS\n"
                                          "    X  OBJ  1\n"
                                          "QUADOBJ\n"
                                          "    X  X  2\n"
                                          "SOS\n"
                                          " S1 SET1\n"
                                          "    X 1\n"
                                          "ENDATA\n";
    const char *quadratic_path = "test_mps_parser_quadratic_tmp.mps";
    CHECK(write_model(quadratic_path, quadratic_model));
    qp_problem_t *problem = read_mps_file(quadratic_path);
    remove(quadratic_path);
    CHECK(problem != NULL);
    CHECK(problem->objective_sparse_matrix_num_nonzeros == 1);
    CHECK(problem->objective_sparse_matrix->row_ptr[0] == 0);
    CHECK(problem->objective_sparse_matrix->row_ptr[1] == 1);
    CHECK(problem->objective_sparse_matrix->col_ind[0] == 0);
    CHECK(check_close(problem->objective_sparse_matrix->val[0], 2.0));
    qp_problem_free(problem);

    static const char unsupported_model[] = "NAME          UNSUPPORTED\n"
                                            "ROWS\n"
                                            " N  OBJ\n"
                                            " L  R1\n"
                                            "COLUMNS\n"
                                            "    X  R1  1\n"
                                            "INDICATORS\n"
                                            " IF X R1 1\n"
                                            "ENDATA\n";
    const char *unsupported_path = "test_mps_parser_unsupported_tmp.mps";
    CHECK(write_model(unsupported_path, unsupported_model));
    problem = read_mps_file(unsupported_path);
    remove(unsupported_path);
    CHECK(problem == NULL);
    return 0;
}

int main(void)
{
    CHECK(test_missing_file() == 0);
    CHECK(test_set_name_less_sections() == 0);
    CHECK(test_integer_bound_defaults() == 0);
    CHECK(test_no_objective_row() == 0);
    CHECK(test_multiline_objective_sense() == 0);
    CHECK(test_optional_and_unsupported_sections() == 0);
    return 0;
}
