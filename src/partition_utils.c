/*
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

#include "partition_utils.h"
#include "utils.h"

#include <float.h>
#include <stdlib.h>

typedef struct
{
    int first;
    int last;
} forbidden_boundary_interval_t;

static int compare_boundary_intervals(const void *left, const void *right)
{
    const forbidden_boundary_interval_t *a = (const forbidden_boundary_interval_t *)left;
    const forbidden_boundary_interval_t *b = (const forbidden_boundary_interval_t *)right;
    return (a->first > b->first) - (a->first < b->first);
}

static int compare_ints(const void *left, const void *right)
{
    int a = *(const int *)left;
    int b = *(const int *)right;
    return (a > b) - (a < b);
}

static int containing_boundary_interval(const forbidden_boundary_interval_t *intervals, int count, int boundary)
{
    int low = 0;
    int high = count - 1;
    while (low <= high)
    {
        int middle = low + (high - low) / 2;
        if (boundary < intervals[middle].first)
            high = middle - 1;
        else if (boundary > intervals[middle].last)
            low = middle + 1;
        else
            return middle;
    }
    return -1;
}

static int project_to_legal_boundary(
    const forbidden_boundary_interval_t *intervals, int count, int total_dimension, int boundary, int direction)
{
    if (total_dimension <= 1)
        return -1;
    if (boundary < 1)
    {
        if (direction < 0)
            return -1;
        boundary = 1;
    }
    if (boundary >= total_dimension)
    {
        if (direction > 0)
            return -1;
        boundary = total_dimension - 1;
    }

    int interval = containing_boundary_interval(intervals, count, boundary);
    if (interval >= 0)
        boundary = direction < 0 ? intervals[interval].first - 1 : intervals[interval].last + 1;
    return boundary > 0 && boundary < total_dimension ? boundary : -1;
}

static void add_cut_candidate(int *candidates, int *count, int candidate)
{
    if (candidate > 0)
        candidates[(*count)++] = candidate;
}

bool optimize_partition_cuts(int total_dimension,
                             int num_partitions,
                             const int *forbidden_starts,
                             const int *forbidden_ends,
                             int num_forbidden_intervals,
                             int *cuts)
{
    if (total_dimension < 0 || num_partitions <= 0 || num_forbidden_intervals < 0 || !cuts ||
        (num_forbidden_intervals > 0 && (!forbidden_starts || !forbidden_ends)))
        return false;

    int cuts_needed = num_partitions - 1;
    if (cuts_needed <= 0)
        return true;
    if (total_dimension < num_partitions)
        return false;

    forbidden_boundary_interval_t *intervals = num_forbidden_intervals > 0
        ? (forbidden_boundary_interval_t *)safe_malloc((size_t)num_forbidden_intervals * sizeof(*intervals))
        : NULL;
    for (int interval = 0; interval < num_forbidden_intervals; ++interval)
    {
        int first = forbidden_starts[interval];
        int last = forbidden_ends[interval];
        if (first < 1 || last < first || last >= total_dimension)
        {
            free(intervals);
            return false;
        }
        intervals[interval].first = first;
        intervals[interval].last = last;
    }
    if (num_forbidden_intervals > 1)
        qsort(intervals, (size_t)num_forbidden_intervals, sizeof(*intervals), compare_boundary_intervals);

    int merged_count = 0;
    for (int interval = 0; interval < num_forbidden_intervals; ++interval)
    {
        if (merged_count == 0 || intervals[interval].first > intervals[merged_count - 1].last + 1)
        {
            intervals[merged_count++] = intervals[interval];
        }
        else if (intervals[interval].last > intervals[merged_count - 1].last)
        {
            intervals[merged_count - 1].last = intervals[interval].last;
        }
    }

    size_t candidate_capacity = 4u * (size_t)num_partitions * (size_t)num_partitions + 4u * num_partitions;
    int *candidates = (int *)safe_malloc(candidate_capacity * sizeof(int));
    int candidate_count = 0;

    /* An optimum with P-1 cuts cannot skip P unused legal boundaries on either side of a target. */
    for (int cut = 1; cut < num_partitions; ++cut)
    {
        int lower = project_to_legal_boundary(intervals, merged_count, total_dimension, cuts[cut], -1);
        int upper = project_to_legal_boundary(intervals, merged_count, total_dimension, cuts[cut], 1);
        for (int step = 0; step < num_partitions && lower > 0; ++step)
        {
            add_cut_candidate(candidates, &candidate_count, lower);
            lower = project_to_legal_boundary(intervals, merged_count, total_dimension, lower - 1, -1);
        }
        for (int step = 0; step < num_partitions && upper > 0; ++step)
        {
            add_cut_candidate(candidates, &candidate_count, upper);
            upper = project_to_legal_boundary(intervals, merged_count, total_dimension, upper + 1, 1);
        }
    }

    int earliest = project_to_legal_boundary(intervals, merged_count, total_dimension, 1, 1);
    int latest = project_to_legal_boundary(intervals, merged_count, total_dimension, total_dimension - 1, -1);
    for (int step = 0; step < num_partitions && earliest > 0; ++step)
    {
        add_cut_candidate(candidates, &candidate_count, earliest);
        earliest = project_to_legal_boundary(intervals, merged_count, total_dimension, earliest + 1, 1);
    }
    for (int step = 0; step < num_partitions && latest > 0; ++step)
    {
        add_cut_candidate(candidates, &candidate_count, latest);
        latest = project_to_legal_boundary(intervals, merged_count, total_dimension, latest - 1, -1);
    }
    free(intervals);

    qsort(candidates, (size_t)candidate_count, sizeof(int), compare_ints);
    int unique_count = 0;
    for (int candidate = 0; candidate < candidate_count; ++candidate)
    {
        if (unique_count == 0 || candidates[candidate] != candidates[unique_count - 1])
            candidates[unique_count++] = candidates[candidate];
    }
    if (unique_count < cuts_needed)
    {
        free(candidates);
        return false;
    }

    size_t table_size = (size_t)cuts_needed * (size_t)unique_count;
    long double *cost = (long double *)safe_malloc(table_size * sizeof(long double));
    int *predecessor = (int *)safe_malloc(table_size * sizeof(int));
    for (int candidate = 0; candidate < unique_count; ++candidate)
    {
        long double delta = (long double)candidates[candidate] - cuts[1];
        cost[candidate] = delta * delta;
        predecessor[candidate] = -1;
    }

    for (int cut = 1; cut < cuts_needed; ++cut)
    {
        long double best_prefix_cost = LDBL_MAX;
        int best_prefix = -1;
        for (int candidate = 0; candidate < unique_count; ++candidate)
        {
            if (candidate > 0)
            {
                long double previous = cost[(size_t)(cut - 1) * unique_count + candidate - 1];
                if (previous < best_prefix_cost)
                {
                    best_prefix_cost = previous;
                    best_prefix = candidate - 1;
                }
            }
            size_t entry = (size_t)cut * unique_count + candidate;
            if (best_prefix < 0)
            {
                cost[entry] = LDBL_MAX;
                predecessor[entry] = -1;
                continue;
            }
            long double delta = (long double)candidates[candidate] - cuts[cut + 1];
            cost[entry] = best_prefix_cost + delta * delta;
            predecessor[entry] = best_prefix;
        }
    }

    int best = -1;
    long double best_cost = LDBL_MAX;
    size_t final_row = (size_t)(cuts_needed - 1) * unique_count;
    for (int candidate = 0; candidate < unique_count; ++candidate)
    {
        if (cost[final_row + candidate] < best_cost)
        {
            best_cost = cost[final_row + candidate];
            best = candidate;
        }
    }
    if (best >= 0)
    {
        for (int cut = cuts_needed - 1; cut >= 0; --cut)
        {
            cuts[cut + 1] = candidates[best];
            best = predecessor[(size_t)cut * unique_count + best];
        }
    }

    free(cost);
    free(predecessor);
    free(candidates);
    return best_cost < LDBL_MAX;
}
