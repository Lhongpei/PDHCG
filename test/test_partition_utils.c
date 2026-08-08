#include "partition_utils.h"

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

int main(void)
{
    int cuts[] = {0, 4, 8, 12};
    const int forbidden_starts[] = {2, 7};
    const int forbidden_ends[] = {5, 10};
    CHECK(optimize_partition_cuts(12, 3, forbidden_starts, forbidden_ends, 2, cuts));
    CHECK(cuts[0] == 0 && cuts[1] > 0 && cuts[1] < cuts[2] && cuts[2] < cuts[3] && cuts[3] == 12);
    CHECK(!(cuts[1] >= 2 && cuts[1] <= 5) && !(cuts[1] >= 7 && cuts[1] <= 10));
    CHECK(!(cuts[2] >= 2 && cuts[2] <= 5) && !(cuts[2] >= 7 && cuts[2] <= 10));

    int joint_cuts[] = {0, 10, 11, 101};
    const int joint_forbidden_starts[] = {2, 11};
    const int joint_forbidden_ends[] = {9, 99};
    CHECK(optimize_partition_cuts(101, 3, joint_forbidden_starts, joint_forbidden_ends, 2, joint_cuts));
    CHECK(joint_cuts[1] == 1 && joint_cuts[2] == 10);

    int impossible_cuts[] = {0, 2, 4, 6};
    const int impossible_start[] = {1};
    const int impossible_end[] = {5};
    CHECK(!optimize_partition_cuts(6, 3, impossible_start, impossible_end, 1, impossible_cuts));

    return 0;
}
