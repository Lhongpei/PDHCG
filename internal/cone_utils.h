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

#pragma once

#include "pdhcg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int cone_length(cone_type_t type, int v_dim);
    int cone_block_length(const cone_blocks_t *blocks, int block);
    int cone_blocks_init_from_specs(cone_blocks_t *blocks,
                                    int num_cones,
                                    const cone_spec_t *specs,
                                    int ambient_dimension,
                                    bool allow_fixed,
                                    const char *label);
    void cone_blocks_clone(cone_blocks_t *dst, const cone_blocks_t *src);
    void cone_blocks_free(cone_blocks_t *blocks);

#ifdef __cplusplus
}
#endif
