//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-758885
//
// All rights reserved.
//
// This file is part of Comb.
//
// For details, see https://github.com/LLNL/Comb
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#include "config.hpp"

#ifdef COMB_ENABLE_CUDA

#include "persistent_launch.hpp"

#include "utils_cuda.hpp"

#ifdef COMB_ENABLE_CUDA_BASIL_BATCH
#include <cooperative_groups.h>
#endif

namespace cuda {

namespace persistent_launch {

namespace detail {

// included here to avoid linker error with clangcuda
#include "batch_exec.cuh"

// Launches a batch kernel and cycles to next buffer
void launch(::detail::MultiBuffer& mb, cudaStream_t stream)
{
#ifdef COMB_ENABLE_CUDA_BASIL_BATCH
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (!getLaunched()) {

      // ensure other thread/device done reading buffer before launch
      while(!mb.cur_buffer_empty());
      // get empty unused buffer
      ::detail::MultiBuffer::buffer_type* device_buffer = mb.get_buffer();

      int blocks_cutoff = 1;
      // TODO decide cutoff below which all blocks read pinned memory directly
      // if (::detail::cuda::get_cuda_arch() >= 600) {
      //    // pascal or newer
      //    blocks_cutoff = 1;
      // } else {
      //    blocks_cutoff = 1;
      // }

      int blocksize = 1024;
      // TODO decide blocksize in a smart way

      int num_blocks = ::COMB::detail::cuda::get_num_sm();

      void* func = NULL;
      void* args[] = { (void*)&device_buffer };

      if (num_blocks < blocks_cutoff) {
         // don't use device cache
         if (get_batch_always_grid_sync()) {
            func = (void*)&block_read_device<::detail::MultiBuffer::shared_buffer_type>;
         } else {
            func = (void*)&block_read_device_few_grid_sync<::detail::MultiBuffer::shared_buffer_type>;
         }
      } else {
         // use device cache
         if (get_batch_always_grid_sync()) {
            func = (void*)&block_read_device<::detail::MultiBuffer::shared_buffer_type>;
         } else {
            func = (void*)&block_read_device_few_grid_sync<::detail::MultiBuffer::shared_buffer_type>;
         }
      }
      cudaCheck(cudaLaunchCooperativeKernel(func, num_blocks, blocksize,
                                            args, 0, stream));
      getLaunched() = true;
   }
#endif
}

void stop(::detail::MultiBuffer& mb, cudaStream_t stream)
{
#ifdef COMB_ENABLE_CUDA_BASIL_BATCH
   if (getLaunched()) {
     mb.done_packing();
     getLaunched() = false;
   }
#endif
}

} // namespace detail

// Start the current batch (launches kernel)
void force_launch(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (!detail::getLaunched()) {
      detail::launch(detail::getMultiBuffer(), stream);
   }
}

// Ensure current batch launched (does nothing)
void force_stop(cudaStream_t stream)
{
   if (detail::getLaunched()) {
      detail::stop(detail::getMultiBuffer(), stream);
   }
}

// Wait for all batches to finish running
void synchronize(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   force_stop(stream);

   // perform synchronization
   cudaCheck(cudaDeviceSynchronize());
}

bool available()
{
   return ::detail::batch_implementation_available();
}

} // namespace persistent_launch

} // namespace cuda

#endif
