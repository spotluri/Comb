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

#ifndef _POL_CUDA_BATCH_HPP
#define _POL_CUDA_BATCH_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_CUDA
#include "batch_launch.hpp"

struct cuda_batch_component
{
  void* ptr = nullptr;
};

struct cuda_batch_group
{
  void* ptr = nullptr;
};

struct cuda_batch_pol {
  static const bool async = true;
  static const char* get_name() { return ( get_batch_always_grid_sync() ? "cudaBatch"      : "cudaBatch_fewgs"      ); }
  using event_type = detail::batch_event_type_ptr;
  using component_type = cuda_batch_component;
  using group_type = cuda_batch_group;
};

template < >
struct ExecContext<cuda_batch_pol> : CudaContext
{
  using pol = cuda_batch_pol;
  using event_type = typename pol::event_type;
  using component_type = typename pol::component_type;
  using group_type = typename pol::group_type;

  using base = CudaContext;

  ExecContext()
    : base()
  { }

  ExecContext(base const& b)
    : base(b)
  { }

  void ensure_waitable()
  {
    cuda::batch_launch::force_launch(base::stream_launch());
  }

  template < typename context >
  void waitOn(context& con)
  {
    con.ensure_waitable();
    base::waitOn(con);
  }

  void synchronize()
  {
    cuda::batch_launch::synchronize(base::stream_launch());
  }

  group_type create_group()
  {
    return group_type{};
  }

  void start_group(group_type)
  {
  }

  void finish_group(group_type)
  {
    cuda::batch_launch::force_launch(base::stream_launch());
  }

  void destroy_group(group_type)
  {

  }

  component_type create_component()
  {
    return component_type{};
  }

  void start_component(group_type, component_type)
  {

  }

  void finish_component(group_type, component_type)
  {

  }

  void destroy_component(component_type)
  {

  }

  event_type createEvent()
  {
    return cuda::batch_launch::createEvent();
  }

  void recordEvent(event_type event)
  {
    return cuda::batch_launch::recordEvent(event, base::stream());
  }

  void finish_component_recordEvent(group_type group, component_type component, event_type event)
  {
    finish_component(group, component);
    recordEvent(event);
  }

  bool queryEvent(event_type event)
  {
    return cuda::batch_launch::queryEvent(event);
  }

  void waitEvent(event_type event)
  {
    cuda::batch_launch::waitEvent(event);
  }

  void destroyEvent(event_type event)
  {
    cuda::batch_launch::destroyEvent(event);
  }

  template < typename body_type >
  void for_all(IdxT begin, IdxT end, body_type&& body)
  {
    cuda::batch_launch::for_all(begin, end, std::forward<body_type>(body), base::stream());
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_2d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
  {
    IdxT len = (end0 - begin0) * (end1 - begin1);
    cuda::batch_launch::for_all(0, len, detail::adapter_2d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, std::forward<body_type>(body)}, base::stream());
    // base::synchronize();
  }

  template < typename body_type >
  void for_all_3d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
  {
    IdxT len = (end0 - begin0) * (end1 - begin1) * (end2 - begin2);
    cuda::batch_launch::for_all(0, len, detail::adapter_3d<typename std::remove_reference<body_type>::type>{begin0, end0, begin1, end1, begin2, end2, std::forward<body_type>(body)}, base::stream());
    // base::synchronize();
  }

};

#endif // COMB_ENABLE_CUDA

#endif // _POL_CUDA_BATCH_HPP
