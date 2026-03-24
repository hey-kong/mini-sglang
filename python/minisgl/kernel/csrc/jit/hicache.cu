#include <minisgl/tensor.h>
#include <minisgl/utils.h>

#include <minisgl/utils.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace device {

namespace details {

#define SGL_DEVICE __forceinline__ __device__

template <int kUnit> inline constexpr auto get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4,
                  "Unsupported memory package size");
  }
}

template <int kUnit> using PackageType = decltype(get_mem_package<kUnit>());

SGL_DEVICE uint1 load_nc(const uint1 *__restrict__ src) {
  uint32_t tmp;
  asm volatile("ld.global.L1::no_allocate.b32 %0,[%1];" : "=r"(tmp) : "l"(src));
  return uint1{tmp};
}

SGL_DEVICE uint2 load_nc(const uint2 *__restrict__ src) {
  uint32_t tmp0, tmp1;
  asm volatile("ld.global.L1::no_allocate.v2.b32 {%0,%1},[%2];"
               : "=r"(tmp0), "=r"(tmp1)
               : "l"(src));
  return uint2{tmp0, tmp1};
}

SGL_DEVICE uint4 load_nc(const uint4 *__restrict__ src) {
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3)
               : "l"(src));
  return uint4{tmp0, tmp1, tmp2, tmp3};
}

SGL_DEVICE void store_nc(uint1 *__restrict__ dst, const uint1 &value) {
  uint32_t tmp = value.x;
  asm volatile("st.global.L1::no_allocate.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
}

SGL_DEVICE void store_nc(uint2 *__restrict__ dst, const uint2 &value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  asm volatile("st.global.L1::no_allocate.v2.b32 [%0],{%1,%2};" ::"l"(dst),
               "r"(tmp0), "r"(tmp1));
}

SGL_DEVICE void store_nc(uint4 *__restrict__ dst, const uint4 &value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  uint32_t tmp2 = value.z;
  uint32_t tmp3 = value.w;
  asm volatile(
      "st.global.L1::no_allocate.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst),
      "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
}

} // namespace details

template <typename T, int N> struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};

template <int64_t kBytes, uint32_t kNumThreads>
SGL_DEVICE auto load_vec(const void *__restrict__ src) {
  static_assert(kBytes % 128 == 0, "kBytes must be multiple of 128 bytes");
  static_assert(128 % kNumThreads == 0, "kNumThreads must divide 128 bytes");
  constexpr uint32_t kLoopCount = kBytes / 128;
  using Package = details::PackageType<128 / kNumThreads>;
  using Storage = AlignedStorage<Package, kLoopCount>;

  const auto src_packed = static_cast<const Package *>(src);
  const auto lane_id = threadIdx.x % kNumThreads;
  Storage vec;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;
    vec.data[i] = details::load_nc(&src_packed[j]);
  }

  return vec;
}

template <int64_t kBytes, uint32_t kNumThreads, typename Storage>
SGL_DEVICE void store_vec(void *__restrict__ dst, const Storage &vec) {
  using Package = std::decay_t<decltype(vec.data[0])>;
  constexpr uint32_t kBytesPerLoop = sizeof(Package) * kNumThreads;
  constexpr uint32_t kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "Invalid Storage configuration");

  const auto dst_packed = static_cast<Package *>(dst);
  const auto lane_id = threadIdx.x % kNumThreads;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;
    details::store_nc(&dst_packed[j], vec.data[i]);
  }
}

} // namespace device

namespace {

#define SGL_HICACHE_KERNEL __global__ __launch_bounds__(kBlockSize, 1)

struct HicacheKernelParams {
  void *__restrict__ k_cache_dst;
  void *__restrict__ v_cache_dst;
  const void *__restrict__ indices_dst;
  void *__restrict__ k_cache_src;
  void *__restrict__ v_cache_src;
  const void *__restrict__ indices_src;
  int64_t kv_cache_src_stride;
  int64_t kv_cache_dst_stride;
  uint32_t length;
  uint32_t num_layers = 0; // only used in all_layer transfer
  uint32_t bulk_bytes = 0; // only used in page-first bulk transfer
};

template <typename T, int64_t kElementSize, uint32_t kUnroll,
          uint32_t kBlockQuota, uint32_t kBlockSize>
SGL_HICACHE_KERNEL void
hicache_transfer_per_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const auto &[k_cache_dst, v_cache_dst, indices_dst,              // dst
               k_cache_src, v_cache_src, indices_src,              // src
               kv_cache_src_stride, kv_cache_dst_stride, length, _, __ // metadata
  ] = params;

  const uint32_t work_id =
      blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    const auto pos_src = static_cast<const T *>(indices_src)[i];
    const auto pos_dst = static_cast<const T *>(indices_dst)[i];
    const auto src_k =
        pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_k =
        pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
    const auto src_v =
        pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_v =
        pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
    const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
    const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
    store_vec<kElementSize, kNumThreads>(dst_k, vec_k);
    store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
  }
}

template <typename T, int64_t kElementSize, uint32_t kUnroll,
          uint32_t kBlockQuota, uint32_t kBlockSize>
SGL_HICACHE_KERNEL void
hicache_transfer_all_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  using src_ptr_t = const void *;
  using dst_ptr_t = void *;

  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const auto &[k_ptr_dst, v_ptr_dst, indices_dst, // dst
               k_ptr_src, v_ptr_src, indices_src, // src
               kv_cache_src_stride, kv_cache_dst_stride, length,
               num_layers, _ // metadata
  ] = params;

  const uint32_t work_id =
      blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    const auto pos_src = static_cast<const T *>(indices_src)[i];
    const auto pos_dst = static_cast<const T *>(indices_dst)[i];
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
      const auto k_cache_src = static_cast<const src_ptr_t *>(k_ptr_src)[layer];
      const auto v_cache_src = static_cast<const src_ptr_t *>(v_ptr_src)[layer];
      const auto k_cache_dst = static_cast<const dst_ptr_t *>(k_ptr_dst)[layer];
      const auto v_cache_dst = static_cast<const dst_ptr_t *>(v_ptr_dst)[layer];
      const auto src_k =
          pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_k =
          pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto src_v =
          pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_v =
          pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
      const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
      store_vec<kElementSize, kNumThreads>(dst_k, vec_k);
      store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
    }
  }
}

template <typename T, int64_t kElementSizeUnused, uint32_t kUnrollUnused, uint32_t kBlockQuota,
          uint32_t kBlockSize>
SGL_HICACHE_KERNEL void
hicache_transfer_all_layer_bulk(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  using pack_t = uint1; // 4B pack for conservative alignment
  constexpr uint32_t kPackBytes = sizeof(pack_t);

  const auto &[k_cache_dst, v_cache_dst, indices_dst, // dst
               k_cache_src, v_cache_src, indices_src, // src
               kv_cache_src_stride, kv_cache_dst_stride, length, num_layers_unused, bulk_bytes] = params;
  const auto num_blocks = min(gridDim.x, static_cast<uint32_t>(kBlockQuota));
  for (uint32_t i = blockIdx.x; i < length; i += num_blocks) {
    const auto pos_src = static_cast<const T *>(indices_src)[i];
    const auto pos_dst = static_cast<const T *>(indices_dst)[i];
    const auto src_k =
        static_cast<const uint8_t *>(pointer::offset(k_cache_src, pos_src * kv_cache_src_stride));
    const auto src_v =
        static_cast<const uint8_t *>(pointer::offset(v_cache_src, pos_src * kv_cache_src_stride));
    auto dst_k = static_cast<uint8_t *>(pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride));
    auto dst_v = static_cast<uint8_t *>(pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride));

    const auto pack_src_k = reinterpret_cast<const pack_t *>(src_k);
    const auto pack_src_v = reinterpret_cast<const pack_t *>(src_v);
    auto pack_dst_k = reinterpret_cast<pack_t *>(dst_k);
    auto pack_dst_v = reinterpret_cast<pack_t *>(dst_v);
    const uint32_t pack_count = bulk_bytes / kPackBytes;
    for (uint32_t p = threadIdx.x; p < pack_count; p += kBlockSize) {
      pack_dst_k[p] = details::load_nc(&pack_src_k[p]);
      pack_dst_v[p] = details::load_nc(&pack_src_v[p]);
    }
    // tail bytes should not happen for current KV shapes, but keep correctness.
    if (threadIdx.x == 0) {
      for (uint32_t b = pack_count * kPackBytes; b < bulk_bytes; ++b) {
        dst_k[b] = src_k[b];
        dst_v[b] = src_v[b];
      }
    }
  }
}

template <int64_t kElementSize, uint32_t kUnroll, uint32_t kBlockQuota,
          uint32_t kBlockSize>
struct HiCacheKernel {
  template <typename T>
  static constexpr auto kernel_one =
      hicache_transfer_per_layer<T, kElementSize, kUnroll, kBlockQuota,
                                 kBlockSize>;
  template <typename T>
  static constexpr auto kernel_all =
      hicache_transfer_all_layer<T, kElementSize, kUnroll, kBlockQuota,
                                 kBlockSize>;
  template <typename T>
  static constexpr auto kernel_bulk =
      hicache_transfer_all_layer_bulk<T, kElementSize, kUnroll, kBlockQuota, kBlockSize>;

  static void run_one(const tvm::ffi::TensorView k_cache_dst,
                      const tvm::ffi::TensorView v_cache_dst,
                      const tvm::ffi::TensorView indices_dst,
                      const tvm::ffi::TensorView k_cache_src,
                      const tvm::ffi::TensorView v_cache_src,
                      const tvm::ffi::TensorView indices_src) {
    using namespace host;

    auto D = SymbolicSize{"head dimension"};
    auto N = SymbolicSize{"src kv stride"};
    auto M = SymbolicSize{"dst kv stride"};
    auto L = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    TensorMatcher({-1, D}) //
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_src)
        .verify(v_cache_src);
    TensorMatcher({-1, D}) //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_dst)
        .verify(v_cache_dst);
    TensorMatcher({L}) //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes,
                 "HicacheKernel: cache dimension mismatch.");

    const auto k_cache_dst_ptr = k_cache_dst.data_ptr();
    const auto v_cache_dst_ptr = v_cache_dst.data_ptr();
    const auto k_cache_src_ptr = k_cache_src.data_ptr();
    const auto v_cache_src_ptr = v_cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto kv_cache_src_stride =
        static_cast<int64_t>(N.unwrap() * dtype_size);
    const auto kv_cache_dst_stride =
        static_cast<int64_t>(M.unwrap() * dtype_size);
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();

    constexpr auto kWorkersPerBlock =
        kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks =
        std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = kv_cache_src_stride,
        .kv_cache_dst_stride = kv_cache_dst_stride,
        .length = length,
    };
    const auto kernel = use_int32 ? kernel_one<int32_t> : kernel_one<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_all(const tvm::ffi::TensorView k_ptr_dst,
                      const tvm::ffi::TensorView v_ptr_dst,
                      const tvm::ffi::TensorView indices_dst,
                      const tvm::ffi::TensorView k_ptr_src,
                      const tvm::ffi::TensorView v_ptr_src,
                      const tvm::ffi::TensorView indices_src,
                      const int64_t kv_src_stride_bytes,
                      const int64_t kv_dst_stride_bytes) {
    using namespace host;

    auto N = SymbolicSize{"num_layers"};
    auto L = SymbolicSize{"indices length"};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({N}) //
        .with_dtype<uint64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(k_ptr_src)
        .verify(v_ptr_src)
        .verify(k_ptr_dst)
        .verify(v_ptr_dst);
    TensorMatcher({L}) //
        .with_dtype<int32_t, int64_t>(dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto k_cache_dst_ptr = k_ptr_dst.data_ptr();
    const auto v_cache_dst_ptr = v_ptr_dst.data_ptr();
    const auto k_cache_src_ptr = k_ptr_src.data_ptr();
    const auto v_cache_src_ptr = v_ptr_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto use_int32 = dtype_.unwrap().bits == 32;
    const auto device = device_.unwrap();

    constexpr auto kWorkersPerBlock =
        kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks =
        std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = kv_src_stride_bytes,
        .kv_cache_dst_stride = kv_dst_stride_bytes,
        .length = length,
        .num_layers = static_cast<uint32_t>(N.unwrap()),
    };
    const auto kernel = use_int32 ? kernel_all<int32_t> : kernel_all<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_bulk(const tvm::ffi::TensorView k_cache_dst,
                       const tvm::ffi::TensorView v_cache_dst,
                       const tvm::ffi::TensorView indices_dst,
                       const tvm::ffi::TensorView k_cache_src,
                       const tvm::ffi::TensorView v_cache_src,
                       const tvm::ffi::TensorView indices_src,
                       const int64_t kv_src_stride_bytes,
                       const int64_t kv_dst_stride_bytes,
                       const int64_t bulk_bytes) {
    using namespace host;

    auto D = SymbolicSize{"head dimension"};
    auto N = SymbolicSize{"src kv stride"};
    auto M = SymbolicSize{"dst kv stride"};
    auto L = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    TensorMatcher({-1, D})
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_src)
        .verify(v_cache_src);
    TensorMatcher({-1, D})
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_dst)
        .verify(v_cache_dst);
    TensorMatcher({L})
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    RuntimeCheck(kv_src_stride_bytes > 0 && kv_dst_stride_bytes > 0 && bulk_bytes > 0,
                 "Hicache bulk: stride and bulk_bytes must be positive.");

    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst.data_ptr(),
        .v_cache_dst = v_cache_dst.data_ptr(),
        .indices_dst = indices_dst.data_ptr(),
        .k_cache_src = k_cache_src.data_ptr(),
        .v_cache_src = v_cache_src.data_ptr(),
        .indices_src = indices_src.data_ptr(),
        .kv_cache_src_stride = kv_src_stride_bytes,
        .kv_cache_dst_stride = kv_dst_stride_bytes,
        .length = length,
        .bulk_bytes = static_cast<uint32_t>(bulk_bytes),
    };
    const auto kernel = use_int32 ? kernel_bulk<int32_t> : kernel_bulk<int64_t>;
    LaunchKernel(kBlockQuota, kBlockSize, device)(kernel, params);
  }
};

#undef SGL_HICACHE_KERNEL

} // namespace
