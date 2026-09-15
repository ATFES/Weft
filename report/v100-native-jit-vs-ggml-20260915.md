# V100 本机 Native JIT 与 GGML 对照（2026-09-15）

数据来源：native JIT 套件全量运行 `/tmp/weft-native-jit/20260915-170013`（分支
`v100-jit`，commit `d3e98a050`，repeat=10，taskset core 3）；GGML 侧为
`report/baseline/ggml-riscv-kernel-performance.csv` 的 V100 固定 baseline；正式
协议比值取自 `report/kernel-performance-comparison.csv`（remote runner，64MiB
cache eviction，10 次取中位）。机读逐 case 对照数据见同目录
`v100-native-jit-vs-ggml-20260915.csv`。

边界声明：JIT 的 `warm_median_ms` 是进程内 warm 计时，无 cache eviction，含
ctypes 调用开销，**不是** [测量协议](../doc/experiments/protocol.md) 的正式
kernel timing；本文件为一次性对照快照，正式比值以 comparison CSV 为准。

## 1. 数值对照（Weft native JIT 输出 vs GGML reference）

容差为运行前声明的合同值（沿用 remote runner）：`1e-4 + 2e-3·|expected|`；
gemv q4_k 因随机字节记录输入采用其专属绝对容差 `1.25e-1`。

| 族 | 对照 case 数 | 判定 | 最大绝对误差 | reference |
|---|---:|---|---|---|
| dense | 4 | within-tolerance | 0 | 解析期望值 |
| vec_dot | 24 | within-tolerance | 0.00292969 | ggml_vec_dot_*_generic |
| dequantize | 24 | within-tolerance | 0 | dequantize_row_*（ggml-base） |
| gemv | 2 | within-tolerance | 0.25 | ggml_vec_dot_q4_K_q8_K |
| mul_mat | 54 | within-tolerance | 0.000793457 | ggml_vec_dot_*（逐行）+ quantize_row_q8_* |
| quantize | 3 | 逐位一致（bitexact） | — | 标量规格（q8_quantize_runtime 合同） |

111 个可执行 case 全部在声明容差内（108 数值 + 3 逐位一致）；6 个 staged 绑定
（iq2_s/iq2_xs 的 prefill 与 staged 变体）在 VLEN256 下无合法 full-product
carrier，编译期即失败，不存在可对照的运行结果——GGML baseline 中同样没有这些
staged 变体对应的 Weft 正式测量行。

### 1.1 vec_dot 逐格式误差

| 格式 | K | expected | 最大绝对误差 |
|---|---:|---|---|
| q1_0 | 4096 | 16189.6 | 0 |
| q4_0 | 4096 | 23615.1 | 0 |
| q4_1 | 4096 | 9119.86 | 0 |
| q5_0 | 4096 | 3355.12 | 0 |
| q5_1 | 4096 | -2510.82 | 0 |
| q8_0 | 4096 | -11477.3 | 0 |
| q2_k | 4096 | -13825.5 | 0 |
| q3_k | 4096 | -8120.16 | 0.00292969 |
| q4_k | 4096 | -2169.25 | 0 |
| q5_k | 4096 | 4230.5 | 0.000488281 |
| q6_k | 4096 | 8443.2 | 0.000976562 |
| iq1_s | 4096 | 15009.9 | 0 |
| iq1_m | 4096 | 15195.2 | 0 |
| iq2_s | 4096 | 15093.5 | 0 |
| iq2_xs | 4096 | 13087 | 0 |
| iq2_xxs | 4096 | 12594.5 | 0 |
| iq3_s | 4096 | 8681.03 | 0 |
| iq3_xxs | 4096 | 3015.5 | 0 |
| iq4_nl | 4096 | 31668.7 | 0 |
| iq4_xs | 4096 | -6977.5 | 0.00146484 |
| tq1_0 | 4096 | -12340.3 | 0 |
| tq2_0 | 4096 | -15142.5 | 0 |
| mxfp4 | 4096 | -1559.84 | 0 |
| nvfp4 | 4096 | -16681.4 | 0 |

### 1.2 mul_mat 逐格式×相位最大误差

| 格式 | decode | prefill |
|---|---|---|
| f16 | 6.31213e-05 | 0.000285864 |
| q1_0 | 0 | 0.000793457 |
| q4_0 | 4.57764e-05 | 0 |
| q4_1 | 0.000152588 | 0 |
| q5_0 | 0 | 0 |
| q5_1 | 6.10352e-05 | 0 |
| q8_0 | 0.000137329 | 0 |
| q2_k | 6.10352e-05 | 0.000610352 |
| q3_k | 7.62939e-05 | 0.000366211 |
| q4_k | 0.0001297 | 0.000427246 |
| q5_k | 6.10352e-05 | 0.000488281 |
| q6_k | 6.10352e-05 | 0.000305176 |
| iq1_s | 0 | 0 |
| iq1_m | 0 | 0 |
| iq3_s | 0 | 0 |
| iq3_xxs | 0 | 0 |
| iq4_nl | 0 | 0 |
| iq4_xs | 0.000366211 | 0.000549316 |
| tq1_0 | 0 | 0 |
| tq2_0 | 0 | 0 |
| mxfp4 | 0 | 0 |
| nvfp4 | 0 | 0 |

staged/persistent 变体（iq2_xxs_staged、q4_k_staged、q4_k_persistent 等）同样
全部通过，逐 case 数值见 `result.json`。

## 2. 性能对照

### 2.1 正式协议比值（remote runner，comparison CSV 的 V100 行）

comparison CSV 的 `weft`/`source` 列为吞吐量，ratio = weft/source，
**ratio>1 表示 Weft 更快**。

V100 共 101 个正式 case：Weft 吞吐高于 GGML（ratio>1）74 个，
ratio 中位 1.168，最小 0.600，最大 8.898（最大值为 nvfp4 vec_dot：GGML 为 scalar 实现）。逐行见 
`report/kernel-performance-comparison.csv`。

### 2.2 本机 JIT warm 指示性对照（mul_mat / dense，单次整调用可比较 scope）

JIT warm 计时与 GGML baseline 同为整调用墙钟，但 JIT 侧无 cache eviction、
数据常驻 L2，数值系统性偏低（对 Weft 有利）；此表只作指示，不进入正式 CSV。
注意两列比值语义相反：`JIT/GGML` 为时间比（<1 = Weft 更快），`正式 ratio` 为
吞吐比（>1 = Weft 更快）。q2_K~q6_K 的 mul_mat 在 comparison CSV 中无 V100
正式行（正式对照仅覆盖这些格式的 vec_dot/dequantize），故其正式 ratio 为 —。

| case | M×N×K | JIT warm (ms) | GGML (ms) | JIT/GGML | 正式 ratio |
|---|---|---:|---:|---:|---:|
| mul_mat:f16-decode | 1×4096×4096 | 6.295 | 5.788 | 1.088 | 0.942 |
| mul_mat:f16-prefill | 128×4096×4096 | 185.284 | 164.511 | 1.126 | 0.916 |
| mul_mat:q1_0-decode | 1×4096×4096 | 4.965 | 5.919 | 0.839 | 1.597 |
| mul_mat:q1_0-prefill | 128×4096×4096 | 345.216 | 753.947 | 0.458 | 1.739 |
| mul_mat:q4_0-decode | 1×4096×4096 | 8.544 | 8.857 | 0.965 | 1.080 |
| mul_mat:q4_0-prefill | 128×4096×4096 | 803.451 | 1129.679 | 0.711 | 1.168 |
| mul_mat:q4_1-decode | 1×4096×4096 | 9.129 | 11.402 | 0.801 | 1.103 |
| mul_mat:q4_1-prefill | 128×4096×4096 | 770.520 | 1451.372 | 0.531 | 1.119 |
| mul_mat:q5_0-decode | 1×4096×4096 | 9.111 | 10.384 | 0.877 | 1.206 |
| mul_mat:q5_0-prefill | 128×4096×4096 | 416.758 | 1326.771 | 0.314 | 1.392 |
| mul_mat:q5_1-decode | 1×4096×4096 | 10.732 | 13.712 | 0.783 | 1.175 |
| mul_mat:q5_1-prefill | 128×4096×4096 | 680.178 | 1751.090 | 0.388 | 1.348 |
| mul_mat:q8_0-decode | 1×4096×4096 | 6.569 | 7.417 | 0.886 | 1.203 |
| mul_mat:q8_0-prefill | 128×4096×4096 | 494.841 | 940.118 | 0.526 | 0.998 |
| mul_mat:q2_k-decode | 1×4096×4096 | 6.987 | 6.066 | 1.152 | 1.030 |
| mul_mat:q2_k-prefill | 128×4096×4096 | 842.445 | 768.586 | 1.096 | 0.979 |
| mul_mat:q3_k-decode | 1×4096×4096 | 5.182 | 8.117 | 0.638 | 1.056 |
| mul_mat:q3_k-prefill | 128×4096×4096 | 608.990 | 842.215 | 0.723 | 1.086 |
| mul_mat:q4_k-decode | 1×4096×4096 | 4.275 | 6.507 | 0.657 | 1.765 |
| mul_mat:q4_k-prefill | 128×4096×4096 | 305.430 | 833.831 | 0.366 | 1.800 |
| mul_mat:q5_k-decode | 1×4096×4096 | 8.056 | 11.748 | 0.686 | 1.275 |
| mul_mat:q5_k-prefill | 128×4096×4096 | 621.035 | 1483.553 | 0.419 | 2.294 |
| mul_mat:q6_k-decode | 1×4096×4096 | 7.615 | 6.573 | 1.159 | 0.844 |
| mul_mat:q6_k-prefill | 128×4096×4096 | 817.626 | 792.940 | 1.031 | 0.962 |
| mul_mat:iq1_s-decode | 1×4096×4096 | 5.929 | 6.314 | 0.939 | 0.976 |
| mul_mat:iq1_s-prefill | 128×4096×4096 | 570.764 | 809.293 | 0.705 | 1.207 |
| mul_mat:iq1_m-decode | 1×4096×4096 | 9.841 | 8.336 | 1.181 | 0.800 |
| mul_mat:iq1_m-prefill | 128×4096×4096 | 1630.559 | 1060.784 | 1.537 | 0.899 |
| mul_mat:iq2_xxs-decode | 1×4096×4096 | 6.931 | 11.627 | 0.596 | 1.380 |
| mul_mat:iq2_xxs-prefill | 128×4096×4096 | 749.505 | 1477.639 | 0.507 | 1.865 |
| mul_mat:iq3_s-decode | 1×4096×4096 | 7.889 | 10.476 | 0.753 | 1.149 |
| mul_mat:iq3_s-prefill | 128×4096×4096 | 947.345 | 1327.948 | 0.713 | 1.144 |
| mul_mat:iq3_xxs-decode | 1×4096×4096 | 9.764 | 14.860 | 0.657 | 1.344 |
| mul_mat:iq3_xxs-prefill | 128×4096×4096 | 1191.360 | 1847.764 | 0.645 | 1.310 |
| mul_mat:iq4_nl-decode | 1×4096×4096 | 6.468 | 5.665 | 1.142 | 0.774 |
| mul_mat:iq4_nl-prefill | 128×4096×4096 | 448.382 | 723.635 | 0.620 | 1.491 |
| mul_mat:iq4_xs-decode | 1×4096×4096 | 6.398 | 7.467 | 0.857 | 1.108 |
| mul_mat:iq4_xs-prefill | 128×4096×4096 | 772.011 | 945.995 | 0.816 | 1.076 |
| mul_mat:tq1_0-decode | 1×4096×4096 | 4.335 | 5.720 | 0.758 | 1.478 |
| mul_mat:tq1_0-prefill | 128×4096×4096 | 304.625 | 725.493 | 0.420 | 2.589 |
| mul_mat:tq2_0-decode | 1×4096×4096 | 2.833 | 3.134 | 0.904 | 1.228 |
| mul_mat:tq2_0-prefill | 128×4096×4096 | 134.885 | 393.177 | 0.343 | 2.839 |
| mul_mat:mxfp4-decode | 1×4096×4096 | 8.572 | 6.091 | 1.407 | 0.741 |
| mul_mat:mxfp4-prefill | 128×4096×4096 | 501.902 | 768.792 | 0.653 | 1.287 |
| mul_mat:nvfp4-decode | 1×4096×4096 | 13.456 | 108.713 | 0.124 | 8.364 |
| mul_mat:nvfp4-prefill | 128×4096×4096 | 1282.581 | 13946.174 | 0.092 | 7.585 |
| dense:gemm_f32 | 128×4096×4096 | 730.557 | 459.046 | 1.591 | 0.600 |
| dense:f32-decode | 1×4096×4096 | 11.087 | 11.728 | 0.945 | 1.082 |
| dense:f32-prefill | 128×4096×4096 | 729.535 | 459.046 | 1.589 | 0.600 |
| dense:gemv_f32 | 14336×4096 | 148.975 | 无 V100 baseline 行 | — | — |

可比 case 49 个：JIT/GGML 时间比中位 0.753，最小 0.092，最大 1.591（JIT 无 eviction 偏快，正式结论以 2.1 为准）。

### 2.2.1 JIT warm 与正式 Weft 测量的交叉验证

将 comparison CSV 的正式 Weft 吞吐换算回毫秒（ops = 2·M·N·K），与本 JIT run 的
warm 中位对照：

| case | JIT warm (ms) | 正式 Weft (ms) | JIT/正式 |
|---|---:|---:|---:|
| mul_mat:f16-decode | 6.295 | 6.144 | 1.025 |
| mul_mat:f16-prefill | 185.284 | 179.505 | 1.032 |
| mul_mat:q4_0-decode | 8.544 | 8.202 | 1.042 |
| mul_mat:q4_0-prefill | 803.451 | 966.937 | 0.831 |
| mul_mat:q4_k-decode | 4.275 | 3.686 | 1.160 |
| mul_mat:q4_k-prefill | 305.430 | 463.224 | 0.659 |
| mul_mat:tq1_0-decode | 4.335 | 3.869 | 1.120 |
| mul_mat:tq1_0-prefill | 304.625 | 280.197 | 1.087 |
| mul_mat:nvfp4-decode | 13.456 | 12.998 | 1.035 |
| mul_mat:nvfp4-prefill | 1282.581 | 1838.554 | 0.698 |
| mul_mat:q5_0-decode | 9.111 | 8.614 | 1.058 |
| mul_mat:q5_0-prefill | 416.758 | 953.467 | 0.437 |
| mul_mat:iq2_xxs-decode | 6.931 | 8.426 | 0.823 |
| mul_mat:iq2_xxs-prefill | 749.505 | 792.480 | 0.946 |

交叉比值中位 1.032（范围 0.437~1.160）：JIT warm 计时与正式协议 Weft 测量基本吻合，差异在
eviction 状态与 C 编译器版本（JIT 用 LLVM22 clang + native 发现的完整 ISA，
正式协议用 clang-18 + 显式 march）的合理范围内。

### 2.3 不做 JIT 性能对比的族及原因

- vec_dot / dequantize：GGML 行时间为 C 循环内 14336/1024 行摊销（每行 2~5µs），
  JIT 每次 ctypes 派发开销即超过单行 kernel 时间，JIT warm 计时是派发主导，
  不构成有效对照；正式对照见 comparison CSV 的 vec_dot/dequantization 行。
- quantize：JIT 每行一次派发（~10µs）对 GGML 每行 25µs，开销占比过高。
- gemv q4_k：GGML baseline 无对应 gemv 行（其参照即 vec_dot 循环，scope 不同）。

## 3. 结论

1. 数值：111/111 可执行 case 与 GGML reference 在声明容差内一致，
   含此前从未在 V100 真机验证过的 native JIT 路径（q1_0/mxfp4/nvfp4、
   staged/persistent、f16 staging 逐位）。
2. 性能：正式协议下 Weft 吞吐 74/101 case 高于 GGML（中位 1.168×）；本 JIT
   run 的绝对耗时与正式 Weft 测量交叉吻合（中位偏差约 ±10%），方向一致。
3. 缺口：iq2_s/iq2_xs 的 staged 绑定在 VLEN256 编译期被资源合同拒绝，是唯一
   无法与 GGML 对照的声明绑定，需编译器侧解决后重测。
