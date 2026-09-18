# V100 Native JIT 验收套件

## 1. 定位

套件对 `weft.compile` / `weft.jit` 的本机原生路径做数值验收：同一编译主链在目标机
内完成 DSL → Canonical Kernel IR → RISC-V Physical IR → intrinsic C → shared
object → 加载 → 调用。测试面以 `examples/kernels/*/tuning.json` 中声明为 v100 的
绑定为唯一权威；未被任何 target 声明绑定的 kernel 文件按 `no-v100-binding` 审计
报告，不做静默跳过。

套件服务于原生调用/ABI 的数值验收，不替代 production 性能 case。正式性能对比走
remote runner 通道（`doc/experiments/` 合同与既有对照 CSV）；本套件的 warm 计时
含义见 §4，不得与 protocol kernel-only timing 混用。

## 2. 入口

```bash
# 构建编译器并查询本机 target（cmake → weft-compile → --query-native-target）
examples/run/weft-jit-build.sh

# 运行：<target> <case-id|family|all> [repeat]，repeat 默认 10
examples/run/weft-jit.sh v100 all 10
examples/run/weft-jit.sh v100 mul_mat 10
examples/run/weft-jit.sh v100 dense:gemv_f32 1
```

`weft-jit.sh` 在 riscv64 本机直接执行；在开发机上经 SSH 远程执行，两端共用同一
入口。目标机环境合同集中在 `examples/run/targets/v100-native-jit.env`：仓库根、
分支（v100-jit）、构建目录、clang 路径、GGML 库路径、taskset core、结果目录。
脚本带分支守卫：仓库不在此分支上时拒绝执行。

## 3. 生命周期与判定

每个可执行 case：

```text
weft.compile（计时）→ 构建声明输入 → cold 调用（计时）
→ 数值/逐位对照 → warm × repeat（计时）
```

判定分三类：

- `numeric`：浮点输出对照，容差沿用对应 remote runner runtime 的声明值
  （`1e-4 + 2e-3·|expected|`；gemv q4_k 采用其专属绝对容差 `1.25e-1`）；
- `bitexact`：量化输出逐位一致（q8 系 quantize）；
- `no-v100-binding`：unbound kernel 文件审计行，只报告绑定状态，无可执行数值。

量化输入与 reference 由 GGML 经 ctypes 生成（iq/tq/mxfp4/nvfp4 查表参数取自
GGML 头文件编译的表库）；GGML 仅在 repro 中提供参考输入/输出，不进入 runtime。

套件级检查覆盖三件事：相同绑定进程内复用、不同绑定专门化、Buffer ABI 合同
（readonly output、wrong rank、wrong encoding、alias overlap、short storage、
misaligned 六条异常路径必须按合同报错）。

## 4. 计时边界

套件记录 `compile_ms` / `cold_call_ms` / `warm_median_ms` 三个独立边界：
`warm_median_ms` 是进程内 warm 整调用墙钟（含 ctypes 派发开销，无 cache
eviction）。对整块调用 case（mul_mat/dense/gemv）它近似 kernel 开销；对单行
kernel case（vec_dot/dequantize/quantize）它由 ctypes 派发主导，**不是** kernel
性能，不得与 GGML baseline 或正式协议数字直接对比。JIT 编译与加载耗时不混入
kernel-only timing。

## 5. 结果产物

每次运行写入 `WEFT_RESULT_ROOT`（默认 `/tmp/weft-native-jit`）下以时间戳命名的
目录：

```text
result.json        全量结构化结果（schema_version/run_id/repeat/cases/suite_checks/summary/target）
summary.csv        逐 case 一行：case, family, verdict_kind, status,
                   max_absolute_error, compile_ms, cold_call_ms, warm_median_ms
suite.json         套件级检查结果
audit.json         测试面审计（selected_bindings / unbound_files / total_kernel_files）
target.json        本机 target（march / vlen_bits / abi / cpus）
environment.json   kernel 与 python 版本
cases/*.json       逐 case 明细（PASS 详情 / FAIL 错误原文）
logs/              环境检查日志
```

测试面规模：v100 声明绑定 117 项（kernel 7、mul-mat 62、vec-dot 24、
row-dequantize 24）+ unbound 审计 13 项，合计 130 行。

## 6. 已知缺口

iq2_s / iq2_xs 的 prefill 与 staged 绑定（共 6 项）在 VLEN256 下无合法
full-product carrier，编译期被资源合同拒绝，不存在可执行数值；GGML baseline 与
正式对照表中同样没有这些 staged 变体的测量行。该缺口需编译器侧解决后重测，
套件不做任何绕过。
