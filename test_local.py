"""W4A16 GEMV 本地正确性测试。

把本文件放在 setup.py 和编译生成的 .so 所在目录，执行：
    python local_test.py

Python 反量化仅用于构造独立的 FP32 参考结果。
测试不调用 torch.mm，也不显式同步 CUDA 设备。
"""

import argparse

import torch
import torch.nn.functional as F

import custom_w4a16_gemv


GROUP_SIZE = 128

# 每项为 (K, N)。小形状覆盖单行、多组和 N 不能被 128 整除的情况。
CASES = [
    (128, 1),
    (256, 5),
    (384, 129),
    (640, 257),
    (4096, 4096),
    (4096, 11008),
    (11008, 4096),
]


def make_inputs(k, n, device, seed):
    """生成题目要求的四个输入；每组的 S、Z 都可以不同。"""
    torch.manual_seed(seed)
    x = torch.randn((1, k), device=device, dtype=torch.float16)

    # 随机字节的低 4 位、高 4 位分别代表两个 0..15 的量化值。
    w_q = torch.randint(
        0, 256, (n, k // 2), device=device, dtype=torch.uint8
    )

    groups = k // GROUP_SIZE
    scales = (
            torch.rand((n, groups), device=device, dtype=torch.float32)
            * 0.045 + 0.005
    ).to(torch.float16)

    # Z 按题意是 FP16，因此也测试包含小数的零点。
    zeros = (
            torch.rand((n, groups), device=device, dtype=torch.float32) * 15.0
    ).to(torch.float16)

    return x, w_q, scales, zeros


def reference_fp32(x, w_q, scales, zeros):
    """独立参考：解包 -> FP32 反量化 -> 逐元素乘法并按 K 求和。"""
    n = w_q.size(0)
    k = x.size(1)

    # 恢复 [N, K] 的逻辑量化矩阵，偶数列用低半字节。
    q = torch.empty((n, k), device=w_q.device, dtype=torch.float32)
    q[:, 0::2] = (w_q & 0x0F).float()
    q[:, 1::2] = ((w_q >> 4) & 0x0F).float()

    # [N, K/G, G] 让每组的 128 个权重共用同一个 S、Z。
    q_grouped = q.reshape(n, k // GROUP_SIZE, GROUP_SIZE)
    weights = (
            (q_grouped - zeros.float().unsqueeze(-1))
            * scales.float().unsqueeze(-1)
    ).reshape(n, k)

    # x.float() 的 [1, K] 会广播到每一行权重。
    return (weights * x.float()).sum(dim=1).unsqueeze(0)


def run_case(k, n, device, seed, label):
    x, w_q, scales, zeros = make_inputs(k, n, device, seed)

    y_pred = custom_w4a16_gemv.gemv_w4a16(
        x, w_q, scales, zeros, group_size=GROUP_SIZE
    )
    y_ref = reference_fp32(x, w_q, scales, zeros)

    assert isinstance(y_pred, torch.Tensor), "输出必须是 Tensor"
    assert y_pred.shape == (1, n), f"输出形状错误：{y_pred.shape}"
    assert y_pred.dtype == torch.float16, f"输出类型错误：{y_pred.dtype}"
    assert y_pred.device == x.device, f"输出设备错误：{y_pred.device}"
    assert y_pred.is_contiguous(), "输出必须 Contiguous"

    pred_float = y_pred.float()
    error = (pred_float - y_ref).abs()
    cosine = F.cosine_similarity(pred_float, y_ref, dim=-1).mean().item()

    # .item() 会等待相关结果可读；不需要 cudaDeviceSynchronize。
    print(
        f"[{label}] K={k}, N={n}, seed={seed}: "
        f"max_abs={error.max().item():.6g}, "
        f"mean_abs={error.mean().item():.6g}, cosine={cosine:.8f}",
        flush=True,
    )

    # 严格使用题目的两项精度门槛，不因当前 Kernel 而放宽。
    torch.testing.assert_close(
        pred_float, y_ref, rtol=1e-2, atol=1e-2
    )
    assert cosine > 0.999, f"余弦相似度未达标：{cosine}"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="W4A16 GEMV 正确性测试")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("本测试需要支持当前 PyTorch 的 CUDA GPU")

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("--device 必须指定 CUDA 设备，例如 cuda:0")

    failed = 0
    total = len(CASES) + 1
    with torch.cuda.device(device):
        for index, (k, n) in enumerate(CASES):
            try:
                run_case(k, n, device, args.seed + index, "current stream")
            except AssertionError as error:
                failed += 1
                print(f"FAIL: {error}", flush=True)
            else:
                print("PASS", flush=True)

        # 在非默认流上生成输入并调用算子，检查当前流的基本使用。
        # 随机数据操作与算子应进入同一条流；无需显式设备同步。
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            try:
                run_case(4096, 129, device, args.seed + len(CASES),
                         "non-default stream")
            except AssertionError as error:
                failed += 1
                print(f"FAIL: {error}", flush=True)
            else:
                print("PASS", flush=True)

    print(f"\n结果：{total - failed}/{total} 通过", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
