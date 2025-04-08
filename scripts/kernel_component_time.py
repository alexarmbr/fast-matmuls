# compute_time: 37047121 cycles, epilogue_time: 191291 cycles

compute_time = 37047121
epilogue_time = 191291
gpu_clock_rate = 1590 * 1e6 # 1590 MHz

compute_time_s = compute_time / gpu_clock_rate
epilogue_time_s = epilogue_time / gpu_clock_rate

compute_time_ms = compute_time_s * 1e3
epilogue_time_ms = epilogue_time_s * 1e3

print(f"compute_time: {compute_time_ms} ms, epilogue_time: {epilogue_time_ms} ms")




