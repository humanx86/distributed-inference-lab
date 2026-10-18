# Distributed Inference Lab

A working lab for learning about and researching distributed inference systems, with a long-term focus on cross-layer performance diagnosis, GPU execution, collective communication and sparse workloads, especially Mixture-of-Experts (MoE).

The central question is how memory use, request scheduling, expert routing and placement, and communication interact to shape latency and throughput, and when a runtime policy is worth its overhead. 

Planned experiments progress from hand-written CUDA kernels and single-GPU model execution, through a real serving engine and multi-GPU communication, to expert-parallel MoE and a bounded study of runtime decisions. The aim is that each result rests on correctness checks, controlled measurements and execution traces, and states what it does not establish.

**Status:** Phase 1: model, memory and execution, in progress

**Hardware:** RTX 5070 Ti Laptop GPU (Blackwell, sm_120, 12 GB GDDR7) for development; datacenter and multi-GPU results are added for the later stages

## Why cross-layer diagnosis

In LLM inference, the layer where a slowdown appears is often not the layer that caused it. Increased latency might trace back to scheduler queueing, KV-cache pressure that triggers preemption and recomputation, host-side kernel launch delays, or one rank holding up an expert-parallel all-to-all because routing sent it more tokens than its peers. 

Nsight Systems can place CPU, GPU and NCCL activity on one timeline, and help reconstruct execution across layers. It shows what ran and when, but connecting that activity to individual requests or experts require batch composition, routing information, and runtime annotations. Traces from separate nodes require validated timestamp alignment, and the instrumentation itself can perturb the timings being measured.

Even within one trace, a long communication bar can mean waiting for a late peer rather than moving data. MoE adds a further complication since expert load depends on the input, so a workload that is balanced on one batch can be skewed on the next one. Straggler detection can name the late rank, but at the collective, a rank delayed because routing sent it more tokens looks the same as one with a hardware or OS problem. Telling them apart needs routing data next to the timeline.

Some production systems address this with continuous, low-overhead observability across CPU, GPU and network layers. SysOM-AI [1], a source of inspiration for this lab, correlates CPU stacks (down to the OS kernel), GPU kernels timings, and NCCL collective events across a training fleet of more than 80,000 GPUs. It targets bulk-synchronous training and profiles per rank rather than per request tracing. Distributed inference requires complementary views since request-level measurements capture latency experienced by individual requests, while rank-level measurements expose communication delays and stragglers that can affect multiple requests during tensor or expert-parallel execution.

This lab works at a much smaller scale and explores related diagnosis questions through controlled inference experiments. Production incidents rarely come with ground truth. A small lab can create it by injecting a known cause, such as a hot expert or a delayed rank, and checking whether the diagnosis recovers it against a no-change control.

## References
[1] Y. Zheng et al., "SysOM-AI: Continuous Cross-Layer Performance Diagnosis for Production AI Training," [arXiv:2603.29235](https://arxiv.org/abs/2603.29235)