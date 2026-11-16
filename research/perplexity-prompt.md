# Perplexity deep research prompt

Status: run 2026-11-13. Only the companion code came back (no written report). Its bandwidth-sharing model is now `simulate_shared` in src/; source archived in Perplexity Output/New Output/Archive/.

```
Research how to build and set parameters for a discrete-event simulation of a bulk file-transfer system (FTP-style, possibly called "NAPM"; if NAPM is a known tool, protocol, or framework, identify it and say how it models transfers).

The model: each record is a file transfer request with file size and request timestamp; some datasets also include actual start and end timestamps. The system has a max spec rate (total bandwidth cap), N concurrent transfer threads, and a per-thread rate cap. The goal is to replay thousands to millions of requests and test whether transfer latency (completion minus request time) meets 1-minute and 5-minute SLAs at p90, p95, and p99.

Answer with proven methods, formulas, and sources:
1. The right queueing model: multi-server FIFO (M/G/c or G/G/c) vs processor-sharing with a global bandwidth cap. When each one applies, and how per-thread rate = min(per-thread cap, max rate / active threads) changes the results.
2. How to estimate the parameters from historical data: deriving per-thread throughput from size/(end - start), queue wait from (start - request), and how to use observed start/end times to validate the simulator, not as inputs.
3. Efficient algorithms for millions of records: heap-based event simulation, handling bandwidth sharing efficiently (rate recalculation per event, virtual-time tricks), and Python libraries (SimPy, NumPy, Ciw) with their real performance at this scale.
4. Fast feasibility checks before simulating: the minimum per-thread rate implied by large-file tail sizes, the utilization limit (Little's law, total bytes / window vs max rate), and how tail latency grows as utilization approaches 1.
5. Parameter sweep strategy: searching N and per-thread rate under the total-rate cap to find the minimum config that passes p99 under 60s and under 300s, plus sensitivity analysis.
6. SLA interpretation: per-request latency SLA vs batch-window drain (all requests in a 5-min window finish within it), and how to compute p90/p95/p99 correctly for each (overall vs per-window, exact vs streaming quantiles like t-digest).
7. Pitfalls: synthetic Poisson arrivals hiding bursts, warm-up/edge effects at the start and end of the replay, TCP slow start and per-file overhead on small files, retries, and file-size distributions with heavy tails.
8. Date-range replays: how to pick test windows from a month of timestamped data (peak day, peak hour, full month), warm-up periods so threads aren't empty at the window start, and whether to report SLA percentiles overall, per day, or per peak hour.

Flag it if any assumption above is wrong or missing (for example, if real FTP throughput per thread isn't constant, or if start/end times should be used differently).

Deliver the full answer as a downloadable Markdown (.md) file: a descriptive title, clear headings, tables where they help, and a numbered source list with links at the end.
```
