# Changelog

## 2026-11-15

- Add background to the README: why Cosmos 3, what is the same and what
  is different from Cosmos Reason 2, what it does better and what it costs,
  and a hardware profile with measured figures and NVIDIA's Jetson figures.
- Add the notebook, its plain-text source and the builder. The tests and
  media are those of the Cosmos Reason 2 first-timers guide, run on the
  Cosmos3-Edge reasoner.
- Add two executed records from `rtx-5090-workstation`. The scores are the
  same in both. The second has slower timings from the middle onward because
  another job was using the CPU.
- Add the vLLM 0.33.0 server test script and its two outputs, the evidence
  for Part 17.
- Add the README with measured results, a comparison with Cosmos Reason 2
  (2B) on the identical tests, the issues log and what is not established.
- Add two charts from the first record to the top of the README.
- Add the repository hygiene test and a test that the notebook matches its
  source.
