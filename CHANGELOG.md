# Changelog

## 2026-11-18

- Add the executed record of a full run on Colab's free T4 GPU runtime,
  with every output, and link it from the README. Note the one visible
  difference from the workstation: a Part 9 thinking answer cut off at
  800 tokens.
- Add the executed record of a full run on a Colab A100 runtime. The same
  Part 9 thinking answer cut off there too, so the README now says the
  difference is between Colab and the workstation, not the T4 alone.

## 2026-11-15

- Link the companion Cosmos Reason 2 guide near the top of the notebook
  and the README, with a direct Colab link, so a reader can open either
  guide first.
- The author line names both models run on the Jetson AGX Thor: Cosmos
  Reason 2 and Cosmos3-Edge.
- Record the Colab A100 run: every cell ran. The About line and the README
  now name both Colab runtimes, T4 and A100.
- Record the first Colab run: every cell ran on the T4 GPU runtime. The
  About line now names Colab and the workstation; the README drops "not run
  on Colab yet".
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
