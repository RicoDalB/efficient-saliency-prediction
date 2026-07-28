# Efficient Visual Saliency Prediction  
## Final Evaluation Insights and Report-Writing Plan

This document was extracted from the completed notebook:

`06_final_evaluation(2).ipynb`

The notebook ran all **54 code cells**, evaluated all four methods on the frozen **2,500-image internal test split**, created the **625-image off-centre subset**, completed the efficiency benchmark, and ended with:

> `All required final-evaluation artifacts exist.`

Therefore, the experimental phase is complete. The next main activity is to interpret these results correctly and write the final six-page report.

---

# 1. Experimental setup that produced these results

## Evaluated methods

| Method | Encoder | Decoder | Purpose |
|---|---|---|---|
| MeanMap | None | None | Fixed centre-prior baseline |
| Light-S | MobileNetV2 | Single-scale | Lightweight learned baseline |
| Light-M | MobileNetV2 | Multi-scale | Main proposed compact model |
| Heavy-M | ResNet-18 | Same multi-scale design | Accuracy-oriented reference |

## Frozen evaluation conditions

- Test samples: **2,500**
- Off-centre samples: **625**
- Input/output size: **192 × 256**
- Decoder width: **32 channels**
- Split seed: **42**
- Accuracy device: **Tesla T4 GPU**
- CPU latency protocol:
  - batch size 1;
  - one CPU thread;
  - 30 warm-up forwards;
  - 200 timed forwards;
  - three repeated timing loops;
  - model forward only, excluding data loading and preprocessing.

## Selected checkpoints

| Model | Best checkpoint epoch | Best validation CC |
|---|---:|---:|
| Light-S | 5 | 0.873851 |
| Light-M | 6 | 0.882194 |
| Heavy-M | 4 | 0.883701 |

The validation-selected checkpoints were correctly used instead of the final training epochs.

---

# 2. Main internal-test results

| Model | KLD ↓ | CC ↑ | SIM ↑ |
|---|---:|---:|---:|
| MeanMap | 0.720233 | 0.556980 | 0.552755 |
| Light-S | 0.342777 | 0.802880 | 0.711626 |
| Light-M | 0.257657 | 0.862401 | 0.753158 |
| Heavy-M | **0.224665** | **0.883490** | **0.775449** |

## Immediate conclusion

The ranking is consistent across all three metrics:

1. **Heavy-M** gives the best accuracy.
2. **Light-M** is second.
3. **Light-S** is third.
4. **MeanMap** is far behind the learned models.

This is an unusually clean result because no metric reverses the overall ranking on the complete test set.

---

# 3. What multi-scale fusion contributed

The most important controlled comparison is:

> **Light-M versus Light-S**

Both models use MobileNetV2. Their central architectural difference is the use of multi-scale feature fusion.

## Accuracy improvement

| Change from Light-S to Light-M | Result |
|---|---:|
| KLD | **−0.085120** |
| Relative KLD reduction | **24.83%** |
| CC | **+0.059521** |
| Relative CC increase | **7.41%** |
| SIM | **+0.041532** |
| Relative SIM increase | **5.84%** |

These are substantial gains, especially because the encoder remains unchanged.

## Per-image evidence

Light-M was better than Light-S on:

| Metric | Fraction of the 2,500 test images |
|---|---:|
| KLD | **85.56%** |
| CC | **87.92%** |
| SIM | **83.96%** |

This is stronger evidence than comparing only the dataset means. The improvement is not caused by a small number of unusually favorable examples: Light-M wins on most test images.

## Cost of multi-scale fusion

| Cost measure | Light-S | Light-M | Change |
|---|---:|---:|---:|
| Parameters | 1.826 M | 1.831 M | **+0.27%** |
| MACs | 0.306 G | 0.309 G | **+1.22%** |
| CPU median latency | 23.78 ms | 30.63 ms | **+28.78%** |
| Estimated CPU FPS | 42.05 | 32.65 | −22.35% |

## Interpretation

The multi-scale decoder adds almost no parameters and very few theoretical operations, but the measured latency increase is larger than the MAC increase.

This is important:

- parameter count alone does not predict real latency;
- feature resizing, skip additions, memory movement, and extra decoder stages can affect wall-clock time;
- the model remains relatively fast, but multi-scale fusion is not computationally free.

## Hypothesis verdict

> **H1 — Light-M improves over Light-S:** strongly supported.

> **H2 — Light-M adds only modest cost:** supported for parameters and MACs, but only partially supported for actual CPU latency.

---

# 4. What the heavier encoder contributed

The second controlled comparison is:

> **Heavy-M versus Light-M**

Both models use the multi-scale design. The main difference is ResNet-18 versus MobileNetV2.

## Accuracy improvement

| Change from Light-M to Heavy-M | Result |
|---|---:|
| KLD | **−0.032992** |
| Relative KLD reduction | **12.80%** |
| CC | **+0.021089** |
| Relative CC increase | **2.45%** |
| SIM | **+0.022291** |
| Relative SIM increase | **2.96%** |

Heavy-M improves every metric, but the absolute gain is smaller than the gain produced by adding multi-scale fusion to the lightweight model.

## Per-image evidence

Heavy-M was better than Light-M on:

| Metric | Fraction of test images |
|---|---:|
| KLD | **71.24%** |
| CC | **70.24%** |
| SIM | **73.08%** |

The larger encoder therefore gives a real and repeatable advantage, but Light-M still wins on roughly 27–30% of individual images.

## Cost of the heavy encoder

| Cost measure | Light-M | Heavy-M | Relative cost |
|---|---:|---:|---:|
| Parameters | 1.831 M | 11.211 M | **6.12×** |
| State dictionary | 7.11 MB | 42.80 MB | **6.02×** |
| MACs | 0.309 G | 1.803 G | **5.83×** |
| CPU median latency | 30.63 ms | 61.86 ms | **2.02×** |
| Estimated CPU FPS | 32.65 | 16.17 | about half |

## Interpretation

Heavy-M is the accuracy winner, but its extra accuracy is expensive.

For approximately:

- six times the parameters;
- almost six times the MACs;
- twice the CPU latency;

it provides:

- +0.021 CC;
- +0.022 SIM;
- −0.033 KLD.

This is the central accuracy–efficiency trade-off of the project.

## Hypothesis verdict

> **H3 — Heavy-M may be more accurate, while Light-M offers the stronger practical trade-off:** supported.

Heavy-M should be described as the **accuracy reference**, while Light-M should be described as the **best compact compromise**.

---

# 5. Comparison against the MeanMap centre prior

All learned models strongly outperform MeanMap.

## Full-test improvement over MeanMap

| Model | Relative KLD reduction | CC gain | SIM gain |
|---|---:|---:|---:|
| Light-S | 52.41% | +0.245900 | +0.158871 |
| Light-M | 64.23% | +0.305421 | +0.200403 |
| Heavy-M | 68.81% | +0.326510 | +0.222694 |

This demonstrates that the networks are not merely reproducing the average spatial prior.

However, the stronger evidence comes from the off-centre analysis.

---

# 6. Off-centre stress test

The top 25% of test images by ground-truth centre-of-mass distance were selected:

- full test: **2,500 images**;
- off-centre subset: **625 images**.

## Full versus off-centre metrics

| Model | Full KLD | Off-centre KLD | Full CC | Off-centre CC | Full SIM | Off-centre SIM |
|---|---:|---:|---:|---:|---:|---:|
| MeanMap | 0.720233 | 1.007342 | 0.556980 | 0.431390 | 0.552755 | 0.458464 |
| Light-S | 0.342777 | 0.392787 | 0.802880 | 0.804567 | 0.711626 | 0.689402 |
| Light-M | 0.257657 | 0.297532 | 0.862401 | 0.868549 | 0.753158 | 0.733768 |
| Heavy-M | 0.224665 | 0.262321 | 0.883490 | 0.887278 | 0.775449 | 0.755766 |

## MeanMap collapse

On the off-centre subset, MeanMap changes by:

- KLD: **+39.86%**
- CC: **−22.55%**
- SIM: **−17.06%**

This is exactly the behavior expected from a fixed central prediction.

## Learned-model robustness

The learned models show:

- KLD increases of approximately 14.6–16.8%;
- SIM reductions of only approximately 2.5–3.1%;
- CC that remains stable or even increases slightly.

The slight CC increase does **not** mean off-centre images are universally easier. It means CC and distribution metrics react differently:

- CC measures linear spatial agreement;
- KLD penalizes probability-mass mismatch;
- SIM measures distribution overlap.

An image can contain one very localized off-centre target that is highly correlated with the prediction, while the prediction still distributes too much probability elsewhere. This can increase CC while worsening KLD and SIM.

## Important interpretation

The off-centre subset is not necessarily a universally harder subset for every neural model.

It is a targeted stress test that specifically reduces the usefulness of the central spatial prior. The result shows:

- MeanMap depends strongly on the prior;
- learned models use image content;
- Heavy-M remains the best off-centre model;
- Light-M preserves almost all of Heavy-M’s robustness at much lower cost.

## Hypothesis verdicts

> **H4 — MeanMap degrades strongly on off-centre scenes:** strongly supported.

> **H5 — Multi-scale fusion improves off-centre robustness:** mostly supported.

Light-M has a smaller KLD increase and a smaller SIM decrease than Light-S. CC is stable for both, so CC alone should not be used to claim stronger robustness.

---

# 7. Training behavior

The combined training curves show a consistent pattern.

## Light-S

- Training loss continues decreasing through the final epoch.
- Validation loss reaches its minimum around epochs 4–5 and then rises.
- Validation CC and SIM plateau after the early epochs.
- The best checkpoint was selected at epoch 5.

This suggests mild overfitting after the best epoch.

## Light-M

- The model converges quickly.
- Validation metrics improve until approximately epoch 6.
- Training loss continues decreasing while validation loss later begins increasing.
- The best checkpoint was selected at epoch 6.

This indicates that checkpoint selection successfully avoided the later overfitting region.

## Heavy-M

- The model reaches its best validation CC earlier, at epoch 4.
- Training loss continues dropping strongly afterward.
- Validation loss begins rising after the optimum.
- The best checkpoint was selected at epoch 4.

The higher-capacity model fits the training data faster and begins overfitting earlier.

## General conclusion from the curves

The curves justify:

- validation-based checkpoint selection;
- limited epoch budgets;
- early stopping;
- using `best.pt` rather than `last.pt`.

Do not report that the models “did not overfit.” A more accurate statement is:

> All models converged rapidly, and a widening train–validation loss gap appeared after the best validation epoch, particularly for Heavy-M. Selecting checkpoints by validation CC prevented the final comparison from using later overfitted states.

---

# 8. Validation-to-test behavior

| Model | Best validation CC | Test CC | Difference |
|---|---:|---:|---:|
| Light-S | 0.873851 | 0.802880 | −0.070971 |
| Light-M | 0.882194 | 0.862401 | −0.019793 |
| Heavy-M | 0.883701 | 0.883490 | −0.000211 |

Light-S shows a much larger validation-to-test decrease than the multi-scale models.

This may indicate that the single-scale model generalizes less reliably to the held-out test split. However, avoid claiming that this proves overfitting by itself because the validation and test subsets may differ in composition.

A careful report sentence is:

> The validation-to-test CC gap was largest for Light-S, whereas Heavy-M reproduced its validation score almost exactly. This suggests stronger held-out robustness for the multi-scale models, although split-specific scene composition may also contribute to the difference.

---

# 9. Qualitative examples

The notebook selected five cases using objective criteria.

## 1. Most-central target — fruit display

Ground truth distributes attention across several product groups.

- MeanMap predicts only a broad central blob.
- Light-S identifies some product regions but overemphasizes the right side.
- Light-M distributes attention over more relevant regions.
- Heavy-M gives the closest multi-region structure.

This example shows why semantic and local information are both useful even when the target centre is near the image centre.

## 2. Most-off-centre target — skier

The salient person is near the lower-right area.

- MeanMap completely misses the target.
- All neural models move their main peak toward the person.
- Light-M obtains very high CC on this image.
- The models still produce some diffuse background activation, which explains why KLD and SIM can disagree with CC.

This is one of the clearest visual demonstrations that the neural models use image content.

## 3. Median Light-M case — people eating

- All neural models identify the principal face/person region.
- The table and secondary people receive weaker attention.
- Heavy-M produces the best numerical result.
- Light-M remains visually close while using a much smaller encoder.

This is a good representative example because it is not chosen from an extreme.

## 4. Worst Light-M case — person near furniture/television

- The target contains several separated salient regions.
- Light-M misses part of the target structure.
- Light-S actually has higher CC than Light-M on this specific image.
- Heavy-M recovers the scene more successfully.

This example is valuable because it shows that multi-scale fusion does not improve every image.

## 5. Largest Light-M gain — airplanes

- The salient regions are spread horizontally and away from the centre.
- Light-S misses much of this structure.
- Light-M detects multiple important regions.
- Heavy-M improves the localization further.

This is strong qualitative evidence for multi-scale spatial fusion.

## General qualitative conclusion

The images support the quantitative findings:

- MeanMap fails whenever attention is not central.
- Light-S often produces simpler or less complete spatial structures.
- Light-M better captures multiple separated salient regions.
- Heavy-M generally refines localization and suppresses mistakes, but not on every image.
- Failure cases remain, especially when attention is divided among several small regions or when the scene has unusual composition.

---

# 10. Main scientific conclusion

A defensible central conclusion is:

> Multi-scale fusion provides the largest architecture-level improvement in this controlled study. Light-M substantially outperforms the single-scale MobileNetV2 baseline while adding only 0.27% parameters and 1.22% MACs, although measured CPU latency increases by 28.8%. Heavy-M achieves the highest KLD, CC, and SIM performance, but requires about 6.1 times the parameters, 5.8 times the MACs, and twice the CPU latency of Light-M. Therefore, Heavy-M is the accuracy winner, whereas Light-M offers the strongest accuracy–efficiency compromise.

The second major conclusion is:

> The MeanMap baseline deteriorates sharply on the off-centre subset, while all learned models remain stable. This confirms that the neural models learn image-dependent attention cues instead of relying only on the SALICON centre prior.

---

# 11. Hypothesis summary

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: Light-M improves over Light-S | **Supported** | Better KLD, CC, and SIM; wins on 84–88% of images |
| H2: Multi-scale cost is modest | **Partially supported** | +0.27% parameters and +1.22% MACs, but +28.8% CPU latency |
| H3: Heavy-M is more accurate but Light-M has a better trade-off | **Supported** | Heavy-M wins accuracy; Light-M is about 6× smaller and 2× faster |
| H4: MeanMap collapses off-centre | **Strongly supported** | CC −22.5%, SIM −17.1%, KLD +39.9% |
| H5: Multi-scale fusion improves off-centre robustness | **Mostly supported** | Light-M has smaller KLD/SIM degradation than Light-S; CC is stable for both |

---

# 12. Claims that are safe to make

You can claim:

- this is a controlled comparison on the selected SALICON split;
- multi-scale fusion improved all primary metrics;
- Heavy-M gave the best absolute accuracy;
- Light-M gave the strongest compact trade-off;
- MeanMap strongly depended on centre bias;
- learned models retained performance on off-centre scenes;
- theoretical cost and measured latency did not scale identically;
- the larger model began overfitting earlier.

---

# 13. Claims to avoid

Do not claim:

- “state of the art”;
- “real-time on every device”;
- “Light-M is universally the best model”;
- “the off-centre subset is harder in every possible sense”;
- “CC alone proves complete robustness”;
- “the models understand human vision”;
- “multi-scale fusion improves every image”;
- “Heavy-M is unnecessary”;
- “the results generalize to every saliency dataset”;
- statistical significance, because no confidence interval or formal hypothesis test was performed;
- seed robustness, because each model was trained once.

---

# 14. The reasoning workflow for writing the report

This is not private hidden reasoning. It is the explicit scientific workflow you should follow while constructing the paper.

## Step 1 — Begin from one research question

Use one clear question:

> Can compact multi-scale fusion improve visual-saliency prediction while preserving a useful efficiency advantage over a conventional heavier CNN, and do the learned models remain effective when the dataset centre prior is less useful?

Every report section should help answer part of this question.

## Step 2 — Separate the question into controlled comparisons

The report narrative should follow this order:

1. **MeanMap versus learned models**  
   Do the models use image content?

2. **Light-S versus Light-M**  
   What is the value of multi-scale fusion under the same encoder?

3. **Light-M versus Heavy-M**  
   What is the value and cost of increasing encoder capacity?

4. **Full test versus off-centre test**  
   How much do the methods rely on the centre prior?

5. **Accuracy versus efficiency**  
   Which model is the most useful practical compromise?

This order produces a logical argument rather than a list of unrelated results.

## Step 3 — Build every claim from evidence

Use this paragraph structure repeatedly:

1. state the observation;
2. give the exact numerical evidence;
3. explain what architectural factor changed;
4. explain the cost or limitation;
5. state the careful conclusion.

Example:

> Light-M improved test CC from 0.8029 to 0.8624 and reduced KLD from 0.3428 to 0.2577 relative to Light-S. Since both models use the same MobileNetV2 encoder, this improvement isolates the contribution of multi-scale fusion. The additional decoder structure increased parameters by only 0.27% and MACs by 1.22%, although CPU latency increased by 28.8%. Multi-scale fusion therefore produced a strong accuracy gain with negligible model-size growth but a measurable runtime cost.

## Step 4 — Use one result for one purpose

Do not make every table prove everything.

- Main metric table: accuracy ranking.
- Paired per-image table: consistency of improvement.
- Parameter/MAC/latency table: efficiency.
- Pareto figure: accuracy–efficiency trade-off.
- Centre-bias table: robustness to the spatial prior.
- Qualitative panel: visual behavior and failure modes.
- Training curves: convergence and checkpoint selection.

## Step 5 — Interpret metric disagreement correctly

When CC, KLD, and SIM behave differently:

- do not hide the disagreement;
- explain what each metric measures;
- avoid inventing an aggregate score;
- use qualitative maps to clarify the difference.

The off-centre result is the best example: learned-model CC slightly improves, while KLD and SIM become somewhat worse.

## Step 6 — Distinguish accuracy winner from recommended model

The report should explicitly separate:

- **best absolute accuracy:** Heavy-M;
- **best compact trade-off:** Light-M;
- **fastest learned model:** Light-S;
- **non-learned spatial prior:** MeanMap.

This prevents a confused conclusion.

## Step 7 — Discuss failures before the conclusion

A convincing report should include:

- Light-M does not win every image;
- Heavy-M is expensive;
- latency increases more than MACs predict;
- the evaluation uses one dataset and one split;
- each model uses one training seed;
- no public SALICON hidden-test evaluation was performed;
- the off-centre subset is defined from target geometry, not a new independent dataset.

This honesty strengthens the report.

## Step 8 — Write the abstract last

The abstract should contain:

1. the visual-saliency problem;
2. the efficiency motivation;
3. the four compared methods;
4. the SALICON evaluation;
5. the most important numerical result;
6. the final trade-off conclusion.

Do not write it until the Results and Conclusion sections are stable.

---

# 15. Recommended six-page report structure

## Title

**Efficient Multi-Scale CNNs for Visual Saliency Prediction**

Alternative:

**Accuracy–Efficiency Trade-offs in Multi-Scale Visual Saliency Prediction**

## Abstract

Approximately 150–200 words.

Include:

- problem;
- compact multi-scale proposal;
- three CNNs plus MeanMap;
- KLD, CC, SIM, parameters, MACs, latency, and off-centre test;
- Heavy-M best accuracy;
- Light-M best compact trade-off;
- MeanMap collapse off-centre.

## I. Introduction and Related Work

Answer:

- What is fixation prediction?
- Why do deep and shallow features complement each other?
- Why is efficiency important?
- Why can centre bias mislead evaluation?
- What exactly does this project contribute?

End with a compact contribution statement.

## II. Dataset and Processing Pipeline

Describe:

- SALICON images and continuous maps;
- deterministic split;
- 10,000 training pairs, 2,500 validation, 2,500 internal test;
- resize to 192 × 256;
- ImageNet RGB normalization;
- target normalization to unit mass;
- fixed local-cache implementation;
- KLD, CC, and SIM.

Do not spend excessive space on Colab engineering details. Mention the cache as a reproducibility and throughput measure, not as a scientific contribution.

## III. Models and Training

Describe:

- MeanMap;
- MobileNetV2 feature taps;
- Light-S deepest-feature decoding;
- Light-M multi-scale fusion;
- Heavy-M ResNet-18 with the same multi-scale decoder;
- common decoder width;
- common training loss;
- AdamW and checkpoint selection by validation CC;
- fair-comparison controls.

Include one architecture diagram.

## IV. Experimental Results

This is the main section.

### A. Accuracy

Present the main KLD/CC/SIM table.

Discuss:

1. learned models versus MeanMap;
2. Light-M versus Light-S;
3. Heavy-M versus Light-M;
4. per-image win fractions.

### B. Efficiency

Present:

- parameters;
- state size;
- MACs;
- CPU median latency.

Explain that Light-M’s latency increase is larger than its MAC increase.

### C. Accuracy–efficiency trade-off

Use the Pareto figure.

State clearly:

- Heavy-M is the accuracy winner;
- Light-M is the strongest compact compromise;
- Light-S is fastest but loses substantial accuracy.

## V. Centre-Bias and Qualitative Analysis

Present:

- full versus off-centre table or plot;
- MeanMap collapse;
- learned-model stability;
- one central example;
- one off-centre example;
- one representative case;
- one failure case;
- one multi-scale improvement case.

Explain metric disagreement where appropriate.

## VI. Conclusion

Answer the original question directly.

Suggested logic:

1. multi-scale fusion is worthwhile;
2. Heavy-M improves accuracy but at high cost;
3. Light-M is the recommended compact model;
4. learned models are not simply exploiting centre bias;
5. future work should include multiple seeds, more datasets, distillation, pruning, or quantization.

---

# 16. Suggested final-results table

Use a compact table similar to this:

| Model | KLD ↓ | CC ↑ | SIM ↑ | Params (M) | MACs (G) | CPU ms ↓ |
|---|---:|---:|---:|---:|---:|---:|
| MeanMap | 0.720 | 0.557 | 0.553 | 0 | 0 | — |
| Light-S | 0.343 | 0.803 | 0.712 | 1.826 | 0.306 | 23.78 |
| Light-M | 0.258 | 0.862 | 0.753 | 1.831 | 0.309 | 30.63 |
| Heavy-M | **0.225** | **0.883** | **0.775** | 11.211 | 1.803 | 61.86 |

Use three decimal places in the report table. Keep the full precision in CSV files.

---

# 17. Suggested centre-bias table

| Model | Full CC ↑ | Off-centre CC ↑ | Full SIM ↑ | Off-centre SIM ↑ |
|---|---:|---:|---:|---:|
| MeanMap | 0.557 | 0.431 | 0.553 | 0.458 |
| Light-S | 0.803 | 0.805 | 0.712 | 0.689 |
| Light-M | 0.862 | 0.869 | 0.753 | 0.734 |
| Heavy-M | **0.883** | **0.887** | **0.775** | **0.756** |

Add KLD in the text or a second compact table if space allows.

---

# 18. Suggested contribution paragraph

> This work presents a controlled study of accuracy and computational efficiency in CNN-based visual saliency prediction. A lightweight single-scale MobileNetV2 baseline is compared with a compact multi-scale model using the same encoder and with a heavier ResNet-18 reference using the same multi-scale design. All learned models are trained and evaluated using identical data splits, preprocessing, loss, checkpoint selection, and metrics. In addition to KLD, CC, and SIM, the study measures parameters, operation counts, and CPU latency. A MeanMap baseline and a ground-truth-defined off-centre subset are further used to quantify dependence on the SALICON centre prior.

---

# 19. Suggested conclusion paragraph

> The results show that multi-scale fusion is the most effective compact architectural change in the controlled comparison. Light-M improves CC from 0.803 to 0.862 over Light-S while increasing parameters by only 0.27% and MACs by 1.22%, although its measured CPU latency is 28.8% higher. Heavy-M achieves the best absolute performance, with a CC of 0.883, but requires approximately 6.1 times the parameters, 5.8 times the MACs, and twice the latency of Light-M. The off-centre analysis further shows that MeanMap deteriorates sharply, whereas the learned models remain stable, confirming that their predictions depend on image content rather than only the dataset centre prior. These findings identify Light-M as the strongest accuracy–efficiency compromise in the evaluated setting.

---

# 20. Remaining practical actions before submission

The notebook’s final Git status showed:

- `src/models.py` modified because the temporary `farward → forward` correction was applied;
- report figures and tables were created but are still untracked.

Before submission:

1. permanently correct `forward` in the GitHub source;
2. place `06_final_evaluation.ipynb` in `notebooks/`;
3. review the generated report figures;
4. commit the final report tables and selected figures;
5. update the README with:
   - project question;
   - model definitions;
   - notebook order;
   - final results;
   - reproduction instructions;
6. write the six-page report;
7. verify that no SALICON data or TAR archive is committed;
8. prepare the final source-code package;
9. compile and proofread the final PDF;
10. submit only after checking the professor’s exact upload requirements.

No additional model training is required unless you discover a genuine implementation error. The present results already form a coherent and complete experimental story.
