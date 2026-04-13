# MoE + Diversity in Robotics
## A Research Landscape & Directions for ICML 2026

**Prepared for:** VLA MOE Diversity Team (ECE 534, Princeton University) – Rhiaan, Tharun, Chirayu, Shlok, Ishaan
**Date:** April 9, 2026
**GitHub:** tharunkumartk/vla-moe-diversity

**Document Overview:** This brief synthesizes the latest research on Mixture-of-Experts (MOE) architectures with diversity objectives for Vision-Language-Action models in robotics. It covers critical findings on orthogonality loss limitations, MoE-VLA architectures, diverse skill learning, and expert diversity mechanisms. The final sections provide actionable recommendations for your project, a compute-efficient experiment plan, and ICML 2026 workshop submission guidance.

---

## 1. Executive Summary

The landscape of MoE-augmented Vision-Language-Action (VLA) models has expanded rapidly since late 2025. At least eight distinct MoE-VLA architectures have been published, with approaches ranging from action-head MoE (AdaMoE, DITEA) to layer-skipping (MoLe-VLA), federated learning (FedVLA), and model merging (MergeVLA). Concurrently, several papers have proposed diversity-promoting mechanisms for MoE, including energy-based observation routing (Di-BM), Gram-Schmidt orthogonalization (MOORE, OMOE), and parameter-free competition (GatePro).

**Critical finding for your project:** Kim (2026) demonstrated that orthogonality regularization in MoE models fails to achieve its stated goal. Weight-space overlap actually increases by up to 114% under regularization, activation-space overlap remains unchanged at ~0.6 regardless, and there is no significant correlation ($r=-0.293$, $p=0.523$) between weight and activation orthogonality. This directly challenges the theoretical basis of your orthogonality loss component.

**Actionable recommendations:** * (1) Consider removing or down-weighting the orthogonality loss in favor of your discriminability loss, which operates on a more principled adversarial basis.
* (2) Investigate Di-BM-style energy-based observation routing as an alternative to output-space diversity penalties.
* (3) Try Gram-Schmidt orthogonalization (MOORE/OMOE) which enforces hard constraints in representation space rather than soft regularization.
* (4) Target the 2nd Workshop on Compositional Learning at ICML 2026 (deadline April 24 AOE) as the primary submission venue; MoE as compositional policy with diversity-driven modular skill learning is a natural fit.

**Key question to resolve experimentally:** Does diversity-aware MoE learn new tasks faster than vanilla MoE or the dense baseline? If so, this adaptation-speed story is compelling for a workshop paper, even if absolute LIBERO-10 performance is not superior. Frame the contribution around efficient skill transfer, not peak benchmark performance.

*References:*
* 1. Kim (2026), https://arxiv.org/abs/2601.00457
* 2. DI-BM, Shen et al. (2026), https://arxiv.org/abs/2601.12397
* 3. MOORE, Hendawy et al. (2023), https://arxiv.org/abs/2311.11385

---

## 2. Project Status & Key Challenges

### 2.1 Current Architecture
Your VLA MOE Diversity system builds on SmolVLA / LeRobot with the following components:

| Component | Details |
| :--- | :--- |
| **VLM Backbone** | 350M params, frozen during MoE training |
| **Action Expert** | 16-layer transformer, 384 hidden dim |
| **MoE Configuration** | $N=8$ SwiGLU experts per layer, top-2 routing via learned router |
| **Trainable Params** | ~100M (parameter-matched to baseline) |
| **Benchmark** | LIBERO-10 (10 manipulation tasks) |
| **Compute** | Princeton Adroit cluster (A100/H100), ~12h per run |

### 2.2 Diversity Objective (Two Components)
1. **Orthogonality Loss:** Penalizes cosine similarity between mean expert output vectors. Aims to push experts toward producing geometrically dissimilar outputs.
2. **Discriminability Loss:** Trains a small MLP classifier (ExpertDiscriminator) to predict which expert produced a given output. Experts are rewarded for producing distinguishable outputs. This is an adversarial/GAN-like mechanism.

### 2.3 Results So Far
MoE with and without diversity loss has performed worse than the dense baseline on LIBERO-10. No configuration has achieved zero-shot transfer to new tasks. The current experiment is testing whether diversity-aware MoE learns new tasks faster (few-shot transfer). This adaptation-speed hypothesis is the most promising direction for a workshop-scale contribution.

### 2.4 Key Open Questions
1. Does diversity-aware MoE converge faster on held-out LIBERO tasks compared to vanilla MoE?
2. Is the orthogonality loss actually helping, or is it interfering with training (per Kim 2026)?
3. Are the 8 experts actually specializing, or are they learning redundant representations?
4. Would alternative diversity mechanisms (Di-BM, MOORE, GatePro) be more effective?

---

## 3. Latest Research on MoE + Diversity

### 3.1 Critical Finding: Orthogonality Loss May Be Broken
**"Geometric Regularization in MoE: The Disconnect Between Weights and Activations"**
*Hyunjun Kim (KAIST) | arXiv:2601.00457 | January 2026 | Paper*

This paper systematically evaluates orthogonality regularization in MoE models (NanoGPT-MOE, ~130M params, 8 experts, top-2 routing) and finds it fails on every front:
* **Key findings:** (a) Weight MSO increases by up to 114% under orthogonality loss, contrary to intent. (b) Activation-space overlap remains ~0.6 regardless of regularization strength. (c) No significant correlation between weight and activation orthogonality ($r=-0.293$, $p=0.523$). (d) Performance effects are inconsistent: -0.9% on WikiText-103, +0.9% on TinyStories, high variance on PTB.
* **Why it fails:** SiLU activations and LayerNorm compress angular differences between expert outputs, increasing activation overlap regardless of weight geometry. Baseline weights are already near-orthogonal (MSO ~ 10-4), so regularization interferes with natural training dynamics.
* **RELEVANCE:** Directly challenges your orthogonality loss component. Your approach penalizes cosine similarity of mean expert outputs (activation-space), which is slightly different from Kim's weight-space target. However, Kim's finding that activation MSO stays ~0.6 regardless of weight orthogonality suggests that even activation-space geometric penalties may be unreliable. The discriminability loss is a more principled mechanism.

### 3.2 MoE for VLA / Robot Manipulation
* **"Expertise Need Not Monopolize: Action-Specialized MoE for VLA" (AdaMoE)** (Shen et al. | Oct 2025): Decouples expert selection from weighting via an independent scale adapter alongside the traditional router. Built on pio VLA. Results: +1.8% LIBERO, +9.3% RoboTwin, +21.5% real-world. Key insight: collaborative expert utilization outperforms winner-takes-all dynamics. **RELEVANCE:** Direct competitor. Uses the same LIBERO benchmark. Their decoupled selection/weighting approach could complement your diversity objectives.
* **"HiMOE-VLA: Hierarchical MoE for Generalist VLA Policies"** (Du et al. | Dec 2025): Hierarchical MoE in the action module handles heterogeneity across embodiments, action spaces, and scenes. Adaptively abstracts multiple heterogeneity sources across layers into shared representations.
* **"DITEA: MoE for VLA in Robotic Manipulation"** (Li & Wang | AAAI 2026): Integrates MoE into the action head of a diffusion-based VLA model. Introduces a Task-Instruction Gate for expert selection, using language instructions to guide routing. **RELEVANCE:** Task-conditioned routing is a promising alternative to your learned router. Language instructions provide strong prior for which expert should handle a task.
* **"MORE: Unlocking Scalability in RL for Quadruped VLA Models"** (Zhao et al. | ICRA 2025): MoE of LORA experts for quadruped VLA, trained with RL-based objectives (Q-function). Demonstrates MoE + RL fine-tuning can leverage mixed-quality data effectively.
* **"FedVLA: Federated VLA Learning with Dual Gating MoE"** (Miao et al. | ICCV 2025): Dual Gating MoE (DGMOE): both tokens and experts decide on activation (bidirectional selection). Expert-Driven Aggregation for federated training. Average expert density ~1.23 per token.
* **"MoLe-VLA: Dynamic Layer-skipping VLA via Mixture-of-Layers"** (Zhang et al. | Mar 2025): Treats each LLM layer as an expert. Spatial-Temporal Aware Router selectively activates layers. +8% success rate on RLBench while reducing FLOPs by up to 5.6x.
* **"ChatVLA / ChatVLA-2: Unified Multimodal Understanding and Robot Control"** (Zhou et al. | 2025): Uses static MoE (separate experts for understanding vs. control) to minimize task interference. ChatVLA-2 adds dynamic MoE with 8 experts, top-2 routing. Phased alignment training.
* **"MergeVLA: Cross-Skill Model Merging Toward a Generalist VLA Agent"** (Fu et al. | Nov 2025): Merges VLA experts trained on different tasks using task masks and sparsely activated LoRA. Replaces self-attention with cross-attention in action expert for composability. 90.2% success on LIBERO across all four task suites after merging.

### 3.3 Diverse Skill Learning with MoE in Robotics
* **"Learning Diverse Skills for Behavior Models with MoE" (Di-BM) [MOST RELEVANT]** (Shen et al. | Jan 2026): Associates each expert with a distinct observation distribution using energy-based models (EBMs). Experts specialize in sub-regions of the observation space rather than being forced apart by output-space penalties. Plug-and-play integration into standard imitation learning. Tested on real-world manipulation tasks with superior data efficiency during fine-tuning on novel tasks. **RELEVANCE:** The most directly relevant paper. Unlike your approach (output-space diversity losses), Di-BM achieves diversity through observation-space partitioning. This is more principled because it lets experts naturally specialize based on what they see, not what they produce. Consider adopting this approach.
* **"Acquiring Diverse Skills using Curriculum RL with MoE" (Di-Skill)** (Celik et al. | ICML 2024): RL method for learning diverse skills using MoE with per-expert context distributions (EBMs). Maximum entropy objective ensures skill diversity. Predates and inspires Di-BM. Demonstrates that decomposed objectives with overlapping sub-regions achieve both diversity and coverage.
* **"Abstracting Robot Manipulation Skills via MoE Policy" (SMP)** (Hao et al. | Jan 2026): Diffusion-based MoE policy learning a compact orthogonal skill basis with sticky routing. Skills are state-adaptive orthogonal action primitives. Adaptive expert activation at inference reduces computational cost. Validated on real dual-arm platform for multi-task and transfer learning. **RELEVANCE:** Also uses orthogonality, but in a variational framework with a whitened action basis. Their orthogonality is on skill bases (action primitives), not expert weights/outputs. Sticky routing prevents rapid expert switching, which may help skill coherence.
* **"MoE-DP: MoE-Enhanced Diffusion Policy for Robust Long-Horizon Manipulation"** (Cheng et al. | Nov 2025): Inserts MoE layer between visual encoder and diffusion model. Achieves interpretable skill decomposition where distinct experts correspond to semantic task primitives (approaching, grasping, placing). 36% relative improvement under disturbance. Enables inference-time subtask rearrangement.
* **"SDP: A Sparse, Reusable, and Flexible Policy for Robot Learning"** (Wang et al. | Jul 2024): MoE within transformer-based diffusion policy. Task-specific routers selectively activate experts. In continual learning, adds new experts + routers while freezing existing ones. Prevents catastrophic forgetting, enables efficient task transfer.

### 3.4 Expert Diversity Mechanisms (LLM / General)
* **"GatePro: Parameter-Free Expert Selection Diversity"** (Zheng et al. | Oct 2025): Identifies most similar expert pairs and introduces localized competition, preventing redundant co-activation. No auxiliary loss or extra parameters. Hot-swappable during any training phase. Tested at Seed-MoE 0.7B/7B and $1.3B/13B$ scales. Reduces expert cosine similarity, increases selection entropy, accelerates expert activation in deeper layers. **RELEVANCE:** Could replace your load-balancing auxiliary loss with a principled competition mechanism. Requires no extra parameters and directly targets the router to prevent functional redundancy.
* **"MOORE: Multi-Task RL with Mixture of Orthogonal Experts"** (Hendawy et al. | Nov 2023): Uses Gram-Schmidt process to orthogonalize expert representations on the Stiefel manifold. Hard constraint (not regularization) in representation space. New SOTA on MetaWorld MT10 and MT50 benchmarks. Exhibits faster convergence than non-orthogonal baselines. **RELEVANCE:** Key alternative to your soft orthogonality loss. MOORE enforces hard orthogonality in representation space via Gram-Schmidt, avoiding the weight-activation disconnect identified by Kim. Could be applied to your expert outputs before routing.
* **"Selective Sinkhorn Routing for Improved Sparse MoE" (SSR)** (Nguyen et al. | Nov 2025): Formulates token-expert assignment as entropy-regularized optimal transport. Replaces auxiliary balance loss entirely. Applying Sinkhorn routing to just 0.1%-1% of steps is sufficient. Achieves faster convergence and robustness to input corruption.
* **"Chain-of-Experts: Unlocking the Communication Power of MoE" (COE)** (Wang et al. | Jun 2025): Sequential expert communication within each layer. Tokens processed iteratively across a chain of experts with re-routing at each step. Increases effective expert combinations by 823x. Reduces validation loss from 1.20 to 1.12 on math reasoning vs. standard MoE.
* **"Diversifying MoE Representation with Orthogonal Optimizer" (OMOE Optimizer)** (Liu et al. | 2023/2024): Identifies the homogeneous representation problem (99% expert similarity). Proposes an orthogonal optimizer with alternating training: experts update parameters orthogonal to subspaces of other experts. Improves fine-tuning performance on GLUE, SuperGLUE, QA, and NER tasks.
* **"HMOE: Heterogeneous Mixture of Experts for Language Modeling"** (Wang et al. | Aug 2024): Experts differ in size, providing diverse capacities. Novel training objective encourages activation of smaller experts. Hybrid size distribution performs best. Complex tokens are routed to larger experts, simple tokens to smaller ones.

### 3.5 Broader Work on Diversity in Robot Data & Skills
* **"Is Diversity All You Need for Scalable Robotic Manipulation?"** (Shi et al. | Jul 2025): Investigates three diversity dimensions: task (what to do), embodiment (which robot), expert (who demonstrates). Finds: (1) task diversity > per-task quantity for transfer; (2) multi-embodiment data is optional for cross-embodiment transfer; (3) expert diversity can be confounding due to velocity multimodality. Proposes distribution debiasing for +15% gains. **RELEVANCE:** Warns that expert diversity can be a double-edged sword. Velocity multimodality from diverse demonstrators can confuse policy learning. Consider whether your MoE experts might be capturing demonstrator-style variation rather than task-relevant skill variation.
* **"DIAYN: Diversity Is All You Need"** (Eysenbach et al. | ICLR 2019): The foundational paper on learning diverse skills via mutual information maximization. Skills are learned to be distinguishable by a discriminator. Your ExpertDiscriminator is conceptually similar but applied at the expert-output level rather than the state-visitation level.

---

## 4. Analysis & Recommendations for Your Project

### 4.1 Why Your Orthogonality Loss May Not Be Working
Kim (2026) found that orthogonality regularization in MoE models neither reduces weight overlap nor translates to functional diversity in activation space. Your implementation penalizes cosine similarity of mean expert outputs, which operates in activation space rather than weight space. This is slightly more promising than Kim's weight-space regularization, but Kim's key finding that activation MSO remains ~0.6 regardless of regularization suggests that even activation-space geometric penalties may be unreliable. 

The fundamental problem is that non-linear transformations (SwiGLU in your case, SiLU in Kim's) and LayerNorm compress angular differences between expert outputs. Penalizing mean output cosine similarity does not prevent experts from producing functionally similar behavior on individual inputs.

**Recommended actions:**
* **Ablate the orthogonality loss:** Run MoE with only the discriminability loss (no orthogonality). If performance improves or stays the same, the orthogonality loss is actively harmful.
* **Increase discriminability loss weight:** Your ExpertDiscriminator approach is more principled because it directly measures functional distinguishability. It is conceptually aligned with DIAYN's mutual information objective. Try increasing its weight relative to the task loss.
* **Switch to Gram-Schmidt orthogonalization (MOORE):** Instead of a soft penalty, use a hard constraint that projects each expert's representation onto the orthogonal complement of other experts' representations. This operates in representation space and avoids the weight-activation gap.
* **Try Di-BM-style observation routing:** Let experts specialize based on observation distributions using energy-based models. This sidesteps the output-space diversity problem entirely.
* **Try GatePro at the router level:** Prevent redundant expert co-activation through localized competition between similar expert pairs. No extra parameters, no auxiliary loss.

### 4.2 Alternative Diversity Mechanisms to Try

| Mechanism | Source | How It Works | Implementation Effort |
| :--- | :--- | :--- | :--- |
| **EBM observation routing** | Di-BM | Per-expert energy-based observation distributions; experts specialize by input region | Medium-High: requires EBM training alongside action model |
| **Gram-Schmidt orthogonalization** | MOORE / OMOE | Hard orthogonality constraint on expert representations via Gram-Schmidt process | Low: modify forward pass to orthogonalize expert outputs |
| **Localized competition (GatePro)** | GatePro | Identify similar expert pairs, prevent co-activation through competition | Low: modify router gating, no extra parameters |
| **Optimal transport routing** | SSR | Sinkhorn-based token-expert assignment replaces auxiliary loss | Medium: replace router with OT-based assignment |
| **Stronger discriminability loss** | Your design | Increase weight of ExpertDiscriminator loss; adversarial training | Minimal: hyperparameter change only |
| **Task-instruction routing** | DITEA | Use language instruction embedding to condition expert selection | Medium: add instruction embedding to router |

### 4.3 Compute-Efficient Experiment Plan for Adroit
Given ~12 hours per experiment on A100 and the April 24 deadline, you have approximately 10-12 experiment slots remaining (accounting for write-up time). Prioritize the following:

| # | Experiment | Purpose | Status |
| :--- | :--- | :--- | :--- |
| **1** | Baseline (dense) on LIBERO-10 | Reference performance | Done |
| **2** | Vanilla MoE (no diversity) on LIBERO-10 | MoE overhead assessment | Done |
| **3** | MoE+ diversity (orth + disc) on LIBERO10- | Full system evaluation | Done |
| **4** | Few-shot transfer to new LIBERO tasks | Test adaptation speed hypothesis | In progress |
| **5** | MoE + discriminability only (no orthogonality) | Ablate orthogonality loss | HIGH PRIORITY |
| **6** | MoE + increased discriminability weight | Stronger adversarial diversity signal | HIGH PRIORITY |
| **7** | Expert utilization analysis across conditions | Verify experts specialize differently | Analysis only |
| **8** | Gram-Schmidt orthogonalization (MOORE-style) | Hard constraint alternative | If time permits |

### 4.4 ICML 2026 Workshop Targets & Timeline
**Primary target:** 2nd Workshop on Compositional Learning: Safety, Interpretability, and Agents5

| Detail | Info |
| :--- | :--- |
| **Deadline** | April 24, 2026 AOE |
| **Notification** | May 15, 2026 AOE |
| **Format** | 4 or 8 pages, ICML format, double-blind, non-archival |
| **Conference** | ICML 2026, Seoul, July 6-11. Workshops July 10-11 |
| **Fit** | Excellent: MoE = compositional policy; diversity = modular skill learning |

**Alternative workshops:**
* **RLXF: RL from World Feedback:** Good fit if you add an RL fine-tuning component (see MoRE's approach). Would require additional experiments.
* **SPIGM:** Structured Probabilistic Inference & Generative Modeling Possible fit if you frame MoE routing as structured probabilistic inference.
* **ColorAl-Workshop page:** Check if the scope includes robotics/embodied Al.

**Recommended paper framing:**
Do not frame the paper around beating LIBERO-10 SOTA (you won't). Instead, frame it as: "Can diversity-aware MoE in VLA action heads accelerate adaptation to new manipulation tasks?" Key narrative elements: 
* (1) Orthogonality loss is unreliable (citing Kim 2026) but discriminability loss shows promise; 
* (2) Diversity-aware MoE experts learn transferable skills; 
* (3) Analysis of expert specialization patterns under different diversity objectives; 
* (4) Compute-efficient approach on SmolVLA.

**Timeline:** ~2 weeks remaining. Days 1-3: finalize transfer experiments + ablations. Days 4-7: expert analysis + additional experiments. Days 8-12: writing. Days 13-14: revision buffer. Submit by April 24 ΑΟΕ.

*References:*
* 4. Kim (2026), https://arxiv.org/abs/2601.00457
* 5. ICML 2026 Workshops, https://blog.icml.cc/2026/04/06/announcing-the-icml-2026-workshops-and-affinity-workshops/

---

## 5. Key Papers Quick Reference

| Paper | Authors | Date | Key Contribution | Rel. |
| :--- | :--- | :--- | :--- | :--- |
| **Geometric Regularization in MoE** | Kim | Jan 2026 | Orthogonality loss fails; weight-activation disconnect | Critical |
| **Di-BM** | Shen et al. | Jan 2026 | EBM observation routing for expert diversity | Critical |
| **AdaMoE** | Shen et al. | Oct 2025 | Decoupled selection/weighting for VLA MOE | High |
| **SMP** | Hao et al. | Jan 2026 | Orthogonal skill basis + sticky routing | High |
| **MOORE** | Hendawy et al. | Nov 2023 | Gram-Schmidt orthogonalization for MTRL | High |
| **GatePro** | Zheng et al. | Oct 2025 | Parameterfree -localized expert competition | High |
| **OMOE Optimizer** | Liu et al. | 2023 | Orthogonal optimizer for MoE diversity | High |
| **MoE-DP** | Cheng et al. | Nov 2025 | MoE in diffusion policy; 36% robustness gain | Med |
| **HIMOE-VLA** | Du et al. | Dec 2025 | Hierarchical MoE for VLA heterogeneity | Med |
| **MergeVLA** | Fu et al. | Nov 2025 | Task masks + cross-attention for VLA merging | Med |
| **Di-Skill** | Celik et al.. | ICML 2024 | Curriculum RL with per-expert EBM contexts | Med |
| **SSR** | Nguyen et al. | Nov 2025 | Optimal transport routing for MoE | Med |
| **Chain-of-Experts** | Wang et al. | Jun 2025 | Sequential expert communication | Med |
| **SDP** | Wang et al. | Jul 2024 | Sparse MoE diffusion policy, continual learning | Med |
| **MORE** | Zhao et al. | ICRA 2025 | MOE LORA + RL for quadruped VLA | Med |
| **FedVLA** | Miao et al. | ICCV 2025 | Dual gating MoE for federated VLA | Low |
| **MoLe-VLA** | Zhang et al. | Mar 2025 | Layer-as-expert, 5.6x FLOPs reduction | Low |
| **HMOE** | Wang et al. | Aug 2024 | Heterogeneous expert sizes | Low |
| **ChatVLA** | Zhou et al. | Feb 2025 | Static MoE for task interference | Low |
| **Diversity for Robotics** | Shi et al. | Jul 2025 | Task diversity > expert diversity for scaling | Low |