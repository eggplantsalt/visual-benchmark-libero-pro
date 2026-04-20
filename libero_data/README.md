---
license: mit
---

# Dataset Card for LIBERO-PRO Perturbation Dataset

This dataset contains the **`bddl`** and **`init`** files of LIBERO-PRO configurations under **object**, **relation**, **semantic**, **task**, and **environment** perturbations. The dataset supports direct integration with the [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) framework.

---

## Dataset Details

### Dataset Description

- **Curated by:** LIBERO-PRO Research Team  
- **Affiliation:** MLLab, Huazhong University of Science and Technology  
- **Language(s) (NLP):** English (instructional text)  
- **License:** MIT  
- **Primary Purpose:** Evaluation of VLA models under structured perturbations  

This dataset extends the original [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO/) by introducing **systematic perturbations** in five dimensions:
1. **Object Perturbation:** Modifies object appearance, color, and scale to test adaptability to visual shifts.  
2. **Position Perturbation:** Relocates objects within feasible spatial bounds to evaluate the model’s adaptability to spatial position changes.  
3. **Semantic Perturbation:** Paraphrases natural language commands to probe linguistic robustness.  
4. **Task Perturbation:** Redefines task logic and target states to test procedural generalization.
5. **Environment Perturbation:** Replaces working environments to evaluate cross-environment robustness.  

Each perturbation includes corresponding **`init` files** (initial environment configurations) and **`bddl` files** (behavioral descriptions in BDDL format).

---

## Uses

**How to use:**
1. Copy all **`.bddl`** files to:
   ```
   LIBERO-PRO/libero/libero/bddl_files/
   ```
2. Copy all **`init`** files to:
   ```
   LIBERO-PRO/libero/libero/init_files/
   ```
3. Follow the quick start instructions provided in the [LIBERO-PRO README](https://github.com/Zxy-MLlab/LIBERO-PRO#readme).

---

## Dataset Structure

Each perturbation category contains:
- **`init/`**: Environment initialization files defining object placement and world state.  
- **`bddl/`**: Task goal definitions in Behavior Domain Definition Language.  

---

## Citation

If you use this dataset, please cite the LIBERO-PRO project:

**BibTeX:**
```bibtex
@misc{zhou2025liberopro,
  title={LIBERO-PRO: Benchmarking Perturbation Robustness of Visual Language Action Models},
  author={Zhou, Xueyang and Tie, Guiyao and Zhang, Guowen and Wang, Hechang and Zhou, Pan},
  year={2025},
  howpublished={\url{https://github.com/Zxy-MLlab/LIBERO-PRO}},
}
```

---

## Dataset Card Authors

- Xueyang Zhou
- Yangming Xu

---

## Dataset Card Contact

For questions or issues, please contact:  
📧 **d202480819@hust.edu.cn**
