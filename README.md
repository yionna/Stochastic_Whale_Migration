# Stochastic Whale Migration

Math modelling project for MATH113B: Mathematical Modelling in Biology  
University of California, Irvine — Winter 2025

A spatially explicit individual-based model (IBM) simulating blue whale transit-foraging behaviour in the Northeast Pacific, built on a two-state semi-Markov framework.

---

## Deliverables

- **[Notebook](notebooks/MATH113B_Whale_Migration.ipynb)** — full simulation and analysis
- **[Final Report](paper/Yunhe_Xu_MATH113B_Final_Writeup.pdf)** — written research writeup with parameter justification and references

---

## Model Overview

The model simulates individual blue whale movement as a two-state random walk (transit vs. foraging), following the framework of Morales et al. (2004). State-switching is governed by a logistic function conditioned on local sea surface temperature (SST) and krill density.

- **Step lengths** follow state-specific Gamma distributions; **turning angles** follow von Mises distributions
- **SST field** is synthetically generated with latitudinal gradient, coastal upwelling, and seasonal variation
- **Krill density field** is patch-based with coastal proximity and latitudinal gradients

Full parameter justification, including sources and modelling assumptions, is documented in the [final report](paper/Yunhe_Xu_MATH113B_Final_Writeup.pdf).

---

## References

- Bailey et al. (2009). *Endangered Species Research.* doi:10.3354/esr00239
- Dodson et al. (2020). *Ecological Modelling*, 432, 109225. doi:10.1016/j.ecolmodel.2020.109225
- Morales et al. (2004). *Ecology*, 85(9), 2436–2445. doi:10.1890/03-0269
- Derville et al. (2025). *Progress in Oceanography*, 231, 103388. doi:10.1016/j.pocean.2024.103388
- Abrahms et al. (2019). *PNAS*, 116(12), 5582–5587. doi:10.1073/pnas.1819031116
- Oestreich et al. (2022). *Functional Ecology*, 36, 882–895. doi:10.1111/1365-2435.14013