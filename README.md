# Stochastic_Whale_Migration
Math modeling project for MATH113B, Math Modeling in Biology (University of California, Irvine), Winter 2025

## Proposed project structure: 
whale_migration/
├── .python-version      # Managed by uv
├── pyproject.toml       # Dependencies (numpy, scipy, cartopy, etc.)
├── uv.lock
├── README.md            # Project documentation
├── data/                # To store cached land masks or results
├── notebooks/
├── ├── MATH113B_Whale_Migration.ipynb # Original course deliverable
│   └── exploration.ipynb # For interactive experimentation
├── paper/
│   └── Yunhe_Xu_MATH113B_Final_Writeup.pdf   # Original draft paper for the course deliverable
└── src/
    └── whale_sim/
        ├── __init__.py
        ├── config.py
        ├── main.py      # Entry point to run simulation
        ├── environment.py # SST, Krill, and Land Mask generation
        ├── agents.py     # Whale class and movement logic
        └── utils.py      # Plotting and stat helper functions


## Running the project
### 1. UV environment
This project is managed by uv version control

