# Legal Knowledge Allocation Using AHP

## Application of Operations Research to Task and Knowledge Management in a Legal Department

This repository contains the computational model, synthetic datasets, and simulation results developed for the monograph **“Aplicação de Pesquisa Operacional para Administração de Tarefas e Conhecimento em um Departamento Jurídico”**, submitted as part of the MBA in Data Science and Analytics.

The research investigates whether different task allocation policies can influence the distribution and concentration of knowledge within a simulated corporate legal department.

## Research Objective

The study develops and evaluates a computational task allocation model based on Operations Research and compares two allocation approaches:

- a **baseline allocation method**, primarily based on analyst specialization and workload; and
- a **multicriteria allocation method based on the Analytic Hierarchy Process (AHP)**, which explicitly incorporates learning as a decision criterion alongside specialization and workload.

The purpose is to analyze whether incorporating learning into task allocation decisions can reduce knowledge concentration among members of a legal department.

## Research Hypotheses

**H1:** A multicriteria task allocation model prioritizing learning reduces knowledge concentration compared with an allocation model predominantly based on specialization.

**H2:** The effect of the allocation model varies according to task incidence, with greater potential for knowledge distribution in legal areas with higher task volumes.

## Methodology

The research uses a simulation environment based entirely on **synthetic data**.

Three main datasets support the model:

- `DB_AT.xlsx` — characteristics of legal tasks and activities;
- `DB_AN.xlsx` — characteristics of simulated analysts;
- `KPMG_Legal_Dept_tab.xlsx` — reference data used to parameterize the distribution of activities across legal practice areas.

Two allocation models are simulated independently.

### Baseline model

The baseline model allocates tasks according to analyst specialization and workload.

### AHP model

The alternative model applies the **Analytic Hierarchy Process (AHP)** using three criteria:

1. Learning
2. Specialization
3. Workload

The pairwise comparison matrix used in the simulation is:

| Criterion | Learning | Specialization | Workload |
|---|---:|---:|---:|
| Learning | 1 | 3 | 5 |
| Specialization | 1/3 | 1 | 1 |
| Workload | 1/5 | 1 | 1 |

Learning progression is represented through a logistic learning function that evolves according to the simulated experience accumulated by each analyst.

## Experimental Design

Four demand scenarios are evaluated:

1. **Base scenario** — baseline distribution of legal activities.
2. **Concentrated scenario** — 50% of incoming tasks are concentrated in Corporate Law (`Societário`).
3. **Overload scenario** — task generation is increased from 5 to 15 tasks per business day.
4. **Concentrated + overload scenario** — combines concentration in Corporate Law with increased task volume.

Each scenario is evaluated with teams of **5, 7, and 10 analysts**.

This produces **12 experimental configurations for each allocation method**, corresponding to **24 simulation runs in total**.

## Repository Structure

```text
legal-knowledge-allocation-ahp/
├── code/
│   ├── ahp/
│   │   └── main.py
│   ├── baseline/
│   │   └── main.py
│   └── README.md
├── data/
│   ├── DB_AN.xlsx
│   ├── DB_AT.xlsx
│   ├── KPMG_Legal_Dept_tab.xlsx
│   └── README.md
├── docs/
├── results/
│   ├── with_AHP/
│   ├── without_AHP/
│   └── README.md
├── .gitignore
├── README.md
└── requirements.txt
```

## Simulation Outputs

Each experimental configuration generates datasets containing:

- complete task allocation worklog;
- knowledge concentration indicators by legal area;
- accumulated knowledge by analyst and legal area;
- consolidated simulation metrics; and
- final learning state.

The complete outputs of all simulated scenarios are available in the [`results`](./results) directory.

## Reproducing the Simulations

The project requires Python and the packages listed in `requirements.txt`.

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run the baseline allocation model:

```bash
python code/baseline/main.py
```

Run the AHP-based allocation model:

```bash
python code/ahp/main.py
```

Both scripts use repository-relative paths to access input data and generate simulation outputs.

## Limitations

The simulation uses exclusively synthetic data. Therefore, its results should be interpreted as evidence regarding the behavior of the proposed allocation models under the specified assumptions rather than as empirical evidence about a particular legal department.

The AHP weights and learning-function parameters represent modeling assumptions and should be empirically calibrated and validated before application in an actual organizational environment.

## Supplementary Material

This repository serves as supplementary material to the monograph and provides access to the complete simulation outputs that could not be presented individually in the main text due to its page limitation.

Its purpose is to support transparency, reproducibility, and further examination of the experimental scenarios.
