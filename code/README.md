# Simulation Code

This directory contains the Python source code used to perform the task allocation simulations described in the monograph *Aplicação de Pesquisa Operacional para Administração de Tarefas e Conhecimento em um Departamento Jurídico*.

## Structure

Two allocation models are implemented:

### `baseline/main.py`

Implements the baseline task allocation model.

Tasks are allocated considering analyst specialization and current workload, without explicitly incorporating learning as a decision criterion.

### `ahp/main.py`

Implements the multicriteria allocation model based on the Analytic Hierarchy Process (AHP).

The model evaluates three criteria:

1. Learning
2. Specialization
3. Workload

The pairwise comparison matrix implemented in the source code is:

| Criterion | Learning | Specialization | Workload |
|---|---:|---:|---:|
| Learning | 1 | 3 | 5 |
| Specialization | 1/3 | 1 | 1 |
| Workload | 1/6 | 1 | 1 |

Learning progression is represented through a logistic learning function whose state evolves according to the simulated experience accumulated by each analyst.

## Input Data

Both models use the input datasets available in the `/data` directory:

- `DB_AT.xlsx`
- `DB_AN.xlsx`
- `KPMG_Legal_Dept_tab.xlsx`

The datasets are synthetic and do not contain confidential information from an actual legal department.

## Experimental Design

Four demand scenarios are evaluated:

1. Base scenario
2. Concentration of tasks in Corporate Law (`Societário`)
3. Workload overload
4. Combination of concentration and overload

Each scenario is simulated with teams of 5, 7, and 10 analysts.

This results in 12 experimental configurations for each allocation model.

## Outputs

Simulation outputs are stored in the `/results` directory, separated into:

- `/results/without_AHP`
- `/results/with_AHP`

Each simulation generates datasets containing the complete worklog, knowledge concentration indicators, analyst-area knowledge distribution, consolidated metrics, and learning state.

## Requirements

Required Python packages are listed in the repository's `requirements.txt` file.

They can be installed with:

```bash
pip install -r requirements.txt
