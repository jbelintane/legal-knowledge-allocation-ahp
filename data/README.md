# Data
 
This directory contains the synthetic input datasets used in the simulations described in the monograph.

No confidential or real-world legal department data are included.

## Files

### `DB_AT.xlsx`

Task database used by the simulation model.

It contains the characteristics of the simulated legal activities, including information related to legal area, activity type, complexity, and estimated execution time.

### `DB_AN.xlsx`

Analyst database used by the simulation model.

It contains the characteristics of the simulated analysts, including their primary legal area, seniority, and availability.

### `KPMG_Legal_Dept_tab.xlsx`

Reference dataset used to parameterize the distribution of legal activities across practice areas in the synthetic simulation environment.

## Role in the simulation

These files serve as inputs for both allocation models:

- the baseline allocation model; and
- the AHP-based multicriteria allocation model.

The simulation outputs generated from these inputs are available in the `/results` directory.
